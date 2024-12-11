import argparse
import os
import pandas as pd
import numpy as np
import cv2
import torch
from tqdm import tqdm
from pathlib import Path
from cellpose import models
from torch.utils.data import DataLoader
from dataloader import TrajPointDataset
from model import VideoClassifier
from train import train_model
from evaluate import predict_and_save, calculate_metrics, parse_csv
from csbdeep.io import save_tiff_imagej_compatible
from track import features_from_masks, tracking
from utils import vis_tracks
from utils import save_napari_animation
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


def main():
    parser = argparse.ArgumentParser(
        description='Cell Behavior Classification'
    )

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')
    subparsers.required = True

    # Train mode parser
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--root_dir',
        type=Path,
        required=True,
        help='Directory containing training videos'
    )

    train_parser.add_argument(
        '--ckpt_dir',
        type=Path,
        default=Path.cwd() / 'ckpts',
        required=False,
        help='Directory to store the model checkpoint file'
    )

    train_parser.add_argument(
        '--segmenter',
        type=str,
        default="cytotorch_0",
        required=False,
        help="Cellpose model used for segmentation"
        )
    train_parser.add_argument(
        '--num_epochs',
        type=int,
        default=500,
        help="Number of epochs for training"
    )
    train_parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.05,
        help='Learning rate for training'
    )
    train_parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        required=False,
        help="Batch size for training"
    )

    # Test mode parser
    test_parser = subparsers.add_parser('predict', help='Test for a single video')
    test_parser.add_argument(
        '--video_path',
        type=Path,
        required=True,
        help='Path to the video to be classified'
    )

    test_parser.add_argument(
        '--segmenter',
        type=str,
        default="cytotorch_0",
        required=False,
        help='Cellpose model for segmentation'
    )

    test_parser.add_argument(
        '--model_path',
        type=Path,
        default=None,
        required=True,
        help='The path of the model used for classification')

    test_parser.add_argument(
        '--save_dir',
        type=Path,
        default=None,
        required=False,
        help='The directory used to store tracking results'
    )

    # Evaluate mode parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument(
        '--root_dir',
        type=Path,
        required=True,
        help='Directory containing videos to be evaluated'
    )

    eval_parser.add_argument(
        '--model_path',
        type=Path,
        default=None,
        required=True,
        help='The path of the model used for evaluation')

    eval_parser.add_argument(
        '--save_dir',
        type=Path,
        default=Path.cwd() / 'results',
        required=False,
        help='The directory to store the csv file of predictions'
    )

    eval_parser.add_argument(
        '--segmenter',
        type=str,
        default="cytotorch_0",
        required=False,
        help="Cellpose model used for segmentation"
    )

    eval_parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        required=False,
        help="Batch size for evaluation"
    )

    args = parser.parse_args()

    # Execute the appropriate function based on the mode
    if args.mode == 'train':
        data_dir = os.path.join(args.root_dir, "trks", args.segmenter)
        train_dataset = TrajPointDataset(data_dir, split="training", augment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)

        val_dataset = TrajPointDataset(data_dir, split="testing")
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=val_dataset.collate_fn)

        model = VideoClassifier(n_features=64, max_sequence_length=20, sample_attention="single")
        best_model = train_model(model, train_dataloader, val_dataloader,
                                 num_epochs=args.num_epochs,
                                 learning_rate=args.learning_rate)
    elif args.mode == 'evaluate':
        print(f"Evaluate with segmenter {args.segmenter}")
        data_dir = os.path.join(args.root_dir, "trks", args.segmenter)
        tst_dataset = TrajPointDataset(data_dir, split="testing")
        tst_dataloader = DataLoader(tst_dataset, batch_size=4, shuffle=False, collate_fn=tst_dataset.collate_fn)
        best_model = VideoClassifier(n_features=64, max_sequence_length=20, sample_attention="single")
        best_model.load_state_dict(torch.load(args.model_path, weights_only=True)["model_state_dict"])

        predict_and_save(best_model,
                         tst_dataloader,
                         os.path.join(args.save_dir, "preds.csv"))
        ground_truth_file = os.path.join(args.root_dir, "test.csv")
        predictions_file = os.path.join(args.save_dir, "preds.csv")

        ground_truth = parse_csv(ground_truth_file)
        predictions = parse_csv(predictions_file)

        eval_results = calculate_metrics(ground_truth, predictions)
        for k, v in eval_results.items():
            print(f"{k}: {v:.4f}")
    elif args.mode == 'predict':
        video_path = Path(args.video_path)
        vid = video_path.stem
        video = cv2.VideoCapture(video_path)
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame[..., 1] if len(frame.shape) > 1 else frame)
        imgs = np.asarray(frames)

        save_dir = Path(args.save_dir) / vid if args.save_dir else video_path.parent / vid
        save_dir.mkdir(parents=True, exist_ok=True)
        save_tiff_imagej_compatible(os.path.join(save_dir, vid + '_imgs.tif'), imgs, axes='TYX')

        model_path = Path.cwd() / "segmenter" / "cellpose_retrained_sgd.pth" \
            if args.segmenter == "cyto_retrained" else os.fspath(models.MODEL_DIR.joinpath(args.segmenter))

        model = models.CellposeModel(gpu=torch.cuda.is_available(),
                                     pretrained_model=str(model_path),
                                     model_type=None if args.segmenter == "cyto_retrained" else "cyto")
        masks = np.zeros_like(imgs)
        for t in (pbar := tqdm(range(imgs.shape[0]))):
            pbar.set_description(f"Segmenting {t}/{imgs.shape[0]} "
                                 f"frame with {args.segmenter}")
            img = imgs[t, ...]
            if args.segmenter == "cyto_retrained":
                labels = model.eval(img,
                                    channels=[0, 0],
                                    cellprob_threshold=0,
                                    min_size=20)[0]
            else:
                labels = model.eval(img,
                                    channels=[0, 0],
                                    diameter=5,
                                    cellprob_threshold=0,
                                    min_size=20)[0]
            masks[t] = labels
        save_tiff_imagej_compatible(os.path.join(save_dir, vid + '_msks.tif'), masks, axes='TYX')
        imgs_path = os.path.join(save_dir, vid + '_imgs.tif')
        msks_path = os.path.join(save_dir, vid + '_msks.tif')

        print("Tracking cells...")
        df = features_from_masks(imgs_path, msks_path, int_thres=50, sz_thres=50)
        if not df.empty:
            trajs = tracking(df, os.path.join(save_dir, f"{vid}_track_trajectories.csv"),
                             max_gap=20, min_len=3, min_int=50)
            if not trajs.empty:
                viewer = vis_tracks(save_dir.parent, vid, track_ids=None)
                save_napari_animation(viewer, os.path.join(save_dir, vid + f"_{args.segmenter}.mp4"), fps=5)
                viewer.close()
                print("Tracking results have been save in " +
                      f"{os.path.join(save_dir, f'{vid}_track_trajectories.csv')}")
            else:
                print("No cells on interest present in the video, "
                      "classifying the video as Class 0!")
                return
        else:
            print("No cells on interest present in the video, "
                  "classifying the video as Class 0!")
            return

        print("Classifying...")
        best_model = VideoClassifier(n_features=64, max_sequence_length=20, sample_attention="single")
        best_model.load_state_dict(torch.load(args.model_path, weights_only=True)["model_state_dict"])
        best_model.eval()

        data = pd.read_csv(os.path.join(save_dir, f"{vid}_track_trajectories.csv"))
        data.replace([np.inf, -np.inf], 0, inplace=True)
        tracks_data = data.loc[:, ['particle', 'frame', 'x', 'y', 'area']]
        track_ids = tracks_data['particle'].unique()

        video_trajs = []
        track_trajs = []
        for track_id in track_ids:
            track = tracks_data.loc[tracks_data['particle'] == track_id].sort_values(by=['frame'])
            pos_idx = track.columns.get_indexer(['frame', 'x', 'y'])
            track_trajs.append(torch.tensor(track.iloc[:, pos_idx].values))
        video_trajs.append(track_trajs)

        padded_trajectories = []
        traj_lengths = []
        video_lengths = torch.tensor([len(traj_list) for traj_list in video_trajs])

        for video_trajs in video_trajs:
            # Get lengths of each trajectory in this video
            lengths = torch.tensor([len(traj) for traj in video_trajs])

            # Create padded tensor for this video's trajectories
            n_trajs = len(video_trajs)
            padded = torch.zeros(n_trajs, 20, 3)

            # Fill in trajectories
            for i, traj in enumerate(video_trajs):
                length = lengths[i]
                padded[i, :length] = traj[:20]  # Truncate if longer than max_length
                padded[i, length:] = traj[length - 1]

            padded_trajectories.append(padded)
            traj_lengths.append(lengths)

        with torch.no_grad():
            outputs = best_model((padded_trajectories, traj_lengths, video_lengths))

            # Convert outputs to predictions
            output = torch.atleast_1d(outputs.squeeze())
            pred = int((output >= 0.5).float().cpu().numpy()[0])
            print(f"Video {vid} is classified as Class {pred}.")


if __name__ == '__main__':
    main()
