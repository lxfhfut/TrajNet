import argparse
import os
import pandas as pd
import numpy as np
import cv2
import torch
from glob import glob
from tqdm import tqdm
from pathlib import Path
from cellpose import models
from torch.utils.data import DataLoader
from dataloader import TrajPointDataset
from model import VideoClassifier
from train import train_model
from evaluate import evaluate_and_save, predict_and_save, calculate_metrics, parse_csv
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
        default="cyto_retrained",
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
        default=0.005,
        help='Learning rate for training'
    )
    train_parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        required=False,
        help="Batch size for training"
    )

    # Predict mode parser
    test_parser = subparsers.add_parser('predict', help='Predict for videos')
    test_parser.add_argument(
        '--root_dir',
        type=Path,
        required=True,
        help='Root directory where data is stored'
    )

    test_parser.add_argument(
        '--segmenter',
        type=str,
        default="cyto_retrained",
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
        help='The directory used to store prediction results'
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
        default="cyto_retrained",
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
        train_dataset = TrajPointDataset(data_dir, split="train", augment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)

        val_dataset = TrajPointDataset(data_dir, split="test_phase1")
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=val_dataset.collate_fn)

        model = VideoClassifier(n_features=64, max_sequence_length=20, sample_attention="single")
        best_model = train_model(model, train_dataloader, val_dataloader,
                                 num_epochs=args.num_epochs,
                                 learning_rate=args.learning_rate,
                                 save_dir=args.ckpt_dir)
    elif args.mode == 'evaluate':
        print(f"Evaluate with segmenter {args.segmenter}")
        data_dir = os.path.join(args.root_dir, "trks", args.segmenter)
        tst_dataset = TrajPointDataset(data_dir, split="test_phase1")
        tst_dataloader = DataLoader(tst_dataset, batch_size=4, shuffle=False, collate_fn=tst_dataset.collate_fn)
        best_models = []
        if Path(args.model_path).is_dir():
            model_paths = glob(os.path.join(args.model_path, "*.pt"))
            for model_path in model_paths:
                best_model = VideoClassifier(n_features=64, max_sequence_length=20, sample_attention="single")
                best_model.load_state_dict(torch.load(model_path, weights_only=True)["model_state_dict"])
                best_models.append(best_model)
        else:
            best_model = VideoClassifier(n_features=64, max_sequence_length=20, sample_attention="single")
            best_model.load_state_dict(torch.load(args.model_path, weights_only=True)["model_state_dict"])
            best_models.append(best_model)

        evaluate_and_save(best_models,
                          tst_dataloader,
                          os.path.join(args.save_dir, "preds.csv"))
        ground_truth_file = os.path.join(args.root_dir, "test_phase1.csv")
        predictions_file = os.path.join(args.save_dir, "preds.csv")

        ground_truth = parse_csv(ground_truth_file)
        predictions = parse_csv(predictions_file)

        eval_results = calculate_metrics(ground_truth, predictions)
        for k, v in eval_results.items():
            print(f"{k}: {v:.4f}")
    elif args.mode == 'predict':
        print(f"Predict with segmenter {args.segmenter}")
        data_dir = os.path.join(args.root_dir, "trks", args.segmenter)
        tst_dataset = TrajPointDataset(data_dir, split="predict")
        tst_dataloader = DataLoader(tst_dataset, batch_size=4, shuffle=False, collate_fn=tst_dataset.collate_fn)
        best_models = []
        if Path(args.model_path).is_dir():
            model_paths = glob(os.path.join(args.model_path, "*.pt"))
            for model_path in model_paths:
                best_model = VideoClassifier(n_features=64, max_sequence_length=20, sample_attention="single")
                best_model.load_state_dict(torch.load(model_path, weights_only=True)["model_state_dict"])
                best_models.append(best_model)
        else:
            best_model = VideoClassifier(n_features=64, max_sequence_length=20, sample_attention="single")
            best_model.load_state_dict(torch.load(args.model_path, weights_only=True)["model_state_dict"])
            best_models.append(best_model)

        predict_and_save(best_models,
                         tst_dataloader,
                         os.path.join(args.save_dir, "preds.csv"))


if __name__ == '__main__':
    main()
