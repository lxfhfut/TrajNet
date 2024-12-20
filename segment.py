import os
import cv2
import numpy as np
from glob import glob
import torch.cuda
from tqdm import tqdm
from pathlib import Path
from cellpose import models
from typing import Dict, List, Tuple, Optional, Union, Any
from csbdeep.io import save_tiff_imagej_compatible


def get_model(root_dir: str, model_name: str) -> models.CellposeModel:
    """
    Initialize and return a CellposeModel instance for cell segmentation.

    Loads either a custom retrained model or a pretrained cellpose model
    based on the model_name parameter.

    Args:
        root_dir (str): Root directory containing model files.
        model_name (str): Name of the model to load.
            Use 'cyto_retrained' for custom model or other cellpose model names.

    Returns:
        models.CellposeModel: Initialized Cellpose model instance.

    Notes:
        - Automatically detects and uses GPU if available
        - Custom model path: {root_dir}/cellpose/annotated/ckpts/models/
        - Pretrained models loaded from cellpose's default model directory
    """
    if model_name == "cyto_retrained":
        model_path = "./segmenter/cellpose_retrained_sgd.pth"
    else:
        model_path = os.fspath(models.MODEL_DIR.joinpath(model_name))
    print(f"Loading model from {model_path}")
    return models.CellposeModel(gpu=torch.cuda.is_available(),
                                pretrained_model=model_path)


def process_video_frames(root_dir: str, vid: str) -> np.ndarray:
    """
    Load and preprocess frames from a video directory.

    Loads PNG frames from the specified directory, sorts them by frame number,
    and extracts the green channel for cell segmentation.

    Args:
        root_dir (str): Root directory containing dataset.
        vid (str): Video identifier/name.

    Returns:
        np.ndarray: Array of preprocessed video frames with shape (T, H, W),
            where T is number of frames, H is height, and W is width.

    Notes:
        - Expects frames as PNG files in {root_dir}/dataset/imgs/{vid}/
        - Frame filenames should be sortable by number
        - Extracts green channel from RGB frames
    """
    # Get list of frame files
    frame_list = glob(os.path.join(root_dir, "dataset", "imgs", vid, "*.png"))

    # Sort frames by number
    sorted_frame_list = sorted(frame_list,
                               key=lambda x: int(x.split('/')[-1].split('.')[0]))

    # Load frames and extract green channel
    frames = [cv2.imread(fl)[..., 1] for fl in sorted_frame_list]
    return np.asarray(frames)


def generate_masks(model: models.CellposeModel,
                   imgs: np.ndarray,
                   model_name: str) -> np.ndarray:
    """
    Generate cell segmentation masks for video frames.

    Uses CellposeModel to detect and segment cells in each frame,
    with parameters optimized for the specific model type.

    Args:
        model (models.CellposeModel): Initialized Cellpose model.
        imgs (np.ndarray): Array of video frames with shape (T, H, W).
        model_name (str): Name of the model being used.

    Returns:
        np.ndarray: Array of segmentation masks with same shape as input.
            Each unique integer represents a different cell.

    Note:
        - Uses single channel for both image and probability map
        - Minimum cell size is 20 pixels
        - Cell diameter parameter (5) only used for non-retrained models
        - Labels are generated frame by frame to manage memory
    """
    masks = np.zeros_like(imgs)

    # Set common parameters for cell detection
    common_params = {
        'channels': [0, 0],  # Use same channel for gray/image and cell probability
        'cellprob_threshold': 0,  # Include all detected cells
        'min_size': 20  # Minimum cell size in pixels
    }

    # Add diameter parameter for non-segmenter models
    if model_name != "cyto_retrained":
        common_params['diameter'] = 5

    # Process each frame
    for t in range(imgs.shape[0]):
        labels = model.eval(imgs[t, ...], **common_params)[0]
        masks[t] = labels

    return masks


def save_results(save_dir: str,
                 vid: str,
                 imgs: np.ndarray,
                 masks: np.ndarray) -> None:
    """
    Save processed images and segmentation masks.

    Saves both original images and generated masks as ImageJ-compatible
    TIFF files with proper time axis encoding.

    Args:
        save_dir (str): Directory to save results.
        vid (str): Video identifier/name.
        imgs (np.ndarray): Array of processed images.
        masks (np.ndarray): Array of generated segmentation masks.

    Note:
        - Creates output directory if it doesn't exist
        - Saves as {vid}_imgs.tif and {vid}_msks.tif
        - Uses TYX axis ordering for ImageJ compatibility
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_tiff_imagej_compatible(os.path.join(save_dir, f'{vid}_imgs.tif'),
                                imgs, axes='TYX')
    save_tiff_imagej_compatible(os.path.join(save_dir, f'{vid}_msks.tif'),
                                masks, axes='TYX')


def segment_single_video(root_dir: str,
                         vid: str,
                         out_dir: str,
                         model_name: str) -> None:
    """
    Process and segment a single video.

    Executes complete segmentation pipeline for one video:
    1. Load model
    2. Process video frames
    3. Generate segmentation masks
    4. Save results

    Args:
        root_dir (str): Root directory containing dataset.
        vid (str): Video identifier/name.
        out_dir (str): Output directory for results.
        model_name (str): Name of the model to use.

    Note:
        - Results are saved in {out_dir}/{model_name}/{vid}/
        - Creates new model instance for each video
    """
    model = get_model(root_dir, model_name)
    imgs = process_video_frames(root_dir, vid)
    masks = generate_masks(model, imgs, model_name)
    save_dir = os.path.join(out_dir, model_name, vid)
    save_results(save_dir, vid, imgs, masks)


def segment_videos(root_dir: str,
                  out_dir: str,
                  model_name: str) -> None:
    """
    Batch process multiple videos for segmentation.

    Processes all video directories in the dataset with progress tracking,
    using a single model instance for efficiency.

    Args:
        root_dir (str): Root directory containing dataset.
        out_dir (str): Output directory for results.
        model_name (str): Name of the model to use.

    Note:
        - Processes videos found in {root_dir}/dataset/imgs/
        - Shows progress bar during processing
        - Reuses same model instance across all videos
        - Results organized by model name and video ID
    """
    # Get list of video folders
    # vid_folders = [f.name for f in os.scandir(os.path.join(root_dir, "dataset", "imgs"))
    #                if f.is_dir()]
    vid_folders = ["04_3"]

    # Initialize model once for all videos
    model = get_model(root_dir, model_name)

    # Process each video
    for vid in tqdm(vid_folders):
        imgs = process_video_frames(root_dir, vid)
        masks = generate_masks(model, imgs, model_name)
        save_dir = os.path.join(out_dir, model_name, vid)
        save_results(save_dir, vid, imgs, masks)


if __name__ == "__main__":
    root_dir = '/Users/lxfhfut/Dropbox/Garvan/CBVCC/'
    out_dir = "/Users/lxfhfut/Desktop/test/"  # './dataset/trks/'
    segment_videos(root_dir, out_dir, "cyto_retrained")
