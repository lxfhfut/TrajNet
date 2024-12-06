"""
Cell segmentation module using Cellpose.

This module provides functionality to segment cells in microscopy videos using the Cellpose
deep learning model. It supports both pre-trained and custom segmenter models for cell detection
and segmentation.

The module processes video frames sequentially and generates binary masks for detected cells.
Results are saved in a compatible format for downstream analysis.
"""

import os
import cv2
import numpy as np
from glob import glob
import torch.cuda
from tqdm import tqdm
from pathlib import Path
from cellpose import models
from csbdeep.io import save_tiff_imagej_compatible


def get_model(root_dir, model_name):
    """
    Initialize and return a CellposeModel instance.

    Args:
        root_dir (str): Root directory containing model files
        model_name (str): Name of the model to load ('cyto_retrained' or other cellpose models)

    Returns:
        models.CellposeModel: Initialized Cellpose model instance

    Notes:
        For 'cyto_retrained', loads from a custom path. Otherwise loads pretrained models
        from cellpose's model directory.
    """
    print(f"Loading model {model_name}")
    if model_name == "cyto_retrained":
        model_path = os.path.join(root_dir, "cellpose", "annotated", "ckpts",
                                  "models", "cellpose_retrained_sgd.pth")
    else:
        model_path = os.fspath(models.MODEL_DIR.joinpath(model_name))

    return models.CellposeModel(gpu=torch.cuda.is_available(),
                                pretrained_model=model_path)


def process_video_frames(root_dir, vid):
    """
    Load and process frames from a video directory.

    Args:
        root_dir (str): Root directory containing dataset
        vid (str): Video identifier

    Returns:
        np.ndarray: Array of processed video frames

    Notes:
        Loads PNG frames and extracts the green channel for processing.
    """
    # Get list of frame files
    frame_list = glob(os.path.join(root_dir, "dataset", "imgs", vid, "*.png"))

    # Sort frames by number
    sorted_frame_list = sorted(frame_list,
                               key=lambda x: int(x.split('/')[-1].split('.')[0]))

    # Load frames and extract green channel
    frames = [cv2.imread(fl)[..., 1] for fl in sorted_frame_list]
    return np.asarray(frames)


def generate_masks(model, imgs, model_name):
    """
    Generate cell masks for all frames in a video.

    Args:
        model (models.CellposeModel): Initialized Cellpose model
        imgs (np.ndarray): Array of video frames
        model_name (str): Name of the model being used

    Returns:
        np.ndarray: Array of generated masks

    Notes:
        Uses different parameters for segmenter vs pretrained models.
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


def save_results(save_dir, vid, imgs, masks):
    """
    Save processed images and masks as TIFF files.

    Args:
        save_dir (str): Directory to save results
        vid (str): Video identifier
        imgs (np.ndarray): Array of processed images
        masks (np.ndarray): Array of generated masks

    Notes:
        Saves files in ImageJ-compatible TIFF format with time axis.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_tiff_imagej_compatible(os.path.join(save_dir, f'{vid}_imgs.tif'),
                                imgs, axes='TYX')
    save_tiff_imagej_compatible(os.path.join(save_dir, f'{vid}_msks.tif'),
                                masks, axes='TYX')


def segment_single_video(root_dir, vid, out_dir, model_name):
    """
    Process and segment a single video.

    Args:
        root_dir (str): Root directory containing dataset
        vid (str): Video identifier
        out_dir (str): Output directory for results
        model_name (str): Name of the model to use

    Notes:
        Handles complete pipeline from loading to saving results.
    """
    model = get_model(root_dir, model_name)
    imgs = process_video_frames(root_dir, vid)
    masks = generate_masks(model, imgs, model_name)
    save_dir = os.path.join(out_dir, model_name, vid)
    save_results(save_dir, vid, imgs, masks)


def segment_videos(root_dir, out_dir, model_name):
    """
    Process multiple videos with progress tracking.

    Args:
        root_dir (str): Root directory containing dataset
        out_dir (str): Output directory for results
        model_name (str): Name of the model to use

    Notes:
        Uses tqdm for progress tracking during batch processing.
    """
    # Get list of video folders
    vid_folders = [f.name for f in os.scandir(os.path.join(root_dir, "dataset", "imgs"))
                   if f.is_dir()]

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
    out_dir = '/Users/lxfhfut/Dropbox/Garvan/CBVCC/dataset/trks/'
    segment_videos(root_dir, out_dir, "cytotorch_0")
