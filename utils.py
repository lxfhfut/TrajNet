import os
import gc
import cv2
import napari
import numpy as np
import pandas as pd
from tqdm import tqdm
import tifffile as tiff
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any


class TrainingVisualizer:
    """
       A class for real-time visualization of training metrics during model training.

       This class creates and updates a matplotlib figure with two subplots:
       one for tracking training/validation loss and another for accuracy metrics.

       Args:
           num_epochs (int): The total number of training epochs.

       Attributes:
           fig (matplotlib.figure.Figure): The main figure object containing both subplots.
           ax1 (matplotlib.axes.Axes): The subplot for loss visualization.
           ax2 (matplotlib.axes.Axes): The subplot for accuracy visualization.
           train_losses (List[float]): List to store training loss values.
           val_losses (List[float]): List to store validation loss values.
           val_accuracies (List[float]): List to store validation accuracy values.
           train_accuracies (List[float]): List to store training accuracy values.
       """
    def __init__(self, num_epochs: int) -> None:
        # Prepare figure with subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 10))
        self.fig.suptitle('Training and Validation Metrics')

        # Training loss plot
        self.ax1.set_title('Training Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')

        # Accuracy plot
        self.ax2.set_title('Validation Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')

        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.train_accuracies = []

        # Line objects for dynamic plotting
        self.train_line, = self.ax1.plot([], [], 'b-')
        self.val_line, = self.ax2.plot([], [], 'r-')

        # Prepare plot
        plt.tight_layout()

    def update(self,
               train_loss: float,
               val_loss: float,
               train_acc: float,
               val_acc: float) -> None:
        """
        Update the visualization with new metrics from the current epoch.

        Args:
            train_loss (float): Current epoch's training loss.
            val_loss (float): Current epoch's validation loss.
            train_acc (float): Current epoch's training accuracy.
            val_acc (float): Current epoch's validation accuracy.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.train_accuracies.append(train_acc)

        # Update loss plot
        self.ax1.clear()
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.plot(self.train_losses, 'b--', label="Training Loss")
        self.ax1.plot(self.val_losses, 'g-', label="Validation Loss")
        self.ax1.legend()

        # Update accuracy plot
        self.ax2.clear()
        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.plot(self.val_accuracies, 'r-', label="Validation Acc.")
        self.ax2.plot(self.train_accuracies, 'm--', label="Training Acc.")
        self.ax2.legend()

        # Redraw
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    def save(self, filename: str = 'training_metrics.png') -> None:
        """
        Save the final visualization

        Args:
            filename: Output filename for the plot
        """
        plt.savefig(filename)
        plt.close(self.fig)


def tracks_csv(csv_path: str,
               track_ids: Optional[np.ndarray] = None,
               sample_rate: int = 1) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
    """
    Read and process tracking data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing tracking data.
        track_ids (np.ndarray, optional): Specific track IDs to process.
            If None, all tracks are processed. Defaults to None.
        sample_rate (int, optional): Sampling rate for track selection.
            Defaults to 1 (use all tracks).

    Returns:
        Tuple containing:
            - np.ndarray: Processed tracks data
            - Dict[str, np.ndarray]: Features dictionary with 'frame' and 'track_id' arrays
            - Dict: Empty graph dictionary (placeholder for future use)
    """
    data = pd.read_csv(csv_path)
    column_idx = data.columns.get_indexer(['particle', 'frame', 'x', 'y'])
    tracks_data = data.iloc[:, column_idx].values
    tracks = []
    if track_ids is None:
        track_ids = np.unique(tracks_data[:, 0])
    else:
        sample_rate = 1

    track_ids_subset = track_ids[::sample_rate]
    for track_id in track_ids_subset:
        row_indices = tracks_data[:, 0] == track_id
        tracks.append(tracks_data[row_indices, :])

    tracks = np.concatenate(tracks, axis=0)
    features = {'frame': tracks[:, 1], 'track_id': tracks[:, 0]}

    graph = {}
    return tracks, features, graph


def vis_tracks(data_dir: str,
               name: str,
               track_ids: Optional[np.ndarray] = None) -> napari.Viewer:
    """
    Visualize tracking data using napari viewer.

    Creates a napari viewer instance with multiple layers:
    - Raw image (green colormap)
    - Mask image (hsv colormap, initially hidden)
    - Points layer showing track positions
    - Tracks layer showing particle trajectories

    Args:
        data_dir (str): Directory containing the image and tracking data.
        name (str): Name of the dataset/experiment.
        track_ids (np.ndarray, optional): Specific track IDs to visualize.
            If None, all tracks are shown. Defaults to None.

    Returns:
        napari.Viewer: Configured napari viewer instance with all layers added.
    """
    viewer = napari.Viewer(ndisplay=2)
    viewer.title = name
    img_path = os.path.join(data_dir, name, name + "_imgs.tif")
    msk_path = os.path.join(data_dir, name, name + "_msks.tif")
    raw_img = tiff.imread(img_path)
    msk_img = tiff.imread(msk_path)
    viewer.add_image(raw_img, name="raw_image", colormap='green')
    viewer.add_image(msk_img, name="msk_image", colormap='hsv', opacity=0.3, visible=False)

    trk_data_csv = os.path.join(data_dir, name, name + '_track_trajectories.csv')
    tracks, feats, graph = tracks_csv(trk_data_csv, track_ids)
    text = {
        'string': '{track_id:.0f}',
        'size': 20,
        'color': 'red',
    }
    vertices = tracks[:, 1:]
    # feats = {'track_id': tracks[:, 0], 'frame': tracks[:, 1]}
    viewer.add_points(vertices, features=feats, text=text, size=0, name='points', visible=True)
    viewer.add_tracks(tracks, features=feats, name='tracks', blending="opaque",
                      colormap="turbo", color_by='frame', tail_width=5)
    return viewer


def save_napari_animation(viewer: napari.Viewer,
                          output_path: str,
                          fps: int = 5) -> None:
    """
    Save a napari viewer animation as an MP4 video file.

    Takes the current state of a napari viewer and creates a video by
    capturing each frame as the time dimension is stepped through.

    Args:
        viewer (napari.Viewer): The napari viewer instance containing the animation.
        output_path (str): Path where the video should be saved (e.g., 'output.mp4').
        fps (int, optional): Frames per second for the output video. Defaults to 30.

    Note:
        - The function assumes the first dimension is time
        - The output video uses MP4V codec
        - The viewer's canvas is captured without any GUI elements
    """
    # Get the dimensions
    dims = viewer.dims
    current_step = dims.current_step

    # Get the time dimension range (first dimension in this case)
    time_range = dims.range[0]
    total_frames = int((time_range.stop - time_range.start) / time_range.step)

    # Get first frame to determine video size
    screenshot = viewer.screenshot(canvas_only=True)
    height, width = screenshot.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        # Save each frame
        for i in tqdm(range(total_frames), desc="Saving frames"):
            # Set the time dimension while keeping others constant
            dims.set_current_step(0, i)

            # Capture the viewer as an image
            screenshot = viewer.screenshot(canvas_only=True)

            # Convert from RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

            # Write the frame
            out.write(bgr_frame)

    finally:
        # Release video writer
        out.release()
        gc.collect()


if __name__ == "__main__":
    data_dir = "/Users/lxfhfut/Dropbox/Garvan/CBVCC/dataset/trks/cyto_retrained/"
    vid = "02_1"
    vis_tracks(data_dir, name=vid)
    napari.run()
