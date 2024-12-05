# Cell Behavior Video Classification (CBVCC)

A deep learning-based tool for analyzing and classifying cell behavior in microscopy videos. This tool performs cell segmentation, tracking, and classification using a combination of custom deep learning models.

## Installation

### Prerequisites
- CUDA-capable GPU (recommended)
- Conda package manager

### Setup Environment

1. Clone the repository:
```bash
git clone https://github.com/lxfhfut/TrajNet.git
cd cbvcc
```

2. Create and activate the conda environment:
```bash
conda env create -f cbvcc.yml
conda activate cbvcc
```

## Usage

The program supports three modes of operation: training, evaluation, and inference on single videos.

### Training Mode

Train a new model on your dataset:

```bash
python main.py train \
    --root_dir /path/to/dataset \
    --ckpt_dir ./checkpoints \
    --segmenter cytotorch_0 \
    --batch_size 32
```

Parameters:
- `--root_dir`: Directory containing training videos and annotations
- `--ckpt_dir`: Directory to save model checkpoints (default: ./ckpts)
- `--segmenter`: Cellpose model for segmentation (default: cytotorch_0)
- `--batch_size`: Training batch size (default: 32)
![Training Progress](./ckpts/training_results_20241129_104939.png)

### Evaluation Mode

Evaluate model performance on a test dataset:

```bash
python main.py evaluate \
    --root_dir /path/to/test/dataset \
    --model_path ./checkpoints/best_model.pt \
    --save_dir ./results \
    --segmenter cytotorch_0 \
    --batch_size 4
```

Parameters:
- `--root_dir`: Directory containing test videos
- `--model_path`: Path to trained model checkpoint
- `--save_dir`: Directory to save evaluation results (default: ./results)
- `--segmenter`: Cellpose model for segmentation (default: cytotorch_0)
- `--batch_size`: Evaluation batch size (default: 4)

### Inference Mode (Single Video)

Analyze and classify a single video:

```bash
python main.py infer \
    --video_path /path/to/video.avi \
    --model_path ./checkpoints/best_model.pt \
    --save_dir ./results \
    --segmenter cytotorch_0
```

Parameters:
- `--video_path`: Path to input video file
- `--model_path`: Path to trained model checkpoint
- `--save_dir`: Directory to save results (default: same as video directory)
- `--segmenter`: Cellpose model for segmentation (default: cytotorch_0)

## Expected Directory Structure

```
dataset/
├── trks/
│   └── cytotorch_0/
│       ├── video1/
│       │   ├── video1_imgs.tif
│       │   ├── video1_msks.tif
│       │   └── video1_track_trajectories.csv
│       └── video2/
├── test.csv
└── train.csv
```

## Output Files

For each processed video, the program generates:
- `_imgs.tif`: Original video frames
- `_msks.tif`: Segmentation masks
- `_track_trajectories.csv`: Cell tracking data
- `_{segmenter}.mp4`: Visualization of tracked cells
- For evaluation: `preds.csv` with classification results

## Notes

- The default segmentation model is 'cytotorch_0'. Other Cellpose models can be specified using the `--segmenter` argument.
- GPU acceleration is automatically used if available.
- For best results, ensure input videos are in a format compatible with OpenCV (.avi recommended).