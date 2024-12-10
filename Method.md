# Method

We present a trajectory-based approach for analyzing and classifying cellular motion patterns in intravital microscopy videos. Our method consists of several key components: cell tracking, feature extraction using a convolutional neural network, attention-based trajectory selection, and classification. The method is designed to be robust to tracking errors while capturing essential motion characteristics.

## Cell Tracking and Trajectory Extraction

The initial step involves tracking cells of interest across video frames. We utilized Cellpose for cell segmentation and combine it with trackpy/laptrack for trajectory generation. For computational efficiency and to reduce noise, we processed only the green channel of each video frame. This process generates $n_i$ trajectories for each video $i$, where each trajectory represents the temporal evolution of a cell's position. To ensure reliable tracking, we focused on cells with high intensity (>50) and substantial size (>50 pixels), which helps filter out potential segmentation and tracking errors.

## Feature Extraction Network

For feature extraction, we employed a shallow two-layer convolutional neural network. This architectural choice is motivated by the observation that key motion features—such as speed, acceleration, and turning angles—can be effectively captured through first and second-order derivatives. The network processes trajectories of varying lengths by:

1. Padding each trajectory to a fixed size of 3×20, where:
   - 3 dimensions represent (x, y, t) coordinates
   - 20 frames represent the maximum temporal extent

2. Applying two convolutional layers with:
   - First layer: 3→16 channels, kernel size 3
   - Second layer: 16→32 channels, kernel size 3
   - ReLU activation and batch normalization

3. Max-pooling and Avg-pooling operations to capture salient motion patterns

## Attention-based Trajectory Selection

To identify and focus on the most informative trajectories within each video, we implemented an attention mechanism that:
1. Computes attention weights for each trajectory's features
2. Generates a video-level representation through weighted averaging of trajectory features
3. Mimics human expert behavior in focusing on salient motion patterns

Let $f_i^j$ denote the feature vector of the j-th trajectory in video $i$. The attention weight $α_i^j$ for each trajectory is computed as:

$$
α_i^j = exp(w^T f_i^j) / Σ_k exp(w^T f_i^k)
$$

where $w$ is a learnable parameter vector. The final video representation $f_i$ for each video is then computed as:

$$
f_i = Σ_j α_i^j f_i^j
$$

This approach allows the model to automatically learn which trajectories are most relevant for classification, similar to how human experts would select trajectories of interest for decision-making.

## Classification

The final classification stage consists of:
- A multilayer perceptron (MLP) for primary classification
- Integration with feature extraction and attention modules for end-to-end training
- After training, this module can be replaced with SVM classifier (but we found that they achieved comparable performance)
## Addressing Overfitting

Given the limited dataset size (210 videos with approximately 500 trajectories), we implemented several strategies to combat overfitting:

### Model Complexity Reduction
- Shallow network architecture
- Limited number of parameters
- Dropout layers (0.5 dropout rate)

### Training Regularization
- Weight decay during optimization

### Data Augmentation
We implemented comprehensive trajectory augmentation techniques:
- Rotation: Random rotation within ±15 degrees
- Scaling: Random scaling between 0.8-1.2×
- Time reversal: Random trajectory reversal
- Speed modification: Random speed adjustments (0.8-1.2×)
- Interpolation: Random trajectory interpolation

## Robustness Considerations

Despite relying on segmentation and tracking results, our method demonstrates robust performance due to:
1. Selective processing of high-quality cell tracks
2. Attention mechanism's ability to focus on reliable trajectories
3. Comprehensive augmentation strategy
