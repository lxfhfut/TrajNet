import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Dict


class TrajFeatNet(nn.Module):
    """
        Neural network module for extracting features from trajectories.

        Processes trajectory data through convolutional layers and attention mechanisms
        to extract meaningful features for classification.

        Args:
            n_features (int): Number of output features. Defaults to 32.
            max_sequence_length (int): Maximum length of input trajectories. Defaults to 20.

        Attributes:
            max_sequence_length (int): Maximum trajectory length
            conv_layers (nn.ModuleList): List of 1D convolutional layers
            batch_norms (nn.ModuleList): List of batch normalization layers
            traj_attention (nn.MultiheadAttention): Trajectory-level attention mechanism
            feature_extractor (nn.Sequential): Final feature extraction layers
    """

    def __init__(self, n_features: int = 32, max_sequence_length: int = 20):
        super(TrajFeatNet, self).__init__()
        self.max_sequence_length = max_sequence_length
        
        # Convolutional feature extraction layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(3, 16, kernel_size=3, padding=1),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            # nn.Conv1d(32, 64, kernel_size=3, padding=1)
        ])
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(16),
            nn.BatchNorm1d(32),
            # nn.BatchNorm1d(64)
        ])
        
        # Trajectory-level attention
        self.traj_attention = nn.MultiheadAttention(
            embed_dim=16,
            num_heads=4,
            batch_first=True
        )
        
        # Final feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(64, n_features),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, batch_data: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
        """
        Process a batch of trajectory data.

        Args:
            batch_data: Tuple containing:
                - padded_trajectories (List[torch.Tensor]): Trajectories [n_trajs, max_length, 3]
                - traj_lengths (List[torch.Tensor]): Length of each trajectory
                - video_lengths (torch.Tensor): Number of trajectories per video

        Returns:
            List[torch.Tensor]: Processed features for each video in the batch

        Note:
            - Applies convolutions with masking based on trajectory lengths
            - Uses global pooling to get trajectory-level features
            - Features are extracted independently for each video
        """
        padded_trajectories, traj_lengths, video_lengths = batch_data
        batch_size = len(padded_trajectories)
        device = padded_trajectories[0].device
        
        # Process each video's trajectories
        video_features = []
        
        for i in range(batch_size):
            trajs = padded_trajectories[i]  # [n_trajs, max_length, 3]
            lengths = traj_lengths[i]  # [n_trajs]
            n_trajs = trajs.size(0)
            
            # Create trajectory-level mask
            traj_mask = torch.arange(self.max_sequence_length, device=device).expand(n_trajs, -1) < lengths.unsqueeze(1)
            
            # Process trajectories
            x = trajs.transpose(1, 2)  # [n_trajs, 3, max_length]
            
            # Apply convolutions with masking
            for conv, bn in zip(self.conv_layers, self.batch_norms):
                x = conv(x)
                x = bn(x)
                x = F.relu(x)
                x = x * traj_mask.unsqueeze(1)
            
            # Prepare for attention [n_trajs, max_length, 64]
            x = x.transpose(1, 2)
            
            # # Apply trajectory-level attention
            # traj_attn_output, _ = self.traj_attention(x, x, x, key_padding_mask=~traj_mask)
            
            # Global pooling for each trajectory
            masked_output = x * traj_mask.unsqueeze(-1)
            traj_features_mean = masked_output.sum(dim=1) / lengths.unsqueeze(1)  # [n_trajs, 64]
            traj_features_max = torch.max(masked_output, dim=1)[0]
            traj_features = torch.concat((traj_features_mean, traj_features_max), dim=1)
            
            video_features.append(self.feature_extractor(traj_features))
        
        return video_features


class SampleAttention(nn.Module):
    """
       Attention mechanism for weighting different samples within a video.

       Supports multiple attention types: single, multi-head, and scaled dot-product.

       Args:
           input_dim (int): Dimension of input features
           attention_type (str): Type of attention mechanism
               Options: 'single', 'multi_head', 'scaled_dot'

       Attributes:
           input_dim (int): Input feature dimension
           attention_type (str): Selected attention mechanism
           attention (nn.Sequential): Single attention network
           num_heads (int): Number of attention heads for multi-head attention
           head_dim (int): Dimension of each attention head
           temperature (float): Scaling factor for dot-product attention
    """

    def __init__(self, input_dim: int, attention_type: str = 'single'):
        super().__init__()
        self.input_dim = input_dim
        self.attention_type = attention_type
        
        if attention_type == 'single':
            self.attention = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
        
        elif attention_type == 'multi_head':
            self.num_heads = 4
            self.head_dim = input_dim // self.num_heads
            self.query = nn.Linear(input_dim, input_dim)
            self.key = nn.Linear(input_dim, input_dim)
            self.value = nn.Linear(input_dim, input_dim)
            self.out = nn.Linear(input_dim, input_dim)
            
        elif attention_type == 'scaled_dot':
            self.temperature = np.sqrt(input_dim)
            self.query = nn.Linear(input_dim, input_dim)
            self.key = nn.Linear(input_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism to input features.

        Args:
            x (torch.Tensor): Input features
                Shape: [seq_len, feature_dim] or [batch_size, seq_len, feature_dim]

        Returns:
            torch.Tensor: Weighted feature representation
                Shape: [feature_dim]

        Note:
            - Handles empty sequences by returning zero tensor
            - Different behavior based on attention_type:
                - single: Direct attention weights
                - multi_head: Multiple attention heads with concatenation
                - scaled_dot: Scaled dot-product attention
        """
        # Handle empty sequence case
        if x.size(0) == 0:  # Empty sequence
            if self.attention_type == 'single':
                return torch.zeros(self.input_dim, device=x.device)
            else:  # multi_head or scaled_dot
                return torch.zeros(self.input_dim, device=x.device)

        if self.attention_type == 'single':
            weights = self.attention(x)
            weights = F.softmax(weights, dim=0)
            return torch.sum(x * weights, dim=0)
            
        elif self.attention_type == 'multi_head':
            # Add batch dimension if not present
            if x.dim() == 2:
                x = x.unsqueeze(0)  # [1, seq_len, feature_dim]

            batch_size, seq_len, _ = x.size()
            
            # Split into multiple heads
            Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Scaled dot-product attention for each head
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            weights = F.softmax(scores, dim=-1)
            
            # Apply attention and concatenate heads
            attention = torch.matmul(weights, V)
            attention = attention.transpose(1, 2).contiguous()
            attention = attention.view(batch_size, seq_len, self.num_heads * self.head_dim)
            
            # Final linear projection and sample pooling
            output = self.out(attention)
            return output.mean(dim=1).squeeze(0)  # Remove batch dim and average across sequence
            
        elif self.attention_type == 'scaled_dot':
            if x.dim() == 2:
                x = x.unsqueeze(0)
            
            Q = self.query(x)
            K = self.key(x)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.temperature
            weights = F.softmax(scores, dim=-1)
            return torch.matmul(weights, x).mean(dim=1).squeeze(0)


class VideoClassifier(nn.Module):
    """
        Complete video classification model using trajectory features.

        Combines trajectory feature extraction with attention-based aggregation
        and binary classification.

        Args:
            n_features (int): Number of features to extract per trajectory. Defaults to 32.
            max_sequence_length (int): Maximum trajectory length. Defaults to 20.
            sample_attention (str): Type of attention for sample weighting. Defaults to 'single'.

        Attributes:
            feat_extractor (TrajFeatNet): Trajectory feature extraction module
            input_dim (int): Dimension of extracted features
            sample_attention (SampleAttention): Attention mechanism for sample weighting
            classifier (nn.Sequential): Classification layers
    """

    def __init__(self, n_features: int = 32,
                 max_sequence_length: int = 20,
                 sample_attention: str = 'single'):

        super().__init__()

        self.feat_extractor = TrajFeatNet(n_features, max_sequence_length)
        self.input_dim = n_features
 
        # Sample attention for weighting different samples
        self.sample_attention = SampleAttention(
            self.input_dim, 
            attention_type=sample_attention
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch_data: Tuple[List[torch.Tensor],
                                          List[torch.Tensor],
                                          torch.Tensor]) -> torch.Tensor:
        """
        Process a batch of videos for classification.

        Args:
            batch_data: Tuple containing:
                - padded_trajectories (List[torch.Tensor]): Trajectories [n_trajs, max_length, 3]
                - traj_lengths (List[torch.Tensor]): Length of each trajectory
                - video_lengths (torch.Tensor): Number of trajectories per video

        Returns:
            torch.Tensor: Classification probabilities [batch_size, 1]

        Note:
            - Extracts features for each trajectory
            - Aggregates features using attention
            - Returns sigmoid probabilities for binary classification
        """
        
        # x is List[Tensor], each of shape (m_i, d), where m_i is the number of trajectories and d is n_features
        x = self.feat_extractor(batch_data)

        # Aggregate features for each video in the batch
        aggregated = []
        for features in x:
            # Apply sample attention
            agg_features = self.sample_attention(features)
            
            aggregated.append(agg_features)
        
        # Stack aggregated features
        x = torch.stack(aggregated)
        
        # Classification
        return self.classifier(x)



