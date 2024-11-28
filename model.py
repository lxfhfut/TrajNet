import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TrajFeatNet(nn.Module):
    def __init__(self, n_features=32, max_sequence_length=20):
        super(TrajFeatNet, self).__init__()
        self.max_sequence_length = max_sequence_length
        
        # Convolutional feature extraction layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(3, 16, kernel_size=3, padding=1),
            # nn.Conv1d(16, 32, kernel_size=3, padding=1),
            # nn.Conv1d(32, 64, kernel_size=3, padding=1)
        ])
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(16),
            # nn.BatchNorm1d(32),
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
            nn.Linear(32, n_features),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, batch_data):
        """
        Args:
            batch_data: Tuple containing:
                - padded_trajectories: List[Tensor] of shape [n_trajs, max_length, 3] for each video
                - traj_lengths: List[Tensor] of trajectory lengths for each video
                - video_lengths: Tensor of number of trajectories per video
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
    """Weight different samples from each video"""
    def __init__(self, input_dim, attention_type='single'):
        super().__init__()
        self.input_dim = input_dim
        self.attention_type = attention_type
        
        if attention_type == 'single':
            self.attention = nn.Sequential(
                nn.Linear(input_dim, 128),
                # nn.Tanh(),
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
    
    def forward(self, x):
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
    def __init__(self, n_features=32, max_sequence_length=20, sample_attention='multi_head'):
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
        
    def forward(self, batch_data):
        """
        Args:
            batch_data: Tuple containing:
                - padded_trajectories: List[Tensor] of shape [n_trajs, max_length, 3] for each video
                - traj_lengths: List[Tensor] of trajectory lengths for each video
                - video_lengths: Tensor of number of trajectories per video
                -
        Returns:
            Tensor of shape [batch_size, 1] with sigmoid probabilities
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



