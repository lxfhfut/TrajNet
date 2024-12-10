import os
import torch
import numpy as np
import pandas as pd
import os.path as osp
from torch.utils.data import Dataset


class TrajPointDataset(Dataset):
    def __init__(self, data_dir, split=None, max_length=20, augment=False):
        self.data_dir = data_dir
        self.max_length = max_length
        self.augment = augment

        # Initialize augmentation parameters if augmentation is enabled
        if augment:
            self.rotation_angle = 15
            self.scale_range = (0.9, 1.1)
            self.noise_level = 0.5
            self.reverse_prob = 0.3
            self.speed_range = (0.9, 1.1)
            self.interp_prob = 0.5
            self.max_points_to_add = 5

        # Load data
        annotations = pd.read_csv(osp.join(osp.dirname(osp.dirname(data_dir)), "train.csv" if split == "training" else "test.csv"))
        vids = [f.name for f in os.scandir(data_dir) if f.is_dir()]
        video_data = []
        labels = []
        video_ids = []

        for vid in vids:
            vid_dir = osp.join(data_dir, vid)

            if os.path.exists(osp.join(vid_dir, f"{vid}_track_trajectories.csv")):
                data = pd.read_csv(osp.join(vid_dir, f"{vid}_track_trajectories.csv"))
            else:
                continue

            result = annotations.loc[annotations['id'] == vid, 'action_label'].values
            if result.size == 1:
                labels.append(result[0])
            else:
                continue

            data.replace([np.inf, -np.inf], 0, inplace=True)
            tracks_data = data.loc[:, ['particle', 'frame', 'x', 'y', 'area']]
            track_ids = tracks_data['particle'].unique()

            track_trajs = []
            for track_id in track_ids:
                track = tracks_data.loc[tracks_data['particle'] == track_id].sort_values(by=['frame'])
                pos_idx = track.columns.get_indexer(['frame', 'x', 'y'])
                track_trajs.append(torch.tensor(track.iloc[:, pos_idx].values))

            video_data.append(track_trajs)
            video_ids.append(vid)

        self.video_data = video_data
        self.labels = labels
        self.video_ids = video_ids

    def __len__(self):
        return len(self.labels)

    def _interp_trajectory(self, traj):
        if len(traj) >= self.max_length:
            return traj

        available_slots = self.max_length - len(traj)
        num_points_to_add = np.random.randint(1, min(self.max_points_to_add + 1, available_slots + 1))

        # Randomly select positions to interpolate
        possible_positions = list(range(len(traj) - 1))
        if not possible_positions:  # If trajectory has only one point
            return traj

        positions_to_interpolate = np.random.choice(
            possible_positions,
            size=min(num_points_to_add, len(possible_positions)),
            replace=False
        )
        # print(f"Before interp.: {traj}")
        new_trajectory = []
        for i in range(len(traj)):
            new_trajectory.append(traj[i])
            if i in positions_to_interpolate:
                # Linear interpolation between current and next point
                t = np.random.random()  # Random point between 0 and 1
                interp_point = t * traj[i] + (1 - t) * traj[i + 1]
                new_trajectory.append(interp_point)
        # print(f"After interp.: {torch.stack(new_trajectory)}")
        return torch.stack(new_trajectory)

    def _rotate_trajectory(self, traj):
        """Randomly rotate trajectory in 2D space"""
        angle = np.random.uniform(-self.rotation_angle, self.rotation_angle)
        angle_rad = np.deg2rad(angle)
        rot_matrix = torch.tensor([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ], dtype=traj.dtype)

        time = traj[:, 0:1]
        xy = traj[:, 1:]
        rotated_xy = xy @ rot_matrix.T

        return torch.cat([time, rotated_xy], dim=1)

    def _scale_trajectory(self, traj):
        """Randomly scale the spatial coordinates"""
        scale = np.random.uniform(*self.scale_range)
        scaled_traj = traj.clone()
        scaled_traj[:, 1:] *= scale
        return scaled_traj

    def _add_noise(self, traj):
        """Add Gaussian noise to spatial coordinates"""
        # print(f"Before noise.: {traj}")
        noise = torch.randn_like(traj[:, 1:]) * self.noise_level
        noisy_traj = traj.clone()
        noisy_traj[:, 1:] += noise
        noisy_traj[:, 1:] = torch.clamp(noisy_traj[:, 1:], min=0.0)
        # print(f"After noise.: {noisy_traj}")

        return noisy_traj

    def _reverse_time(self, traj):
        """Randomly reverse the trajectory"""
        if np.random.random() < self.reverse_prob:
            reversed_traj = torch.flip(traj, [0])
            reversed_traj[:, 0] = reversed_traj[-1, 0] - reversed_traj[:, 0]
            return reversed_traj
        return traj

    def _modify_speed(self, traj):
        """Randomly modify the speed of the trajectory"""
        speed_factor = np.random.uniform(*self.speed_range)
        modified_traj = traj.clone()
        modified_traj[:, 0] *= speed_factor
        return modified_traj

    def _augment_trajectory(self, traj):
        """Apply random augmentations to trajectory"""
        augmented = traj.clone()

        augmentations = [
            (self._rotate_trajectory, 0.5),
            (self._scale_trajectory, 0.5),
            # (self._add_noise, 0.5),
            (self._reverse_time, 0.3),
            (self._modify_speed, 0.3),
            (self._interp_trajectory, 0.3)
        ]

        for aug_fn, prob in augmentations:
            if np.random.random() < prob:
                augmented = aug_fn(augmented)

        return augmented

    def __getitem__(self, idx):
        trajectories = self.video_data[idx]

        if self.augment:
            # Augment each trajectory in the video
            augmented_trajectories = []
            for traj in trajectories:
                aug_traj = self._augment_trajectory(traj)
                augmented_trajectories.append(aug_traj)
            trajectories = augmented_trajectories

        return trajectories, self.labels[idx], self.video_ids[idx]

    def collate_fn(self, batch):
        return prepare_batch(batch, self.max_length)


# prepare_batch function remains the same
def prepare_batch(batch, max_length=20):
    # Unpack batch
    trajectory_lists, labels, video_ids = zip(*batch)

    # Process each video's trajectories
    padded_trajectories = []
    traj_lengths = []
    video_lengths = torch.tensor([len(traj_list) for traj_list in trajectory_lists])

    for video_trajs in trajectory_lists:
        # Get lengths of each trajectory in this video
        lengths = torch.tensor([len(traj) for traj in video_trajs])

        # Create padded tensor for this video's trajectories
        n_trajs = len(video_trajs)
        padded = torch.zeros(n_trajs, max_length, 3)

        # Fill in trajectories
        for i, traj in enumerate(video_trajs):
            length = lengths[i]
            padded[i, :length] = traj[:max_length]  # Truncate if longer than max_length
            padded[i, length:] = traj[length-1]

        padded_trajectories.append(padded)
        traj_lengths.append(lengths)

    return padded_trajectories, traj_lengths, video_lengths, torch.tensor(labels), video_ids
