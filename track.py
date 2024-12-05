import os
import numpy as np
import pandas as pd
import trackpy as tp
import os.path as osp
import tifffile as tiff
from laptrack import LapTrack
from skimage.measure import regionprops


def features_from_masks(imgs_path, masks_path, int_thres=20, sz_thres=50):
    """
        Extract cell features from image and mask pairs.

        Args:
            imgs_path (str): Path to original images TIFF file
            masks_path (str): Path to segmentation masks TIFF file
            int_thres (int): Intensity threshold for cell detection
            sz_thres (int): Size threshold for cell detection

        Returns:
            pd.DataFrame: DataFrame containing extracted cell features

        Notes:
            Features include position, size, shape, and intensity measurements.
    """

    imgs = tiff.imread(imgs_path)
    msks = tiff.imread(masks_path)

    imgs_xyt = np.moveaxis(imgs, [0, 1, 2], [2, 1, 0])
    msks_xyt = np.moveaxis(msks, [0, 1, 2], [2, 1, 0])
    frame = []
    x = []
    y = []
    mass = []
    size_x = []
    size_y = []
    area = []
    solidity = []
    eccentricity = []
    intensity_mean = []

    for t in range(imgs_xyt.shape[-1]):
        props = regionprops(msks_xyt[..., t], imgs_xyt[..., t])
        sorted_props = sorted(props, key=lambda x: x.label)
        for prop in sorted_props:
            if prop.intensity_mean < int_thres or prop.area < sz_thres:
                continue
            frame.append(t)
            # probably can try centroid_weighted to see any improvement
            x.append(prop.centroid[1])
            y.append(prop.centroid[0])
            intensity_mean.append(prop.intensity_mean)
            mass.append(prop.area * prop.intensity_mean)
            size_x.append(prop.bbox[3] - prop.bbox[1])
            size_y.append(prop.bbox[2] - prop.bbox[0])
            area.append(prop.area)
            eccentricity.append(prop.eccentricity)
            solidity.append(prop.solidity)

    d = {'x': x,
         'y': y,
         'mass': mass,
         'size_x': size_x,
         'size_y': size_y,
         'eccentricity': eccentricity,
         'solidity': solidity,
         'intensity_mean': intensity_mean,
         'area': area,
         'frame': frame}
    df = pd.DataFrame(data=d)
    return df


def tracking(df, result_csv, tracker="laptrack", max_gap=12, min_len=5, min_int=50):
    """
        Perform cell tracking using specified algorithm.

        Args:
            df (pd.DataFrame): DataFrame containing cell features
            result_csv (str): Path to save tracking results
            tracker (str): Tracking algorithm to use ('laptrack' or 'trackpy')
            max_gap (int): Maximum allowed frame gap between tracks
            min_len (int): Minimum track length to keep
            min_int (int): Minimum intensity threshold

        Returns:
            pd.DataFrame: DataFrame containing track information

        Notes:
            Supports both LapTrack and TrackPy algorithms with configurable parameters.
        """
    if tracker == "laptrack":
        lt = LapTrack(
            track_dist_metric="sqeuclidean",
            splitting_dist_metric="sqeuclidean",
            merging_dist_metric="sqeuclidean",
            track_cost_cutoff=max_gap**2,
            splitting_cost_cutoff=False,  # or False for non-splitting case
            merging_cost_cutoff=False,  # or False for non-merging case
        )

        track_df, split_df, merge_df = lt.predict_dataframe(
            df,
            coordinate_cols=[
                "x",
                "y",
            ],  # the column names for the coordinates
            frame_col="frame",
            only_coordinate_cols=False
        )
        column_mapping = {'track_id': 'particle', 'frame_y': 'frame'}
        t = track_df.rename(columns=column_mapping)

    elif tracker == "trackpy":
        t = tp.link(df, max_gap, pos_columns=['y', 'x'],
                    memory=3, adaptive_stop=5, adaptive_step=0.9)
    else:
        pass

    t1 = tp.filter_stubs(t, min_len)
    t2 = t1[t1['intensity_mean'] > min_int]
    print('Number of tracks:', t2['particle'].nunique())

    if not t2.empty:
        t2.to_csv(result_csv, index=False)

    return t2


def extract_track_feats(csv_path):
    """
        Extract statistical features from cell tracks.

        Args:
            csv_path (str): Path to CSV file containing track data

        Notes:
            Calculates and saves various track statistics including:
            - Displacement and duration
            - Velocity and acceleration metrics
            - Direction changes and angular metrics
            - Track shape characteristics (curvature, tortuosity)
    """
    data = pd.read_csv(csv_path)
    data.replace([np.inf, -np.inf], 0, inplace=True)
    tracks_data = data.loc[:, ['particle', 'frame', 'x', 'y', 'area']]
    track_ids = tracks_data['particle'].unique()

    # define features
    track_displacement = []
    track_duration = []
    track_length = []
    track_velocity_mean = []
    track_velocity_max = []
    track_velocity_std = []
    track_velocity_variation = []
    track_speed_mean = []
    track_acceleration = []
    track_direction_change = []
    track_direction_change_max = []
    track_curvature_mean = []
    track_curvature_max = []
    track_angular_velocity_mean = []
    track_angular_velocity_max = []
    track_tortuosity = []
    for track_id in track_ids:
        track = tracks_data.loc[tracks_data['particle'] == track_id].sort_values(by=['frame'])

        # track displacement length
        pos_idx = track.columns.get_indexer(['x', 'y'])
        pos_first = track.iloc[0, pos_idx].values
        pos_last = track.iloc[-1, pos_idx].values
        trk_displacement = np.linalg.norm(pos_first - pos_last)
        track_displacement.append(trk_displacement)

        # track duration
        duration = track['frame'].max() - track['frame'].min() + 1
        track_duration.append(duration)

        # track length
        diff_xy = track[['x', 'y']].diff().fillna(0).values
        trk_length = np.sum(np.linalg.norm(diff_xy, axis=1))
        track_length.append(trk_length)

        # track velocity
        diff_time = track['frame'].diff().fillna(0).values
        velocity = np.divide(np.linalg.norm(diff_xy[1:, :], axis=1), diff_time[1:])
        track_velocity_mean.append(np.mean(velocity))
        track_velocity_max.append(np.max(velocity))
        track_velocity_std.append(np.std(velocity))
        track_velocity_variation.append(np.std(velocity) / np.mean(velocity))
        track_speed_mean.append(trk_length / (duration - 1))

        # track acceleration
        acceleration = np.divide(velocity[1:] - velocity[:-1], diff_time[2:])
        track_acceleration.append(np.abs(np.mean(acceleration)))

        # Instantaneous Direction Change
        directions = np.arctan2(diff_xy[1:, 1], diff_xy[1:, 0])
        direction_changes = np.abs(np.diff(directions))
        direction_changes = np.mod(direction_changes + np.pi, 2 * np.pi) - np.pi
        track_direction_change.append(np.mean(np.abs(direction_changes)))
        track_direction_change_max.append(np.max(np.abs(direction_changes)))

        # Curvature
        curvature = []
        for i in range(1, len(track) - 1):
            v1 = track.iloc[i, pos_idx].values - track.iloc[i - 1, pos_idx].values
            v2 = track.iloc[i + 1, pos_idx].values - track.iloc[i, pos_idx].values
            area = np.abs(np.cross(v1, v2)) / 2
            side_lengths = [np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v2 - v1)]
            if area:
                circumradius = np.prod(side_lengths) / (4 * area)
                curvature.append(circumradius)

        if len(curvature) == 0:
            track_curvature_mean.append(0)
            track_curvature_max.append(0)
        else:
            track_curvature_mean.append(np.mean(curvature))
            track_curvature_max.append(np.max(curvature))

        # Angular Velocity
        angular_velocity = direction_changes / diff_time[1:-1]
        track_angular_velocity_mean.append(np.mean(np.abs(angular_velocity)))
        track_angular_velocity_max.append(np.max(np.abs(angular_velocity)))
        track_tortuosity.append(trk_length / trk_displacement if trk_displacement > 0 else np.inf)

    track_stats = {
        'ID': track_ids,
        'Track Displacement': track_displacement,
        'Track Duration': track_duration,
        'Track Length': track_length,
        'Track Speed Max': track_velocity_max,
        'Track Speed Mean': track_velocity_mean,
        'Track Speed StdDev': track_velocity_std,
        'Track Speed Variation': track_velocity_variation,
        'Track Veloc Mean': track_speed_mean,
        'Track Acceleration': track_acceleration,
        'Track Direction Change Mean': track_direction_change,
        'Track Direction Change Max': track_direction_change_max,
        'Track Curvature Mean': track_curvature_mean,
        'Track Curvature Max': track_curvature_max,
        'Track Angular Velocity Mean': track_angular_velocity_mean,
        'Track Angular Velocity Max': track_angular_velocity_max,
        'Track Tortuosity': track_tortuosity,
    }

    df = pd.DataFrame(track_stats)
    if not df.empty:
        df.to_csv(csv_path.replace("trajectories", "statistics"), index=False, float_format='%.4f')


def track_single_video(data_dir,  vid, int_thres=50, sz_thres=50):
    vid_dir = osp.join(data_dir, vid)
    imgs_path = osp.join(vid_dir, f"{vid}_imgs.tif")
    msks_path = osp.join(vid_dir, f"{vid}_msks.tif")

    df = features_from_masks(imgs_path, msks_path, int_thres, sz_thres)
    if not df.empty:
        trajs = tracking(df, osp.join(vid_dir, f"{vid}_track_trajectories.csv"),
                         max_gap=18, min_len=3, min_int=int_thres)
        if not trajs.empty:
            extract_track_feats(osp.join(vid_dir, f"{vid}_track_trajectories.csv"))


def track_videos(data_dir, int_thres=50, sz_thres=50):
    vids = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    for vid in vids:
        track_single_video(data_dir, vid, int_thres, sz_thres)


if __name__ == "__main__":
    data_dir = "/Users/lxfhfut/Dropbox/Garvan/CBVCC/dataset/trks/cytotorch_0"
    track_videos(data_dir, int_thres=50, sz_thres=50)

