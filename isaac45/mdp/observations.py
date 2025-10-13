from __future__ import annotations

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# from scipy import stats
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from scipy import stats
import numpy as np
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, wrap_to_pi, euler_xyz_from_quat


def base_lin_vel_ideal(env: ManagerBasedRLEnv) -> torch.tensor:
    """Compute root linear velocity in the asset's root frame using the ideal controller.
       Since the ideal controller does not update velocity automatically, it is manually inferred from actions.
    """
    if hasattr(env, 'actions') and env.actions is not None:

        velocity = env.actions.clone()
        velocity[:, 1] = 0.0
        zero_col = torch.zeros(velocity.shape[0], 1, device=velocity.device)  # shape (num_envs, 1)
        velocity_padded = torch.cat((velocity, zero_col), dim=1) 

        return velocity_padded

    return torch.zeros((1, 3), dtype=torch.float32)


def base_ang_vel_ideal(env: ManagerBasedRLEnv) -> torch.tensor:
    """Compute root angular velocity in the asset's root frame using the ideal controller.
       Since angular velocity is not updated by the ideal controller, infer from actions.
    """
    if hasattr(env, 'actions') and env.actions is not None:

        velocity = env.actions.clone()
        velocity[:, 0] = 0.0
        zero_col = torch.zeros(velocity.shape[0], 1, device=velocity.device)  # shape (num_envs, 1)
        velocity_padded = torch.cat((velocity, zero_col), dim=1) 
        velocity_padded[:, [1, 2]] = velocity_padded[:, [2, 1]]

        return velocity_padded*0.8
    return torch.zeros((1, 3), dtype=torch.float32)


def get_orig_depth_images(env: ManagerBasedEnv) -> torch.Tensor:
    """Get original depth images from the camera. 
    """
    camera_depth_image = env.scene["camera"].data.output["distance_to_camera"] # tensor of shape [num_env, height, width, 1] with depth values in meters (Attention: Extra dimension for the channels w.r.t other IsaacLab versions < 1.2.0).
        
    return camera_depth_image

def get_depth_images(env: ManagerBasedEnv, scale_factor: float, depth_threshold: float) -> torch.Tensor:
    """Get processed depth images from the camera. 
    
    - scale_factor: factor to down/up scale the resolution of the original depth image.
    - depth_threshold: threshold to truncate the depth image values.

    """
    
    camera_depth_image = env.scene["camera"].data.output["distance_to_camera"] # tensor of shape [num_env, height, width, 1] with depth values in meters (Attention: Extra dimension for the channels w.r.t other IsaacLab versions < 1.2.0).
    
    depth_image_processed = process_depth_image(camera_depth_image, scale_factor, depth_threshold) # tensor of shape [num_env, 1, scale_factor*height, scale_factor*width] with normalized depth values.
    
    return depth_image_processed 

def process_depth_image (depth_image: torch.Tensor, scale_factor: float, depth_threshold: float) -> torch.Tensor:
    """Process original depth images (downsample, truncate, normalize, and re-organize shape to be fed to SB3). 
    
    - scale_factor: factor to down/up scale the resolution of the original depth image.
    - depth_threshold: threshold to truncate the depth image values.
    
    """
    # Downsample
    depth_image_prep = depth_image.permute(0, 3, 1, 2)  # Needed for the interpolate function. Takes inputs of size [batch_size, channels, height, width]
    depth_image_downsampled = F.interpolate(depth_image_prep, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    # Truncate and normalize depth values
    depth_image_processed = torch.clamp(depth_image_downsampled / depth_threshold, 0, 1)

    return  depth_image_processed

def get_depth_images_switched_channel(env: ManagerBasedEnv, depth_threshold: float) -> torch.Tensor:
    """Get original depth images from the camera with channel-first convention (for SB3). 
    """
    camera_depth_image = env.scene["camera"].data.output["distance_to_camera"] # tensor of shape [num_env, height, width, 1] with depth values in meters (Attention: Extra dimension for the channels w.r.t other IsaacLab versions < 1.2.0).
    
    depth_image_processed = camera_depth_image.permute(0, 3, 1, 2)             # tensor of shape [num_env, 1, height, width]
    depth_image_processed[depth_image_processed == 0] = 4.0

    depth_image_processed = torch.clamp(depth_image_processed / depth_threshold, 0, 1)

    return depth_image_processed 


def get_1d_depth(env: ManagerBasedEnv, depth_threshold: float) -> torch.Tensor:
    """Return a 1D 'lidar-like' depth measurement by selecting the middle row of the depth image.
        Useful for simplified observations.
    """
    camera_depth_image = env.scene["camera"].data.output["distance_to_camera"]
    camera_depth_image[camera_depth_image == 0] = 4.0

    # Remove channel dimension and get shape info
    camera_depth_image = camera_depth_image.squeeze(-1)  # -> [num_env, H, W]
    num_envs, H, W = camera_depth_image.shape
    
    # Select the middle row (or another row of interest)
    row_index = H // 2
    lidar_1d = camera_depth_image[:, row_index, :]  # -> [num_env, W]

    # Normalize and clamp
    lidar_1d_clipped = torch.clamp(lidar_1d / depth_threshold, 0, 1)  # Values in [0, 1]
    return lidar_1d_clipped


def get_rotated_egocentric_mapping_ONE_HOT(env: ManagerBasedEnv, egocentric_map_half_size: float) -> torch.Tensor:
    """
    Get egocentric image of the generated traversability map, rotated to align with the drone's orientation.
    
    - env: Environment object containing the map and drone state.
    - egocentric_map_size: Radius (in cells) around the drone around wich we create the egocentric map. The map will have size (2*egocentric_map_size + 1)
    """

    egocentric_map = torch.zeros(env.num_envs, 2*egocentric_map_half_size + 1, 2*egocentric_map_half_size + 1, dtype=torch.int, device=env.device)
    
    if hasattr(env, 'env_map'):

        env_map = env.env_map

        gridmaps = env_map.environment_map.to(torch.int32)          # Tensor of size (num_envs, grid_num[0], grid_num[1]).
        drone_locs = env_map.drone_loc_in_gridmap                   # Tensor of size (num_envs, 2).
        drone_2D_orientation = env_map.drone_orientation_2D         # Tensor of shape (num_envs) with yaw.

        cos_theta = torch.cos(drone_2D_orientation)                  # tensor of shape (num_envs) with x unit vector component
        sin_theta = torch.sin(drone_2D_orientation)                  # tensor of shape (num_envs) with y unit vector component
        Rz_drone = torch.stack([torch.stack([cos_theta, -sin_theta], dim=1), torch.stack([sin_theta, cos_theta], dim=1)], dim=0).permute(1, 0, 2)  # Tensor of shape (num_envs, 2, 2) with 2D rotation around z axis of the drone 

        # Compute coordinates of the cells to select from the gridmap, for all environments
        x = torch.arange(- egocentric_map_half_size,  egocentric_map_half_size + 1, device=env.device)
        y = torch.arange(- egocentric_map_half_size,  egocentric_map_half_size + 1, device=env.device)
        xv, yv = torch.meshgrid(x, y, indexing="ij")
        shift_coords = torch.stack([xv.flatten(), yv.flatten()], dim=1).unsqueeze(0).expand(env.num_envs, -1, -1).float()  # Shape: (num_envs, 2*egocentric_map_half_size + 1, 2)
     
        rotated_shift_coords = torch.bmm(Rz_drone, shift_coords.permute(0, 2, 1)).permute(0, 2, 1)      # Tensor of shape: (num_envs, 2*egocentric_map_half_size + 1, 2) with rotated shift coordinates (following the drone's orientation)
        gridmap_coords =  rotated_shift_coords + drone_locs.unsqueeze(1)                                # Tensor of shape: (num_envs, 2*egocentric_map_half_size + 1, 2)
        gridmap_coords = gridmap_coords.long()  

        valid_mask = ((gridmap_coords[:, :, 0] >= 0) & (gridmap_coords[:, :, 0] < gridmaps.shape[1]) & (gridmap_coords[:, :, 1] >= 0) & (gridmap_coords[:, :, 1] < gridmaps.shape[2])) # Boolean tensor of shape  (num_envs, 2*egocentric_map_half_size + 1)

        gridmap_valid_coords = gridmap_coords[valid_mask]  # Tensor of shape: (valid gridmap coords in all envs, 2)
        valid_shift_coords = shift_coords[valid_mask]  # Tensor of shape: (valid gridmap coords in all envs, 2)
        env_indices = torch.arange(env.num_envs, device=env.device).unsqueeze(1).expand(-1, valid_mask.size(1))[valid_mask].long()  # Tensor of shape: (valid gridmap coords in all envs)

        # Compute egocentric map indices
        lx = (valid_shift_coords[:, 0] + egocentric_map_half_size).long()
        ly = (valid_shift_coords[:, 1] + egocentric_map_half_size).long()

        # Write values to the egocentric map
        egocentric_map[env_indices, lx, ly] = gridmaps[env_indices, gridmap_valid_coords[:, 0], gridmap_valid_coords[:, 1]]

        # Mark the drone's location in the map
        egocentric_map = egocentric_map.flip(1).flip(2)
        egocentric_map[egocentric_map == 3] = 1
        egocentric_map[:, egocentric_map_half_size , egocentric_map_half_size] = 3

    else:
        print('[INFO]: Mapping class not initialized yet')

    # Normalize and reshape the map
    egocentric_map_one_hot_encoded = F.one_hot(egocentric_map.to(dtype=torch.int64), num_classes=4)  # tensor of shape [num_env, 2*egocentric_map_half_size + 1, 2*egocentric_map_half_size + 1, num_classes]
    egocentric_map_processed = egocentric_map_one_hot_encoded.permute(0, 3, 1, 2).float() # tensor of shape [num_env, num_classes, 2*egocentric_map_half_size + 1, 2*egocentric_map_half_size + 1]

    return egocentric_map_processed

def get_id_semantic_images_switched_channel_ONE_HOT(env: ManagerBasedEnv, num_classes: float) -> torch.Tensor:
    """Get original ID semantic images from the camera with channel-first convention (for SB3). Use when colorize_semantic_segmentation = False in the camera configuration.
    """

    semantic_image = env.scene["camera"].data.output['semantic_segmentation'] # tensor of shape [num_env, height, width, 1] with semantic IDs (Attention: Extra dimension for the channels w.r.t other IsaacLab versions < 1.2.0).
    semantic_image_one_hot_encoded = F.one_hot(semantic_image.to(dtype=torch.int64).squeeze(3), num_classes=num_classes) # tensor of shape [num_env, height, width, num_classes]
    semantic_image_processed =  semantic_image_one_hot_encoded.permute(0, 3, 1, 2).float() # tensor of shape [num_env, num_classes, height, width]

    return  semantic_image_processed

def get_oneline_semantic(env: ManagerBasedEnv, num_classes: float) -> torch.Tensor: 
    """Return a single row from the semantic segmentation image.
    """
    semantic_image = env.scene["camera"].data.output['semantic_segmentation'].permute(0, 3, 1, 2).float()
    oneline_sem = semantic_image[ :,:, 24, :]
    return oneline_sem


def get_subgoal_vector(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Subgoal in body frame (dx, dy) plus heading error and planar distance."""
    if not hasattr(env, "env_map") or env.env_map is None:
        return torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)
    env_map = env.env_map
    if not hasattr(env_map, "current_subgoal_world"):
        return torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)

    subgoal = env_map.current_subgoal_world
    active = getattr(env_map, "subgoal_active", torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    robot_pos = env.scene["robot"].data.root_pos_w
    vec_world = subgoal - robot_pos
    robot_quat = env.scene["robot"].data.root_quat_w
    vec_body = quat_rotate_inverse(robot_quat, vec_world)

    yaw_robot = euler_xyz_from_quat(robot_quat)[2]
    yaw_target = torch.atan2(vec_world[:, 1], vec_world[:, 0])
    heading_error = wrap_to_pi(yaw_target - yaw_robot)
    distance = torch.linalg.norm(vec_world[:, :2], dim=1)

    obs = torch.stack([vec_body[:, 0], vec_body[:, 1], heading_error, distance], dim=1)
    obs[~active] = 0.0
    return obs
