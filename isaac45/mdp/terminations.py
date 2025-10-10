from __future__ import annotations

import torch
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
import numpy as np
import datetime


def quat_to_euler(quaternions):
    """Converts a batch of quaternions to Euler angles (roll, pitch, yaw)."""
    qw, qx, qy, qz = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2))
    pitch = torch.asin(2.0 * (qw * qy - qz * qx))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    
    return roll, pitch, yaw


def drone_flips_upsidedown(env: ManagerBasedRLEnv):
    """Terminate episode if drone has flipped upside down"""

    robot_rotation = env.scene["robot"].data.root_quat_w
    roll, pitch, _ = quat_to_euler(robot_rotation)
    flipped = (torch.abs(roll) > 0.4 * torch.pi) | (torch.abs(pitch) > 0.4 * torch.pi)
    # if flipped:
    #     print("drone flipped")
    return flipped

def drone_crashes_single_contact_sensor(env:ManagerBasedRLEnv, force_threshold: float) -> torch.Tensor:
    contact_force = env.scene["contact_forces"].data.net_forces_w           # (num_envs, 5, 3)
   
    force_magnitude = torch.norm(contact_force, dim=2)                      # (num_envs, 5)

    result = torch.any(force_magnitude > force_threshold, dim=1)  
    # env_map = env.env_map
    # gridmaps = env_map.environment_map  
    # for env_idx in torch.where(result)[0].tolist():
    #     timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     env_mapping = gridmaps[env_idx].clone()
    #     env_mapping[env_mapping == 3] = 1  # Optional transformation

    #     fig, ax = plt.subplots(figsize=(9, 6))
    #     ax.imshow(env_mapping.cpu(), cmap='viridis', origin='lower')
    #     trajectory = env_map.drone_trajectory[env_idx].cpu().numpy()  # (N, 2) shape
    #     # current_pos_np = current_pos.cpu().numpy()
        
    #     ax.plot(trajectory[:, 1], trajectory[:, 0], color='blue', linewidth=1, label='Drone Trajectory')
    #     # axes.scatter(current_pos_np[1], current_pos_np[0], color='red', s=30, label='Current Position')
    #     ax.set_title(f'Env {env_idx} - Traversability Map at {timestamp}')

    #     ax.axis('off')
    #     plt.tight_layout()
    #     filename = f'/workspace/isaaclab/DRL_UAV_Indoor_Exploration/images/occ_map_collision_{env_idx}_{timestamp}.png'
    #     plt.savefig(filename)
    #     plt.close(fig)
    return result



def drone_covers_fixed_area(env: ManagerBasedRLEnv, num_cells_to_cover: int) -> torch.Tensor:
    """Terminate episode if the drone explores a fixed area.
        
        - env_map: Class that handles the mapping of the environment. 
        - num_cells_to_cover: Area of the gridmap to consider the environment as explored.

       """
    
    # Update the map here (not in the rewards) as in the environment step function the terminations are computed before the rewards.
    env_map = env.env_map
    camera_depth_image = env.scene["camera"].data.output["distance_to_camera"]                                  # tensor of shape [num_env, height, width, 1] with depth values in meters
    drone_pose = env.scene["robot"].data.root_state_w[:,:7]                                                     # tensor of shape [num_env, 7] = [num_env, pos (x,y,z), quat(q_w,q_x,q_y,q_z)] in the simulation world (inertial) frame.

    env_map.update_environment_map(camera_depth_image.squeeze(3), drone_pose)                                      
    gridmaps = env_map.environment_map                                                                          # tensor of size (num_envs, grid_height, grid_width)
    non_zero_cells_per_env = torch.sum(gridmaps != 0, dim=(1, 2))                                               # tensor of size (num_envs)
    
    result = non_zero_cells_per_env >= num_cells_to_cover
    for env_idx in torch.where(result)[0].tolist():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        env_mapping = gridmaps[env_idx].clone()
        env_mapping[env_mapping == 3] = 1  # Optional transformation

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(env_mapping.cpu(), cmap='viridis', origin='lower')
        trajectory = env_map.drone_trajectory[env_idx].cpu().numpy()  # (N, 2) shape
        # current_pos_np = current_pos.cpu().numpy()
        
        ax.plot(trajectory[:, 1], trajectory[:, 0], color='blue', linewidth=1, label='Drone Trajectory')
        # axes.scatter(current_pos_np[1], current_pos_np[0], color='red', s=30, label='Current Position')
        ax.set_title(f'Env {env_idx} - Traversability Map at {timestamp}')

        ax.axis('off')
        plt.tight_layout()
        filename = f'/workspace/isaaclab/DRL_UAV_Indoor_Exploration/images/occ_map_fully_explored{env_idx}_{timestamp}.png'
        plt.savefig(filename)
        plt.close(fig)
    if torch.any(result):
        print('Environment fully explored!: ', result)
        

    return result

def drone_covers_fixed_area_4_SCENES(env: ManagerBasedRLEnv, num_cells_to_cover: dict  # keys: 'A', 'B', 'C', 'D'
) -> torch.Tensor:
    """
    Positive reward for covering a fixed area (in number of cells) across four scenes (A, B, C, D).
    
    Args:
        env: Environment object.
        num_cells_to_cover: Dict with required covered cells per scene, e.g., {'A': 400, 'B': 350, ...}
        
    Returns:
        Tensor of rewards per environment.
    """

    # Hardcoded scene limits: (x_min, x_max), (y_min, y_max)
    scene_limits = {
        'A': ((0, 22), (0, 18)),
        'B': ((25, 43), (0, 18)),
        'C': ((0, 18), (20, 44)),
        'D': ((20, 44), (20, 44)),
    }
    env_map = env.env_map
    camera_depth_image = env.scene["camera"].data.output["distance_to_camera"]                                  # tensor of shape [num_env, height, width, 1] with depth values in meters
    drone_pose = env.scene["robot"].data.root_state_w[:,:7]                                                     # tensor of shape [num_env, 7] = [num_env, pos (x,y,z), quat(q_w,q_x,q_y,q_z)] in the simulation world (inertial) frame.

    env_map.update_environment_map(camera_depth_image.squeeze(3), drone_pose)                                      
    gridmaps = env_map.environment_map                                                                          # tensor of size (num_envs, grid_height, grid_width)
    non_zero_cells_per_env = torch.sum(gridmaps != 0, dim=(1, 2))
    # env_map = env.env_map
    # gridmaps = env_map.environment_map  # (num_envs, grid_height, grid_width)
    # non_zero_cells_per_env = torch.sum(gridmaps != 0, dim=(1, 2))

    drone_positions = env.scene["robot"].data.root_pos_w - env.scene.env_origins  # (num_envs, 3)
    result = torch.zeros_like(non_zero_cells_per_env, dtype=torch.float32, device=env.device)

    for scene_key, ((x_min, x_max), (y_min, y_max)) in scene_limits.items():
        in_scene = (
            (drone_positions[:, 0] >= x_min) & (drone_positions[:, 0] <= x_max) &
            (drone_positions[:, 1] >= y_min) & (drone_positions[:, 1] <= y_max)
        )
        
        threshold = num_cells_to_cover[scene_key]
        # enough_coverage = non_zero_cells_per_env >= threshold
        scene_rewards = (non_zero_cells_per_env >= threshold) & in_scene
        result[scene_rewards] = 1.0
        result = result.bool()
    for env_idx in torch.where(result)[0].tolist():
        print("finished in an environment!")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        env_mapping = gridmaps[env_idx].clone()
        env_mapping[env_mapping == 3] = 1  # Optional transformation

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(env_mapping.cpu(), cmap='viridis', origin='lower')
        trajectory = env_map.drone_trajectory[env_idx].cpu().numpy()  # (N, 2) shape
        # current_pos_np = current_pos.cpu().numpy()
        
        ax.plot(trajectory[:, 1], trajectory[:, 0], color='blue', linewidth=1, label='Drone Trajectory')
        # axes.scatter(current_pos_np[1], current_pos_np[0], color='red', s=30, label='Current Position')
        ax.set_title(f'Env {env_idx} - Traversability Map at {timestamp}')

        ax.axis('off')
        plt.tight_layout()
        filename = f'/workspace/isaaclab/DRL_UAV_Indoor_Exploration/images/occ_map_fully_explored{env_idx}_{timestamp}.png'
        plt.savefig(filename)
        plt.close(fig)
    return result
