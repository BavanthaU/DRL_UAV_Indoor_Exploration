from __future__ import annotations

import torch

import matplotlib.pyplot as plt
import os
import h5py

from typing import TYPE_CHECKING
import numpy as np

from isaaclab.utils.math import wrap_to_pi, quat_from_matrix, quat_rotate_inverse, yaw_quat
from isaaclab.scene import InteractiveSceneCfg

from .common import get_raycast_planar_min_distance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def quat_to_euler(quaternions):
    """Converts a batch of quaternions to Euler angles (roll, pitch, yaw)."""
    qw, qx, qy, qz = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2))
    pitch = torch.asin(2.0 * (qw * qy - qz * qx))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    
    return roll, pitch, yaw


def check_collision_single_contact_sensor (env: ManagerBasedRLEnv, M: float, N:float, force_threshold: float) -> torch.Tensor:
    """Penalty for crashing against an obstacle. Uses a single contact sensor whose scope is the whole drone body.
    
    Args:
        env: Isaac RL environment containing drones and contact sensor data.
        M: Negative reward applied to drones that collide.
        N: Positive reward applied to drones that continue without collision.
        force_threshold: Force magnitude above which a contact is considered a collision.

    Returns:
        Tensor of shape (num_envs,) containing reward per environment. 
    """

    contact_force = env.scene["contact_forces"].data.net_forces_w # (num_envs, 5, 3)             
    force_magnitude = torch.norm(contact_force, dim=2) # (num_envs, 5)

    # Determine which environments/drones experienced a collision, when magnitude above threshold
    terminated_envs = torch.any(force_magnitude > force_threshold, dim=1) # (num_envs)
    rewards = torch.full(terminated_envs.shape, N, dtype=torch.float32).to(torch.device(env.device))
    rewards[terminated_envs] = M
    
    return rewards

def smooth_flight_reward(env: ManagerBasedRLEnv, accel_penalty: float = 0.1, ang_accel_penalty: float = 0.05) -> torch.Tensor:
    """Reward term to encourage smooth drone motion by penalizing sudden acceleration and angular velocity changes.

    Args:
        env: Isaac RL environment
        accel_penalty: Weight for the linear acceleration penalty.
        ang_accel_penalty: Weight for the angular acceleration penalty.

    Returns:
        Tensor of shape (num_envs,) containing smoothness reward for each environment.
    """

    linear_vel = env.scene["robot"].data.root_lin_vel_w  # (num_envs, 3)
    
    # Compute linear acceleration 
    linear_accel = torch.diff(linear_vel, dim=0, prepend=torch.zeros_like(linear_vel[:1]))  # (num_envs, 3)
    accel_magnitude = torch.norm(linear_accel, dim=1)  # (num_envs)

    angular_velocity = env.scene["robot"].data.root_ang_vel_w  # (num_envs, 3)
    
    # Compute angular acceleration 
    angular_accel = torch.diff(angular_velocity, dim=0, prepend=torch.zeros_like(angular_velocity[:1]))  # (num_envs, 3)
    ang_accel_magnitude = torch.norm(angular_accel, dim=1)  # (num_envs)

    # Compute penalty (higher acceleration gives more negative reward)
    smoothness_reward = -(accel_penalty * accel_magnitude + ang_accel_penalty * ang_accel_magnitude)

    return smoothness_reward




def drone_flips_upsidedown(env: ManagerBasedRLEnv)-> torch.Tensor:
    """Terminate episode if drone has flipped upside down"""

    robot_rotation = env.scene["robot"].data.root_quat_w
    roll, pitch, _ = quat_to_euler(robot_rotation)
    flipped = (torch.abs(roll) > 0.4 * torch.pi) | (torch.abs(pitch) > 0.4 * torch.pi)
    
    return -1*flipped

def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive.
        Args:
            env: Isaac RL environment
    """
    return (~env.termination_manager.terminated).float()


def near_collision(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reward function to penalize drones when they come close to obstacles based on raycasting data. Ultimately not used because I had problems with the raycaster detecting meshes. 

    - Intended as a test for the raycaster to detect near-collisions.
    - Assigns a negative reward if any ray detects a wall within a threshold distance (1 meter).

    Args:
        env: Isaac RL environment containing drones and raycaster data.

    Returns:
        Tensor of shape (num_envs,) with per-drone rewards:
            - -5.0 if any ray detects a wall within 1m
            - 0.0 otherwise
    """

    # Get robot and wall positions for all environments
    robot_pos = env.scene["robot"].data.root_pos_w  # (num_envs, 3)

    ray_hits = env.scene["ray_caster"].data.ray_hits_w  # (num_envs, num_rays, 3)

    # Extract only the x and y components
    robot_pos_xy = robot_pos[:, :2]  # (num_envs, 2)
    distance_to_wall_xy = ray_hits[:, 1:-1, :2]  # (num_envs, num_rays, 2)

    # Calculate the difference between robot's position and wall's distance in x and y
    diff_xy = distance_to_wall_xy - robot_pos_xy[:, None, :]  # (num_envs, num_rays, 2)
    distance = (diff_xy[:, :, 0]**2 + diff_xy[:, :, 1]**2)**(.5)

    condition = torch.abs(distance) < 1  # (num_envs, num_rays)
    total_reward  = torch.where(condition.any(dim=1), -5.0, 0.0) 

    return total_reward


def raycast_proximity_penalty(
    env: ManagerBasedRLEnv,
    near_threshold: float = 1.0,
    crash_threshold: float = 0.2,
    near_penalty: float = -0.5,
    crash_penalty: float = -10.0,
    vertical_tolerance: float = 1.5,
) -> torch.Tensor:
    """
    Penalize the drone when ray-cast measurements indicate close proximity to obstacles.

    Args:
        env: Isaac RL environment containing the ray-caster sensor.
        near_threshold: Distance below which a mild penalty is applied.
        crash_threshold: Distance below which a severe penalty (crash) is applied.
        near_penalty: Reward value applied when the nearest obstacle is within ``near_threshold``.
        crash_penalty: Reward value applied when the nearest obstacle is within ``crash_threshold``.
        vertical_tolerance: Maximum vertical separation between sensor and hit to consider the ray valid.

    Returns:
        Tensor of shape ``(num_envs,)`` with the proximity penalties.
    """
    distances = get_raycast_planar_min_distance(env, vertical_tolerance=vertical_tolerance)
    rewards = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    near_mask = distances < near_threshold
    rewards[near_mask] = near_penalty

    crash_mask = distances < crash_threshold
    rewards[crash_mask] = crash_penalty

    return rewards


def subgoal_progress_reward(
    env: ManagerBasedRLEnv,
    progress_weight: float = 0.5,
    reach_bonus: float = 5.0,
    tolerance: float = 1.5,
) -> torch.Tensor:
    """Reward for moving toward and reaching the current frontier subgoal."""
    if not hasattr(env, "env_map") or env.env_map is None:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    env_map = env.env_map
    if not hasattr(env_map, "current_subgoal_world"):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    active = getattr(env_map, "subgoal_active", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    if not active.any():
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    robot_pos = env.scene["robot"].data.root_pos_w
    subgoal = env_map.current_subgoal_world
    dist = torch.linalg.norm(subgoal[:, :2] - robot_pos[:, :2], dim=1)
    prev = env_map.prev_subgoal_distance
    progress = prev - dist
    reward = progress_weight * progress
    reached = (dist <= tolerance) & active
    reward[reached] += reach_bonus
    reward[~active] = 0.0
    env_map.prev_subgoal_distance = dist
    return reward


def curiosity_intrinsic_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Intrinsic reward encouraging visitation of novel cells."""
    if not hasattr(env, "env_map") or env.env_map is None:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    env_map = env.env_map
    if not hasattr(env_map, "curiosity_reward"):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return env_map.curiosity_reward.detach()


def area_coverage (env: ManagerBasedRLEnv) -> torch.Tensor:
    """Positive reward equal to the new map area seen in each step.
        Args:
            env: Isaac RL environment.

        Returns:
            Tensor of shape (num_envs,) with reward for each environment.
    """
    # Access the environment map and the difference in newly observed area
    env_map = env.env_map
    area_diff = env_map.reward_area_difference

    # Initialize rewards with the positive reward for new area
    rewards = area_diff.to(torch.float32).to(env.device) # tensor of shape (num_envs)                                    

    # Reset map and for environments that have terminated
    ended_envs = env.termination_manager.dones
    env_map.reset_environment_map(ended_envs)

    return rewards

def area_coverage_and_loop_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Positive reward equal to the new map area seen in each step and penalize drones that loop in already explored regions.

    - Positive reward: Equal to the number of new map cells seen in the current step.
    - Loop penalty: Small negative reward if the drone does not discover new cells for multiple consecutive steps.

    Args:
        env: Isaac RL environment containing the environment map and exploration data.

    Returns:
        Tensor of shape (num_envs,) with reward for each environment.
    
    """
    # Access the environment map and the difference in newly observed area
    env_map = env.env_map
    area_diff = env_map.reward_area_difference

    # Initialize rewards with the positive reward for new area
    rewards = area_diff.to(torch.float32).to(env.device)  # tensor of shape (num_envs)      

    # Initialize loop counter if it does not exist
    if not hasattr(env, "no_new_cells_counter"):
        env.no_new_cells_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    # Identify which drones did not discover new cells
    no_new_cells = area_diff == 0
    new_cells_found = area_diff > 0

    # Update the counter: increment if no new cells, reset if new cells found
    env.no_new_cells_counter += no_new_cells.to(torch.int32)
    env.no_new_cells_counter *= no_new_cells.to(torch.int32)  # reset to 0 where new cells are found

    # Apply penalty only if for 20 new steps no new cells have been found. 
    penalty_mask = env.no_new_cells_counter > 20
    rewards += penalty_mask.to(torch.float32) * -0.05

    # Reset map and looping counter and for environments that have terminated
    ended_envs = env.termination_manager.dones
    env_map.reset_environment_map(ended_envs)
    env.no_new_cells_counter[ended_envs] = 0

    return rewards
    
def fixed_area_covered (env: ManagerBasedRLEnv, num_cells_to_cover: int) -> torch.Tensor:
    """ Positive reward for covering a fixed area (in number of cells).
        Args:
            env: Isaac RL environment.
            num_cells_to_cover: required number of cells to cover in the occupancy map

        Returns:
            Tensor of shape (num_envs,) with reward for each environment.

    """
    env_map = env.env_map
    gridmaps = env_map.environment_map  # tensor of size (num_envs, grid_height, grid_width)
    non_zero_cells_per_env = torch.sum(gridmaps != 0, dim=(1, 2))

    explored_envs = non_zero_cells_per_env >= num_cells_to_cover # (num_envs)
    rewards = explored_envs.float().to(env.device)
    
    return rewards

    
def fixed_area_covered_4_offices(env: ManagerBasedRLEnv, num_cells_to_cover: dict  # keys: 'A', 'B', 'C', 'D'
) -> torch.Tensor:
    """
    Positive reward for covering a fixed area (in number of cells) across four scenes (A, B, C, D).
    
    Args:
        env: Isaac RL environment.
        num_cells_to_cover: Dict with required covered cells per scene, for example {'A': 400, 'B': 350, ...}
        
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
    gridmaps = env_map.environment_map  # (num_envs, grid_height, grid_width)
    non_zero_cells_per_env = torch.sum(gridmaps != 0, dim=(1, 2))

    drone_positions = env.scene["robot"].data.root_pos_w - env.scene.env_origins  # (num_envs, 3)
    rewards = torch.zeros_like(non_zero_cells_per_env, dtype=torch.float32, device=env.device)

    for scene_key, ((x_min, x_max), (y_min, y_max)) in scene_limits.items():
        in_scene = (
            (drone_positions[:, 0] >= x_min) & (drone_positions[:, 0] <= x_max) &
            (drone_positions[:, 1] >= y_min) & (drone_positions[:, 1] <= y_max)
        )

        threshold = num_cells_to_cover[scene_key]
        # enough_coverage = non_zero_cells_per_env >= threshold
        scene_rewards = (non_zero_cells_per_env >= threshold) & in_scene
        rewards[scene_rewards] = 1.0

    return rewards


def doorway_reward_per_drone(env, doorway_radius=0.3):
    """
    Rewards drones for entering new doorways within 0.3m.
    Updates env.visited_doorways to track individual drone progress.
    
    Args:
        env: Isaac RL environment.
        doorway_radius: Distance threshold to count a doorway entry.

    Returns:
        Tensor of shape (num_envs,) with reward for each environment.
    """
    # Compute positions of drones relative to the environment origin (shape: [num_envs, 3])
    drone_positions = env.scene["robot"].data.root_pos_w - env.scene.env_origins 

    # Take only x,y coordinate of drones
    positions_xy = drone_positions[:, :2]  # [num_envs, 2]
    if not hasattr(env, "visited_doorways"):
        env.visited_doorways = [set() for _ in range(env.num_envs)]

    num_envs = env.num_envs
    rewards = torch.zeros(num_envs, dtype=torch.float32, device=env.device)

    # Predefined doorway positions for all environments (x, y)
    doorway_positions = torch.tensor(
        [
            # Environment A
            [4.0, 6.0], [10.5, 9.0], [15.0, 7.5],
            # Environment B
            [37.0, 6.0], [40.0, 10.5], [35.5, 12.0], [32.5, 12.0],
            # Environment C
            [3.0, 24.5], [3.0, 27.5], [3.0, 36.5], [3.0, 39.5],
            [6.0, 27.5], [6.0, 42.5], [6.0, 30.5]
        ],
        device=env.device,
        dtype=torch.float32
    )
    
    visited = env.visited_doorways  # list of sets (one per env/drone)

    # Loop over all environments/drones
    for env_id in range(num_envs):
        # Check each doorway for this drone
        for door_idx, doorway_pos in enumerate(doorway_positions):
            # Skip doorways already visited by this drone
            if door_idx in visited[env_id]:
                continue
            
            # Compute planar distance to the doorway
            dist = torch.norm(positions_xy[env_id] - doorway_pos)
            
            # If drone is within doorway_radius of an unvisited doorway
            if dist <= doorway_radius:
                rewards[env_id] = 1.0  # reward for entering a new doorway
                # print(f"environment {env_id} has gone through a door!")  # optional debug logging
                visited[env_id].add(door_idx)  # mark doorway as visited
                break  # Only give one reward per timestep

    return rewards


def penalize_idle_behavior(env: ManagerBasedRLEnv, idle_penalty: float = -0.1, motion_threshold: float = 0.05) -> torch.Tensor:
    """
    Penalize the drone when it neither moves forward nor rotates.

    - idle_penalty: Negative reward when the drone is idle.
    - motion_threshold: Minimum magnitude to consider movement as significant.

    Args:
        env: Isaac RL environment.
        idle_penalty: weight of reward
        motion_threshold: value above action must be to avoid penalty

    Returns: 
        Tensor of shape (num_envs,) with reward for each environment.
    """
    # Check if action exists
    if hasattr(env, 'actions') and env.actions is not None:
        # Get the current action values
        actions = env.actions.clone()  # Shape: (num_envs, action_dim)

        # Check if actions are above threshold
        forward_motion = torch.abs(actions[:, 0]) > motion_threshold  # Forward velocity
        rotational_motion = torch.abs(actions[:, 1]) > motion_threshold  # Rotational velocity

        # Identify drones that are idle
        idle_mask = ~(forward_motion | rotational_motion)

        # Initialize zero rewards
        rewards = torch.zeros(actions.shape[0], dtype=torch.float32).to(env.device)

        # Apply penalty to idle drones
        rewards[idle_mask] = idle_penalty
    
        return rewards
    else: 
        num_envs = env.num_envs
        rewards = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
    return rewards
