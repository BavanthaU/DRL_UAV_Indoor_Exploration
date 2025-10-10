from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Literal

import carb
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

import random
import numpy as np
from scipy.spatial.transform import Rotation as R


def reset_drone_position_semantics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float],
    z_position: float):
    """Reset the drone's joints and position (x, y, z) independently for a set of environments.

    Args:
        env: The environment manager containing the scene and robots.
        env_ids: Tensor of environment indices to reset.
        velocity_range: Tuple specifying min and max velocities for randomization (forward and angular).
        z_position: Fixed z-coordinate for the drone.
    """

    # Get the robot object from the scene and the origins of each scene. 
    robot = env.scene["robot"]    
    scene_origin = env.scene.env_origins

    # All possible rooms in the office environments, with 0.3m distance from walls. 
    # Only offices 1, 2 and 3 are used in the final RL algorithms and of each office only 4 rooms are taken as starting positions.
    # This is done so the change of starting in each environment is the same. 
    rooms = [
        #Office 1
        {"x_range": (0.9, 1.8), "y_range": (1.0, 2.7)},
        {"x_range": (3.5, 6.9), "y_range": (1.0, 4.5)},
        # {"x_range": (4.2, 10.7), "y_range": (10.3, 11.9)},
        {"x_range": (12.7, 13.5), "y_range": (10.3, 14.6)},
        {"x_range": (4.2, 10.9), "y_range": (13.7, 14.6)},
        # {"x_range": (16.2, 18.7), "y_range": (5.1, 9.6)},
        
        # Office 2 (circle with small offices)
        {"x_range": (29.2, 36.2), "y_range": (3.4, 8.2)},
        {"x_range": (28.6, 33.2), "y_range": (12.8, 14.5)},
        {"x_range": (34.7, 38.9), "y_range": (12.9, 14.2)},
        {"x_range": (40.8, 42.1), "y_range": (9.7, 13.8)},

        # Office 3 (small offices and bigger ones along corridor)
        {"x_range": (0.8, 2.4), "y_range": (20.6, 23.6)},
        {"x_range": (0.8, 2.4), "y_range": (28.2, 31.4)},
        # {"x_range": (0.8, 2.4), "y_range": (32.6, 35.7)},
        # {"x_range": (0.8, 2.4), "y_range": (40.3, 43.2)},
        {"x_range": (6.7, 10.9), "y_range": (38.7, 43.2)},
        {"x_range": (6.7, 10.9), "y_range": (29.8, 37.2)},
        {"x_range": (6.7, 10.9), "y_range": (20.2, 28.2)},

        # Office 4 (big circular corridor with offices at corners and in middle)
        # {"x_range": (23.5, 25.1), "y_range": (20.7, 25.0)},
        # {"x_range": (20.6, 22.2), "y_range": (20.7, 24.5)},
        # {"x_range": (20.5, 25.2), "y_range": (39.3, 43.3)},
        # {"x_range": (39.2, 43.4), "y_range": (39.3, 43.3)},
        
        # # {"x_range": (39.0, 43.0), "y_range": (20.8, 22.1)},
        # {"x_range": (39.0, 43.0), "y_range": (23.5, 24.6)},
        # {"x_range": (28.7, 35.0), "y_range": (30.0, 34.2)},
    ]

    # Generate random positions in a randomly selected room and orientations for each environment
    position_orientation = torch.tensor([
        [
            # Random x within room bounds plus environment origin
            random.uniform((room := random.choice(rooms))["x_range"][0], room["x_range"][1]) + scene_origin[env_id, 0],
            # Random y within room bounds plus environment origin
            random.uniform(room["y_range"][0], room["y_range"][1]) + scene_origin[env_id, 1],
            # Fixed z position
            z_position,
            # Convert random yaw rotation to quaternion (w, x, y, z)
            *R.from_euler('z', random.uniform(-1*np.pi, np.pi)).as_quat()[[3, 0, 1, 2]]
        ]
        for env_id in env_ids
    ], device=env.device, dtype=torch.float32)

    # Generate random velocities. (not used in RL algorithms)
    velocities = torch.tensor([
        [
            random.uniform(velocity_range[0], velocity_range[1]),  # x vel
            0,  # y vel
            0.0,  # z vel
            0.0, 0.0, random.uniform(velocity_range[0], velocity_range[1])  # angular velocity [roll, pitch, yaw]
        ]
        for _ in env_ids
    ], device=env.device, dtype=torch.float32)

    # Apply the computed positions and orientations to the robot in the simulation
    robot.write_root_link_pose_to_sim(position_orientation, env_ids=env_ids)
    robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    
    # Also reset visited doorways: 
    if hasattr(env, "visited_doorways"):
        for env_id in env_ids:
            env.visited_doorways[env_id] = set()


def reset_drone_eval_envA(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """
    Reset the drone in evaluation environment A to predefined start positions and orientations.
    """
    
    # Get the robot object from the scene
    robot = env.scene["robot"]
    
    # Scene origin offsets for each environment
    scene_origin = env.scene.env_origins

    # Initialize the start positions queue if it does not exist
    # Each entry: [x, y, z, roll, pitch, yaw]
    if not hasattr(env, "start_positions_queue"):
        env.start_positions_queue = [
        [-8.0, 27.5, 2.0, 0, 0, 0.0],
        [-8.0, 27.5, 2.0, 0, 0, np.pi * 0.5],
        [-8.0, 27.5, 2.0, 0, 0, np.pi],
        [-8.0, 27.5, 2.0, 0, 0, -np.pi * 0.5],

        [-15.0, 27.5, 2.0, 0, 0, 0.0],
        [-15.0, 27.5, 2.0, 0, 0, np.pi * 0.5],
        [-15.0, 27.5, 2.0, 0, 0, np.pi],
        [-15.0, 27.5, 2.0, 0, 0, -np.pi * 0.5],

        [-19.0, 27.5, 2.0, 0, 0, 0.0],
        [-19.0, 27.5, 2.0, 0, 0, np.pi * 0.5],
        [-19.0, 27.5, 2.0, 0, 0, np.pi],
        [-19.0, 27.5, 2.0, 0, 0, -np.pi * 0.5],

        [-8.0, 31.0, 2.0, 0, 0, 0.0],
        [-8.0, 31.0, 2.0, 0, 0, np.pi * 0.5],
        [-8.0, 31.0, 2.0, 0, 0, np.pi],
        [-8.0, 31.0, 2.0, 0, 0, -np.pi * 0.5],

        [-15.0, 31.0, 2.0, 0, 0, 0.0],
        [-15.0, 31.0, 2.0, 0, 0, np.pi * 0.5],
        [-15.0, 31.0, 2.0, 0, 0, np.pi],
        [-15.0, 31.0, 2.0, 0, 0, -np.pi * 0.5],
        
        [-19.0, 31.0, 2.0, 0, 0, 0.0],
        [-19.0, 31.0, 2.0, 0, 0, np.pi * 0.5],
        [-19.0, 31.0, 2.0, 0, 0, np.pi],
        [-19.0, 31.0, 2.0, 0, 0, -np.pi * 0.5],

        [-15.0, 38.0, 2.0, 0, 0, 0.0],
        [-15.0, 38.0, 2.0, 0, 0, np.pi * 0.5],
        [-15.0, 38.0, 2.0, 0, 0, np.pi],
        [-15.0, 38.0, 2.0, 0, 0, -np.pi * 0.5],

        [-19.0, 38.0, 2.0, 0, 0, 0.0],
        [-19.0, 38.0, 2.0, 0, 0, np.pi * 0.5],
        [-19.0, 38.0, 2.0, 0, 0, np.pi],
        [-19.0, 38.0, 2.0, 0, 0, -np.pi * 0.5],

        [-15.0, 44.6, 2.0, 0, 0, 0.0],
        [-15.0, 44.6, 2.0, 0, 0, np.pi * 0.5],
        [-15.0, 44.6, 2.0, 0, 0, np.pi],
        [-15.0, 44.6, 2.0, 0, 0, -np.pi * 0.5],
            
        [-19.0, 44.6, 2.0, 0, 0, 0.0],
        [-19.0, 44.6, 2.0, 0, 0, np.pi * 0.5],
        [-19.0, 44.6, 2.0, 0, 0, np.pi],
        [-19.0, 44.6, 2.0, 0, 0, -np.pi * 0.5],

        [-15.0, 47.7, 2.0, 0, 0, 0.0],
        [-15.0, 47.7, 2.0, 0, 0, np.pi * 0.5],
        [-15.0, 47.7, 2.0, 0, 0, np.pi],
        [-15.0, 47.7, 2.0, 0, 0, -np.pi * 0.5],

        [-19.0, 47.7, 2.0, 0, 0, 0.0],
        [-19.0, 47.7, 2.0, 0, 0, np.pi * 0.5],
        [-19.0, 47.7, 2.0, 0, 0, np.pi],
        [-19.0, 47.7, 2.0, 0, 0, -np.pi * 0.5],

        [-8.0, 38.0, 2.0, 0, 0, 0.0],
        [-8.0, 38.0, 2.0, 0, 0, np.pi * 0.5],
        [-8.0, 38.0, 2.0, 0, 0, np.pi],
        [-8.0, 38.0, 2.0, 0, 0, -np.pi * 0.5],
            
        [-8.0, 44.6, 2.0, 0, 0, 0.0],
        [-8.0, 44.6, 2.0, 0, 0, np.pi * 0.5],
        [-8.0, 44.6, 2.0, 0, 0, np.pi],
        [-8.0, 44.6, 2.0, 0, 0, -np.pi * 0.5],

        [-8.0, 47.7, 2.0, 0, 0, 0.0],
        [-8.0, 47.7, 2.0, 0, 0, np.pi * 0.5],
        [-8.0, 47.7, 2.0, 0, 0, np.pi],
        [-8.0, 47.7, 2.0, 0, 0, -np.pi * 0.5]]
        env.start_position_index = 0

    # Get current position and increment index
    idx = env.start_position_index % len(env.start_positions_queue)
    position = env.start_positions_queue[idx]
    env.start_position_index += 1  # persist index across calls

    # Apply position to environment
    x, y, z, roll, pitch, yaw = position
    quat = torch.tensor(
        [*R.from_euler('z', yaw).as_quat()[[3, 0, 1, 2]]  # Quaternion (w, x, y, z)
        ],
        device=env.device, dtype=torch.float32
    )
    position_orientation = torch.tensor([
        [
            x + scene_origin[env_id, 0],
            y + scene_origin[env_id, 1],
            z,
            *quat
        ]
        for env_id in env_ids
    ], device=env.device)

    velocities = torch.zeros((len(env_ids), 6), device=env.device)
    robot.write_root_link_pose_to_sim(position_orientation, env_ids=env_ids)
    robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)



def reset_drone_eval_envB(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """
    Reset the drone in evaluation environment B to predefined start positions and orientations.
    """
    
    # Get the robot object from the scene
    robot = env.scene["robot"]
    
    # Scene origin offsets for each environment
    scene_origin = env.scene.env_origins

    # Initialize the start positions queue if it does not exist
    # Each entry: [x, y, z, roll, pitch, yaw]
    if not hasattr(env, "start_positions_queue"):
        env.start_positions_queue = [
        [-3, 38, 2.0, 0, 0, 0],
        [-3, 38, 2.0, 0, 0, np.pi * 0.5],
        [-3, 38, 2.0, 0, 0, -np.pi * 0.5],
        [-3, 38, 2.0, 0, 0, np.pi],

        [2, 38, 2.0, 0, 0, 0],
        [2, 38, 2.0, 0, 0, np.pi * 0.5],
        [2, 38, 2.0, 0, 0, -np.pi * 0.5],
        [2, 38, 2.0, 0, 0, np.pi],

        [2, 44, 2.0, 0, 0, 0],
        [2, 44, 2.0, 0, 0, np.pi * 0.5],
        [2, 44, 2.0, 0, 0, -np.pi * 0.5],
        [2, 44, 2.0, 0, 0, np.pi],

        [2, 50, 2.0, 0, 0, 0],
        [2, 50, 2.0, 0, 0, np.pi * 0.5],
        [2, 50, 2.0, 0, 0, -np.pi * 0.5],
        [2, 50, 2.0, 0, 0, np.pi],

        [2, 56, 2.0, 0, 0, 0],
        [2, 56, 2.0, 0, 0, np.pi * 0.5],
        [2, 56, 2.0, 0, 0, -np.pi * 0.5],
        [2, 56, 2.0, 0, 0, np.pi],

        [0, 50, 2.0, 0, 0, 0],
        [0, 50, 2.0, 0, 0, np.pi * 0.5],
        [0, 50, 2.0, 0, 0, -np.pi * 0.5],
        [0, 50, 2.0, 0, 0, np.pi],

        [0, 57, 2.0, 0, 0, 0],
        [0, 57, 2.0, 0, 0, np.pi * 0.5],
        [0, 57, 2.0, 0, 0, -np.pi * 0.5],
        [0, 57, 2.0, 0, 0, np.pi],

        [-3, 50, 2.0, 0, 0, 0],
        [-3, 50, 2.0, 0, 0, np.pi * 0.5],
        [-3, 50, 2.0, 0, 0, -np.pi * 0.5],
        [-3, 50, 2.0, 0, 0, np.pi],

        [-3, 54, 2.0, 0, 0, 0],
        [-3, 54, 2.0, 0, 0, np.pi * 0.5],
        [-3, 54, 2.0, 0, 0, -np.pi * 0.5],
        [-3, 54, 2.0, 0, 0, np.pi],

        [-22, 53, 2.0, 0, 0, 0],
        [-22, 53, 2.0, 0, 0, np.pi * 0.5],
        [-22, 53, 2.0, 0, 0, -np.pi * 0.5],
        [-22, 53, 2.0, 0, 0, np.pi],

        [-22, 57, 2.0, 0, 0, 0],
        [-22, 57, 2.0, 0, 0, np.pi * 0.5],
        [-22, 57, 2.0, 0, 0, -np.pi * 0.5],
        [-22, 57, 2.0, 0, 0, np.pi],

        [-18, 57, 2.0, 0, 0, 0],
        [-18, 57, 2.0, 0, 0, np.pi * 0.5],
        [-18, 57, 2.0, 0, 0, -np.pi * 0.5],
        [-18, 57, 2.0, 0, 0, np.pi],

        [-18, 53, 2.0, 0, 0, 0],
        [-18, 53, 2.0, 0, 0, np.pi * 0.5],
        [-18, 53, 2.0, 0, 0, -np.pi * 0.5],
        [-18, 53, 2.0, 0, 0, np.pi],

        [-14, 53, 2.0, 0, 0, 0],
        [-14, 53, 2.0, 0, 0, np.pi * 0.5],
        [-14, 53, 2.0, 0, 0, -np.pi * 0.5],
        [-14, 53, 2.0, 0, 0, np.pi],

        [-14, 57, 2.0, 0, 0, 0],
        [-14, 57, 2.0, 0, 0, np.pi * 0.5],
        [-14, 57, 2.0, 0, 0, -np.pi * 0.5],
        [-14, 57, 2.0, 0, 0, np.pi],

        [-9, 52.5, 2.0, 0, 0, 0],
        [-9, 52.5, 2.0, 0, 0, np.pi * 0.5],
        [-9, 52.5, 2.0, 0, 0, -np.pi * 0.5],
        [-9, 52.5, 2.0, 0, 0, np.pi],

        [-7, 58.5, 2.0, 0, 0, 0],
        [-7, 58.5, 2.0, 0, 0, np.pi * 0.5],
        [-7, 58.5, 2.0, 0, 0, -np.pi * 0.5],
        [-7, 58.5, 2.0, 0, 0, np.pi]]
        env.start_position_index = 0

    # Get current position and increment index
    idx = env.start_position_index % len(env.start_positions_queue)
    position = env.start_positions_queue[idx]
    env.start_position_index += 1  # persist index across calls

    # Apply position to environment
    x, y, z, roll, pitch, yaw = position
    quat = torch.tensor(
        [*R.from_euler('z', yaw).as_quat()[[3, 0, 1, 2]]  # Quaternion (w, x, y, z)
        ],
        device=env.device, dtype=torch.float32
    )
    position_orientation = torch.tensor([
        [
            x + scene_origin[env_id, 0],
            y + scene_origin[env_id, 1],
            z,
            *quat
        ]
        for env_id in env_ids
    ], device=env.device)

    velocities = torch.zeros((len(env_ids), 6), device=env.device)
    robot.write_root_link_pose_to_sim(position_orientation, env_ids=env_ids)
    robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def reset_drone_eval_envC(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """
    Reset the drone in evaluation environment C to predefined start positions and orientations.
    """
    
    # Get the robot object from the scene
    robot = env.scene["robot"]
    
    # Scene origin offsets for each environment
    scene_origin = env.scene.env_origins

    # Initialize the start positions queue if it does not exist
    # Each entry: [x, y, z, roll, pitch, yaw]
    if not hasattr(env, "start_positions_queue"):
        env.start_positions_queue = [
        [-20.4, 12.5, 2.0, 0, 0, 0.0],
        [-20.4, 12.5, 2.0, 0, 0, np.pi * 0.5],
        [-20.4, 12.5, 2.0, 0, 0, np.pi],
        [-20.4, 12.5, 2.0, 0, 0, -np.pi * 0.5],

        [-20.4, 8.5, 2.0, 0, 0, 0.0],
        [-20.4, 8.5, 2.0, 0, 0, np.pi * 0.5],
        [-20.4, 8.5, 2.0, 0, 0, np.pi],
        [-20.4, 8.5, 2.0, 0, 0, -np.pi * 0.5],

        [-15.5, 10.15, 2.0, 0, 0, 0.0],
        [-15.5, 10.15, 2.0, 0, 0, np.pi * 0.5],
        [-15.5, 10.15, 2.0, 0, 0, np.pi],
        [-15.5, 10.15, 2.0, 0, 0, -np.pi * 0.5],

        [-16.5, -10.5, 2.0, 0, 0, 0.0],
        [-16.5, -10.5, 2.0, 0, 0, np.pi * 0.5],
        [-16.5, -10.5, 2.0, 0, 0, np.pi],
        [-16.5, -10.5, 2.0, 0, 0, -np.pi * 0.5],

        [-9.75, 2.4, 2.0, 0, 0, 0.0],
        [-9.75, 2.4, 2.0, 0, 0, np.pi * 0.5],
        [-9.75, 2.4, 2.0, 0, 0, np.pi],
        [-9.75, 2.4, 2.0, 0, 0, -np.pi * 0.5],

        [-9.75, -2.4, 2.0, 0, 0, 0.0],
        [-9.75, -2.4, 2.0, 0, 0, np.pi * 0.5],
        [-9.75, -2.4, 2.0, 0, 0, np.pi],
        [-9.75, -2.4, 2.0, 0, 0, -np.pi * 0.5],

        [-9.75, -7.2, 2.0, 0, 0, 0.0],
        [-9.75, -7.2, 2.0, 0, 0, np.pi * 0.5],
        [-9.75, -7.2, 2.0, 0, 0, np.pi],
        [-9.75, -7.2, 2.0, 0, 0, -np.pi * 0.5],

        [-9.75, 7.2, 2.0, 0, 0, 0.0],
        [-9.75, 7.2, 2.0, 0, 0, np.pi * 0.5],
        [-9.75, 7.2, 2.0, 0, 0, np.pi],
        [-9.75, 7.2, 2.0, 0, 0, -np.pi * 0.5],

        [-4.5, -7.2, 2.0, 0, 0, 0.0],
        [-4.5, -7.2, 2.0, 0, 0, np.pi * 0.5],
        [-4.5, -7.2, 2.0, 0, 0, np.pi],
        [-4.5, -7.2, 2.0, 0, 0, -np.pi * 0.5],

        [-4.5, 7.2, 2.0, 0, 0, 0.0],
        [-4.5, 7.2, 2.0, 0, 0, np.pi * 0.5],
        [-4.5, 7.2, 2.0, 0, 0, np.pi],
        [-4.5, 7.2, 2.0, 0, 0, -np.pi * 0.5],

        [-4.5, -4.2, 2.0, 0, 0, 0.0],
        [-4.5, -4.2, 2.0, 0, 0, np.pi * 0.5],
        [-4.5, -4.2, 2.0, 0, 0, np.pi],
        [-4.5, -4.2, 2.0, 0, 0, -np.pi * 0.5],

        [-4.5, 4.2, 2.0, 0, 0, 0.0],
        [-4.5, 4.2, 2.0, 0, 0, np.pi * 0.5],
        [-4.5, 4.2, 2.0, 0, 0, np.pi],
        [-4.5, 4.2, 2.0, 0, 0, -np.pi * 0.5],

        [0.75, -7.2, 2.0, 0, 0, 0.0],
        [0.75, -7.2, 2.0, 0, 0, np.pi * 0.5],
        [0.75, -7.2, 2.0, 0, 0, np.pi],
        [0.75, -7.2, 2.0, 0, 0, -np.pi * 0.5],

        [0.75, 7.2, 2.0, 0, 0, 0.0],
        [0.75, 7.2, 2.0, 0, 0, np.pi * 0.5],
        [0.75, 7.2, 2.0, 0, 0, np.pi],
        [0.75, 7.2, 2.0, 0, 0, -np.pi * 0.5],

        [0.75, -2.4, 2.0, 0, 0, 0.0],
        [0.75, -2.4, 2.0, 0, 0, np.pi * 0.5],
        [0.75, -2.4, 2.0, 0, 0, np.pi],
        [0.75, -2.4, 2.0, 0, 0, -np.pi * 0.5],
        
        [0.75, 2.4, 2.0, 0, 0, 0.0],
        [0.75, 2.4, 2.0, 0, 0, np.pi * 0.5],
        [0.75, 2.4, 2.0, 0, 0, np.pi],
        [0.75, 2.4, 2.0, 0, 0, -np.pi * 0.5],]
        env.start_position_index = 0

    # Get current position and increment index
    idx = env.start_position_index % len(env.start_positions_queue)
    position = env.start_positions_queue[idx]
    env.start_position_index += 1  # persist index across calls

    # Apply position to environment
    x, y, z, roll, pitch, yaw = position
    quat = torch.tensor(
        [*R.from_euler('z', yaw).as_quat()[[3, 0, 1, 2]]  # Quaternion (w, x, y, z)
        ],
        device=env.device, dtype=torch.float32
    )
    position_orientation = torch.tensor([
        [
            x + scene_origin[env_id, 0],
            y + scene_origin[env_id, 1],
            z,
            *quat
        ]
        for env_id in env_ids
    ], device=env.device)

    velocities = torch.zeros((len(env_ids), 6), device=env.device)
    robot.write_root_link_pose_to_sim(position_orientation, env_ids=env_ids)
    robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)
