"""
Note:
This script was slightly modified from code originally written by a previous student, Andrea Bravo i Forn. 
For more variants of the occupancy map see her code. 
Modifications made:
- Converted radial distance depth image to planar distance Z, 
- Changed grid size to suit smaller doorways, 
- Changed default Isaac Sim camera parameters to Zed X one wide lens camera parameters,
- Changed fov to suit new camera parameters
- Changed height threshold to match new drone height. 
"""

import torch
import torch.nn.functional as F

import kornia.morphology as km
import kornia.contrib as kc
import math
import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import h5py

from isaaclab.utils.math import matrix_from_quat, euler_xyz_from_quat
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth

class BasicEnvironmentModel:
    def __init__(self, num_envs: int, device:str, tf_local_to_global_world_frame: torch.Tensor):

        self.num_envs = num_envs
        self.depth_threshold = 4.0 # 8.0
        self.device = torch.device(device)

        # Variables for PointCloud Generation
        self._image_changed = True
        self._height = None
        self._width = None
        self._Iinv = None
        self._cam_to_img_mat = None
        self._img_pixs_ones = None
        self._PointCloud = None

        self._focal_length = 2.208 # mm
        self._focus_distance = 28 # m
        self._horizontal_aperture = 5.76 # mm
        self._vertical_aperture = 3.24 # mm

        # Variables for Grid-Map generation
        self._grid_size = 0.3          # 0.5, 0.25                                     
        self._grid_orig = (-10, 10)                                                 # In meters (world frame)
        self._grid_orig_tensor = torch.tensor(self._grid_orig, device=self.device) 
        self._grid_num = (25, 50)       # (25, 50), (50, 100)                       # In number of cells
        self._height_threshold = (2.6,)
        self.env_indices = None

        # Transformation from camera frame to drone body frame (fixed)
        self.R_camera_offset = Rotation.from_euler('xz', [-90, -90], degrees=True) # P_bf =  (TF from camera to body frame) * (TF from ROS frame to world frame) * P_cf
        self.t_camera_offset = torch.tensor([0.0, 0.0, -0.25], device = self.device)

        self.E_camera_offset = torch.eye(4, device = self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.E_camera_offset[:, 0:3,0:3] = torch.tensor(self.R_camera_offset.as_matrix(), device = self.device)
        self.E_camera_offset[:, 0:3,3] = self.t_camera_offset

        self.E = torch.eye(4, device = self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)

        self.tf_local_to_global_world_frame = tf_local_to_global_world_frame
        self.drone_grid_loc = torch.zeros(self.num_envs, 2, device = self.device)


        # General variables
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.int, device = self.device)
        self._old_grid_area = torch.zeros(self.num_envs, dtype=torch.int, device = self.device)
        self._environment_map = torch.zeros((self.num_envs, self._grid_num[0], self._grid_num[1]), device = self.device, dtype=torch.uint8) # Initialize num_envs gridmaps of increasing size
        self.area_diff = torch.zeros(self.num_envs, dtype=torch.int, device = self.device)
        self.visitation_counts = torch.zeros((self.num_envs, self._grid_num[0], self._grid_num[1]), device=self.device, dtype=torch.float32)
        self.curiosity_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.curiosity_beta = 0.1
        self.frontier_snapshot = {i: None for i in range(self.num_envs)}
        self.manager_mode = "heuristic"
        self.test_global_planner = False
        self._test_map_interval = 1000
        self._last_test_map_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # For monitoring progress in RL
        self.count = torch.zeros(self.num_envs, dtype=torch.int, device = self.device)

        # Frontier planning support
        self.frontier_tolerance = 1.5
        self.frontier_distance_weight = 0.1
        self.subgoal_height = 2.0
        self.current_subgoal_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.subgoal_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_subgoal_distance = torch.zeros(self.num_envs, device=self.device)
        grid_x = (torch.arange(self._grid_num[0], device=self.device, dtype=torch.float32) + 0.5) * self._grid_size + self._grid_orig_tensor[0]
        grid_y = (torch.arange(self._grid_num[1], device=self.device, dtype=torch.float32) + 0.5) * self._grid_size + self._grid_orig_tensor[1]
        self._cell_centers_x, self._cell_centers_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        self._frontier_kernel = torch.ones(1, 1, 3, 3, device=self.device)
        self._gain_kernel = torch.ones(1, 1, 5, 5, device=self.device)
        self.manager = None
        self.manager_max_candidates = 8
        self.subgoal_reward_accum = torch.zeros(self.num_envs, device=self.device)
        self.subgoal_steps = torch.zeros(self.num_envs, device=self.device)

    def register_manager(self, manager, mode: str = "rl") -> None:
        """Attach a frontier manager that will select subgoals."""
        self.manager = manager
        self.manager_mode = mode
        if hasattr(manager, "max_candidates"):
            self.manager_max_candidates = int(manager.max_candidates)

        # Frontier planning support
        self.frontier_tolerance = 1.5
        self.frontier_distance_weight = 0.1
        self.subgoal_height = 2.0
        self.current_subgoal_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.subgoal_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_subgoal_distance = torch.zeros(self.num_envs, device=self.device)
        grid_x = (torch.arange(self._grid_num[0], device=self.device, dtype=torch.float32) + 0.5) * self._grid_size + self._grid_orig_tensor[0]
        grid_y = (torch.arange(self._grid_num[1], device=self.device, dtype=torch.float32) + 0.5) * self._grid_size + self._grid_orig_tensor[1]
        self._cell_centers_x, self._cell_centers_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        self._frontier_kernel = torch.ones(1, 1, 3, 3, device=self.device)
        self._gain_kernel = torch.ones(1, 1, 5, 5, device=self.device)

    
    @property
    def environment_map(self) -> torch.Tensor:
        return self._environment_map
    
    @property
    def reward_area_difference(self) -> torch.Tensor:
        return self.area_diff
    
    @property
    def drone_loc_in_gridmap(self) -> torch.Tensor:
        return self.drone_grid_loc
    
    @property
    def grid_size(self) -> float:
        return self._grid_size
    
    @property
    def grid_origin(self) -> torch.Tensor:
        return self._grid_orig_tensor
    
    @property
    def tf_local_to_global_world(self) -> torch.Tensor:
        return self.tf_local_to_global_world_frame
        
    def compute_intrinsics(self)-> torch.Tensor:

        fx = self._width*self._focal_length / self._horizontal_aperture
        fy = self._height*self._focal_length / self._vertical_aperture
        I = torch.tensor([[fx, 0, 0.5*self._width], [0, fy, 0.5*self._height], [0, 0, 1]], device = self.device)        
        
        return I
    
    def compute_extrinsics(self, drone_pose: torch.Tensor):

        # Idea: Compute the transformation that expresses the camera frame in the world frame: T(world frame-> drone frame) + T(drone frame -> camera frame)
        # Camera frame (ROS convention): Z - towards the picture, X - Right, Y - Down
        # Drone frame (FLU convention): Z - up wrt body, X - forward wrt body, Y - left wrt body
        # World frame (FLU convention)

        # T(drone body frame -> world frame)
        R_drone= matrix_from_quat(drone_pose[:, 3:7])   # tensor of size (N,3,3)
        t_drone = drone_pose[:, :3]                     # tensor of size (N,3)

        E_drone = torch.eye(4, device = self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        E_drone[:, 0:3,0:3] = R_drone
        E_drone[:, 0:3,3] = t_drone - self.tf_local_to_global_world_frame

        # Extrinsic matrix: P(in world frame) = T(world frame-> drone body frame)* T(drone body frame -> camera frame)* P(in camera frame)
        self.E = torch.matmul(E_drone, self.E_camera_offset)  # tensor of shape [num_env, 4, 4]
        
    def generate_PointCloud (self, depth_images: torch.Tensor):

        # Check and Update Image Size
        if self._height is None or self._height != depth_images.shape[1]:
            self._height = depth_images.shape[1]
            self._image_changed = True

        if self._width is None or self._width != depth_images.shape[2]:
            self._width = depth_images.shape[2]
            self._image_changed = True

        # Get the Homogeneous Image Coordinates of each pixel 
        if self._image_changed:
            
            u = torch.arange(0, self._height, device = self.device)                                                                           
            v = torch.arange(0, self._width, device = self.device)                                                                           
            img_pixs = torch.meshgrid(u, v, indexing="ij")                                                          # np.mgrid[0: Height, 0: Width] -> Array of size [2, Height, Width]. 1st Dim slice [0]: Row index /  1st Dim slice [1]: Column index
            img_pixs = torch.stack(img_pixs, dim=0).reshape(2, -1)                                                  # 2D Array of size (2, Height * Width). Each column of img_pixs corresponds to a single pixel's (row, col) = (u, v) coordinate in the original depth image (sorted by rows).
            img_pixs[[0, 1], :] = img_pixs[[1, 0], :]                                                               # Swap (row, col) into (col, row).
            Img_pixs_ones = torch.cat((img_pixs, torch.ones((1, img_pixs.shape[1]), device = self.device)), dim=0)  # 2D Array of size (3, Height * Width). Transforming from inhomogenious to homogeneous pixel coordinets (adding a row of 1s at the end): (col, row, 1)

        # Transform Homogeneous Image Coordinates (in 2D) to Homogeneous Camera Coordinates (in 3D)

            # Compute intinsic camera parameters
            if self._Iinv is None:
                I = self.compute_intrinsics()
                self._Iinv = torch.linalg.inv(I)

            self._cam_to_img_mat = torch.matmul(self._Iinv, Img_pixs_ones).unsqueeze(0).repeat(self.num_envs, 1, 1)  # 2D Array of size (num_envs, 3, Height * Width). Doesn't change if the input depth images don't change dimensions. Mathematically: K_i^(-1)*[c,r,1]^T
            
            self.env_indices = torch.arange(self.num_envs, device = self.device).unsqueeze(1).expand(-1, self._height*self._width)       # Array of size (num_envs, height * width) with the environment index of each point in grid_locs
            self._image_changed = False  

        # Get 3D PointCloud in Camera Coordinates
        depth_images = depth_images.reshape(self.num_envs, self._height * self._width)  # Array of size (num_envs, height * width): depth values sorted by rows (corresponding to pixels in Img_pixs_ones) 

        finite_mask = torch.isfinite(depth_images)                          # Array of size (num_envs, height * width)
        threshold_mask = depth_images < (self.depth_threshold - 0.1)
        combined_mask = finite_mask & threshold_mask
        depth_images[~combined_mask] = 0                                    # Set to 0 invalid depth values. This will add points to the 3D PointCloud at (0,0,0) - in the camera frame. In the world frame this added points will be in the drone position (later on we will use a mask to delete them).
        norm_of_direction_vector = torch.linalg.norm(self._cam_to_img_mat, dim=1) # Get norm along the (X,Y,Z) dimension

        # Convert radial distance d to planar distance Z
        Z = depth_images / norm_of_direction_vector

        # Now use Z in your final calculation
        points_in_cam =  self._cam_to_img_mat * Z.unsqueeze(1)
        # points_in_cam =  self._cam_to_img_mat * depth_images.unsqueeze(1)   # Array of size (num_envs, 3, height * width)
        points_in_cam = torch.cat((points_in_cam, torch.ones(self.num_envs, 1, self._height*self._width, device = self.device)), dim=1) # Array of size (num_envs, 4, height * width).Transforming from inhomogenious to homogeneous pixel coordinets (adding a row of 1s at the end). [X,Y,Z,1] in camera coordinates

        # Create a mask to filter out non-valid points
        self.invalid_points_mask = (points_in_cam[:, :3, :]== 0).all(dim=1) # Boolean array of size (num_envs, height * width) -> Filter out non-valid points in the PointCloud
        
        # Transform 3D PointCloud to World Coordinates
        points_in_world = torch.matmul(self.E, points_in_cam)

        self._PointCloud = points_in_world[:,:3,:].permute(0, 2, 1)         # Array of size (num_envs, height * width, 3)

        return
    
    def expand_grid(self, min_grid_locs: torch.Tensor, max_grid_locs: torch.Tensor):

       # Shift origin if necessary (to accommodate points with negative indices)
        shift_x = 0
        shift_y = 0
        if min_grid_locs[0] < 0:
            shift_x = abs(min_grid_locs[0].item())
        if min_grid_locs[1] < 0:
            shift_y = abs(min_grid_locs[1].item())

        # Calculate the new grid size by considering both the positive and negative shifts
        new_grid_size_x = max(max_grid_locs[0].item() + 1, self._grid_num[0]) + shift_x
        new_grid_size_y = max(max_grid_locs[1].item() + 1, self._grid_num[1]) + shift_y

        # Update the origin based on the shift
        self._grid_orig = (self._grid_orig[0] - shift_x * self._grid_size, self._grid_orig[1] - shift_y * self._grid_size)
        self._grid_orig_tensor = torch.tensor(self._grid_orig, device=self.device)

        # Create a new expanded grid
        new_grid = torch.zeros((self.num_envs, new_grid_size_x, new_grid_size_y), device = self.device, dtype=torch.uint8)

        # Copy old grid values into the new grid, shifted if necessary
        new_grid[:, shift_x:shift_x + self._grid_num[0], shift_y:shift_y + self._grid_num[1]] = self._environment_map

        # Update the grid size and environment map
        self._grid_num = (new_grid_size_x, new_grid_size_y)
        self._environment_map = new_grid

        return shift_x, shift_y
          
    def update_gridmap_from_PointCloud (self, drone_pose: torch.Tensor)-> torch.Tensor:
        
        # Project PointCloud on 2D grid
        grid_locs = torch.floor((self._PointCloud[:,:,:2] - self._grid_orig_tensor) / self._grid_size).to(torch.int) # tensor of size (num_envs, height * width, 2). PointCloud projection on the X-Y plane + asign each 2D point to a gridmap cell

        # Check if we need to expand the grid
        min_grid_locs = torch.min(grid_locs.reshape(-1, 2), dim=0).values # min_grid_locs[0]: Minimum X coordinate // min_grid_locs[1]: Minimum Y coordinate.
        max_grid_locs = torch.max(grid_locs.reshape(-1, 2), dim=0).values # max_grid_locs[0]: Maximum X coordinate // max_grid_locs[1]: Maximum Y coordinate.

        # If points fall outside the current grid bounds, expand the grid
        if min_grid_locs[0] < 0 or min_grid_locs[1] < 0 or max_grid_locs[0] >= self._grid_num[0] or max_grid_locs[1] >= self._grid_num[1]:
            shift_x, shift_y = self.expand_grid(min_grid_locs, max_grid_locs)
            grid_locs += torch.tensor([shift_x, shift_y], device=self.device)
        
        # Filter points based on height
        high_filter_idx = self._PointCloud[:,:,2] < self._height_threshold[0]       # Boolean array of size (num_envs, height * width) -> Filter out ceiling

        # Filter out non-valid points
        high_filter_valid_idx = ~self.invalid_points_mask & high_filter_idx

        # Mark seen space
        local_grid = torch.zeros((self.num_envs, self._grid_num[0], self._grid_num[1]), device = self.device, dtype=torch.uint8)

        env_indices_floor = self.env_indices[high_filter_valid_idx]                                                      # Array of size (#selected points across all environments)
        local_grid[env_indices_floor, grid_locs[high_filter_valid_idx][:,0], grid_locs[high_filter_valid_idx][:,1]] = 1  # Mark seen space in local map

        self.drone_grid_loc = torch.floor(((drone_pose[:, [0, 1]] - self.tf_local_to_global_world_frame[:, [0, 1]]) - self._grid_orig_tensor) / self._grid_size).to(torch.int) # Array of size (num_envs, 2) with drone position in each environment     
        
        # Update the environment map
        self._environment_map[(self._environment_map == 0) & (local_grid == 1)] = 1

        num_occupied_cells = torch.sum(self._environment_map > 0, dim=(1, 2), dtype=torch.int)    
      
        return num_occupied_cells
        
    def update_environment_map(self, depth_images: torch.Tensor, drone_pose: torch.Tensor):

        self.compute_extrinsics(drone_pose)

        self.generate_PointCloud(depth_images)

        new_grid_area = self.update_gridmap_from_PointCloud(drone_pose)  # tensor of integers of size (num_envs)

        self.area_diff = new_grid_area - self._old_grid_area         # tensor of integers of size (num_envs)
        assert torch.all(self.area_diff > -0.001), f"Unexpected area differences: {self.area_diff}. All values must be >= -0.001."
        
        self._old_grid_area = new_grid_area
        self._episode_steps += 1
        self._update_subgoals(drone_pose)
        self._update_curiosity(drone_pose[:, :3])
        if getattr(self, "test_global_planner", False):
            self._initialize_test_frontier(drone_pose[:, :3])
            self._maybe_log_test_map()


    def reset_environment_map(self, idx_reset):
        self._environment_map[idx_reset] = torch.zeros((self._grid_num[0], self._grid_num[1]), device = self.device, dtype=torch.uint8)
        self._old_grid_area [idx_reset] = 0
        self._episode_steps [idx_reset] = 0
        self.count[idx_reset] +=1 
        self.subgoal_active[idx_reset] = False
        self.prev_subgoal_distance[idx_reset] = 0.0
        if self.manager is not None:
            env_ids = torch.nonzero(idx_reset, as_tuple=False).squeeze(-1)
            if env_ids.ndim == 0 and env_ids.numel() > 0:
                env_ids = env_ids.unsqueeze(0)
            if env_ids.numel() > 0:
                rewards = self.subgoal_reward_accum[env_ids].clone()
                steps = self.subgoal_steps[env_ids].clone()
                done_flags = torch.ones_like(env_ids, dtype=torch.bool, device=self.device)
                self.manager.on_subgoal_complete(env_ids, rewards, steps, done_flags=done_flags)
                self.manager.on_reset(env_ids)
        self.current_subgoal_world[idx_reset] = 0.0
        self.subgoal_reward_accum[idx_reset] = 0.0
        self.subgoal_steps[idx_reset] = 0.0
        self.visitation_counts[idx_reset] = 0.0
        self.curiosity_reward[idx_reset] = 0.0

        return

    def _update_subgoals(self, drone_pose: torch.Tensor):
        if self._environment_map is None:
            return
        self.subgoal_reward_accum += self.area_diff.to(torch.float32)
        self.subgoal_steps += 1
        drone_pos = drone_pose[:, :3]
        if not self.subgoal_active.any():
            need_new = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            current_dist = torch.linalg.norm(self.current_subgoal_world[:, :2] - drone_pos[:, :2], dim=1)
            need_new = (~self.subgoal_active) | (current_dist <= self.frontier_tolerance)
            self.prev_subgoal_distance[~need_new & self.subgoal_active] = current_dist[~need_new & self.subgoal_active]
        coverage_ratio = (self._environment_map != 0).float().mean(dim=(1, 2))
        if need_new.any():
            env_ids = torch.nonzero(need_new, as_tuple=False).squeeze(-1)
            if env_ids.ndim == 0 and env_ids.numel() > 0:
                env_ids = env_ids.unsqueeze(0)
            if env_ids.numel() > 0:
                rewards = self.subgoal_reward_accum[env_ids].clone()
                steps = self.subgoal_steps[env_ids].clone()
                if self.manager is not None:
                    done_flags = torch.zeros_like(env_ids, dtype=torch.bool, device=self.device)
                    self.manager.on_subgoal_complete(env_ids, rewards, steps, done_flags=done_flags)
                self.subgoal_reward_accum[env_ids] = 0.0
                self.subgoal_steps[env_ids] = 0.0
            self._select_frontier_targets(drone_pos, need_new, coverage_ratio)
        if self.manager is not None and getattr(self.manager, "training", False):
            self.manager.update()

    def _select_frontier_targets(self, drone_pos: torch.Tensor, mask: torch.Tensor, coverage_ratio: torch.Tensor):
        if mask.ndim == 0:
            mask = mask.unsqueeze(0)
        unknown = (self._environment_map == 0).float().unsqueeze(1)
        free = (self._environment_map == 1).float().unsqueeze(1)
        # cells that are free and touch unknown
        neighbor_unknown = F.conv2d(unknown, self._frontier_kernel, padding=1)
        frontier_mask = (neighbor_unknown > 0.0) & (free > 0.0)
        if not frontier_mask.any():
            self.subgoal_active[mask] = False
            return
        gain = F.conv2d(unknown, self._gain_kernel, padding=2)
        gain = gain.squeeze(1)
        frontier_mask = frontier_mask.squeeze(1)
        centers_x = self._cell_centers_x.unsqueeze(0)
        centers_y = self._cell_centers_y.unsqueeze(0)
        dx = centers_x - drone_pos[:, 0].unsqueeze(1).unsqueeze(2)
        dy = centers_y - drone_pos[:, 1].unsqueeze(1).unsqueeze(2)
        distances = torch.sqrt(dx * dx + dy * dy + 1e-6)
        scores = gain - self.frontier_distance_weight * distances
        scores = scores.masked_fill(~frontier_mask, float("-inf"))

        env_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if env_ids.ndim == 0 and env_ids.numel() > 0:
            env_ids = env_ids.unsqueeze(0)

        for env_idx in env_ids.tolist():
            candidate_mask = frontier_mask[env_idx]
            if not candidate_mask.any():
                self.subgoal_active[env_idx] = False
                self.frontier_snapshot[env_idx] = None
                continue
            scores_env = scores[env_idx][candidate_mask]
            valid_count = int(torch.sum(torch.isfinite(scores_env)).item())
            if valid_count == 0:
                self.subgoal_active[env_idx] = False
                self.frontier_snapshot[env_idx] = None
                continue
            coords = candidate_mask.nonzero(as_tuple=False)
            gains_env = gain[env_idx][candidate_mask]
            distances_env = distances[env_idx][candidate_mask]
            world_x = self._cell_centers_x[coords[:, 0], coords[:, 1]]
            world_y = self._cell_centers_y[coords[:, 0], coords[:, 1]]
            headings = torch.atan2(world_y - drone_pos[env_idx, 1], world_x - drone_pos[env_idx, 0])

            sorted_scores, order = scores_env.sort(descending=True)
            take = min(self.manager_max_candidates, sorted_scores.numel())
            order = order[:take]
            if take == 0:
                self.subgoal_active[env_idx] = False
                self.frontier_snapshot[env_idx] = None
                continue
            cand_feat = torch.stack(
                [gains_env[order], distances_env[order], sorted_scores[:take], headings[order]],
                dim=1,
            )
            cand_pos = torch.stack([world_x[order], world_y[order]], dim=1)
            coords_ordered = coords[order]

            heuristic_choice = 0
            choice = heuristic_choice
            if self.manager is not None:
                chosen_idx = self.manager.select_frontier(
                    env_idx,
                    cand_feat,
                    cand_pos,
                    coverage_ratio[env_idx].item(),
                    float(self._episode_steps[env_idx].item()),
                    heuristic_action=heuristic_choice,
                )
                if chosen_idx is not None:
                    choice = max(0, min(int(chosen_idx), take - 1))

            self.current_subgoal_world[env_idx, 0] = cand_pos[choice, 0]
            self.current_subgoal_world[env_idx, 1] = cand_pos[choice, 1]
            self.current_subgoal_world[env_idx, 2] = self.subgoal_height
            self.subgoal_active[env_idx] = True
            new_dist = torch.linalg.norm(self.current_subgoal_world[env_idx, :2] - drone_pos[env_idx, :2])
            self.prev_subgoal_distance[env_idx] = new_dist
            self._store_frontier_snapshot(env_idx, candidate_mask, coords_ordered, choice)

        inactive = mask & (~self.subgoal_active)
        if inactive.any():
            self.subgoal_active[inactive] = False
            for idx in torch.nonzero(inactive, as_tuple=False).squeeze(-1).tolist():
                self.frontier_snapshot[idx] = None

    def _store_frontier_snapshot(self, env_idx: int, frontier_mask_env: torch.Tensor, coords_ordered: torch.Tensor, choice_idx: int) -> None:
        vis = torch.zeros_like(frontier_mask_env, dtype=torch.uint8).cpu()
        mask_cpu = frontier_mask_env.cpu()
        vis[mask_cpu] = 1
        coords_cpu = coords_ordered.cpu() if coords_ordered is not None else None
        if coords_cpu is not None and coords_cpu.numel() > 0 and 0 <= choice_idx < coords_cpu.shape[0]:
            cx, cy = coords_cpu[choice_idx].tolist()
            vis[int(cx), int(cy)] = 2
        drone_cell = self.drone_grid_loc[env_idx]
        gx, gy = int(drone_cell[0].item()), int(drone_cell[1].item())
        if 0 <= gx < vis.shape[0] and 0 <= gy < vis.shape[1]:
            vis[gx, gy] = 3
        self.frontier_snapshot[env_idx] = vis

    def _initialize_test_frontier(self, drone_pos: torch.Tensor) -> None:
        if not getattr(self, "test_global_planner", False):
            self._last_test_map_step = torch.zeros_like(self._last_test_map_step)
            return
        for env_idx in range(self.num_envs):
            if self.subgoal_active[env_idx]:
                continue
            grid_loc = self.drone_grid_loc[env_idx]
            if torch.any(grid_loc < 0):
                continue
            target = grid_loc + torch.tensor([0, 1], device=self.device)
            target[0] = torch.clamp(target[0], 0, self._grid_num[0] - 1)
            target[1] = torch.clamp(target[1], 0, self._grid_num[1] - 1)
            world_x = self._grid_orig_tensor[0] + (target[0].float() + 0.5) * self._grid_size
            world_y = self._grid_orig_tensor[1] + (target[1].float() + 0.5) * self._grid_size
            self.current_subgoal_world[env_idx, 0] = world_x
            self.current_subgoal_world[env_idx, 1] = world_y
            self.current_subgoal_world[env_idx, 2] = self.subgoal_height
            self.subgoal_active[env_idx] = True
            self.prev_subgoal_distance[env_idx] = torch.linalg.norm(
                self.current_subgoal_world[env_idx, :2] - drone_pos[env_idx, :2]
            )
            mask = torch.zeros_like(self._environment_map[env_idx], dtype=torch.bool)
            mask[int(target[0].item()), int(target[1].item())] = True
            coords = torch.tensor([[int(target[0].item()), int(target[1].item())]], device=self.device)
            self._store_frontier_snapshot(env_idx, mask, coords, 0)
        self.test_global_planner = False

    def _update_curiosity(self, drone_pos: torch.Tensor):
        grid_locs = self.drone_grid_loc.clone()
        valid_x = (grid_locs[:, 0] >= 0) & (grid_locs[:, 0] < self._grid_num[0])
        valid_y = (grid_locs[:, 1] >= 0) & (grid_locs[:, 1] < self._grid_num[1])
        valid = valid_x & valid_y
        if not valid.any():
            self.curiosity_reward[:] = 0.0
            return
        env_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if env_ids.ndim == 0:
            env_ids = env_ids.unsqueeze(0)
        self.curiosity_reward[:] = 0.0
        for env_idx in env_ids.tolist():
            gx = int(grid_locs[env_idx, 0].item())
            gy = int(grid_locs[env_idx, 1].item())
            self.visitation_counts[env_idx, gx, gy] += 1.0
            visits = self.visitation_counts[env_idx, gx, gy]
            self.curiosity_reward[env_idx] = float(self.curiosity_beta / torch.sqrt(visits.clamp(min=1.0)))

    def save_observations(self, drone_pose: torch.Tensor, depth_images: torch.Tensor):
            
            with h5py.File(self.save_path_data, 'a') as f:
                
                group_name = f'timestep_{self._episode_steps[0]}'
                print(group_name)
                try:
                    timestep_group = f.create_group(group_name)
                except ValueError:
                    print(f"Group '{group_name}' already exists. Accessing the existing group.")
                    timestep_group = f[group_name]

                timestep_group.attrs['timestep'] = self._episode_steps.cpu().numpy()
                
                # Store data
                PointCloud = timestep_group.create_dataset('PointCloud', data=self._PointCloud[0].cpu().numpy())
                Drone_pose = timestep_group.create_dataset('Drone_pose', data=drone_pose[0].cpu().numpy())
                Depth_image = timestep_group.create_dataset('Depth_image', data=depth_images[0].cpu().numpy())
                Reward = timestep_group.create_dataset('Reward', data=self.area_diff[0].cpu().numpy())
                area = torch.sum(self._environment_map != 0, dim=(1, 2))
                Area = timestep_group.create_dataset('Area', data=area.cpu().numpy())
                Map = timestep_group.create_dataset('Map', data=self._environment_map[0].cpu().numpy())

    def downsample_point_cloud(self, point_cloud, voxel_size):

        if point_cloud.size == 0:
            print("Warning: Point cloud is empty.")
            return np.array([])  # or handle as needed

        # Step 1: Determine the minimum coordinates (origin of the grid)
        min_coords = np.min(point_cloud, axis=0)
        
        # Step 2: Calculate voxel indices for each point
        voxel_indices = np.floor((point_cloud - min_coords) / voxel_size).astype(int)
        
        # Step 3: Combine points in the same voxel
        # We use np.unique to find unique voxel indices and aggregate points within the same voxel
        unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
        
        # Step 4: Compute the centroid of points within each voxel
        downsampled_points = np.zeros((unique_voxels.shape[0], 3))
        for i in range(unique_voxels.shape[0]):
            downsampled_points[i] = np.mean(point_cloud[inverse_indices == i], axis=0)
        
        return downsampled_points

    
class EnvironmentModelFOVTraversability (BasicEnvironmentModel):
    def __init__(self, num_envs: int, device:str, tf_local_to_global_world_frame: torch.Tensor):

        super().__init__(num_envs, device, tf_local_to_global_world_frame)

        # Variables for building the gridmap
        self._height_threshold = (1.9-0.1, 1.9+0.1)

        self.drone_2D_orientation = torch.zeros(self.num_envs, device = self.device)
        self.neigbour_cells_offset_drone = torch.stack([torch.tensor([1, 1, 1, 0, 0, 0, -1, -1, -1], device=self.device),torch.tensor([1, 0, -1, 1, 0, -1, 1, 0, -1], device=self.device)], dim=1)  # tensor of shape (9, 2)

        # Varaibles for adding the unseen 2D FOV to the map
        self.max_fov_distance =  13 #8 #1 2                                                 # num of cells in the gridmap. Adjust depending on gridmap granularity / PC depth threshold  (0.5 res, 8.0 depth -> 12 // 0.5 res, 4.0 depth -> 8)
        self.fov_x = 2 * math.atan(self._horizontal_aperture / (2 * self._focal_length))   # in radinas
        num_rays = 9  #5, 3                                                                   # Number of rays to cast. Adjust along with self.kernel_size_fov.
        self.half_fov_angle = self.fov_x / 2 - 0.1 # -0.2
        self.angles = torch.linspace(-self.half_fov_angle, self.half_fov_angle, num_rays, device=self.device)
        
        self.neigbour_cells_offset = {torch.tensor([0,0,0,0,0,0,0,0], device=self.device), torch.tensor([1,1,1,0,0,-1,-1,-1], device=self.device),torch.tensor([-1,0,1,-1,1,-1,0,1], device=self.device)}
        self.num_neigbour_cells = 8
        
        # Compute collision reward
        self.K = None     
        self.size = None                                             
        self.L = None 
        self.n = None 
        self.iter = 0

        # For monitoring when training
        self.drone_trajectory = [torch.empty((0, 2), dtype=torch.int, device=self.device) for _ in range(self.num_envs)]
        self.environment_map_ended_episodes = {i: None for i in range(self.num_envs)}
        self.drone_trajectory_ended_episodes = {i: None for i in range(self.num_envs)}
        self.frontier_snapshot = {i: None for i in range(self.num_envs)}

    
    @property
    def wandb_environment_map_dict(self) -> dict:
        return self.environment_map_ended_episodes
    
    @property
    def wandb_drone_traj_dict(self) -> dict:
        return self.drone_trajectory_ended_episodes

    @property
    def wandb_frontier_map_dict(self) -> dict:
        return self.frontier_snapshot
    
    @property
    def drone_orientation_2D(self) -> torch.Tensor:
        return self.drone_2D_orientation
    
    def reset_wandb_dicts_ended_episodes(self):
        self.environment_map_ended_episodes = {i: None for i in range(self.num_envs)}
        self.drone_trajectory_ended_episodes = {i: None for i in range(self.num_envs)}

    def update_gridmap_from_PointCloud (self, drone_pose: torch.Tensor)-> torch.Tensor:
        
        # Project PointCloud on 2D grid
        grid_locs = torch.floor((self._PointCloud[:,:,:2] - self._grid_orig_tensor) / self._grid_size).to(torch.int) # tensor of size (num_envs, height * width, 2). PointCloud projection on the X-Y plane + change origin to grid corner + asign each 2D point to a gridmap cell

        # Get drone position (row, col) and orientation (yaw) in the gridmap 
        self.drone_grid_loc = torch.floor(((drone_pose[:, [0, 1]] - self.tf_local_to_global_world_frame[:, [0, 1]]) - self._grid_orig_tensor) / self._grid_size).to(torch.int) # Array of size (num_envs, 2) with drone position in each environment     
        
        self.drone_2D_orientation = euler_xyz_from_quat(drone_pose[:,3:7])[2]  # tensor of shape (num_envs) with yaw in each environment
        cos_orientation = torch.cos(self.drone_2D_orientation)                 # tensor of shape (num_envs) with x-coordinate of the unit vector indicating the drone direction in each env
        sin_orientation = torch.sin(self.drone_2D_orientation)                 # tensor of shape (num_envs) with y-coordinate of the unit vector indicating the drone direction in each env

        FOV_end = torch.stack([self.max_fov_distance * self._grid_size * cos_orientation, self.max_fov_distance * self._grid_size * sin_orientation], dim=1)  # tensor of shape (num_envs, 2)
        drone_FOV_end = self.drone_grid_loc + torch.floor(FOV_end / self._grid_size).to(torch.int)                                                            # tensor of shape (num_envs, 2)

        drone_with_neighbors = (self.drone_grid_loc.unsqueeze(1) + self.neigbour_cells_offset_drone.unsqueeze(0))                                             # tensor of shape (num_envs, 9, 2)
        
        combined_locs = torch.cat([grid_locs.reshape(-1, 2), drone_FOV_end, drone_with_neighbors.reshape(-1, 2)], dim=0)

        # Check if we need to expand the grid
        min_grid_locs = torch.min(combined_locs, dim=0).values # min_grid_locs[0]: Minimum X coordinate // min_grid_locs[1]: Minimum Y coordinate.
        max_grid_locs = torch.max(combined_locs, dim=0).values # max_grid_locs[0]: Maximum X coordinate // max_grid_locs[1]: Maximum Y coordinate.

        # If points fall outside the current grid bounds, expand the grid
        if min_grid_locs[0] < 0 or min_grid_locs[1] < 0 or max_grid_locs[0] >= self._grid_num[0] or max_grid_locs[1] >= self._grid_num[1]:
            shift_x, shift_y = self.expand_grid(min_grid_locs, max_grid_locs)
            grid_locs += torch.tensor([shift_x, shift_y], device=self.device)
            self.drone_grid_loc += torch.tensor([shift_x, shift_y], device=self.device)
        
        # Filter points based on height.
        high_filter_obs_idx = self._PointCloud[:,:,2] < self._height_threshold[1]         # Boolean array of size (num_envs, height * width) -> Filter out points right above the UAV level
        low_filter_obs_idx = self._PointCloud[:,:,2] > self._height_threshold[0]          # Boolean array of size (num_envs, height * width) -> Filter out points right below the UAV level
        obstacle_idx = high_filter_obs_idx & low_filter_obs_idx                           # Boolean array of size (num_envs, height * width) -> Select obstacles at the UAV level
        
        # Filter out non-valid points
        obstacle_valid_idx = ~self.invalid_points_mask & obstacle_idx

        # Mark non-traversable space (obstacles)
        local_grid = torch.zeros((self.num_envs, self._grid_num[0], self._grid_num[1]), device = self.device, dtype=torch.uint8)
       
        env_indices_obstacles = self.env_indices[obstacle_valid_idx]                                                     # Array of size (#selected points across all environments)
        local_grid[env_indices_obstacles, grid_locs[obstacle_valid_idx][:,0], grid_locs[obstacle_valid_idx][:,1]] = 2    # Mark obstacles in local map
        
        # Smooth local traversable and non-traversable regions
        # local_grid_obstacles = (local_grid == 2)
        # closed_grid_obstacles = km.closing(local_grid_obstacles.unsqueeze(1), torch.ones(self.kernel_size_closing_obstacles, device = self.device ))
        # local_grid[((local_grid == 0) | (local_grid == 1)) & (closed_grid_obstacles.squeeze(1) == 1)] = 2

        # Update non-traversable in the environment map
        self._environment_map[((self._environment_map == 0) | (self._environment_map == 1)) & (local_grid == 2)] = 2

        # Mark FOV cone in the local gridmap
        # get raycasted directions wrt world frame (they were wrt drone orientation)
        cos_angles = cos_orientation.unsqueeze(1) * torch.cos(self.angles) - sin_orientation.unsqueeze(1) * torch.sin(self.angles)  # tensor of shape (num_envs, num_angles)
        sin_angles = sin_orientation.unsqueeze(1) * torch.cos(self.angles) + cos_orientation.unsqueeze(1) * torch.sin(self.angles)  # tensor of shape (num_envs, num_angles)
        
        # Initialize ray coordinates at drone position
        ray_x = self.drone_grid_loc[:, 0].unsqueeze(1).float().repeat(1, cos_angles.shape[1])                  # tensor of shape (num_envs, num_angles). Contains row cell coordinate in the gridmap of the end cell of the beam that is being ray-traced
        ray_y = self.drone_grid_loc[:, 1].unsqueeze(1).float().repeat(1, sin_angles.shape[1])                  # tensor of shape (num_envs, num_angles). Contains column cell coordinate in the gridmap of the end cell of the beam that is being ray-traced

        # Iterate over ray steps (maximum FOV distance)
        for _ in range(self.max_fov_distance):
            
            # Advance rays
            ray_x += cos_angles                                            
            ray_y += sin_angles

            # Convert to integer grid indices
            grid_x = ray_x.long()
            grid_y = ray_y.long()

            # Check bounds
            valid_mask = (grid_x >= 0) & (grid_y >= 0) & (grid_x < self._grid_num[0]) & (grid_y < self._grid_num[1]) # boolean tensor mask of shape (num_envs, num_angles) 

            # Stop processing rays out of bounds
            if not valid_mask.any():
                break

            # Mark cells as seen or stop rays at obstacles
            indices = torch.nonzero(valid_mask, as_tuple=True)              # Tuple of 2 elements: 2 tensors of size (num "alive" ray ends in all env) with the rows and column indices of valid elements in the mask (indices[0] = tensor with the env num of the valid elements, indices[1]= tensor with the angle num of the valid elements).
            grid_x_valid = grid_x[valid_mask]                               # Tensor of size (num "alive" ray ends in all env) with row index in the local gridmap of the ray ends
            grid_y_valid = grid_y[valid_mask]                               # Tensor of size (num "alive" ray ends in all env) with col index in the local gridmap of the ray ends
            local_grid_indices = (indices[0], grid_x_valid, grid_y_valid)   # Tuple of 3 elements: 3 tensors with coordinates of the "alive" ray traced ends in the local gridmap. Each tensor of shape: num_alive_ray_ends_all_envs.
            
            # Stop raycasting when hitting an obstacle (or a neigbourhood of it)
            neighbourhood_local_grid_indices = tuple((a.unsqueeze(1) + b).flatten() for a, b in zip(local_grid_indices, self.neigbour_cells_offset))  # Tuple of 3 elements: 3 tensors with coordinates of the 8-neighbouring cells of the "alive" ray traced ends in the local gridmap. Each tensor of shape: 8*num_alive_ray_ends_all_envs.
            
            neighbourhood_local_grid_indices_stacked = torch.stack(neighbourhood_local_grid_indices, dim=0)  # Tensor of shape (3, 8*num_alive_ray_ends)
            valid_neighbourhood_mask = (                                                                     # Tensor of booleans of shape (8*num_alive_ray_ends)
                (neighbourhood_local_grid_indices_stacked[1] >= 0) & 
                (neighbourhood_local_grid_indices_stacked[1] < self._grid_num[0]) &
                (neighbourhood_local_grid_indices_stacked[2] >= 0) & 
                (neighbourhood_local_grid_indices_stacked[2] < self._grid_num[1])
            )
            if (~valid_neighbourhood_mask).any():
                local_grid_indices_stacked = torch.stack(local_grid_indices, dim=0).repeat_interleave(self.num_neigbour_cells, dim=1)  # Tensor of shape (3, 8*num_alive_ray_ends)
                neighbourhood_local_grid_indices_stacked[:, ~valid_neighbourhood_mask] = local_grid_indices_stacked [:, ~valid_neighbourhood_mask]
                neighbourhood_local_grid_indices = tuple(neighbourhood_local_grid_indices_stacked[i] for i in range(3))
            
            is_ray_end_invalid = (self._environment_map[local_grid_indices] == 2)  # Size: (num "alive" ray ends)
            is_neighborhood_invalid = ((self._environment_map[neighbourhood_local_grid_indices] == 2)).view(-1, self.num_neigbour_cells).any(dim=1)  # Size: (num "alive" ray ends)           
            mask_elements = is_ray_end_invalid | is_neighborhood_invalid   # boolean tensor of size (num "alive" ray ends in all env) that indicates wihch alive ray ends impinge on an obstacle (or a neigbourhood of it).
            
            stop_mask =  torch.zeros_like(ray_x, dtype=torch.bool)
            stop_mask[indices[0][mask_elements], indices[1][mask_elements]] = 1
            if stop_mask.any():
                ray_x[stop_mask] = float('inf')  # Mark these rays as stopped
                ray_y[stop_mask] = float('inf')

            # Mark traversable region
            local_grid[local_grid_indices] = torch.where(local_grid[local_grid_indices] == 0, 1, local_grid[local_grid_indices])

        # local_grid_fov = (local_grid == 1)
        # closed_grid_fov = km.closing(local_grid_fov.unsqueeze(1), torch.ones(self.kernel_size_closing_fov, device = self.device ))
        # local_grid[(local_grid == 0) & (closed_grid_fov.squeeze(1) == 1)] = 1

        # Update added FOV the environment map         
        self._environment_map[(self._environment_map == 0) & (local_grid == 1)] = 1

        num_occupied_cells = torch.sum(self._environment_map > 0, dim=(1, 2), dtype=torch.int)    
      
        return num_occupied_cells
    
    def visualize_items(self, drone_pose:torch.Tensor, depth_images:torch.Tensor, local_grid:torch.Tensor):
        # Filter points based on height.
        high_filter_idx = self._PointCloud[0,:,2] < self._height_threshold[2]             
        high_filter_obs_idx = self._PointCloud[0,:,2] < self._height_threshold[1]         
        low_filter_obs_idx = self._PointCloud[0,:,2] > self._height_threshold[0]          
        obstacle_idx = high_filter_obs_idx & low_filter_obs_idx                           

        PointCloud_floor = self._PointCloud[0,high_filter_idx,:]
        PointCloud_obstacles = self._PointCloud[0,obstacle_idx, :]

        fig = plt.figure(figsize=(18, 15))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1])  # Define layout proportions

        # Plot 2D Occupancy Grid
        ax_occ = fig.add_subplot(gs[0, 0])  # Top-left
        occupancy_img = ax_occ.imshow(self._environment_map[0].cpu(), cmap='gray_r', origin='lower')
        cbar_occ = plt.colorbar(occupancy_img, ax=ax_occ)
        cbar_occ.set_label('Occupancy')
        ax_occ.set_title('Occupancy Grid (2D)')
        ax_occ.set_xlabel('X axis')
        ax_occ.set_ylabel('Y axis')

        # Plot Local gridmap
        ax_depth = fig.add_subplot(gs[0, 1])  # Top-right
        depth_img = ax_depth.imshow(local_grid[0].cpu(), cmap='gray_r', origin='lower')
        cbar_depth = plt.colorbar(depth_img, ax=ax_depth)
        cbar_depth.set_label('Occupancy')
        ax_depth.set_title('Local Gridmap')
        ax_depth.set_xlabel('X axis')
        ax_depth.set_ylabel('Y axis')

        # Plot Depth Image
        ax_depth = fig.add_subplot(gs[1, 0])  # Bottom-left
        depth_img = ax_depth.imshow(depth_images[0].cpu(), cmap='gray_r', origin='lower')
        cbar_depth = plt.colorbar(depth_img, ax=ax_depth)
        cbar_depth.set_label('Depth Intensity')
        ax_depth.set_title('Depth Image')
        ax_depth.set_xlabel('X axis')
        ax_depth.set_ylabel('Y axis')

        # 3D Point Cloud
        ax_3d = fig.add_subplot(gs[1, 1], projection='3d')  # Bottom right

        # Plot drone pose
        ax_3d.scatter(drone_pose[:, 0].cpu(), drone_pose[:, 1].cpu(), drone_pose[:, 2].cpu(),
                    c='g', marker='o', s=100, label='Drone Pose')

        # Downsampled Point Cloud (floor)
        Downsampled_PointCloud = self.downsample_point_cloud(PointCloud_floor.cpu().numpy(), 0.01)
        if Downsampled_PointCloud.size > 0:
            ax_3d.scatter(Downsampled_PointCloud[:, 0], Downsampled_PointCloud[:, 1], Downsampled_PointCloud[:, 2],
                        c='y', marker='o', label='Point Cloud Floor')

        # Downsampled Point Cloud (obstacles)
        Downsampled_PointCloud_obstacles = self.downsample_point_cloud(PointCloud_obstacles.cpu().numpy(), 0.01)
        if Downsampled_PointCloud_obstacles.size > 0:
            ax_3d.scatter(Downsampled_PointCloud_obstacles[:, 0], Downsampled_PointCloud_obstacles[:, 1],
                        Downsampled_PointCloud_obstacles[:, 2], c='m', marker='o', label='Point Cloud Obstacles')

        # 3D plot labels, legend, and limits
        ax_3d.set_xlabel('X Position')
        ax_3d.set_ylabel('Y Position')
        ax_3d.set_zlabel('Z Position')
        ax_3d.legend(loc='best')

        # Optional: Adjust limits
        # min_x, max_x = np.min(self._PointCloud[0][:, 0].cpu().numpy()), np.max(self._PointCloud[0][:, 0].cpu().numpy())
        # min_y, max_y = np.min(self._PointCloud[0][:, 1].cpu().numpy()), np.max(self._PointCloud[0][:, 1].cpu().numpy())
        # min_z, max_z = np.min(self._PointCloud[0][:, 2].cpu().numpy()), np.max(self._PointCloud[0][:, 2].cpu().numpy())

        min_x, max_x = np.array([5 , -5])
        min_y, max_y = np.array([15 , 50])
        min_z, max_z = np.array([0 , 3])

        ax_3d.set_xlim([min_x, max_x])
        ax_3d.set_ylim([min_y, max_y])
        ax_3d.set_zlim([min_z, max_z])

        # Add ground plane (optional)
        x_plane = np.linspace(min_x, max_x, 10)
        y_plane = np.linspace(min_y, max_y, 10)
        X, Y = np.meshgrid(x_plane, y_plane)
        Z = np.zeros_like(X)  # Ground plane at Z=0
        ax_3d.plot_surface(X, Y, Z, alpha=0.3, facecolor='grey')

        # Finalize and save
        plt.tight_layout()
        filename = '/workspace/isaac_sim_data/drone_recordings/explored_office_map/combined_figure_local_global_PC.png'
        plt.savefig(filename)
        plt.close(fig)
        
    def update_environment_map(self, depth_images: torch.Tensor, drone_pose: torch.Tensor):
        self.compute_extrinsics(drone_pose)

        self.generate_PointCloud(depth_images)

        new_grid_area = self.update_gridmap_from_PointCloud(drone_pose)  # tensor of integers of size (num_envs)

        if getattr(self, "test_global_planner", False):
            interval = getattr(self, "_test_map_interval", 1000)
            if interval > 0:
                current_step = int(self._episode_steps[0].item()) if self._episode_steps.numel() > 0 else 0
                if current_step % interval == 0:
                    env_idx = 0
                    self.environment_map_ended_episodes[env_idx] = (
                        self._environment_map[env_idx].detach().to("cpu").clone()
                    )
                    if hasattr(self, "drone_trajectory"):
                        self.drone_trajectory_ended_episodes[env_idx] = self.drone_trajectory[env_idx].clone()
                    frontier_vis = self.frontier_snapshot.get(env_idx)
                    if isinstance(frontier_vis, torch.Tensor):
                        self.frontier_snapshot[env_idx] = frontier_vis.clone()
                    else:
                        self.frontier_snapshot[env_idx] = frontier_vis
                    self._last_test_map_step[env_idx] = self._episode_steps[env_idx]
        
        self.area_diff = new_grid_area - self._old_grid_area         # tensor of integers of size (num_envs)
        self.area_diff[self._episode_steps == 0]= 0                  # Force the fist step to give area_diff =0. Otherwise the reward will be too big (everything is seen for the first time)
        assert torch.all(self.area_diff > -0.001), f"Unexpected area differences: {self.area_diff}. All values must be >= -0.001."

        self._old_grid_area = new_grid_area
        self._episode_steps += 1

        for i in range(self.num_envs):
            self.drone_trajectory[i] = torch.cat((self.drone_trajectory[i], self.drone_grid_loc[i].unsqueeze(0)), dim=0)

    def reset_environment_map(self, idx_reset):     # idx_reset: tensor of shape (num_envs) with 0/1 for the environments that don't/ do need to be reset.
        
        true_indices = torch.nonzero(idx_reset, as_tuple=False).squeeze()       # tensor of shape (num_envs to be reset)
        if true_indices.ndim == 0:                                              # If there is only 1 indx to reset, true_indices is a scalar. Convert it to a 1D tensor.
            true_indices = true_indices.unsqueeze(0)

        for idx in true_indices:

            map = self._environment_map[idx].clone()
            map[map == 3] = 1
            map[int(self.drone_grid_loc[idx,0]), int(self.drone_grid_loc[idx,1])] = 3

            self.environment_map_ended_episodes[idx.item()] = map
            self.drone_trajectory_ended_episodes[idx.item()] = self.drone_trajectory[idx]

        self._environment_map[idx_reset] = torch.zeros((self._grid_num[0], self._grid_num[1]), device = self.device, dtype=torch.uint8)
        self._old_grid_area [idx_reset] = 0
        self._episode_steps [idx_reset] = 0
        self.count[idx_reset] +=1 

        self.drone_trajectory = [torch.empty((0, 2), dtype=torch.int, device=self.device) if idx_reset[idx] == 1 else traj for idx, traj in enumerate(self.drone_trajectory)]
        self.subgoal_active[idx_reset] = False
        self.prev_subgoal_distance[idx_reset] = 0.0
        self.current_subgoal_world[idx_reset] = 0.0

        return
