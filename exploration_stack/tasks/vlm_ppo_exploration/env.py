from __future__ import annotations

try:
    import gymnasium as gym
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.envs import DirectRLEnv
    from isaaclab.sensors import TiledCamera
    from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "IsaacVlmPpoUavExplorationEnv requires Isaac Lab and PyTorch. Use the "
        "debug mock PPO script outside Isaac Lab."
    ) from exc

from .env_cfg import IsaacVlmPpoUavExplorationEnvCfg
from .intrinsic_rewards import NewCellCountCuriosity, SemanticNovelty
from .observations import crop_grid, depth_line_from_camera, frontier_mask_from_visited
from .reward_terms import RewardWeights, combine_rewards, compute_extrinsic_reward
from .terminations import altitude_out_of_bounds, local_map_complete


class IsaacVlmPpoUavExplorationEnv(DirectRLEnv):
    """Standalone VLM-PPO UAV exploration environment."""

    cfg: IsaacVlmPpoUavExplorationEnvCfg

    def __init__(self, cfg: IsaacVlmPpoUavExplorationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        grid_shape = (self.cfg.map_cfg.grid_size, self.cfg.map_cfg.grid_size)
        self._visited = torch.zeros((self.num_envs, *grid_shape), dtype=torch.bool, device=self.device)
        self._trajectory = torch.zeros_like(self._visited)
        self._mapped_free_cells = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._prev_mapped_free_cells = torch.zeros_like(self._mapped_free_cells)
        self._frontier_closed_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._home_xy_w = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self._prev_home_distance = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._stuck_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._new_cell_curiosity = NewCellCountCuriosity(
            self.num_envs,
            grid_shape,
            device=self.device,
        )
        self._semantic_novelty = SemanticNovelty(self.num_envs)
        self._reward_weights = RewardWeights(
            w_frontier_closure=self.cfg.reward_cfg.w_frontier_closure,
            w_map_progress=self.cfg.reward_cfg.w_map_progress,
            w_return_home_progress=self.cfg.reward_cfg.w_return_home_progress,
            w_collision=self.cfg.reward_cfg.w_collision,
            w_near_obstacle=self.cfg.reward_cfg.w_near_obstacle,
            w_time=self.cfg.reward_cfg.w_time,
            w_idle=self.cfg.reward_cfg.w_idle,
            w_oscillation=self.cfg.reward_cfg.w_oscillation,
            w_action_smoothness=self.cfg.reward_cfg.w_action_smoothness,
            w_altitude_error=self.cfg.reward_cfg.w_altitude_error,
            beta_count=self.cfg.reward_cfg.beta_count,
            beta_rnd=self.cfg.reward_cfg.beta_rnd,
            beta_semantic_novelty=self.cfg.reward_cfg.beta_semantic_novelty,
        )
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for key in [
                "extrinsic",
                "new_cell_count",
                "rnd",
                "semantic_novelty",
                "frontier_closure",
                "map_progress",
                "return_home_progress",
                "collision",
                "near_obstacle",
                "time",
                "idle",
                "oscillation",
                "action_smoothness",
                "altitude_error",
                "new_cells",
                "mapped_free_cells",
                "mapped_cell_delta",
                "frontier_count",
                "frontier_closed",
                "return_home_phase",
                "home_distance",
                "invalid_state",
                "invalid_nonfinite",
                "invalid_quaternion",
                "invalid_map_bounds",
                "invalid_altitude",
                "invalid_linear_speed",
                "invalid_angular_speed",
            ]
        }

    def _setup_scene(self):
        if self.cfg.use_office_asset:
            office_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.office_usd_path)
            office_cfg.func(
                "/World/envs/env_.*/Office",
                office_cfg,
                translation=self.cfg.office_translation,
            )
        self._robot = Articulation(self.cfg.robot)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions[:] = self._actions
        self._actions = torch.nan_to_num(actions.clone(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        desired_body_xy = torch.stack(
            [
                self._actions[:, 0] * self.cfg.action_cfg.max_vx_mps,
                self._actions[:, 1] * self.cfg.action_cfg.max_vy_mps,
            ],
            dim=-1,
        )
        root_pos_w = self._safe_root_pos_w()
        root_quat_w = self._safe_root_quat_w()
        root_lin_vel_w = torch.nan_to_num(self._robot.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
        root_ang_vel_w = torch.nan_to_num(self._robot.data.root_ang_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
        rot = matrix_from_quat(root_quat_w)
        desired_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        desired_vel_w[:, :2] = torch.bmm(rot[:, :2, :2], desired_body_xy.unsqueeze(-1)).squeeze(-1)
        vel_error = desired_vel_w[:, :2] - root_lin_vel_w[:, :2]
        force_xy = self.cfg.action_cfg.kp_xy_velocity * self._robot_mass * vel_error
        altitude = root_pos_w[:, 2]
        vz = root_lin_vel_w[:, 2]
        z_force = (
            self._robot_weight
            + self._robot_mass * self.cfg.action_cfg.kp_z * (self.cfg.action_cfg.target_altitude_m - altitude)
            - self._robot_mass * self.cfg.action_cfg.kd_z * vz
        )
        self._thrust[:, 0, 0] = force_xy[:, 0]
        self._thrust[:, 0, 1] = force_xy[:, 1]
        self._thrust[:, 0, 2] = z_force.clamp(0.0, 2.5 * self._robot_weight)
        yaw_rate_cmd = self._actions[:, 2] * self.cfg.action_cfg.max_yaw_rate_radps
        yaw_rate_error = yaw_rate_cmd - root_ang_vel_w[:, 2]
        max_rp_torque = float(self.cfg.action_cfg.max_roll_pitch_torque_nm)
        max_yaw_torque = float(self.cfg.action_cfg.max_yaw_torque_nm)
        self._moment[:, 0, 0] = (-self.cfg.action_cfg.angular_damping * root_ang_vel_w[:, 0]).clamp(
            -max_rp_torque, max_rp_torque
        )
        self._moment[:, 0, 1] = (-self.cfg.action_cfg.angular_damping * root_ang_vel_w[:, 1]).clamp(
            -max_rp_torque, max_rp_torque
        )
        self._moment[:, 0, 2] = (self.cfg.action_cfg.kp_yaw_rate * yaw_rate_error).clamp(
            -max_yaw_torque, max_yaw_torque
        )

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
            is_global=True,
        )

    def _get_observations(self) -> dict:
        self._update_map_memory()
        rgb = torch.nan_to_num(self._tiled_camera.data.output["rgb"].float(), nan=0.0, posinf=255.0, neginf=0.0) / 255.0
        rgb = rgb[..., :3].permute(0, 3, 1, 2).contiguous()
        depth = self._tiled_camera.data.output.get("depth")
        depth_line = depth_line_from_camera(depth, width=64)
        if depth_line is None:
            depth_line = torch.zeros(self.num_envs, 64, device=self.device)
        root_pos_w = self._safe_root_pos_w()
        centers = self._world_to_grid(root_pos_w[:, :2])
        occupancy = self._visited.to(torch.long)
        frontier = frontier_mask_from_visited(self._visited)
        obs = {
            "camera_rgb": rgb,
            "map_crop": crop_grid(occupancy, centers, self.cfg.map_cfg.crop_size),
            "frontier_mask": crop_grid(frontier.long(), centers, self.cfg.map_cfg.crop_size),
            "trajectory_mask": crop_grid(self._trajectory.long(), centers, self.cfg.map_cfg.crop_size),
            "depth_line": depth_line,
            "semantic_line": torch.zeros(self.num_envs, 64, device=self.device),
            "subgoal_features": self._subgoal_features(centers, frontier),
        }
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        invalid_terms = self._invalid_state_terms()
        invalid_state = self._combine_invalid_terms(invalid_terms)
        root_pos_w = self._safe_root_pos_w()
        occupancy = self._visited.to(torch.long)
        new_cell_reward, new_counts = self._new_cell_curiosity.update(occupancy)
        self._stuck_counter = torch.where(
            (new_counts > 0) | invalid_state,
            torch.zeros_like(self._stuck_counter),
            self._stuck_counter + 1,
        )
        semantic_novelty = torch.zeros_like(new_cell_reward)
        rnd_reward = torch.zeros_like(new_cell_reward)
        collision = invalid_state
        clearance = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        low_motion = torch.linalg.norm(self._actions[:, :2], dim=-1) < 0.05
        no_progress = new_counts <= 0
        idle = low_motion & no_progress
        yaw_flip = torch.sign(self._actions[:, 2]) != torch.sign(self._prev_actions[:, 2])
        yaw_flip &= torch.abs(self._actions[:, 2]) > 0.1
        yaw_flip &= torch.abs(self._prev_actions[:, 2]) > 0.1
        frontier = frontier_mask_from_visited(self._visited)
        frontier_count = frontier.flatten(start_dim=1).sum(dim=1).float()
        frontier_closed_now = (frontier_count <= 0) & (
            self._mapped_free_cells >= self.cfg.map_cfg.min_mapped_cells_for_completion
        )
        self._frontier_closed_counter = torch.where(
            frontier_closed_now,
            self._frontier_closed_counter + 1,
            torch.zeros_like(self._frontier_closed_counter),
        )
        frontier_complete = local_map_complete(
            frontier_count,
            self._frontier_closed_counter,
            self.cfg.map_cfg.frontier_closed_steps,
        )
        home_distance = torch.linalg.norm(root_pos_w[:, :2] - self._home_xy_w, dim=-1)
        home_distance = torch.nan_to_num(home_distance, nan=0.0, posinf=0.0, neginf=0.0)
        return_home = self._return_home_phase()
        return_home_progress = (self._prev_home_distance - home_distance).clamp_min(0.0) * return_home.float()
        return_home_progress = torch.nan_to_num(return_home_progress, nan=0.0, posinf=0.0, neginf=0.0)
        extrinsic, extrinsic_terms = compute_extrinsic_reward(
            map_progress=new_cell_reward.detach(),
            frontier_closed=frontier_complete,
            collision=collision,
            esdf_clearance=clearance,
            safety_radius=0.45,
            dt=self.step_dt,
            idle_mask=idle,
            yaw_flip=yaw_flip,
            action=self._actions,
            prev_action=self._prev_actions,
            altitude=root_pos_w[:, 2],
            target_altitude=self.cfg.action_cfg.target_altitude_m,
            return_home_progress=return_home_progress,
            weights=self._reward_weights,
        )
        reward, intrinsic_terms = combine_rewards(
            extrinsic,
            new_cell_reward,
            rnd_reward,
            semantic_novelty,
            self._reward_weights,
        )
        mapped_cell_delta = (self._mapped_free_cells - self._prev_mapped_free_cells).clamp_min(0.0)
        self._prev_mapped_free_cells[:] = self._mapped_free_cells
        self._prev_home_distance[:] = home_distance
        all_terms = {
            **extrinsic_terms,
            **intrinsic_terms,
            "new_cells": new_counts,
            "mapped_free_cells": self._mapped_free_cells,
            "mapped_cell_delta": mapped_cell_delta,
            "frontier_count": frontier_count,
            "frontier_closed": frontier_complete.float(),
            "return_home_phase": return_home.float(),
            "home_distance": home_distance,
            "invalid_state": invalid_state.float(),
            **{key: value.float() for key, value in invalid_terms.items()},
        }
        for key, value in all_terms.items():
            value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if key in self._episode_sums:
                self._episode_sums[key] += value.detach()
            all_terms[key] = value
        if "log" not in self.extras:
            self.extras["log"] = {}
        for key, value in all_terms.items():
            self.extras["log"][f"Reward/{key}"] = float(value.mean().detach().cpu())
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        invalid_state = self._invalid_state_mask()
        root_pos_w = self._safe_root_pos_w()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        altitude_done = altitude_out_of_bounds(
            self._robot.data.root_pos_w[:, 2],
            self.cfg.action_cfg.min_altitude_m,
            self.cfg.action_cfg.max_altitude_m,
        )
        frontier = frontier_mask_from_visited(self._visited)
        frontier_count = frontier.flatten(start_dim=1).sum(dim=1).float()
        frontier_complete = local_map_complete(
            frontier_count,
            self._frontier_closed_counter,
            self.cfg.map_cfg.frontier_closed_steps,
        )
        home_distance = torch.linalg.norm(root_pos_w[:, :2] - self._home_xy_w, dim=-1)
        returned_home = self._return_home_phase() & (home_distance <= self.cfg.map_cfg.home_reached_radius_m)
        stuck = self._stuck_counter >= self.cfg.map_cfg.stuck_steps
        return invalid_state | altitude_done | frontier_complete | returned_home | stuck, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._log_resets(env_ids)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._visited[env_ids] = False
        self._trajectory[env_ids] = False
        self._mapped_free_cells[env_ids] = 0.0
        self._prev_mapped_free_cells[env_ids] = 0.0
        self._frontier_closed_counter[env_ids] = 0
        self._prev_home_distance[env_ids] = 0.0
        self._stuck_counter[env_ids] = 0
        self._new_cell_curiosity.reset(env_ids)
        self._semantic_novelty.reset(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        root_state[:, 2] = self.cfg.action_cfg.target_altitude_m
        root_state[:, 0:2] += torch.zeros_like(root_state[:, 0:2]).uniform_(-1.0, 1.0)
        self._home_xy_w[env_ids] = root_state[:, :2]
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]), env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._update_map_memory(env_ids=env_ids)
        self._prev_mapped_free_cells[env_ids] = self._mapped_free_cells[env_ids]
        self._new_cell_curiosity.prime(self._visited, env_ids)
        self._stuck_counter[env_ids] = 0

    def _update_map_memory(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(self.device).long()
        else:
            ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        centers = self._world_to_grid(self._safe_root_pos_w()[:, :2])
        radius = self.cfg.map_cfg.sensor_radius_cells
        for env_id in ids.tolist():
            row = int(centers[env_id, 0].item())
            col = int(centers[env_id, 1].item())
            row0, row1 = max(0, row - radius), min(self.cfg.map_cfg.grid_size, row + radius + 1)
            col0, col1 = max(0, col - radius), min(self.cfg.map_cfg.grid_size, col + radius + 1)
            self._visited[env_id, row0:row1, col0:col1] = True
            self._trajectory[env_id, row, col] = True
        new_counts = self._visited.flatten(start_dim=1).sum(dim=1)
        self._mapped_free_cells = new_counts.float()

    def _world_to_grid(self, xy_w):
        xy_w = torch.where(torch.isfinite(xy_w), xy_w, self._terrain.env_origins[:, :2])
        xy_local = xy_w - self._terrain.env_origins[:, :2]
        center = self.cfg.map_cfg.grid_size // 2
        cells = torch.floor(xy_local / self.cfg.map_cfg.resolution_m).long() + center
        return cells.clamp(0, self.cfg.map_cfg.grid_size - 1)

    def _subgoal_features(self, centers, frontier):
        root_pos_w = self._safe_root_pos_w()
        features = torch.zeros(self.num_envs, 5, device=self.device)
        for env_id in range(self.num_envs):
            if self._return_home_phase()[env_id]:
                vector = self._home_xy_w[env_id] - root_pos_w[env_id, :2]
                distance = torch.linalg.norm(vector)
                bearing = torch.atan2(vector[1], vector[0])
                features[env_id, 0] = distance
                features[env_id, 1] = bearing
                features[env_id, 3] = 1.0
                features[env_id, 4] = self._stuck_counter[env_id].float() / max(1, self.cfg.map_cfg.stuck_steps)
                continue
            cells = torch.nonzero(frontier[env_id], as_tuple=False)
            if cells.numel() == 0:
                continue
            delta = cells.float() - centers[env_id].float().unsqueeze(0)
            dist = torch.linalg.norm(delta, dim=1)
            best = torch.argmin(dist)
            target = cells[best].float()
            vector = (target - centers[env_id].float()) * self.cfg.map_cfg.resolution_m
            distance = torch.linalg.norm(vector)
            bearing = torch.atan2(vector[1], vector[0])
            features[env_id, 0] = distance
            features[env_id, 1] = bearing
            features[env_id, 2] = dist[best] * self.cfg.map_cfg.resolution_m
            features[env_id, 3] = 1.0
            features[env_id, 4] = self._stuck_counter[env_id].float() / max(1, self.cfg.map_cfg.stuck_steps)
        return features

    def _return_home_phase(self):
        elapsed_s = self.episode_length_buf.float() * float(self.step_dt)
        return elapsed_s >= float(self.cfg.map_cfg.return_home_after_s)

    def _log_resets(self, env_ids):
        if "log" not in self.extras:
            self.extras["log"] = {}
        for key, values in self._episode_sums.items():
            episode_values = torch.nan_to_num(values[env_ids], nan=0.0, posinf=0.0, neginf=0.0)
            self.extras["log"][f"Episode/{key}"] = float(episode_values.mean().detach().cpu())
            values[env_ids] = 0.0

    def _invalid_state_mask(self):
        return self._combine_invalid_terms(self._invalid_state_terms())

    def _invalid_state_terms(self):
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        root_lin_vel_w = self._robot.data.root_lin_vel_w
        root_ang_vel_w = self._robot.data.root_ang_vel_w
        tensors = [root_pos_w, root_quat_w, root_lin_vel_w, root_ang_vel_w]
        invalid_nonfinite = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for tensor in tensors:
            invalid_nonfinite |= ~torch.isfinite(tensor).all(dim=-1)
        quat_norm = torch.linalg.norm(torch.nan_to_num(root_quat_w, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
        invalid_quaternion = quat_norm <= 1.0e-6
        finite_pos = torch.isfinite(root_pos_w).all(dim=-1)
        local_xy = torch.linalg.norm(
            torch.nan_to_num(root_pos_w[:, :2] - self._terrain.env_origins[:, :2], nan=0.0, posinf=0.0, neginf=0.0),
            dim=-1,
        )
        altitude = root_pos_w[:, 2]
        invalid_map_bounds = finite_pos & (local_xy > self._map_radius_limit_m())
        invalid_altitude = torch.isfinite(altitude) & (
            (altitude < self.cfg.action_cfg.min_altitude_m) | (altitude > self.cfg.action_cfg.max_altitude_m)
        )
        finite_lin_vel = torch.isfinite(root_lin_vel_w).all(dim=-1)
        finite_ang_vel = torch.isfinite(root_ang_vel_w).all(dim=-1)
        lin_speed = torch.linalg.norm(torch.nan_to_num(root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
        ang_speed = torch.linalg.norm(torch.nan_to_num(root_ang_vel_w, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
        invalid_linear_speed = finite_lin_vel & (lin_speed > 20.0)
        invalid_angular_speed = finite_ang_vel & (ang_speed > 100.0)
        return {
            "invalid_nonfinite": invalid_nonfinite,
            "invalid_quaternion": invalid_quaternion,
            "invalid_map_bounds": invalid_map_bounds,
            "invalid_altitude": invalid_altitude,
            "invalid_linear_speed": invalid_linear_speed,
            "invalid_angular_speed": invalid_angular_speed,
        }

    @staticmethod
    def _combine_invalid_terms(invalid_terms):
        invalid = None
        for value in invalid_terms.values():
            invalid = value.clone() if invalid is None else invalid | value
        return invalid

    def _safe_root_pos_w(self):
        root_pos_w = self._robot.data.root_pos_w
        fallback = torch.zeros_like(root_pos_w)
        fallback[:, :2] = self._home_xy_w
        fallback[:, 2] = self.cfg.action_cfg.target_altitude_m
        finite_root = torch.where(torch.isfinite(root_pos_w), root_pos_w, fallback)
        return torch.where(self._unsafe_pose_mask(root_pos_w).unsqueeze(-1), fallback, finite_root)

    def _safe_root_quat_w(self):
        quat = torch.nan_to_num(self._robot.data.root_quat_w, nan=0.0, posinf=0.0, neginf=0.0)
        norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        identity = torch.zeros_like(quat)
        identity[:, 0] = 1.0
        invalid = norm.squeeze(-1) <= 1e-6
        quat = torch.where(invalid.unsqueeze(-1), identity, quat / norm.clamp_min(1e-6))
        return quat

    def _unsafe_pose_mask(self, root_pos_w):
        finite_pose = torch.isfinite(root_pos_w).all(dim=-1)
        sanitized = torch.nan_to_num(root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
        local_xy = torch.linalg.norm(sanitized[:, :2] - self._terrain.env_origins[:, :2], dim=-1)
        max_z = max(10.0, float(self.cfg.action_cfg.max_altitude_m) * 4.0)
        return (~finite_pose) | (local_xy > self._map_radius_limit_m() * 4.0) | (sanitized[:, 2].abs() > max_z)

    def _map_radius_limit_m(self):
        half_width = 0.5 * float(self.cfg.map_cfg.grid_size) * float(self.cfg.map_cfg.resolution_m)
        sensor_margin = float(self.cfg.map_cfg.sensor_radius_cells) * float(self.cfg.map_cfg.resolution_m)
        return half_width + sensor_margin
