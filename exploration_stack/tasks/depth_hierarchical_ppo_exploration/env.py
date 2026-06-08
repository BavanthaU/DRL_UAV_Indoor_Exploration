from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("IsaacDepthHierarchicalPpoUavExplorationEnv requires PyTorch and Isaac Lab.") from exc

from exploration_stack.depth_hierarchical.local_explorer import LocalDoneEvaluator
from exploration_stack.depth_hierarchical.map_memory import CANDIDATE_FEATURE_DIM, FREE, OCCUPIED, UNKNOWN, VISITED, MapMemory, MapMemoryConfig
from exploration_stack.tasks.vlm_ppo_exploration.env import IsaacVlmPpoUavExplorationEnv
from exploration_stack.tasks.vlm_ppo_exploration.mapping import depth_line_samples
from exploration_stack.tasks.vlm_ppo_exploration.observations import depth_line_from_camera, frontier_mask_from_occupancy
from exploration_stack.tasks.vlm_ppo_exploration.terminations import local_map_complete

from .env_cfg import IsaacDepthHierarchicalPpoUavExplorationEnvCfg


class IsaacDepthHierarchicalPpoUavExplorationEnv(IsaacVlmPpoUavExplorationEnv):
    """Isaac Lab depth-only task.

    This class reuses the quadrotor physics, depth camera, and occupancy-map
    integration from the existing simulator task, but it does not expose RGB,
    VLM embeddings, semantic lines, global target IDs, frontier IDs, or A* paths
    to the local explorer.
    """

    cfg: IsaacDepthHierarchicalPpoUavExplorationEnvCfg

    def __init__(self, cfg: IsaacDepthHierarchicalPpoUavExplorationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_depth_hierarchy_buffers()
        self._local_done_evaluator = LocalDoneEvaluator()

    def _init_depth_hierarchy_buffers(self):
        history = int(self.cfg.depth_hierarchy_cfg.history_steps)
        rays = int(self.cfg.depth_hierarchy_cfg.depth_ray_count)
        scalars = int(self.cfg.depth_hierarchy_cfg.depth_scalar_dim)
        self._depth_ray_history = torch.zeros(self.num_envs, history, rays, dtype=torch.float32, device=self.device)
        self._action_history = torch.zeros(self.num_envs, history, 3, dtype=torch.float32, device=self.device)
        self._depth_scalar_history = torch.zeros(self.num_envs, history, scalars, dtype=torch.float32, device=self.device)
        self._new_cell_metric_history = torch.zeros(self.num_envs, history, dtype=torch.float32, device=self.device)
        self._last_new_cell_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._previous_candidate_outcome = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)

    def _get_observations(self) -> dict:
        self._update_map_memory()
        self._prime_initial_observation_maps()
        if not hasattr(self, "_depth_ray_history"):
            self._init_depth_hierarchy_buffers()
        depth = self._tiled_camera.data.output.get("depth")
        depth_line = depth_line_from_camera(
            depth,
            width=int(self.cfg.depth_hierarchy_cfg.depth_ray_count),
            max_depth_m=float(self.cfg.map_cfg.depth_max_range_m),
        )
        if depth_line is None:
            depth_line = torch.full(
                (self.num_envs, int(self.cfg.depth_hierarchy_cfg.depth_ray_count)),
                float(self.cfg.map_cfg.depth_max_range_m),
                device=self.device,
            )
        depth_metric = depth_line * float(self.cfg.map_cfg.depth_max_range_m)
        mapped_delta = (self._mapped_free_cells - self._prev_mapped_free_cells).clamp_min(0.0)
        self._last_new_cell_step = torch.where(
            mapped_delta > 0,
            self.episode_length_buf.long(),
            self._last_new_cell_step,
        )
        self._new_cell_metric_history = torch.roll(self._new_cell_metric_history, shifts=-1, dims=1)
        self._new_cell_metric_history[:, -1] = mapped_delta.detach()
        self._depth_ray_history = torch.roll(self._depth_ray_history, shifts=-1, dims=1)
        self._depth_ray_history[:, -1] = depth_metric.detach()
        self._action_history = torch.roll(self._action_history, shifts=-1, dims=1)
        self._action_history[:, -1] = self._actions.detach()
        self._depth_scalar_history = torch.roll(self._depth_scalar_history, shifts=-1, dims=1)
        self._depth_scalar_history[:, -1] = self._depth_scalars(depth_metric, mapped_delta)
        candidate_features, candidate_mask = self._candidate_tensors()
        obs = {
            "depth_rays": self._depth_ray_history.clone(),
            "previous_action": self._action_history.clone(),
            "depth_scalars": self._depth_scalar_history.clone(),
            "candidate_features": candidate_features,
            "candidate_mask": candidate_mask,
            "map_crop": self._map_one_hot(),
            "global_map_stats": self._global_map_stats(),
            "previous_candidate_outcome": self._previous_candidate_outcome.clone(),
        }
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        invalid_terms = self._invalid_state_terms()
        invalid_state = self._combine_invalid_terms(invalid_terms)
        depth = self._tiled_camera.data.output.get("depth")
        distances, _ = depth_line_samples(
            depth,
            width=int(self.cfg.depth_hierarchy_cfg.depth_ray_count),
            min_depth_m=float(self.cfg.map_cfg.depth_min_range_m),
            max_depth_m=float(self.cfg.map_cfg.depth_max_range_m),
        )
        if distances is None:
            min_depth = torch.full((self.num_envs,), float(self.cfg.map_cfg.depth_max_range_m), device=self.device)
        else:
            min_depth = distances.min(dim=-1).values
        near_obstacle = (float(self.cfg.action_cfg.target_altitude_m) * 0.0 + (0.45 - min_depth).clamp_min(0.0)).float()
        altitude_error = (self._safe_root_pos_w()[:, 2] - float(self.cfg.action_cfg.target_altitude_m)).abs()
        action_smoothness = (self._actions - self._prev_actions).square().mean(dim=-1)
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        reward = reward - 8.0 * invalid_state.float()
        reward = reward - 0.5 * near_obstacle
        reward = reward - 0.2 * altitude_error
        reward = reward - 0.02 * action_smoothness
        reward = reward - 0.001
        mapped_cell_delta = (self._mapped_free_cells - self._prev_mapped_free_cells).clamp_min(0.0)
        self._prev_mapped_free_cells[:] = self._mapped_free_cells
        all_terms = {
            "depth_safety_base": reward,
            "collision": invalid_state.float(),
            "near_obstacle": near_obstacle,
            "altitude_error": altitude_error,
            "action_smoothness": action_smoothness,
            "new_cells_metric_only": mapped_cell_delta,
            "mapped_free_cells": self._mapped_free_cells,
            "mapped_cell_delta": mapped_cell_delta,
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
        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        invalid_state = self._invalid_state_mask(update_altitude_counters=True)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        frontier = frontier_mask_from_occupancy(self._map_occupancy)
        frontier_count = frontier.flatten(start_dim=1).sum(dim=1).float()
        frontier_complete = local_map_complete(
            frontier_count,
            self._frontier_closed_counter,
            self.cfg.map_cfg.frontier_closed_steps,
        )
        no_candidates = ~self._candidate_tensors()[1].any(dim=1)
        stuck = self._stuck_counter >= self.cfg.map_cfg.stuck_steps
        return invalid_state | frontier_complete | no_candidates | stuck, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_depth_ray_history"):
            return
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._depth_ray_history[env_ids] = 0.0
        self._action_history[env_ids] = 0.0
        self._depth_scalar_history[env_ids] = 0.0
        self._new_cell_metric_history[env_ids] = 0.0
        self._last_new_cell_step[env_ids] = self.episode_length_buf[env_ids].long()
        self._previous_candidate_outcome[env_ids] = 0.0

    def _depth_scalars(self, depth_metric, mapped_delta):
        batch, rays = depth_metric.shape
        split = max(1, rays // 3)
        scalars = torch.zeros(batch, int(self.cfg.depth_hierarchy_cfg.depth_scalar_dim), device=self.device)
        scalars[:, 0] = depth_metric.min(dim=-1).values
        scalars[:, 1] = depth_metric.mean(dim=-1)
        scalars[:, 2] = depth_metric.std(dim=-1)
        scalars[:, 3] = depth_metric[:, :split].mean(dim=-1)
        scalars[:, 4] = depth_metric[:, split : 2 * split].mean(dim=-1)
        scalars[:, 5] = depth_metric[:, 2 * split :].mean(dim=-1)
        scalars[:, 6] = self._new_cell_metric_history.sum(dim=1) + mapped_delta
        elapsed_since_new = (self.episode_length_buf.long() - self._last_new_cell_step).float() * float(self.step_dt)
        scalars[:, 7] = elapsed_since_new
        return torch.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)

    def _candidate_tensors(self):
        features = torch.zeros(
            self.num_envs,
            int(self.cfg.depth_hierarchy_cfg.max_candidates),
            int(self.cfg.depth_hierarchy_cfg.candidate_feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.zeros(self.num_envs, int(self.cfg.depth_hierarchy_cfg.max_candidates), dtype=torch.bool, device=self.device)
        centers = self._world_to_grid(self._safe_root_pos_w()[:, :2])
        yaw = self._root_yaw_w()
        for env_id in range(self.num_envs):
            memory = MapMemory(
                MapMemoryConfig(
                    height=int(self.cfg.map_cfg.grid_size),
                    width=int(self.cfg.map_cfg.grid_size),
                    resolution_m=float(self.cfg.map_cfg.resolution_m),
                    max_candidates=int(self.cfg.depth_hierarchy_cfg.max_candidates),
                    candidate_feature_dim=CANDIDATE_FEATURE_DIM,
                )
            )
            memory.grid[:] = self._convert_isaac_grid(
                self._map_occupancy[env_id],
                self._trajectory[env_id],
            ).detach().cpu().numpy()
            memory.visited[:] = self._trajectory[env_id].detach().cpu().numpy().astype(bool)
            env_features, env_mask, _ = memory.candidate_tensors(
                (int(centers[env_id, 0].item()), int(centers[env_id, 1].item())),
                yaw_rad=float(yaw[env_id].detach().cpu()),
                device=self.device,
            )
            count = min(features.shape[1], env_features.shape[1])
            features[env_id, :count] = env_features[0, :count]
            mask[env_id, :count] = env_mask[0, :count]
        return features, mask

    def _map_one_hot(self):
        converted = self._convert_isaac_grid(self._map_occupancy, self._trajectory)
        return torch.stack(
            [
                converted == UNKNOWN,
                converted == FREE,
                converted == OCCUPIED,
                converted == VISITED,
            ],
            dim=1,
        ).float()

    def _global_map_stats(self):
        converted = self._convert_isaac_grid(self._map_occupancy, self._trajectory)
        total = float(converted.shape[1] * converted.shape[2])
        return torch.stack(
            [
                (converted == UNKNOWN).flatten(start_dim=1).sum(dim=1) / total,
                (converted == FREE).flatten(start_dim=1).sum(dim=1) / total,
                (converted == OCCUPIED).flatten(start_dim=1).sum(dim=1) / total,
                (converted == VISITED).flatten(start_dim=1).sum(dim=1) / total,
                self._trajectory.flatten(start_dim=1).sum(dim=1).float() / total,
                torch.full((self.num_envs,), float(self.cfg.map_cfg.resolution_m), device=self.device),
                torch.full((self.num_envs,), float(self.cfg.map_cfg.grid_size), device=self.device),
                torch.full((self.num_envs,), float(self.cfg.map_cfg.grid_size), device=self.device),
            ],
            dim=-1,
        )

    def _convert_isaac_grid(self, grid, trajectory):
        converted = torch.full_like(grid, UNKNOWN)
        converted = torch.where(grid == 1, torch.full_like(converted, FREE), converted)
        converted = torch.where(grid == 2, torch.full_like(converted, OCCUPIED), converted)
        converted = torch.where(trajectory.bool(), torch.full_like(converted, VISITED), converted)
        return converted
