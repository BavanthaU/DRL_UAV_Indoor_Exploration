from __future__ import annotations

from exploration_stack.tasks.vlm_ppo_exploration.env_cfg import (
    ENVIRONMENT_DIR,
    IsaacVlmPpoUavExplorationEnvCfg,
    configclass,
)


@configclass
class DepthHierarchicalPolicyCfg:
    history_steps: int = 8
    depth_ray_count: int = 64
    depth_scalar_dim: int = 8
    max_candidates: int = 64
    candidate_feature_dim: int = 24
    local_done_no_new_cells_s: float = 4.0
    local_done_repeated_ratio: float = 0.75


@configclass
class IsaacDepthHierarchicalPpoUavExplorationEnvCfg(IsaacVlmPpoUavExplorationEnvCfg):
    """Isaac Lab config for the depth-only hierarchical PPO/RND baseline."""

    office_usd_path: str = str(ENVIRONMENT_DIR / "TrainEnvOffice1.usd")
    action_space = 3
    observation_space = {
        "depth_rays": [8, 64],
        "previous_action": [8, 3],
        "depth_scalars": [8, 8],
        "candidate_features": [64, 24],
        "candidate_mask": [64],
        "map_crop": [4, 64, 64],
        "global_map_stats": 8,
        "previous_candidate_outcome": 6,
    }
    depth_hierarchy_cfg = DepthHierarchicalPolicyCfg()
