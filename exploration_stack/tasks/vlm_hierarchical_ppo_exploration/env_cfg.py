from __future__ import annotations

from exploration_stack.tasks.vlm_ppo_exploration.env_cfg import IsaacVlmPpoUavExplorationEnvCfg, configclass


@configclass
class HierarchicalPolicyCfg:
    planner_feature_mode: str = "features_only"
    use_privileged_map_for_training: bool = False
    max_candidates: int = 16
    option_interval_steps: int = 12
    planner_dropout_prob: float = 0.2
    astar_feature_dropout_prob: float = 0.2
    frontier_candidate_dropout_prob: float = 0.1
    safety_shield_enabled: bool = True


@configclass
class IsaacVlmHierarchicalPpoUavExplorationEnvCfg(IsaacVlmPpoUavExplorationEnvCfg):
    """Isaac Lab config for learned hierarchical VLM-PPO exploration."""

    hierarchy_cfg = HierarchicalPolicyCfg()
