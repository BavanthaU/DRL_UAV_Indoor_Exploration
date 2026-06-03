from __future__ import annotations

from exploration_stack.tasks.vlm_ppo_exploration.env import IsaacVlmPpoUavExplorationEnv

from .env_cfg import IsaacVlmHierarchicalPpoUavExplorationEnvCfg


class IsaacVlmHierarchicalPpoUavExplorationEnv(IsaacVlmPpoUavExplorationEnv):
    """Separate hierarchical task wrapper over the Isaac VLM exploration env."""

    cfg: IsaacVlmHierarchicalPpoUavExplorationEnvCfg
