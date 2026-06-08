from __future__ import annotations

TASK_ID = "Isaac-Depth-Hierarchical-PPO-UAV-Exploration-v0"


def register_task() -> None:
    try:
        import gymnasium as gym
    except ImportError:
        return
    if TASK_ID in gym.registry:
        return
    gym.register(
        id=TASK_ID,
        entry_point="exploration_stack.depth_hierarchical.runtime.mock_env:DepthHierarchicalMockEnv",
        disable_env_checker=True,
    )
