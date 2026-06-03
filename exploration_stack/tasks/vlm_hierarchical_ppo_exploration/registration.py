from __future__ import annotations

TASK_ID = "Isaac-VLM-Hierarchical-PPO-UAV-Exploration-v0"


def register_task() -> None:
    try:
        import gymnasium as gym
    except ImportError:
        return
    if TASK_ID in gym.registry:
        return
    gym.register(
        id=TASK_ID,
        entry_point=(
            "exploration_stack.tasks.vlm_hierarchical_ppo_exploration.env:"
            "IsaacVlmHierarchicalPpoUavExplorationEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "exploration_stack.tasks.vlm_hierarchical_ppo_exploration.env_cfg:"
                "IsaacVlmHierarchicalPpoUavExplorationEnvCfg"
            ),
        },
    )
