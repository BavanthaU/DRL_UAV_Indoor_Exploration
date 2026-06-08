from __future__ import annotations

from _depth_hierarchical_common import parse_depth_train_args
from exploration_stack.depth_hierarchical.runtime import run_isaac_joint_training, run_joint_mock_training


def main():
    args, config, simulation_app = parse_depth_train_args(
        "Train depth hierarchical PPO/RND with alternating local and high-level updates.",
        "configs/depth_hierarchical_ppo/train_joint_rnd_temporal_wandb.yaml",
    )
    if config.get("environment", {}).get("backend", "debug") == "debug":
        return run_joint_mock_training(config, args)
    return run_isaac_joint_training(config, args, simulation_app=simulation_app)


if __name__ == "__main__":
    main()
