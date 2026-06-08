from __future__ import annotations

from _depth_hierarchical_common import parse_depth_args
from exploration_stack.depth_hierarchical.runtime import run_profile


def main():
    args, config = parse_depth_args(
        "Profile depth hierarchical PPO/RND map, planning, and policy utilities.",
        "configs/depth_hierarchical_ppo/train_joint_rnd_temporal_wandb.yaml",
    )
    return run_profile(config, args)


if __name__ == "__main__":
    main()
