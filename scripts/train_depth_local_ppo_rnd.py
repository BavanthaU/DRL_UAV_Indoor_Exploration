from __future__ import annotations

from _depth_hierarchical_common import parse_depth_args
from exploration_stack.depth_hierarchical.runtime import run_local_mock_training


def main():
    args, config = parse_depth_args(
        "Train the depth local PPO explorer with TemporalDepthRND.",
        "configs/depth_hierarchical_ppo/train_local_depth_ppo_rnd_temporal.yaml",
    )
    return run_local_mock_training(config, args)


if __name__ == "__main__":
    main()
