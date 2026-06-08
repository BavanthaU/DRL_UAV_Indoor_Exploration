from __future__ import annotations

from _depth_hierarchical_common import parse_depth_args
from exploration_stack.depth_hierarchical.runtime import run_high_level_mock_training


def main():
    args, config = parse_depth_args(
        "Train the learned high-level PPO candidate selector with MapCandidateRND.",
        "configs/depth_hierarchical_ppo/train_high_level_selector_rnd.yaml",
    )
    return run_high_level_mock_training(config, args)


if __name__ == "__main__":
    main()
