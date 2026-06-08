from __future__ import annotations

from _depth_hierarchical_common import parse_depth_args
from exploration_stack.depth_hierarchical.runtime import run_mock_smoke


def main():
    args, config = parse_depth_args(
        "Evaluate the depth hierarchical PPO/RND baseline in mock mode.",
        "configs/depth_hierarchical_ppo/eval_rnd_baseline.yaml",
    )
    steps = int(args.steps or config.get("environment", {}).get("max_steps", 50))
    return run_mock_smoke(steps=steps, device=str(args.device or config.get("device", "cpu")))


if __name__ == "__main__":
    main()
