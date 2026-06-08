from __future__ import annotations

import argparse

from _depth_hierarchical_common import REPO_ROOT
from exploration_stack.depth_hierarchical.runtime import run_mock_smoke


def main():
    parser = argparse.ArgumentParser(description="Run a fast mock smoke test for depth hierarchical PPO/RND.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    return run_mock_smoke(steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()
