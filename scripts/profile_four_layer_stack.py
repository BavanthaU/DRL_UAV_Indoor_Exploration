#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_four_layer_smoke_test import build_mock_runner


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the mock four-layer stack.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    start = time.perf_counter()
    total_steps = 0
    for _ in range(args.episodes):
        runner = build_mock_runner()
        result = runner.run_episode(max_steps=args.steps)
        total_steps += result.steps
    elapsed = time.perf_counter() - start
    print(f"profile: episodes={args.episodes}, steps={total_steps}, elapsed_s={elapsed:.4f}, hz={total_steps / elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
