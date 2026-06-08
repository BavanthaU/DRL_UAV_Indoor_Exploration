from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exploration_stack.depth_hierarchical.runtime import load_config


def add_depth_args(parser: argparse.ArgumentParser, default_config: str, *, include_device: bool = True):
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--task", type=str, default="Isaac-Depth-Hierarchical-PPO-UAV-Exploration-v0")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    if include_device:
        parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--record_video", action="store_true", default=False)


def parse_depth_args(description: str, default_config: str):
    parser = argparse.ArgumentParser(description=description)
    add_depth_args(parser, default_config)
    args = parser.parse_args()
    return args, load_config(args.config)


def parse_depth_train_args(description: str, default_config: str):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=default_config)
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(pre_args.config)
    backend = config.get("environment", {}).get("backend", "debug")
    simulation_app = None
    if backend == "debug":
        parser = argparse.ArgumentParser(description=description)
        add_depth_args(parser, default_config, include_device=True)
        args = parser.parse_args()
    else:
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description=description)
        add_depth_args(parser, default_config, include_device=False)
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        if getattr(args, "record_video", False):
            args.enable_cameras = True
        simulation_app = AppLauncher(args).app
    return args, load_config(args.config), simulation_app
