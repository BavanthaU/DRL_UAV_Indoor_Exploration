from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _vlm_ppo_common import (
    IsaacPolicyEnvAdapter,
    add_common_args,
    build_log_dir,
    git_commit,
    load_config,
    make_debug_env,
    resolve_wandb_config,
    write_run_config,
)

HIERARCHICAL_TASK_ID = "Isaac-VLM-Hierarchical-PPO-UAV-Exploration-v0"
DEFAULT_CONFIG = "configs/vlm_hierarchical_ppo/debug_mock.yaml"


def parse_hierarchical_train_args() -> tuple[argparse.Namespace, dict[str, Any], Any | None]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(pre_args.config)
    env_backend = config.get("environment", {}).get("backend", "debug")
    simulation_app = None
    if env_backend == "debug":
        parser = argparse.ArgumentParser(description="Train hierarchical VLM-PPO UAV explorer.")
        add_common_args(parser, include_device=True)
        parser.set_defaults(config=DEFAULT_CONFIG, task=HIERARCHICAL_TASK_ID)
        args, _ = parser.parse_known_args()
    else:
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description="Train hierarchical VLM-PPO UAV explorer in Isaac Lab.")
        add_common_args(parser, include_device=False)
        parser.set_defaults(config=DEFAULT_CONFIG, task=HIERARCHICAL_TASK_ID)
        AppLauncher.add_app_launcher_args(parser)
        args, _ = parser.parse_known_args()
        if getattr(args, "record_video", False):
            args.enable_cameras = True
        simulation_app = AppLauncher(args).app
    config = load_config(args.config)
    return args, config, simulation_app


def make_hierarchical_isaac_env(config: dict[str, Any], args: argparse.Namespace):
    import gymnasium as gym

    import exploration_stack.tasks.vlm_hierarchical_ppo_exploration  # noqa: F401
    from exploration_stack.tasks.vlm_hierarchical_ppo_exploration.env_cfg import (
        IsaacVlmHierarchicalPpoUavExplorationEnvCfg,
    )

    env_cfg = IsaacVlmHierarchicalPpoUavExplorationEnvCfg()
    ppo_cfg = config.get("ppo", {})
    vlm_cfg = config.get("vlm", {})
    hierarchy_cfg = config.get("hierarchy", {})
    map_cfg = config.get("environment", {}).get("map", {})
    action_cfg = config.get("environment", {}).get("action", {})
    scene_cfg = config.get("environment", {}).get("scene", {})
    env_cfg.scene.num_envs = args.num_envs if args.num_envs is not None else int(ppo_cfg.get("num_envs", env_cfg.scene.num_envs))
    if getattr(args, "device", None) is not None:
        env_cfg.sim.device = args.device
    image_size = int(vlm_cfg.get("image_size", 224))
    env_cfg.tiled_camera.width = image_size
    env_cfg.tiled_camera.height = image_size
    env_cfg.observation_space["camera_rgb"] = [3, image_size, image_size]
    for key, value in map_cfg.items():
        if hasattr(env_cfg.map_cfg, key):
            setattr(env_cfg.map_cfg, key, value)
    for key, value in action_cfg.items():
        if hasattr(env_cfg.action_cfg, key):
            setattr(env_cfg.action_cfg, key, value)
    for key, value in scene_cfg.items():
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, value)
    for key, value in hierarchy_cfg.items():
        if hasattr(env_cfg.hierarchy_cfg, key):
            setattr(env_cfg.hierarchy_cfg, key, value)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if getattr(args, "record_video", False) else None)
    return IsaacPolicyEnvAdapter(env)


def make_env(config: dict[str, Any], args: argparse.Namespace):
    env_backend = config.get("environment", {}).get("backend", "debug")
    return make_debug_env(config, args) if env_backend == "debug" else make_hierarchical_isaac_env(config, args)


def build_hierarchical_model(config: dict[str, Any], action_dim: int = 3):
    from exploration_stack.hierarchy import HierarchicalActorCritic, HierarchicalActorCriticConfig
    from exploration_stack.vlm_frontend import VLMPolicyEncoderConfig

    vlm_cfg = config.get("vlm", {})
    hierarchy_cfg = config.get("hierarchy", {})
    encoder_cfg = VLMPolicyEncoderConfig(
        vlm_backend=vlm_cfg.get("backend", "mobileclip"),
        model_name=vlm_cfg.get("model_name"),
        image_mode=vlm_cfg.get("image_mode", "camera_plus_map"),
        freeze_vlm_encoder=bool(vlm_cfg.get("freeze_vlm_encoder", True)),
        train_vlm_lora=bool(vlm_cfg.get("train_vlm_lora", False)),
        latent_dim=int(vlm_cfg.get("latent_dim", 256)),
        memory_type=vlm_cfg.get("memory_type", "gru"),
        memory_steps=int(vlm_cfg.get("memory_steps", 8)),
        image_size=int(vlm_cfg.get("image_size", 224)),
        use_depth_line=bool(vlm_cfg.get("use_depth_line", True)),
        use_semantic_line=bool(vlm_cfg.get("use_semantic_line", True)),
        use_map_crop=bool(vlm_cfg.get("use_map_crop", True)),
        use_frontier_mask=bool(vlm_cfg.get("use_frontier_mask", True)),
        depth_line_dim=int(vlm_cfg.get("depth_line_dim", 64)),
        semantic_line_dim=int(vlm_cfg.get("semantic_line_dim", 64)),
        subgoal_feature_dim=int(vlm_cfg.get("subgoal_feature_dim", 5)),
        lora_vlm_encoder=vlm_cfg.get("lora_vlm_encoder", {"enabled": False, "rank": 8, "target_modules": []}),
    )
    return HierarchicalActorCritic(
        HierarchicalActorCriticConfig(
            encoder=encoder_cfg,
            action_dim=action_dim,
            log_std_init=float(config.get("ppo", {}).get("log_std_init", 0.0)),
            max_candidates=int(hierarchy_cfg.get("max_candidates", 16)),
            candidate_feature_dim=int(hierarchy_cfg.get("candidate_feature_dim", 18)),
            candidate_hidden_dim=int(hierarchy_cfg.get("candidate_hidden_dim", 128)),
            candidate_heads=int(hierarchy_cfg.get("candidate_heads", 4)),
            candidate_layers=int(hierarchy_cfg.get("candidate_layers", 1)),
            option_hidden_dim=int(hierarchy_cfg.get("option_hidden_dim", 256)),
            option_embedding_dim=int(hierarchy_cfg.get("option_embedding_dim", 32)),
            planner_feature_mode=hierarchy_cfg.get("planner_feature_mode", "features_only"),
            use_privileged_map_for_training=bool(hierarchy_cfg.get("use_privileged_map_for_training", False)),
            use_planner_dropout=bool(hierarchy_cfg.get("use_planner_dropout", True)),
            planner_dropout_prob=float(hierarchy_cfg.get("planner_dropout_prob", 0.2)),
            astar_feature_dropout_prob=float(hierarchy_cfg.get("astar_feature_dropout_prob", 0.2)),
            frontier_candidate_dropout_prob=float(hierarchy_cfg.get("frontier_candidate_dropout_prob", 0.1)),
        )
    )


def build_hierarchical_trainer(config: dict[str, Any], args: argparse.Namespace, model, log_dir: Path):
    import torch

    from exploration_stack.rl.ppo import HierarchicalPPOConfig, HierarchicalPPOTrainerAdapter

    ppo_cfg = config.get("ppo", {})
    backend = args.ppo_backend or ppo_cfg.get("backend", "torch_reference")
    if backend != "torch_reference":
        raise ValueError("Hierarchical PPO currently supports the torch_reference backend.")
    max_iterations = args.max_iterations if args.max_iterations is not None else int(ppo_cfg.get("max_iterations", 1))
    device = getattr(args, "device", None) or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer_cfg = HierarchicalPPOConfig(
        rollout_length=int(ppo_cfg.get("rollout_length", 64)),
        minibatches=int(ppo_cfg.get("minibatches", 4)),
        learning_epochs=int(ppo_cfg.get("learning_epochs", 5)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_param=float(ppo_cfg.get("clip_param", 0.2)),
        entropy_coef=float(ppo_cfg.get("entropy_coef", 0.01)),
        value_loss_coef=float(ppo_cfg.get("value_loss_coef", 1.0)),
        learning_rate=float(ppo_cfg.get("learning_rate", 3.0e-4)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 1.0)),
        normalize_rewards=bool(ppo_cfg.get("normalize_rewards", True)),
        option_interval_steps=int(config.get("hierarchy", {}).get("option_interval_steps", 12)),
        max_option_age_steps=int(config.get("hierarchy", {}).get("max_option_age_steps", 20)),
        safety_shield_enabled=bool(config.get("hierarchy", {}).get("safety_shield_enabled", True)),
        safety_min_depth_fraction=float(config.get("hierarchy", {}).get("safety_min_depth_fraction", 0.08)),
        log_dir=str(log_dir),
    )
    wandb_config = resolve_wandb_config(config, args, log_dir)
    return HierarchicalPPOTrainerAdapter(model, trainer_cfg, device=device, wandb_config=wandb_config), max_iterations


__all__ = [
    "DEFAULT_CONFIG",
    "HIERARCHICAL_TASK_ID",
    "build_hierarchical_model",
    "build_hierarchical_trainer",
    "build_log_dir",
    "git_commit",
    "load_config",
    "make_env",
    "parse_hierarchical_train_args",
    "write_run_config",
]
