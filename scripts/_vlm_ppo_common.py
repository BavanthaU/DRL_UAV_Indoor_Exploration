from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except ImportError:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config {config_path} must contain a mapping at top level.")
    return data


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def add_common_args(parser: argparse.ArgumentParser, *, include_device: bool = True) -> None:
    parser.add_argument("--config", type=str, default="configs/vlm_ppo_explorer/debug_mock_train.yaml")
    parser.add_argument("--task", type=str, default="Isaac-VLM-PPO-UAV-Exploration-v0")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ppo_backend", type=str, default=None, choices=["torch_reference", "skrl", "rsl_rl"])
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--record_video", action="store_true", default=False)
    if include_device:
        parser.add_argument("--device", type=str, default=None)


def parse_train_args() -> tuple[argparse.Namespace, dict[str, Any], Any | None]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="configs/vlm_ppo_explorer/debug_mock_train.yaml")
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(pre_args.config)
    env_backend = config.get("environment", {}).get("backend", "debug")
    simulation_app = None
    if env_backend == "debug":
        parser = argparse.ArgumentParser(description="Train VLM-PPO UAV explorer.")
        add_common_args(parser, include_device=True)
        args, _ = parser.parse_known_args()
    else:
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description="Train VLM-PPO UAV explorer in Isaac Lab.")
        add_common_args(parser, include_device=False)
        AppLauncher.add_app_launcher_args(parser)
        args, _ = parser.parse_known_args()
        if getattr(args, "record_video", False):
            args.enable_cameras = True
        simulation_app = AppLauncher(args).app
    return args, config, simulation_app


def run_id(prefix: str | None = None) -> str:
    suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{suffix}" if prefix else suffix


def build_log_dir(config: dict[str, Any], args: argparse.Namespace) -> Path:
    name = args.run_id or run_id(config.get("run_name"))
    root = Path(config.get("logging", {}).get("root", "logs/vlm_ppo_explorer"))
    return REPO_ROOT / root / name


def make_debug_env(config: dict[str, Any], args: argparse.Namespace):
    from exploration_stack.tasks.vlm_ppo_exploration.debug_env import DebugVlmPpoEnvConfig, DebugVlmPpoVectorEnv

    env_cfg = config.get("environment", {})
    ppo_cfg = config.get("ppo", {})
    vlm_cfg = config.get("vlm", {})
    num_envs = args.num_envs if args.num_envs is not None else int(ppo_cfg.get("num_envs", env_cfg.get("num_envs", 2)))
    device = args.device or config.get("device", "cpu")
    cfg = DebugVlmPpoEnvConfig(
        num_envs=num_envs,
        grid_size=int(env_cfg.get("grid_size", 32)),
        crop_size=int(env_cfg.get("crop_size", 16)),
        image_size=int(vlm_cfg.get("image_size", 64)),
        max_steps=int(env_cfg.get("max_steps", 64)),
        sensor_radius_cells=int(env_cfg.get("sensor_radius_cells", 3)),
        success_threshold=float(env_cfg.get("success_threshold", 0.60)),
        stuck_steps=int(env_cfg.get("stuck_steps", 24)),
        device=device,
        seed=int(args.seed if args.seed is not None else config.get("seed", 7)),
    )
    return DebugVlmPpoVectorEnv(cfg)


class IsaacPolicyEnvAdapter:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.unwrapped.num_envs

    def reset(self):
        result = self.env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        return self._policy_obs(obs)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, extras = result
            done = terminated | truncated
        else:
            obs, reward, done, extras = result
        reward_terms = {}
        for key, value in (extras or {}).get("log", {}).items():
            if key.startswith("Reward/"):
                reward_terms[key.removeprefix("Reward/")] = value
        return self._policy_obs(obs), reward, done, {"reward_terms": reward_terms}

    def close(self):
        self.env.close()

    @staticmethod
    def _policy_obs(obs):
        return obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs


def make_isaac_env(config: dict[str, Any], args: argparse.Namespace):
    import gymnasium as gym

    import exploration_stack.tasks.vlm_ppo_exploration  # noqa: F401
    from exploration_stack.tasks.vlm_ppo_exploration.env_cfg import IsaacVlmPpoUavExplorationEnvCfg

    env_cfg = IsaacVlmPpoUavExplorationEnvCfg()
    ppo_cfg = config.get("ppo", {})
    vlm_cfg = config.get("vlm", {})
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
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if getattr(args, "record_video", False) else None)
    return IsaacPolicyEnvAdapter(env)


def build_model(config: dict[str, Any], action_dim: int = 3):
    from exploration_stack.rl.ppo.vlm_actor_critic import VLMActorCritic
    from exploration_stack.vlm_frontend import VLMPolicyEncoderConfig

    vlm_cfg = config.get("vlm", {})
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
    return VLMActorCritic(encoder_cfg, action_dim=action_dim, log_std_init=float(config.get("ppo", {}).get("log_std_init", 0.0)))


def build_trainer(config: dict[str, Any], args: argparse.Namespace, model, log_dir: Path):
    import torch

    from exploration_stack.rl.ppo.rsl_rl_adapter import RslRlPPOTrainerAdapter
    from exploration_stack.rl.ppo.skrl_adapter import SkrlPPOTrainerAdapter
    from exploration_stack.rl.ppo.trainer_adapter import TorchPPOConfig, TorchPPOTrainerAdapter
    from exploration_stack.tasks.vlm_ppo_exploration.rnd import RNDConfig, RNDIntrinsicReward

    ppo_cfg = config.get("ppo", {})
    backend = args.ppo_backend or ppo_cfg.get("backend", "torch_reference")
    max_iterations = args.max_iterations if args.max_iterations is not None else int(ppo_cfg.get("max_iterations", 1))
    device = getattr(args, "device", None) or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    rnd_cfg = config.get("intrinsic", {}).get("rnd", {})
    rnd = None
    rnd_beta = float(rnd_cfg.get("beta", ppo_cfg.get("rnd_beta", 0.0))) if rnd_cfg.get("enabled", False) else 0.0
    if rnd_beta > 0.0:
        prompt_count = 12
        image_embedding_count = 2 if model.encoder.cfg.image_mode == "camera_plus_map" else 1
        rnd_input_dim = (
            model.encoder.cfg.latent_dim
            + prompt_count * image_embedding_count
            + model.encoder.vlm.embedding_dim
            + model.encoder.cfg.depth_line_dim
        )
        rnd = RNDIntrinsicReward(
            RNDConfig(
                input_dim=rnd_input_dim,
                hidden_dim=int(rnd_cfg.get("hidden_dim", 128)),
                output_dim=int(rnd_cfg.get("output_dim", 64)),
                learning_rate=float(rnd_cfg.get("learning_rate", 1.0e-4)),
                reward_clip=float(rnd_cfg.get("reward_clip", 5.0)),
            ),
            device=device,
        )
    torch_cfg = TorchPPOConfig(
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
        normalize_observations=bool(ppo_cfg.get("normalize_observations", True)),
        normalize_rewards=bool(ppo_cfg.get("normalize_rewards", True)),
        mixed_precision=bool(ppo_cfg.get("mixed_precision", False)),
        rnd_beta=rnd_beta,
        log_dir=str(log_dir),
    )
    if backend == "torch_reference":
        return TorchPPOTrainerAdapter(model, torch_cfg, device=device, rnd=rnd), max_iterations
    if backend == "skrl":
        return SkrlPPOTrainerAdapter(model, torch_cfg, device=device), max_iterations
    if backend == "rsl_rl":
        return RslRlPPOTrainerAdapter(model, torch_cfg, device=device), max_iterations
    raise ValueError(f"Unsupported PPO backend '{backend}'")


def write_run_config(trainer, config: dict[str, Any]) -> None:
    trainer.logger.write_config(config, git_commit=git_commit())
