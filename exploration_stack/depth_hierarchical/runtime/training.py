from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from exploration_stack.depth_hierarchical.curiosity import MapCandidateRND, TemporalDepthRND
from exploration_stack.depth_hierarchical.high_level import HighLevelCandidateSelectorPPO, compute_high_level_reward
from exploration_stack.depth_hierarchical.local_explorer import (
    LocalDepthActorCritic,
    LocalDoneEvaluator,
    compute_local_reward,
)
from exploration_stack.depth_hierarchical.map_memory import CANDIDATE_FEATURE_DIM, MapMemory
from exploration_stack.depth_hierarchical.transit import AStarTransitNavigator, ActiveMappingDuringTransit
from exploration_stack.rl.ppo.base import PPOTrainResult

from .mock_env import DepthHierarchicalMockEnv, DepthHierarchicalMockEnvConfig


class IsaacDepthPolicyEnvAdapter:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.unwrapped.num_envs

    @property
    def device(self):
        return self.env.unwrapped.device

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


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except ImportError:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping.")
    return data


def env_from_config(config: dict[str, Any], args: argparse.Namespace):
    env_cfg = config.get("environment", {})
    ppo_cfg = config.get("ppo", {})
    return DepthHierarchicalMockEnv(
        DepthHierarchicalMockEnvConfig(
            num_envs=int(args.num_envs or ppo_cfg.get("num_envs", env_cfg.get("num_envs", 2))),
            max_steps=int(env_cfg.get("max_steps", 64)),
            map_size=int(env_cfg.get("map_size", 32)),
            device=str(getattr(args, "device", None) or config.get("device", "cpu")),
            seed=int(getattr(args, "seed", None) or config.get("seed", 7)),
        )
    )


def isaac_env_from_config(config: dict[str, Any], args: argparse.Namespace):
    import gymnasium as gym

    import exploration_stack.tasks.depth_hierarchical_ppo_exploration  # noqa: F401
    from exploration_stack.tasks.depth_hierarchical_ppo_exploration.env_cfg import (
        IsaacDepthHierarchicalPpoUavExplorationEnvCfg,
    )

    env_cfg = IsaacDepthHierarchicalPpoUavExplorationEnvCfg()
    ppo_cfg = config.get("ppo", {})
    env_settings = config.get("environment", {})
    map_cfg = env_settings.get("map", {})
    action_cfg = env_settings.get("action", {})
    scene_cfg = env_settings.get("scene", {})
    depth_cfg = config.get("local_explorer", {})
    high_cfg = config.get("high_level", {})
    env_cfg.scene.num_envs = int(args.num_envs or ppo_cfg.get("num_envs", env_settings.get("num_envs", env_cfg.scene.num_envs)))
    if getattr(args, "device", None) is not None:
        env_cfg.sim.device = args.device
    if "office_usd_path" in env_settings:
        usd_path = Path(str(env_settings["office_usd_path"]))
        if not usd_path.is_absolute():
            usd_path = Path.cwd() / usd_path
        env_cfg.office_usd_path = str(usd_path)
    if "use_office_asset" in env_settings:
        env_cfg.use_office_asset = bool(env_settings["use_office_asset"])
    for key, value in map_cfg.items():
        if hasattr(env_cfg.map_cfg, key):
            setattr(env_cfg.map_cfg, key, value)
    for key, value in action_cfg.items():
        if hasattr(env_cfg.action_cfg, key):
            setattr(env_cfg.action_cfg, key, value)
    for key, value in scene_cfg.items():
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, value)
        elif hasattr(env_cfg.scene, key):
            setattr(env_cfg.scene, key, value)
    if "history_steps" in depth_cfg:
        env_cfg.depth_hierarchy_cfg.history_steps = int(depth_cfg["history_steps"])
    if "max_candidates" in high_cfg:
        env_cfg.depth_hierarchy_cfg.max_candidates = int(high_cfg["max_candidates"])
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if getattr(args, "record_video", False) else None)
    return IsaacDepthPolicyEnvAdapter(env)


def run_isaac_joint_training(config: dict[str, Any], args: argparse.Namespace, simulation_app=None):
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    env = isaac_env_from_config(config, args)
    device = torch.device(str(getattr(args, "device", None) or config.get("device") or env.device))
    policy = LocalDepthActorCritic().to(device)
    depth_rnd = TemporalDepthRND(device=device)
    selector = HighLevelCandidateSelectorPPO().to(device)
    map_rnd = MapCandidateRND(device=device)
    local_optimizer = torch.optim.Adam(policy.parameters(), lr=float(config.get("ppo", {}).get("learning_rate", 3.0e-4)))
    high_optimizer = torch.optim.Adam(selector.parameters(), lr=float(config.get("ppo", {}).get("high_level_learning_rate", 3.0e-4)))
    max_iterations = int(args.max_iterations or config.get("ppo", {}).get("max_iterations", 1000))
    rollout_length = int(config.get("ppo", {}).get("rollout_length", 64))
    action_cfg = config.get("environment", {}).get("action", {})
    action_scale = torch.tensor(
        [
            float(action_cfg.get("max_vx_mps", 1.0)),
            float(action_cfg.get("max_vy_mps", 0.5)),
            float(action_cfg.get("max_yaw_rate_radps", 1.0)),
        ],
        dtype=torch.float32,
        device=device,
    ).clamp_min(1.0e-6)
    log_root = Path(config.get("logging", {}).get("root", "logs/depth_hierarchical_ppo"))
    run_name = config.get("run_name", "isaac_depth_hierarchical_rnd")
    checkpoint_dir = log_root / run_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    obs = _to_device(env.reset(), device)
    total_steps = 0
    last_metrics = {}
    for update in range(max_iterations):
        local_losses = []
        high_losses = []
        depth_rnd_losses = []
        map_rnd_losses = []
        coverage = 0.0
        for _ in range(rollout_length):
            action, info = policy.act(obs)
            env_action = (action / action_scale).clamp(-1.0, 1.0)
            next_obs, env_reward, done, env_info = env.step(env_action.detach())
            next_obs = _to_device(next_obs, device)
            rnd_reward, rnd_raw = depth_rnd(
                obs,
                collision=_term_tensor(env_info, "collision", env.num_envs, device).bool(),
                min_depth=obs["depth_scalars"][:, -1, 0],
            )
            reward = torch.nan_to_num(_as_tensor(env_reward, device).float(), nan=0.0, posinf=0.0, neginf=0.0)
            reward = reward + rnd_reward.detach()
            local_loss = -(info["log_prob"] * reward.detach()).mean() + 0.5 * info["value"].square().mean()
            local_optimizer.zero_grad(set_to_none=True)
            local_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config.get("ppo", {}).get("max_grad_norm", 1.0)))
            local_optimizer.step()
            depth_rnd_losses.append(depth_rnd.update_predictor(obs))

            candidate_mask = obs["candidate_mask"].bool()
            if candidate_mask.any():
                high_out = selector(obs["candidate_features"], candidate_mask, obs["map_crop"], obs["global_map_stats"])
                selected_features = obs["candidate_features"][torch.arange(env.num_envs, device=device), high_out.selected_index]
                map_reward, map_raw = map_rnd(
                    obs["map_crop"],
                    selected_features,
                    obs["global_map_stats"],
                    obs["previous_candidate_outcome"],
                )
                new_cells = _term_tensor(env_info, "new_cells_metric_only", env.num_envs, device)
                high_reward, realized_gain = compute_high_level_reward(
                    map_rnd_reward=map_reward,
                    new_free_cells=new_cells,
                    unknown_to_known_cells=new_cells,
                    missing_patch_filled=torch.zeros_like(map_reward),
                    new_connected_region_entered=(new_cells > 0).float(),
                    astar_path_length=selected_features[:, 10].clamp_min(0.0),
                    unreachable=(~candidate_mask.any(dim=1)).float(),
                    no_gain_after_reach=(new_cells <= 0).float(),
                    collision_during_transit_or_local=_term_tensor(env_info, "collision", env.num_envs, device),
                    revisit_ratio_after_candidate=selected_features[:, -2].clamp(0.0, 1.0),
                )
                high_loss = -(high_out.log_prob * high_reward.detach()).mean() + 0.5 * high_out.value.square().mean()
                high_optimizer.zero_grad(set_to_none=True)
                high_loss.backward()
                torch.nn.utils.clip_grad_norm_(selector.parameters(), float(config.get("ppo", {}).get("max_grad_norm", 1.0)))
                high_optimizer.step()
                map_rnd_losses.append(map_rnd.update_predictor(obs["map_crop"], selected_features.detach(), obs["global_map_stats"], obs["previous_candidate_outcome"]))
                high_losses.append(float(high_loss.detach().cpu()))
            local_losses.append(float(local_loss.detach().cpu()))
            coverage = float(_term_tensor(env_info, "mapped_free_cells", env.num_envs, device).mean().detach().cpu())
            obs = next_obs
            total_steps += env.num_envs
        last_metrics = {
            "update": update + 1,
            "timesteps": total_steps,
            "local_loss": _mean(local_losses),
            "high_level_loss": _mean(high_losses),
            "depth_rnd_predictor_loss": _mean(depth_rnd_losses),
            "map_rnd_predictor_loss": _mean(map_rnd_losses),
            "mapped_free_cells": coverage,
        }
        print(json.dumps(last_metrics, sort_keys=True))
    checkpoint_path = checkpoint_dir / f"depth_hierarchical_update_{max_iterations:06d}.pt"
    torch.save(
        {
            "local_policy": policy.state_dict(),
            "depth_rnd": depth_rnd.state_dict(),
            "high_level_selector": selector.state_dict(),
            "map_rnd": map_rnd.state_dict(),
            "config": config,
            "metrics": last_metrics,
        },
        checkpoint_path,
    )
    if hasattr(env, "close"):
        env.close()
    if simulation_app is not None:
        simulation_app.close()
    print(f"[INFO] Isaac depth hierarchical training finished: updates={max_iterations} timesteps={total_steps}")
    print(f"[INFO] Checkpoint saved: {checkpoint_path}")
    return PPOTrainResult(timesteps=total_steps, updates=max_iterations, metrics=last_metrics, checkpoint_path=str(checkpoint_path))


def run_local_mock_training(config: dict[str, Any], args: argparse.Namespace):
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    env = env_from_config(config, args)
    obs = env.reset()
    policy = LocalDepthActorCritic().to(env.device)
    rnd = TemporalDepthRND(device=env.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config.get("ppo", {}).get("learning_rate", 3.0e-4)))
    max_iterations = int(args.max_iterations or config.get("ppo", {}).get("max_iterations", 1))
    last = {}
    for _ in range(max_iterations):
        out = policy(obs)
        rnd_reward, rnd_raw = rnd(obs, min_depth=obs["depth_scalars"][:, -1, 0])
        next_obs, _, _, info = env.step(out.action.detach())
        local_reward = compute_local_reward(
            rnd_reward,
            collision=info["collision"],
            near_obstacle_penalty=(obs["depth_scalars"][:, -1, 0] < 0.45).float(),
            altitude_error=torch.zeros(env.num_envs, device=env.device),
            action_smoothness=(out.action - obs["previous_action"][:, -1]).square().mean(dim=-1),
            new_cells_metric_only=info["new_cells"],
        )
        loss = -(out.log_prob * local_reward.detach()).mean() + 0.5 * out.value.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        rnd_loss = rnd.update_predictor(obs)
        obs = next_obs
        last = {"local_loss": float(loss.detach().cpu()), "depth_rnd_raw": float(rnd_raw.mean().cpu()), "depth_rnd_predictor_loss": rnd_loss}
    print(json.dumps(last, sort_keys=True))
    return last


def run_high_level_mock_training(config: dict[str, Any], args: argparse.Namespace):
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    device = torch.device(getattr(args, "device", None) or config.get("device", "cpu"))
    memory = _seed_memory()
    candidate_features, candidate_mask, candidates = memory.candidate_tensors((16, 16), device=device)
    selector = HighLevelCandidateSelectorPPO().to(device)
    map_rnd = MapCandidateRND(device=device)
    optimizer = torch.optim.Adam(selector.parameters(), lr=float(config.get("ppo", {}).get("learning_rate", 3.0e-4)))
    max_iterations = int(args.max_iterations or config.get("ppo", {}).get("max_iterations", 1))
    last = {}
    for _ in range(max_iterations):
        map_crop = memory.map_crop_tensor(device=device)
        global_stats = torch.as_tensor(memory.global_stats()[None], device=device)
        out = selector(candidate_features, candidate_mask, map_crop, global_stats)
        selected_features = candidate_features[torch.arange(candidate_features.shape[0], device=device), out.selected_index]
        previous_outcome = torch.zeros(candidate_features.shape[0], 6, device=device)
        map_rnd_reward, map_rnd_raw = map_rnd(map_crop, selected_features, global_stats, previous_outcome)
        reward, gain = compute_high_level_reward(
            map_rnd_reward=map_rnd_reward,
            new_free_cells=torch.full_like(map_rnd_reward, 12.0),
            unknown_to_known_cells=torch.full_like(map_rnd_reward, 24.0),
            missing_patch_filled=torch.ones_like(map_rnd_reward),
            new_connected_region_entered=torch.ones_like(map_rnd_reward),
            astar_path_length=selected_features[:, 10],
            unreachable=torch.zeros_like(map_rnd_reward),
            no_gain_after_reach=torch.zeros_like(map_rnd_reward),
            collision_during_transit_or_local=torch.zeros_like(map_rnd_reward),
            revisit_ratio_after_candidate=selected_features[:, -2].clamp(0.0, 1.0),
        )
        loss = -(out.log_prob * reward.detach()).mean() + 0.5 * out.value.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        map_loss = map_rnd.update_predictor(map_crop, selected_features.detach(), global_stats, previous_outcome)
        last = {"high_level_loss": float(loss.detach().cpu()), "map_rnd_raw": float(map_rnd_raw.mean().cpu()), "candidate_realized_gain": float(gain.mean().cpu()), "map_rnd_predictor_loss": map_loss}
    print(json.dumps(last, sort_keys=True))
    return last


def run_joint_mock_training(config: dict[str, Any], args: argparse.Namespace):
    local = run_local_mock_training(config, args)
    high = run_high_level_mock_training(config, args)
    metrics = {**local, **high}
    print(json.dumps({"joint": metrics}, sort_keys=True))
    return metrics


def run_mock_smoke(steps: int = 50, device: str = "cpu"):
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    env = DepthHierarchicalMockEnv(DepthHierarchicalMockEnvConfig(num_envs=2, max_steps=max(steps, 1), device=device))
    obs = env.reset()
    policy = LocalDepthActorCritic().to(env.device)
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for _ in range(steps):
        with torch.no_grad():
            action = policy(obs).action
        obs, _, done, _ = env.step(action)
    metrics = {"steps": steps, "coverage_percent": env.coverage_percent(), "done_any": bool(done.any().cpu())}
    print(json.dumps(metrics, sort_keys=True))
    return metrics


def run_profile(config: dict[str, Any], args: argparse.Namespace):
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    start = time.perf_counter()
    run_mock_smoke(steps=int(getattr(args, "steps", None) or 20), device=str(getattr(args, "device", None) or config.get("device", "cpu")))
    memory = _seed_memory()
    navigator = AStarTransitNavigator()
    active_mapping = ActiveMappingDuringTransit()
    candidates = memory.extract_candidates((16, 16))
    plan = navigator.plan_to_selected_candidate((16, 16), candidates[0], memory.grid) if candidates else None
    yaw, gain = active_mapping.choose_yaw(memory.grid.tolist(), (16, 16), 0.0)
    metrics = {
        "elapsed_s": time.perf_counter() - start,
        "candidate_count": len(candidates),
        "astar_reachable": bool(plan.reachable) if plan else False,
        "active_mapping_gain": float(gain),
        "active_mapping_yaw": float(yaw),
    }
    print(json.dumps(metrics, sort_keys=True))
    return metrics


def _seed_memory():
    memory = MapMemory()
    center = 16
    free = [(r, c) for r in range(center - 4, center + 5) for c in range(center - 4, center + 5)]
    occ = [(center - 5, c) for c in range(center - 5, center + 6)]
    memory.mark_observed(free_cells=free, occupied_cells=occ, agent_cell=(center, center))
    return memory


def _to_device(obs: dict, device):
    return {key: _as_tensor(value, device) for key, value in obs.items()}


def _as_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.tensor(value, dtype=torch.float32, device=device)


def _term_tensor(env_info, key: str, num_envs: int, device):
    value = env_info.get("reward_terms", {}).get(key, 0.0)
    if isinstance(value, torch.Tensor):
        value = value.to(device).float()
        return value if value.ndim > 0 else value.expand(num_envs)
    return torch.full((num_envs,), float(value), dtype=torch.float32, device=device)


def _mean(values):
    return float(sum(values) / max(1, len(values)))
