from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

from .base import PPOTrainResult, TrainerAdapter
from .rollout_logging import StructuredRunLogger
from .vlm_actor_critic import VLMActorCritic
from exploration_stack.tasks.vlm_ppo_exploration.reward_normalizer import RewardNormalizer


@dataclass
class TorchPPOConfig:
    rollout_length: int = 64
    minibatches: int = 4
    learning_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    learning_rate: float = 3.0e-4
    max_grad_norm: float = 1.0
    normalize_observations: bool = True
    normalize_rewards: bool = True
    mixed_precision: bool = False
    rnd_beta: float = 0.0
    log_dir: str = "logs/vlm_ppo_explorer/debug"


class TorchPPOTrainerAdapter(TrainerAdapter):
    """Reference PPO trainer for custom VLM actor-critic integration."""

    def __init__(self, model: VLMActorCritic, cfg: TorchPPOConfig, *, device: str = "cpu", rnd=None):
        if torch is None:
            raise RuntimeError("TorchPPOTrainerAdapter requires PyTorch.")
        self.model = model.to(device)
        self.cfg = cfg
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.logger = StructuredRunLogger(cfg.log_dir)
        self.reward_normalizer = RewardNormalizer().to(self.device) if cfg.normalize_rewards else None
        self.rnd = rnd.to(self.device) if rnd is not None else None

    def train(self, env, *, max_iterations: int) -> PPOTrainResult:
        obs = self._to_device(env.reset())
        total_timesteps = 0
        last_metrics: dict[str, float] = {}
        for update in range(max_iterations):
            rollout = self._collect_rollout(env, obs)
            obs = rollout["next_obs"]
            total_timesteps += rollout["actions"].shape[0] * rollout["actions"].shape[1]
            metrics = self._update(rollout)
            reward_terms = rollout.get("reward_terms", {})
            for key, value in reward_terms.items():
                metrics[f"reward/{key}"] = float(value)
            self.logger.log(update, metrics)
            last_metrics = metrics
        return PPOTrainResult(timesteps=total_timesteps, updates=max_iterations, metrics=last_metrics)

    def _collect_rollout(self, env, obs):
        obs_buf: list[dict] = []
        actions, log_probs, values, rewards, dones = [], [], [], [], []
        reward_term_sums: dict[str, float] = {}
        for _ in range(self.cfg.rollout_length):
            with torch.no_grad():
                action, info = self.model.act(obs)
            next_obs, reward, done, info_env = env.step(action.detach())
            reward = self._as_tensor(reward).detach()
            rnd_metrics = {}
            if self.rnd is not None and self.cfg.rnd_beta > 0.0:
                rnd_input = self._build_rnd_input(obs, info["aux"])
                rnd_reward, raw_error = self.rnd(rnd_input)
                predictor_loss = self.rnd.update_predictor(rnd_input)
                reward = reward + self.cfg.rnd_beta * rnd_reward.detach()
                rnd_metrics = {
                    "rnd_reward": float(rnd_reward.mean().detach().cpu()),
                    "rnd_raw_error": float(raw_error.mean().detach().cpu()),
                    "rnd_predictor_loss": predictor_loss,
                }
            reward_to_store = self.reward_normalizer(reward) if self.reward_normalizer is not None else reward
            obs_buf.append({key: value.detach() for key, value in obs.items()})
            actions.append(action.detach())
            log_probs.append(info["log_prob"].detach())
            values.append(info["value"].detach())
            rewards.append(reward_to_store.detach())
            dones.append(self._as_tensor(done).float().detach())
            for key, value in info_env.get("reward_terms", {}).items():
                reward_term_sums[key] = reward_term_sums.get(key, 0.0) + float(self._as_tensor(value).mean().cpu())
            for key, value in rnd_metrics.items():
                reward_term_sums[key] = reward_term_sums.get(key, 0.0) + value
            obs = self._to_device(next_obs)
        with torch.no_grad():
            next_value = self.model.forward(obs).value.detach()
        rollout = {
            "obs": self._stack_obs(obs_buf),
            "actions": torch.stack(actions),
            "log_probs": torch.stack(log_probs),
            "values": torch.stack(values),
            "rewards": torch.stack(rewards),
            "dones": torch.stack(dones),
            "next_value": next_value,
            "next_obs": obs,
            "reward_terms": reward_term_sums,
        }
        rollout["advantages"], rollout["returns"] = self._compute_gae(rollout)
        return rollout

    def _compute_gae(self, rollout):
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"]
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(rewards.shape[1], device=self.device)
        next_value = rollout["next_value"]
        for step in reversed(range(rewards.shape[0])):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.cfg.gamma * next_value * mask - values[step]
            last_advantage = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * last_advantage
            advantages[step] = last_advantage
            next_value = values[step]
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-8))
        return advantages, returns

    def _update(self, rollout):
        batch_size = rollout["actions"].shape[0] * rollout["actions"].shape[1]
        minibatch_size = max(1, batch_size // self.cfg.minibatches)
        flat = self._flatten_rollout(rollout)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(self.cfg.learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb = indices[start : start + minibatch_size]
                obs_mb = {key: value[mb] for key, value in flat["obs"].items()}
                out = self.model.evaluate_actions(obs_mb, flat["actions"][mb])
                ratio = torch.exp(out.log_prob - flat["log_probs"][mb])
                unclipped = ratio * flat["advantages"][mb]
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * flat["advantages"][mb]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(out.value, flat["returns"][mb])
                entropy = out.entropy.mean()
                loss = policy_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                metrics["policy_loss"] += float(policy_loss.detach().cpu())
                metrics["value_loss"] += float(value_loss.detach().cpu())
                metrics["entropy"] += float(entropy.detach().cpu())
        denom = max(1, self.cfg.learning_epochs * math.ceil(batch_size / minibatch_size))
        return {key: value / denom for key, value in metrics.items()}

    def _flatten_rollout(self, rollout):
        return {
            "obs": {key: value.flatten(0, 1) for key, value in rollout["obs"].items()},
            "actions": rollout["actions"].flatten(0, 1),
            "log_probs": rollout["log_probs"].flatten(0, 1),
            "advantages": rollout["advantages"].flatten(0, 1),
            "returns": rollout["returns"].flatten(0, 1),
        }

    def _stack_obs(self, obs_buf: list[dict]):
        keys = obs_buf[0].keys()
        return {key: torch.stack([obs[key] for obs in obs_buf]) for key in keys}

    def _to_device(self, obs: dict):
        return {key: self._as_tensor(value).to(self.device) for key, value in obs.items()}

    def _as_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.tensor(value, dtype=torch.float32, device=self.device)

    def _build_rnd_input(self, obs: dict, aux: dict):
        parts = [aux["z_actor"].detach()]
        if "prompt_similarity" in aux:
            parts.append(aux["prompt_similarity"].detach())
        if "map_embedding" in aux:
            parts.append(aux["map_embedding"].detach())
        elif "camera_embedding" in aux:
            parts.append(aux["camera_embedding"].detach())
        if "depth_line" in obs:
            parts.append(obs["depth_line"].detach().float().flatten(start_dim=1))
        return torch.cat(parts, dim=-1)

    def save(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "cfg": self.cfg}, output)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
