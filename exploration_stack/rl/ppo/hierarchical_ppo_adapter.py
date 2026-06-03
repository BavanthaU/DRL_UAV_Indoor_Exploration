from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from exploration_stack.hierarchy import (
    DepthLineSafetyShield,
    DepthLineSafetyShieldConfig,
    HierarchicalActorCritic,
    OPTION_NAMES,
    OptionManager,
    OptionManagerConfig,
)
from exploration_stack.tasks.vlm_ppo_exploration.reward_normalizer import RewardNormalizer

from .base import PPOTrainResult, TrainerAdapter
from .losses import ppo_clipped_policy_loss, value_mse_loss
from .rollout_logging import StructuredRunLogger
from .rollout_storage_hierarchical import flatten_hierarchical_rollout, stack_observations


@dataclass
class HierarchicalPPOConfig:
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
    normalize_rewards: bool = True
    option_interval_steps: int = 12
    max_option_age_steps: int = 20
    safety_shield_enabled: bool = True
    safety_min_depth_fraction: float = 0.08
    log_dir: str = "logs/vlm_hierarchical_ppo/debug"


class HierarchicalPPOTrainerAdapter(TrainerAdapter):
    """Reference PPO trainer for the learned semantic-option hierarchy."""

    def __init__(
        self,
        model: HierarchicalActorCritic,
        cfg: HierarchicalPPOConfig,
        *,
        device: str = "cpu",
        wandb_config: dict | None = None,
    ):
        if torch is None:
            raise RuntimeError("HierarchicalPPOTrainerAdapter requires PyTorch.")
        self.model = model.to(device)
        self.cfg = cfg
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.logger = StructuredRunLogger(cfg.log_dir, wandb_config=wandb_config)
        self.reward_normalizer = RewardNormalizer().to(self.device) if cfg.normalize_rewards else None
        self.safety_shield = DepthLineSafetyShield(
            DepthLineSafetyShieldConfig(
                enabled=cfg.safety_shield_enabled,
                min_depth_fraction=cfg.safety_min_depth_fraction,
            )
        )
        self.option_manager: OptionManager | None = None

    def train(self, env, *, max_iterations: int) -> PPOTrainResult:
        obs = self._to_device(env.reset())
        total_timesteps = 0
        last_metrics: dict[str, float] = {}
        for update in range(max_iterations):
            rollout = self._collect_rollout(env, obs)
            obs = rollout["next_obs"]
            total_timesteps += rollout["actions"].shape[0] * rollout["actions"].shape[1]
            metrics = self._update(rollout)
            metrics.update(self._rollout_metrics(rollout))
            metrics["timesteps"] = total_timesteps
            for key, value in rollout.get("reward_terms", {}).items():
                metrics[f"reward/{key}"] = float(value)
            self.logger.log(update, metrics)
            last_metrics = metrics
        return PPOTrainResult(timesteps=total_timesteps, updates=max_iterations, metrics=last_metrics)

    def _collect_rollout(self, env, obs):
        self._ensure_option_manager(env.num_envs)
        obs_buf: list[dict] = []
        actions, options, candidates = [], [], []
        log_probs, values, rewards, dones = [], [], [], []
        reward_term_sums: dict[str, float] = {}
        safety_interventions = 0.0
        for _ in range(self.cfg.rollout_length):
            with torch.no_grad():
                output = self._act_with_option_manager(obs)
                action = output.action
                info = {
                    "log_prob": output.log_prob,
                    "value": output.value,
                    "option": output.option,
                    "candidate": output.candidate,
                }
                action, interventions = self.safety_shield(action, obs)
                if interventions.any():
                    evaluated = self.model.forward(
                        obs,
                        actions=action,
                        options=info["option"],
                        candidates=info["candidate"],
                        training=False,
                    )
                    info["log_prob"] = evaluated.log_prob
                    info["value"] = evaluated.value
            next_obs, reward, done, info_env = env.step(action.detach())
            reward = self._as_tensor(reward).detach()
            reward_to_store = self.reward_normalizer(reward) if self.reward_normalizer is not None else reward
            obs_buf.append({key: value.detach() for key, value in obs.items()})
            actions.append(action.detach())
            options.append(info["option"].detach())
            candidates.append(info["candidate"].detach())
            log_probs.append(info["log_prob"].detach())
            values.append(info["value"].detach())
            rewards.append(reward_to_store.detach())
            dones.append(self._as_tensor(done).float().detach())
            for key, value in info_env.get("reward_terms", {}).items():
                reward_term_sums[key] = reward_term_sums.get(key, 0.0) + float(self._as_tensor(value).mean().cpu())
            safety_interventions += float(interventions.float().sum().detach().cpu())
            if "safety_interventions" in info_env:
                safety_interventions += float(self._as_tensor(info_env["safety_interventions"]).sum().cpu())
            done_tensor = self._as_tensor(done).bool()
            if done_tensor.any():
                self.option_manager.reset(torch.nonzero(done_tensor, as_tuple=False).flatten())
            obs = self._to_device(next_obs)
        with torch.no_grad():
            next_value = self.model.forward(obs, training=False).value.detach()
        rollout = {
            "obs": stack_observations(obs_buf),
            "actions": torch.stack(actions),
            "options": torch.stack(options),
            "candidates": torch.stack(candidates),
            "log_probs": torch.stack(log_probs),
            "values": torch.stack(values),
            "rewards": torch.stack(rewards),
            "dones": torch.stack(dones),
            "next_value": next_value,
            "next_obs": obs,
            "reward_terms": reward_term_sums,
            "safety_interventions": safety_interventions,
        }
        rollout["advantages"], rollout["returns"] = self._compute_gae(rollout)
        return rollout

    def _ensure_option_manager(self, num_envs: int):
        if self.option_manager is None or self.option_manager.option.shape[0] != num_envs:
            self.option_manager = OptionManager(
                num_envs,
                OptionManagerConfig(
                    option_interval_steps=self.cfg.option_interval_steps,
                    max_option_age_steps=self.cfg.max_option_age_steps,
                ),
                device=self.device,
            )
            self.option_manager.reset()

    def _act_with_option_manager(self, obs):
        proposal = self.model.forward(obs, training=False)
        resample = self.option_manager.should_resample(proposal.termination_prob)
        option = torch.where(resample, proposal.option, self.option_manager.option)
        candidate = torch.where(resample, proposal.candidate, self.option_manager.candidate)
        output = self.model.forward(obs, options=option, candidates=candidate, training=False)
        self.option_manager.update(output.option, output.candidate, resample)
        return output

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
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
        return advantages, returns

    def _update(self, rollout):
        batch_size = rollout["actions"].shape[0] * rollout["actions"].shape[1]
        minibatch_size = max(1, batch_size // self.cfg.minibatches)
        flat = flatten_hierarchical_rollout(rollout)
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "entropy/local": 0.0,
            "entropy/option": 0.0,
            "entropy/candidate": 0.0,
        }
        updates = 0
        for _ in range(self.cfg.learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb = indices[start : start + minibatch_size]
                obs_mb = {key: value[mb] for key, value in flat["obs"].items()}
                out = self.model.evaluate_actions(
                    obs_mb,
                    flat["actions"][mb],
                    flat["options"][mb],
                    flat["candidates"][mb],
                )
                policy_loss = ppo_clipped_policy_loss(
                    out.log_prob,
                    flat["log_probs"][mb],
                    flat["advantages"][mb],
                    self.cfg.clip_param,
                )
                value_loss = value_mse_loss(out.value, flat["returns"][mb])
                entropy = out.entropy.mean()
                loss = policy_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                metrics["policy_loss"] += float(policy_loss.detach().cpu())
                metrics["value_loss"] += float(value_loss.detach().cpu())
                metrics["entropy"] += float(entropy.detach().cpu())
                metrics["entropy/local"] += float(out.local_entropy.mean().detach().cpu())
                metrics["entropy/option"] += float(out.option_entropy.mean().detach().cpu())
                metrics["entropy/candidate"] += float(out.candidate_entropy.mean().detach().cpu())
                updates += 1
        denom = max(1, updates)
        return {key: value / denom for key, value in metrics.items()}

    def _rollout_metrics(self, rollout):
        metrics: dict[str, float] = {
            "safety/interventions": float(rollout.get("safety_interventions", 0.0)),
            "rollout/mean_reward": float(rollout["rewards"].mean().detach().cpu()),
        }
        counts = torch.bincount(rollout["options"].flatten(), minlength=len(OPTION_NAMES)).float()
        total = counts.sum().clamp_min(1.0)
        for idx, name in enumerate(OPTION_NAMES):
            metrics[f"option_fraction/{name}"] = float((counts[idx] / total).detach().cpu())
        return metrics

    def _to_device(self, obs: dict):
        return {key: self._as_tensor(value).to(self.device) for key, value in obs.items()}

    def _as_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.tensor(value, dtype=torch.float32, device=self.device)

    def save(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "cfg": self.cfg}, output)
        self.logger.log_artifact(
            output,
            artifact_type="model",
            aliases=["latest"],
            enabled_key="log_checkpoints",
            metadata={"checkpoint_path": str(output)},
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
