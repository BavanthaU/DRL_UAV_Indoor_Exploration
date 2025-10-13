from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class _PendingDecision:
    obs: torch.Tensor
    action: int
    valid_count: int
    env_idx: int


class FrontierRLManager:
    """Lightweight policy-gradient agent that selects frontier subgoals."""

    def __init__(
        self,
        device: str,
        max_candidates: int = 8,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        batch_size: int = 32,
        training: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.max_candidates = max_candidates
        self.feature_dim = 4  # [gain, distance, score, heading]
        self.extra_dim = 2  # coverage ratio, time step (normalized)
        self.obs_dim = self.max_candidates * (self.feature_dim + 1) + self.extra_dim
        self.policy = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.max_candidates),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.training = training
        self._pending: Dict[int, _PendingDecision] = {}
        self._buffer: List[tuple[torch.Tensor, int, int, float]] = []
        self._reward_gamma = gamma

    def _build_obs(
        self,
        candidate_features: torch.Tensor,
        coverage: float,
        timestep: float,
    ) -> torch.Tensor:
        """Pad candidate features to fixed length and append masks + global features."""
        k = candidate_features.shape[0]
        obs_feat = torch.zeros(self.max_candidates, self.feature_dim, device=self.device)
        mask = torch.zeros(self.max_candidates, device=self.device)
        if k > 0:
            use = min(k, self.max_candidates)
            obs_feat[:use] = candidate_features[:use].to(self.device)
            mask[:use] = 1.0
        obs_flat = obs_feat.flatten()
        mask_flat = mask
        extra = torch.tensor(
            [coverage, timestep],
            dtype=torch.float32,
            device=self.device,
        )
        obs = torch.cat([obs_flat, mask_flat, extra], dim=0)
        return obs

    def select_frontier(
        self,
        env_idx: int,
        candidate_features: torch.Tensor,
        candidate_positions: torch.Tensor,
        coverage: float,
        timestep: float,
    ) -> Optional[int]:
        """Select a frontier index for the given environment."""
        k = candidate_features.shape[0]
        if k == 0:
            return None
        obs = self._build_obs(candidate_features, coverage, timestep)
        logits = self.policy(obs.unsqueeze(0)).squeeze(0)
        mask = torch.full((self.max_candidates,), -1e9, device=self.device)
        mask[:k] = 0.0
        logits = logits + mask
        if self.training:
            if torch.rand(1, device=self.device).item() < self.epsilon:
                action_idx = int(torch.randint(0, k, (1,), device=self.device).item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action_idx = int(dist.sample().item())
        else:
            valid_logits = logits[:k]
            action_idx = int(torch.argmax(valid_logits).item())
        if action_idx >= k:
            action_idx = k - 1

        if self.training:
            self._pending[env_idx] = _PendingDecision(
                obs=obs.detach(),
                action=action_idx,
                valid_count=k,
                env_idx=env_idx,
            )
        return action_idx

    def on_subgoal_complete(
        self,
        env_ids: torch.Tensor,
        rewards: torch.Tensor,
        steps: torch.Tensor,
        done_flags: Optional[torch.Tensor] = None,
    ) -> None:
        """Finalize decisions once their accumulated reward is available."""
        if not self.training:
            self._pending.clear()
            return
        if env_ids is None or env_ids.numel() == 0:
            return
        env_ids_cpu = env_ids.detach().cpu().tolist()
        rewards_cpu = rewards.detach().cpu().tolist()
        counts = steps.detach().cpu().tolist()
        for idx, rew, step_count in zip(env_ids_cpu, rewards_cpu, counts):
            decision = self._pending.pop(idx, None)
            if decision is None:
                continue
            discounted_reward = rew
            if step_count > 0:
                discounted_reward /= max(step_count, 1.0)
            self._buffer.append(
                (
                    decision.obs.clone(),
                    decision.action,
                    decision.valid_count,
                    float(discounted_reward),
                )
            )

    def on_reset(self, env_ids: torch.Tensor) -> None:
        """Clear pending decisions when environments reset."""
        if env_ids is None or env_ids.numel() == 0:
            return
        for idx in env_ids.detach().cpu().tolist():
            self._pending.pop(idx, None)

    def update(self) -> None:
        """Perform a policy-gradient update if enough samples collected."""
        if not self.training:
            return
        if len(self._buffer) < self.batch_size:
            return
        batch = self._buffer
        self._buffer = []

        obs_batch = torch.stack([item[0] for item in batch]).to(self.device)
        actions = torch.tensor([item[1] for item in batch], dtype=torch.long, device=self.device)
        counts = torch.tensor([item[2] for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([item[3] for item in batch], dtype=torch.float32, device=self.device)

        logits = self.policy(obs_batch)
        mask = torch.full((obs_batch.shape[0], self.max_candidates), -1e9, device=self.device)
        for i, count in enumerate(counts):
            mask[i, : int(count.item())] = 0.0
        logits = logits + mask
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        advantages = rewards - rewards.mean()
        if advantages.std() > 1e-6:
            advantages = advantages / (advantages.std() + 1e-6)
        loss = -(advantages * log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
