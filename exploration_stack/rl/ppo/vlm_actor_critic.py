from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
    from torch.distributions import Normal
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    Normal = None

from exploration_stack.vlm_frontend import VLMPolicyEncoder, VLMPolicyEncoderConfig


@dataclass
class ActorCriticOutput:
    action: "torch.Tensor"
    log_prob: "torch.Tensor"
    entropy: "torch.Tensor"
    value: "torch.Tensor"
    aux: dict[str, "torch.Tensor"]
    mean: "torch.Tensor"
    std: "torch.Tensor"


class VLMActorCritic(nn.Module if nn is not None else object):
    """Gaussian actor and scalar critic on top of VLMPolicyEncoder latents."""

    def __init__(self, encoder_cfg: VLMPolicyEncoderConfig, *, action_dim: int = 3, log_std_init: float = 0.0):
        if nn is None:
            raise RuntimeError("VLMActorCritic requires PyTorch.")
        super().__init__()
        self.encoder = VLMPolicyEncoder(encoder_cfg)
        self.action_dim = action_dim
        latent_dim = encoder_cfg.latent_dim
        self.actor_mean = nn.Linear(latent_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.critic = nn.Linear(latent_dim, 1)

    def forward(self, obs: dict[str, "torch.Tensor"], actions: "torch.Tensor | None" = None) -> ActorCriticOutput:
        enc = self.encoder(obs)
        mean = torch.tanh(self.actor_mean(enc.z_actor))
        log_std = self.actor_log_std.clamp(-5.0, 2.0)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if actions is None:
            action = dist.rsample()
        else:
            action = actions
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(enc.z_critic).squeeze(-1)
        aux = dict(enc.aux)
        aux["z_actor"] = enc.z_actor
        aux["z_critic"] = enc.z_critic
        return ActorCriticOutput(
            action=action.clamp(-1.0, 1.0),
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            aux=aux,
            mean=mean,
            std=std,
        )

    def act(self, obs: dict[str, "torch.Tensor"], *, deterministic: bool = False) -> tuple["torch.Tensor", dict[str, Any]]:
        with torch.no_grad():
            output = self.forward(obs)
        action = output.mean if deterministic else output.action
        return action.clamp(-1.0, 1.0), {
            "log_prob": output.log_prob,
            "value": output.value,
            "aux": output.aux,
        }

    def evaluate_actions(self, obs: dict[str, "torch.Tensor"], actions: "torch.Tensor") -> ActorCriticOutput:
        return self.forward(obs, actions=actions)
