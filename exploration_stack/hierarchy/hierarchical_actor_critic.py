from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical, Normal
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    Categorical = None
    Normal = None

from exploration_stack.planning.planner_dropout import PlannerDropoutConfig, PlannerFeatureDropout
from exploration_stack.vlm_frontend import VLMPolicyEncoder, VLMPolicyEncoderConfig

from .candidate_builder import CandidateBuilder, CandidateBuilderConfig
from .candidate_encoder import CandidateEncoderConfig, CandidateSetEncoder
from .option_masking import build_option_mask
from .option_policy import OptionPolicyConfig, SemanticOptionPolicy
from .option_types import NUM_OPTIONS


@dataclass
class HierarchicalActorCriticConfig:
    encoder: VLMPolicyEncoderConfig = field(default_factory=VLMPolicyEncoderConfig)
    action_dim: int = 3
    log_std_init: float = 0.0
    max_candidates: int = 16
    candidate_feature_dim: int = 18
    candidate_hidden_dim: int = 128
    candidate_heads: int = 4
    candidate_layers: int = 1
    option_hidden_dim: int = 256
    option_embedding_dim: int = 32
    planner_feature_mode: str = "features_only"
    use_privileged_map_for_training: bool = False
    use_planner_dropout: bool = True
    planner_dropout_prob: float = 0.2
    astar_feature_dropout_prob: float = 0.2
    frontier_candidate_dropout_prob: float = 0.1


@dataclass
class HierarchicalActorCriticOutput:
    action: "torch.Tensor"
    log_prob: "torch.Tensor"
    local_log_prob: "torch.Tensor"
    option_log_prob: "torch.Tensor"
    candidate_log_prob: "torch.Tensor"
    entropy: "torch.Tensor"
    local_entropy: "torch.Tensor"
    option_entropy: "torch.Tensor"
    candidate_entropy: "torch.Tensor"
    value: "torch.Tensor"
    aux: dict[str, "torch.Tensor"]
    mean: "torch.Tensor"
    std: "torch.Tensor"
    option: "torch.Tensor"
    candidate: "torch.Tensor"
    option_logits: "torch.Tensor"
    candidate_logits: "torch.Tensor"
    candidate_mask: "torch.Tensor"
    termination_prob: "torch.Tensor"
    option_confidence: "torch.Tensor"


class HierarchicalActorCritic(nn.Module if nn is not None else object):
    """Learned semantic-option policy plus learned local continuous control."""

    def __init__(self, cfg: HierarchicalActorCriticConfig):
        if nn is None:
            raise RuntimeError("HierarchicalActorCritic requires PyTorch.")
        super().__init__()
        self.cfg = cfg
        self.encoder = VLMPolicyEncoder(cfg.encoder)
        self.candidate_builder = CandidateBuilder(
            CandidateBuilderConfig(
                max_candidates=cfg.max_candidates,
                feature_dim=cfg.candidate_feature_dim,
                planner_feature_mode=cfg.planner_feature_mode,
                use_privileged_map_for_training=cfg.use_privileged_map_for_training,
            )
        )
        self.planner_dropout = PlannerFeatureDropout(
            PlannerDropoutConfig(
                planner_dropout_prob=cfg.planner_dropout_prob,
                astar_feature_dropout_prob=cfg.astar_feature_dropout_prob,
                frontier_candidate_dropout_prob=cfg.frontier_candidate_dropout_prob,
            )
        )
        self.candidate_encoder = CandidateSetEncoder(
            CandidateEncoderConfig(
                feature_dim=cfg.candidate_feature_dim,
                hidden_dim=cfg.candidate_hidden_dim,
                heads=cfg.candidate_heads,
                layers=cfg.candidate_layers,
            )
        )
        self.option_policy = SemanticOptionPolicy(
            OptionPolicyConfig(
                context_dim=cfg.encoder.latent_dim,
                candidate_dim=cfg.candidate_hidden_dim,
                hidden_dim=cfg.option_hidden_dim,
                num_options=NUM_OPTIONS,
            )
        )
        self.option_embedding = nn.Embedding(NUM_OPTIONS, cfg.option_embedding_dim)
        local_dim = cfg.encoder.latent_dim + cfg.candidate_hidden_dim + cfg.option_embedding_dim
        self.local_actor = nn.Sequential(
            nn.Linear(local_dim, cfg.option_hidden_dim),
            nn.LayerNorm(cfg.option_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.option_hidden_dim, cfg.action_dim),
        )
        self.actor_log_std = nn.Parameter(torch.full((cfg.action_dim,), float(cfg.log_std_init)))
        self.critic = nn.Sequential(
            nn.Linear(cfg.encoder.latent_dim + cfg.candidate_hidden_dim, cfg.option_hidden_dim),
            nn.LayerNorm(cfg.option_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.option_hidden_dim, 1),
        )

    def forward(
        self,
        obs: dict[str, "torch.Tensor"],
        actions: "torch.Tensor | None" = None,
        options: "torch.Tensor | None" = None,
        candidates: "torch.Tensor | None" = None,
        *,
        deterministic: bool = False,
        training: bool = True,
    ) -> HierarchicalActorCriticOutput:
        enc = self.encoder(obs)
        candidate_features, candidate_mask = self.candidate_builder.build(obs)
        if self.cfg.use_planner_dropout and self.training and training:
            candidate_features, candidate_mask = self.planner_dropout(candidate_features, candidate_mask, training=True)
        candidate_tokens, candidate_pooled = self.candidate_encoder(candidate_features, candidate_mask)
        option_logits, candidate_logits, termination_prob, option_confidence = self.option_policy(
            enc.z_actor,
            candidate_tokens,
            candidate_pooled,
            candidate_mask,
        )
        option_mask = build_option_mask(candidate_features, candidate_mask)
        option_logits = torch.nan_to_num(option_logits, nan=0.0, posinf=0.0, neginf=0.0)
        option_logits = option_logits.masked_fill(~option_mask, -1.0e9)
        candidate_logits = torch.nan_to_num(candidate_logits, nan=-1.0e9, posinf=0.0, neginf=-1.0e9)
        option_dist = Categorical(logits=option_logits)
        candidate_dist = Categorical(logits=candidate_logits)
        if options is None:
            option = option_logits.argmax(dim=-1) if deterministic else option_dist.sample()
        else:
            option = options.long()
        if candidates is None:
            candidate = candidate_logits.argmax(dim=-1) if deterministic else candidate_dist.sample()
        else:
            candidate = candidates.long()
        selected_candidate = self._gather_candidate(candidate_tokens, candidate)
        option_embed = self.option_embedding(option)
        local_context = torch.cat([enc.z_actor, selected_candidate, option_embed], dim=-1)
        local_context = torch.nan_to_num(local_context, nan=0.0, posinf=0.0, neginf=0.0)
        mean = torch.nan_to_num(torch.tanh(self.local_actor(local_context)), nan=0.0, posinf=1.0, neginf=-1.0)
        log_std = self.actor_log_std.clamp(-5.0, 2.0)
        std = log_std.exp().expand_as(mean)
        local_dist = Normal(mean, std)
        if actions is None:
            action = mean if deterministic else local_dist.rsample()
        else:
            action = actions
        local_log_prob = local_dist.log_prob(action).sum(dim=-1)
        option_log_prob = option_dist.log_prob(option)
        candidate_log_prob = candidate_dist.log_prob(candidate)
        local_entropy = local_dist.entropy().sum(dim=-1)
        option_entropy = option_dist.entropy()
        candidate_entropy = candidate_dist.entropy()
        value = self.critic(torch.cat([enc.z_critic, candidate_pooled], dim=-1)).squeeze(-1)
        aux = dict(enc.aux)
        aux.update(
            {
                "z_actor": enc.z_actor,
                "z_critic": enc.z_critic,
                "candidate_features": candidate_features,
                "candidate_mask": candidate_mask,
                "option_mask": option_mask,
                "option_logits": option_logits,
                "candidate_logits": candidate_logits,
                "termination_prob": termination_prob,
                "option_confidence": option_confidence,
                "selected_option": option,
                "selected_candidate": candidate,
            }
        )
        return HierarchicalActorCriticOutput(
            action=action.clamp(-1.0, 1.0),
            log_prob=local_log_prob + option_log_prob + candidate_log_prob,
            local_log_prob=local_log_prob,
            option_log_prob=option_log_prob,
            candidate_log_prob=candidate_log_prob,
            entropy=local_entropy + option_entropy + candidate_entropy,
            local_entropy=local_entropy,
            option_entropy=option_entropy,
            candidate_entropy=candidate_entropy,
            value=value,
            aux=aux,
            mean=mean,
            std=std,
            option=option,
            candidate=candidate,
            option_logits=option_logits,
            candidate_logits=candidate_logits,
            candidate_mask=candidate_mask,
            termination_prob=termination_prob,
            option_confidence=option_confidence,
        )

    def act(self, obs: dict[str, "torch.Tensor"], *, deterministic: bool = False) -> tuple["torch.Tensor", dict[str, Any]]:
        with torch.no_grad():
            output = self.forward(obs, deterministic=deterministic, training=False)
        action = output.mean if deterministic else output.action
        return action.clamp(-1.0, 1.0), {
            "log_prob": output.log_prob,
            "value": output.value,
            "aux": output.aux,
            "option": output.option,
            "candidate": output.candidate,
            "option_log_prob": output.option_log_prob,
            "candidate_log_prob": output.candidate_log_prob,
            "local_log_prob": output.local_log_prob,
        }

    def evaluate_actions(
        self,
        obs: dict[str, "torch.Tensor"],
        actions: "torch.Tensor",
        options: "torch.Tensor",
        candidates: "torch.Tensor",
    ) -> HierarchicalActorCriticOutput:
        return self.forward(obs, actions=actions, options=options, candidates=candidates, training=True)

    @staticmethod
    def _gather_candidate(candidate_tokens, candidate):
        gather_index = candidate.view(-1, 1, 1).expand(-1, 1, candidate_tokens.shape[-1])
        return torch.gather(candidate_tokens, dim=1, index=gather_index).squeeze(1)
