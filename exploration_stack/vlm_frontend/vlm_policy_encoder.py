from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .affordance_maps import affordance_summary_from_prompt_similarity
from .base import build_vlm_encoder, prompt_similarity, prompt_uncertainty, torch, nn
from .map_renderer import MapRenderConfig, MapRenderer
from .prompt_bank import PromptBank


@dataclass
class VLMPolicyEncoderConfig:
    vlm_backend: str = "mobileclip"
    model_name: str | None = None
    image_mode: str = "camera_plus_map"
    freeze_vlm_encoder: bool = True
    train_vlm_lora: bool = False
    latent_dim: int = 256
    memory_type: str = "gru"
    memory_steps: int = 8
    image_size: int = 224
    use_depth_line: bool = True
    use_semantic_line: bool = True
    use_map_crop: bool = True
    use_frontier_mask: bool = True
    depth_line_dim: int = 64
    semantic_line_dim: int = 64
    subgoal_feature_dim: int = 5
    lora_vlm_encoder: dict[str, Any] | None = None


@dataclass
class VLMPolicyEncoderOutput:
    z_actor: "torch.Tensor"
    z_critic: "torch.Tensor"
    aux: dict[str, "torch.Tensor"]
    memory_state: "torch.Tensor | None" = None


class VLMPolicyEncoder(nn.Module if nn is not None else object):
    """VLM-driven observation encoder for actor/critic PPO heads."""

    def __init__(self, cfg: VLMPolicyEncoderConfig, prompt_bank: PromptBank | None = None):
        if nn is None:
            raise RuntimeError("VLMPolicyEncoder requires PyTorch.")
        super().__init__()
        self.cfg = cfg
        self.prompt_bank = prompt_bank or PromptBank()
        self.vlm = build_vlm_encoder(
            cfg.vlm_backend,
            model_name=cfg.model_name,
            embedding_dim=max(64, min(cfg.latent_dim, 256)),
            freeze=cfg.freeze_vlm_encoder,
        )
        self.map_renderer = MapRenderer(MapRenderConfig(image_size=cfg.image_size))
        vlm_dim = self.vlm.embedding_dim
        image_embedding_count = 0
        if cfg.image_mode in ("camera_only", "camera_plus_map"):
            image_embedding_count += 1
        if cfg.image_mode in ("map_only", "camera_plus_map"):
            image_embedding_count += 1
        prompt_dim = len(self.prompt_bank) * max(1, image_embedding_count)
        numeric_dim = 0
        numeric_dim += cfg.depth_line_dim if cfg.use_depth_line else 0
        numeric_dim += cfg.semantic_line_dim if cfg.use_semantic_line else 0
        numeric_dim += cfg.subgoal_feature_dim
        fused_dim = vlm_dim * max(1, image_embedding_count) + prompt_dim + numeric_dim
        self.input_proj = nn.Sequential(nn.Linear(fused_dim, cfg.latent_dim), nn.LayerNorm(cfg.latent_dim), nn.SiLU())
        if cfg.memory_type == "gru":
            self.memory = nn.GRU(input_size=cfg.latent_dim, hidden_size=cfg.latent_dim, batch_first=True)
        elif cfg.memory_type in ("none", None):
            self.memory = None
        else:
            raise ValueError(f"Unsupported memory_type '{cfg.memory_type}'")
        self.actor_proj = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim), nn.SiLU())
        self.critic_proj = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim), nn.SiLU())
        self.coverage_head = nn.Linear(cfg.latent_dim, 1)
        self.loop_head = nn.Linear(cfg.latent_dim, 1)

    def forward(self, obs: dict[str, "torch.Tensor"], memory_state=None) -> VLMPolicyEncoderOutput:
        embeddings = []
        similarities = []
        text_embeddings = None
        camera_embedding = None
        map_embedding = None
        if self.cfg.image_mode in ("camera_only", "camera_plus_map"):
            camera = self._require(obs, "camera_rgb")
            text_embeddings = self.vlm.encode_text(self.prompt_bank.prompts).to(camera.device)
            camera_embedding = self.vlm.encode_image(camera)
            embeddings.append(camera_embedding)
            similarities.append(prompt_similarity(camera_embedding, text_embeddings))
        if self.cfg.image_mode in ("map_only", "camera_plus_map"):
            occupancy = self._require(obs, "map_crop")
            rendered_map = self.map_renderer.render(
                occupancy,
                frontier_mask=obs.get("frontier_mask"),
                trajectory_mask=obs.get("trajectory_mask"),
            )
            if text_embeddings is None:
                text_embeddings = self.vlm.encode_text(self.prompt_bank.prompts).to(rendered_map.device)
            map_embedding = self.vlm.encode_image(rendered_map)
            embeddings.append(map_embedding)
            similarities.append(prompt_similarity(map_embedding, text_embeddings))
        fused_parts = embeddings + similarities
        batch = fused_parts[0].shape[0]
        if self.cfg.use_depth_line:
            fused_parts.append(self._pad_or_trim(obs.get("depth_line"), self.cfg.depth_line_dim, batch, fused_parts[0].device))
        if self.cfg.use_semantic_line:
            fused_parts.append(self._pad_or_trim(obs.get("semantic_line"), self.cfg.semantic_line_dim, batch, fused_parts[0].device))
        fused_parts.append(self._pad_or_trim(obs.get("subgoal_features"), self.cfg.subgoal_feature_dim, batch, fused_parts[0].device))
        fused = torch.cat(fused_parts, dim=-1)
        latent = self.input_proj(fused)
        if self.memory is not None:
            latent_seq = latent.unsqueeze(1)
            latent_out, memory_state = self.memory(latent_seq, memory_state)
            latent = latent_out[:, -1]
        prompt_sim = torch.cat(similarities, dim=-1)
        affordance = affordance_summary_from_prompt_similarity(similarities[0], self.prompt_bank)
        aux = {
            "prompt_similarity": prompt_sim,
            "doorway_likelihood": affordance.doorway_likelihood,
            "corridor_likelihood": affordance.corridor_likelihood,
            "open_space_likelihood": affordance.open_space_likelihood,
            "dead_end_likelihood": affordance.dead_end_likelihood,
            "collision_risk": affordance.collision_risk,
            "frontier_value": affordance.frontier_value,
            "frontier_utility": affordance.frontier_value,
            "revisit_likelihood": affordance.revisit_likelihood,
            "coverage_delta_prediction": self.coverage_head(latent).squeeze(-1),
            "loop_probability": torch.sigmoid(self.loop_head(latent).squeeze(-1)),
            "uncertainty": prompt_uncertainty(similarities[0]),
        }
        if camera_embedding is not None:
            aux["camera_embedding"] = camera_embedding
        if map_embedding is not None:
            aux["map_embedding"] = map_embedding
        return VLMPolicyEncoderOutput(
            z_actor=self.actor_proj(latent),
            z_critic=self.critic_proj(latent),
            aux=aux,
            memory_state=memory_state,
        )

    @staticmethod
    def _require(obs: dict, key: str):
        if key not in obs:
            raise KeyError(f"Observation missing required key '{key}' for VLMPolicyEncoder")
        return obs[key]

    @staticmethod
    def _pad_or_trim(value, target_dim: int, batch: int, device):
        if value is None:
            return torch.zeros(batch, target_dim, device=device)
        value = value.float()
        if value.ndim > 2:
            value = value.flatten(start_dim=1)
        if value.shape[-1] == target_dim:
            return value
        if value.shape[-1] > target_dim:
            return value[:, :target_dim]
        pad = torch.zeros(value.shape[0], target_dim - value.shape[-1], device=value.device, dtype=value.dtype)
        return torch.cat([value, pad], dim=-1)
