from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised in dependency-free shells
    torch = None
    nn = None
    F = None

from .prompt_bank import PromptBank


@dataclass
class VLMEncoderOutput:
    image_embedding: Any
    text_embeddings: Any
    patch_tokens: Any | None = None


class VLMEncoder(nn.Module if nn is not None else object):
    """Base class for image/text encoders used by the exploration policy."""

    embedding_dim: int

    def encode_image(self, images):
        raise NotImplementedError

    def encode_text(self, prompts: list[str]):
        raise NotImplementedError

    def forward(self, images, prompts: list[str]) -> VLMEncoderOutput:
        return VLMEncoderOutput(
            image_embedding=self.encode_image(images),
            text_embeddings=self.encode_text(prompts),
            patch_tokens=None,
        )


class MockVLMEncoder(VLMEncoder):
    """Deterministic lightweight encoder for tests and CPU smoke runs only."""

    def __init__(self, embedding_dim: int = 64):
        if nn is None:
            raise RuntimeError("MockVLMEncoder requires PyTorch.")
        super().__init__()
        self.embedding_dim = embedding_dim
        self.image_proj = nn.LazyLinear(embedding_dim)
        self.register_buffer("_device_ref", torch.empty(0))

    def encode_image(self, images):
        if images.ndim != 4:
            raise ValueError(f"Expected images [B,C,H,W], got shape {tuple(images.shape)}")
        flat = images.float().flatten(start_dim=1) / 255.0 if images.max() > 2 else images.float().flatten(start_dim=1)
        return F.normalize(self.image_proj(flat), dim=-1)

    def encode_text(self, prompts: list[str]):
        rows = []
        for prompt in prompts:
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            values = []
            while len(values) < self.embedding_dim:
                values.extend((byte / 255.0) * 2.0 - 1.0 for byte in digest)
                digest = hashlib.sha256(digest).digest()
            rows.append(values[: self.embedding_dim])
        return F.normalize(torch.tensor(rows, dtype=torch.float32, device=self._device_ref.device), dim=-1)


def build_vlm_encoder(
    backend: str,
    *,
    model_name: str | None = None,
    embedding_dim: int = 64,
    freeze: bool = True,
) -> VLMEncoder:
    if backend == "mock":
        return MockVLMEncoder(embedding_dim=embedding_dim)
    if backend == "mobileclip":
        from .mobileclip_encoder import MobileCLIPEncoder

        return MobileCLIPEncoder(model_name=model_name, freeze=freeze)
    if backend == "siglip":
        from .siglip_encoder import SigLIPEncoder

        return SigLIPEncoder(model_name=model_name, freeze=freeze)
    raise ValueError(f"Unsupported VLM backend '{backend}'. Use mobileclip, siglip, or mock.")


def prompt_similarity(image_embedding, text_embeddings):
    image_embedding = torch.nan_to_num(image_embedding, nan=0.0, posinf=0.0, neginf=0.0)
    text_embeddings = torch.nan_to_num(text_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    image_embedding = F.normalize(image_embedding, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    return torch.nan_to_num(image_embedding @ text_embeddings.T, nan=0.0, posinf=0.0, neginf=0.0)


def prompt_uncertainty(similarities):
    similarities = torch.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.softmax(similarities, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    entropy = entropy / max(1, similarities.shape[-1])
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.shape[-1] > 1 else 0.0)
    return torch.clamp(entropy + (1.0 - margin), 0.0, 1.0)
