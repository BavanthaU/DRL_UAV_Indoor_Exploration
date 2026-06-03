from __future__ import annotations

from typing import Any

from .base import VLMEncoder, torch, nn, F


class MobileCLIPEncoder(VLMEncoder):
    """MobileCLIP frontend using an installed MobileCLIP/open_clip-style model."""

    def __init__(self, model_name: str | None = None, *, freeze: bool = True):
        if nn is None:
            raise RuntimeError("MobileCLIPEncoder requires PyTorch.")
        super().__init__()
        self.model_name = self._normalize_model_name(model_name or "MobileCLIP-S1")
        self.freeze = freeze
        self.model, self.preprocess = self._load_model(self.model_name)
        self.embedding_dim = self._infer_embedding_dim()
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    def _load_model(self, model_name: str) -> tuple[Any, Any]:
        try:
            import mobileclip
        except ImportError:
            mobileclip = None
        if mobileclip is not None:
            return mobileclip.create_model_and_transforms(model_name)[0:2]

        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "MobileCLIP backend requires the `mobileclip` package or an "
                "open_clip-compatible MobileCLIP model. Install it in the active "
                "Isaac Lab conda environment with:\n"
                "  python -m pip install open_clip_torch\n"
                "or run `python -m pip install -r requirements.txt`. Use "
                "vlm.backend=mock only for CPU/debug tests."
            ) from exc
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="datacompdr")
        tokenizer = open_clip.get_tokenizer(model_name)
        self._tokenizer = tokenizer
        return model, preprocess

    def _infer_embedding_dim(self) -> int:
        if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
            return int(self.model.text_projection.shape[-1])
        return int(getattr(self.model, "embed_dim", 512))

    def encode_image(self, images):
        images = torch.nan_to_num(images.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        with torch.set_grad_enabled(not self.freeze):
            if hasattr(self.model, "encode_image"):
                embedding = self.model.encode_image(images)
            else:
                raise RuntimeError("Installed MobileCLIP model does not expose encode_image().")
        embedding = torch.nan_to_num(embedding.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return F.normalize(embedding, dim=-1)

    def encode_text(self, prompts: list[str]):
        if not hasattr(self, "_tokenizer"):
            try:
                import open_clip
                self._tokenizer = open_clip.get_tokenizer(self.model_name)
            except ImportError as exc:
                raise RuntimeError("Text encoding requires open_clip tokenizer for this MobileCLIP backend.") from exc
        tokens = self._tokenizer(prompts).to(next(self.model.parameters()).device)
        with torch.set_grad_enabled(not self.freeze):
            embedding = self.model.encode_text(tokens)
        embedding = torch.nan_to_num(embedding.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return F.normalize(embedding, dim=-1)

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        aliases = {
            "mobileclip_s0": "MobileCLIP-S0",
            "mobileclip-s0": "MobileCLIP-S0",
            "mobileclip_s1": "MobileCLIP-S1",
            "mobileclip-s1": "MobileCLIP-S1",
            "mobileclip_s2": "MobileCLIP-S2",
            "mobileclip-s2": "MobileCLIP-S2",
            "mobileclip_b": "MobileCLIP-B",
            "mobileclip-b": "MobileCLIP-B",
        }
        return aliases.get(model_name.lower(), model_name)
