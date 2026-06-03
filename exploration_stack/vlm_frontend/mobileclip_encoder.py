from __future__ import annotations

from typing import Any

from .base import VLMEncoder, torch, nn, F


class MobileCLIPEncoder(VLMEncoder):
    """MobileCLIP frontend using an installed MobileCLIP/open_clip-style model."""

    def __init__(self, model_name: str | None = None, *, freeze: bool = True):
        if nn is None:
            raise RuntimeError("MobileCLIPEncoder requires PyTorch.")
        super().__init__()
        self.model_name = model_name or "MobileCLIP-S1"
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
                "open_clip-compatible MobileCLIP model. Install the backend or "
                "use vlm_backend=mock only for debug tests."
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
        with torch.set_grad_enabled(not self.freeze):
            if hasattr(self.model, "encode_image"):
                embedding = self.model.encode_image(images.float())
            else:
                raise RuntimeError("Installed MobileCLIP model does not expose encode_image().")
        return F.normalize(embedding.float(), dim=-1)

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
        return F.normalize(embedding.float(), dim=-1)

