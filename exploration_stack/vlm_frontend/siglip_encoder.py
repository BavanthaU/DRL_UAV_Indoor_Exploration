from __future__ import annotations

from .base import VLMEncoder, torch, nn, F


class SigLIPEncoder(VLMEncoder):
    """SigLIP encoder through Hugging Face Transformers."""

    def __init__(self, model_name: str | None = None, *, freeze: bool = True):
        if nn is None:
            raise RuntimeError("SigLIPEncoder requires PyTorch.")
        super().__init__()
        self.model_name = model_name or "google/siglip-base-patch16-224"
        self.freeze = freeze
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "SigLIP backend requires transformers. Install transformers and "
                "ensure the selected model is available locally or downloadable."
            ) from exc
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.embedding_dim = int(getattr(self.model.config, "projection_dim", getattr(self.model.config, "hidden_size", 768)))
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    def encode_image(self, images):
        with torch.set_grad_enabled(not self.freeze):
            if hasattr(self.model, "get_image_features"):
                embedding = self.model.get_image_features(pixel_values=images.float())
            else:
                embedding = self.model.vision_model(pixel_values=images.float()).pooler_output
        return F.normalize(embedding.float(), dim=-1)

    def encode_text(self, prompts: list[str]):
        tokens = self.processor(text=prompts, padding=True, return_tensors="pt")
        tokens = {key: value.to(next(self.model.parameters()).device) for key, value in tokens.items()}
        with torch.set_grad_enabled(not self.freeze):
            if hasattr(self.model, "get_text_features"):
                embedding = self.model.get_text_features(**tokens)
            else:
                embedding = self.model.text_model(**tokens).pooler_output
        return F.normalize(embedding.float(), dim=-1)

