from __future__ import annotations

from gymnasium import spaces
import torch as th
import torch.nn as nn

try:
    import timm
except Exception as exc:  # pragma: no cover - handled at runtime
    raise ImportError("timm is required for ConvNeXtV2 backbones.") from exc

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict


class _ConvNeXtV2NanoBackbone(nn.Module):
    """ConvNeXtV2-Nano backbone adapted for arbitrary input channels using timm."""

    def __init__(self, in_channels: int, out_dim: int, freeze_backbone: bool = True):
        super().__init__()
        try:
            self.backbone = timm.create_model(
                "convnextv2_nano.fcmae_ft_in22k_in1k",
                pretrained=True,
                num_classes=0,
                global_pool="avg",
                in_chans=in_channels,
            )
        except Exception as exc:
            raise ImportError(
                "Failed to load ConvNeXtV2 Nano from timm. Ensure timm is installed and includes this checkpoint."
            ) from exc

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self._out_channels = getattr(self.backbone, "num_features", None)
        if self._out_channels is None:
            raise RuntimeError("ConvNeXtV2 backbone did not expose num_features attribute.")

        self.head = nn.Sequential(
            nn.LayerNorm(self._out_channels),
            nn.Linear(self._out_channels, out_dim),
            nn.GELU(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        feats = self.backbone(x)  # (B, num_features)
        return self.head(feats)


class ConvNeXtV2NanoFrozenExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses a pretrained ConvNeXtV2 Nano backbone (via timm) for image observations
    and lightweight MLPs for vector observations. By default the backbone weights are frozen so RL only
    tunes the projection heads.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        image_out: int = 256,
        vector_out: int = 64,
        features_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__(observation_space, features_dim=1)
        assert isinstance(observation_space, spaces.Dict), "Observation space must be Dict."

        self.image_extractors = nn.ModuleDict()
        self.vector_extractors = nn.ModuleDict()

        for key, space in observation_space.spaces.items():
            if is_image_space(space, normalized_image=True):
                channels = space.shape[0] if len(space.shape) == 3 else 1
                self.image_extractors[key] = _ConvNeXtV2NanoBackbone(
                    in_channels=channels,
                    out_dim=image_out,
                    freeze_backbone=freeze_backbone,
                )
            else:
                flat_dim = get_flattened_obs_dim(space)
                self.vector_extractors[key] = nn.Sequential(
                    nn.Linear(flat_dim, vector_out),
                    nn.LayerNorm(vector_out),
                    nn.GELU(),
                )

        concat_dim = len(self.image_extractors) * image_out + len(self.vector_extractors) * vector_out
        if concat_dim == 0:
            concat_dim = features_dim

        if concat_dim != features_dim:
            self._head = nn.Sequential(
                nn.Linear(concat_dim, features_dim),
                nn.LayerNorm(features_dim),
                nn.GELU(),
            )
            self._features_dim = features_dim
        else:
            self._head = nn.Identity()
            self._features_dim = concat_dim

    def forward(self, observations: TensorDict) -> th.Tensor:
        feats = []
        for key, extractor in self.image_extractors.items():
            x = observations[key]
            if x.dtype == th.uint8:
                x = x.float() / 255.0
            feats.append(extractor(x))
        for key, extractor in self.vector_extractors.items():
            x = observations[key]
            feats.append(extractor(x.view(x.size(0), -1)))
        if feats:
            x = th.cat(feats, dim=1)
        else:
            batch = next(iter(observations.values())).shape[0]
            x = th.zeros((batch, self._features_dim), device=self.device)
        return self._head(x)
