from __future__ import annotations

from typing import Dict

from gymnasium import spaces
import torch as th
import torch.nn as nn
from torchvision import models

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict


class _MobileNetBackbone(nn.Module):
    """MobileNet v3 small backbone adapted for arbitrary input channels."""

    def __init__(self, in_channels: int, out_dim: int, freeze_backbone: bool = True):
        super().__init__()
        weights = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = weights.features
        first_conv = self.backbone[0][0]
        if first_conv.in_channels != in_channels:
            new_conv = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            with th.no_grad():
                if in_channels == 1:
                    new_conv.weight.copy_(first_conv.weight.sum(dim=1, keepdim=True))
                elif in_channels == 2:
                    new_conv.weight[:, :2].copy_(first_conv.weight[:, :2])
                    new_conv.weight[:, 2:].zero_()
                else:
                    repeat = th.cat([first_conv.weight] * ((in_channels + 2) // 3), dim=1)
                    new_conv.weight.copy_(repeat[:, :in_channels])
                if new_conv.bias is not None:
                    new_conv.bias.copy_(first_conv.bias)
            self.backbone[0][0] = new_conv

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        last_channel = weights.classifier[0].in_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_channel, out_dim),
            nn.SiLU(True),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)


class MobileNetFrozenExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses a pretrained MobileNet backbone for image-like observations
    and small MLPs for vector observations. The backbone can be frozen so RL only tunes the policy head.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        image_out: int = 128,
        vector_out: int = 64,
        features_dim: int = 128,
        freeze_backbone: bool = True,
    ):
        super().__init__(observation_space, features_dim=1)
        assert isinstance(observation_space, spaces.Dict), "Observation space must be Dict."

        self.image_extractors = nn.ModuleDict()
        self.vector_extractors = nn.ModuleDict()

        for key, space in observation_space.spaces.items():
            if is_image_space(space, normalized_image=True):
                channels = space.shape[0] if len(space.shape) == 3 else 1
                self.image_extractors[key] = _MobileNetBackbone(
                    in_channels=channels,
                    out_dim=image_out,
                    freeze_backbone=freeze_backbone,
                )
            else:
                flat_dim = get_flattened_obs_dim(space)
                self.vector_extractors[key] = nn.Sequential(
                    nn.Linear(flat_dim, vector_out),
                    nn.SiLU(True),
                )

        concat_dim = len(self.image_extractors) * image_out + len(self.vector_extractors) * vector_out
        if concat_dim == 0:
            concat_dim = features_dim

        if concat_dim != features_dim:
            self._head = nn.Sequential(
                nn.Linear(concat_dim, features_dim),
                nn.SiLU(True),
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
            # fallback empty obs
            batch = next(iter(observations.values())).shape[0]
            x = th.zeros((batch, self._features_dim), device=self.device)
        return self._head(x)
