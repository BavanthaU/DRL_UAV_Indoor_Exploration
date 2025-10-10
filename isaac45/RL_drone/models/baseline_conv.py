# RL_drone/models/baseline_extractor.py
from typing import List, Tuple, Optional, Dict
from gymnasium import spaces
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict


# -------- utils --------
def _prep_image(x: th.Tensor) -> th.Tensor:
    """Accept (N,C,H,W) or (N,H,W); uint8 or float; returns float in [0..1], (N,C,H,W)."""
    if x.ndim == 3:  # (N,H,W) -> (N,1,H,W)
        x = x.unsqueeze(1)
    x = x.float()
    if x.dtype == th.uint8 or (th.isfinite(x).any() and x.max() > 1.5):
        x = x / 255.0
    return x


# -------- building blocks for 2D CNN --------
class SE(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        mid = max(1, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, mid, 1), nn.SiLU(True),
            nn.Conv2d(mid, c, 1), nn.Sigmoid(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x * self.fc(self.pool(x))


class DWSeparableConv(nn.Module):
    """Depthwise conv + pointwise conv with GroupNorm + SiLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, dilation: int = 1, groups: int = 16):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride=stride, padding=pad, dilation=dilation, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act = nn.SiLU(True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.gn(x)
        return self.act(x)


class ResDWBlock(nn.Module):
    """
    Residual block with depthwise-separable convs.
    Optionally includes Squeeze-Excite and a projection for channel/stride changes.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dilation: int = 1,
        use_se: bool = True,
        groups: int = 16,
    ):
        super().__init__()
        self.conv1 = DWSeparableConv(in_ch, out_ch, kernel=3, stride=stride, dilation=dilation, groups=groups)
        self.conv2 = DWSeparableConv(out_ch, out_ch, kernel=3, stride=1, dilation=dilation, groups=groups)
        self.se = SE(out_ch) if use_se else nn.Identity()
        self.proj = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            )
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )
        self.act = nn.SiLU(True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        identity = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return self.act(x + identity)


class ASPPLite(nn.Module):
    """Light ASPP: parallel dilated 3x3 convs + 1x1 + global avg pooling."""
    def __init__(self, c: int, out_c: int, dilations: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(c, out_c, 1, bias=False),
            *[nn.Conv2d(c, out_c, 3, padding=d, dilation=d, bias=False) for d in dilations],
        ])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, out_c, 1, bias=False),
        )
        self.norm = nn.GroupNorm(num_groups=min(16, out_c * (len(dilations) + 2)), num_channels=out_c * (len(dilations) + 2))
        self.act = nn.SiLU(True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        feats = [b(x) for b in self.branches]
        g = self.gap(x)
        g = th.nn.functional.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        out = th.cat(feats + [g], dim=1)
        out = self.norm(out)
        return self.act(out)


# -------- deeper ego-map CNN --------
class DeeperEgoMapCNN(nn.Module):
    """
    Deeper encoder for ego maps.
      stem -> stages[*] (residual DW blocks, first block downsamples) -> ASPP-lite (optional) -> GAP -> MLP
    Configurable depth/width; still small and fast.
    """
    def __init__(
        self,
        in_ch: int,
        out_dim: int = 128,
        base_channels: int = 64,
        stages: Tuple[int, ...] = (2, 2, 2),     # blocks per stage
        use_se: bool = True,
        use_aspp: bool = True,
        aspp_out: Optional[int] = None,          # None -> equals last stage channels
        groups: int = 16,
    ):
        super().__init__()
        c1 = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, c1), num_channels=c1),
            nn.SiLU(True),
        )

        chans = [c1, c1 * 2, c1 * 4] if len(stages) >= 3 else [c1] * len(stages)
        self.stages = nn.ModuleList()
        in_c = c1
        for i, n_blocks in enumerate(stages):
            out_c = chans[i] if i < len(chans) else chans[-1]
            blocks = []
            for b in range(n_blocks):
                stride = 2 if (b == 0 and i > 0) else 1      # downsample at stage entry (except stage 0)
                dilation = 1 if i < 2 else 2                 # a touch of dilation in deeper stage
                blocks.append(ResDWBlock(in_c, out_c, stride=stride, dilation=dilation, use_se=use_se, groups=groups))
                in_c = out_c
            self.stages.append(nn.Sequential(*blocks))

        last_c = in_c
        self.aspp = ASPPLite(last_c, aspp_out or last_c) if use_aspp else None
        post_c = (aspp_out or last_c) * (len((1, 2, 4)) + 2) if use_aspp else last_c

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(post_c, out_dim),
            nn.SiLU(True),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = _prep_image(x)
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        if self.aspp is not None:
            x = self.aspp(x)
        x = self.pool(x)
        return self.head(x)


# -------- simple 1D encoder for line features --------
class SimpleLine1DEncoder(nn.Module):
    def __init__(self, in_len: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.SiLU(True),
            nn.Conv1d(32, 32, kernel_size=5, dilation=2, padding=4), nn.SiLU(True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim), nn.SiLU(True),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        if x.ndim >= 3:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x.unsqueeze(1))


# -------- Combined extractor (2D ego map + 1D lines) --------
class BaselineEgoMapAndLinesExtractor(BaseFeaturesExtractor):
    """
    Image-like keys (ego maps) -> DeeperEgoMapCNN
    Vector/line keys (semantics/depth lines) -> SimpleLine1DEncoder
    Concatenate -> optional Linear -> features_dim
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        image_out: int = 128,
        line_out: int = 64,
        features_dim: int = 128,
        normalized_image: bool = True,   # kept for is_image_space behavior parity
        # DeeperEgoMapCNN hyperparams:
        ego_base_channels: int = 64,
        ego_stages: Tuple[int, ...] = (2, 2, 2),
        ego_use_se: bool = True,
        ego_use_aspp: bool = True,
        ego_aspp_out: Optional[int] = None,
        ego_groups: int = 16,
    ):
        super().__init__(observation_space, features_dim=1)
        assert isinstance(observation_space, spaces.Dict), "Expect Dict observation space"

        self.img_keys, self.vec_keys = [], []
        for k, sp in observation_space.spaces.items():
            if is_image_space(sp, normalized_image=normalized_image):
                self.img_keys.append(k)
            else:
                self.vec_keys.append(k)

        # Per-key encoders
        self.image_extractors = nn.ModuleDict()
        for k in self.img_keys:
            shape = observation_space.spaces[k].shape
            in_ch = shape[0] if len(shape) == 3 else 1
            self.image_extractors[k] = DeeperEgoMapCNN(
                in_ch=in_ch,
                out_dim=image_out,
                base_channels=ego_base_channels,
                stages=ego_stages,
                use_se=ego_use_se,
                use_aspp=ego_use_aspp,
                aspp_out=ego_aspp_out,
                groups=ego_groups,
            )

        self.line_extractors = nn.ModuleDict()
        for k in self.vec_keys:
            L = get_flattened_obs_dim(observation_space.spaces[k])
            self.line_extractors[k] = SimpleLine1DEncoder(in_len=L, out_dim=line_out)

        # Feature dims
        concat_dim = len(self.img_keys) * image_out + len(self.vec_keys) * line_out
        if concat_dim == 0:
            concat_dim = features_dim

        if features_dim != concat_dim:
            self.post = nn.Sequential(nn.Linear(concat_dim, features_dim), nn.SiLU(True))
            self._features_dim = features_dim
        else:
            self.post = nn.Identity()
            self._features_dim = concat_dim

    def forward(self, obs: TensorDict) -> th.Tensor:
        img_feats = [enc(obs[k]) for k, enc in self.image_extractors.items()]
        vec_feats = [enc(obs[k]) for k, enc in self.line_extractors.items()]

        if img_feats or vec_feats:
            x = th.cat(img_feats + vec_feats, dim=1)
        else:
            # fallback (unlikely)
            any_tensor = next(iter(obs.values()))
            x = th.zeros((any_tensor.shape[0], self._features_dim), device=any_tensor.device)

        return self.post(x)
