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


# -------- Combined extractor (2D ego map + 1D lines) with deterministic ordering --------
class BaselineEgoMapAndLinesExtractor(BaseFeaturesExtractor):
    """
    Image-like keys (ego maps) -> DeeperEgoMapCNN
    Vector/line keys (semantics/depth lines) -> SimpleLine1DEncoder
    Concatenate (in a deterministic, user-specified order) -> optional Linear -> features_dim

    Args:
        ordered_keys: optional iterable of keys specifying processing/concat order. If None,
                      uses the Dict's insertion order.
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
        # NEW:
        ordered_keys: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__(observation_space, features_dim=1)
        assert isinstance(observation_space, spaces.Dict), "Expect Dict observation space"

        # ---- resolve key order ----
        all_keys = list(observation_space.spaces.keys())
        if ordered_keys is None:
            keys = all_keys
        else:
            missing = [k for k in ordered_keys if k not in observation_space.spaces]
            if missing:
                raise KeyError(f"ordered_keys contains missing keys: {missing}")
            keys = list(ordered_keys)

        # Keep type+order for forward concatenation
        self.proc_order: list[Tuple[str, str]] = []  # ("img"|"vec", key)

        # Per-type encoders
        self.image_extractors = nn.ModuleDict()
        self.line_extractors = nn.ModuleDict()

        concat_dim = 0
        for k in keys:
            sp = observation_space.spaces[k]
            if is_image_space(sp, normalized_image=normalized_image):
                shape = sp.shape
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
                self.proc_order.append(("img", k))
                concat_dim += image_out
            else:
                L = get_flattened_obs_dim(sp)
                self.line_extractors[k] = SimpleLine1DEncoder(in_len=L, out_dim=line_out)
                self.proc_order.append(("vec", k))
                concat_dim += line_out

        if concat_dim == 0:
            concat_dim = features_dim

        if features_dim != concat_dim:
            self.post = nn.Sequential(nn.Linear(concat_dim, features_dim), nn.SiLU(True))
            self._features_dim = features_dim
        else:
            self.post = nn.Identity()
            self._features_dim = concat_dim

    def forward(self, obs: TensorDict) -> th.Tensor:
        feats = []
        for kind, k in self.proc_order:
            if kind == "img":
                feats.append(self.image_extractors[k](obs[k]))
            else:
                feats.append(self.line_extractors[k](obs[k]))

        if feats:
            x = th.cat(feats, dim=1)
        else:
            # fallback (unlikely)
            any_tensor = next(iter(obs.values()))
            x = th.zeros((any_tensor.shape[0], self._features_dim), device=any_tensor.device)

        return self.post(x)


# inside baseline_extractor.py, after BaselineEgoMapAndLinesExtractor

class ILSACBaseline(nn.Module):
    """
    IL/SAC head using BaselineEgoMapAndLinesExtractor.

    Matches SB3-style policy_kwargs:
      - activation_fn (e.g., nn.ELU)
      - log_std_init (float, e.g., -0.5)
      - net_arch: dict(pi=[...], qf=[...])
      - features_extractor_kwargs: forwarded to BaselineEgoMapAndLinesExtractor
      - share_features_extractor: bool (this head shares by design)
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        act_dim: int,
        device: str | th.device,
        *,
        activation_fn=nn.ELU,                 # <- matches policy_kwargs.activation_fn
        log_std_init: float = -0.5,           # <- matches policy_kwargs.log_std_init
        net_arch: Optional[Dict] = None,      # <- matches policy_kwargs.net_arch
        features_extractor_kwargs: Optional[Dict] = None,   # <- policy_kwargs.features_extractor_kwargs
        share_features_extractor: bool = True,              # <- policy_kwargs.share_features_extractor
        action_low: Optional[th.Tensor | list[float]] = None,
        action_high: Optional[th.Tensor | list[float]] = None,
    ):
        super().__init__()
        self.device = th.device(device)

        # ---- feature extractor (shared) ----
        fx_kwargs = dict(features_dim=128)  # default; will be overridden below if provided
        if features_extractor_kwargs:
            fx_kwargs.update(features_extractor_kwargs)

        self.feature_extractor = BaselineEgoMapAndLinesExtractor(
            observation_space=observation_space, **fx_kwargs
        )
        features_dim = self.feature_extractor._features_dim

        # ---- net_arch parsing ----
        # Defaults to your dict(pi=[256,128], qf=[256,256]) if not provided
        if net_arch is None:
            net_arch = dict(pi=[256, 128], qf=[256, 256])
        pi_layers = net_arch.get("pi", [256, 128])
        qf_layers = net_arch.get("qf", [256, 256])

        Act = activation_fn  # class, e.g., nn.ELU
        def mlp(sizes):
            layers = []
            in_f = sizes[0]
            for out_f in sizes[1:]:
                layers += [nn.Linear(in_f, out_f), Act()]
                in_f = out_f
            return nn.Sequential(*layers), in_f

        # ---- ACTOR ----
        actor_sizes = [features_dim] + list(pi_layers)
        self.actor_fc, actor_out = mlp(actor_sizes)
        self.actor_head = nn.Linear(actor_out, act_dim)
        self.log_std  = nn.Linear(actor_out, act_dim)
        nn.init.xavier_uniform_(self.log_std.weight)
        with th.no_grad():
            self.log_std.bias.fill_(log_std_init)  # SB3-like log_std_init

        # ---- CRITIC ----
        critic_sizes = [features_dim + act_dim] + list(qf_layers)
        self.critic_fc, critic_out = mlp(critic_sizes)
        self.critic_head = nn.Linear(critic_out, 1)

        # ---- action bounds ----
        if action_low is None:
            action_low = [0.0, -1.0]
        if action_high is None:
            action_high = [1.0, 1.0]
        self.register_buffer("action_low",  th.as_tensor(action_low,  dtype=th.float32).view(1, -1))
        self.register_buffer("action_high", th.as_tensor(action_high, dtype=th.float32).view(1, -1))
        assert self.action_low.shape[1] == act_dim and self.action_high.shape[1] == act_dim

        self.reset_parameters()

    # ---- helpers ----
    def _bounded(self, raw: th.Tensor) -> th.Tensor:
        u = th.tanh(raw)          # [-1,1]
        u01 = (u + 1.0) * 0.5     # [0,1]
        return self.action_low + u01 * (self.action_high - self.action_low)

    # ---- forward ----
    def forward(self, observations: Dict[str, th.Tensor], action: Optional[th.Tensor] = None):
        feats = self.feature_extractor(observations)

        h = self.actor_fc(feats)
        raw_action = self.actor_head(h)
        pi_mean = self._bounded(raw_action)

        raw_log_std = self.log_std(h).clamp(min=-20, max=2)
        pi_std = raw_log_std.exp()

        if action is None:
            action = pi_mean

        q_in = th.cat([feats, action], dim=-1)
        q_value = self.critic_head(self.critic_fc(q_in))
        return q_value, pi_mean, pi_std

    def act(self, observations: Dict[str, th.Tensor], deterministic: bool = True) -> th.Tensor:
        with th.no_grad():
            feats = self.feature_extractor(observations)
            h = self.actor_fc(feats)
            raw_mean = self.actor_head(h)
            raw_log_std = self.log_std(h).clamp(min=-20, max=2)
            std = raw_log_std.exp()
            if deterministic:
                return self._bounded(raw_mean)
            return self._bounded(raw_mean + th.randn_like(std) * std)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        nn.init.xavier_uniform_(self.critic_head.weight, gain=0.01)
        nn.init.constant_(self.critic_head.bias, 0.0)

