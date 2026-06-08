from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


class RunningMeanStd(nn.Module if nn is not None else object):
    def __init__(self, shape=(), epsilon: float = 1.0e-4):
        if nn is None:
            raise RuntimeError("RunningMeanStd requires PyTorch.")
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(float(epsilon), dtype=torch.float32))

    @torch.no_grad()
    def update(self, x):
        x = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        if x.numel() == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = delta.square() * self.count * batch_count / total_count
        self.mean.copy_(new_mean)
        self.var.copy_((m_a + m_b + correction) / total_count)
        self.count.copy_(total_count)

    def normalize(self, x, eps: float = 1.0e-8):
        return (x - self.mean) / torch.sqrt(self.var + eps)
