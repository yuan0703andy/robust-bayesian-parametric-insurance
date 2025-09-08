from typing import Tuple
import torch


def _softplus_hinge(x: torch.Tensor, tau: float) -> torch.Tensor:
    # 近似 ReLU(x) = max(x,0)；tau 越小越接近硬 hinge
    return tau * torch.log1p(torch.exp(x / tau))

def _ramp_soft(x: torch.Tensor, t0: float, t1: float, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """平滑的線性 ramp: clip((x - t0)/(t1 - t0), 0, 1) 的可微近似。
       回傳 (ramp, dr/dx)
    """
    w = max(t1 - t0, 1e-6)
    a = x - t0
    b = x - t1
    num = _softplus_hinge(a, tau) - _softplus_hinge(b, tau)  # ≈ clip(x-t0, 0, w)
    ramp = num / w
    sig_a = torch.sigmoid(a / tau)
    sig_b = torch.sigmoid(b / tau)
    drdx = (sig_a - sig_b) / w
    return ramp, drdx