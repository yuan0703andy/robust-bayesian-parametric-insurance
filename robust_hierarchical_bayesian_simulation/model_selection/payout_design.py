# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# 0) 小工具：數值穩定 & 退火
# ------------------------------------------------------------
def clamp_eps(eps: torch.Tensor, lo: float = 1e-6, hi: float = 1. - 1e-6):
    return torch.clamp(eps, lo, hi)

def anneal_tau(epoch: int, milestones=(0, 50, 100), taus=(1.0, 0.3, 0.1)) -> float:
    """簡易退火：epoch < 50 → 1.0, <100 → 0.3, 之後 → 0.1"""
    for m, t in zip(milestones[::-1], taus[::-1]):
        if epoch >= m:
            return t
    return taus[0]


# ------------------------------------------------------------
# 1) Max-intensity 聚合 + Soft/Hard 賠付層
# ------------------------------------------------------------
def max_intensity(V_circle: torch.Tensor) -> torch.Tensor:
    """
    V_circle: [N, M, H] 或 [N, H] 的風速樣本 (圈內醫院)
    回傳:
      若輸入[N, M, H] → S_samples: [N, M]
      若輸入[N, H]   → S:         [N]
    """
    return V_circle.max(dim=-1).values

def softplus_hinge(x: torch.Tensor, tau: float) -> torch.Tensor:
    # (x)+ 的平滑近似；tau 越小越接近 ReLU
    return tau * F.softplus(x / tau)

def payout_train(S: torch.Tensor, K: float, alpha: float, Cap: float, tau: float) -> torch.Tensor:
    """
    可微平滑賠付（訓練用）
    S: [N] 或 [N, M] 的 intensity（可已是 max-intensity）
    回傳同 shape 的 payout
    """
    hinge = softplus_hinge(S - K, tau)
    return torch.clamp(alpha * hinge, max=Cap)

def payout_eval(S: torch.Tensor, K: float, alpha: float, Cap: float) -> torch.Tensor:
    """
    硬條款賠付（評估/報價用）
    """
    hinge = torch.clamp(S - K, min=0.0)
    return torch.clamp(alpha * hinge, max=Cap)


# ------------------------------------------------------------
# 2) 觸發機率（硬/平滑）與啟動率
# ------------------------------------------------------------
def trigger_rate_from_samples(
    S_samples: torch.Tensor,  # [N, M]
    K: float,
    smooth: bool = False,
    tau: float = 0.1
) -> torch.Tensor:
    """
    回傳標量張量：平均觸發率 (over N & M)
    """
    if smooth:
        trig = torch.sigmoid((S_samples - K) / tau)
    else:
        trig = (S_samples >= K).float()
    return trig.mean()


# ------------------------------------------------------------
# 3) 二分法解 K 以達成目標啟動率
# ------------------------------------------------------------
@torch.no_grad()
def solve_K_bisection(
    S_samples: torch.Tensor,  # [N, M]
    target_trigger: float,
    K_low: float,
    K_high: float,
    smooth: bool = False,
    tau: float = 0.1,
    tol: float = 1e-4,
    max_iter: int = 64
) -> float:
    low, high = float(K_low), float(K_high)
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        rt = trigger_rate_from_samples(S_samples, mid, smooth=smooth, tau=tau).item()
        if abs(rt - target_trigger) < tol:
            return mid
        if rt > target_trigger:
            # 門檻太低 → 太容易觸發 → 提高 K
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


# ------------------------------------------------------------
# 4) Monte-Carlo 期望賠付與配平預算 (解 alpha)
# ------------------------------------------------------------
def expected_payout_from_samples(
    S_samples: torch.Tensor,  # [N, M]
    K: float,
    alpha: float,
    Cap: float,
    hard: bool = True,
    tau: float = 0.1,
    return_eventwise: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    回傳：
        全樣本平均 E[Payout]（標量）
        若 return_eventwise=True，另回 [N] 的事件級 E[Payout_i]
    """
    if hard:
        P = payout_eval(S_samples, K, alpha, Cap)  # [N, M]
    else:
        P = payout_train(S_samples, K, alpha, Cap, tau=tau)

    eventwise = P.mean(dim=1)  # [N]
    overall = eventwise.mean() # 標量
    return (overall, eventwise) if return_eventwise else overall


@torch.no_grad()
def solve_alpha_for_budget(
    S_samples: torch.Tensor,  # [N, M]
    K: float,
    Cap: float,
    budget: float,
    hard: bool = True,
    tau: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 64,
    alpha_low: float = 0.0,
    alpha_high: Optional[float] = None
) -> float:
    """
    用二分法解 alpha，使 E[Payout] = budget。
    會自動擴張 alpha_high 直到超過 budget。
    """
    # 若未提供 alpha_high，先用倍增法找到一個上界
    if alpha_high is None:
        a = 1.0
        for _ in range(32):
            val = expected_payout_from_samples(S_samples, K, a, Cap, hard=hard, tau=tau).item()
            if val >= budget:
                alpha_high = a
                break
            a *= 2.0
        if alpha_high is None:  # 仍沒達標 → Cap 太低或 Budget 太高
            alpha_high = a

    lo, hi = float(alpha_low), float(alpha_high)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = expected_payout_from_samples(S_samples, K, mid, Cap, hard=hard, tau=tau).item()
        if abs(val - budget) < tol:
            return mid
        if val < budget:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------
# 5) 基差風險（MC 版）：MSE / MAE / (1 - corr)
# ------------------------------------------------------------
@torch.no_grad()
def basis_risk_mc(
    S_samples: torch.Tensor,   # [N, M]
    L: torch.Tensor,           # [N] 實際損失（同單位）
    K: float,
    alpha: float,
    Cap: float,
    metric: Literal["mse", "mae", "1-corr"] = "mse",
    hard: bool = True,
    tau: float = 0.1,
) -> float:
    if hard:
        P = payout_eval(S_samples, K, alpha, Cap)  # [N, M]
    else:
        P = payout_train(S_samples, K, alpha, Cap, tau=tau)
    # MC 估計 E[(Payout - L)^p]：先在樣本維度 M 平均
    P_bar = P.mean(dim=1)  # [N]

    if metric == "mse":
        val = torch.mean((P_bar - L)**2).item()
    elif metric == "mae":
        val = torch.mean(torch.abs(P_bar - L)).item()
    elif metric == "1-corr":
        x = P_bar - P_bar.mean()
        y = L - L.mean()
        denom = (x.std(unbiased=False) * y.std(unbiased=False) + 1e-12)
        corr = torch.mean(x * y) / denom
        val = float(1.0 - corr)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return val


# ------------------------------------------------------------
# 6) 掃 Cap（每個 Cap 內部會解 alpha），以最小化基差
# ------------------------------------------------------------
@torch.no_grad()
def tune_cap_under_budget(
    S_samples: torch.Tensor,  # [N, M]
    L: torch.Tensor,          # [N]
    K: float,
    budget: float,
    cap_grid: torch.Tensor,   # [G]，單位同 payout
    metric: Literal["mse", "mae", "1-corr"] = "mse",
    hard: bool = True,
    tau: float = 0.1,
) -> Tuple[float, float, float]:
    """
    回傳: best_cap, best_alpha, best_metric
    """
    best = (None, None, math.inf)
    for Cap in cap_grid.tolist():
        alpha = solve_alpha_for_budget(S_samples, K, Cap, budget, hard=hard, tau=tau)
        score = basis_risk_mc(S_samples, L, K, alpha, Cap, metric=metric, hard=hard, tau=tau)
        if score < best[2]:
            best = (Cap, alpha, score)
    return best  # (Cap*, alpha*, metric*)


# ------------------------------------------------------------
# 7) （可選）ε-contamination & 溫度化：把這段接到你的 VI/ELBO
# ------------------------------------------------------------
def contaminated_loglik(
    log_f: torch.Tensor,     # [N] 或 [N, ...] 來自主模型似然 log p_f(y|θ)
    log_g: torch.Tensor,     # [N] 或 [N, ...] 來自「污染/重尾」備用模型 log p_g(y)
    eps: torch.Tensor | float
) -> torch.Tensor:
    """
    穩定計算 log[(1-ε)f + ε g]。
    在 ELBO 裡把 log_f 改成這個值即可（其餘 KL 不變）。
    """
    if not torch.is_tensor(eps):
        eps = torch.tensor(eps, device=log_f.device, dtype=log_f.dtype)
    eps = clamp_eps(eps)
    a = torch.log1p(-eps) + log_f
    b = torch.log(eps) + log_g
    return torch.logaddexp(a, b)

def tempered_loglik(loglik: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    溫度化（likelihood tempering）：把 outlier 影響變小（T>1）。
    最簡單的做法：ELBO = E_q[ (1/T) * loglik ] - KL
    """
    return loglik / max(temperature, 1e-8)


# ------------------------------------------------------------
# 8) 一步到位的條款蒐參（包裝）
# ------------------------------------------------------------
@dataclass
class ClauseTuningResult:
    K: float
    Cap: float
    alpha: float
    trigger_rate: float
    budget: float
    basis_score: float


@torch.no_grad()
def tune_clause_parameters(
    V_circle_samples: torch.Tensor,   # [N, M, H] 或 [N, H] 或已經是 [N, M] 的 S
    L: torch.Tensor,                  # [N]
    target_trigger: float,            # 目標啟動率 π*
    budget: float,                    # 預算 (mean payout)
    K_bounds: Tuple[float, float],    # (K_min, K_max)
    cap_grid: torch.Tensor,           # e.g. torch.linspace(0.1, 5.0, 30) * 單位
    metric: Literal["mse", "mae", "1-corr"] = "mse",
    smooth_trig: bool = False,
    trig_tau: float = 0.1,
    hard: bool = True,
    tau: float = 0.1,
) -> ClauseTuningResult:
    # 1) 形成 S_samples (Max-intensity)
    if V_circle_samples.dim() == 3:
        S_samples = max_intensity(V_circle_samples)   # [N, M]
    elif V_circle_samples.dim() == 2:
        # [N, M]：視為已是 S_samples
        S_samples = V_circle_samples
    else:
        # [N, H] 無 MC → 做個假樣本維度 M=1
        S_samples = max_intensity(V_circle_samples).unsqueeze(1)

    # 2) 解 K 以達啟動率
    K = solve_K_bisection(
        S_samples, target_trigger,
        K_low=K_bounds[0], K_high=K_bounds[1],
        smooth=smooth_trig, tau=trig_tau
    )

    # 3) 掃 Cap，內部每個 Cap 解 alpha，挑基差最小
    best_cap, best_alpha, best_basis = tune_cap_under_budget(
        S_samples, L, K, budget, cap_grid, metric=metric, hard=hard, tau=tau
    )

    # 4) 回報硬條款下實際觸發率
    trig = trigger_rate_from_samples(S_samples, K, smooth=False, tau=trig_tau).item()

    return ClauseTuningResult(
        K=K, Cap=best_cap, alpha=best_alpha,
        trigger_rate=trig, budget=budget, basis_score=best_basis
    )

