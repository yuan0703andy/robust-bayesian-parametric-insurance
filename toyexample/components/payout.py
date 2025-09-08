"""Differentiable payout functions moved from toy_example_complete.py."""
from typing import Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
from ..utils.smoothing_tools import _ramp_soft, _softplus_hinge

# ============================================================================
# 4. 可微分保險賠付函數（Steinmann產品 + Sigmoid逼近）
# ============================================================================

class DifferentiablePayoutFunction(nn.Module):
    """分層線性賠付（Stepped linear）。
       訓練：softplus-hinge 平滑（tau_ramp）；評估：硬條款（piecewise linear）。
    """
    def __init__(self, product_config: Dict, verbose: bool = False):
        super().__init__()
        thr = torch.tensor(product_config['thresholds'], dtype=torch.float32)   # [T]
        rat = torch.tensor(product_config['ratios'],     dtype=torch.float32)   # [T] 累積比例，遞增至 ≤1
        self.register_buffer('thresholds', thr)
        self.register_buffer('ratios',     rat)

        # 若未提供 widths，就用相鄰 threshold 的差；最後一段沿用倒數第二段寬
        if 'widths' in product_config:
            widths = torch.tensor(product_config['widths'], dtype=torch.float32)
        else:
            d = torch.diff(thr)
            last = d[-1] if d.numel() > 0 else torch.tensor(1.0)
            widths = torch.cat([d, last.view(1)])
        self.register_buffer('widths', widths)

        self.max_payout = float(product_config['max_payout'])
        # 平滑溫度（退火目標）：大→平滑、小→近硬條款
        self.tau_ramp  = float(product_config.get('tau_ramp', 0.5))
        # 在 eval() 時是否強制硬條款
        self.eval_hard = bool(product_config.get('hard_eval', True))
        self.verbose   = verbose

        if self.verbose:
            print(f"💰 初始化分層線性賠付: thresholds={thr.tolist()}, ratios={rat.tolist()}, widths={widths.tolist()}")
            print(f"   max_payout={self.max_payout:.0f}, tau_ramp={self.tau_ramp}, eval_hard={self.eval_hard}")

    def set_tau(self, tau: float):
        self.tau_ramp = float(tau)

    def forward(self, loss_distribution_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Delta method：把 Loss 的 LogNormal 推到 Payout 的 LogNormal 近似"""
        mu_log    = loss_distribution_params['mu_log']         # (B,H,E)
        sigma_log = loss_distribution_params['sigma_log']

        EX = torch.exp(mu_log + 0.5 * sigma_log**2)            # 代表點 E[X]
        payout_det, gprime = self._payout_and_derivative(EX, training=self.training)

        varX = (torch.exp(sigma_log**2) - 1.0) * torch.exp(2*mu_log + sigma_log**2)
        varY = (gprime**2) * varX

        EY   = torch.clamp(payout_det, min=1e-6)
        cv2  = torch.clamp(varY / (EY**2), min=0.0)
        sigma_payout_log = torch.sqrt(torch.log1p(cv2))
        mu_payout_log    = torch.log(EY) - 0.5 * sigma_payout_log**2

        return {
            'mu_payout_log':    mu_payout_log,
            'sigma_payout_log': sigma_payout_log,
            'payout_values':    payout_det
        }

    def _payout_and_derivative(self, loss_values: torch.Tensor, training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """多階分層線性：每階 i 的增量 dr_i = ratios[i]-ratios[i-1]，
           在 [thr_i, thr_i+width_i] 線性從 0→1；總賠付 = max_payout * ∑ dr_i * ramp_i。
           訓練用平滑 ramp，eval()/報價用硬 ramp。
        """
        thresholds, ratios, widths = self.thresholds, self.ratios, self.widths
        payout_ratio = torch.zeros_like(loss_values)
        d_ratio_dx   = torch.zeros_like(loss_values)

        prev = 0.0
        for i in range(len(thresholds)):
            dr = float(ratios[i].item() - prev)
            if dr <= 0:
                prev = float(ratios[i]); continue
            t  = float(thresholds[i].item())
            w  = float(widths[i].item())

            if training and not self.eval_hard:
                ramp, drdx = _ramp_soft(loss_values, t, t + w, self.tau_ramp)
            else:
                # 硬條款：ramp = clamp((x - t)/w, 0, 1)；其導數在區間內為 1/w
                r = (loss_values - t) / max(w, 1e-6)
                ramp = torch.clamp(r, 0.0, 1.0)
                drdx = ((loss_values > t) & (loss_values < t + w)).to(loss_values.dtype) / max(w, 1e-6)

            payout_ratio = payout_ratio + dr * ramp
            d_ratio_dx   = d_ratio_dx   + dr * drdx
            prev = float(ratios[i])

        payout_values = payout_ratio * self.max_payout
        gprime        = d_ratio_dx   * self.max_payout
        return payout_values, gprime

# %%
# ============================================================================
# 4.1 參數型 cat-in-circle 賠付目標（事件層觸發）
# ============================================================================

def _smooth_max(x: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    return tau * torch.logsumexp(x / tau, dim=-1)

def _indemnity_from_loss(loss: torch.Tensor, deductible: float = 0.0, limit: float = float('inf')) -> torch.Tensor:
    pay = torch.clamp(loss - deductible, min=0.0)
    if torch.isfinite(torch.tensor(limit, device=loss.device)):
        pay = torch.clamp(pay, max=limit)
    return pay

class CatInCirclePayout(nn.Module):
    """基於最大風速觸發的參數型賠付：事件層目標生成器。
    - 對每個 site（此處以醫院作為 site）取半徑R內的站點最大風速（用平滑max近似）
    - 指標 I = max_{circle} wind
    - 賠付_site = ramp(I; trigger→exhaustion) * site_limit
    - 事件總賠付 = 各 site 賠付加總（可選加總限額）
    備註：本模組僅用於產生 y_target（監管/產品定義），不參與梯度。
    """
    def __init__(self, distance_matrix_km: np.ndarray, radius_km: float,
                 trigger_ms: float, exhaustion_ms: float,
                 site_limits: np.ndarray, smooth_tau: float = 0.5,
                 payout_cap: float = None, site_weights: np.ndarray = None):
        super().__init__()
        D = torch.tensor(distance_matrix_km, dtype=torch.float32)
        self.register_buffer('D', D)
        H = D.shape[0]
        self.N_site, self.N_sta = H, H
        self.R = float(radius_km)
        self.trigger = float(trigger_ms)
        self.exhaustion = float(exhaustion_ms)
        self.smooth_tau = float(smooth_tau)
        self.register_buffer('site_limits', torch.tensor(site_limits, dtype=torch.float32))
        if site_weights is None:
            self.register_buffer('site_weights', torch.ones(H, dtype=torch.float32))
        else:
            self.register_buffer('site_weights', torch.tensor(site_weights, dtype=torch.float32))
        self.payout_cap = payout_cap
        # 建立圓形遮罩
        mask = (D <= self.R).float()
        empty = (mask.sum(dim=1) == 0)
        if torch.any(empty):
            nearest_idx = torch.argmin(D[empty], dim=1)
            mask[empty, :] = 0.0
            mask[empty, nearest_idx] = 1.0
        self.register_buffer('mask', mask)

    def forward(self, winds_ms: torch.Tensor, tau: float = None, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # winds_ms: [H,E]
        tau_eff = float(self.smooth_tau if tau is None else tau)
        mask = self.mask  # [H,H]

        if hard:
            # 硬 max（符合條款）
            masked = winds_ms.unsqueeze(0).expand(self.N_site, -1, -1)  # [H,H,E]
            masked[mask == 0] = -1e6
            I_site = masked.max(dim=1).values  # [H,E]
        else:
            # 平滑 max（訓練穩定）
            masked = winds_ms.unsqueeze(0).expand(self.N_site, -1, -1)
            big_neg = torch.tensor(-1e6, device=winds_ms.device, dtype=winds_ms.dtype)
            masked = masked + (mask.unsqueeze(-1) - 1.0) * (-big_neg)
            I_site = tau_eff * torch.logsumexp(masked / tau_eff, dim=1)

        width = max(self.exhaustion - self.trigger, 1e-6)
        if hard:
            ramp = torch.clamp((I_site - self.trigger) / width, 0.0, 1.0)
        else:
            ramp, _ = _ramp_soft(I_site, self.trigger, self.exhaustion, tau_eff)

        payout_site = ramp * self.site_limits.unsqueeze(-1)
        total = (self.site_weights.unsqueeze(-1) * payout_site).sum(dim=0)  # [E]
        if self.payout_cap is not None:
            total = torch.clamp(total, max=self.payout_cap)
        return total, I_site

print("✅ 可微分保險賠付函數定義完成")

__all__ = [
    'DifferentiablePayoutFunction',
    'CatInCirclePayout',
    '_smooth_max',
    '_indemnity_from_loss'
]
