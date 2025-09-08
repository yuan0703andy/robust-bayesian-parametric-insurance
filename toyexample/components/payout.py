"""Differentiable payout functions moved from toy_example_complete.py."""
from typing import Dict, Tuple
import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# 4. 可微分保險賠付函數（Steinmann產品 + Sigmoid逼近）
# ============================================================================

class DifferentiablePayoutFunction(nn.Module):
        """可微分的保險賠付函數 - 「合約引擎」"""
        
        def __init__(self, product_config: Dict, verbose: bool = False):
            super().__init__()
            
            # Steinmann產品配置
            thresholds = torch.tensor(product_config['thresholds'], dtype=torch.float32)
            ratios = torch.tensor(product_config['ratios'], dtype=torch.float32)  
            max_payout = float(product_config['max_payout'])
            steepness = float(product_config.get('steepness', 0.1))
            
            # 註冊為不可訓練參數
            self.register_buffer('thresholds', thresholds)
            self.register_buffer('ratios', ratios)
            self.max_payout = max_payout
            self.steepness = steepness
            self.verbose = verbose
            
            if self.verbose:
                print(f"💰 初始化保險產品: 閾值={thresholds.tolist()}, 比例={ratios.tolist()}")
                print(f"   最大賠付: ${max_payout/1e6:.1f}M, 陡峭度: {steepness}")
        
        def forward(self, loss_distribution_params: Dict[str, torch.Tensor]
                   ) -> Dict[str, torch.Tensor]:
            """
            使用 delta method 將損失的 LogNormal 分佈推至賠付的 LogNormal 近似。
            """
            mu_log = loss_distribution_params['mu_log']   # (B,H,E)
            sigma_log = loss_distribution_params['sigma_log']

            # 代表點：E[X] for LogNormal
            EX = torch.exp(mu_log + 0.5 * sigma_log**2)

            # g(EX) 與 g'(EX)
            payout_det, gprime = self._payout_and_derivative(EX)

            # Var[X] for LogNormal
            varX = (torch.exp(sigma_log**2) - 1.0) * torch.exp(2*mu_log + sigma_log**2)

            # Delta method: Var[Y] ≈ (g'(E[X]))^2 Var[X]
            varY = (gprime**2) * varX

            # 以匹配一二矩方式近似 Y ~ LogNormal(mu_Y, sigma_Y)
            EY = torch.clamp(payout_det, min=1e-6)
            cv2 = torch.clamp(varY / (EY**2), min=0.0)
            sigma_payout_log = torch.sqrt(torch.log1p(cv2))
            mu_payout_log = torch.log(EY) - 0.5 * sigma_payout_log**2

            return {
                'mu_payout_log': mu_payout_log,
                'sigma_payout_log': sigma_payout_log,
                'payout_values': payout_det
            }
        
        def _payout_and_derivative(self, loss_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            計算合約賠付 g(x) 與對損失的一階導數 g'(x)。
            內部以百萬美元為Sigmoid入參，導數需乘 1e-6 還原到美元尺度。
            返回:
              payout_values: g(x)
              gprime: dg/dx
            """
            loss_millions = loss_values / 1e6
            total_payout = torch.zeros_like(loss_values)
            slope_sum = torch.zeros_like(loss_values)  # 累積 ∑ Δr_i * s_i(1-s_i) / k

            prev_ratio = 0.0
            for thr, ratio in zip(self.thresholds, self.ratios):
                dr = float(ratio) - prev_ratio
                if dr <= 0:
                    prev_ratio = float(ratio)
                    continue
                thr_m = float(thr) / 1e6
                s = torch.sigmoid((loss_millions - thr_m) / self.steepness)  # s_i
                total_payout += dr * s
                slope_sum += dr * s * (1 - s) / self.steepness              # s_i(1-s_i)/k
                prev_ratio = float(ratio)

            payout_values = total_payout * self.max_payout                  # g(x)
            gprime = slope_sum * (self.max_payout / 1e6)                    # dg/dx 還原到美元尺度
            return payout_values, gprime

print("✅ 可微分保險賠付函數定義完成")

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

    def forward(self, winds_ms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # winds_ms: [H, E]
        masked = winds_ms.unsqueeze(0).expand(self.N_site, -1, -1)
        big_neg = torch.tensor(-1e6, device=winds_ms.device, dtype=winds_ms.dtype)
        masked = masked + (self.mask.unsqueeze(-1) - 1.0) * (-big_neg)
        # site指標 I_site: [H, E]
        I_site = _smooth_max(masked, tau=self.smooth_tau)
        # 線性ramp到賠付比例
        width = max(self.exhaustion - self.trigger, 1e-6)
        ramp = torch.clamp((I_site - self.trigger) / width, 0.0, 1.0)
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
