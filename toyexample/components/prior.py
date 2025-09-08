
"""Prior/Likelihood enums and processor moved from toy_example_complete.py."""
from typing import Dict, Any
import numpy as np
import torch
from torch.distributions import Normal, LogNormal, StudentT
from enum import Enum

from config import PriorScenario, LikelihoodFamily

# --- Utility: safe expansion of sigma_obs to match mu_loss (batch, H, E)
def _expand_sigma_obs_to_mu(mu_loss: torch.Tensor, sigma_obs: torch.Tensor) -> torch.Tensor:
    """Expand sigma_obs to match shape of mu_loss (batch, H, E).
    Accepts (batch,), (batch,H), or already (batch,H,E). Handles 0-d scalar as well.
    """
    if isinstance(sigma_obs, (int, float)):
        sigma_obs = torch.tensor(sigma_obs, device=mu_loss.device)
    if sigma_obs.dim() == 0:  # scalar
        return sigma_obs.view(1, 1, 1).expand_as(mu_loss)
    if sigma_obs.dim() == 1:  # (batch,)
        return sigma_obs.view(-1, 1, 1).expand_as(mu_loss)
    if sigma_obs.dim() == 2:  # (batch, H)
        return sigma_obs.unsqueeze(-1).expand_as(mu_loss)
    if sigma_obs.dim() == 3:  # (batch, H, E)
        return sigma_obs
    raise ValueError(f"sigma_obs has unsupported shape {tuple(sigma_obs.shape)}")

# %%
# ============================================================================
# 6. Prior/Likelihood 處理器 - 真正實現不同組合的數學差異
# ============================================================================


class PriorLikelihoodProcessor:
    """Prior/Likelihood處理器 - 將配置轉換為實際的數學實現"""
    
    @staticmethod
    def get_prior_parameters(prior_scenario: PriorScenario, n_params: int = 9) -> Dict[str, Any]:
        """
        根據先驗情境獲取先驗參數 - 改進版本支持更靈活的a, b, v₀學習
        
        擴展參數空間:
        - theta[0-3]: 層次效應參數 (log σ_α, log σ_β, log σ_δ, log ρ_spatial)
        - theta[4]: log(vulnerability_a) - Emanuel函數的風險係數  
        - theta[5]: log(vulnerability_b) - Emanuel函數的指數
        - theta[6]: log(sigma_obs) - 觀測誤差
        - theta[7]: log(v_threshold) - 可學習的閾值風速v₀
        - theta[8]: 額外污染參數或其他擴展
        
        Returns:
            Dict with 'mu_prior' and 'sigma_prior' for N(μ, σ²I)
        """
        # Emanuel 2011的歷史參考值: a=0.0039, b=2.04, v₀=25.7 m/s
        emanuel_a_log = np.log(0.0039)  # ≈ -5.54
        emanuel_b_log = np.log(2.04)    # ≈ 0.71
        emanuel_v0_log = np.log(25.7)   # ≈ 3.25
        
        if prior_scenario == PriorScenario.NON_INFORMATIVE:
                # 非信息先驗: 極大方差，讓數據主導學習
                mu_prior = torch.zeros(n_params)
                # 對a, b, v₀使用更大的方差來提高學習靈活性
                sigma_prior = torch.tensor([
                    5.0,   # log σ_α (區域效應) 
                    5.0,   # log σ_β (個體效應)
                    5.0,   # log σ_δ (空間效應)
                    2.0,   # log ρ_spatial (空間相關性)
                    3.0,   # log(vulnerability_a) - 大方差促進學習
                    2.0,   # log(vulnerability_b) - 中等方差
                    3.0,   # log(sigma_obs) - 觀測誤差
                    1.5,   # log(v_threshold) - 閾值風速
                    2.0    # 額外參數
                ][:n_params])
                
        elif prior_scenario == PriorScenario.WEAK_INFORMATIVE:
            # 弱信息先驗: 以Emanuel值為中心，中等方差
            mu_prior = torch.tensor([
                0.0,           # log σ_α 
                0.0,           # log σ_β 
                0.0,           # log σ_δ 
                1.0,           # log ρ_spatial (默認10km相關長度)
                emanuel_a_log, # log(vulnerability_a) - Emanuel參考
                emanuel_b_log, # log(vulnerability_b) - Emanuel參考  
                np.log(1e6),   # log(sigma_obs) - 默認觀測誤差
                emanuel_v0_log,# log(v_threshold) - Emanuel參考
                0.0            # 額外參數
            ][:n_params])
            
            sigma_prior = torch.tensor([
                2.0,   # 層次效應可有中等變化
                2.0, 
                2.0,
                1.0,   # 空間相關性
                1.5,   # a參數允許較大變化
                1.0,   # b參數相對穩定  
                2.0,   # 觀測誤差變化
                0.8,   # v₀在Emanuel基礎上中等變化
                1.5    # 額外參數
            ][:n_params])
            
        elif prior_scenario == PriorScenario.OPTIMISTIC:
            # 樂觀先驗: 期望較低的脆弱度（較小的a, b）
            mu_prior = torch.tensor([
                -0.5,          # 較小的層次效應
                -0.5, 
                -0.5,
                1.5,           # 更強的空間相關性
                emanuel_a_log - 0.5,  # a偏小 (更樂觀)
                emanuel_b_log - 0.3,  # b偏小 (更樂觀)
                np.log(8e5),   # 較小的觀測誤差
                emanuel_v0_log + 0.2, # v₀偏高 (需更強風才損失)
                -0.5           # 樂觀額外效應
            ][:n_params])
            
            sigma_prior = torch.tensor([
                1.5, 1.5, 1.5, 0.8,  # 層次和空間效應
                1.2,   # a的不確定性
                0.8,   # b相對確定
                1.8,   # 觀測誤差
                0.6,   # v₀變化較小
                1.2    # 額外參數
            ][:n_params])
            
        elif prior_scenario == PriorScenario.PESSIMISTIC:
            # 悲觀先驗: 期望較高的脆弱度（較大的a, b）
            mu_prior = torch.tensor([
                0.3,           # 較大的層次效應
                0.3,
                0.3, 
                0.8,           # 較弱的空間相關性
                emanuel_a_log + 0.4,  # a偏大 (更悲觀)
                emanuel_b_log + 0.2,  # b偏大 (更悲觀)
                np.log(1.2e6), # 較大的觀測誤差
                emanuel_v0_log - 0.15,# v₀偏低 (較弱風就損失)
                0.3            # 悲觀額外效應
            ][:n_params])
            
            sigma_prior = torch.tensor([
                2.0, 2.0, 2.0, 1.2,  # 更大的層次效應不確定性
                1.8,   # a的較大不確定性  
                1.2,   # b的中等不確定性
                2.5,   # 較大觀測誤差不確定性
                0.9,   # v₀的中等變化
                1.8    # 額外參數不確定性
            ][:n_params])
                
        else:
            raise ValueError(f"未知的先驗情境: {prior_scenario}")
            
        return {
            'mu_prior': mu_prior,
            'sigma_prior': sigma_prior,
            'scenario': prior_scenario,
            'emanuel_reference': {
                'a_log': emanuel_a_log,
                'b_log': emanuel_b_log, 
                'v0_log': emanuel_v0_log
            }
        }
    
    @staticmethod
    def compute_likelihood_logprob(observed_losses: torch.Tensor, 
                                       predicted_params: Dict[str, torch.Tensor],
                                       likelihood_family: LikelihoodFamily) -> torch.Tensor:
            """
            根據似然族計算log likelihood
            
            Args:
                observed_losses: (n_hospitals, n_events) 觀測損失
                predicted_params: 模型預測的分佈參數
                likelihood_family: 似然函數族
                
            Returns:
                log_likelihood: 標量張量
            """
            if likelihood_family == LikelihoodFamily.NORMAL:
                # 正態似然: Loss ~ N(μ, σ²)
                mu_loss = predicted_params['mu_loss']  # (batch, hospitals, events)
                sigma_obs_raw = predicted_params.get('sigma_obs', torch.tensor(1e6, device=mu_loss.device))
                std = _expand_sigma_obs_to_mu(mu_loss, sigma_obs_raw)
                
                # 計算log probability（平均而非總和，避免規模依賴）
                dist = Normal(mu_loss, std)
                log_prob = dist.log_prob(observed_losses.unsqueeze(0)).mean(dim=(1, 2))
                
            elif likelihood_family == LikelihoodFamily.LOGNORMAL:
                # 對數正態似然: Loss ~ LogNormal(μ_log, σ_log²)
                mu_log = predicted_params['mu_log']  # (batch, hospitals, events)
                sigma_log = predicted_params['sigma_log']
                # 最後一道安全閘：清理 NaN/Inf 與合理範圍
                mu_log = torch.nan_to_num(mu_log, nan=0.0, posinf=20.0, neginf=-60.0)
                sigma_log = torch.nan_to_num(sigma_log, nan=1e-3, posinf=2.5, neginf=1e-3)
                sigma_log = torch.clamp(sigma_log, 1e-6, 5.0)
                
                dist = LogNormal(mu_log, sigma_log)
                eps = 1e-3
                log_prob = dist.log_prob((observed_losses.unsqueeze(0)).clamp_min(eps)).mean(dim=(1, 2))
                
            elif likelihood_family == LikelihoodFamily.STUDENT_T:
                # Student-t似然: 重尾分佈，對異常值更穩健
                mu_loss = predicted_params['mu_loss']  # (batch, hospitals, events)
                sigma_obs_raw = predicted_params.get('sigma_obs', torch.tensor(1e6, device=mu_loss.device))
                scale = _expand_sigma_obs_to_mu(mu_loss, sigma_obs_raw)
                df = 3.0  # 自由度，較小值產生更重的尾部
                
                dist = StudentT(df, mu_loss, scale)
                log_prob = dist.log_prob(observed_losses.unsqueeze(0)).mean(dim=(1, 2))
                
            else:
                raise ValueError(f"未知的似然族: {likelihood_family}")
                
            # 最後在樣本維做平均，得到「平均 NLL」的對偶
            return log_prob.mean()
    
    print("✅ Prior/Likelihood處理器定義完成")

__all__ = [
    'PriorScenario', 
    'LikelihoodFamily', 
    '_expand_sigma_obs_to_mu', 
    'PriorLikelihoodProcessor'
]