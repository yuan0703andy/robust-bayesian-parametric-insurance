"""Unified End-to-End VI Model moved from toy_example_complete.py."""
from typing import Dict, Tuple, TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, LogNormal, StudentT

from components.prior import PriorScenario, LikelihoodFamily, PriorLikelihoodProcessor, _expand_sigma_obs_to_mu
from .hbm import DifferentiableHierarchicalBayesianModel
from components.payout import DifferentiablePayoutFunction, CatInCirclePayout, _indemnity_from_loss

if TYPE_CHECKING:
    from .data import SimulatedSpatialData

# GPU configuration (optional - for multi-GPU support)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_MULTI_GPU = torch.cuda.device_count() > 1
GPU_DEVICES = list(range(torch.cuda.device_count())) if USE_MULTI_GPU else [0]

# ============================================================================
# 5. 統一的端到端VI模型
# ============================================================================

class UnifiedEndToEndVIModel(nn.Module):
    """
    統一的端到端變分推斷模型
    
    集成ε-contamination robust方法:
    - Prior contamination: p_ε(θ) = (1-ε_p) * p₀(θ) + ε_p * q_p(θ)  
    - Likelihood contamination: L_ε(θ) = (1-ε_l) * L₀(θ) + ε_l * q_l(θ)
    """
    
    def __init__(self, n_hospitals: int, n_regions: int, n_events: int,
                    distance_matrix: np.ndarray, product_config: Dict,
                    n_hbm_params: int = 7, epsilon_prior: float = 0.0, 
                    epsilon_likelihood: float = 0.0,
                    prior_scenario: PriorScenario = PriorScenario.NON_INFORMATIVE,
                    likelihood_family: LikelihoodFamily = LikelihoodFamily.LOGNORMAL,
                    verbose: bool = False,
                    lambda_kl: float = 0.05,
                    lambda_br: float = 1.0,
                    payout_scale: float = None):
        super().__init__()
        
        self.n_hbm_params = n_hbm_params
        self.epsilon_prior = epsilon_prior         # 先驗污染係數 
        self.epsilon_likelihood = epsilon_likelihood  # 似然污染係數
        self.prior_scenario = prior_scenario       # 先驗情境
        self.likelihood_family = likelihood_family # 似然函數族
        self.verbose = verbose
        self.lambda_kl = float(lambda_kl)
        self.lambda_br = float(lambda_br)
        # 純CRPS模式（不加入likelihood項），符合 ℒ_BR 定義
        
        # 獲取具體的先驗參數
        prior_params = PriorLikelihoodProcessor.get_prior_parameters(prior_scenario, n_hbm_params)
        
        # 變分參數 φ = (μ_θ, log_σ_θ) - 使用適應性初始化
        # 讓 q(θ) 的均值就落在 p(θ) 的中心，避免一開始爆尺度
        self.mu_theta = nn.Parameter(prior_params['mu_prior'].clone())

        # 後驗初始標準差保守一點（例如 0.3），避免過度抖動
        init_sigma = torch.full_like(prior_params['sigma_prior'], 0.3)
        self.log_sigma_theta = nn.Parameter(torch.log(init_sigma))
        
        # 註冊先驗參數為buffer（不可訓練）
        self.register_buffer('prior_mu', prior_params['mu_prior'])
        self.register_buffer('prior_sigma', prior_params['sigma_prior'])
        
        # 子模組
        self.hbm = DifferentiableHierarchicalBayesianModel(
            n_hospitals, n_regions, n_events, distance_matrix, verbose=verbose
        )
        self.payout_function = DifferentiablePayoutFunction(product_config, verbose=verbose)
        # 事件層參數型賠付目標（cat-in-circle），若配置齊全則啟用
        self.param_target = None
        try:
            radius_km = product_config.get('radius_km', None)
            trigger_ms = product_config.get('trigger_ms', None)
            exhaustion_ms = product_config.get('exhaustion_ms', None)
            site_limits = product_config.get('site_limits', None)
            payout_cap = product_config.get('payout_cap', None)
            if (radius_km is not None) and (trigger_ms is not None) and (exhaustion_ms is not None):
                if site_limits is None:
                    # 每院平均分配 max_payout（若未提供）
                    site_limits = np.full(n_hospitals, float(product_config.get('max_payout', 1.0)) / max(n_hospitals,1))
                self.param_target = CatInCirclePayout(
                    distance_matrix_km=distance_matrix,
                    radius_km=radius_km,
                    trigger_ms=trigger_ms,
                    exhaustion_ms=exhaustion_ms,
                    site_limits=np.asarray(site_limits, dtype=np.float32),
                    payout_cap=payout_cap,
                    smooth_tau=product_config.get('smooth_tau', 0.5)
                )
        except Exception:
            self.param_target = None

        # payout 尺度：使用事件層可達的合計限額，避免過度縮放CRPS
        if payout_scale is None:
            if self.param_target is not None and hasattr(self.param_target, 'site_limits'):
                # 事件層上限 = 各 site 限額加總；若有 cap 就取 cap
                cap = float(self.param_target.payout_cap) if self.param_target.payout_cap is not None else None
                limit_sum = float(self.param_target.site_limits.sum().item())
                self.payout_scale = cap if cap is not None else limit_sum
            else:
                # 後備：單院上限
                self.payout_scale = float(product_config.get('max_payout', 1.0))
        else:
            self.payout_scale = float(payout_scale)
        
        if self.verbose:
            print(f"🧠 統一VI模型初始化: {n_hbm_params}個HBM參數")
            print(f"   先驗情境: {prior_scenario.value}")
            print(f"   似然族: {likelihood_family.value}")
            print(f"   ε-contamination: Prior={epsilon_prior:.3f}, Likelihood={epsilon_likelihood:.3f}")
            if epsilon_prior > 0 or epsilon_likelihood > 0:
                print(f"   🛡️  啟用Robust貝氏模式")
    
    def forward(self, hazard_intensities: torch.Tensor,
                exposure_values: torch.Tensor,
                observed_losses: torch.Tensor,
                n_samples: int = 10,
                spatial_data: 'SimulatedSpatialData' = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向傳播計算 Basis-Risk-Aware ELBO，返回張量以便 DataParallel 聚合。
        返回順序: (total_loss, elbo, crps_term, kl_div)
        """
        # 1. VI採樣 θ ~ q_φ(θ)
        theta_samples = self._sample_theta(n_samples)
        
        # 2. 提取區域分配（如果提供），並確保放在正確設備
        region_assignments = None
        if spatial_data is not None and hasattr(spatial_data, 'region_assignments'):
            region_assignments = torch.tensor(spatial_data.region_assignments, dtype=torch.long, device=hazard_intensities.device)
        
        # 3. 損失分佈 G(θ)
        loss_dist_params = self.hbm(hazard_intensities, exposure_values, theta_samples, region_assignments)
        
        # 4. 賠付分佈 F(θ)
        payout_dist_params = self.payout_function(loss_dist_params)

        # (A) NLL: 資料配適（loss likelihood）
        loglik = PriorLikelihoodProcessor.compute_likelihood_logprob(
            observed_losses, loss_dist_params, self.likelihood_family
        )
        nll = -loglik

        # (B) KL(q||p_ε)
        kl_div = self._compute_kl_divergence_with_prior(theta_samples)

        # (C) 分布型基差風險：事件層 CRPS on payout（單位化+MC回退版）
        scale = max(getattr(self, "payout_scale", 1.0), 1.0)
        Y_samples = self._sample_total_payout_from_loss(loss_dist_params, n_pred_samples=50)  # [S,E]
        
        if self.param_target is not None:
            with torch.no_grad():
                y_target, _ = self.param_target(hazard_intensities)
        else:
            with torch.no_grad():
                # 後備：以損失總額的 indemnity 當作理想賠付
                loss_total = observed_losses.sum(dim=0)
                y_target = _indemnity_from_loss(loss_total, deductible=0.0, limit=scale)
        
        # 嘗試解析式 CRPS（原版，已在金額域）
        crps_closed_form = self._crps_event_level(Y_samples, y_target)
        crps_u_closed = crps_closed_form / scale  # 轉為單位化
        
        # 用正確的 MC-CRPS 作為回退（S=32~64 足夠穩）
        S_mc = 32
        Y_mc = self._sample_total_payout_from_loss(loss_dist_params, n_pred_samples=S_mc)  # [S,E] 金額
        crps_u_mc = self._mc_crps_unitless(Y_mc, y_target, scale)  # [E] 單位化
        
        # 檢測解析式是否穩定
        bad = (~torch.isfinite(crps_u_closed)) | (crps_u_closed > 1.0) | (crps_u_closed < 0.0)
        if bad.any():
            print(f"⚠️ 解析式CRPS不穩定: {bad.sum().item()}/{len(bad)} 事件切換到MC-CRPS")
            print(f"   closed形式範圍: [{crps_u_closed.min():.3f}, {crps_u_closed.max():.3f}]")
            print(f"   MC形式範圍: [{crps_u_mc.min():.3f}, {crps_u_mc.max():.3f}]")
        
        crps_u = torch.where(bad, crps_u_mc, crps_u_closed).clamp_(0.0, 1.0)
        
        # 訓練損失用單位化版（更穩定）
        crps_scaled = crps_u

        # 數值保護（其他項）
        nll = torch.nan_to_num(nll, nan=1e6, posinf=1e6, neginf=1e6)
        kl_div = torch.nan_to_num(kl_div, nan=0.0, posinf=1e6, neginf=1e6)

        total_loss = nll + self.lambda_kl * kl_div + self.lambda_br * crps_scaled
        elbo = -total_loss
        total_loss = -elbo
        
        return total_loss, elbo, crps_scaled.detach(), kl_div.detach()

    def _sample_total_payout_from_loss(self, loss_params: Dict[str, torch.Tensor], n_pred_samples: int = 50) -> torch.Tensor:
        mu_log = loss_params['mu_log']
        sigma_log = loss_params['sigma_log']
        if mu_log.dim() == 2:
            mu_log = mu_log.unsqueeze(0); sigma_log = sigma_log.unsqueeze(0)
        B, H, E = mu_log.shape
        S = int(n_pred_samples)
        eps = torch.randn(S, B, H, E, device=mu_log.device)
        X = torch.exp(mu_log.unsqueeze(0) + sigma_log.unsqueeze(0) * eps)  # (S,B,H,E)
        payout_det, _ = self.payout_function._payout_and_derivative(X, training=self.training)
        Y = payout_det.sum(dim=2)  # (S,B,E) 合計各院
        return Y.mean(dim=1)       # (S,E) 對 batch 平均

    def _crps_event_level(self, Y_samples: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        # Y_samples: [S,E], y_target: [E]
        S, E = Y_samples.shape
        term1 = torch.mean(torch.abs(Y_samples - y_target.unsqueeze(0)))
        Ys, _ = torch.sort(Y_samples, dim=0)
        idx = torch.arange(1, S+1, device=Ys.device, dtype=Ys.dtype).view(S, 1)
        coeff = 2 * idx - (S + 1)
        term2 = (Ys * coeff).sum(dim=0) / (S * S)
        crps = term1 - term2.mean()
        return torch.clamp(crps, min=0.0)

    def _mc_crps_unitless(self, samples: torch.Tensor, y_true: torch.Tensor, 
                          scale: float, eps: float = 1e-12) -> torch.Tensor:
        """
        正確的 Monte-Carlo CRPS（單位化 0~1）
        samples: [S, E]  (事件層預測樣本，單位: 金額)
        y_true : [E]     (事件層真值(實際理賠)，單位: 金額)
        scale  : float   (事件層上限，例如 30e6)
        return : [E]     (單位化 CRPS，0~1)
        """
        S = samples.shape[0]
        # 單位化到 [0, 1] 的金額比例域
        X = (samples / (scale + eps)).clamp(0.0, 1.0)      # [S,E]
        Y = (y_true  / (scale + eps)).clamp(0.0, 1.0)      # [E]

        # term1 = E|X - y|
        term1 = (X - Y.unsqueeze(0)).abs().mean(dim=0)     # [E]

        # term2 = E|X - X'|
        # 以 pairwise L1 距離估計；S<=64 直接做就好
        P = (X.unsqueeze(0) - X.unsqueeze(1)).abs()        # [S,S,E]
        term2 = P.mean(dim=(0,1))                          # [E]  == (1/S^2)Σ|x_i-x_j|

        crps_u = term1 - 0.5 * term2                       # [E]
        
        # 檢查數值問題並給出診斷信息
        bad_mask = ~torch.isfinite(crps_u)
        if bad_mask.any():
            print(f"⚠️ MC-CRPS數值問題: {bad_mask.sum().item()}/{len(crps_u)} 事件有NaN/Inf")
            print(f"   term1範圍: [{term1.min():.3f}, {term1.max():.3f}]")
            print(f"   term2範圍: [{term2.min():.3f}, {term2.max():.3f}]")
        
        # 數值保護（仍保持在 unitless 域）
        crps_u = crps_u.nan_to_num(nan=1.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        return crps_u

    def evaluate_metrics(self, hazard_intensities: torch.Tensor,
                         exposure_values: torch.Tensor,
                         observed_losses: torch.Tensor,
                         n_samples: int = 10,
                         spatial_data: 'SimulatedSpatialData' = None,
                         n_pred_samples: int = 50,
                         contam_scale_eval: float = 2.0) -> Dict[str, torch.Tensor]:
        """
        評估用工具：同時計算 CRPS 與 robust log-likelihood 指標（不影響訓練）。
        - CRPS 依據 payout F(θ) 進行，保持可微分設計（僅用於評估時無需梯度）。
        - Robust log-likelihood 使用 ε_like 的混合：log((1-ε) p0 + ε pc)，在HBM likelihood層。
        """
        with torch.no_grad():
            # 1) 產生分佈參數
            theta_samples = self._sample_theta(n_samples)
            region_assignments = None
            if spatial_data is not None and hasattr(spatial_data, 'region_assignments'):
                region_assignments = torch.tensor(spatial_data.region_assignments, dtype=torch.long, device=hazard_intensities.device)
            loss_dist_params = self.hbm(hazard_intensities, exposure_values, theta_samples, region_assignments)
            payout_dist_params = self.payout_function(loss_dist_params)

            # 2) CRPS（取均值）
            # 事件層目標賠付（訓練可用軟，評估/報價用硬）
            if self.param_target is not None:
                y_target, _ = self.param_target(hazard_intensities, hard=True)  # 評估硬條款
            else:
                loss_total = observed_losses.sum(dim=0)
                y_target = _indemnity_from_loss(loss_total, deductible=0.0, limit=self.payout_scale)

            crps_scores = self._compute_crps_batch(y_target, payout_dist_params, n_pred_samples=n_pred_samples)
            crps_mean = torch.mean(crps_scores)
            
            
            # 3) 基礎 log-likelihood（不混合）
            base_loglik = PriorLikelihoodProcessor.compute_likelihood_logprob(
                observed_losses, loss_dist_params, self.likelihood_family
            )

            # 4) 混合 robust log-likelihood（僅評估用）
            eps_like = float(self.epsilon_likelihood)
            if eps_like > 0:
                robust_ll = self._robust_loglikelihood_mixture_eval(
                    observed_losses, loss_dist_params, contam_scale_eval
                )
            else:
                robust_ll = base_loglik

        return {
            'crps_mean': crps_mean,
            'base_loglik_mean': base_loglik,
            'robust_loglik_mean': robust_ll,
            'epsilon_likelihood': torch.tensor(self.epsilon_likelihood),
            'likelihood_family': torch.tensor(0)  # placeholder for logging
        }

    def _robust_loglikelihood_mixture_eval(self, observed_losses: torch.Tensor,
                                            predicted_params: Dict[str, torch.Tensor],
                                            contam_scale: float = 2.0) -> torch.Tensor:
        """僅在評估時使用的 ε_like 混合 log-likelihood（不參與訓練目標）。"""
        mu_log = predicted_params.get('mu_log')
        sigma_log = predicted_params.get('sigma_log')
        mu_loss = predicted_params.get('mu_loss')
        sigma_obs = predicted_params.get('sigma_obs')
        device = mu_log.device
        eps = torch.tensor(float(self.epsilon_likelihood), device=device)

        if self.likelihood_family == LikelihoodFamily.LOGNORMAL:
            base_dist = LogNormal(mu_log, sigma_log)
            contam_dist = LogNormal(mu_log, contam_scale * sigma_log)
            eps_min = 1e-3
            obs_safe = (observed_losses.unsqueeze(0)).clamp_min(eps_min)
            log_p0 = base_dist.log_prob(obs_safe).sum(dim=(1, 2))
            log_pc = contam_dist.log_prob(obs_safe).sum(dim=(1, 2))
        elif self.likelihood_family == LikelihoodFamily.NORMAL:
            std = _expand_sigma_obs_to_mu(mu_loss, sigma_obs)
            base_dist = Normal(mu_loss, std)
            contam_dist = Normal(mu_loss, contam_scale * std)
            log_p0 = base_dist.log_prob(observed_losses.unsqueeze(0)).sum(dim=(1, 2))
            log_pc = contam_dist.log_prob(observed_losses.unsqueeze(0)).sum(dim=(1, 2))
        elif self.likelihood_family == LikelihoodFamily.STUDENT_T:
            df_base, df_c = 3.0, 2.0
            scale = _expand_sigma_obs_to_mu(mu_loss, sigma_obs)
            base_dist = StudentT(df_base, mu_loss, scale)
            contam_dist = StudentT(df_c, mu_loss, contam_scale * scale)
            log_p0 = base_dist.log_prob(observed_losses.unsqueeze(0)).sum(dim=(1, 2))
            log_pc = contam_dist.log_prob(observed_losses.unsqueeze(0)).sum(dim=(1, 2))
        else:
            raise ValueError(f"未知的似然族: {self.likelihood_family}")

        log_mix = torch.logsumexp(torch.stack([
            torch.log(1 - eps) + log_p0,
            torch.log(eps) + log_pc
        ], dim=0), dim=0)

        return log_mix.mean()
    
    def _sample_theta(self, n_samples: int) -> torch.Tensor:
        """使用重參數化技巧採樣HBM參數"""
        sigma_theta = torch.exp(self.log_sigma_theta)
        
        # 重參數化: θ = μ + σ * ε, ε ~ N(0,1)
        device = self.mu_theta.device
        epsilon = torch.randn(n_samples, self.n_hbm_params, device=device)
        theta_samples = self.mu_theta.unsqueeze(0) + sigma_theta.unsqueeze(0) * epsilon
        
        return theta_samples

    def compute_elbo_loss(self, hazard_intensities: torch.Tensor,
                        exposure_values: torch.Tensor, 
                        observed_losses: torch.Tensor,
                        n_samples: int = 10,
                        spatial_data: 'SimulatedSpatialData' = None) -> Dict[str, torch.Tensor]:
        """兼容性包裝：調用 forward 並輸出字典格式。"""
        total_loss, elbo, crps_term, kl_div = self.forward(
            hazard_intensities, exposure_values, observed_losses, n_samples, spatial_data
        )
        return {
            'total_loss': total_loss,
            'elbo': elbo,
            'crps_term': crps_term,
            'kl_term': kl_div
        }
    
    def _compute_crps_batch(self, y_target: torch.Tensor,
                            payout_dist_params: Dict[str, torch.Tensor],
                            n_pred_samples: int = 50) -> torch.Tensor:
        """
        批次計算 CRPS（非負；越小越好）
        CRPS(F,y) = E|X - y| - 0.5 E|X - X'|
        - 重參數化取樣：X = exp(μ + σ·ε)
        - 排序係數法：O(S log S) 計算 E|X - X'|，避免 S^2 記憶體
        回傳 shape 為 (batch,)；若無 batch，回傳 (1,)
        """
        mu_log = payout_dist_params['mu_payout_log']   # (...= [B,H,E] 或 [H,E])
        sigma_log = payout_dist_params['sigma_payout_log']
        device = mu_log.device
        
        if mu_log.dim() == 2:
            mu_log = mu_log.unsqueeze(0); sigma_log = sigma_log.unsqueeze(0)
        B, H, E = mu_log.shape
        S = int(n_pred_samples)
        eps = torch.randn(S, B, H, E, device=device)
        X = torch.exp(mu_log.unsqueeze(0) + sigma_log.unsqueeze(0) * eps)  # (S,B,H,E)
        X = torch.nan_to_num(X, nan=0.0, posinf=1e12, neginf=0.0)

        # 事件層賠付樣本（合併醫院）
        X_event = X.sum(dim=2)  # (S,B,E)

        # y_target: [E] → [1,1,E] → broadcast 到 (S,B,E)
        y = y_target.view(1, 1, E)

        # CRPS(F,y) = E|X - y| - 0.5 E|X - X'|
        term1 = torch.mean(torch.abs(X_event - y), dim=0)        # (B,E)
        Xs, _ = torch.sort(X_event, dim=0)                       # (S,B,E)
        idx = torch.arange(1, S+1, device=device, dtype=Xs.dtype).view(S, 1, 1)
        coeff = 2 * idx - (S + 1)
        term2 = (Xs * coeff).sum(dim=0) / (S * S)
        crps_b = torch.clamp(term1 - term2, min=0.0).mean(dim=1) # (B,)
        return crps_b

    def _compute_kl_divergence_with_prior(self, theta_samples: torch.Tensor) -> torch.Tensor:
        """
        數值安全版 KL(q||p_ε)。避免 -inf、inf、NaN 外溢。
        """
        # 參數護欄
        prior_mu = self.prior_mu
        prior_sigma = torch.clamp(self.prior_sigma, min=1e-6, max=1e6)

        log_sigma_theta_safe = torch.clamp(self.log_sigma_theta, min=-20.0, max=20.0)
        sigma_theta = torch.clamp(torch.exp(log_sigma_theta_safe), min=1e-6, max=1e6)
        mu_theta = torch.nan_to_num(self.mu_theta, nan=0.0, posinf=0.0, neginf=0.0)

        if self.epsilon_prior <= 0:
            # 閉式解（數值安全）
            term1 = (sigma_theta**2 + (mu_theta - prior_mu)**2) / (prior_sigma**2)
            term2 = 2 * torch.log(prior_sigma) - 2 * log_sigma_theta_safe - 1
            kl = 0.5 * torch.sum(torch.nan_to_num(term1 + term2, nan=0.0, posinf=1e6, neginf=1e6))
            return torch.clamp(torch.nan_to_num(kl, nan=1e6, posinf=1e6, neginf=1e6), 0.0, 1e9)

        # ε-contamination 混合先驗路徑（Monte Carlo）
        eps = torch.tensor(float(self.epsilon_prior), device=theta_samples.device, dtype=theta_samples.dtype)
        eps = torch.clamp(eps, 1e-8, 1 - 1e-8)

        # log q_φ(θ)
        log_q = torch.sum(
            -0.5 * ((theta_samples - mu_theta.unsqueeze(0))**2 / sigma_theta.unsqueeze(0)**2)
            - 0.5 * torch.log(2 * torch.pi * sigma_theta.unsqueeze(0)**2), dim=1
        )

        # 基礎先驗 N(prior_mu, prior_sigma²)
        log_p0 = torch.sum(
            -0.5 * ((theta_samples - prior_mu.unsqueeze(0))**2 / prior_sigma.unsqueeze(0)**2)
            - 0.5 * torch.log(2 * torch.pi * prior_sigma.unsqueeze(0)**2), dim=1
        )

        # 污染先驗 N(prior_mu, (κ prior_sigma)²)
        kappa = 3.0
        contamination_sigma = torch.clamp(kappa * prior_sigma, min=1e-6, max=1e6)
        log_qp = torch.sum(
            -0.5 * ((theta_samples - prior_mu.unsqueeze(0))**2 / contamination_sigma.unsqueeze(0)**2)
            - 0.5 * torch.log(2 * torch.pi * contamination_sigma.unsqueeze(0)**2), dim=1
        )

        log_w0 = torch.log1p(-eps)   # log(1-ε)
        log_we = torch.log(eps)      # log(ε)

        log_mix = torch.logsumexp(torch.stack([log_w0 + log_p0, log_we + log_qp], dim=0), dim=0)
        kl_mc = torch.mean(log_q - log_mix)
        kl_mc = torch.nan_to_num(kl_mc, nan=1e6, posinf=1e6, neginf=1e6)
        return torch.clamp(kl_mc, 0.0, 1e9)
    
    # 預留：若需加入似然相關的評估函數，可在此擴展（不影響VI目標）
    
    # 移除重複的KL散度計算，統一使用_compute_kl_divergence_with_prior

    # 多GPU並行支持方法
    def to_multi_gpu(self):
        """跳過DataParallel，僅移動到單一設備（暫時止血修復）。"""
        self.to(device)
        return self
    
    def parallel_mcmc_sampling(self, n_total_samples: int, chunk_size: int = None) -> torch.Tensor:
        """
        並行MCMC採樣 - 在多個GPU上分佈採樣
        
        Args:
            n_total_samples: 總採樣數
            chunk_size: 每個GPU的批次大小
        """
        if not USE_MULTI_GPU or len(GPU_DEVICES) <= 1:
            # 單GPU fallback
            return self._sample_theta(n_total_samples)
        
        if chunk_size is None:
            chunk_size = max(1, n_total_samples // len(GPU_DEVICES))
        
        print(f"🔄 並行MCMC採樣: {n_total_samples}個樣本分佈於{len(GPU_DEVICES)}個GPU")
        
        # 分配採樣任務到不同GPU
        samples_per_gpu = []
        remaining_samples = n_total_samples
        
        for i, gpu_id in enumerate(GPU_DEVICES):
            if i == len(GPU_DEVICES) - 1:  # 最後一個GPU處理剩餘樣本
                gpu_samples = remaining_samples
            else:
                gpu_samples = min(chunk_size, remaining_samples)
            
            if gpu_samples > 0:
                # 在指定GPU上採樣
                with torch.cuda.device(gpu_id):
                    # 移動變分參數到當前GPU
                    mu_theta_gpu = self.mu_theta.to(f"cuda:{gpu_id}")
                    log_sigma_theta_gpu = self.log_sigma_theta.to(f"cuda:{gpu_id}")
                    sigma_theta_gpu = torch.exp(log_sigma_theta_gpu)
                    
                    # 重參數化採樣
                    epsilon = torch.randn(gpu_samples, self.n_hbm_params, device=f"cuda:{gpu_id}")
                    theta_samples_gpu = (mu_theta_gpu.unsqueeze(0) + 
                                    sigma_theta_gpu.unsqueeze(0) * epsilon)
                    
                    samples_per_gpu.append(theta_samples_gpu.to(device))  # 移動到主設備
                    remaining_samples -= gpu_samples
                    
                    print(f"   GPU {gpu_id}: 採樣 {gpu_samples} 個樣本")
    
        # 合併所有GPU的結果
        if samples_per_gpu:
            all_samples = torch.cat(samples_per_gpu, dim=0)
            print(f"✅ 並行採樣完成: {all_samples.shape}")
            return all_samples
        else:
            return self._sample_theta(n_total_samples)

print("✅ 統一端到端VI模型定義完成 (支持雙GPU)")

__all__ = [
    'UnifiedEndToEndVIModel',
    'device',
    'USE_MULTI_GPU', 
    'GPU_DEVICES'
]
