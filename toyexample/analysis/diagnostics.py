
from ..components.config import PriorScenario, LikelihoodFamily, ModelConfiguration
from ..components.prior import PriorLikelihoodProcessor, _expand_sigma_obs_to_mu
from ..core.model import UnifiedEndToEndVIModel
from ..utils.gpu_setup import device, USE_MULTI_GPU, GPU_DEVICES
from typing import Dict, List
from enum import Enum
from typing import Any
import numpy as np
import torch
from torch.distributions import Normal, LogNormal, StudentT


# %%
def demonstrate_prior_likelihood_combinations():
    """演示Prior/Likelihood組合系統"""
    print("🧪 演示Prior/Likelihood組合系統")
    print("="*80)
    
    # 開始演示
    
    # 獲取所有配置
    configs = ModelConfiguration.get_comprehensive_test_configs()
    
    print(f"📊 測試配置總覽 ({len(configs)}個組合):")
    print("-"*80)
    print(f"{'#':<3} {'配置名稱':<45} {'先驗':<12} {'似然':<12} {'ε_prior':<8} {'ε_like':<8}")
    print("-"*80)
    
    for i, config in enumerate(configs):
        print(f"{i+1:<3} {config['name'][:44]:<45} "
              f"{config['prior_scenario'].value[:11]:<12} "
              f"{config['likelihood_family'].value[:11]:<12} "
              f"{config['epsilon_prior']:<8.2f} "
              f"{config['epsilon_likelihood']:<8.2f}")
    
    print("-"*80)
    print(f"\n📋 Prior情境詳細說明:")
    
    # 演示先驗參數
    for prior_scenario in [PriorScenario.NON_INFORMATIVE, PriorScenario.WEAK_INFORMATIVE, 
                          PriorScenario.OPTIMISTIC, PriorScenario.PESSIMISTIC]:
        prior_params = PriorLikelihoodProcessor.get_prior_parameters(prior_scenario)
        print(f"   {prior_scenario.value}:")
        print(f"     μ = {prior_params['mu_prior'][:4].tolist()}")
        print(f"     σ = {prior_params['sigma_prior'][:4].tolist()}")
    
    print(f"\n🎯 Likelihood族說明:")
    print("   normal: 正態分佈 - 標準假設，對異常值敏感")
    print("   lognormal: 對數正態 - 適合損失建模，右偏分佈")
    print("   student_t: Student-t - 重尾分佈，對異常值穩健")
    
    print(f"\n🛡️ ε-contamination穩健程度:")
    print("   基線(0.0, 0.0): 無污染保護")
    print("   中等(0.08, 0.10): 適中穩健性")
    print("   極高(0.15, 0.18): 最大穩健性")
    
    print(f"\n💡 組合邏輯:")
    print("   Prior情境 × Likelihood族 × 穩健程度 = 完整測試矩陣")
    print("   每種組合針對不同的風險預期和建模假設")
    print("   ε-contamination提供對模型誤指定的保護")

    # 已整合到上方單一 __main__ 入口


def verify_robust_statistics_implementation():
    """驗證 robust Bayesian ε-contamination 是否生效的簡易診斷。"""
    print("\n🧪 驗證 Robust Statistics 實作 (KL 對比)")
    H, R, E = 10, 2, 5
    dist_mat = np.abs(np.random.randn(H, H)).astype(np.float32)
    dist_mat = (dist_mat + dist_mat.T) / 2
    np.fill_diagonal(dist_mat, 0.0)
    product = {
        'name': 'Diagnostic Single',
        'thresholds': [1e6, 999e6, 999e6, 999e6],
        'ratios': [1.0, 0.0, 0.0, 0.0],
        'max_payout': 2e6,
        'steepness': 0.1
    }
    model_standard = UnifiedEndToEndVIModel(
        n_hospitals=H, n_regions=R, n_events=E,
        distance_matrix=dist_mat, product_config=product,
        epsilon_prior=0.0, epsilon_likelihood=0.0,
        prior_scenario=PriorScenario.WEAK_INFORMATIVE,
        likelihood_family=LikelihoodFamily.LOGNORMAL
    )
    model_robust = UnifiedEndToEndVIModel(
        n_hospitals=H, n_regions=R, n_events=E,
        distance_matrix=dist_mat, product_config=product,
        epsilon_prior=0.1, epsilon_likelihood=0.15,
        prior_scenario=PriorScenario.WEAK_INFORMATIVE,
        likelihood_family=LikelihoodFamily.LOGNORMAL
    )
    theta_standard = model_standard._sample_theta(100)
    theta_robust = model_robust._sample_theta(100)
    kl_standard = model_standard._compute_kl_divergence_with_prior(theta_standard)
    kl_robust = model_robust._compute_kl_divergence_with_prior(theta_robust)
    diff = torch.abs(kl_robust - kl_standard).item()
    print(f"   KL(standard) = {kl_standard:.3f}")
    print(f"   KL(robust)   = {kl_robust:.3f}")
    print(f"   |Δ|          = {diff:.3f}")
    if diff < 1e-2:
        print("⚠️  KL 幾乎無差，檢查 epsilon 是否正確傳遞或先驗方差設置是否過大")
    else:
        print("✅  Epsilon 已影響 KL，robust 先驗生效")


def diagnose_kl_issue(model: UnifiedEndToEndVIModel):
    """診斷KL散度上升的原因：列出後驗/先驗的均值與方差差異與KL分解。"""
    with torch.no_grad():
        # 後驗參數 q(θ)
        mu_q = model.mu_theta.detach().cpu().numpy()
        sigma_q = torch.exp(model.log_sigma_theta).detach().cpu().numpy()

        # 先驗參數 p(θ)
        mu_p = model.prior_mu.detach().cpu().numpy()
        sigma_p = model.prior_sigma.detach().cpu().numpy()

        print("\n🔍 KL診斷：")
        print("參數 | 後驗μ | 先驗μ | 差異 | 後驗σ | 先驗σ | 比率")
        print("-" * 70)

        for i in range(len(mu_q)):
            diff = mu_q[i] - mu_p[i]
            ratio = sigma_q[i] / sigma_p[i] if sigma_p[i] != 0 else np.inf
            print(f"θ[{i}] | {mu_q[i]:6.3f} | {mu_p[i]:6.3f} | "
                  f"{diff:+6.3f} | {sigma_q[i]:6.3f} | {sigma_p[i]:6.3f} | {ratio:6.3f}")

        # KL 的均值項與方差項貢獻（對角高斯的閉式分解）
        with np.errstate(divide='ignore', invalid='ignore'):
            kl_mean = 0.5 * np.sum(((mu_q - mu_p) ** 2) / (sigma_p ** 2))
            var_ratio = (sigma_q ** 2) / (sigma_p ** 2)
            # 防止 log(0) 與負值
            safe_ratio = np.clip(sigma_q / sigma_p, 1e-12, 1e12)
            kl_var = 0.5 * np.sum(var_ratio - 1 - 2 * np.log(safe_ratio))

        total_kl = kl_mean + kl_var
        print("\nKL分解：")
        print(f"  均值項貢獻: {kl_mean:.3f}")
        print(f"  方差項貢獻: {kl_var:.3f}")
        print(f"  總KL: {total_kl:.3f}")


def diagnose_kl_per_dim(model: UnifiedEndToEndVIModel, topk: int = 5):
    """快速診斷：逐維KL，找出貢獻最大的參數維度。
    使用標準對角高斯 KL 條款（非ε混合）做近似拆解：
      KL_i = 0.5 * [ (σ_i^2 + (μ_i-μ0_i)^2)/σ0_i^2 + 2 log σ0_i - 2 log σ_i - 1 ]
    """
    base = model.module if hasattr(model, 'module') else model
    with torch.no_grad():
        mu = base.mu_theta.detach()
        log_sigma = base.log_sigma_theta.detach()
        mu0 = base.prior_mu
        sigma0 = torch.clamp(base.prior_sigma, min=1e-8)
        sigma = torch.exp(log_sigma)

        kl_per_dim = 0.5 * ((sigma**2 + (mu - mu0)**2) / (sigma0**2) + 2*torch.log(sigma0) - 2*log_sigma - 1)
        total_kl = kl_per_dim.sum().item()

        k = int(min(topk, kl_per_dim.numel()))
        top_vals, top_idx = torch.topk(kl_per_dim, k=k)

        print("\n🔎 KL 逐維診斷：")
        print(f"KL sum: {total_kl:.6f}")
        print("Top dims (idx, value):")
        for i in range(k):
            idx = int(top_idx[i].item())
            val = float(top_vals[i].item())
            dmu = float((mu[idx] - mu0[idx]).item())
            ratio = float((sigma[idx] / sigma0[idx]).item())
            print(f"  #{idx:02d}: KL={val:.6f} | μ-μ0={dmu:+.4f} | σ/σ0={ratio:.4f}")

        # 也輸出完整向量（如需外部分析）
        return {
            'kl_per_dim': kl_per_dim.cpu().numpy(),
            'mu_minus_mu0': (mu - mu0).cpu().numpy(),
            'sigma_over_sigma0': (sigma / sigma0).cpu().numpy(),
            'top_idx': top_idx.cpu().numpy(),
            'top_vals': top_vals.cpu().numpy(),
            'kl_sum': total_kl,
        }
