#!/usr/bin/env python3
"""
CRPS-Compatible MCMC Log-Probability Functions
CRPS相容的MCMC對數概率函數

實現CRPS導向的logp函數，用於MCMC採樣器（如NUTS）
這些函數將CRPS優化目標轉換為概率分佈形式，
使其能與標準MCMC採樣器一起使用。

核心概念：
- 將 CRPS(y, F(θ)) 轉換為 logp(θ|y) ∝ -CRPS(y, F(θ)) + log(prior(θ))
- 支持基差風險最小化的參數保險設計
- 實現可微分的CRPS計算用於梯度採樣

Author: Research Team
Date: 2025-01-17
Version: 1.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings

# Try importing PyTorch for differentiable operations
try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
    TorchTensor = torch.Tensor
except ImportError:
    TORCH_AVAILABLE = False
    TorchTensor = "torch.Tensor"

# Try importing JAX for MCMC integration (replaces PyMC)
try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.stats as jsp
    from jax import random, grad, jit, vmap
    from jax.scipy.special import logsumexp, erf
    from functools import partial
    JAX_AVAILABLE = True
    print(f"✅ JAX 版本: {jax.__version__} (replacing PyMC)")
    jax.config.update("jax_enable_x64", True)
except ImportError:
    JAX_AVAILABLE = False

from scipy import stats
from scipy.stats import norm


class CRPSLogProbabilityFunction:
    """
    CRPS相容的對數概率函數
    
    將CRPS優化目標轉換為MCMC採樣器可使用的logp函數
    """
    
    def __init__(self, 
                 parametric_payout_function: Callable,
                 crps_weight: float = 1.0,
                 basis_risk_weight: float = 1.0,
                 prior_weight: float = 1.0):
        """
        初始化CRPS logp函數
        
        Args:
            parametric_payout_function: 參數型賠付函數
            crps_weight: CRPS項目權重
            basis_risk_weight: 基差風險權重
            prior_weight: 先驗權重
        """
        self.parametric_payout_function = parametric_payout_function
        self.crps_weight = crps_weight
        self.basis_risk_weight = basis_risk_weight
        self.prior_weight = prior_weight
        
    def crps_gaussian_logp(self, y_true, mu, sigma):
        """
        高斯分佈的CRPS對數概率
        
        將CRPS轉換為負對數概率：logp ∝ -CRPS
        """
        # 計算標準化殘差
        z = (y_true - mu) / sigma
        
        # 高斯CRPS公式
        crps = sigma * (z * (2 * norm.cdf(z) - 1) + 
                       2 * norm.pdf(z) - 
                       1 / np.sqrt(np.pi))
        
        # 轉換為負對數概率（較小的CRPS = 較高的概率）
        # 使用溫度參數來控制集中度
        temperature = 1.0
        logp = -self.crps_weight * crps / temperature
        
        return logp
    
    def crps_ensemble_logp(self, y_true, forecast_samples):
        """
        基於ensemble的CRPS對數概率
        """
        N = len(y_true)
        M = forecast_samples.shape[1] if len(forecast_samples.shape) > 1 else len(forecast_samples)
        
        crps_scores = []
        for i in range(N):
            y = y_true[i]
            if len(forecast_samples.shape) > 1:
                forecasts = forecast_samples[i]
            else:
                forecasts = forecast_samples
            
            # CRPS近似計算
            crps = np.mean(np.abs(forecasts - y)) - 0.5 * np.mean(
                np.abs(forecasts[:, None] - forecasts[None, :])
            )
            crps_scores.append(crps)
        
        # 轉換為對數概率
        total_crps = np.mean(crps_scores)
        logp = -self.crps_weight * total_crps
        
        return logp
    
    def basis_risk_aware_logp(self, 
                            theta: np.ndarray,
                            observed_losses: np.ndarray,
                            parametric_features: np.ndarray) -> float:
        """
        基差風險導向的對數概率函數
        
        結合CRPS和基差風險最小化
        
        Args:
            theta: 模型參數
            observed_losses: 觀測損失
            parametric_features: 參數特徵（如風速、氣壓等）
            
        Returns:
            對數概率值
        """
        # 1. 計算參數型賠付
        payout_distribution = self.parametric_payout_function(
            theta, parametric_features
        )
        
        # 2. 計算CRPS
        if len(payout_distribution.shape) > 1:
            # Ensemble預測
            crps_logp = self.crps_ensemble_logp(observed_losses, payout_distribution)
        else:
            # 高斯近似
            mu = np.mean(payout_distribution)
            sigma = np.std(payout_distribution) + 1e-6  # 避免除零
            crps_logp = np.sum(self.crps_gaussian_logp(observed_losses, mu, sigma))
        
        # 3. 計算基差風險懲罰
        mean_payout = np.mean(payout_distribution, axis=1) if len(payout_distribution.shape) > 1 else payout_distribution
        
        # 不對稱基差風險：重懲罰賠不夠的情況
        under_compensation = np.maximum(0, observed_losses - mean_payout)
        over_compensation = np.maximum(0, mean_payout - observed_losses)
        
        basis_risk = 2.0 * np.mean(under_compensation) + 0.5 * np.mean(over_compensation)
        basis_risk_logp = -self.basis_risk_weight * basis_risk
        
        # 4. 先驗對數概率
        prior_logp = self._compute_prior_logp(theta)
        
        # 5. 總對數概率
        total_logp = crps_logp + basis_risk_logp + self.prior_weight * prior_logp
        
        return total_logp
    
    def _compute_prior_logp(self, theta: np.ndarray) -> float:
        """計算先驗對數概率"""
        # 預設使用正態先驗
        # θ ~ N(0, I)
        prior_logp = np.sum(norm.logpdf(theta, loc=0, scale=1))
        return prior_logp


class JAXCRPSLogProbability:
    """
    JAX專用的CRPS對數概率函數
    
    提供與JAX MCMC採樣器整合的介面
    """
    
    def __init__(self, 
                 observed_losses: np.ndarray,
                 parametric_features: np.ndarray,
                 parametric_payout_function: Callable):
        """
        初始化JAX CRPS logp
        
        Args:
            observed_losses: 觀測損失數據
            parametric_features: 參數特徵數據
            parametric_payout_function: 賠付函數
        """
        if not JAX_AVAILABLE:
            raise ImportError("需要JAX來使用JAXCRPSLogProbability")
            
        self.observed_losses = jnp.array(observed_losses)
        self.parametric_features = jnp.array(parametric_features)
        self.parametric_payout_function = parametric_payout_function
        
    def create_crps_logp_function(self) -> Callable:
        """
        創建CRPS log probability函數用於JAX MCMC
        
        Returns:
            JAX可微分的log probability函數
        """
        
        @jit
        def crps_logp_func(theta_array):
            """內部CRPS logp函數"""            
            # 計算參數型賠付（簡化版本：線性預測）
            linear_pred = jnp.dot(self.parametric_features, theta_array[:-1])
            log_sigma = theta_array[-1]
            sigma = jnp.exp(log_sigma)  # 確保正數
            
            # 高斯CRPS計算（JAX版本）
            z = (self.observed_losses - linear_pred) / sigma
            
            # 使用JAX操作
            phi_z = jnp.exp(-0.5 * z**2) / jnp.sqrt(2 * jnp.pi)  # 標準正態PDF
            Phi_z = 0.5 * (1 + erf(z / jnp.sqrt(2)))             # 標準正態CDF
            
            crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / jnp.sqrt(jnp.pi))
            
            # 轉換為負對數概率
            total_crps = jnp.sum(crps)
            logp = -total_crps
            
            # 添加prior logp (簡單正態先驗)
            prior_logp = jnp.sum(jsp.norm.logpdf(theta_array[:-1], loc=0.0, scale=1.0))
            prior_logp += jsp.norm.logpdf(log_sigma, loc=0.0, scale=1.0)
            
            return logp + prior_logp
        
        return crps_logp_func


class TorchCRPSLogProbability:
    """
    PyTorch專用的CRPS對數概率函數
    
    提供與PyTorch優化器和HMC採樣器整合的介面
    """
    
    def __init__(self, 
                 observed_losses,
                 parametric_features):
        """
        初始化PyTorch CRPS logp
        """
        if not TORCH_AVAILABLE:
            raise ImportError("需要PyTorch來使用TorchCRPSLogProbability")
            
        self.observed_losses = observed_losses
        self.parametric_features = parametric_features
        
    def crps_logp_pytorch(self, 
                         theta,
                         require_grad: bool = True):
        """
        PyTorch版本的CRPS對數概率
        
        支持自動微分，可用於HMC採樣
        """
        if require_grad:
            theta = theta.requires_grad_(True)
        
        # 線性預測
        linear_pred = torch.matmul(self.parametric_features, theta[:-1])
        log_sigma = theta[-1]
        sigma = torch.exp(log_sigma)
        
        # 標準化殘差
        z = (self.observed_losses - linear_pred) / sigma
        
        # 高斯CRPS計算
        normal_dist = Normal(0, 1)
        phi_z = torch.exp(normal_dist.log_prob(z))  # PDF
        Phi_z = normal_dist.cdf(z)                  # CDF
        
        crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / np.sqrt(np.pi))
        
        # 總CRPS
        total_crps = torch.sum(crps)
        
        # 先驗對數概率（正態先驗）
        prior_logp = torch.sum(Normal(0, 1).log_prob(theta))
        
        # 總對數概率
        logp = -total_crps + prior_logp
        
        return logp


def create_nuts_compatible_logp(observed_losses: np.ndarray,
                               parametric_features: np.ndarray,
                               parametric_payout_function: Callable,
                               framework: str = "pymc") -> Callable:
    """
    創建與NUTS採樣器相容的CRPS logp函數
    
    Args:
        observed_losses: 觀測損失
        parametric_features: 參數特徵
        parametric_payout_function: 賠付函數
        framework: 使用的框架 ("pymc" 或 "pytorch")
        
    Returns:
        NUTS相容的logp函數
    """
    
    if framework == "jax":
        if not JAX_AVAILABLE:
            raise ImportError("JAX not available")
        
        crps_logp = JAXCRPSLogProbability(
            observed_losses=observed_losses,
            parametric_features=parametric_features,
            parametric_payout_function=parametric_payout_function
        )
        
        return crps_logp.create_crps_logp_function()
        
    elif framework == "pytorch":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        observed_tensor = torch.tensor(observed_losses, dtype=torch.float32)
        features_tensor = torch.tensor(parametric_features, dtype=torch.float32)
        
        crps_logp = TorchCRPSLogProbability(
            observed_losses=observed_tensor,
            parametric_features=features_tensor
        )
        
        return crps_logp.crps_logp_pytorch
        
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def test_crps_logp_functions():
    """測試CRPS logp函數"""
    print("🧪 測試CRPS對數概率函數...")
    
    # 生成測試數據
    n_obs = 100
    n_features = 3
    
    np.random.seed(42)
    X = np.random.randn(n_obs, n_features)
    true_theta = np.array([1.0, -0.5, 0.3, 0.1])  # 包含sigma參數
    
    # 生成觀測損失
    y_pred = X @ true_theta[:-1]
    y_obs = y_pred + np.random.normal(0, np.exp(true_theta[-1]), n_obs)
    y_obs = np.maximum(y_obs, 0)  # 確保非負
    
    # 測試基本CRPS logp
    crps_logp = CRPSLogProbabilityFunction(
        parametric_payout_function=lambda theta, X: X @ theta[:-1]
    )
    
    logp_value = crps_logp.basis_risk_aware_logp(
        theta=true_theta,
        observed_losses=y_obs,
        parametric_features=X
    )
    
    print(f"✅ 基本CRPS logp測試: {logp_value:.4f}")
    
    # 測試PyTorch版本（如果可用）
    if TORCH_AVAILABLE:
        torch_logp = TorchCRPSLogProbability(
            observed_losses=torch.tensor(y_obs, dtype=torch.float32),
            parametric_features=torch.tensor(X, dtype=torch.float32)
        )
        
        torch_theta = torch.tensor(true_theta, dtype=torch.float32)
        torch_logp_value = torch_logp.crps_logp_pytorch(torch_theta)
        
        print(f"✅ PyTorch CRPS logp測試: {torch_logp_value.item():.4f}")
    
    # 測試梯度計算
    if TORCH_AVAILABLE:
        torch_theta = torch.tensor(true_theta, dtype=torch.float32, requires_grad=True)
        torch_logp_value = torch_logp.crps_logp_pytorch(torch_theta)
        
        # 計算梯度
        torch_logp_value.backward()
        grad_norm = torch.norm(torch_theta.grad).item()
        
        print(f"✅ 梯度計算測試: 梯度範數 = {grad_norm:.4f}")
    
    print("✅ CRPS logp函數測試完成")


if __name__ == "__main__":
    test_crps_logp_functions()