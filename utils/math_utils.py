#!/usr/bin/env python3
"""
Mathematical Utilities Module
數學工具模組

提供整個框架中使用的數學工具函數

核心功能:
- CRPS計算和相關統計
- 貝氏統計工具
- 數值優化輔助
- 分布處理工具

Author: Research Team  
Date: 2025-01-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from scipy import stats, optimize
from scipy.special import erf, gamma
import warnings

# ========================================
# CRPS 相關函數
# ========================================

def crps_empirical(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    計算經驗CRPS (Continuous Ranked Probability Score)
    
    CRPS = E|Y - X| - 0.5 * E|X - X'|
    
    Parameters:
    -----------
    y_true : np.ndarray
        真實值
    y_pred : np.ndarray
        預測值集合
        
    Returns:
    --------
    float
        CRPS值
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.ndim == 0:
        y_true = y_true.reshape(1)
    
    # 確保維度匹配
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
    
    crps_values = []
    
    for i in range(len(y_true)):
        obs = y_true[i]
        forecast = y_pred[i] if y_pred.shape[0] > 1 else y_pred[0]
        
        # CRPS計算
        term1 = np.mean(np.abs(forecast - obs))
        term2 = 0.5 * np.mean(np.abs(forecast[:, None] - forecast[None, :]))
        
        crps_val = term1 - term2
        crps_values.append(crps_val)
    
    return np.mean(crps_values)

def crps_normal(y_true: float, mu: float, sigma: float) -> float:
    """
    正態分布的解析CRPS計算
    
    Parameters:
    -----------
    y_true : float
        觀測值
    mu : float
        正態分布均值
    sigma : float
        正態分布標準差
        
    Returns:
    --------
    float
        CRPS值
    """
    if sigma <= 0:
        return abs(y_true - mu)
    
    z = (y_true - mu) / sigma
    
    # 解析公式
    crps = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 
                   2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    
    return crps

def crps_lognormal(y_true: float, mu: float, sigma: float) -> float:
    """
    對數正態分布的CRPS計算
    
    Parameters:
    -----------
    y_true : float
        觀測值
    mu : float
        對數正態分布的log-scale均值
    sigma : float
        對數正態分布的log-scale標準差
        
    Returns:
    --------
    float
        CRPS值
    """
    if y_true <= 0:
        return np.inf
    
    if sigma <= 0:
        return abs(y_true - np.exp(mu))
    
    # 使用數值積分近似
    # 這是簡化實現，完整版本需要更複雜的積分
    samples = np.random.lognormal(mu, sigma, 1000)
    return crps_empirical(np.array([y_true]), samples.reshape(1, -1))

def crps_ensemble(y_true: np.ndarray, ensemble: np.ndarray) -> np.ndarray:
    """
    計算集合預測的CRPS
    
    Parameters:
    -----------
    y_true : np.ndarray
        真實值 (n_observations,)
    ensemble : np.ndarray
        集合預測 (n_observations, n_ensemble_members)
        
    Returns:
    --------
    np.ndarray
        每個觀測的CRPS值
    """
    y_true = np.asarray(y_true)
    ensemble = np.asarray(ensemble)
    
    if ensemble.ndim == 1:
        ensemble = ensemble.reshape(-1, 1)
    
    if y_true.shape[0] != ensemble.shape[0]:
        raise ValueError("觀測值和集合預測的數量不匹配")
    
    crps_values = []
    
    for i in range(len(y_true)):
        obs = y_true[i]
        forecast = ensemble[i, :]
        
        # 經驗CRPS計算
        term1 = np.mean(np.abs(forecast - obs))
        
        # 計算集合成員間的平均絕對差異
        n_members = len(forecast)
        term2 = 0.0
        for j in range(n_members):
            for k in range(n_members):
                term2 += np.abs(forecast[j] - forecast[k])
        term2 /= (2 * n_members**2)
        
        crps_val = term1 - term2
        crps_values.append(crps_val)
    
    return np.array(crps_values)

# ========================================
# 貝氏統計工具
# ========================================

def effective_sample_size(samples: np.ndarray, max_lag: int = None) -> float:
    """
    計算有效樣本大小
    
    Parameters:
    -----------
    samples : np.ndarray
        MCMC樣本
    max_lag : int, optional
        最大滯後
        
    Returns:
    --------
    float
        有效樣本大小
    """
    n = len(samples)
    if max_lag is None:
        max_lag = min(n // 4, 200)
    
    # 去除均值
    centered = samples - np.mean(samples)
    
    # 計算自相關函數
    autocorrs = []
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr = 1.0
        else:
            valid_pairs = n - lag
            if valid_pairs <= 0:
                break
            
            numerator = np.sum(centered[:-lag] * centered[lag:])
            denominator = np.sum(centered**2)
            
            if denominator == 0:
                autocorr = 0
            else:
                autocorr = numerator / denominator
        
        autocorrs.append(autocorr)
        
        # 停止條件：自相關變為負數或很小
        if lag > 0 and autocorr <= 0.05:
            break
    
    # 計算積分自相關時間
    tau_int = 1 + 2 * np.sum(autocorrs[1:])
    
    # 有效樣本大小
    n_eff = n / tau_int if tau_int > 0 else n
    
    return max(1.0, n_eff)

def rhat_statistic(chains: List[np.ndarray]) -> float:
    """
    計算Gelman-Rubin R-hat統計量
    
    Parameters:
    -----------
    chains : List[np.ndarray]
        多條MCMC鏈
        
    Returns:
    --------
    float
        R-hat值
    """
    if len(chains) < 2:
        return 1.0
    
    chains = [np.asarray(chain) for chain in chains]
    m = len(chains)  # 鏈數
    n = len(chains[0])  # 每條鏈的長度
    
    # 檢查鏈長度一致性
    if not all(len(chain) == n for chain in chains):
        min_len = min(len(chain) for chain in chains)
        chains = [chain[:min_len] for chain in chains]
        n = min_len
    
    if n < 2:
        return 1.0
    
    # 計算鏈內和鏈間方差
    chain_means = [np.mean(chain) for chain in chains]
    grand_mean = np.mean(chain_means)
    
    # 鏈內方差
    W = np.mean([np.var(chain, ddof=1) for chain in chains])
    
    # 鏈間方差
    B = n * np.var(chain_means, ddof=1)
    
    # 估計的方差
    var_hat = ((n - 1) * W + B) / n
    
    # R-hat
    if W == 0:
        return 1.0
    
    rhat = np.sqrt(var_hat / W)
    return rhat

def hdi_interval(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """
    計算最高密度區間 (Highest Density Interval)
    
    Parameters:
    -----------
    samples : np.ndarray
        樣本
    alpha : float
        顯著水平
        
    Returns:
    --------
    Tuple[float, float]
        (下界, 上界)
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    
    interval_width = int(np.ceil((1 - alpha) * n))
    
    if interval_width >= n:
        return sorted_samples[0], sorted_samples[-1]
    
    # 尋找最窄的區間
    min_width = np.inf
    best_lower = sorted_samples[0]
    best_upper = sorted_samples[-1]
    
    for i in range(n - interval_width + 1):
        lower = sorted_samples[i]
        upper = sorted_samples[i + interval_width - 1]
        width = upper - lower
        
        if width < min_width:
            min_width = width
            best_lower = lower
            best_upper = upper
    
    return best_lower, best_upper

# ========================================
# 數值優化工具
# ========================================

def robust_minimize(func: Callable, 
                   x0: np.ndarray,
                   bounds: Optional[List[Tuple[float, float]]] = None,
                   method: str = "L-BFGS-B",
                   max_attempts: int = 5) -> optimize.OptimizeResult:
    """
    穩健的數值優化，會嘗試多個起始點
    
    Parameters:
    -----------
    func : Callable
        目標函數
    x0 : np.ndarray
        初始猜測
    bounds : List[Tuple[float, float]], optional
        變數邊界
    method : str
        優化方法
    max_attempts : int
        最大嘗試次數
        
    Returns:
    --------
    optimize.OptimizeResult
        優化結果
    """
    best_result = None
    best_fun = np.inf
    
    for attempt in range(max_attempts):
        try:
            # 添加隨機擾動到初始猜測
            if attempt > 0:
                if bounds is not None:
                    # 在邊界內隨機選擇起始點
                    perturbed_x0 = []
                    for i, (lower, upper) in enumerate(bounds):
                        perturbed_x0.append(np.random.uniform(lower, upper))
                    current_x0 = np.array(perturbed_x0)
                else:
                    # 添加高斯噪聲
                    noise_scale = 0.1 * (attempt / max_attempts)
                    current_x0 = x0 + np.random.normal(0, noise_scale, size=x0.shape)
            else:
                current_x0 = x0
            
            # 執行優化
            result = optimize.minimize(
                func, 
                current_x0, 
                method=method, 
                bounds=bounds
            )
            
            # 記錄最佳結果
            if result.success and result.fun < best_fun:
                best_result = result
                best_fun = result.fun
                
        except Exception as e:
            warnings.warn(f"優化嘗試 {attempt + 1} 失敗: {e}")
            continue
    
    if best_result is None:
        # 如果所有嘗試都失敗，返回一個虛擬結果
        class DummyResult:
            def __init__(self):
                self.success = False
                self.fun = np.inf
                self.x = x0
                self.message = "所有優化嘗試都失敗"
        
        return DummyResult()
    
    return best_result

def constrained_optimization(objective: Callable,
                           constraints: List[Dict],
                           x0: np.ndarray,
                           bounds: Optional[List[Tuple[float, float]]] = None) -> optimize.OptimizeResult:
    """
    約束優化求解器
    
    Parameters:
    -----------
    objective : Callable
        目標函數
    constraints : List[Dict]
        約束條件列表
    x0 : np.ndarray
        初始猜測
    bounds : List[Tuple[float, float]], optional
        變數邊界
        
    Returns:
    --------
    optimize.OptimizeResult
        優化結果
    """
    try:
        result = optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        return result
    except Exception as e:
        warnings.warn(f"約束優化失敗: {e}")
        # 回退到無約束優化
        return robust_minimize(objective, x0, bounds)

# ========================================
# 分布處理工具
# ========================================

def fit_distribution(data: np.ndarray, 
                    distribution_name: str = "auto") -> Dict[str, Any]:
    """
    擬合分布到數據
    
    Parameters:
    -----------
    data : np.ndarray
        數據
    distribution_name : str
        分布名稱，"auto"為自動選擇
        
    Returns:
    --------
    Dict[str, Any]
        擬合結果
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]  # 移除NaN
    
    if len(data) == 0:
        return {"distribution": None, "parameters": None, "aic": np.inf}
    
    distributions_to_try = {
        "normal": stats.norm,
        "lognormal": stats.lognorm,
        "gamma": stats.gamma,
        "beta": stats.beta,
        "exponential": stats.expon,
        "weibull": stats.weibull_min
    }
    
    if distribution_name != "auto":
        if distribution_name in distributions_to_try:
            distributions_to_try = {distribution_name: distributions_to_try[distribution_name]}
        else:
            raise ValueError(f"不支援的分布: {distribution_name}")
    
    best_fit = None
    best_aic = np.inf
    
    for name, distribution in distributions_to_try.items():
        try:
            # 特殊處理某些分布
            if name == "beta":
                # Beta分布需要數據在[0,1]範圍內
                if np.any(data < 0) or np.any(data > 1):
                    continue
            elif name == "lognormal":
                # 對數正態分布需要正數據
                if np.any(data <= 0):
                    continue
            
            # 擬合分布
            params = distribution.fit(data)
            
            # 計算對數似然
            log_likelihood = np.sum(distribution.logpdf(data, *params))
            
            # 計算AIC
            k = len(params)  # 參數數量
            aic = 2 * k - 2 * log_likelihood
            
            if aic < best_aic:
                best_aic = aic
                best_fit = {
                    "distribution": name,
                    "distribution_object": distribution,
                    "parameters": params,
                    "log_likelihood": log_likelihood,
                    "aic": aic
                }
                
        except Exception as e:
            warnings.warn(f"無法擬合 {name} 分布: {e}")
            continue
    
    return best_fit if best_fit is not None else {"distribution": None, "parameters": None, "aic": np.inf}

def sample_from_mixture(components: List[Dict], 
                       weights: np.ndarray, 
                       n_samples: int) -> np.ndarray:
    """
    從混合分布中抽樣
    
    Parameters:
    -----------
    components : List[Dict]
        分布組件列表
    weights : np.ndarray
        混合權重
    n_samples : int
        樣本數量
        
    Returns:
    --------
    np.ndarray
        樣本
    """
    weights = np.asarray(weights)
    weights = weights / np.sum(weights)  # 標準化
    
    # 決定每個組件的樣本數量
    component_samples = np.random.multinomial(n_samples, weights)
    
    all_samples = []
    
    for i, (component, n_comp_samples) in enumerate(zip(components, component_samples)):
        if n_comp_samples == 0:
            continue
        
        distribution = component["distribution_object"]
        params = component["parameters"]
        
        samples = distribution.rvs(*params, size=n_comp_samples)
        all_samples.extend(samples)
    
    # 隨機打亂
    all_samples = np.array(all_samples)
    np.random.shuffle(all_samples)
    
    return all_samples

def test_math_utils():
    """測試數學工具功能"""
    print("🧪 測試數學工具模組...")
    
    # 測試CRPS計算
    print("✅ 測試CRPS計算:")
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.random.normal([1.1, 1.9, 3.2], 0.5, (3, 100))
    crps = crps_empirical(y_true, y_pred)
    print(f"   經驗CRPS: {crps:.4f}")
    
    # 測試R-hat統計量
    print("✅ 測試R-hat統計量:")
    chain1 = np.random.normal(0, 1, 1000)
    chain2 = np.random.normal(0.1, 1, 1000)
    rhat = rhat_statistic([chain1, chain2])
    print(f"   R-hat: {rhat:.4f}")
    
    # 測試分布擬合
    print("✅ 測試分布擬合:")
    test_data = np.random.lognormal(0, 1, 500)
    fit_result = fit_distribution(test_data, "auto")
    print(f"   最佳分布: {fit_result['distribution']}")
    
    print("✅ 數學工具測試完成")

if __name__ == "__main__":
    test_math_utils()