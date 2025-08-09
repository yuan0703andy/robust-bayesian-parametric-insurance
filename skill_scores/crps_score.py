"""
Continuous Ranked Probability Score (CRPS) for probabilistic forecasts
機率性預測的連續分級機率評分
"""

import numpy as np
from scipy import integrate


def calculate_crps(observations, forecasts_ensemble=None, forecasts_mean=None, forecasts_std=None):
    """
    計算 CRPS (Continuous Ranked Probability Score)
    
    Parameters:
    -----------
    observations : array-like
        觀測值
    forecasts_ensemble : array-like, optional
        集合預測 (shape: n_samples x n_ensemble_members)
    forecasts_mean : array-like, optional
        高斯分布預測的平均值
    forecasts_std : array-like, optional
        高斯分布預測的標準差
        
    Returns:
    --------
    float or array
        CRPS 值
    """
    
    observations = np.array(observations)
    
    if forecasts_ensemble is not None:
        # 集合預測的 CRPS
        return crps_ensemble(observations, forecasts_ensemble)
    
    elif forecasts_mean is not None and forecasts_std is not None:
        # 高斯分布的 CRPS
        return crps_gaussian(observations, forecasts_mean, forecasts_std)
    
    else:
        raise ValueError("Must provide either forecasts_ensemble or (forecasts_mean, forecasts_std)")


def crps_ensemble(observations, forecasts):
    """
    計算集合預測的 CRPS
    CRPS = E|X - Y| - 0.5 * E|X - X'|
    其中 X, X' 是獨立的預測樣本, Y 是觀測值
    """
    
    observations = np.array(observations)
    forecasts = np.array(forecasts)
    
    if forecasts.ndim == 1:
        forecasts = forecasts.reshape(1, -1)
    
    if len(observations) != forecasts.shape[0]:
        if len(observations) == 1:
            observations = np.repeat(observations, forecasts.shape[0])
        else:
            raise ValueError("Observations and forecasts dimension mismatch")
    
    crps_values = []
    
    for i in range(len(observations)):
        obs = observations[i]
        forecast_ensemble = forecasts[i]
        
        # E|X - Y|: 預測與觀測的平均絕對誤差
        term1 = np.mean(np.abs(forecast_ensemble - obs))
        
        # 0.5 * E|X - X'|: 預測成員間的平均絕對差異
        n_members = len(forecast_ensemble)
        if n_members > 1:
            pairwise_diffs = []
            for j in range(n_members):
                for k in range(j + 1, n_members):
                    pairwise_diffs.append(abs(forecast_ensemble[j] - forecast_ensemble[k]))
            term2 = 0.5 * np.mean(pairwise_diffs)
        else:
            term2 = 0
        
        crps_values.append(term1 - term2)
    
    return np.array(crps_values)


def crps_gaussian(observations, mu, sigma):
    """
    計算高斯分布預測的 CRPS (解析解)
    
    Parameters:
    -----------
    observations : array-like
        觀測值
    mu : array-like
        預測分布的平均值
    sigma : array-like
        預測分布的標準差
        
    Returns:
    --------
    array
        CRPS 值
    """
    
    from scipy.stats import norm
    
    observations = np.array(observations)
    mu = np.array(mu)
    sigma = np.array(sigma)
    
    # 標準化
    z = (observations - mu) / sigma
    
    # 高斯 CRPS 解析解 - 修復 method 問題
    # 確保 norm.cdf 和 norm.pdf 返回數值而不是 method 對象
    cdf_values = np.array([norm.cdf(zi) if np.isscalar(zi) else norm.cdf(zi) for zi in np.atleast_1d(z)])
    pdf_values = np.array([norm.pdf(zi) if np.isscalar(zi) else norm.pdf(zi) for zi in np.atleast_1d(z)])
    
    if np.isscalar(z):
        cdf_values = cdf_values[0]
        pdf_values = pdf_values[0]
    
    crps = sigma * (z * (2 * cdf_values - 1) + 2 * pdf_values - 1 / np.sqrt(np.pi))
    
    return crps


def calculate_crps_skill_score(observations, forecasts_ensemble=None, forecasts_mean=None, 
                              forecasts_std=None, baseline_forecasts=None):
    """
    計算 CRPS Skill Score (CRPSS)
    CRPSS = 1 - (CRPS_forecast / CRPS_baseline)
    
    Parameters:
    -----------
    observations : array-like
        觀測值
    forecasts_ensemble : array-like, optional
        模型集合預測
    forecasts_mean, forecasts_std : array-like, optional
        模型高斯預測參數
    baseline_forecasts : dict, optional
        基準線預測，應包含相同格式的預測
        
    Returns:
    --------
    float
        CRPS Skill Score
    """
    
    # 計算模型 CRPS
    crps_model = calculate_crps(
        observations, forecasts_ensemble, forecasts_mean, forecasts_std
    )
    
    if baseline_forecasts is None:
        # 使用氣候學基準線 (觀測值的平均和標準差)
        climatology_mean = np.full_like(observations, np.mean(observations))
        climatology_std = np.full_like(observations, np.std(observations))
        crps_baseline = crps_gaussian(observations, climatology_mean, climatology_std)
    else:
        # 使用提供的基準線預測
        crps_baseline = calculate_crps(
            observations,
            baseline_forecasts.get('ensemble'),
            baseline_forecasts.get('mean'),
            baseline_forecasts.get('std')
        )
    
    # 計算平均 CRPS
    mean_crps_model = np.mean(crps_model)
    mean_crps_baseline = np.mean(crps_baseline)
    
    if mean_crps_baseline == 0:
        return float('inf') if mean_crps_model == 0 else float('-inf')
    
    crpss = 1 - (mean_crps_model / mean_crps_baseline)
    
    return crpss


def create_ensemble_from_parametric_payouts(payouts, damages, n_ensemble=100, uncertainty_factor=0.2):
    """
    從參數型賠付創建集合預測，用於 CRPS 計算
    
    Parameters:
    -----------
    payouts : array-like
        參數型保險賠付
    damages : array-like
        實際損失 (用於校準不確定性)
    n_ensemble : int
        集合成員數量
    uncertainty_factor : float
        不確定性因子 (相對於賠付金額的標準差)
        
    Returns:
    --------
    array
        集合預測 (shape: n_events x n_ensemble)
    """
    
    payouts = np.array(payouts)
    n_events = len(payouts)
    
    ensemble_forecasts = np.zeros((n_events, n_ensemble))
    
    for i in range(n_events):
        payout = payouts[i]
        
        # 基於賠付金額設定不確定性
        if payout > 0:
            # 對有賠付的事件，添加相對不確定性
            std_dev = payout * uncertainty_factor
            ensemble_forecasts[i, :] = np.random.normal(payout, std_dev, n_ensemble)
            # 確保非負
            ensemble_forecasts[i, :] = np.maximum(ensemble_forecasts[i, :], 0)
        else:
            # 對無賠付的事件，小機率有少量賠付
            small_payout_prob = 0.1
            for j in range(n_ensemble):
                if np.random.random() < small_payout_prob:
                    # 小額隨機賠付
                    max_possible = np.max(payouts) * 0.1 if np.max(payouts) > 0 else 0
                    ensemble_forecasts[i, j] = np.random.exponential(max_possible * 0.1)
                else:
                    ensemble_forecasts[i, j] = 0
    
    return ensemble_forecasts


def analyze_crps_components(observations, forecasts_ensemble):
    """
    分析 CRPS 的分解和診斷統計
    
    Parameters:
    -----------
    observations : array-like
        觀測值
    forecasts_ensemble : array-like
        集合預測
        
    Returns:
    --------
    dict
        CRPS 分析結果
    """
    
    crps_values = crps_ensemble(observations, forecasts_ensemble)
    
    # 集合統計
    ensemble_means = np.mean(forecasts_ensemble, axis=1)
    ensemble_stds = np.std(forecasts_ensemble, axis=1)
    
    # 分析結果
    return {
        'mean_crps': np.mean(crps_values),
        'median_crps': np.median(crps_values),
        'crps_std': np.std(crps_values),
        'ensemble_spread_mean': np.mean(ensemble_stds),
        'bias': np.mean(ensemble_means - observations),
        'correlation': np.corrcoef(ensemble_means, observations)[0, 1] if np.std(ensemble_means) > 0 else 0,
        'n_samples': len(observations)
    }