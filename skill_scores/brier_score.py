"""
Brier Score and Brier Skill Score for binary event prediction
二元事件預測的布萊爾評分和技能評分
"""

import numpy as np
from sklearn.metrics import brier_score_loss


def calculate_brier_score(observations, predicted_probabilities):
    """
    計算 Brier Score
    BS = (1/N) * Σ(forecast_prob - outcome)²
    
    Parameters:
    -----------
    observations : array-like
        二元觀測結果 (0 or 1)
    predicted_probabilities : array-like
        預測機率 (0 to 1)
        
    Returns:
    --------
    float
        Brier Score (0 to 1, 0為完美預測)
    """
    return brier_score_loss(observations, predicted_probabilities)


def calculate_brier_skill_score(observations, predicted_probabilities, baseline_probabilities=None):
    """
    計算 Brier Skill Score (BSS)
    BSS = 1 - (BS_forecast / BS_baseline)
    
    Parameters:
    -----------
    observations : array-like
        二元觀測結果 (0 or 1)
    predicted_probabilities : array-like
        模型預測機率
    baseline_probabilities : array-like, optional
        基準線預測機率，若無則使用氣候頻率
        
    Returns:
    --------
    float
        Brier Skill Score (-∞ to 1, 1為完美預測, >0才有技能)
    """
    
    obs = np.array(observations)
    pred_prob = np.array(predicted_probabilities)
    
    bs_forecast = calculate_brier_score(obs, pred_prob)
    
    if baseline_probabilities is None:
        # 使用氣候頻率作為基準線
        climatology = np.mean(obs)
        baseline_probabilities = np.full_like(obs, climatology)
    
    bs_baseline = calculate_brier_score(obs, baseline_probabilities)
    
    if bs_baseline == 0:
        return float('inf') if bs_forecast == 0 else float('-inf')
    
    bss = 1 - (bs_forecast / bs_baseline)
    
    return bss


def brier_score_decomposition(observations, predicted_probabilities):
    """
    Brier Score 分解: BS = Reliability - Resolution + Uncertainty
    
    Parameters:
    -----------
    observations : array-like
        二元觀測結果
    predicted_probabilities : array-like
        預測機率
        
    Returns:
    --------
    dict
        Brier Score 分解結果
    """
    
    obs = np.array(observations)
    pred_prob = np.array(predicted_probabilities)
    
    # 基本統計
    n = len(obs)
    o_bar = np.mean(obs)  # 氣候頻率
    bs = calculate_brier_score(obs, pred_prob)
    
    # 將預測機率分組來計算可靠性和解析度
    # 使用十分位數分組
    prob_bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(pred_prob, prob_bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(prob_bins) - 2)
    
    reliability = 0
    resolution = 0
    
    for i in range(len(prob_bins) - 1):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            n_k = np.sum(mask)
            o_k = np.mean(obs[mask])  # 該組的觀測頻率
            p_k = np.mean(pred_prob[mask])  # 該組的預測機率
            
            reliability += (n_k / n) * (p_k - o_k) ** 2
            resolution += (n_k / n) * (o_k - o_bar) ** 2
    
    uncertainty = o_bar * (1 - o_bar)
    
    return {
        'brier_score': bs,
        'reliability': reliability,
        'resolution': resolution,
        'uncertainty': uncertainty,
        'verification': reliability - resolution + uncertainty  # 應等於 BS
    }


def convert_payouts_to_binary_probabilities(payouts, damages, threshold_type='median'):
    """
    將賠付金額轉換為二元事件機率
    用於 Brier Score 計算
    
    Parameters:
    -----------
    payouts : array-like
        參數型保險賠付金額
    damages : array-like
        實際損失金額
    threshold_type : str
        閾值類型 ('median', 'mean', 'percentile_75')
        
    Returns:
    --------
    tuple
        (binary_observations, predicted_probabilities)
    """
    
    payouts = np.array(payouts)
    damages = np.array(damages)
    
    # 定義損失事件閾值
    if threshold_type == 'median':
        threshold = np.median(damages[damages > 0]) if np.any(damages > 0) else 0
    elif threshold_type == 'mean':
        threshold = np.mean(damages[damages > 0]) if np.any(damages > 0) else 0
    elif threshold_type == 'percentile_75':
        threshold = np.percentile(damages[damages > 0], 75) if np.any(damages > 0) else 0
    else:
        threshold = 0
    
    # 二元觀測結果: 損失是否超過閾值
    binary_obs = (damages > threshold).astype(int)
    
    # 預測機率: 基於賠付金額的標準化機率
    max_payout = np.max(payouts) if np.max(payouts) > 0 else 1
    predicted_prob = np.clip(payouts / max_payout, 0, 1)
    
    return binary_obs, predicted_prob