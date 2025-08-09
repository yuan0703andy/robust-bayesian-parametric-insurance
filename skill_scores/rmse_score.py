"""
Root Mean Square Error (RMSE) skill score for parametric insurance
參數型保險的均方根誤差技能評分
"""

import numpy as np
from sklearn.metrics import mean_squared_error


def calculate_rmse(observations, predictions):
    """
    計算 RMSE
    
    Parameters:
    -----------
    observations : array-like
        觀測值 (實際損失)
    predictions : array-like
        預測值 (參數型賠付)
        
    Returns:
    --------
    float
        RMSE 值
    """
    return np.sqrt(mean_squared_error(observations, predictions))


def calculate_rmse_skill_score(observations, predictions, baseline_predictions=None):
    """
    計算 RMSE Skill Score (RMSE-SS)
    RMSE-SS = 1 - (RMSE_model / RMSE_baseline)
    
    Parameters:
    -----------
    observations : array-like
        觀測值 (實際損失)
    predictions : array-like
        模型預測值 (參數型賠付)
    baseline_predictions : array-like, optional
        基準線預測值 (如氣候平均)，若無則使用觀測值平均
        
    Returns:
    --------
    float
        RMSE Skill Score (-∞ to 1, 1為完美預測)
    """
    
    rmse_model = calculate_rmse(observations, predictions)
    
    if baseline_predictions is None:
        # 使用氣候平均作為基準線
        baseline_predictions = np.full_like(observations, np.mean(observations))
    
    rmse_baseline = calculate_rmse(observations, baseline_predictions)
    
    if rmse_baseline == 0:
        return float('inf') if rmse_model == 0 else float('-inf')
    
    rmse_ss = 1 - (rmse_model / rmse_baseline)
    
    return rmse_ss


def analyze_rmse_components(observations, predictions):
    """
    分析 RMSE 的組成成分
    
    Parameters:
    -----------
    observations : array-like
        觀測值
    predictions : array-like
        預測值
        
    Returns:
    --------
    dict
        RMSE 分析結果
    """
    
    obs = np.array(observations)
    pred = np.array(predictions)
    
    # 基本統計
    rmse = calculate_rmse(obs, pred)
    bias = np.mean(pred - obs)  # 偏差
    scatter = np.std(pred - obs)  # 散布
    correlation = np.corrcoef(obs, pred)[0, 1] if np.std(pred) > 0 else 0
    
    return {
        'rmse': rmse,
        'bias': bias,
        'scatter': scatter,
        'correlation': correlation,
        'n_samples': len(obs)
    }