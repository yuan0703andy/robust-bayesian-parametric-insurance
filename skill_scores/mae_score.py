"""
Mean Absolute Error (MAE) skill score for parametric insurance
參數型保險的平均絕對誤差技能評分
"""

import numpy as np
from sklearn.metrics import mean_absolute_error


def calculate_mae(observations, predictions):
    """
    計算 MAE
    
    Parameters:
    -----------
    observations : array-like
        觀測值 (實際損失)
    predictions : array-like
        預測值 (參數型賠付)
        
    Returns:
    --------
    float
        MAE 值
    """
    return mean_absolute_error(observations, predictions)


def calculate_mae_skill_score(observations, predictions, baseline_predictions=None):
    """
    計算 MAE Skill Score (MAE-SS)
    MAE-SS = 1 - (MAE_model / MAE_baseline)
    
    Parameters:
    -----------
    observations : array-like
        觀測值 (實際損失)
    predictions : array-like
        模型預測值 (參數型賠付)
    baseline_predictions : array-like, optional
        基準線預測值，若無則使用觀測值平均
        
    Returns:
    --------
    float
        MAE Skill Score (-∞ to 1, 1為完美預測)
    """
    
    mae_model = calculate_mae(observations, predictions)
    
    if baseline_predictions is None:
        # 使用氣候平均作為基準線
        baseline_predictions = np.full_like(observations, np.mean(observations))
    
    mae_baseline = calculate_mae(observations, baseline_predictions)
    
    if mae_baseline == 0:
        return float('inf') if mae_model == 0 else float('-inf')
    
    mae_ss = 1 - (mae_model / mae_baseline)
    
    return mae_ss


def calculate_median_absolute_error(observations, predictions):
    """
    計算中位數絕對誤差 (MedAE)
    對異常值更穩健的誤差指標
    
    Parameters:
    -----------
    observations : array-like
        觀測值
    predictions : array-like
        預測值
        
    Returns:
    --------
    float
        MedAE 值
    """
    return np.median(np.abs(np.array(observations) - np.array(predictions)))


def analyze_mae_components(observations, predictions):
    """
    分析 MAE 的組成成分和穩健性指標
    
    Parameters:
    -----------
    observations : array-like
        觀測值
    predictions : array-like
        預測值
        
    Returns:
    --------
    dict
        MAE 分析結果
    """
    
    obs = np.array(observations)
    pred = np.array(predictions)
    errors = pred - obs
    
    return {
        'mae': calculate_mae(obs, pred),
        'medae': calculate_median_absolute_error(obs, pred),
        'mean_error': np.mean(errors),  # 平均誤差 (偏差)
        'median_error': np.median(errors),  # 中位數誤差
        'max_absolute_error': np.max(np.abs(errors)),
        'percentile_95_error': np.percentile(np.abs(errors), 95),
        'n_samples': len(obs)
    }