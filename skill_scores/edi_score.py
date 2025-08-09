"""
Extreme Dependence Index (EDI) for extreme event prediction skill
極端事件預測技能的極端依賴指數
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def calculate_edi(observations, predictions, extreme_threshold_obs=90, extreme_threshold_pred=90):
    """
    計算極端依賴指數 (EDI)
    
    EDI 衡量模型在極端事件上的預測技能
    EDI = P(Y > y_extreme | X > x_extreme) - P(Y > y_extreme)
    
    Parameters:
    -----------
    observations : array-like
        觀測值 (實際損失)
    predictions : array-like
        預測值 (參數型賠付)
    extreme_threshold_obs : float
        觀測值的極端事件閾值 (百分位數)
    extreme_threshold_pred : float
        預測值的極端事件閾值 (百分位數)
        
    Returns:
    --------
    float
        EDI 值 (0 to 1, 越高表示極端事件預測技能越好)
    """
    
    obs = np.array(observations)
    pred = np.array(predictions)
    
    # 計算極端事件閾值
    obs_threshold = np.percentile(obs[obs > 0], extreme_threshold_obs) if np.any(obs > 0) else 0
    pred_threshold = np.percentile(pred[pred > 0], extreme_threshold_pred) if np.any(pred > 0) else 0
    
    # 定義極端事件
    extreme_obs = obs > obs_threshold
    extreme_pred = pred > pred_threshold
    
    if np.sum(extreme_pred) == 0:
        return 0.0  # 無法計算條件機率
    
    # 條件機率: P(Y > y_extreme | X > x_extreme)
    conditional_prob = np.sum(extreme_obs & extreme_pred) / np.sum(extreme_pred)
    
    # 邊際機率: P(Y > y_extreme)
    marginal_prob = np.mean(extreme_obs)
    
    # EDI
    edi = conditional_prob - marginal_prob
    
    return edi


def calculate_edi_skill_score(observations, predictions, baseline_predictions=None, 
                             extreme_threshold=90):
    """
    計算 EDI Skill Score
    EDI-SS = (EDI_model - EDI_baseline) / (EDI_perfect - EDI_baseline)
    
    Parameters:
    -----------
    observations : array-like
        觀測值
    predictions : array-like
        模型預測值
    baseline_predictions : array-like, optional
        基準線預測值
    extreme_threshold : float
        極端事件閾值 (百分位數)
        
    Returns:
    --------
    float
        EDI Skill Score
    """
    
    # 模型 EDI
    edi_model = calculate_edi(observations, predictions, extreme_threshold, extreme_threshold)
    
    if baseline_predictions is None:
        # 使用隨機基準線 (無技能預測)
        baseline_predictions = np.random.permutation(predictions)
    
    # 基準線 EDI
    edi_baseline = calculate_edi(observations, baseline_predictions, extreme_threshold, extreme_threshold)
    
    # 完美預測的 EDI (預測 = 觀測)
    edi_perfect = calculate_edi(observations, observations, extreme_threshold, extreme_threshold)
    
    if edi_perfect == edi_baseline:
        return 0.0
    
    edi_ss = (edi_model - edi_baseline) / (edi_perfect - edi_baseline)
    
    return edi_ss


def calculate_extreme_hit_rate(observations, predictions, extreme_threshold=90):
    """
    計算極端事件命中率
    
    Parameters:
    -----------
    observations : array-like
        觀測值
    predictions : array-like
        預測值
    extreme_threshold : float
        極端事件閾值 (百分位數)
        
    Returns:
    --------
    dict
        極端事件統計
    """
    
    obs = np.array(observations)
    pred = np.array(predictions)
    
    # 定義極端事件閾值
    obs_threshold = np.percentile(obs[obs > 0], extreme_threshold) if np.any(obs > 0) else 0
    pred_threshold = np.percentile(pred[pred > 0], extreme_threshold) if np.any(pred > 0) else 0
    
    # 極端事件標識
    extreme_obs = obs > obs_threshold
    extreme_pred = pred > pred_threshold
    
    # 混淆矩陣
    true_positive = np.sum(extreme_obs & extreme_pred)    # 命中
    false_positive = np.sum(~extreme_obs & extreme_pred)  # 虛警
    true_negative = np.sum(~extreme_obs & ~extreme_pred)  # 正確拒絕
    false_negative = np.sum(extreme_obs & ~extreme_pred)  # 漏報
    
    # 計算各種指標
    total = len(obs)
    
    # 命中率 (Hit Rate / Sensitivity / Recall)
    hit_rate = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    
    # 虛警率 (False Alarm Rate)
    false_alarm_rate = false_positive / (false_positive + true_negative) if (false_positive + true_negative) > 0 else 0
    
    # 準確率 (Precision)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    
    # F1 分數
    f1_score = 2 * (precision * hit_rate) / (precision + hit_rate) if (precision + hit_rate) > 0 else 0
    
    return {
        'hit_rate': hit_rate,
        'false_alarm_rate': false_alarm_rate,
        'precision': precision,
        'f1_score': f1_score,
        'true_positive': true_positive,
        'false_positive': false_positive,
        'true_negative': true_negative,
        'false_negative': false_negative,
        'extreme_obs_count': np.sum(extreme_obs),
        'extreme_pred_count': np.sum(extreme_pred),
        'obs_threshold': obs_threshold,
        'pred_threshold': pred_threshold
    }


def analyze_extreme_events_dependence(observations, predictions, thresholds=[75, 90, 95, 99]):
    """
    分析不同閾值下的極端事件依賴性
    
    Parameters:
    -----------
    observations : array-like
        觀測值
    predictions : array-like
        預測值
    thresholds : list
        要分析的百分位閾值
        
    Returns:
    --------
    dict
        不同閾值下的 EDI 和相關統計
    """
    
    results = {}
    
    for threshold in thresholds:
        edi = calculate_edi(observations, predictions, threshold, threshold)
        hit_stats = calculate_extreme_hit_rate(observations, predictions, threshold)
        
        results[f'threshold_{threshold}'] = {
            'edi': edi,
            'hit_rate': hit_stats['hit_rate'],
            'false_alarm_rate': hit_stats['false_alarm_rate'],
            'precision': hit_stats['precision'],
            'f1_score': hit_stats['f1_score'],
            'extreme_obs_count': hit_stats['extreme_obs_count'],
            'extreme_pred_count': hit_stats['extreme_pred_count']
        }
    
    return results


def calculate_extremal_correlation(observations, predictions, extreme_threshold=90):
    """
    計算極端值間的相關性
    
    Parameters:
    -----------
    observations : array-like
        觀測值
    predictions : array-like
        預測值
    extreme_threshold : float
        極端事件閾值 (百分位數)
        
    Returns:
    --------
    dict
        極端值相關性分析
    """
    
    obs = np.array(observations)
    pred = np.array(predictions)
    
    # 提取極端值
    obs_threshold = np.percentile(obs[obs > 0], extreme_threshold) if np.any(obs > 0) else 0
    pred_threshold = np.percentile(pred[pred > 0], extreme_threshold) if np.any(pred > 0) else 0
    
    extreme_obs_mask = obs > obs_threshold
    extreme_pred_mask = pred > pred_threshold
    
    # 只保留至少一個是極端值的情況
    extreme_mask = extreme_obs_mask | extreme_pred_mask
    
    if np.sum(extreme_mask) < 3:  # 需要足夠的樣本計算相關性
        return {
            'pearson_r': np.nan,
            'spearman_r': np.nan,
            'n_extreme_pairs': np.sum(extreme_mask)
        }
    
    extreme_obs = obs[extreme_mask]
    extreme_pred = pred[extreme_mask]
    
    # 計算相關性
    pearson_r, pearson_p = pearsonr(extreme_obs, extreme_pred)
    spearman_r, spearman_p = spearmanr(extreme_obs, extreme_pred)
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n_extreme_pairs': np.sum(extreme_mask),
        'extreme_obs_mean': np.mean(extreme_obs),
        'extreme_pred_mean': np.mean(extreme_pred)
    }