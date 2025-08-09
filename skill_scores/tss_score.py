"""
True Skill Statistic (TSS) for binary event prediction
二元事件預測的真實技能統計
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def calculate_tss(observations, predictions, threshold=None):
    """
    計算 True Skill Statistic (TSS)
    TSS = Sensitivity + Specificity - 1 = TPR - FPR
    
    也稱為 Youden's J statistic 或 Peirce Skill Score
    
    Parameters:
    -----------
    observations : array-like
        二元觀測結果 (0 or 1)
    predictions : array-like
        連續預測值或二元預測結果
    threshold : float, optional
        將連續預測轉為二元的閾值，若無則自動選擇最佳閾值
        
    Returns:
    --------
    float
        TSS 值 (-1 to 1, 1為完美預測, 0為無技能)
    """
    
    obs = np.array(observations, dtype=int)
    pred = np.array(predictions)
    
    # 如果預測值是連續的，需要二元化
    if len(np.unique(pred)) > 2:
        if threshold is None:
            # 找到最佳閾值 (最大化 TSS)
            thresholds = np.percentile(pred, np.linspace(10, 90, 81))
            tss_scores = []
            
            for t in thresholds:
                binary_pred = (pred >= t).astype(int)
                if len(np.unique(binary_pred)) > 1:  # 確保不是全部相同
                    tss = _calculate_tss_from_binary(obs, binary_pred)
                    tss_scores.append(tss)
                else:
                    tss_scores.append(-1)
            
            if tss_scores:
                best_idx = np.argmax(tss_scores)
                threshold = thresholds[best_idx]
            else:
                threshold = np.median(pred)
        
        binary_pred = (pred >= threshold).astype(int)
    else:
        binary_pred = pred.astype(int)
    
    return _calculate_tss_from_binary(obs, binary_pred)


def _calculate_tss_from_binary(observations, binary_predictions):
    """計算二元預測的 TSS"""
    
    # 計算混淆矩陣
    tn, fp, fn, tp = confusion_matrix(observations, binary_predictions).ravel()
    
    # 計算 Sensitivity (True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # 計算 Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # TSS = Sensitivity + Specificity - 1
    tss = sensitivity + specificity - 1
    
    return tss


def calculate_tss_skill_score(observations, predictions, baseline_predictions=None, threshold=None):
    """
    計算 TSS Skill Score
    由於 TSS 本身就是技能分數，這裡計算相對於基準線的改善
    
    Parameters:
    -----------
    observations : array-like
        觀測值
    predictions : array-like
        模型預測值
    baseline_predictions : array-like, optional
        基準線預測值
    threshold : float, optional
        二元化閾值
        
    Returns:
    --------
    float
        TSS 改善量 (model_TSS - baseline_TSS)
    """
    
    tss_model = calculate_tss(observations, predictions, threshold)
    
    if baseline_predictions is None:
        # 使用隨機預測作為基準線 (TSS = 0)
        tss_baseline = 0
    else:
        tss_baseline = calculate_tss(observations, baseline_predictions, threshold)
    
    tss_improvement = tss_model - tss_baseline
    
    return tss_improvement


def find_optimal_threshold(observations, predictions, metric='tss'):
    """
    找到最佳二元化閾值
    
    Parameters:
    -----------
    observations : array-like
        二元觀測結果
    predictions : array-like
        連續預測值
    metric : str
        優化指標 ('tss', 'f1', 'youden')
        
    Returns:
    --------
    dict
        最佳閾值和對應的統計量
    """
    
    obs = np.array(observations, dtype=int)
    pred = np.array(predictions)
    
    # 生成候選閾值
    thresholds = np.percentile(pred, np.linspace(5, 95, 91))
    thresholds = np.unique(thresholds)  # 去除重複
    
    best_score = -float('inf')
    best_threshold = np.median(pred)
    best_stats = {}
    
    for threshold in thresholds:
        binary_pred = (pred >= threshold).astype(int)
        
        # 確保預測有變異性
        if len(np.unique(binary_pred)) < 2:
            continue
        
        # 計算統計量
        stats = calculate_binary_classification_stats(obs, binary_pred)
        
        if metric == 'tss':
            score = stats['tss']
        elif metric == 'f1':
            score = stats['f1_score']
        elif metric == 'youden':
            score = stats['sensitivity'] + stats['specificity'] - 1
        else:
            score = stats['tss']  # 默認使用 TSS
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_stats = stats.copy()
            best_stats['threshold'] = threshold
    
    return best_stats


def calculate_binary_classification_stats(observations, binary_predictions):
    """
    計算二元分類的完整統計量
    
    Parameters:
    -----------
    observations : array-like
        二元觀測結果
    binary_predictions : array-like
        二元預測結果
        
    Returns:
    --------
    dict
        分類統計量
    """
    
    obs = np.array(observations, dtype=int)
    pred = np.array(binary_predictions, dtype=int)
    
    # 混淆矩陣
    tn, fp, fn, tp = confusion_matrix(obs, pred).ravel()
    
    # 基本比率
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0     # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0          # Negative Predictive Value
    
    # 計算各種技能分數
    tss = sensitivity + specificity - 1
    hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) > 0 else 0  # Heidke Skill Score
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # 偏差統計
    bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0  # Frequency Bias
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'tss': tss,
        'hss': hss,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'bias': bias,
        'n_positive': tp + fn,
        'n_negative': tn + fp
    }


def analyze_roc_curve(observations, predictions):
    """
    分析 ROC 曲線和相關統計
    
    Parameters:
    -----------
    observations : array-like
        二元觀測結果
    predictions : array-like
        連續預測值 (機率或分數)
        
    Returns:
    --------
    dict
        ROC 分析結果
    """
    
    obs = np.array(observations, dtype=int)
    pred = np.array(predictions)
    
    try:
        # ROC 曲線
        fpr, tpr, thresholds = roc_curve(obs, pred)
        auc = roc_auc_score(obs, pred)
        
        # 找到最佳 TSS 點 (最大 TPR - FPR)
        tss_values = tpr - fpr
        best_idx = np.argmax(tss_values)
        best_threshold = thresholds[best_idx]
        best_tss = tss_values[best_idx]
        best_tpr = tpr[best_idx]
        best_fpr = fpr[best_idx]
        
        return {
            'auc': auc,
            'best_threshold': best_threshold,
            'best_tss': best_tss,
            'best_tpr': best_tpr,
            'best_fpr': best_fpr,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    except ValueError as e:
        # 處理只有一個類別的情況
        return {
            'auc': np.nan,
            'best_threshold': np.nan,
            'best_tss': np.nan,
            'error': str(e)
        }


def convert_payouts_to_binary_events(payouts, damages, method='threshold'):
    """
    將參數型賠付轉換為二元事件預測，用於 TSS 計算
    
    Parameters:
    -----------
    payouts : array-like
        參數型保險賠付
    damages : array-like
        實際損失
    method : str
        轉換方法 ('threshold', 'quantile', 'relative')
        
    Returns:
    --------
    tuple
        (binary_observations, binary_predictions_or_scores)
    """
    
    payouts = np.array(payouts)
    damages = np.array(damages)
    
    # 定義觀測的二元事件 (是否有顯著損失)
    damage_threshold = np.percentile(damages[damages > 0], 50) if np.any(damages > 0) else 0
    binary_obs = (damages > damage_threshold).astype(int)
    
    if method == 'threshold':
        # 簡單閾值法：是否有賠付
        binary_pred = (payouts > 0).astype(int)
        return binary_obs, binary_pred
    
    elif method == 'quantile':
        # 基於分位數的預測分數
        pred_scores = np.zeros_like(payouts)
        if np.max(payouts) > 0:
            pred_scores = payouts / np.max(payouts)  # 標準化到 [0,1]
        return binary_obs, pred_scores
    
    elif method == 'relative':
        # 相對於損失分布的預測分數
        pred_scores = np.zeros_like(payouts)
        max_damage = np.max(damages) if np.max(damages) > 0 else 1
        pred_scores = np.minimum(payouts / max_damage, 1.0)
        return binary_obs, pred_scores
    
    else:
        raise ValueError(f"Unknown method: {method}")