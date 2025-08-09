"""
Skill Score Evaluator
技能評分評估器

Unified skill score evaluation for parametric insurance products.
統一的參數型保險產品技能評分評估。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# Import from skill_scores module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skill_scores import (
    calculate_rmse, calculate_mae,
    calculate_crps, calculate_crps_skill_score,
    calculate_edi, calculate_edi_skill_score,
    calculate_tss, calculate_tss_skill_score,
    calculate_brier_score, calculate_brier_skill_score
)

class SkillScoreType(Enum):
    """技能評分類型"""
    RMSE = "rmse"
    MAE = "mae"
    CRPS = "crps"
    CRPSS = "crpss"
    EDI = "edi"
    EDISS = "ediss"
    TSS = "tss"
    BRIER = "brier"
    BRIESS = "briess"
    CORRELATION = "correlation"

@dataclass
class SkillScoreResult:
    """技能評分結果"""
    score_type: SkillScoreType
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = None

class SkillScoreEvaluator:
    """
    統一技能評分評估器
    
    提供各種技能評分的統一計算介面，支援Bootstrap信賴區間。
    """
    
    def __init__(self, bootstrap_samples: int = 1000, confidence_level: float = 0.95):
        """
        初始化評估器
        
        Parameters:
        -----------
        bootstrap_samples : int
            Bootstrap樣本數
        confidence_level : float
            信賴水準
        """
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
    
    def evaluate_single_score(self, 
                            score_type: SkillScoreType,
                            predictions: np.ndarray,
                            observations: np.ndarray,
                            **kwargs) -> SkillScoreResult:
        """
        計算單一技能評分
        
        Parameters:
        -----------
        score_type : SkillScoreType
            評分類型
        predictions : np.ndarray
            預測值
        observations : np.ndarray
            觀測值
        **kwargs : dict
            額外參數
            
        Returns:
        --------
        SkillScoreResult
            評分結果
        """
        
        # 計算主要分數
        if score_type == SkillScoreType.RMSE:
            score = calculate_rmse(observations, predictions)
        elif score_type == SkillScoreType.MAE:
            score = calculate_mae(observations, predictions)
        elif score_type == SkillScoreType.CRPS:
            # CRPS需要分布信息，這裡使用簡化版本
            score = calculate_crps(predictions, observations)
        elif score_type == SkillScoreType.CORRELATION:
            score = np.corrcoef(observations, predictions)[0, 1] if len(observations) > 1 else 0.0
        else:
            raise ValueError(f"不支援的評分類型: {score_type}")
        
        # 計算Bootstrap信賴區間
        ci = self._calculate_bootstrap_ci(score_type, predictions, observations)
        
        return SkillScoreResult(
            score_type=score_type,
            value=score,
            confidence_interval=ci,
            metadata=kwargs
        )
    
    def evaluate_multiple_scores(self,
                                score_types: List[SkillScoreType],
                                predictions: np.ndarray,
                                observations: np.ndarray,
                                **kwargs) -> Dict[SkillScoreType, SkillScoreResult]:
        """
        計算多個技能評分
        
        Parameters:
        -----------
        score_types : List[SkillScoreType]
            評分類型列表
        predictions : np.ndarray
            預測值
        observations : np.ndarray
            觀測值
        **kwargs : dict
            額外參數
            
        Returns:
        --------
        Dict[SkillScoreType, SkillScoreResult]
            評分結果字典
        """
        
        results = {}
        for score_type in score_types:
            results[score_type] = self.evaluate_single_score(
                score_type, predictions, observations, **kwargs
            )
        
        return results
    
    def evaluate_products(self,
                         products: List[Dict[str, Any]],
                         observations: np.ndarray,
                         score_types: List[SkillScoreType] = None) -> pd.DataFrame:
        """
        評估多個產品的技能評分
        
        Parameters:
        -----------
        products : List[Dict[str, Any]]
            產品列表，每個產品包含預測值
        observations : np.ndarray
            觀測值
        score_types : List[SkillScoreType], optional
            評分類型列表
            
        Returns:
        --------
        pd.DataFrame
            產品評分結果表
        """
        
        if score_types is None:
            score_types = [SkillScoreType.RMSE, SkillScoreType.MAE, 
                          SkillScoreType.CORRELATION]
        
        results = []
        
        for product in products:
            product_id = product.get('product_id', 'unknown')
            predictions = product.get('payouts', product.get('predictions', []))
            
            if len(predictions) == 0:
                continue
                
            product_scores = self.evaluate_multiple_scores(
                score_types, np.array(predictions), observations
            )
            
            row = {'product_id': product_id}
            for score_type, result in product_scores.items():
                row[score_type.value] = result.value
                if result.confidence_interval:
                    row[f'{score_type.value}_ci_lower'] = result.confidence_interval[0]
                    row[f'{score_type.value}_ci_upper'] = result.confidence_interval[1]
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def _calculate_bootstrap_ci(self,
                              score_type: SkillScoreType,
                              predictions: np.ndarray,
                              observations: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        計算Bootstrap信賴區間
        
        Parameters:
        -----------
        score_type : SkillScoreType
            評分類型
        predictions : np.ndarray
            預測值
        observations : np.ndarray
            觀測值
            
        Returns:
        --------
        Optional[Tuple[float, float]]
            信賴區間 (下界, 上界)
        """
        
        try:
            n = len(observations)
            bootstrap_scores = []
            
            for _ in range(self.bootstrap_samples):
                # Bootstrap重新抽樣
                indices = np.random.choice(n, size=n, replace=True)
                boot_pred = predictions[indices]
                boot_obs = observations[indices]
                
                # 計算評分
                if score_type == SkillScoreType.RMSE:
                    score = calculate_rmse(boot_obs, boot_pred)
                elif score_type == SkillScoreType.MAE:
                    score = calculate_mae(boot_obs, boot_pred)
                elif score_type == SkillScoreType.CORRELATION:
                    score = np.corrcoef(boot_obs, boot_pred)[0, 1] if len(boot_obs) > 1 else 0.0
                else:
                    continue
                
                bootstrap_scores.append(score)
            
            if len(bootstrap_scores) > 0:
                alpha = 1 - self.confidence_level
                lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
                upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
                return (lower, upper)
            
        except Exception:
            pass
        
        return None
    
    def compare_products(self,
                        products_a: List[Dict[str, Any]],
                        products_b: List[Dict[str, Any]],
                        observations: np.ndarray,
                        score_types: List[SkillScoreType] = None) -> Dict[str, Any]:
        """
        比較兩組產品的技能評分
        
        Parameters:
        -----------
        products_a : List[Dict[str, Any]]
            第一組產品
        products_b : List[Dict[str, Any]]
            第二組產品
        observations : np.ndarray
            觀測值
        score_types : List[SkillScoreType], optional
            評分類型列表
            
        Returns:
        --------
        Dict[str, Any]
            比較結果
        """
        
        if score_types is None:
            score_types = [SkillScoreType.RMSE, SkillScoreType.MAE]
        
        results_a = self.evaluate_products(products_a, observations, score_types)
        results_b = self.evaluate_products(products_b, observations, score_types)
        
        comparison = {
            'group_a_stats': {},
            'group_b_stats': {},
            'improvement': {},
            'significance_test': {}
        }
        
        for score_type in score_types:
            score_col = score_type.value
            
            if score_col in results_a.columns and score_col in results_b.columns:
                scores_a = results_a[score_col].dropna()
                scores_b = results_b[score_col].dropna()
                
                comparison['group_a_stats'][score_col] = {
                    'mean': scores_a.mean(),
                    'std': scores_a.std(),
                    'best': scores_a.min() if score_type in [SkillScoreType.RMSE, SkillScoreType.MAE] else scores_a.max()
                }
                
                comparison['group_b_stats'][score_col] = {
                    'mean': scores_b.mean(),
                    'std': scores_b.std(),
                    'best': scores_b.min() if score_type in [SkillScoreType.RMSE, SkillScoreType.MAE] else scores_b.max()
                }
                
                # 計算改善百分比
                if score_type in [SkillScoreType.RMSE, SkillScoreType.MAE]:
                    improvement = (scores_a.mean() - scores_b.mean()) / scores_a.mean() * 100
                else:
                    improvement = (scores_b.mean() - scores_a.mean()) / scores_a.mean() * 100
                
                comparison['improvement'][score_col] = improvement
        
        return comparison