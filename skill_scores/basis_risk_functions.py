"""
Basis Risk Functions for Parametric Insurance
參數型保險基差風險函數

提供各種基差風險的數學定義和計算方法，從 bayesian_decision_theory.py 遷移而來
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

class BasisRiskType(Enum):
    """基差風險類型"""
    ABSOLUTE = "absolute"
    ASYMMETRIC_UNDER = "asymmetric_under"
    WEIGHTED_ASYMMETRIC = "weighted_asymmetric"
    QUADRATIC = "quadratic"
    
@dataclass 
class BasisRiskLossFunction:
    """基差風險損失函數"""
    risk_type: BasisRiskType
    w_under: float = 1.0     # 賠不夠的懲罰權重
    w_over: float = 0.3      # 賠多了的懲罰權重
    
    def calculate_loss(self, actual_loss: float, payout: float) -> float:
        """計算基差風險損失"""
        
        if self.risk_type == BasisRiskType.ABSOLUTE:
            # 絕對基差風險
            return abs(actual_loss - payout)
            
        elif self.risk_type == BasisRiskType.ASYMMETRIC_UNDER:
            # 不對稱基差風險 (只懲罰賠不夠)
            return max(0, actual_loss - payout)
            
        elif self.risk_type == BasisRiskType.WEIGHTED_ASYMMETRIC:
            # 加權不對稱基差風險
            under_coverage = max(0, actual_loss - payout)
            over_coverage = max(0, payout - actual_loss)
            return self.w_under * under_coverage + self.w_over * over_coverage
            
        elif self.risk_type == BasisRiskType.QUADRATIC:
            # 二次基差風險
            return (actual_loss - payout) ** 2
            
        else:
            raise ValueError(f"Unsupported risk type: {self.risk_type}")

class BasisRiskCalculator:
    """基差風險計算器"""
    
    def __init__(self):
        """初始化基差風險計算器"""
        pass
    
    @staticmethod
    def calculate_absolute_basis_risk(actual_loss: float, payout: float) -> float:
        """絕對基差風險: |actual - payout|"""
        return abs(actual_loss - payout)
    
    @staticmethod 
    def calculate_asymmetric_basis_risk(actual_loss: float, payout: float) -> float:
        """不對稱基差風險: max(0, actual - payout)"""
        return max(0, actual_loss - payout)
    
    @staticmethod
    def calculate_weighted_asymmetric_basis_risk(
        actual_loss: float, 
        payout: float,
        w_under: float = 2.0,
        w_over: float = 0.5
    ) -> float:
        """加權不對稱基差風險"""
        under_coverage = max(0, actual_loss - payout)
        over_coverage = max(0, payout - actual_loss)
        return w_under * under_coverage + w_over * over_coverage
    
    @staticmethod
    def calculate_quadratic_basis_risk(actual_loss: float, payout: float) -> float:
        """二次基差風險: (actual - payout)²"""
        return (actual_loss - payout) ** 2
    
    def calculate_portfolio_basis_risk(self,
                                     actual_losses: np.ndarray,
                                     payouts: np.ndarray,
                                     risk_type: BasisRiskType = BasisRiskType.WEIGHTED_ASYMMETRIC,
                                     **risk_params) -> Dict[str, float]:
        """
        計算投資組合基差風險統計
        
        Parameters:
        -----------
        actual_losses : np.ndarray
            真實損失陣列
        payouts : np.ndarray
            賠付陣列
        risk_type : BasisRiskType
            基差風險類型
        **risk_params : dict
            風險參數 (如 w_under, w_over)
            
        Returns:
        --------
        Dict[str, float]
            基差風險統計結果
        """
        
        if len(actual_losses) != len(payouts):
            raise ValueError("actual_losses 和 payouts 長度必須相等")
        
        # 計算個別基差風險
        individual_risks = []
        
        for actual, payout in zip(actual_losses, payouts):
            if risk_type == BasisRiskType.ABSOLUTE:
                risk = self.calculate_absolute_basis_risk(actual, payout)
            elif risk_type == BasisRiskType.ASYMMETRIC_UNDER:
                risk = self.calculate_asymmetric_basis_risk(actual, payout)
            elif risk_type == BasisRiskType.WEIGHTED_ASYMMETRIC:
                risk = self.calculate_weighted_asymmetric_basis_risk(
                    actual, payout, 
                    w_under=risk_params.get('w_under', 2.0),
                    w_over=risk_params.get('w_over', 0.5)
                )
            elif risk_type == BasisRiskType.QUADRATIC:
                risk = self.calculate_quadratic_basis_risk(actual, payout)
            else:
                raise ValueError(f"Unsupported risk type: {risk_type}")
            
            individual_risks.append(risk)
        
        individual_risks = np.array(individual_risks)
        
        # 計算統計指標
        stats = {
            'mean_basis_risk': float(np.mean(individual_risks)),
            'median_basis_risk': float(np.median(individual_risks)),
            'std_basis_risk': float(np.std(individual_risks)),
            'max_basis_risk': float(np.max(individual_risks)),
            'min_basis_risk': float(np.min(individual_risks)),
            'total_basis_risk': float(np.sum(individual_risks)),
            'basis_risk_95th_percentile': float(np.percentile(individual_risks, 95)),
            'basis_risk_5th_percentile': float(np.percentile(individual_risks, 5))
        }
        
        return stats
    
    def analyze_coverage_breakdown(self,
                                 actual_losses: np.ndarray,
                                 payouts: np.ndarray) -> Dict[str, Any]:
        """
        分析保險覆蓋情況分解
        
        Returns:
        --------
        Dict[str, Any]
            覆蓋情況分析結果
        """
        
        n_total = len(actual_losses)
        
        # 分類事件
        perfect_match = np.isclose(actual_losses, payouts, rtol=1e-3)
        under_coverage = actual_losses > payouts
        over_coverage = actual_losses < payouts
        no_loss_no_payout = (actual_losses == 0) & (payouts == 0)
        loss_no_payout = (actual_losses > 0) & (payouts == 0)
        no_loss_payout = (actual_losses == 0) & (payouts > 0)
        
        # 計算覆蓋統計
        under_coverage_amount = np.sum(np.maximum(0, actual_losses - payouts))
        over_coverage_amount = np.sum(np.maximum(0, payouts - actual_losses))
        
        breakdown = {
            'total_events': n_total,
            'perfect_match_count': int(np.sum(perfect_match)),
            'under_coverage_count': int(np.sum(under_coverage)),
            'over_coverage_count': int(np.sum(over_coverage)),
            'no_loss_no_payout_count': int(np.sum(no_loss_no_payout)),
            'loss_no_payout_count': int(np.sum(loss_no_payout)),
            'no_loss_payout_count': int(np.sum(no_loss_payout)),
            
            'perfect_match_rate': float(np.mean(perfect_match)),
            'under_coverage_rate': float(np.mean(under_coverage)),
            'over_coverage_rate': float(np.mean(over_coverage)),
            'trigger_rate': float(np.mean(payouts > 0)),
            'loss_rate': float(np.mean(actual_losses > 0)),
            
            'total_under_coverage_amount': float(under_coverage_amount),
            'total_over_coverage_amount': float(over_coverage_amount),
            'average_under_coverage': float(under_coverage_amount / max(1, np.sum(under_coverage))),
            'average_over_coverage': float(over_coverage_amount / max(1, np.sum(over_coverage))),
            
            'coverage_efficiency': float(1 - (under_coverage_amount + over_coverage_amount) / max(1, np.sum(actual_losses)))
        }
        
        return breakdown

def create_basis_risk_function(risk_type: str = "weighted_asymmetric",
                             w_under: float = 2.0,
                             w_over: float = 0.5) -> BasisRiskLossFunction:
    """
    創建基差風險損失函數的便利函數
    
    Parameters:
    -----------
    risk_type : str
        風險類型 ("absolute", "asymmetric_under", "weighted_asymmetric", "quadratic")
    w_under : float
        賠不夠的懲罰權重
    w_over : float
        賠多了的懲罰權重
        
    Returns:
    --------
    BasisRiskLossFunction
        基差風險損失函數
    """
    
    risk_type_enum = BasisRiskType(risk_type)
    
    return BasisRiskLossFunction(
        risk_type=risk_type_enum,
        w_under=w_under,
        w_over=w_over
    )