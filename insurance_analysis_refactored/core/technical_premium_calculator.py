#!/usr/bin/env python3
"""
Technical Premium Calculator Module
技術保費計算器模組

專門負責進階技術保費計算，包含:
- 期望賠付計算
- 風險資本要求 (Solvency II framework)
- VaR 計算
- 完整保費分解

Author: Research Team
Date: 2025-01-10
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import norm, gamma
from abc import ABC, abstractmethod

# 導入基礎數據結構
from .parametric_engine import ParametricProduct


@dataclass
class TechnicalPremiumConfig:
    """技術保費計算配置"""
    risk_free_rate: float = 0.02         # 無風險利率
    risk_loading_factor: float = 0.20     # 風險載入係數
    solvency_ratio: float = 1.25          # 償付能力比率
    expense_ratio: float = 0.15           # 費用率
    profit_margin: float = 0.10           # 利潤率
    confidence_level: float = 0.995       # 信心水準 (VaR計算)
    monte_carlo_samples: int = 10000      # 蒙特卡羅樣本數


@dataclass
class TechnicalPremiumResult:
    """技術保費計算結果"""
    expected_payout: float
    payout_std: float
    risk_capital: float
    risk_loading: float
    net_risk_premium: float
    expense_loading: float
    gross_premium: float
    profit_loading: float
    technical_premium: float
    loss_ratio: float
    expense_ratio: float
    profit_ratio: float
    combined_ratio: float
    value_at_risk: float
    regulatory_capital: float


class ExpectedPayoutCalculator(ABC):
    """期望賠付計算器抽象基類"""
    
    @abstractmethod
    def calculate_expected_payout(self, 
                                product_params: ParametricProduct,
                                hazard_indices: np.ndarray,
                                monte_carlo_samples: int = 10000) -> Tuple[float, float]:
        """計算期望賠付和標準差"""
        pass


class GammaDistributionPayoutCalculator(ExpectedPayoutCalculator):
    """使用Gamma分佈擬合的期望賠付計算器"""
    
    def calculate_expected_payout(self, 
                                product_params: ParametricProduct,
                                hazard_indices: np.ndarray,
                                monte_carlo_samples: int = 10000) -> Tuple[float, float]:
        """
        使用Gamma分佈擬合災害指標，計算期望賠付
        
        Parameters:
        -----------
        product_params : ParametricProduct
            產品參數
        hazard_indices : np.ndarray
            歷史災害指標
        monte_carlo_samples : int
            蒙特卡羅樣本數
            
        Returns:
        --------
        Tuple[float, float]
            (期望賠付, 賠付標準差)
        """
        # 擬合災害分佈
        if len(hazard_indices) > 0:
            # 使用 Gamma 分佈擬合災害指標
            shape, loc, scale = gamma.fit(hazard_indices, floc=0)
            
            # 蒙特卡羅模擬
            simulated_indices = gamma.rvs(shape, loc=loc, scale=scale, 
                                        size=monte_carlo_samples)
        else:
            # 如果沒有歷史數據，使用默認分佈
            simulated_indices = np.random.gamma(2, 25, monte_carlo_samples)
        
        # 計算賠付 - 支持多閾值產品
        payouts = np.zeros(len(simulated_indices))
        
        for i, index_value in enumerate(simulated_indices):
            # 階梯式賠付邏輯
            for j, threshold in enumerate(product_params.trigger_thresholds):
                if index_value >= threshold:
                    payouts[i] = product_params.payout_amounts[j]
        
        # 應用賠付限額
        payouts = np.minimum(payouts, product_params.max_payout)
        
        expected_payout = np.mean(payouts)
        payout_std = np.std(payouts)
        
        return expected_payout, payout_std


class SolvencyIIRiskCapitalCalculator:
    """基於Solvency II框架的風險資本計算器"""
    
    def __init__(self, config: TechnicalPremiumConfig):
        """
        初始化風險資本計算器
        
        Parameters:
        -----------
        config : TechnicalPremiumConfig
            技術保費計算配置
        """
        self.config = config
    
    def calculate_value_at_risk(self, 
                              expected_payout: float, 
                              payout_std: float) -> float:
        """
        計算風險價值 (VaR)
        
        Parameters:
        -----------
        expected_payout : float
            期望賠付
        payout_std : float
            賠付標準差
            
        Returns:
        --------
        float
            風險價值
        """
        # VaR 計算 (假設賠付為正態分佈)
        var_level = norm.ppf(self.config.confidence_level)
        value_at_risk = expected_payout + var_level * payout_std
        
        return max(0, value_at_risk)  # 確保非負
    
    def calculate_regulatory_capital(self, value_at_risk: float) -> float:
        """
        計算監管資本要求
        
        Parameters:
        -----------
        value_at_risk : float
            風險價值
            
        Returns:
        --------
        float
            監管資本要求
        """
        return value_at_risk * self.config.solvency_ratio
    
    def calculate_risk_capital(self, 
                             expected_payout: float, 
                             payout_std: float) -> Tuple[float, float, float]:
        """
        計算風險資本要求
        
        基於 Solvency II 框架的簡化版本
        
        Parameters:
        -----------
        expected_payout : float
            期望賠付
        payout_std : float
            賠付標準差
            
        Returns:
        --------
        Tuple[float, float, float]
            (VaR, 監管資本, 風險資本)
        """
        # 1. 計算 VaR
        value_at_risk = self.calculate_value_at_risk(expected_payout, payout_std)
        
        # 2. 監管資本 = VaR × 償付能力比率
        regulatory_capital = self.calculate_regulatory_capital(value_at_risk)
        
        # 3. 風險資本 = 監管資本 - 期望賠付
        risk_capital = max(0, regulatory_capital - expected_payout)
        
        return value_at_risk, regulatory_capital, risk_capital


class TechnicalPremiumCalculator:
    """進階技術保費計算器"""
    
    def __init__(self, 
                 config: TechnicalPremiumConfig,
                 payout_calculator: Optional[ExpectedPayoutCalculator] = None):
        """
        初始化技術保費計算器
        
        Parameters:
        -----------
        config : TechnicalPremiumConfig
            技術保費計算配置
        payout_calculator : ExpectedPayoutCalculator, optional
            期望賠付計算器，默認使用Gamma分佈計算器
        """
        self.config = config
        self.payout_calculator = payout_calculator or GammaDistributionPayoutCalculator()
        self.risk_capital_calculator = SolvencyIIRiskCapitalCalculator(config)
    
    def calculate_technical_premium(self,
                                  product_params: ParametricProduct,
                                  hazard_indices: np.ndarray) -> TechnicalPremiumResult:
        """
        計算技術保費的完整分解
        
        Technical Premium = Expected Payout + Risk Loading + Expenses + Profit
        
        Parameters:
        -----------
        product_params : ParametricProduct
            產品參數
        hazard_indices : np.ndarray
            災害指標數據
            
        Returns:
        --------
        TechnicalPremiumResult
            技術保費詳細分解
        """
        # 1. 計算期望賠付和變異性
        expected_payout, payout_std = self.payout_calculator.calculate_expected_payout(
            product_params, hazard_indices, self.config.monte_carlo_samples
        )
        
        # 2. 計算風險資本
        value_at_risk, regulatory_capital, risk_capital = \
            self.risk_capital_calculator.calculate_risk_capital(expected_payout, payout_std)
        
        # 3. 風險載入 (Risk Loading)
        risk_loading = (risk_capital * self.config.risk_free_rate + 
                       expected_payout * self.config.risk_loading_factor)
        
        # 4. 淨風險保費
        net_risk_premium = expected_payout + risk_loading
        
        # 5. 費用載入
        expense_loading = net_risk_premium * self.config.expense_ratio / (1 - self.config.expense_ratio)
        
        # 6. 毛保費
        gross_premium = net_risk_premium + expense_loading
        
        # 7. 利潤載入
        profit_loading = gross_premium * self.config.profit_margin / (1 - self.config.profit_margin)
        
        # 8. 技術保費 (最終保費)
        technical_premium = gross_premium + profit_loading
        
        # 9. 計算各項比率
        loss_ratio = expected_payout / technical_premium if technical_premium > 0 else 0
        expense_ratio_actual = expense_loading / technical_premium if technical_premium > 0 else 0
        profit_ratio = profit_loading / technical_premium if technical_premium > 0 else 0
        combined_ratio = (expected_payout + expense_loading) / technical_premium if technical_premium > 0 else 0
        
        return TechnicalPremiumResult(
            expected_payout=expected_payout,
            payout_std=payout_std,
            risk_capital=risk_capital,
            risk_loading=risk_loading,
            net_risk_premium=net_risk_premium,
            expense_loading=expense_loading,
            gross_premium=gross_premium,
            profit_loading=profit_loading,
            technical_premium=technical_premium,
            loss_ratio=loss_ratio,
            expense_ratio=expense_ratio_actual,
            profit_ratio=profit_ratio,
            combined_ratio=combined_ratio,
            value_at_risk=value_at_risk,
            regulatory_capital=regulatory_capital
        )
    
    def calculate_premium_breakdown_summary(self, result: TechnicalPremiumResult) -> Dict[str, float]:
        """
        獲取保費分解摘要
        
        Parameters:
        -----------
        result : TechnicalPremiumResult
            技術保費計算結果
            
        Returns:
        --------
        Dict[str, float]
            保費分解摘要
        """
        return {
            'expected_payout': result.expected_payout,
            'payout_std': result.payout_std,
            'risk_capital': result.risk_capital,
            'risk_loading': result.risk_loading,
            'net_risk_premium': result.net_risk_premium,
            'expense_loading': result.expense_loading,
            'gross_premium': result.gross_premium,
            'profit_loading': result.profit_loading,
            'technical_premium': result.technical_premium,
            'loss_ratio': result.loss_ratio,
            'expense_ratio': result.expense_ratio,
            'profit_ratio': result.profit_ratio,
            'combined_ratio': result.combined_ratio,
            'value_at_risk': result.value_at_risk,
            'regulatory_capital': result.regulatory_capital
        }


def create_standard_technical_premium_calculator(
        risk_free_rate: float = 0.02,
        risk_loading_factor: float = 0.20,
        solvency_ratio: float = 1.25,
        expense_ratio: float = 0.15,
        profit_margin: float = 0.10,
        confidence_level: float = 0.995) -> TechnicalPremiumCalculator:
    """
    創建標準技術保費計算器
    
    Parameters:
    -----------
    risk_free_rate : float
        無風險利率
    risk_loading_factor : float
        風險載入係數
    solvency_ratio : float
        償付能力比率
    expense_ratio : float
        費用率
    profit_margin : float
        利潤率
    confidence_level : float
        VaR信心水準
        
    Returns:
    --------
    TechnicalPremiumCalculator
        標準技術保費計算器
    """
    config = TechnicalPremiumConfig(
        risk_free_rate=risk_free_rate,
        risk_loading_factor=risk_loading_factor,
        solvency_ratio=solvency_ratio,
        expense_ratio=expense_ratio,
        profit_margin=profit_margin,
        confidence_level=confidence_level
    )
    
    return TechnicalPremiumCalculator(config)