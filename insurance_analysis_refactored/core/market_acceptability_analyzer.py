#!/usr/bin/env python3
"""
Market Acceptability Analyzer Module
市場接受度分析器模組

專門負責分析參數保險產品的市場接受度，包含:
- 產品複雜度評估
- 觸發頻率吸引力分析
- 保費可負擔性評估
- 綜合市場接受度評分

Author: Research Team
Date: 2025-01-10
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# 導入基礎數據結構
from .parametric_engine import ParametricProduct


class ProductComplexityLevel(Enum):
    """產品複雜度等級"""
    VERY_LOW = "very_low"      # 非常簡單
    LOW = "low"                # 簡單
    MEDIUM = "medium"          # 中等
    HIGH = "high"              # 複雜
    VERY_HIGH = "very_high"    # 非常複雜


@dataclass
class MarketAcceptabilityConfig:
    """市場接受度分析配置"""
    optimal_trigger_rate: float = 0.20           # 最佳觸發率
    trigger_rate_tolerance: float = 0.15         # 觸發率容忍度
    market_premium_benchmark: float = 1.5        # 市場保費基準比率
    complexity_weight: float = 0.3               # 複雜度權重
    frequency_weight: float = 0.4                # 頻率權重  
    affordability_weight: float = 0.3            # 可負擔性權重


@dataclass
class MarketAcceptabilityResult:
    """市場接受度分析結果"""
    complexity_score: float              # 複雜度評分 (0-1, 越低越複雜)
    simplicity_score: float              # 簡單度評分 (1-複雜度)
    frequency_appeal: float              # 頻率吸引力 (0-1, 越高越好)
    affordability: float                 # 可負擔性 (0-1, 越高越好)  
    overall_acceptability: float         # 綜合接受度 (0-1, 越高越好)
    complexity_level: ProductComplexityLevel  # 複雜度等級
    trigger_rate_deviation: float       # 觸發率偏離度
    premium_ratio: float                # 保費比率


class ProductComplexityEvaluator(ABC):
    """產品複雜度評估器抽象基類"""
    
    @abstractmethod
    def evaluate_complexity(self, product: ParametricProduct) -> Tuple[float, ProductComplexityLevel]:
        """評估產品複雜度"""
        pass


class StandardComplexityEvaluator(ProductComplexityEvaluator):
    """標準複雜度評估器"""
    
    def evaluate_complexity(self, product: ParametricProduct) -> Tuple[float, ProductComplexityLevel]:
        """
        根據產品類型和觸發條件數量評估複雜度
        
        Parameters:
        -----------
        product : ParametricProduct
            保險產品
            
        Returns:
        --------
        Tuple[float, ProductComplexityLevel]
            (複雜度評分, 複雜度等級)
        """
        # 基礎複雜度評分
        base_complexity = 0.0
        
        # 根據觸發閾值數量計算複雜度
        num_thresholds = len(product.trigger_thresholds)
        
        if num_thresholds == 1:
            base_complexity = 0.1  # 單一觸發條件，複雜度很低
            level = ProductComplexityLevel.VERY_LOW
        elif num_thresholds == 2:
            base_complexity = 0.25  # 雙重觸發條件，低複雜度
            level = ProductComplexityLevel.LOW
        elif num_thresholds == 3:
            base_complexity = 0.4   # 三重觸發條件，中等複雜度
            level = ProductComplexityLevel.MEDIUM
        elif num_thresholds == 4:
            base_complexity = 0.6   # 四重觸發條件，較高複雜度
            level = ProductComplexityLevel.HIGH
        else:
            base_complexity = 0.8   # 更多觸發條件，非常複雜
            level = ProductComplexityLevel.VERY_HIGH
        
        # 根據賠付函數類型調整複雜度
        payout_type_adjustment = {
            "step": 0.0,        # 階梯函數最簡單
            "linear": 0.1,      # 線性函數較簡單
            "exponential": 0.2, # 指數函數較複雜
            "sigmoid": 0.2      # Sigmoid函數較複雜
        }
        
        payout_complexity = payout_type_adjustment.get(
            product.payout_function_type.value if hasattr(product.payout_function_type, 'value') else str(product.payout_function_type),
            0.1
        )
        
        # 最終複雜度評分
        final_complexity = min(1.0, base_complexity + payout_complexity)
        
        return final_complexity, level


class TriggerFrequencyAnalyzer:
    """觸發頻率分析器"""
    
    def __init__(self, config: MarketAcceptabilityConfig):
        """
        初始化觸發頻率分析器
        
        Parameters:
        -----------
        config : MarketAcceptabilityConfig
            市場接受度分析配置
        """
        self.config = config
    
    def calculate_trigger_frequency_appeal(self, trigger_rate: float) -> Tuple[float, float]:
        """
        計算觸發頻率吸引力
        
        太高或太低的觸發率都會降低市場接受度
        最佳觸發率約在配置的最佳範圍內
        
        Parameters:
        -----------
        trigger_rate : float
            觸發率 (0-1)
            
        Returns:
        --------
        Tuple[float, float]
            (頻率吸引力評分, 觸發率偏離度)
        """
        optimal_rate = self.config.optimal_trigger_rate
        tolerance = self.config.trigger_rate_tolerance
        
        # 計算偏離度
        rate_deviation = abs(trigger_rate - optimal_rate)
        
        # 使用高斯函數模擬市場偏好
        appeal = np.exp(-0.5 * (rate_deviation / tolerance) ** 2)
        
        return appeal, rate_deviation
    
    def get_trigger_rate_assessment(self, trigger_rate: float) -> str:
        """
        獲取觸發率評估描述
        
        Parameters:
        -----------
        trigger_rate : float
            觸發率
            
        Returns:
        --------
        str
            評估描述
        """
        if trigger_rate < 0.05:
            return "觸發率過低，客戶可能認為保障不足"
        elif trigger_rate < 0.1:
            return "觸發率偏低，適合保守型客戶"
        elif trigger_rate <= 0.3:
            return "觸發率適中，市場接受度較高"
        elif trigger_rate <= 0.5:
            return "觸發率偏高，保費可能較昂貴"
        else:
            return "觸發率過高，保費負擔沉重"


class PremiumAffordabilityAnalyzer:
    """保費可負擔性分析器"""
    
    def __init__(self, config: MarketAcceptabilityConfig):
        """
        初始化保費可負擔性分析器
        
        Parameters:
        -----------
        config : MarketAcceptabilityConfig
            市場接受度分析配置
        """
        self.config = config
    
    def calculate_premium_affordability(self,
                                      technical_premium: float,
                                      expected_payout: float) -> Tuple[float, float]:
        """
        計算保費可負擔性
        
        保費與期望賠付的比率越接近市場基準，接受度越高
        
        Parameters:
        -----------
        technical_premium : float
            技術保費
        expected_payout : float
            期望賠付
            
        Returns:
        --------
        Tuple[float, float]
            (可負擔性評分, 保費比率)
        """
        if expected_payout <= 0:
            return 0.0, float('inf')
        
        premium_ratio = technical_premium / expected_payout
        benchmark = self.config.market_premium_benchmark
        
        # 計算偏離度
        ratio_deviation = abs(premium_ratio - benchmark) / benchmark
        
        # 比率偏離基準越遠，接受度越低
        affordability = max(0, 1 - ratio_deviation)
        
        return affordability, premium_ratio
    
    def get_affordability_assessment(self, premium_ratio: float) -> str:
        """
        獲取可負擔性評估描述
        
        Parameters:
        -----------
        premium_ratio : float
            保費比率
            
        Returns:
        --------
        str
            評估描述
        """
        if premium_ratio < 1.2:
            return "保費相對便宜，高性價比"
        elif premium_ratio <= 1.8:
            return "保費合理，市場接受度高"
        elif premium_ratio <= 2.5:
            return "保費偏貴，需權衡保障與成本"
        else:
            return "保費過高，市場接受度低"


class MarketAcceptabilityAnalyzer:
    """市場接受度綜合分析器"""
    
    def __init__(self, 
                 config: Optional[MarketAcceptabilityConfig] = None,
                 complexity_evaluator: Optional[ProductComplexityEvaluator] = None):
        """
        初始化市場接受度分析器
        
        Parameters:
        -----------
        config : MarketAcceptabilityConfig, optional
            分析配置，默認使用標準配置
        complexity_evaluator : ProductComplexityEvaluator, optional
            複雜度評估器，默認使用標準評估器
        """
        self.config = config or MarketAcceptabilityConfig()
        self.complexity_evaluator = complexity_evaluator or StandardComplexityEvaluator()
        self.frequency_analyzer = TriggerFrequencyAnalyzer(self.config)
        self.affordability_analyzer = PremiumAffordabilityAnalyzer(self.config)
    
    def analyze_market_acceptability(self,
                                   product: ParametricProduct,
                                   technical_premium: float,
                                   expected_payout: float,
                                   trigger_rate: float) -> MarketAcceptabilityResult:
        """
        計算綜合市場接受度
        
        Parameters:
        -----------
        product : ParametricProduct
            保險產品
        technical_premium : float
            技術保費
        expected_payout : float
            期望賠付
        trigger_rate : float
            觸發率
            
        Returns:
        --------
        MarketAcceptabilityResult
            市場接受度分析結果
        """
        # 1. 產品複雜度評估
        complexity_score, complexity_level = self.complexity_evaluator.evaluate_complexity(product)
        simplicity_score = 1 - complexity_score
        
        # 2. 觸發頻率吸引力評估
        frequency_appeal, trigger_rate_deviation = \
            self.frequency_analyzer.calculate_trigger_frequency_appeal(trigger_rate)
        
        # 3. 保費可負擔性評估
        affordability, premium_ratio = \
            self.affordability_analyzer.calculate_premium_affordability(
                technical_premium, expected_payout
            )
        
        # 4. 計算綜合評分 (加權平均)
        weights = {
            'simplicity': self.config.complexity_weight,
            'frequency': self.config.frequency_weight,
            'affordability': self.config.affordability_weight
        }
        
        # 確保權重和為1
        total_weight = sum(weights.values())
        if total_weight != 1.0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        overall_acceptability = (
            weights['simplicity'] * simplicity_score +
            weights['frequency'] * frequency_appeal +
            weights['affordability'] * affordability
        )
        
        return MarketAcceptabilityResult(
            complexity_score=complexity_score,
            simplicity_score=simplicity_score,
            frequency_appeal=frequency_appeal,
            affordability=affordability,
            overall_acceptability=overall_acceptability,
            complexity_level=complexity_level,
            trigger_rate_deviation=trigger_rate_deviation,
            premium_ratio=premium_ratio
        )
    
    def get_detailed_assessment(self, result: MarketAcceptabilityResult) -> Dict[str, str]:
        """
        獲取詳細評估報告
        
        Parameters:
        -----------
        result : MarketAcceptabilityResult
            分析結果
            
        Returns:
        --------
        Dict[str, str]
            詳細評估報告
        """
        # 獲取各項評估描述
        trigger_assessment = self.frequency_analyzer.get_trigger_rate_assessment(
            self.config.optimal_trigger_rate - result.trigger_rate_deviation
        )
        
        affordability_assessment = self.affordability_analyzer.get_affordability_assessment(
            result.premium_ratio
        )
        
        # 綜合評估
        if result.overall_acceptability >= 0.8:
            overall_assessment = "市場接受度極高，推薦產品"
        elif result.overall_acceptability >= 0.6:
            overall_assessment = "市場接受度良好，可考慮推出"
        elif result.overall_acceptability >= 0.4:
            overall_assessment = "市場接受度中等，需改進設計"
        else:
            overall_assessment = "市場接受度較低，建議重新設計"
        
        return {
            'complexity_assessment': f"產品複雜度: {result.complexity_level.value}",
            'trigger_assessment': trigger_assessment,
            'affordability_assessment': affordability_assessment,
            'overall_assessment': overall_assessment,
            'recommendations': self._generate_recommendations(result)
        }
    
    def _generate_recommendations(self, result: MarketAcceptabilityResult) -> str:
        """生成改進建議"""
        recommendations = []
        
        if result.simplicity_score < 0.5:
            recommendations.append("簡化觸發條件，減少產品複雜度")
        
        if result.frequency_appeal < 0.5:
            recommendations.append(f"調整觸發率至最佳範圍 ({self.config.optimal_trigger_rate:.1%} ± {self.config.trigger_rate_tolerance:.1%})")
        
        if result.affordability < 0.5:
            recommendations.append("優化保費定價，提高性價比")
        
        return " | ".join(recommendations) if recommendations else "產品設計良好，無需特別調整"


def create_standard_market_analyzer(optimal_trigger_rate: float = 0.20,
                                  market_benchmark: float = 1.5) -> MarketAcceptabilityAnalyzer:
    """
    創建標準市場接受度分析器
    
    Parameters:
    -----------
    optimal_trigger_rate : float
        最佳觸發率
    market_benchmark : float
        市場保費基準
        
    Returns:
    --------
    MarketAcceptabilityAnalyzer
        標準市場接受度分析器
    """
    config = MarketAcceptabilityConfig(
        optimal_trigger_rate=optimal_trigger_rate,
        market_premium_benchmark=market_benchmark
    )
    
    return MarketAcceptabilityAnalyzer(config)