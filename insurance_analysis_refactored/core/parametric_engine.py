"""
Core Parametric Insurance Data Structures
核心參數型保險數據結構

This module provides core data structures and basic functionality for parametric insurance:
- Core data classes (ParametricProduct, ProductPerformance)
- Enumerations (ParametricIndexType, PayoutFunctionType)
- Basic product creation functionality

Note: Advanced functionality has been moved to specialized modules:
- Spatial analysis: enhanced_spatial_analysis.py
- Performance evaluation: skill_evaluator.py  
- Portfolio optimization: product_manager.py
- Premium calculation: technical_premium_calculator.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ParametricIndexType(Enum):
    """參數指標類型"""
    CAT_IN_CIRCLE = "cat_in_circle"
    MAXIMUM_WIND = "maximum_wind"
    AVERAGE_WIND = "average_wind"
    DURATION_ABOVE_THRESHOLD = "duration_above_threshold"

class PayoutFunctionType(Enum):
    """賠付函數類型"""
    STEP = "step"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SIGMOID = "sigmoid"

@dataclass
class ParametricProduct:
    """參數型保險產品數據類"""
    product_id: str
    name: str
    description: str
    index_type: ParametricIndexType
    payout_function_type: PayoutFunctionType
    trigger_thresholds: List[float]
    payout_amounts: List[float]
    max_payout: float
    technical_premium: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProductPerformance:
    """產品績效數據類"""
    product_id: str
    rmse: float
    mae: float
    correlation: float
    hit_rate: float
    false_alarm_rate: float
    coverage_ratio: float
    basis_risk: float
    skill_scores: Dict[str, float] = field(default_factory=dict)
    technical_metrics: Dict[str, float] = field(default_factory=dict)


class ParametricInsuranceEngine:
    """
    輕量級參數型保險引擎
    
    專注於產品創建和基本管理，複雜功能使用專業模組：
    - 空間分析: enhanced_spatial_analysis.py
    - 績效評估: skill_evaluator.py
    - 產品管理: product_manager.py
    - 保費計算: technical_premium_calculator.py
    """
    
    def __init__(self):
        self.products = {}
    
    def create_parametric_product(self,
                                product_id: str,
                                name: str,
                                description: str,
                                index_type: ParametricIndexType,
                                payout_function_type: PayoutFunctionType,
                                trigger_thresholds: List[float],
                                payout_amounts: List[float],
                                max_payout: float,
                                **kwargs) -> ParametricProduct:
        """
        創建參數型保險產品
        
        Parameters:
        -----------
        product_id : str
            產品ID
        name : str
            產品名稱
        description : str
            產品描述
        index_type : ParametricIndexType
            參數指標類型
        payout_function_type : PayoutFunctionType
            賠付函數類型
        trigger_thresholds : List[float]
            觸發閾值列表
        payout_amounts : List[float]
            賠付金額列表
        max_payout : float
            最大賠付金額
        **kwargs : dict
            額外參數
            
        Returns:
        --------
        ParametricProduct
            創建的產品
        """
        
        product = ParametricProduct(
            product_id=product_id,
            name=name,
            description=description,
            index_type=index_type,
            payout_function_type=payout_function_type,
            trigger_thresholds=trigger_thresholds,
            payout_amounts=payout_amounts,
            max_payout=max_payout,
            metadata=kwargs
        )
        
        self.products[product_id] = product
        return product
    
    def get_product(self, product_id: str) -> Optional[ParametricProduct]:
        """獲取產品"""
        return self.products.get(product_id)
    
    def list_products(self) -> List[ParametricProduct]:
        """列出所有產品"""
        return list(self.products.values())


# Legacy functionality moved to specialized modules:
# - CatInCircleExtractor -> enhanced_spatial_analysis.py 
# - PayoutFunction classes -> technical_premium_calculator.py or saffir_simpson_products.py
# - evaluate_product_performance -> skill_evaluator.py
# - optimize_product_portfolio -> product_manager.py  
# - calculate_technical_premium -> technical_premium_calculator.py
# - calculate_correct_step_payouts -> deprecated (only used in backup files)
# - calculate_crps_score -> skill_scores module