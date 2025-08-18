#!/usr/bin/env python3
"""
Utilities Module
工具模組

提供整個框架使用的通用工具函數

主要組件:
- math_utils: 數學工具函數
- plotting_utils: 繪圖工具 (待實現)
- validation_utils: 驗證工具 (待實現)

Author: Research Team
Date: 2025-01-17
"""

from .math_utils import (
    # CRPS相關函數
    crps_empirical,
    crps_normal,
    crps_lognormal,
    crps_ensemble,
    
    # 貝氏統計工具
    effective_sample_size,
    rhat_statistic,
    hdi_interval,
    
    # 數值優化工具
    robust_minimize,
    constrained_optimization,
    
    # 分布處理工具
    fit_distribution,
    sample_from_mixture
)

__all__ = [
    # CRPS函數
    "crps_empirical",
    "crps_normal", 
    "crps_lognormal",
    "crps_ensemble",
    
    # 貝氏統計
    "effective_sample_size",
    "rhat_statistic",
    "hdi_interval",
    
    # 數值優化
    "robust_minimize",
    "constrained_optimization",
    
    # 分布工具
    "fit_distribution",
    "sample_from_mixture"
]