"""
VI+MCMC Framework Module
變分推論與MCMC框架模組

支援兩階段工作流程：
1. VI 快速篩選和基差風險導向訓練
2. MCMC 精確驗證和後驗分析
"""

# 階段1: VI 快速訓練
from .basis_risk_vi import (
    DifferentiableCRPS,
    ParametricPayoutFunction,
    BasisRiskAwareVI
)

from .vi_screener import VIScreener

# 階段2: MCMC 精確驗證  
from .mcmc_validator import MCMCValidator

# 共用組件
from .climada_data_loader import CLIMADADataLoader

# 核心階層貝葉斯模型（已整合空間效應）
from .parametric_bayesian_hierarchy import (
    ParametricHierarchicalModel, ModelSpec, VulnerabilityData,
    CovarianceFunction, SpatialConfig, PriorScenario, LikelihoodFamily
)

__all__ = [
    # VI 快速訓練
    'DifferentiableCRPS',
    'ParametricPayoutFunction', 
    'BasisRiskAwareVI',
    'VIScreener',
    
    # MCMC 精確驗證
    'MCMCValidator',
    
    # 共用組件
    'CLIMADADataLoader',
    
    # 核心階層貝葉斯模型（已整合空間效應）
    'ParametricHierarchicalModel', 
    'ModelSpec', 
    'VulnerabilityData',
    'CovarianceFunction', 
    'SpatialConfig', 
    'PriorScenario', 
    'LikelihoodFamily'
]