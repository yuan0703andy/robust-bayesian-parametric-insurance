#!/usr/bin/env python3
"""
Hierarchical Modeling Module - Stage 3
階層建模模組 - 階段3

階層貝氏模型建構和空間效應建模

主要組件:
- core_model: 核心階層模型類別
- prior_specifications: 先驗規格和枚舉定義
- likelihood_families: 似然函數族和配置
- spatial_effects: 空間效應建模 (待實現)

Author: Research Team
Date: 2025-01-17
"""

from .prior_specifications import (
    ModelSpec,
    VulnerabilityData,
    LikelihoodFamily,
    PriorScenario,
    VulnerabilityFunctionType,
    ContaminationDistribution,
    CovarianceFunction,
    SpatialConfig,
    get_prior_parameters,
    validate_model_spec
)

from .likelihood_families import (
    MCMCConfig,
    DiagnosticResult,
    HierarchicalModelResult,
    LikelihoodBuilder,
    ContaminationMixture,
    VulnerabilityFunctionBuilder,
    check_convergence,
    recommend_mcmc_adjustments
)

from .core_model import ParametricHierarchicalModel

__all__ = [
    # 主要類別
    "ParametricHierarchicalModel",
    "ModelSpec",
    "VulnerabilityData",
    "MCMCConfig",
    "DiagnosticResult", 
    "HierarchicalModelResult",
    "SpatialConfig",
    
    # 枚舉
    "LikelihoodFamily",
    "PriorScenario",
    "VulnerabilityFunctionType",
    "ContaminationDistribution",
    "CovarianceFunction",
    
    # 建構器
    "LikelihoodBuilder",
    "ContaminationMixture",
    "VulnerabilityFunctionBuilder",
    
    # 工具函數
    "get_prior_parameters",
    "validate_model_spec",
    "check_convergence",
    "recommend_mcmc_adjustments"
]