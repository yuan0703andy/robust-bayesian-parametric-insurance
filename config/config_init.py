#!/usr/bin/env python3
"""
Configuration Module
配置模組

統一的配置管理系統

主要組件:
- model_configs: 模型配置
- computation_configs: 計算配置
- validation_configs: 驗證配置

Author: Research Team
Date: 2025-01-17
"""

from .model_configs import (
    # 工作流程和複雜度
    WorkflowStage,
    ModelComplexity,
    
    # 配置結構
    IntegratedFrameworkConfig,
    DataProcessingConfig,
    RobustPriorsConfig,
    HierarchicalModelingConfig,
    VIScreeningConfig,
    CRPSFrameworkConfig,
    MCMCValidationConfig,
    PosteriorAnalysisConfig,
    ParametricInsuranceConfig,
    ComputationConfig,
    
    # 預設配置生成器
    create_quick_prototype_config,
    create_standard_analysis_config,
    create_comprehensive_research_config,
    create_hpc_optimized_config,
    create_epsilon_contamination_focused_config
)

__all__ = [
    # 枚舉
    "WorkflowStage",
    "ModelComplexity",
    
    # 主要配置類別
    "IntegratedFrameworkConfig",
    "ComputationConfig",
    
    # 階段配置
    "DataProcessingConfig",
    "RobustPriorsConfig", 
    "HierarchicalModelingConfig",
    "VIScreeningConfig",
    "CRPSFrameworkConfig",
    "MCMCValidationConfig",
    "PosteriorAnalysisConfig",
    "ParametricInsuranceConfig",
    
    # 配置生成器
    "create_quick_prototype_config",
    "create_standard_analysis_config",
    "create_comprehensive_research_config", 
    "create_hpc_optimized_config",
    "create_epsilon_contamination_focused_config"
]