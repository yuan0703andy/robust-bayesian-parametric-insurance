"""
Robust Hierarchical Bayesian Simulation Package
魯棒階層貝葉斯模擬包

支援兩階段工作流程：
1. VI 快速篩選和基差風險導向訓練
2. MCMC 精確驗證和後驗分析

模組化組件：
3. 階層建模 (hierarchical_modeling)
4. 模型選擇 (model_selection)
5. 超參數優化 (hyperparameter_optimization)
6. MCMC驗證 (mcmc_validation)
7. 後驗分析 (posterior_analysis)
8. 魯棒先驗 (robust_priors)

注意: spatial_data_processor 已移至專案根目錄的 data_processing 模組
"""

# 階層建模 - 核心功能
try:
    from .hierarchical_modeling import (
        build_hierarchical_model, 
        get_portfolio_loss_predictions, 
        validate_model_inputs,
        ParametricHierarchicalModel
    )
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False

# 模型選擇和VI
try:
    from .model_selection import (
        BasisRiskAwareVI,
        ModelSelector,
        DifferentiableCRPS,
        ParametricPayoutFunction
    )
    MODEL_SELECTION_AVAILABLE = True
except ImportError:
    MODEL_SELECTION_AVAILABLE = False

# MCMC驗證
try:
    from .mcmc_validation import (
        CRPSMCMCValidator,
        setup_gpu_environment
    )
    MCMC_AVAILABLE = True
except ImportError:
    MCMC_AVAILABLE = False

# 後驗分析
try:
    from .posterior_analysis import (
        CredibleIntervalCalculator,
        PosteriorApproximation,
        PosteriorPredictiveChecker
    )
    POSTERIOR_AVAILABLE = True
except ImportError:
    POSTERIOR_AVAILABLE = False

# 魯棒先驗
try:
    from .robust_priors import (
        EpsilonEstimator,
        DoubleEpsilonContamination,
        EpsilonContaminationSpec
    )
    ROBUST_PRIORS_AVAILABLE = True
except ImportError:
    ROBUST_PRIORS_AVAILABLE = False

# 超參數優化
try:
    from .hyperparameter_optimization import (
        AdaptiveHyperparameterOptimizer,
        WeightSensitivityAnalyzer
    )
    HYPERPARAM_AVAILABLE = True
except ImportError:
    HYPERPARAM_AVAILABLE = False

# 配置管理
try:
    from .config import create_standard_analysis_config, ModelComplexity
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# GPU設置
try:
    from .gpu_setup import GPUConfig
    GPU_AVAILABLE = True  
except ImportError:
    GPU_AVAILABLE = False

# 版本信息
__version__ = "2.0.0"

__all__ = []

# 添加可用的組件到 __all__
if HIERARCHICAL_AVAILABLE:
    __all__.extend([
        'build_hierarchical_model',
        'get_portfolio_loss_predictions', 
        'validate_model_inputs',
        'ParametricHierarchicalModel'
    ])

if MODEL_SELECTION_AVAILABLE:
    __all__.extend([
        'BasisRiskAwareVI',
        'ModelSelector',
        'DifferentiableCRPS',
        'ParametricPayoutFunction'
    ])

if MCMC_AVAILABLE:
    __all__.extend([
        'CRPSMCMCValidator',
        'setup_gpu_environment'
    ])

if POSTERIOR_AVAILABLE:
    __all__.extend([
        'CredibleIntervalCalculator',
        'PosteriorApproximation',
        'PosteriorPredictiveChecker'
    ])

if ROBUST_PRIORS_AVAILABLE:
    __all__.extend([
        'EpsilonEstimator',
        'DoubleEpsilonContamination',
        'EpsilonContaminationSpec'
    ])

if HYPERPARAM_AVAILABLE:
    __all__.extend([
        'AdaptiveHyperparameterOptimizer',
        'WeightSensitivityAnalyzer'
    ])

if CONFIG_AVAILABLE:
    __all__.extend([
        'create_standard_analysis_config',
        'ModelComplexity'
    ])

if GPU_AVAILABLE:
    __all__.append('GPUConfig')

# 模組狀態信息
def get_module_status():
    """獲取模組可用狀態"""
    status = "🎯 Robust Hierarchical Bayesian Simulation Package Status:\n"
    status += f"{'✅' if HIERARCHICAL_AVAILABLE else '❌'} Hierarchical Modeling: {HIERARCHICAL_AVAILABLE}\n"
    status += f"{'✅' if MODEL_SELECTION_AVAILABLE else '❌'} Model Selection: {MODEL_SELECTION_AVAILABLE}\n"
    status += f"{'✅' if MCMC_AVAILABLE else '❌'} MCMC Validation: {MCMC_AVAILABLE}\n"
    status += f"{'✅' if POSTERIOR_AVAILABLE else '❌'} Posterior Analysis: {POSTERIOR_AVAILABLE}\n"
    status += f"{'✅' if ROBUST_PRIORS_AVAILABLE else '❌'} Robust Priors: {ROBUST_PRIORS_AVAILABLE}\n"
    status += f"{'✅' if HYPERPARAM_AVAILABLE else '❌'} Hyperparameter Optimization: {HYPERPARAM_AVAILABLE}\n"
    status += f"{'✅' if CONFIG_AVAILABLE else '❌'} Configuration: {CONFIG_AVAILABLE}\n"
    status += f"{'✅' if GPU_AVAILABLE else '❌'} GPU Setup: {GPU_AVAILABLE}\n"
    status += f"\n注意: SpatialDataProcessor 已移至專案根目錄的 data_processing 模組"
    return status