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

# 版本信息
__version__ = "2.0.0"

# 配置管理
try:
    from .config import create_standard_analysis_config, ModelComplexity
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

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

# 超參數優化
try:
    from .hyperparameter_optimization import (
        AdaptiveHyperparameterOptimizer,
        WeightSensitivityAnalyzer
    )
    HYPERPARAM_AVAILABLE = True
except ImportError:
    HYPERPARAM_AVAILABLE = False

# MCMC驗證
try:
    from .mcmc_validation import CRPSMCMCValidator
    MCMC_VALIDATOR_AVAILABLE = True
except ImportError:
    MCMC_VALIDATOR_AVAILABLE = False

# GPU 環境配置
try:
    from .gpu_setup.gpu_config import setup_gpu_environment
    GPU_SETUP_AVAILABLE = True
except ImportError:
    GPU_SETUP_AVAILABLE = False

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

# 構建動態 __all__ 列表
__all__ = []

if CONFIG_AVAILABLE:
    __all__.extend(['create_standard_analysis_config', 'ModelComplexity'])

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

if HYPERPARAM_AVAILABLE:
    __all__.extend([
        'AdaptiveHyperparameterOptimizer',
        'WeightSensitivityAnalyzer'
    ])

if MCMC_VALIDATOR_AVAILABLE:
    __all__.append('CRPSMCMCValidator')

if GPU_SETUP_AVAILABLE:
    __all__.append('setup_gpu_environment')

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

# 模組狀態信息
def get_module_status():
    """獲取模組可用狀態"""
    status = "🎯 Robust Hierarchical Bayesian Simulation Package Status:\n"
    status += f"{'✅' if CONFIG_AVAILABLE else '❌'} Configuration: {CONFIG_AVAILABLE}\n"
    status += f"{'✅' if HIERARCHICAL_AVAILABLE else '❌'} Hierarchical Modeling: {HIERARCHICAL_AVAILABLE}\n"
    status += f"{'✅' if MODEL_SELECTION_AVAILABLE else '❌'} Model Selection: {MODEL_SELECTION_AVAILABLE}\n"
    status += f"{'✅' if HYPERPARAM_AVAILABLE else '❌'} Hyperparameter Optimization: {HYPERPARAM_AVAILABLE}\n"
    status += f"{'✅' if MCMC_VALIDATOR_AVAILABLE else '❌'} MCMC Validation: {MCMC_VALIDATOR_AVAILABLE}\n"
    status += f"{'✅' if GPU_SETUP_AVAILABLE else '❌'} GPU Setup: {GPU_SETUP_AVAILABLE}\n"
    status += f"{'✅' if POSTERIOR_AVAILABLE else '❌'} Posterior Analysis: {POSTERIOR_AVAILABLE}\n"
    status += f"{'✅' if ROBUST_PRIORS_AVAILABLE else '❌'} Robust Priors: {ROBUST_PRIORS_AVAILABLE}\n"
    status += f"\n注意: SpatialDataProcessor 已移至專案根目錄的 data_processing 模組"
    return status

# 便利函數：檢查單個組件是否可用
def is_component_available(component_name: str) -> bool:
    """
    檢查特定組件是否可用
    
    Parameters:
    -----------
    component_name : str
        組件名稱 ('config', 'hierarchical', 'model_selection', etc.)
        
    Returns:
    --------
    bool : 組件是否可用
    """
    availability_map = {
        'config': CONFIG_AVAILABLE,
        'hierarchical': HIERARCHICAL_AVAILABLE,
        'model_selection': MODEL_SELECTION_AVAILABLE,
        'hyperparam': HYPERPARAM_AVAILABLE,
        'mcmc': MCMC_VALIDATOR_AVAILABLE,
        'gpu_setup': GPU_SETUP_AVAILABLE,
        'posterior': POSTERIOR_AVAILABLE,
        'robust_priors': ROBUST_PRIORS_AVAILABLE
    }
    
    return availability_map.get(component_name, False)

# 如果在直接執行時，顯示模組狀態
if __name__ == "__main__":
    print(f"🌀 Robust Hierarchical Bayesian Simulation Package v{__version__}")
    print(get_module_status())