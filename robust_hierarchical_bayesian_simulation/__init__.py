"""
Robust Hierarchical Bayesian Simulation Package
魯棒階層貝葉斯模擬包

支援兩階段工作流程：
1. VI 快速篩選和基差風險導向訓練
2. MCMC 精確驗證和後驗分析

新增模組化組件：
3. 空間數據處理 (SpatialDataProcessor)
4. 階層模型構建 (build_hierarchical_model) 
5. 投資組合優化 (PortfolioOptimizer)
"""

# 階段1: VI 快速訓練
try:
    from .basis_risk_vi import (
        DifferentiableCRPS,
        ParametricPayoutFunction,
        BasisRiskAwareVI
    )
    from .vi_screener import VIScreener
    VI_AVAILABLE = True
except ImportError:
    VI_AVAILABLE = False

# 階段2: MCMC 精確驗證  
try:
    from .mcmc_validator import MCMCValidator
    MCMC_AVAILABLE = True
except ImportError:
    MCMC_AVAILABLE = False

# 共用組件
try:
    from .climada_data_loader import CLIMADADataLoader
    CLIMADA_LOADER_AVAILABLE = True
except ImportError:
    CLIMADA_LOADER_AVAILABLE = False

# 核心階層貝葉斯模型（已整合空間效應）
try:
    from .parametric_bayesian_hierarchy import (
        ParametricHierarchicalModel, ModelSpec, VulnerabilityData,
        CovarianceFunction, SpatialConfig, PriorScenario, LikelihoodFamily
    )
    HIERARCHY_AVAILABLE = True
except ImportError:
    HIERARCHY_AVAILABLE = False

# 新的模組化組件 (修正現有硬編碼問題)
# 注意: spatial_data_processor 已移至 data_processing 模組
# 注意: hierarchical_model_builder 已移至 hierarchical_modeling 子模組
from .hierarchical_modeling import (
    build_hierarchical_model, 
    get_portfolio_loss_predictions, 
    validate_model_inputs
)
from .portfolio_optimizer import PortfolioOptimizer, ProductAllocation

# 版本信息
__version__ = "1.0.0"

__all__ = [
    # 新的模組化組件 (主要接口)
    # 注意: SpatialDataProcessor 等已移至 data_processing 模組
    'build_hierarchical_model',
    'get_portfolio_loss_predictions', 
    'validate_model_inputs',
    'PortfolioOptimizer',
    'ProductAllocation'
]

# 添加可用的舊組件
if VI_AVAILABLE:
    __all__.extend([
        'DifferentiableCRPS',
        'ParametricPayoutFunction', 
        'BasisRiskAwareVI',
        'VIScreener'
    ])

if MCMC_AVAILABLE:
    __all__.append('MCMCValidator')

if CLIMADA_LOADER_AVAILABLE:
    __all__.append('CLIMADADataLoader')

if HIERARCHY_AVAILABLE:
    __all__.extend([
        'ParametricHierarchicalModel', 
        'ModelSpec', 
        'VulnerabilityData',
        'CovarianceFunction', 
        'SpatialConfig', 
        'PriorScenario', 
        'LikelihoodFamily'
    ])

# 模組狀態信息
def get_module_status():
    """獲取模組可用狀態"""
    status = "🎯 Robust Hierarchical Bayesian Simulation Package Status:\n"
    status += f"✅ 模組化組件: SpatialDataProcessor, build_hierarchical_model, PortfolioOptimizer\n"
    status += f"{'✅' if VI_AVAILABLE else '❌'} VI components: {VI_AVAILABLE}\n"
    status += f"{'✅' if MCMC_AVAILABLE else '❌'} MCMC components: {MCMC_AVAILABLE}\n" 
    status += f"{'✅' if CLIMADA_LOADER_AVAILABLE else '❌'} CLIMADA loader: {CLIMADA_LOADER_AVAILABLE}\n"
    status += f"{'✅' if HIERARCHY_AVAILABLE else '❌'} Parametric hierarchy: {HIERARCHY_AVAILABLE}\n"
    return status