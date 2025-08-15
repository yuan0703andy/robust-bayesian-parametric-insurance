"""
Robust Bayesian Analysis Framework
穩健貝氏分析框架

Clean, modular implementation of robust Bayesian methods for parametric insurance analysis.

Core Modules:
=============

1. posterior_mixture_approximation.py - 後驗混合分布近似 (MPE)
2. parametric_bayesian_hierarchy.py - 參數化階層貝氏模型
3. robust_model_ensemble_analyzer.py - 穩健模型集合分析器 (M = Γ_f × Γ_π)
4. minimax_credible_intervals.py - 極小化極大可信區間計算
5. parametric_product_optimizer.py - 參數化產品優化器 (決策理論)

Supporting Modules:
==================
- spatial_effects.py - 空間效應模組 (醫院脆弱度的空間相關性建模)
- density_ratio_theory.py - 密度比理論框架 (Γ = {P : dP/dP₀ ≤ γ(x)})
- epsilon_contamination.py - ε-污染理論框架 (π(θ) = (1-ε)π₀(θ) + εq(θ))
- posterior_predictive_checks.py - 後驗預測檢查 (PPC)
- climada_uncertainty_quantification.py - CLIMADA不確定性量化
- mcmc_environment_config.py - MCMC環境配置
- basis_risk_weight_sensitivity.py - 基差風險權重敏感性分析

Usage Examples:
===============

# 1. Mixed Predictive Estimation
from bayesian.mixed_predictive_estimation import MixedPredictiveEstimation
mpe = MixedPredictiveEstimation()
result = mpe.fit_mixture(posterior_samples, "normal", n_components=3)

# 2. Hierarchical Model
from bayesian.hierarchical_model_parametric import ParametricHierarchicalModel, ModelSpec
spec = ModelSpec(likelihood_family="normal", prior_scenario="weak_informative")
model = ParametricHierarchicalModel(spec)
result = model.fit(observations)

# 3. Model Class Analysis
from bayesian.model_class_analyzer import ModelClassAnalyzer
analyzer = ModelClassAnalyzer()
results = analyzer.analyze_model_class(observations)

# 4. Robust Credible Intervals
from bayesian.robust_credible_intervals import RobustCredibleIntervalCalculator
calculator = RobustCredibleIntervalCalculator()
robust_interval = calculator.compute_robust_interval(posterior_samples_dict, "theta")

# 5. Bayesian Decision Theory
from bayesian.bayesian_decision_theory import BayesianDecisionOptimizer
optimizer = BayesianDecisionOptimizer()
decision_result = optimizer.optimize_expected_risk(product, posterior_samples, hazard_indices, losses)
"""

# =============================================================================
# Core Independent Modules (5個核心模組)
# =============================================================================

# 1. Posterior Mixture Approximation (混合預測估計)
from .posterior_mixture_approximation import (
    MixedPredictiveEstimation,
    MPEResult, 
    MPEConfig,
    fit_gaussian_mixture,
    sample_from_gaussian_mixture
)

# 1a. Posterior Predictive Checks (後驗預測檢查) - NEW
from .posterior_predictive_checks import (
    PPCValidator,
    PPCComparator,
    quick_ppc,
    compare_distributions
)

# 2. Parametric Bayesian Hierarchy (參數化階層貝氏模型)
from .parametric_bayesian_hierarchy import (
    ParametricHierarchicalModel,
    ModelSpec,
    MCMCConfig,
    DiagnosticResult,
    HierarchicalModelResult,
    LikelihoodFamily,
    PriorScenario,
    VulnerabilityData,           # 新增：脆弱度數據結構
    VulnerabilityFunctionType,   # 新增：脆弱度函數類型
    create_model_spec,
    quick_fit
)

# 3. Robust Model Ensemble Analyzer (穩健模型集合分析器)
from .robust_model_ensemble_analyzer import (
    ModelClassAnalyzer,
    ModelClassSpec,
    ModelClassResult,
    AnalyzerConfig,
    quick_model_class_analysis
)

# 4. Minimax Credible Intervals (極小化極大可信區間)
from .minimax_credible_intervals import (
    RobustCredibleIntervalCalculator,
    IntervalResult,
    IntervalComparison,
    IntervalOptimizationMethod,
    CalculatorConfig,
    compute_robust_credible_interval,
    compare_credible_intervals
)

# 5. Parametric Product Optimizer (參數化產品優化器)
from .parametric_product_optimizer import (
    BayesianDecisionOptimizer,
    ProductParameters,
    ProductSpace,
    DecisionResult,
    GammaMinimaxResult,
    OptimizerConfig,
    OptimizationMethod
)

# =============================================================================
# Supporting Modules
# =============================================================================

# Density Ratio Theory (密度比理論框架)
from .density_ratio_theory import (
    RobustBayesianFramework,
    DensityRatioClass,
    ModelSelectionCriterion,
    ModelConfiguration,
    ModelComparisonResult
)

# ε-Contamination Theory (ε-污染理論框架) - NEW
from .epsilon_contamination import (
    EpsilonContaminationClass,
    EpsilonContaminationSpec,
    ContaminationEstimateResult,
    ContaminationDistributionClass,
    create_typhoon_contamination_spec,
    quick_contamination_analysis,
    demonstrate_dual_process_nature
)

# CLIMADA Uncertainty Quantification (CLIMADA不確定性量化)
from .climada_uncertainty_quantification import (
    ProbabilisticLossDistributionGenerator,
)

# MCMC Environment Configuration (MCMC環境配置)
from .mcmc_environment_config import configure_pymc_environment

# Basis Risk Weight Sensitivity (基差風險權重敏感性)
from .basis_risk_weight_sensitivity import (
    WeightSensitivityAnalyzer,
)

# CPU Optimization Config (CPU優化配置) - UPDATED
def get_cpu_optimized_mcmc_config(n_cores=None, quick_test=False):
    """Get CPU-optimized MCMC configuration"""
    import multiprocessing
    if n_cores is None:
        n_cores = min(multiprocessing.cpu_count(), 8)  # Cap at 8 cores
    
    if quick_test:
        return {
            "n_samples": 200,
            "n_warmup": 100,
            "n_chains": 2,
            "cores": min(n_cores, 2),
            "target_accept": 0.80,
            "backend": "pytensor"
        }
    else:
        return {
            "n_samples": 1000,
            "n_warmup": 500,
            "n_chains": min(4, n_cores),
            "cores": n_cores,
            "target_accept": 0.85,
            "backend": "pytensor"
        }

# =============================================================================
# Public API (5個核心獨立模組)
# =============================================================================

__all__ = [
    # === 5個核心獨立模組 ===
    'MixedPredictiveEstimation',              # 混合預測估計
    'PPCValidator', 'PPCComparator',          # 後驗預測檢查
    'ParametricHierarchicalModel',            # 參數化階層模型
    'ModelClassAnalyzer',                     # 模型集合分析器
    'RobustCredibleIntervalCalculator',       # 穩健可信區間
    'BayesianDecisionOptimizer',              # 貝氏決策理論
    
    # === 核心模組的配置類 ===
    'ModelSpec', 'MCMCConfig', 'ModelClassSpec', 'AnalyzerConfig',
    'MPEConfig', 'CalculatorConfig', 'OptimizerConfig',
    
    # === 核心模組的結果類 ===
    'MPEResult', 'HierarchicalModelResult', 'ModelClassResult', 
    'IntervalResult', 'DecisionResult', 'GammaMinimaxResult',
    
    # === 便利函數 ===
    'fit_gaussian_mixture', 'quick_ppc', 'compare_distributions',
    'create_model_spec', 'quick_fit',
    'quick_model_class_analysis', 'compute_robust_credible_interval',
    
    # === 理論基礎 ===
    'RobustBayesianFramework', 'DensityRatioClass',
    'EpsilonContaminationClass', 'quick_contamination_analysis',
    
    # === 支持組件 ===
    'ProbabilisticLossDistributionGenerator',
    'WeightSensitivityAnalyzer',
    'configure_pymc_environment',
    
    # === CPU優化 ===
    'get_cpu_optimized_mcmc_config',
    
    # === 新增：空間效應組件 ===
    'SpatialEffectsAnalyzer',            # 空間效應分析器
    'SpatialConfig',                     # 空間分析配置
    'SpatialEffectsResult',              # 空間分析結果
    'CovarianceFunction',                # 協方差函數類型
    'create_standard_spatial_config',    # 創建標準空間配置
    'quick_spatial_analysis'             # 快速空間分析
]

# 導入空間效應模組
try:
    from .spatial_effects import (
        SpatialEffectsAnalyzer,
        SpatialConfig,
        SpatialEffectsResult,
        CovarianceFunction,
        create_standard_spatial_config,
        quick_spatial_analysis
    )
    HAS_SPATIAL_EFFECTS = True
except ImportError:
    HAS_SPATIAL_EFFECTS = False
    import warnings
    warnings.warn("空間效應模組不可用")

__version__ = "3.0.0"  # Modular architecture version
__author__ = "Robust Bayesian Analysis Team"

# =============================================================================
# Module Information
# =============================================================================

def get_module_info():
    """Get module architecture information"""
    return {
        'version': __version__,
        'core_modules': [
            'mixed_predictive_estimation.py - 混合預測估計 (MPE)',
            'hierarchical_model_parametric.py - 參數化階層貝氏模型', 
            'model_class_analyzer.py - 模型集合分析器 (M = Γ_f × Γ_π)',
            'robust_credible_intervals.py - 穩健可信區間計算',
            'bayesian_decision_theory.py - 貝氏決策理論'
        ],
        'supporting_modules': [
            'robust_bayesian_analysis.py - 密度比理論框架',
            'robust_bayesian_uncertainty.py - 不確定性量化',
            'pymc_config.py - PyMC環境配置',
            'weight_sensitivity_analyzer.py - 權重敏感性分析'
        ],
        'key_features': [
            '完全獨立的模組設計',
            '每個模組可單獨import使用',
            'PyMC fallback機制',
            '完整的測試覆蓋',
            '清晰的功能分離'
        ]
    }

def validate_installation():
    """驗證模組安裝狀態"""
    validation_results = {
        'core_modules': {},
        'supporting_modules': {},
        'dependencies': []
    }
    
    # 測試5個核心模組
    core_modules = [
        ('mixed_predictive_estimation', 'MixedPredictiveEstimation'),
        ('hierarchical_model_parametric', 'ParametricHierarchicalModel'),
        ('model_class_analyzer', 'ModelClassAnalyzer'),
        ('robust_credible_intervals', 'RobustCredibleIntervalCalculator'),
        ('bayesian_decision_theory', 'BayesianDecisionOptimizer')
    ]
    
    for module_name, class_name in core_modules:
        try:
            exec(f"from .{module_name} import {class_name}")
            validation_results['core_modules'][module_name] = True
        except ImportError as e:
            validation_results['core_modules'][module_name] = False
            validation_results['dependencies'].append(f"{module_name}: {str(e)}")
    
    return validation_results

# =============================================================================
# Quick Start Guide
# =============================================================================

def show_quick_start():
    """顯示快速入門指南"""
    guide = """
    🚀 Robust Bayesian Analysis Framework v3.0.0
    ============================================
    
    5個核心獨立模組，每個都可單獨使用：
    
    📊 1. Mixed Predictive Estimation (MPE)
    ```python
    from bayesian import MixedPredictiveEstimation
    mpe = MixedPredictiveEstimation()
    result = mpe.fit_mixture(samples, "normal", n_components=3)
    new_samples = mpe.sample_from_mixture(1000, result)
    ```
    
    🏗️ 2. Parametric Hierarchical Model
    ```python
    from bayesian import ParametricHierarchicalModel, ModelSpec
    spec = ModelSpec("normal", "weak_informative")
    model = ParametricHierarchicalModel(spec)
    result = model.fit(observations)
    ```
    
    🔍 3. Model Class Analyzer (M = Γ_f × Γ_π)
    ```python
    from bayesian import ModelClassAnalyzer
    analyzer = ModelClassAnalyzer()
    results = analyzer.analyze_model_class(observations)
    theta_range = analyzer.compute_posterior_range('theta')
    ```
    
    🛡️ 4. Robust Credible Intervals
    ```python
    from bayesian import RobustCredibleIntervalCalculator
    calculator = RobustCredibleIntervalCalculator()
    robust_interval = calculator.compute_robust_interval(
        posterior_samples_dict, "theta", alpha=0.05
    )
    ```
    
    🎯 5. Bayesian Decision Theory
    ```python
    from bayesian import BayesianDecisionOptimizer, ProductParameters
    optimizer = BayesianDecisionOptimizer()
    product = ProductParameters(trigger_threshold=50, payout_amount=1e8)
    result = optimizer.optimize_expected_risk(
        product, posterior_samples, hazard_indices, losses
    )
    ```
    
    ✅ 驗證安裝：
    ```python
    from bayesian import validate_installation
    print(validate_installation())
    ```
    """
    print(guide)

# Auto-show quick start on import (interactive mode only)
def _show_import_help():
    try:
        import sys
        if hasattr(sys, 'ps1'):  # Interactive mode
            print("🧠 Robust Bayesian Analysis Framework v3.0.0")
            print("   5個獨立模組架構，支持單獨import")
            print("   使用 show_quick_start() 查看快速指南")
            print("   使用 validate_installation() 驗證安裝")
    except:
        pass

_show_import_help()