"""
Robust Bayesian Analysis Framework
穩健貝氏分析框架

A clean, well-structured Bayesian framework for parametric insurance analysis with:
- 4 core modules with clear separation of concerns
- Integration with skill_scores and insurance_analysis_refactored
- Mixed Predictive Estimation (MPE) throughout
- Density ratio constraints for robustness

Core Architecture:
================

1. robust_bayesian_analysis.py (THEORY/FRAMEWORK)
   - Mathematical foundation: Density ratio class framework (Γ = {P : dP/dP₀ ≤ γ(x)})
   - Core theoretical components: model selection, robustness evaluation
   - Provides: DensityRatioClass, RobustBayesianFramework

2. hierarchical_bayesian_model.py  
   - 4-level hierarchical Bayesian model
   - Level 1: Observation Model (Y|θ, σ²)
   - Level 2: Process Model (θ|φ, τ²)
   - Level 3: Parameter Model (φ|α, β)
   - Level 4: Hyperparameter Model (α, β)
   - Mixed Predictive Estimation (MPE) implementation

3. robust_bayesian_uncertainty.py
   - Probabilistic loss distribution generation
   - Hazard intensity spatial correlation noise
   - Exposure value log-normal uncertainty
   - Vulnerability function parameter uncertainty
   - Integration with CLIMADA

4. robust_bayesian_analyzer.py (MAIN USER INTERFACE)
   - Primary analyzer integrating ALL components
   - Complete workflow: skill_scores + insurance + all theory
   - Provides: RobustBayesianAnalyzer (recommended for users)
   - One-stop interface for comprehensive Bayesian analysis

Usage Examples:
==============

🚀 Primary Interface (Recommended - 正確的兩階段整合流程):
```python
from bayesian import RobustBayesianAnalyzer
from skill_scores.basis_risk_functions import BasisRiskType

# Initialize analyzer
analyzer = RobustBayesianAnalyzer(
    density_ratio_constraint=2.0,
    n_monte_carlo_samples=500,
    n_mixture_components=3
)

# 🎯 NEW: Integrated two-phase optimization (方法一 → 方法二連貫流程)
# This automatically executes:
# 1. Method 1: Build candidate models → Fit all models → Skill evaluation → Select champion
# 2. Method 2: Use champion model's posterior → Define basis risk loss → Minimize expected loss
results = analyzer.integrated_bayesian_optimization(
    observations=train_losses,           # Method 1: training data for model fitting
    validation_data=validation_losses,   # Method 1: validation for model selection
    hazard_indices=hazard_indices,       # Method 2: hazard indices for optimization
    actual_losses=actual_losses_matrix,  # Method 2: loss scenarios for basis risk calculation
    product_bounds={'trigger_threshold': (30, 60), 'payout_amount': (1e7, 1e9)},
    basis_risk_type=BasisRiskType.WEIGHTED_ASYMMETRIC,
    w_under=2.0, w_over=0.5
)

# Access results
champion_model = results['phase_1_model_comparison']['champion_model']
optimal_product = results['phase_2_decision_optimization']['optimal_product']
theoretical_compliance = results['integration_validation']['theoretical_compliance']

print(f"Champion Model: {champion_model['name']} (CRPS: {champion_model['crps_score']:.3e})")
print(f"Optimal Product: Threshold={optimal_product['trigger_threshold']:.1f}, "
      f"Payout=${optimal_product['payout_amount']:.1e}")
print(f"Theory Compliance: {theoretical_compliance}")
```

🔧 Legacy Methods (已移除 - Removed for API Simplification):
```
❌ 舊方法已移除 (Old methods removed):
   - fit_and_compare_models()      # 獨立的方法一實現
   - optimize_product_parameters() # 獨立的方法二實現
   
💡 移除原因:
   1. 簡化API，避免用戶混淆
   2. 強制使用正確的理論框架  
   3. 確保方法一→方法二的連貫流程
   4. 減少維護負擔
   
✅ 新的統一介面:
   只使用 integrated_bayesian_optimization() 即可獲得完整功能
```

🔧 Advanced Component Usage (For Researchers/Developers):
```python
# Direct access to underlying components if needed
from bayesian import (
    RobustBayesianFramework,           # Density ratio framework
    HierarchicalBayesianModel,         # 4-level hierarchical + MPE
    ProbabilisticLossDistributionGenerator,  # Uncertainty quantification
)

# Use components separately for research
framework = RobustBayesianFramework(density_ratio_constraint=2.0)
hierarchical = HierarchicalBayesianModel(config)
uncertainty_gen = ProbabilisticLossDistributionGenerator()
```

Integration with External Modules:

# 1. Robust analysis with multiple models
framework = RobustBayesianFramework(density_ratio_constraint=2.0)
comparison_results = framework.compare_all_models(observed_losses)

# 2. Hierarchical model with MPE
config = HierarchicalModelConfig(n_mixture_components=3)
hierarchical_model = HierarchicalBayesianModel(config)
hierarchical_result = hierarchical_model.fit(observed_losses)

# 3. Uncertainty quantification
uncertainty_generator = ProbabilisticLossDistributionGenerator(
    n_monte_carlo_samples=500
)
probabilistic_losses = uncertainty_generator.generate_probabilistic_loss_distributions(
    tc_hazard, exposure_main, impact_func_set
)

# 4. Density ratio constraints
density_ratio = DensityRatioClass(gamma_constraint=2.0)
density_ratio.set_reference_prior("normal", loc=0, scale=1)
```

Integration with External Modules:
```python
# Skill scores are automatically integrated
# Results include CRPS, EDI, TSS evaluations

# Insurance products can be provided or auto-generated
from insurance_analysis_refactored.core import ParametricInsuranceEngine

engine = ParametricInsuranceEngine()
products = engine.generate_steinmann_products(indices)

results = analyzer.comprehensive_bayesian_analysis(
    ...,
    parametric_products=products
)
```
"""

# =============================================================================
# Core Bayesian Components (Clean 4-Module Architecture)
# =============================================================================

# 1. Robust Bayesian Analysis (Density Ratio Framework)
from .robust_bayesian_analysis import (
    RobustBayesianFramework,
    DensityRatioClass,
    ModelSelectionCriterion,
    ModelConfiguration,
    ModelComparisonResult
)

# 2. Hierarchical Bayesian Model (4-Level + MPE)
from .hierarchical_bayesian_model import (
    HierarchicalBayesianModel,
    HierarchicalModelConfig,
    HierarchicalModelResult,
    MixedPredictiveEstimation
)

# 3. Robust Bayesian Uncertainty Quantification
from .robust_bayesian_uncertainty import (
    ProbabilisticLossDistributionGenerator,
    integrate_robust_bayesian_with_parametric_insurance
)

# 4. Main Analyzer (Integration Hub)
from .robust_bayesian_analyzer import (
    RobustBayesianAnalyzer
)

# 5. Weight Sensitivity Analysis
from .weight_sensitivity_analyzer import (
    WeightSensitivityAnalyzer,
    extend_robust_analyzer_with_weight_sensitivity
)

# NOTE: Model comparison and decision theory functionality has been migrated
# to robust_bayesian_analyzer.py for better integration and reduced complexity.
# Basis risk functions moved to skill_scores/basis_risk_functions.py for reusability.

# =============================================================================
# Public API (Clean & Simple)
# =============================================================================

__all__ = [
    # === Main Interface (Recommended) ===
    'RobustBayesianAnalyzer',                           # Primary analyzer (includes model comparison + decision theory)
    
    # === Core Components ===
    # Robust Analysis
    'RobustBayesianFramework',                          # Density ratio framework
    'DensityRatioClass',                                # Density ratio constraints
    'ModelSelectionCriterion',                          # Model selection enum
    'ModelConfiguration',                               # Model config dataclass
    'ModelComparisonResult',                            # Model comparison results
    
    # Hierarchical Model
    'HierarchicalBayesianModel',                        # 4-level hierarchical model
    'HierarchicalModelConfig',                          # Hierarchical config
    'HierarchicalModelResult',                          # Hierarchical results
    'MixedPredictiveEstimation',                        # MPE implementation
    
    # Uncertainty Quantification
    'ProbabilisticLossDistributionGenerator',           # Probabilistic loss generator
    'integrate_robust_bayesian_with_parametric_insurance',  # Integration function
    
    # Weight Sensitivity Analysis
    'WeightSensitivityAnalyzer',                        # Weight sensitivity analysis
    'extend_robust_analyzer_with_weight_sensitivity',   # Integration with RobustBayesianAnalyzer
]

__version__ = "2.0.0"  # Clean architecture version
__author__ = "Robust Bayesian Analysis Team"

# =============================================================================
# Module Information & Usage Guide
# =============================================================================

def get_module_info():
    """Get module architecture information"""
    return {
        'version': __version__,
        'core_modules': [
            'robust_bayesian_analysis.py - Density ratio framework',
            'hierarchical_bayesian_model.py - 4-level hierarchical + MPE', 
            'robust_bayesian_uncertainty.py - Uncertainty quantification',
            'robust_bayesian_analyzer.py - Main integration hub',
            'weight_sensitivity_analyzer.py - Weight sensitivity analysis'
        ],
        'external_integrations': [
            'skill_scores/ - CRPS, EDI, TSS evaluation',
            'insurance_analysis_refactored/ - Product design'
        ],
        'key_features': [
            'Mixed Predictive Estimation (MPE)',
            'Density ratio constraints (Γ = {P : dP/dP₀ ≤ γ(x)})',
            'Hierarchical uncertainty propagation',
            'Skill score evaluation',
            'Insurance product integration'
        ]
    }

def get_quick_start_guide():
    """Get quick start usage guide"""
    return """
    🚀 Quick Start Guide
    
    === 📊 完整報告生成 (推薦) ===
    ```python
    from bayesian import create_quick_report, BayesianReportGenerator, ReportConfig
    
    # 快速報告生成
    report_path = create_quick_report(
        posterior_samples=posterior_samples,
        model_results=model_results,
        observed_data=observed_data,
        title="我的貝氏分析報告"
    )
    
    # 自定義報告配置
    config = ReportConfig(
        title="詳細分析報告", 
        include_plots=True,
        output_format=ReportFormat.HTML
    )
    generator = BayesianReportGenerator(config=config)
    report_path = generator.generate_comprehensive_report(
        posterior_samples=posterior_samples,
        model_results=model_results,
        observed_data=observed_data
    )
    ```
    
    === 🔍 基本分析 ===
    ```python
    from bayesian import RobustBayesianAnalyzer
    
    analyzer = RobustBayesianAnalyzer()
    results = analyzer.comprehensive_bayesian_analysis(
        tc_hazard, exposure_main, impact_func_set, observed_losses
    )
    ```
    
    === 🔧 組件分析 ===
    ```python
    from bayesian import RobustBayesianFramework, HierarchicalBayesianModel, WeightSensitivityAnalyzer
    
    # Density ratio framework
    framework = RobustBayesianFramework(density_ratio_constraint=2.0)
    model_comparison = framework.compare_all_models(observed_losses)
    
    # Hierarchical model with MPE
    hierarchical = HierarchicalBayesianModel(config)
    hierarchical_result = hierarchical.fit(observed_losses)
    
    # Weight sensitivity analysis
    weight_analyzer = WeightSensitivityAnalyzer()
    sensitivity_results = weight_analyzer.analyze_weight_sensitivity(
        products, actual_losses, wind_indices
    )
    ```
    
    === ✨ 主要功能 ===
    • 🧠 完整貝氏分析流程
    • 📊 自動生成綜合報告 (HTML/Text/JSON)
    • 🔍 MCMC 收斂診斷 (R-hat, ESS, MCSE)
    • 🛡️ 穩健性和敏感度分析
    • 📈 模型比較和選擇 (AIC, BIC, Bayes Factor)
    • 🎨 自動生成分析圖表
    • 🔗 整合 skill_scores 和 insurance 模組
    """

def validate_installation():
    """Validate that all components are properly installed"""
    validation_results = {
        'core_modules': True,
        'skill_scores': False,
        'insurance_module': False,
        'climada': False,
        'dependencies': []
    }
    
    # Check skill_scores
    try:
        import sys, os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from skill_scores import calculate_crps
        validation_results['skill_scores'] = True
    except ImportError:
        validation_results['dependencies'].append('skill_scores module not found')
    
    # Check insurance module
    try:
        from insurance_analysis_refactored.core import ParametricInsuranceEngine
        validation_results['insurance_module'] = True
    except ImportError:
        validation_results['dependencies'].append('insurance_analysis_refactored module not found')
    
    # Check CLIMADA
    try:
        from climada.engine import ImpactCalc
        validation_results['climada'] = True
    except ImportError:
        validation_results['dependencies'].append('CLIMADA not available')
    
    return validation_results

# =============================================================================
# Helpful Import Messages
# =============================================================================

def _show_import_help():
    """Show helpful import information"""
    try:
        import sys
        if hasattr(sys, 'ps1'):  # Interactive mode only
            print("🧠 Robust Bayesian Analysis Framework v2.0.0 Loaded")
            print("   Clean 4-module architecture with external integrations")
            print("   Primary interface: RobustBayesianAnalyzer")
            print("")
            print("   📚 Use get_quick_start_guide() for usage examples")
            print("   🔍 Use validate_installation() to check dependencies")
            print("   ℹ️  Use get_module_info() for architecture details")
            
            # Quick validation
            validation = validate_installation()
            if not validation['skill_scores']:
                print("   ⚠️  skill_scores module not found - some functionality limited")
            if not validation['insurance_module']:
                print("   ⚠️  insurance_analysis_refactored module not found - using simplified evaluation")
    except:
        pass

# Auto-show help on import
_show_import_help()

# =============================================================================
# Configuration and Settings
# =============================================================================

# Default configuration for all components
DEFAULT_CONFIG = {
    'density_ratio_constraint': 2.0,
    'n_monte_carlo_samples': 500,
    'n_mixture_components': 3,
    'hazard_uncertainty_std': 0.15,
    'exposure_uncertainty_log_std': 0.20,
    'vulnerability_uncertainty_std': 0.10,
    'mcmc_samples': 2000,
    'mcmc_warmup': 1000,
    'mcmc_chains': 4
}

def get_default_config():
    """Get default configuration for all Bayesian components"""
    return DEFAULT_CONFIG.copy()

def set_global_config(**kwargs):
    """Set global configuration parameters"""
    for key, value in kwargs.items():
        if key in DEFAULT_CONFIG:
            DEFAULT_CONFIG[key] = value
        else:
            print(f"Warning: Unknown configuration parameter '{key}' ignored")

# =============================================================================
# Legacy Support (Backward Compatibility)
# =============================================================================

# For backward compatibility with existing code
try:
    from .bayesian_insurance_integration import BayesianInsuranceAnalyzer as LegacyBayesianInsuranceAnalyzer
    
    def BayesianInsuranceAnalyzer(*args, **kwargs):
        """Legacy wrapper - use RobustBayesianAnalyzer instead"""
        import warnings
        warnings.warn(
            "BayesianInsuranceAnalyzer is deprecated. Use RobustBayesianAnalyzer instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return RobustBayesianAnalyzer(*args, **kwargs)
    
    __all__.append('BayesianInsuranceAnalyzer')  # Legacy support
    
except ImportError:
    pass  # Legacy module not available

# Export legacy integration function with new name
integrate_bayesian_with_insurance_products = integrate_robust_bayesian_with_parametric_insurance
__all__.append('integrate_bayesian_with_insurance_products')