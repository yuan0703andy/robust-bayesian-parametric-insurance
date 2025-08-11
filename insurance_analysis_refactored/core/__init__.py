"""
Core Insurance Analysis Module - Parametric Insurance Framework
核心保險分析模組 - 參數型保險框架

This module provides the core parametric insurance analysis framework including:
- Parametric insurance product design and optimization
- Skill score evaluation and performance assessment
- Product portfolio management and optimization
- Unified analysis framework for insurance workflows

Focus: Pure insurance product functionality without Bayesian dependencies.
All Bayesian and uncertainty quantification methods are in the bayesian/ module.
"""

# Legacy deterministic components (for backward compatibility)
from .parametric_engine import (
    ParametricInsuranceEngine, ParametricProduct, ProductPerformance,
    ParametricIndexType, PayoutFunctionType
)
from .skill_evaluator import (
    SkillScoreEvaluator, SkillScoreType, SkillScoreResult
)
from .product_manager import (
    InsuranceProductManager, ProductPortfolio, ProductStatus
)
# analysis_framework module not found - removed temporarily
# from .analysis_framework import (
#     UnifiedAnalysisFramework, AnalysisConfig, AnalysisType, AnalysisResults,
#     EvaluationMode, ProductDesignType
# )
# Input adapter functionality now available through UnifiedAnalysisFramework
from .input_adapters import InputAdapter, CLIMADAInputAdapter, BayesianInputAdapter
# Legacy Bayesian components moved to bayesian/ module
# Keeping only minimal references for backward compatibility

# NOTE: Bayesian components moved to bayesian/ module
# All Bayesian-related functionality is now centralized in the bayesian/ module
# unified_probabilistic_framework_full module not found - removed temporarily
# from .unified_probabilistic_framework_full import (
#     UnifiedProbabilisticFramework,
#     UnifiedFrameworkConfig,
#     UnifiedFrameworkResult,
#     AnalysisPhase
# )

# Advanced Technical Premium Analysis Modules (Modularized from 08_technical_premium_analysis.py)
from .technical_premium_calculator import (
    TechnicalPremiumCalculator,
    TechnicalPremiumConfig,
    TechnicalPremiumResult,
    ExpectedPayoutCalculator,
    GammaDistributionPayoutCalculator,
    SolvencyIIRiskCapitalCalculator,
    create_standard_technical_premium_calculator
)
from .market_acceptability_analyzer import (
    MarketAcceptabilityAnalyzer,
    MarketAcceptabilityConfig,
    MarketAcceptabilityResult,
    ProductComplexityLevel,
    create_standard_market_analyzer
)
from .multi_objective_optimizer import (
    MultiObjectiveOptimizer,
    OptimizationConfig,
    MultiObjectiveResult,
    ParetoSolution,
    OptimizationObjective,
    DecisionPreferenceType,
    create_standard_multi_objective_optimizer
)
from .technical_premium_visualizer import (
    TechnicalPremiumVisualizer,
    DecisionSupportReportGenerator,
    create_standard_visualizer,
    create_standard_report_generator
)


__all__ = [
    # Main unified framework - temporarily disabled
    # 'UnifiedProbabilisticFramework',           # Main entry point
    # 'UnifiedFrameworkConfig',
    # 'UnifiedFrameworkResult', 
    # 'AnalysisPhase',
    
    # Core insurance product components
    'ParametricInsuranceEngine',
    'SkillScoreEvaluator', 
    'InsuranceProductManager',
    # 'UnifiedAnalysisFramework',  # temporarily disabled
    
    # Input adapters
    'InputAdapter',
    'CLIMADAInputAdapter', 
    'BayesianInputAdapter',
    
    # Advanced Technical Premium Analysis Components
    'TechnicalPremiumCalculator',              # Advanced premium calculation
    'TechnicalPremiumConfig',
    'TechnicalPremiumResult',
    'ExpectedPayoutCalculator',
    'GammaDistributionPayoutCalculator',
    'SolvencyIIRiskCapitalCalculator',
    'create_standard_technical_premium_calculator',
    
    'MarketAcceptabilityAnalyzer',             # Market acceptability analysis
    'MarketAcceptabilityConfig',
    'MarketAcceptabilityResult',
    'ProductComplexityLevel',
    'create_standard_market_analyzer',
    
    'MultiObjectiveOptimizer',                 # Multi-objective optimization & Pareto analysis
    'OptimizationConfig',
    'MultiObjectiveResult',
    'ParetoSolution',
    'OptimizationObjective',
    'DecisionPreferenceType',
    'create_standard_multi_objective_optimizer',
    
    'TechnicalPremiumVisualizer',              # Visualization and reporting
    'DecisionSupportReportGenerator',
    'create_standard_visualizer',
    'create_standard_report_generator',
    
    # Core insurance data classes
    'ParametricProduct',
    'ProductPerformance',
    'SkillScoreResult',
    'ProductPortfolio',
    # 'AnalysisConfig',  # temporarily disabled
    # 'AnalysisResults',  # temporarily disabled
    
    # Insurance-focused enums
    'ParametricIndexType',
    'PayoutFunctionType',
    'SkillScoreType',
    'ProductStatus',
    # 'AnalysisType',      # temporarily disabled
    # 'EvaluationMode',    # temporarily disabled 
    # 'ProductDesignType'  # temporarily disabled
]

__version__ = "3.0.0"  # Major version bump for new probabilistic framework
__author__ = "Advanced Probabilistic Insurance Analysis Team"

# Framework usage guidance
def get_framework_usage_guide():
    """
    Get usage guidance for the refactored modular framework
    """
    return """
    🚀 REFACTORED INSURANCE ANALYSIS FRAMEWORK USAGE GUIDE
    
    ✅ CORE DATA STRUCTURES (parametric_engine.py):
    
    ```python
    from insurance_analysis_refactored.core import (
        ParametricProduct,          # Core product data structure
        ProductPerformance,         # Performance metrics data
        ParametricIndexType,        # Index type enums
        PayoutFunctionType,         # Payout function enums
        ParametricInsuranceEngine   # Basic product creation
    )
    
    # Create products using lightweight engine
    engine = ParametricInsuranceEngine()
    product = engine.create_parametric_product(
        product_id="TEST_001",
        name="Test Product",
        description="Test parametric product",
        index_type=ParametricIndexType.CAT_IN_CIRCLE,
        payout_function_type=PayoutFunctionType.STEP,
        trigger_thresholds=[33.0, 42.0, 58.0],
        payout_amounts=[1e8, 3e8, 5e8],
        max_payout=5e8
    )
    ```
    
    ✅ SPECIALIZED MODULES (for complex functionality):
    
    ```python
    # Spatial analysis
    from insurance_analysis_refactored.core.enhanced_spatial_analysis import EnhancedCatInCircleAnalyzer
    
    # Product generation
    from insurance_analysis_refactored.core.saffir_simpson_products import generate_steinmann_2023_products
    
    # Performance evaluation 
    from insurance_analysis_refactored.core.skill_evaluator import SkillScoreEvaluator
    
    # Product management
    from insurance_analysis_refactored.core.product_manager import InsuranceProductManager
    
    # Premium calculation
    from insurance_analysis_refactored.core.technical_premium_calculator import TechnicalPremiumCalculator
    ```
    
    ✅ ADVANCED TECHNICAL PREMIUM ANALYSIS:
    
    ```python
    from insurance_analysis_refactored.core import (
        TechnicalPremiumCalculator,
        MarketAcceptabilityAnalyzer, 
        MultiObjectiveOptimizer,
        TechnicalPremiumVisualizer,
        create_standard_technical_premium_calculator,
        create_standard_market_analyzer,
        create_standard_multi_objective_optimizer,
        create_standard_visualizer
    )
    
    # Complete workflow
    premium_calc = create_standard_technical_premium_calculator()
    market_analyzer = create_standard_market_analyzer()
    optimizer = create_standard_multi_objective_optimizer(premium_calc, market_analyzer)
    visualizer = create_standard_visualizer()
    
    # Run multi-objective optimization
    results = optimizer.optimize(candidate_products, actual_losses, hazard_indices, config)
    
    # Generate visualization and reports
    visualizer.visualize_multi_objective_results(results)
    ```
    
    ⚠️  REMOVED FUNCTIONALITY:
    - CatInCircleExtractor -> use enhanced_spatial_analysis.py
    - PayoutFunction classes -> use technical_premium_calculator.py 
    - calculate_correct_step_payouts -> deprecated
    - calculate_crps_score -> use skill_scores module
    
    ✅ BAYESIAN ANALYSIS (separate module):
    
    ```python
    from bayesian import (
        RobustBayesianAnalyzer,
        HierarchicalBayesianModel,
        ProbabilisticLossDistributionGenerator,
        WeightSensitivityAnalyzer
    )
    ```
    
    📋 REFACTORING SUMMARY:
    - parametric_engine.py: Simplified to core data structures only
    - Complex functionality moved to specialized modules
    - Removed duplicate implementations
    - Maintained backward compatibility for data structures
    - Fixed import issues in __init__.py
    """