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
from .analysis_framework import (
    UnifiedAnalysisFramework, AnalysisConfig, AnalysisType, AnalysisResults,
    # Merged from unified_product_engine.py
    EvaluationMode, ProductDesignType
)
# Input adapter functionality now available through UnifiedAnalysisFramework
from .input_adapters import InputAdapter, CLIMADAInputAdapter, BayesianInputAdapter
# Legacy Bayesian components moved to bayesian/ module
# Keeping only minimal references for backward compatibility

# NOTE: Bayesian components moved to bayesian/ module
# All Bayesian-related functionality is now centralized in the bayesian/ module
from .unified_probabilistic_framework_full import (
    UnifiedProbabilisticFramework,
    UnifiedFrameworkConfig,
    UnifiedFrameworkResult,
    AnalysisPhase
)


__all__ = [
    # Main unified framework
    'UnifiedProbabilisticFramework',           # Main entry point
    'UnifiedFrameworkConfig',
    'UnifiedFrameworkResult',
    'AnalysisPhase',
    
    # Core insurance product components
    'ParametricInsuranceEngine',
    'SkillScoreEvaluator', 
    'InsuranceProductManager',
    'UnifiedAnalysisFramework',
    
    # Input adapters (merged from unified_product_engine.py)
    'InputAdapter',
    'CLIMADAInputAdapter', 
    'BayesianInputAdapter',
    
    # Core insurance data classes
    'ParametricProduct',
    'ProductPerformance',
    'SkillScoreResult',
    'ProductPortfolio',
    'AnalysisConfig',
    'AnalysisResults',
    
    # Insurance-focused enums
    'ParametricIndexType',
    'PayoutFunctionType',
    'SkillScoreType',
    'ProductStatus',
    'AnalysisType',
    # Merged from unified_product_engine.py
    'EvaluationMode',
    'ProductDesignType'
]

__version__ = "3.0.0"  # Major version bump for new probabilistic framework
__author__ = "Advanced Probabilistic Insurance Analysis Team"

# Framework usage guidance
def get_framework_usage_guide():
    """
    Get usage guidance for the new probabilistic framework
    """
    return """
    🚀 INSURANCE ANALYSIS FRAMEWORK USAGE GUIDE
    
    For parametric insurance product design and evaluation:
    
    ```python
    from insurance_analysis_refactored.core import (
        ParametricInsuranceEngine,
        SkillScoreEvaluator,
        InsuranceProductManager
    )
    
    # Create parametric insurance engine
    engine = ParametricInsuranceEngine()
    
    # Generate Steinmann-style products
    from .saffir_simpson_products import generate_steinmann_2023_products
    steinmann_structures, metadata = generate_steinmann_2023_products()
    
    # Convert to ParametricProduct objects
    products = []
    for structure in steinmann_structures:
        product = engine.create_parametric_product(
            product_id=structure.product_id,
            name=f"Steinmann Product {structure.product_id}",
            description=f"{structure.structure_type} threshold product",
            index_type=ParametricIndexType.CAT_IN_CIRCLE,
            payout_function_type=PayoutFunctionType.STEP,
            trigger_thresholds=structure.thresholds,
            payout_amounts=[structure.max_payout * p for p in structure.payouts],
            max_payout=structure.max_payout
        )
        products.append(product)
    
    # Evaluate product performance
    evaluator = SkillScoreEvaluator()
    performance = evaluator.evaluate_products(products, observed_losses)
    
    # Manage product portfolio
    manager = InsuranceProductManager()
    portfolio = manager.create_portfolio("main", product_ids, weights)
    ```
    
    For Bayesian analysis, use the bayesian/ module:
    
    ```python
    from bayesian import (
        RobustBayesianAnalyzer,
        HierarchicalBayesianModel,
        ProbabilisticLossDistributionGenerator
    )
    ```
    
    For visualization, use the visualization/ module:
    
    ```python
    from visualization import (
        BayesianVisualization,
        SteinmannVisualization
    )
    ```
    """