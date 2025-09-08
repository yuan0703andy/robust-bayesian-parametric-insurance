"""
Stage 4: Model Selection with VI
階段4：使用VI的模型海選

Two-Step Architecture Implementation:
Step 1: Hierarchical Loss Predictor - Pure loss prediction using CRPS-VI
Step 2: Product Evaluator - Evaluate 350 products using trained predictor

Legacy components also available for backward compatibility.
"""

# Step 1: Pure Hierarchical Loss Predictor
from .hierarchical_loss_predictor_vi import HierarchicalLossPredictorVI

# Step 2: Product Evaluator  
from .product_evaluator import ProductEvaluator, BasisRiskCalculator

# Legacy components (for backward compatibility)
from .basis_risk_vi import (
    DifferentiableCRPS,
    ParametricPayoutFunction,
    BasisRiskAwareVI
)

from .model_selector import (
    ModelCandidate,
    HyperparameterConfig,
    ModelSelectionResult,
    ModelSelectorWithHyperparamOptimization
)

# Provide convenient aliases
ModelSelector = ModelSelectorWithHyperparamOptimization

__all__ = [
    # Two-Step Architecture (Primary)
    'HierarchicalLossPredictorVI',   # Step 1: Loss Predictor
    'ProductEvaluator',              # Step 2: Product Evaluator
    'BasisRiskCalculator',           # Supporting utility
    
    # Legacy VI components (Secondary)
    'DifferentiableCRPS',
    'ParametricPayoutFunction', 
    'BasisRiskAwareVI',
    
    # Legacy Model selection (Secondary)
    'ModelCandidate',
    'HyperparameterConfig',
    'ModelSelectionResult',
    'ModelSelectorWithHyperparamOptimization',
    'ModelSelector'  # 便捷別名
]