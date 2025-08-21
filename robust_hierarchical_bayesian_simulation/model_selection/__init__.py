"""
Stage 4: Model Selection with VI
階段4：使用VI的模型海選

Model selection with hyperparameter optimization using Basis-Risk-Aware VI.
"""

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

__all__ = [
    # VI components
    'DifferentiableCRPS',
    'ParametricPayoutFunction', 
    'BasisRiskAwareVI',
    
    # Model selection
    'ModelCandidate',
    'HyperparameterConfig',
    'ModelSelectionResult',
    'ModelSelectorWithHyperparamOptimization'
]