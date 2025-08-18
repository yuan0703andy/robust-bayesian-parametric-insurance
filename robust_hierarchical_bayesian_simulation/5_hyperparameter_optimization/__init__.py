"""
Stage 5: Hyperparameter Optimization
階段5：超參數優化

Advanced hyperparameter optimization strategies for CRPS-based model selection.
"""

from .hyperparameter_optimizer import (
    HyperparameterSearchSpace,
    AdaptiveHyperparameterOptimizer,
    CrossValidatedHyperparameterSearch
)

# Keep weight_sensitivity for backward compatibility but mark as deprecated
from .weight_sensitivity import (
    WeightSensitivityConfig,
    WeightSensitivityAnalyzer
)

import warnings

def __getattr__(name):
    if name in ['WeightSensitivityConfig', 'WeightSensitivityAnalyzer']:
        warnings.warn(
            f"{name} is deprecated. Use AdaptiveHyperparameterOptimizer instead.",
            DeprecationWarning,
            stacklevel=2
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core hyperparameter optimization
    'HyperparameterSearchSpace',
    'AdaptiveHyperparameterOptimizer',
    'CrossValidatedHyperparameterSearch',
    
    # Deprecated (for backward compatibility)
    'WeightSensitivityConfig',
    'WeightSensitivityAnalyzer'
]