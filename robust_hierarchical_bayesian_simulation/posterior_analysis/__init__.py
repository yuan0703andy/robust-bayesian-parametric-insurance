"""
Stage 7: Posterior Analysis
階段7：後驗分析

Posterior approximation, credible intervals, and predictive checks.
"""

try:
    from .credible_intervals import RobustCredibleIntervalCalculator as CredibleIntervalCalculator
except ImportError:
    CredibleIntervalCalculator = None

try:
    from .posterior_approximation import MixedPredictiveEstimation as PosteriorApproximation
except ImportError:
    PosteriorApproximation = None

try:
    from .predictive_checks import PPCValidator as PosteriorPredictiveChecker
except ImportError:
    PosteriorPredictiveChecker = None