#!/usr/bin/env python3
"""
Robust Priors Module - Stage 2
穩健先驗模組 - 階段2

ε-contamination 穩健貝氏先驗建模

主要組件:
- contamination_theory: 污染理論基礎
- prior_contamination: 先驗污染分析
- likelihood_contamination: 似然污染分析 (待實現)
- double_contamination: 雙重污染分析 (待實現)

Author: Research Team
Date: 2025-01-17
"""

from .contamination_theory import (
    EpsilonContaminationSpec,
    ContaminationDistributionClass,
    ContaminationEstimateResult,
    RobustPosteriorResult,
    contamination_bound,
    worst_case_risk,
    ContaminationDistributionGenerator
)

from .prior_contamination import PriorContaminationAnalyzer

__all__ = [
    "EpsilonContaminationSpec",
    "ContaminationDistributionClass", 
    "ContaminationEstimateResult",
    "RobustPosteriorResult",
    "PriorContaminationAnalyzer",
    "contamination_bound",
    "worst_case_risk",
    "ContaminationDistributionGenerator"
]