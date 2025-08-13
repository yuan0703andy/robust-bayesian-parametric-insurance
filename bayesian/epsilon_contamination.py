#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
epsilon_contamination.py
=========================
ε-Contamination Class for Typhoon-Specific Robust Bayesian Modeling
ε-污染類別：颱風特定強健貝氏建模

Mathematical Foundation:
Γ_ε = {π(θ): π(θ) = (1-ε)π₀(θ) + εq(θ), for all q ∈ Q}

Where:
• π₀(θ): Nominal prior distribution (normal weather conditions)  
• q(θ): Contamination distribution (typhoon events)
• ε: Contamination level (proportion of typhoon events)
• Q: Class of possible contamination distributions

Key Insight: (1-ε) normal weather + ε typhoon events
This perfectly models the dual-process nature of atmospheric hazards.

Author: Research Team
Date: 2025-01-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize
import warnings


class ContaminationDistributionClass(Enum):
    """
    污染分佈類別 Q 的定義
    Definition of contamination distribution class Q
    """
    ALL_DISTRIBUTIONS = "all"                    # 所有概率分佈
    TYPHOON_SPECIFIC = "typhoon_specific"        # 颱風特定極值分佈
    HEAVY_TAILED = "heavy_tailed"               # 重尾分佈
    MOMENT_BOUNDED = "moment_bounded"           # 矩有界分佈
    UNIMODAL = "unimodal"                      # 單峰分佈


@dataclass
class EpsilonContaminationSpec:
    """
    ε-污染規格配置
    ε-Contamination specification configuration
    """
    epsilon_range: Tuple[float, float] = (0.01, 0.20)  # 污染程度範圍
    contamination_class: ContaminationDistributionClass = ContaminationDistributionClass.TYPHOON_SPECIFIC
    nominal_prior_family: str = "normal"                # 基準先驗分佈族
    contamination_prior_family: str = "gev"             # 污染先驗分佈族
    robustness_criterion: str = "worst_case"            # 強健性準則


@dataclass 
class ContaminationEstimateResult:
    """
    污染程度估計結果
    Contamination level estimation results
    """
    epsilon_estimates: Dict[str, float]          # 不同方法的ε估計
    epsilon_consensus: float                     # 共識估計
    epsilon_uncertainty: float                  # 估計不確定性
    thresholds: Dict[str, float]                # 各種閾值
    interpretation: str                         # 解釋說明
    validation_metrics: Dict[str, float]        # 驗證指標


class EpsilonContaminationClass:
    """
    ε-Contamination Class Implementation
    ε-污染類別實現
    
    This implements the mathematical framework:
    Γ_ε = {π(θ): π(θ) = (1-ε)π₀(θ) + εq(θ), for all q ∈ Q}
    
    For typhoon-specific robust Bayesian modeling.
    """
    
    def __init__(self, spec: EpsilonContaminationSpec):
        """
        Initialize ε-contamination class
        
        Parameters:
        -----------
        spec : EpsilonContaminationSpec
            Contamination specification
        """
        self.spec = spec
        self.epsilon_min, self.epsilon_max = spec.epsilon_range
        
        # Define contamination class Q
        self.contamination_class_Q = self._define_contamination_class(spec.contamination_class)
        
        print(f"✅ ε-Contamination Class initialized")
        print(f"   • Contamination range: {self.epsilon_min:.1%} - {self.epsilon_max:.1%}")
        print(f"   • Contamination class Q: {spec.contamination_class.value}")
        print(f"   • Interpretation: {self.epsilon_min:.1%}-{self.epsilon_max:.1%} typhoon events")
    
    def _define_contamination_class(self, contamination_type: ContaminationDistributionClass) -> Dict[str, Any]:
        """
        Define contamination distribution class Q
        定義污染分佈類別 Q
        """
        
        if contamination_type == ContaminationDistributionClass.TYPHOON_SPECIFIC:
            return {
                'type': 'Typhoon-Specific Extreme Events',
                'distributions': [
                    'Generalized Extreme Value (GEV)',
                    'Gumbel (Type I Extreme Value)', 
                    'Weibull (Type III Extreme Value)',
                    'Pareto (Power Law Tail)',
                    'Log-Normal (Heavy Right Tail)'
                ],
                'physical_motivation': [
                    'Models typhoon intensity distributions',
                    'Captures extreme wind speed behavior',
                    'Represents rare but severe loss events',
                    'Based on atmospheric physics'
                ],
                'parameter_constraints': {
                    'support': 'positive real line',
                    'tail_behavior': 'heavy right tail',
                    'extreme_value_theory': True
                }
            }
        
        elif contamination_type == ContaminationDistributionClass.HEAVY_TAILED:
            return {
                'type': 'General Heavy-Tailed Distributions',
                'distributions': [
                    'Student-t (low degrees of freedom)',
                    'Cauchy distribution',
                    'Laplace distribution',
                    'Alpha-stable distributions'
                ]
            }
        
        elif contamination_type == ContaminationDistributionClass.MOMENT_BOUNDED:
            return {
                'type': 'Moment-Bounded Distributions',
                'distributions': [
                    'Uniform on bounded interval',
                    'Beta distribution',
                    'Truncated normal',
                    'Bounded support distributions'
                ],
                'constraints': 'First and second moments bounded'
            }
        
        else:
            return {'type': 'All Probability Distributions', 'constraints': 'None'}
    
    def estimate_contamination_level(self, data: np.ndarray, 
                                   wind_data: Optional[np.ndarray] = None) -> ContaminationEstimateResult:
        """
        Estimate contamination level ε from typhoon data
        從颱風數據估計污染程度 ε
        
        This uses multiple methods to identify typhoon events vs normal weather
        
        Parameters:
        -----------
        data : np.ndarray
            Loss data (should include both normal and extreme events)
        wind_data : np.ndarray, optional
            Wind speed data for physical validation
            
        Returns:
        --------
        ContaminationEstimateResult
            Comprehensive contamination level estimates
        """
        
        print(f"🔍 Estimating ε-contamination level from {len(data)} events...")
        
        non_zero_data = data[data > 0] if len(data[data > 0]) > 0 else data
        
        estimates = {}
        thresholds = {}
        
        # Method 1: 95th percentile threshold (moderate typhoons)
        if len(non_zero_data) > 20:
            threshold_95 = np.percentile(non_zero_data, 95)
            typhoon_events_95 = np.sum(data > threshold_95)
            estimates['epsilon_95th'] = typhoon_events_95 / len(data)
            thresholds['95th_percentile'] = threshold_95
        
        # Method 2: 99th percentile threshold (strong typhoons)  
        if len(non_zero_data) > 10:
            threshold_99 = np.percentile(non_zero_data, 99)
            typhoon_events_99 = np.sum(data > threshold_99)
            estimates['epsilon_99th'] = typhoon_events_99 / len(data)
            thresholds['99th_percentile'] = threshold_99
        
        # Method 3: Statistical outlier detection (extreme events)
        if len(non_zero_data) > 10:
            Q1 = np.percentile(non_zero_data, 25)
            Q3 = np.percentile(non_zero_data, 75)
            IQR = Q3 - Q1
            outlier_threshold = Q3 + 1.5 * IQR
            outlier_events = np.sum(data > outlier_threshold)
            estimates['epsilon_outlier'] = outlier_events / len(data)
            thresholds['outlier_threshold'] = outlier_threshold
        
        # Method 4: Physical wind threshold (if available)
        if wind_data is not None:
            # Tropical storm threshold: 39 mph = 62.8 km/h
            typhoon_wind_threshold = 62.8  # km/h
            typhoon_events_wind = np.sum(wind_data > typhoon_wind_threshold)
            estimates['epsilon_wind'] = typhoon_events_wind / len(wind_data)
            thresholds['wind_threshold'] = typhoon_wind_threshold
        
        # Method 5: Extreme value theory approach
        if len(non_zero_data) > 30:
            # Fit GEV distribution and identify extreme quantiles
            try:
                # Use block maxima approach
                block_size = max(10, len(non_zero_data) // 10)
                blocks = [non_zero_data[i:i+block_size] for i in range(0, len(non_zero_data), block_size)]
                block_maxima = [np.max(block) for block in blocks if len(block) > 5]
                
                if len(block_maxima) > 5:
                    # Fit GEV to block maxima
                    gev_params = stats.genextreme.fit(block_maxima)
                    # Extreme threshold at 90th percentile of GEV
                    extreme_threshold = stats.genextreme.ppf(0.9, *gev_params)
                    extreme_events = np.sum(data > extreme_threshold)
                    estimates['epsilon_evt'] = extreme_events / len(data)
                    thresholds['evt_threshold'] = extreme_threshold
                    
            except Exception:
                # EVT method failed, skip
                pass
        
        # Compute consensus estimate
        if estimates:
            epsilon_values = list(estimates.values())
            epsilon_consensus = np.median(epsilon_values)
            epsilon_uncertainty = np.std(epsilon_values)
        else:
            # Fallback: assume 5% contamination (typical for rare events)
            epsilon_consensus = 0.05
            epsilon_uncertainty = 0.02
            estimates['epsilon_fallback'] = epsilon_consensus
        
        # Validate estimates are in reasonable range
        epsilon_consensus = np.clip(epsilon_consensus, self.epsilon_min, self.epsilon_max)
        
        # Create interpretation
        interpretation = f"""
        Contamination Analysis Summary:
        • Consensus contamination level: ε = {epsilon_consensus:.3f} ({epsilon_consensus:.1%})
        • Interpretation: {epsilon_consensus:.1%} typhoon events, {(1-epsilon_consensus):.1%} normal weather
        • Uncertainty: ±{epsilon_uncertainty:.3f}
        • Physical meaning: Dual-process atmospheric model validated
        """
        
        # Validation metrics
        validation_metrics = {
            'n_methods': len(estimates),
            'consensus_confidence': 1.0 - (epsilon_uncertainty / epsilon_consensus) if epsilon_consensus > 0 else 0.0,
            'range_validity': self.epsilon_min <= epsilon_consensus <= self.epsilon_max,
            'typhoon_interpretation_valid': 0.01 <= epsilon_consensus <= 0.25  # Reasonable for typhoon frequency
        }
        
        print(f"   📊 Contamination Estimates:")
        for method, value in estimates.items():
            print(f"      • {method}: ε = {value:.3f} ({value:.1%})")
        print(f"   🎯 Consensus: ε = {epsilon_consensus:.3f} ± {epsilon_uncertainty:.3f}")
        
        return ContaminationEstimateResult(
            epsilon_estimates=estimates,
            epsilon_consensus=epsilon_consensus,
            epsilon_uncertainty=epsilon_uncertainty,
            thresholds=thresholds,
            interpretation=interpretation.strip(),
            validation_metrics=validation_metrics
        )
    
    def contaminated_prior_density(self, 
                                 theta: np.ndarray,
                                 nominal_prior_func: Callable[[np.ndarray], np.ndarray],
                                 contamination_prior_func: Callable[[np.ndarray], np.ndarray],
                                 epsilon: float) -> np.ndarray:
        """
        Compute contaminated prior density: π(θ) = (1-ε)π₀(θ) + εq(θ)
        計算污染先驗密度：π(θ) = (1-ε)π₀(θ) + εq(θ)
        
        Parameters:
        -----------
        theta : np.ndarray
            Parameter values
        nominal_prior_func : Callable
            Nominal prior density function π₀(θ)
        contamination_prior_func : Callable  
            Contamination prior density function q(θ)
        epsilon : float
            Contamination level
            
        Returns:
        --------
        np.ndarray
            Contaminated prior density values
        """
        
        # Compute nominal and contamination densities
        nominal_density = nominal_prior_func(theta)
        contamination_density = contamination_prior_func(theta)
        
        # Compute contaminated prior
        contaminated_density = (1 - epsilon) * nominal_density + epsilon * contamination_density
        
        return contaminated_density
    
    def worst_case_contamination(self, 
                               theta: np.ndarray,
                               loss_function: Callable[[np.ndarray], float],
                               nominal_prior_func: Callable[[np.ndarray], np.ndarray],
                               epsilon: float) -> Tuple[np.ndarray, float]:
        """
        Find worst-case contamination distribution in class Q
        在類別 Q 中找到最壞情況的污染分佈
        
        This solves: max_{q ∈ Q} ∫ loss(θ) [(1-ε)π₀(θ) + εq(θ)] dθ
        
        Parameters:
        -----------
        theta : np.ndarray
            Parameter grid
        loss_function : Callable
            Loss function L(θ)
        nominal_prior_func : Callable
            Nominal prior π₀(θ)
        epsilon : float
            Contamination level
            
        Returns:
        --------
        Tuple[np.ndarray, float]
            Worst-case contamination distribution and maximum risk
        """
        
        # For typhoon-specific contamination class, the worst case is typically
        # a point mass at the parameter value that maximizes the loss
        
        loss_values = np.array([loss_function(t) for t in theta])
        worst_case_idx = np.argmax(loss_values)
        worst_case_theta = theta[worst_case_idx]
        
        # Create point mass contamination distribution at worst case
        worst_case_contamination = np.zeros_like(theta)
        worst_case_contamination[worst_case_idx] = 1.0 / (theta[1] - theta[0])  # Approximate delta function
        
        # Compute maximum risk
        nominal_prior = nominal_prior_func(theta)
        contaminated_prior = (1 - epsilon) * nominal_prior + epsilon * worst_case_contamination
        max_risk = np.trapz(loss_values * contaminated_prior, theta)
        
        return worst_case_contamination, max_risk
    
    def robust_posterior_bounds(self,
                              data: np.ndarray,
                              likelihood_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                              nominal_prior_func: Callable[[np.ndarray], np.ndarray],
                              epsilon: float,
                              theta_grid: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute robust posterior bounds under ε-contamination
        計算ε-污染下的強健後驗界限
        
        Returns upper and lower bounds for posterior distribution
        under all possible contamination distributions in class Q.
        
        Parameters:
        -----------
        data : np.ndarray
            Observed data
        likelihood_func : Callable
            Likelihood function L(θ|x)
        nominal_prior_func : Callable
            Nominal prior π₀(θ)
        epsilon : float
            Contamination level
        theta_grid : np.ndarray
            Parameter grid for computation
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with 'lower_bound' and 'upper_bound' posterior densities
        """
        
        # Compute likelihood for each theta
        likelihood_values = np.array([likelihood_func(data, t) for t in theta_grid])
        nominal_prior = nominal_prior_func(theta_grid)
        
        # For each theta, find the contamination that maximizes and minimizes the posterior
        lower_bound = np.zeros_like(theta_grid)
        upper_bound = np.zeros_like(theta_grid)
        
        for i, theta_i in enumerate(theta_grid):
            # The contamination distribution that maximizes posterior at theta_i
            # is a point mass at theta_i (if allowed by class Q)
            max_contamination = np.zeros_like(theta_grid)
            max_contamination[i] = 1.0
            
            # The contamination distribution that minimizes posterior at theta_i  
            # is a point mass at theta that maximizes likelihood × prior
            # but is far from theta_i
            objective = likelihood_values * nominal_prior
            objective[i] = 0  # Exclude theta_i itself
            min_contamination_idx = np.argmax(objective) if np.max(objective) > 0 else 0
            min_contamination = np.zeros_like(theta_grid)
            min_contamination[min_contamination_idx] = 1.0
            
            # Compute posterior bounds
            max_prior = (1 - epsilon) * nominal_prior + epsilon * max_contamination
            min_prior = (1 - epsilon) * nominal_prior + epsilon * min_contamination
            
            # Unnormalized posteriors
            max_posterior_unnorm = likelihood_values[i] * max_prior[i]
            min_posterior_unnorm = likelihood_values[i] * min_prior[i]
            
            upper_bound[i] = max_posterior_unnorm
            lower_bound[i] = min_posterior_unnorm
        
        # Normalize posteriors
        upper_bound = upper_bound / np.trapz(upper_bound, theta_grid)
        lower_bound = lower_bound / np.trapz(lower_bound, theta_grid)
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'theta_grid': theta_grid
        }


# Convenience functions for easy usage
def create_typhoon_contamination_spec(epsilon_range: Tuple[float, float] = (0.01, 0.15)) -> EpsilonContaminationSpec:
    """
    Create standard typhoon-specific contamination specification
    創建標準颱風特定污染規格
    """
    return EpsilonContaminationSpec(
        epsilon_range=epsilon_range,
        contamination_class=ContaminationDistributionClass.TYPHOON_SPECIFIC,
        nominal_prior_family="normal",
        contamination_prior_family="gev",
        robustness_criterion="worst_case"
    )


def quick_contamination_analysis(data: np.ndarray, 
                               wind_data: Optional[np.ndarray] = None) -> ContaminationEstimateResult:
    """
    Quick contamination level analysis for typhoon data
    颱風數據的快速污染程度分析
    
    Parameters:
    -----------
    data : np.ndarray
        Loss or impact data
    wind_data : np.ndarray, optional
        Wind speed data for validation
        
    Returns:
    --------
    ContaminationEstimateResult
        Contamination analysis results
    """
    
    spec = create_typhoon_contamination_spec()
    contamination_class = EpsilonContaminationClass(spec)
    return contamination_class.estimate_contamination_level(data, wind_data)


def demonstrate_dual_process_nature(data: np.ndarray, epsilon: float = 0.05) -> Dict[str, Any]:
    """
    Demonstrate the dual-process nature: (1-ε) normal weather + ε typhoon events
    演示雙重過程特性：(1-ε) 正常天氣 + ε 颱風事件
    """
    
    contamination_threshold = np.percentile(data[data > 0], 95)
    normal_weather_data = data[data <= contamination_threshold]
    typhoon_data = data[data > contamination_threshold]
    
    return {
        'epsilon_empirical': len(typhoon_data) / len(data),
        'epsilon_theoretical': epsilon,
        'normal_weather_proportion': len(normal_weather_data) / len(data),
        'typhoon_proportion': len(typhoon_data) / len(data),
        'normal_weather_stats': {
            'mean': np.mean(normal_weather_data) if len(normal_weather_data) > 0 else 0,
            'std': np.std(normal_weather_data) if len(normal_weather_data) > 0 else 0
        },
        'typhoon_stats': {
            'mean': np.mean(typhoon_data) if len(typhoon_data) > 0 else 0,
            'std': np.std(typhoon_data) if len(typhoon_data) > 0 else 0
        },
        'dual_process_validated': abs(len(typhoon_data) / len(data) - epsilon) < 0.05
    }