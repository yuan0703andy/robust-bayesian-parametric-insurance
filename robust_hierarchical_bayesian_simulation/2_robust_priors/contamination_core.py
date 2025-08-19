#!/usr/bin/env python3
"""
Contamination Core Module
污染核心模組

核心ε-contamination理論與基本實現
整合自 contamination_theory.py 和部分 epsilon_contamination.py

數學基礎:
Γ_ε = {π(θ): π(θ) = (1-ε)π₀(θ) + εq(θ), for all q ∈ Q}

核心功能:
- 基本類型定義與配置
- 理論計算函數
- 污染分布生成器
- 雙重ε-contamination實現

Author: Research Team  
Date: 2025-08-19
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from scipy import stats
import warnings

# ========================================
# 基本類型定義
# ========================================

class ContaminationDistributionClass(Enum):
    """污染分佈類別 Q 的定義"""
    ALL_DISTRIBUTIONS = "all"                    # 所有概率分佈
    TYPHOON_SPECIFIC = "typhoon_specific"        # 颱風特定極值分佈
    HEAVY_TAILED = "heavy_tailed"               # 重尾分佈
    MOMENT_BOUNDED = "moment_bounded"           # 矩有界分佈
    UNIMODAL = "unimodal"                      # 單峰分佈

class RobustnessCriterion(Enum):
    """強健性準則"""
    WORST_CASE = "worst_case"                    # 最壞情況
    AVERAGE_CASE = "average_case"                # 平均情況
    MINIMAX = "minimax"                          # 最小最大
    MAXIMUM_ENTROPY = "maximum_entropy"          # 最大熵

class EstimationMethod(Enum):
    """ε估計方法"""
    EMPIRICAL_FREQUENCY = "empirical_frequency"  # 經驗頻率
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"   # K-S檢驗
    ANDERSON_DARLING = "anderson_darling"        # A-D檢驗
    BAYESIAN_MODEL_SELECTION = "bayesian_model_selection"  # 貝氏模型選擇
    CROSS_VALIDATION = "cross_validation"        # 交叉驗證

# ========================================
# 配置結構
# ========================================

@dataclass
class EpsilonContaminationSpec:
    """ε-污染規格配置"""
    epsilon_range: Tuple[float, float] = (0.01, 0.20)  # 污染程度範圍
    contamination_class: ContaminationDistributionClass = ContaminationDistributionClass.TYPHOON_SPECIFIC
    nominal_prior_family: str = "normal"                # 基準先驗分佈族
    contamination_prior_family: str = "gev"             # 污染先驗分佈族
    robustness_criterion: RobustnessCriterion = RobustnessCriterion.WORST_CASE
    
    # 颱風特定參數
    typhoon_frequency_per_year: float = 3.2            # 年均颱風頻率
    simulation_years: int = 365                         # 模擬天數
    
    def __post_init__(self):
        # 類型轉換支援
        if isinstance(self.contamination_class, str):
            self.contamination_class = ContaminationDistributionClass(self.contamination_class)
        if isinstance(self.robustness_criterion, str):
            self.robustness_criterion = RobustnessCriterion(self.robustness_criterion)
    
    @property
    def empirical_epsilon(self) -> float:
        """基於颱風頻率的經驗ε值"""
        return self.typhoon_frequency_per_year / self.simulation_years

@dataclass 
class ContaminationEstimateResult:
    """污染程度估計結果"""
    epsilon_estimates: Dict[str, float]          # 不同方法的ε估計
    epsilon_consensus: float                     # 共識估計
    epsilon_uncertainty: float                  # 估計不確定性
    thresholds: Dict[str, float]                # 各種閾值
    test_statistics: Dict[str, float]           # 檢驗統計量
    p_values: Dict[str, float]                  # p值
    method_weights: Dict[str, float]            # 方法權重
    
    def __post_init__(self):
        if not self.epsilon_estimates:
            raise ValueError("至少需要一個ε估計")
        
        # 如果沒有提供共識估計，計算加權平均
        if self.epsilon_consensus == 0:
            if self.method_weights:
                self.epsilon_consensus = sum(
                    self.method_weights.get(method, 1.0) * estimate 
                    for method, estimate in self.epsilon_estimates.items()
                ) / sum(self.method_weights.values())
            else:
                self.epsilon_consensus = np.mean(list(self.epsilon_estimates.values()))

@dataclass
class RobustPosteriorResult:
    """穩健後驗結果"""
    worst_case_posterior: Dict[str, np.ndarray]   # 最壞情況後驗
    best_case_posterior: Dict[str, np.ndarray]    # 最佳情況後驗
    robust_credible_intervals: Dict[str, Tuple[float, float]]  # 穩健可信區間
    sensitivity_analysis: Dict[str, Any]          # 敏感性分析
    robustness_measures: Dict[str, float]         # 穩健性指標

# ========================================
# 理論計算函數
# ========================================

def contamination_bound(epsilon: float, 
                       base_measure: float, 
                       contamination_measure: float) -> Tuple[float, float]:
    """
    計算污染邊界
    對於 π(θ) = (1-ε)π₀(θ) + εq(θ)，計算測度的邊界
    """
    lower_bound = (1 - epsilon) * base_measure
    upper_bound = (1 - epsilon) * base_measure + epsilon * contamination_measure
    return lower_bound, upper_bound

def worst_case_risk(epsilon: float,
                   risk_function: callable,
                   base_distribution: Any,
                   contamination_class: ContaminationDistributionClass) -> float:
    """計算最壞情況風險 max_{q∈Q} R[(1-ε)π₀ + εq]"""
    base_risk = risk_function(base_distribution)
    
    # 根據污染類別估計最大可能風險增加
    if contamination_class == ContaminationDistributionClass.HEAVY_TAILED:
        contamination_multiplier = 10.0  # 重尾分布可能帶來的風險倍數
    elif contamination_class == ContaminationDistributionClass.TYPHOON_SPECIFIC:
        contamination_multiplier = 5.0   # 颱風特定風險倍數
    else:
        contamination_multiplier = 2.0   # 一般情況
    
    worst_case = (1 - epsilon) * base_risk + epsilon * contamination_multiplier * base_risk
    return worst_case

def compute_robustness_radius(epsilon_max: float,
                            base_posterior: np.ndarray,
                            contamination_distributions: List[np.ndarray]) -> float:
    """計算穩健性半徑"""
    base_mean = np.mean(base_posterior)
    base_std = np.std(base_posterior)
    
    max_deviation = 0.0
    
    for contamination_samples in contamination_distributions:
        # 計算污染後的後驗
        contaminated_posterior = (
            (1 - epsilon_max) * base_posterior[:len(contamination_samples)] + 
            epsilon_max * contamination_samples
        )
        
        contaminated_mean = np.mean(contaminated_posterior)
        contaminated_std = np.std(contaminated_posterior)
        
        # 計算標準化偏差
        mean_deviation = abs(contaminated_mean - base_mean) / base_std
        std_deviation = abs(contaminated_std - base_std) / base_std
        
        total_deviation = mean_deviation + std_deviation
        max_deviation = max(max_deviation, total_deviation)
    
    return max_deviation

def create_mixed_distribution(base_dist: Any, 
                            contamination_dist: Any, 
                            epsilon: float) -> Any:
    """創建混合分布 π(θ) = (1-ε)π₀(θ) + εq(θ)"""
    
    class MixedDistribution:
        def __init__(self, base, contamination, eps):
            self.base = base
            self.contamination = contamination
            self.epsilon = eps
        
        def rvs(self, size=1000):
            """生成混合分布樣本"""
            n_contamination = int(size * self.epsilon)
            n_base = size - n_contamination
            
            if hasattr(self.base, 'rvs'):
                base_samples = self.base.rvs(n_base)
            else:
                base_samples = np.random.choice(self.base, n_base)
            
            if hasattr(self.contamination, 'rvs'):
                contamination_samples = self.contamination.rvs(n_contamination)
            else:
                contamination_samples = np.random.choice(self.contamination, n_contamination)
            
            mixed_samples = np.concatenate([base_samples, contamination_samples])
            np.random.shuffle(mixed_samples)
            return mixed_samples
        
        def mean(self):
            """混合分布均值"""
            if hasattr(self.base, 'mean'):
                base_mean = self.base.mean()
            else:
                base_mean = np.mean(self.base)
                
            if hasattr(self.contamination, 'mean'):
                contamination_mean = self.contamination.mean()
            else:
                contamination_mean = np.mean(self.contamination)
            
            return (1 - self.epsilon) * base_mean + self.epsilon * contamination_mean
    
    return MixedDistribution(base_dist, contamination_dist, epsilon)

# ========================================
# 污染分布生成器
# ========================================

class ContaminationDistributionGenerator:
    """污染分布生成器"""
    
    @staticmethod
    def generate_typhoon_specific(location: float = 0, 
                                scale: float = 1, 
                                shape: float = 0.1) -> stats.genextreme:
        """生成颱風特定的極值分布"""
        return stats.genextreme(c=shape, loc=location, scale=scale)
    
    @staticmethod
    def generate_heavy_tailed(df: float = 3, 
                            location: float = 0, 
                            scale: float = 1) -> stats.t:
        """生成重尾分布 (Student-t)"""
        return stats.t(df=df, loc=location, scale=scale)
    
    @staticmethod
    def generate_moment_bounded(a: float = -2, 
                              b: float = 2) -> stats.uniform:
        """生成矩有界分布 (均勻分布)"""
        return stats.uniform(loc=a, scale=b-a)

# ========================================
# 雙重污染實現 (Double ε-contamination)
# ========================================

class DoubleEpsilonContamination:
    """
    Double ε-contamination Implementation
    實現 Prior + Likelihood 雙重 ε-contamination
    
    Mathematical Foundation:
    Prior: π(θ) = (1-ε₁) × π₀(θ) + ε₁ × πc(θ)
    Likelihood: p(y|θ) = (1-ε₂) × L₀(y|θ) + ε₂ × Lc(y|θ)
    """
    
    def __init__(self, 
                 epsilon_prior: float = 0.1, 
                 epsilon_likelihood: float = 0.1,
                 prior_contamination_type: str = 'heavy_tailed',
                 likelihood_contamination_type: str = 'outliers'):
        self.epsilon_prior = epsilon_prior
        self.epsilon_likelihood = epsilon_likelihood
        self.prior_contamination_type = prior_contamination_type
        self.likelihood_contamination_type = likelihood_contamination_type
        
    def create_contaminated_prior(self, base_prior_params: Dict) -> Dict:
        """Create contaminated prior: π(θ) = (1-ε₁) × π₀(θ) + ε₁ × πc(θ)"""
        contaminated_params = base_prior_params.copy()
        
        if self.prior_contamination_type == 'heavy_tailed':
            # Add heavy tails to prior
            contaminated_params['scale'] *= (1 + 2 * self.epsilon_prior)
            contaminated_params['df'] = max(2, base_prior_params.get('df', 30) * (1 - self.epsilon_prior))
            
        elif self.prior_contamination_type == 'extreme_value':
            # Mix with extreme value distribution
            contaminated_params['location'] += self.epsilon_prior * base_prior_params.get('scale', 1) * 3
            contaminated_params['shape'] = 0.1 + self.epsilon_prior * 0.3  # GEV shape parameter
            
        elif self.prior_contamination_type == 'misspecified':
            # Systematic misspecification
            contaminated_params['location'] *= (1 - self.epsilon_prior * 0.5)
            contaminated_params['scale'] *= (1 + self.epsilon_prior)
            
        contaminated_params['contamination_info'] = {
            'epsilon': self.epsilon_prior,
            'type': self.prior_contamination_type,
            'base_params': base_prior_params
        }
        
        return contaminated_params
    
    def create_contaminated_likelihood(self, data: np.ndarray, clean_fraction: float = None) -> np.ndarray:
        """Create contaminated likelihood: p(y|θ) = (1-ε₂) × L₀(y|θ) + ε₂ × Lc(y|θ)"""
        n = len(data)
        
        if clean_fraction is None:
            clean_fraction = 1 - self.epsilon_likelihood
            
        n_clean = int(n * clean_fraction)
        n_contaminated = n - n_clean
        
        # Separate clean and contaminated data
        clean_data = data[:n_clean].copy()
        
        if self.likelihood_contamination_type == 'outliers':
            # Add outliers
            contaminated_data = np.concatenate([
                data[n_clean:n_clean + n_contaminated//2],
                np.random.uniform(data.min() * 3, data.max() * 3, n_contaminated - n_contaminated//2)
            ])
            
        elif self.likelihood_contamination_type == 'measurement_error':
            # Add measurement errors
            contaminated_data = data[n_clean:] + np.random.normal(0, data.std() * 2, n_contaminated)
            
        elif self.likelihood_contamination_type == 'extreme_events':
            # Add extreme events (e.g., typhoons)
            contaminated_data = np.random.exponential(data.mean() * 5, n_contaminated)
            
        else:
            contaminated_data = data[n_clean:]
            
        # Combine data
        mixed_data = np.concatenate([clean_data, contaminated_data])
        np.random.shuffle(mixed_data)
        
        return mixed_data
    
    def compute_robust_posterior(self, 
                                data: np.ndarray,
                                base_prior_params: Dict,
                                likelihood_params: Dict) -> Dict:
        """Compute robust posterior under double contamination"""
        # Create contaminated prior
        contaminated_prior = self.create_contaminated_prior(base_prior_params)
        
        # Process data with contamination
        contaminated_data = self.create_contaminated_likelihood(data)
        
        # Compute posterior (simplified analytical approximation)
        n = len(contaminated_data)
        data_mean = np.mean(contaminated_data)
        data_var = np.var(contaminated_data)
        
        # Prior parameters
        prior_mean = contaminated_prior.get('location', 0)
        prior_var = contaminated_prior.get('scale', 1) ** 2
        
        # Posterior computation (conjugate update for Normal-Normal model)
        posterior_precision = 1/prior_var + n/data_var
        posterior_var = 1/posterior_precision
        posterior_mean = (prior_mean/prior_var + n*data_mean/data_var) / posterior_precision
        
        # Robustness adjustment for double contamination
        robustness_factor = (1 - self.epsilon_prior) * (1 - self.epsilon_likelihood)
        effective_sample_size = n * robustness_factor
        
        # Inflated uncertainty due to contamination
        posterior_var_robust = posterior_var / robustness_factor
        
        return {
            'posterior_mean': posterior_mean,
            'posterior_std': np.sqrt(posterior_var_robust),
            'effective_sample_size': effective_sample_size,
            'epsilon_prior': self.epsilon_prior,
            'epsilon_likelihood': self.epsilon_likelihood,
            'robustness_factor': robustness_factor,
            'contamination_impact': {
                'prior_shift': abs(prior_mean - base_prior_params.get('location', 0)),
                'variance_inflation': posterior_var_robust / posterior_var,
                'sample_size_reduction': (n - effective_sample_size) / n
            }
        }

# ========================================
# 便利函數
# ========================================

def create_typhoon_contamination_spec(epsilon_range: Tuple[float, float] = (0.01, 0.15)) -> EpsilonContaminationSpec:
    """創建標準颱風特定污染規格"""
    return EpsilonContaminationSpec(
        epsilon_range=epsilon_range,
        contamination_class=ContaminationDistributionClass.TYPHOON_SPECIFIC,
        nominal_prior_family="normal",
        contamination_prior_family="gev",
        robustness_criterion=RobustnessCriterion.WORST_CASE
    )

def demonstrate_dual_process_nature(data: np.ndarray, epsilon: float = 0.05) -> Dict[str, Any]:
    """演示雙重過程特性：(1-ε) 正常天氣 + ε 颱風事件"""
    
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

# ========================================
# 模組導出
# ========================================

__all__ = [
    # 類型定義
    'ContaminationDistributionClass',
    'RobustnessCriterion', 
    'EstimationMethod',
    # 配置結構
    'EpsilonContaminationSpec',
    'ContaminationEstimateResult',
    'RobustPosteriorResult',
    # 理論函數
    'contamination_bound',
    'worst_case_risk',
    'compute_robustness_radius',
    'create_mixed_distribution',
    # 核心類別
    'ContaminationDistributionGenerator',
    'DoubleEpsilonContamination',
    # 便利函數
    'create_typhoon_contamination_spec',
    'demonstrate_dual_process_nature'
]