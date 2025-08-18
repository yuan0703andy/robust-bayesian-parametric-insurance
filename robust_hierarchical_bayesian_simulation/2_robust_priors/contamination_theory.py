#!/usr/bin/env python3
"""
Contamination Theory Module
污染理論模組

從 epsilon_contamination.py 拆分出的理論基礎部分
包含基本的數學理論、枚舉定義和核心概念

數學基礎:
Γ_ε = {π(θ): π(θ) = (1-ε)π₀(θ) + εq(θ), for all q ∈ Q}

核心功能:
- 污染分布類別定義
- ε-contamination 規格配置
- 基本理論工具

Author: Research Team  
Date: 2025-01-17
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from scipy import stats

# ========================================
# 基本枚舉定義
# ========================================

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
    """
    ε-污染規格配置
    ε-Contamination specification configuration
    """
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
    """
    污染程度估計結果
    Contamination level estimation results
    """
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
    """
    穩健後驗結果
    Robust posterior analysis results
    """
    worst_case_posterior: Dict[str, np.ndarray]   # 最壞情況後驗
    best_case_posterior: Dict[str, np.ndarray]    # 最佳情況後驗
    robust_credible_intervals: Dict[str, Tuple[float, float]]  # 穩健可信區間
    sensitivity_analysis: Dict[str, Any]          # 敏感性分析
    robustness_measures: Dict[str, float]         # 穩健性指標

# ========================================
# 理論基礎函數
# ========================================

def contamination_bound(epsilon: float, 
                       base_measure: float, 
                       contamination_measure: float) -> Tuple[float, float]:
    """
    計算污染邊界
    
    對於 π(θ) = (1-ε)π₀(θ) + εq(θ)，計算測度的邊界
    
    Parameters:
    -----------
    epsilon : float
        污染程度
    base_measure : float
        基準分布的測度值
    contamination_measure : float
        污染分布的測度值
        
    Returns:
    --------
    Tuple[float, float]
        (下界, 上界)
    """
    lower_bound = (1 - epsilon) * base_measure
    upper_bound = (1 - epsilon) * base_measure + epsilon * contamination_measure
    return lower_bound, upper_bound

def worst_case_risk(epsilon: float,
                   risk_function: callable,
                   base_distribution: Any,
                   contamination_class: ContaminationDistributionClass) -> float:
    """
    計算最壞情況風險
    
    max_{q∈Q} R[(1-ε)π₀ + εq]
    
    Parameters:
    -----------
    epsilon : float
        污染程度
    risk_function : callable
        風險函數
    base_distribution : 
        基準分布
    contamination_class : ContaminationDistributionClass
        污染分布類別
        
    Returns:
    --------
    float
        最壞情況風險
    """
    # 簡化實現 - 實際應用中需要更複雜的優化
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
    """
    計算穩健性半徑
    
    衡量在給定污染程度下後驗的變化範圍
    
    Parameters:
    -----------
    epsilon_max : float
        最大污染程度
    base_posterior : np.ndarray
        基準後驗樣本
    contamination_distributions : List[np.ndarray]
        可能的污染分布樣本
        
    Returns:
    --------
    float
        穩健性半徑
    """
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

def sensitivity_to_epsilon(epsilon_values: np.ndarray,
                         base_distribution: Any,
                         contamination_distribution: Any,
                         metric_function: callable) -> Dict[str, np.ndarray]:
    """
    分析對ε的敏感性
    
    Parameters:
    -----------
    epsilon_values : np.ndarray
        ε值範圍
    base_distribution : 
        基準分布
    contamination_distribution : 
        污染分布
    metric_function : callable
        度量函數
        
    Returns:
    --------
    Dict[str, np.ndarray]
        敏感性分析結果
    """
    metrics = []
    
    for eps in epsilon_values:
        # 模擬混合分布
        mixed_distribution = create_mixed_distribution(
            base_distribution, contamination_distribution, eps
        )
        
        # 計算度量
        metric_value = metric_function(mixed_distribution)
        metrics.append(metric_value)
    
    metrics = np.array(metrics)
    
    return {
        "epsilon_values": epsilon_values,
        "metric_values": metrics,
        "sensitivity": np.gradient(metrics, epsilon_values),
        "max_sensitivity": np.max(np.abs(np.gradient(metrics, epsilon_values))),
        "sensitivity_at_zero": np.gradient(metrics, epsilon_values)[0] if len(metrics) > 1 else 0
    }

def create_mixed_distribution(base_dist: Any, 
                            contamination_dist: Any, 
                            epsilon: float) -> Any:
    """
    創建混合分布 π(θ) = (1-ε)π₀(θ) + εq(θ)
    
    Parameters:
    -----------
    base_dist : 
        基準分布
    contamination_dist : 
        污染分布
    epsilon : float
        混合權重
        
    Returns:
    --------
    混合分布對象
    """
    # 這是簡化實現，實際應用中需要更複雜的分布處理
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
        """
        生成颱風特定的極值分布
        
        Parameters:
        -----------
        location : float
            位置參數
        scale : float
            尺度參數
        shape : float
            形狀參數
            
        Returns:
        --------
        scipy.stats.genextreme
            廣義極值分布
        """
        return stats.genextreme(c=shape, loc=location, scale=scale)
    
    @staticmethod
    def generate_heavy_tailed(df: float = 3, 
                            location: float = 0, 
                            scale: float = 1) -> stats.t:
        """
        生成重尾分布 (Student-t)
        
        Parameters:
        -----------
        df : float
            自由度
        location : float
            位置參數
        scale : float
            尺度參數
            
        Returns:
        --------
        scipy.stats.t
            Student-t分布
        """
        return stats.t(df=df, loc=location, scale=scale)
    
    @staticmethod
    def generate_moment_bounded(a: float = -2, 
                              b: float = 2) -> stats.uniform:
        """
        生成矩有界分布 (均勻分布)
        
        Parameters:
        -----------
        a : float
            下界
        b : float
            上界
            
        Returns:
        --------
        scipy.stats.uniform
            均勻分布
        """
        return stats.uniform(loc=a, scale=b-a)

def test_contamination_theory():
    """測試污染理論功能"""
    print("🧪 測試污染理論模組...")
    
    # 測試基本配置
    print("✅ 測試ε-contamination規格:")
    spec = EpsilonContaminationSpec()
    print(f"   經驗ε值: {spec.empirical_epsilon:.4f}")
    print(f"   污染類別: {spec.contamination_class.value}")
    
    # 測試污染邊界計算
    print("✅ 測試污染邊界:")
    lower, upper = contamination_bound(0.1, 5.0, 15.0)
    print(f"   邊界: [{lower:.2f}, {upper:.2f}]")
    
    # 測試分布生成器
    print("✅ 測試分布生成器:")
    typhoon_dist = ContaminationDistributionGenerator.generate_typhoon_specific()
    samples = typhoon_dist.rvs(100)
    print(f"   颱風分布樣本均值: {np.mean(samples):.3f}")
    
    print("✅ 污染理論測試完成")

if __name__ == "__main__":
    test_contamination_theory()