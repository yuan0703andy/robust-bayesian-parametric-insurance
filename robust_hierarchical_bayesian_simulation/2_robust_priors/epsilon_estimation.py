#!/usr/bin/env python3
"""
Epsilon Estimation Module
ε值估計模組

專門處理ε-contamination模型中ε值的估計功能
整合自 epsilon_contamination.py 和 prior_contamination.py

核心功能:
- 多種ε估計方法實現
- 數據驅動的污染程度分析
- 方法比較與驗證
- 先驗污染分析

Author: Research Team  
Date: 2025-08-19
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import stats
from scipy.optimize import minimize
import warnings

# 從核心模組導入
try:
    from .contamination_core import (
        EpsilonContaminationSpec, ContaminationEstimateResult, EstimationMethod,
        ContaminationDistributionClass, ContaminationDistributionGenerator
    )
except ImportError:
    from contamination_core import (
        EpsilonContaminationSpec, ContaminationEstimateResult, EstimationMethod,
        ContaminationDistributionClass, ContaminationDistributionGenerator
    )

# ========================================
# ε估計核心類別
# ========================================

class EpsilonEstimator:
    """
    ε值估計器
    實現多種污染程度估計方法
    """
    
    def __init__(self, spec: EpsilonContaminationSpec):
        self.spec = spec
        self.contamination_generator = ContaminationDistributionGenerator()
        self.estimation_cache: Dict[str, ContaminationEstimateResult] = {}
        
        print(f"📊 ε估計器初始化")
        print(f"   污染類別: {self.spec.contamination_class.value}")
        print(f"   ε範圍: {self.spec.epsilon_range[0]:.3f} - {self.spec.epsilon_range[1]:.3f}")
    
    def estimate_contamination_level(self, data: np.ndarray, 
                                   wind_data: Optional[np.ndarray] = None) -> ContaminationEstimateResult:
        """
        從颱風數據估計污染程度 ε
        使用多種方法識別颱風事件 vs 正常天氣
        """
        print(f"🔍 估計ε-contamination level從 {len(data)} 事件...")
        
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
        epsilon_consensus = np.clip(epsilon_consensus, self.spec.epsilon_range[0], self.spec.epsilon_range[1])
        
        # Validation metrics
        validation_metrics = {
            'n_methods': len(estimates),
            'consensus_confidence': 1.0 - (epsilon_uncertainty / epsilon_consensus) if epsilon_consensus > 0 else 0.0,
            'range_validity': self.spec.epsilon_range[0] <= epsilon_consensus <= self.spec.epsilon_range[1],
            'typhoon_interpretation_valid': 0.01 <= epsilon_consensus <= 0.25  # Reasonable for typhoon frequency
        }
        
        print(f"   📊 污染估計:")
        for method, value in estimates.items():
            print(f"      • {method}: ε = {value:.3f} ({value:.1%})")
        print(f"   🎯 共識: ε = {epsilon_consensus:.3f} ± {epsilon_uncertainty:.3f}")
        
        return ContaminationEstimateResult(
            epsilon_estimates=estimates,
            epsilon_consensus=epsilon_consensus,
            epsilon_uncertainty=epsilon_uncertainty,
            thresholds=thresholds,
            test_statistics={},  # Will be filled by method-specific estimators
            p_values={},        # Will be filled by method-specific estimators
            method_weights={}   # Can be added later
        )
    
    def estimate_from_statistical_tests(self, 
                                       data: np.ndarray,
                                       methods: List[EstimationMethod] = None) -> ContaminationEstimateResult:
        """使用統計檢驗方法估計ε值"""
        if methods is None:
            methods = [
                EstimationMethod.EMPIRICAL_FREQUENCY,
                EstimationMethod.KOLMOGOROV_SMIRNOV,
                EstimationMethod.ANDERSON_DARLING
            ]
        
        print(f"📈 統計檢驗估計ε值 (n={len(data)})...")
        
        epsilon_estimates = {}
        test_statistics = {}
        p_values = {}
        
        # Generate base distribution
        base_dist = self._create_base_distribution(data)
        
        for method in methods:
            print(f"   使用方法: {method.value}")
            
            if method == EstimationMethod.EMPIRICAL_FREQUENCY:
                epsilon, test_stat, p_val = self._estimate_empirical_frequency(data, base_dist)
            elif method == EstimationMethod.KOLMOGOROV_SMIRNOV:
                epsilon, test_stat, p_val = self._estimate_ks_test(data, base_dist)
            elif method == EstimationMethod.ANDERSON_DARLING:
                epsilon, test_stat, p_val = self._estimate_ad_test(data, base_dist)
            elif method == EstimationMethod.BAYESIAN_MODEL_SELECTION:
                epsilon, test_stat, p_val = self._estimate_bayesian_selection(data, base_dist)
            else:
                print(f"   ⚠️ 方法 {method.value} 尚未實現")
                continue
            
            epsilon_estimates[method.value] = epsilon
            test_statistics[method.value] = test_stat
            p_values[method.value] = p_val
            
            print(f"      ε估計: {epsilon:.4f}")
        
        # Compute consensus estimate and uncertainty
        consensus_epsilon = np.mean(list(epsilon_estimates.values()))
        uncertainty = np.std(list(epsilon_estimates.values()))
        
        # Set thresholds
        thresholds = {
            "lower_bound": max(0.0, consensus_epsilon - 2 * uncertainty),
            "upper_bound": min(1.0, consensus_epsilon + 2 * uncertainty),
            "empirical_threshold": self.spec.empirical_epsilon
        }
        
        result = ContaminationEstimateResult(
            epsilon_estimates=epsilon_estimates,
            epsilon_consensus=consensus_epsilon,
            epsilon_uncertainty=uncertainty,
            thresholds=thresholds,
            test_statistics=test_statistics,
            p_values=p_values,
            method_weights={}
        )
        
        # Cache result
        cache_key = f"statistical_{len(data)}_{hash(data.tobytes())}"
        self.estimation_cache[cache_key] = result
        
        print(f"✅ ε統計估計完成: {consensus_epsilon:.4f} ± {uncertainty:.4f}")
        return result
    
    def _create_base_distribution(self, data: np.ndarray):
        """創建基準分布"""
        if self.spec.nominal_prior_family == "normal":
            return stats.norm(loc=np.mean(data), scale=np.std(data))
        elif self.spec.nominal_prior_family == "lognormal":
            # 確保數據為正
            positive_data = data[data > 0]
            if len(positive_data) == 0:
                return stats.norm(loc=0, scale=1)
            log_data = np.log(positive_data)
            return stats.lognorm(s=np.std(log_data), scale=np.exp(np.mean(log_data)))
        elif self.spec.nominal_prior_family == "gamma":
            # 使用moment matching
            mean_val = np.mean(data)
            var_val = np.var(data)
            if var_val <= 0 or mean_val <= 0:
                return stats.gamma(a=1, scale=1)
            beta = mean_val / var_val
            alpha = mean_val * beta
            return stats.gamma(a=alpha, scale=1/beta)
        else:
            # 預設使用正態分布
            return stats.norm(loc=np.mean(data), scale=np.std(data))
    
    def _estimate_empirical_frequency(self, data: np.ndarray, base_dist) -> Tuple[float, float, float]:
        """經驗頻率法估計ε"""
        # 計算極值頻率
        threshold = np.percentile(data, 95)  # 95%分位數作為極值閾值
        extreme_count = np.sum(data > threshold)
        
        # 估計ε為極值頻率
        epsilon = extreme_count / len(data)
        
        # 簡單的統計量
        test_statistic = extreme_count
        p_value = 1 - stats.binom.cdf(extreme_count - 1, len(data), 0.05)
        
        return epsilon, test_statistic, p_value
    
    def _estimate_ks_test(self, data: np.ndarray, base_dist) -> Tuple[float, float, float]:
        """Kolmogorov-Smirnov檢驗法估計ε"""
        # 對基準分布進行K-S檢驗
        ks_statistic, ks_p_value = stats.kstest(data, base_dist.cdf)
        
        # 將K-S統計量轉換為ε估計
        # 這是啟發式轉換，實際應用中可能需要更精確的映射
        epsilon = min(0.5, ks_statistic * 2)  # 將D統計量映射到[0, 0.5]
        
        return epsilon, ks_statistic, ks_p_value
    
    def _estimate_ad_test(self, data: np.ndarray, base_dist) -> Tuple[float, float, float]:
        """Anderson-Darling檢驗法估計ε"""
        # 對基準分布進行A-D檢驗
        try:
            ad_statistic, critical_values, significance_levels = stats.anderson(data)
            
            # 根據A-D統計量估計ε
            # 使用與critical_values的比較
            if len(critical_values) > 0:
                # 標準化A-D統計量
                normalized_ad = ad_statistic / critical_values[2]  # 使用5%水平的臨界值
                epsilon = min(0.5, normalized_ad * 0.1)  # 啟發式映射
            else:
                epsilon = 0.05
            
            # 計算近似p值
            p_value = 1.0 / (1.0 + ad_statistic)  # 簡化的p值估計
            
            return epsilon, ad_statistic, p_value
            
        except Exception as e:
            print(f"   ⚠️ A-D檢驗失敗: {e}")
            return 0.05, 0.0, 1.0
    
    def _estimate_bayesian_selection(self, data: np.ndarray, base_dist) -> Tuple[float, float, float]:
        """貝氏模型選擇法估計ε"""
        # 這是簡化實現，實際應用中需要完整的貝氏框架
        
        # 嘗試不同的ε值，計算邊際似然
        epsilon_candidates = np.linspace(0.01, 0.3, 20)
        log_marginal_likelihoods = []
        
        for eps in epsilon_candidates:
            # 創建混合分布
            contamination_dist = self._create_contamination_distribution(data)
            
            # 計算混合分布的對數似然
            n_base = int(len(data) * (1 - eps))
            n_contamination = len(data) - n_base
            
            if n_base > 0 and n_contamination > 0:
                # 簡化的似然計算
                base_ll = np.sum(base_dist.logpdf(data[:n_base]))
                contamination_ll = np.sum(contamination_dist.logpdf(data[n_base:]))
                total_ll = base_ll + contamination_ll
            else:
                total_ll = np.sum(base_dist.logpdf(data))
            
            log_marginal_likelihoods.append(total_ll)
        
        # 找到最大似然對應的ε
        best_idx = np.argmax(log_marginal_likelihoods)
        best_epsilon = epsilon_candidates[best_idx]
        
        # 計算Bayes factor作為檢驗統計量
        max_ll = log_marginal_likelihoods[best_idx]
        null_ll = np.sum(base_dist.logpdf(data))  # ε=0的似然
        bayes_factor = 2 * (max_ll - null_ll)
        
        # 近似p值
        p_value = stats.chi2.sf(bayes_factor, df=1)
        
        return best_epsilon, bayes_factor, p_value
    
    def _create_contamination_distribution(self, data: np.ndarray):
        """創建污染分布"""
        if self.spec.contamination_class == ContaminationDistributionClass.TYPHOON_SPECIFIC:
            return self.contamination_generator.generate_typhoon_specific(
                location=np.mean(data), 
                scale=np.std(data) * 2,  # 更大的變異性
                shape=0.2
            )
        elif self.spec.contamination_class == ContaminationDistributionClass.HEAVY_TAILED:
            return self.contamination_generator.generate_heavy_tailed(
                df=3, 
                location=np.mean(data), 
                scale=np.std(data) * 1.5
            )
        else:
            # 預設使用正態分布但參數不同
            return stats.norm(loc=np.mean(data), scale=np.std(data) * 2)

# ========================================
# 先驗污染分析器 (從prior_contamination.py整合)
# ========================================

class PriorContaminationAnalyzer:
    """先驗污染分析器"""
    
    def __init__(self, spec: EpsilonContaminationSpec):
        self.spec = spec
        self.contamination_generator = ContaminationDistributionGenerator()
        
        print(f"🔬 先驗污染分析器初始化")
        print(f"   污染類別: {self.spec.contamination_class.value}")
        print(f"   基準先驗: {self.spec.nominal_prior_family}")
    
    def analyze_prior_robustness(self, 
                                epsilon_range: np.ndarray = None,
                                parameter_of_interest: str = "mean") -> Dict[str, Any]:
        """分析先驗的穩健性"""
        if epsilon_range is None:
            epsilon_range = np.linspace(0.0, 0.3, 31)
        
        print(f"🔍 分析先驗穩健性 (參數: {parameter_of_interest})...")
        
        # 創建基準和污染分布
        base_samples = self._generate_base_prior_samples(1000)
        contamination_samples = self._generate_contamination_prior_samples(1000)
        
        results = {
            "epsilon_range": epsilon_range,
            "parameter_values": [],
            "parameter_bounds": [],
            "robustness_metrics": {}
        }
        
        for eps in epsilon_range:
            # 創建混合先驗樣本
            n_contamination = int(1000 * eps)
            n_base = 1000 - n_contamination
            
            mixed_samples = np.concatenate([
                base_samples[:n_base],
                contamination_samples[:n_contamination]
            ])
            
            # 計算關注參數
            if parameter_of_interest == "mean":
                param_value = np.mean(mixed_samples)
            elif parameter_of_interest == "variance":
                param_value = np.var(mixed_samples)
            elif parameter_of_interest == "quantile_95":
                param_value = np.percentile(mixed_samples, 95)
            else:
                param_value = np.mean(mixed_samples)  # 預設
            
            results["parameter_values"].append(param_value)
            
            # 計算可信區間
            ci_lower = np.percentile(mixed_samples, 2.5)
            ci_upper = np.percentile(mixed_samples, 97.5)
            results["parameter_bounds"].append((ci_lower, ci_upper))
        
        # 計算穩健性指標
        param_values = np.array(results["parameter_values"])
        results["robustness_metrics"] = {
            "max_deviation": np.max(param_values) - np.min(param_values),
            "relative_deviation": (np.max(param_values) - np.min(param_values)) / np.abs(param_values[0]),
            "sensitivity_at_zero": np.gradient(param_values, epsilon_range)[0] if len(param_values) > 1 else 0,
            "max_sensitivity": np.max(np.abs(np.gradient(param_values, epsilon_range)))
        }
        
        print(f"✅ 穩健性分析完成")
        print(f"   最大偏差: {results['robustness_metrics']['max_deviation']:.4f}")
        print(f"   相對偏差: {results['robustness_metrics']['relative_deviation']:.2%}")
        
        return results
    
    def _generate_base_prior_samples(self, n_samples: int) -> np.ndarray:
        """生成基準先驗樣本"""
        if self.spec.nominal_prior_family == "normal":
            return np.random.normal(0, 1, n_samples)
        elif self.spec.nominal_prior_family == "lognormal":
            return np.random.lognormal(0, 1, n_samples)
        elif self.spec.nominal_prior_family == "gamma":
            return np.random.gamma(2, 1, n_samples)
        else:
            return np.random.normal(0, 1, n_samples)
    
    def _generate_contamination_prior_samples(self, n_samples: int) -> np.ndarray:
        """生成污染先驗樣本"""
        if self.spec.contamination_class == ContaminationDistributionClass.TYPHOON_SPECIFIC:
            # 使用廣義極值分布
            return stats.genextreme.rvs(c=0.2, loc=0, scale=2, size=n_samples)
        elif self.spec.contamination_class == ContaminationDistributionClass.HEAVY_TAILED:
            # 使用Student-t分布
            return stats.t.rvs(df=3, loc=0, scale=2, size=n_samples)
        else:
            # 預設使用不同參數的正態分布
            return np.random.normal(0, 3, n_samples)

# ========================================
# 分析工具函數
# ========================================

def quick_contamination_analysis(data: np.ndarray, 
                               wind_data: Optional[np.ndarray] = None) -> ContaminationEstimateResult:
    """颱風數據的快速污染程度分析"""
    
    try:
        from .contamination_core import create_typhoon_contamination_spec
    except ImportError:
        from contamination_core import create_typhoon_contamination_spec
        
    spec = create_typhoon_contamination_spec()
    estimator = EpsilonEstimator(spec)
    return estimator.estimate_contamination_level(data, wind_data)

def compare_estimation_methods(data: np.ndarray,
                             true_epsilon: float = None) -> pd.DataFrame:
    """比較不同的ε估計方法"""
    try:
        from .contamination_core import create_typhoon_contamination_spec
    except ImportError:
        from contamination_core import create_typhoon_contamination_spec
    
    spec = create_typhoon_contamination_spec()
    estimator = EpsilonEstimator(spec)
    
    # 使用所有可用方法估計
    result = estimator.estimate_contamination_level(data)
    
    # 構建比較表
    comparison_data = []
    
    for method_name, epsilon_est in result.epsilon_estimates.items():
        row = {
            "方法": method_name,
            "ε估計": epsilon_est,
            "檢驗統計量": result.test_statistics.get(method_name, np.nan),
            "p值": result.p_values.get(method_name, np.nan)
        }
        
        if true_epsilon is not None:
            row["絕對誤差"] = abs(epsilon_est - true_epsilon)
            row["相對誤差"] = abs(epsilon_est - true_epsilon) / true_epsilon
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

# ========================================
# 模組導出
# ========================================

__all__ = [
    'EpsilonEstimator',
    'PriorContaminationAnalyzer',
    'quick_contamination_analysis',
    'compare_estimation_methods'
]