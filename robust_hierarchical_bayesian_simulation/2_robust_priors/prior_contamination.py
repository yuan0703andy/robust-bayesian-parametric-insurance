#!/usr/bin/env python3
"""
Prior Contamination Module
先驗污染模組

從 epsilon_contamination.py 拆分出的先驗污染功能
專門處理先驗分布的ε-contamination建模

數學基礎:
π_ε(θ) = (1-ε)π₀(θ) + εq(θ)

核心功能:
- 先驗污染建模
- ε值估計
- 先驗穩健性分析

Author: Research Team  
Date: 2025-01-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings

# 從其他模組導入
try:
    from .contamination_theory import (
        EpsilonContaminationSpec, ContaminationEstimateResult, 
        ContaminationDistributionClass, EstimationMethod,
        ContaminationDistributionGenerator
    )
except ImportError:
    # 如果相對導入失敗，嘗試絕對導入
    try:
        from contamination_theory import (
            EpsilonContaminationSpec, ContaminationEstimateResult, 
            ContaminationDistributionClass, EstimationMethod,
            ContaminationDistributionGenerator
        )
    except ImportError:
        # 如果都失敗，嘗試從當前目錄導入
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from contamination_theory import (
            EpsilonContaminationSpec, ContaminationEstimateResult, 
            ContaminationDistributionClass, EstimationMethod,
            ContaminationDistributionGenerator
        )

# ========================================
# 先驗污染分析器
# ========================================

class PriorContaminationAnalyzer:
    """
    先驗污染分析器
    
    專門處理先驗分布的ε-contamination建模和分析
    """
    
    def __init__(self, spec: EpsilonContaminationSpec):
        """
        初始化先驗污染分析器
        
        Parameters:
        -----------
        spec : EpsilonContaminationSpec
            ε-contamination規格配置
        """
        self.spec = spec
        self.contamination_generator = ContaminationDistributionGenerator()
        
        # 結果緩存
        self.estimation_cache: Dict[str, ContaminationEstimateResult] = {}
        
        print(f"🔬 先驗污染分析器初始化完成")
        print(f"   污染類別: {self.spec.contamination_class.value}")
        print(f"   基準先驗: {self.spec.nominal_prior_family}")
        print(f"   污染先驗: {self.spec.contamination_prior_family}")
    
    def estimate_epsilon_from_data(self, 
                                 data: np.ndarray,
                                 methods: List[EstimationMethod] = None) -> ContaminationEstimateResult:
        """
        從數據估計ε值
        
        Parameters:
        -----------
        data : np.ndarray
            觀測數據
        methods : List[EstimationMethod], optional
            使用的估計方法
            
        Returns:
        --------
        ContaminationEstimateResult
            ε估計結果
        """
        if methods is None:
            methods = [
                EstimationMethod.EMPIRICAL_FREQUENCY,
                EstimationMethod.KOLMOGOROV_SMIRNOV,
                EstimationMethod.ANDERSON_DARLING
            ]
        
        print(f"📊 從數據估計ε值 (n={len(data)})...")
        
        epsilon_estimates = {}
        test_statistics = {}
        p_values = {}
        
        # 生成基準分布
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
        
        # 計算共識估計和不確定性
        consensus_epsilon = np.mean(list(epsilon_estimates.values()))
        uncertainty = np.std(list(epsilon_estimates.values()))
        
        # 設定閾值
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
            method_weights={}  # 可以後續添加權重
        )
        
        # 緩存結果
        cache_key = f"data_{len(data)}_{hash(data.tobytes())}"
        self.estimation_cache[cache_key] = result
        
        print(f"✅ ε估計完成: {consensus_epsilon:.4f} ± {uncertainty:.4f}")
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
    
    def analyze_prior_robustness(self, 
                                epsilon_range: np.ndarray = None,
                                parameter_of_interest: str = "mean") -> Dict[str, Any]:
        """
        分析先驗的穩健性
        
        Parameters:
        -----------
        epsilon_range : np.ndarray, optional
            ε值範圍
        parameter_of_interest : str
            關注的參數
            
        Returns:
        --------
        Dict[str, Any]
            穩健性分析結果
        """
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
    
    def compare_estimation_methods(self, 
                                 data: np.ndarray,
                                 true_epsilon: float = None) -> pd.DataFrame:
        """
        比較不同的ε估計方法
        
        Parameters:
        -----------
        data : np.ndarray
            觀測數據
        true_epsilon : float, optional
            真實的ε值（如果已知）
            
        Returns:
        --------
        pd.DataFrame
            方法比較結果
        """
        print(f"📈 比較ε估計方法...")
        
        # 使用所有可用方法估計
        all_methods = [
            EstimationMethod.EMPIRICAL_FREQUENCY,
            EstimationMethod.KOLMOGOROV_SMIRNOV,
            EstimationMethod.ANDERSON_DARLING,
            EstimationMethod.BAYESIAN_MODEL_SELECTION
        ]
        
        result = self.estimate_epsilon_from_data(data, all_methods)
        
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
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print(f"✅ 方法比較完成")
        if true_epsilon is not None:
            print(f"   真實ε值: {true_epsilon:.4f}")
            best_method_idx = comparison_df["絕對誤差"].idxmin()
            best_method = comparison_df.loc[best_method_idx, "方法"]
            print(f"   最佳方法: {best_method}")
        
        return comparison_df

def test_prior_contamination():
    """測試先驗污染功能"""
    print("🧪 測試先驗污染模組...")
    
    # 創建測試數據
    np.random.seed(42)
    # 模擬混合數據：90%正態 + 10%極值
    normal_data = np.random.normal(0, 1, 900)
    extreme_data = np.random.exponential(3, 100)
    test_data = np.concatenate([normal_data, extreme_data])
    np.random.shuffle(test_data)
    
    # 創建分析器
    spec = EpsilonContaminationSpec(
        contamination_class=ContaminationDistributionClass.TYPHOON_SPECIFIC
    )
    analyzer = PriorContaminationAnalyzer(spec)
    
    # 測試ε估計
    print("✅ 測試ε估計:")
    result = analyzer.estimate_epsilon_from_data(test_data)
    print(f"   共識ε值: {result.epsilon_consensus:.4f}")
    
    # 測試穩健性分析
    print("✅ 測試穩健性分析:")
    robustness = analyzer.analyze_prior_robustness()
    print(f"   最大偏差: {robustness['robustness_metrics']['max_deviation']:.4f}")
    
    # 測試方法比較
    print("✅ 測試方法比較:")
    comparison = analyzer.compare_estimation_methods(test_data, true_epsilon=0.1)
    print(f"   比較結果: {len(comparison)} 個方法")
    
    print("✅ 先驗污染測試完成")

if __name__ == "__main__":
    test_prior_contamination()