"""
Posterior Analysis Reporter Module
後驗分析報告模組

提供後驗分布的深度分析，包括 Prior vs Posterior 比較、
分布特徵分析、參數相關性、預測分布評估等。

Key Features:
- Prior vs Posterior 分布比較
- 後驗分布特徵分析 (偏度、峰度、多模態)
- 參數間相關性分析
- 預測分布評估
- 後驗預測檢驗
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from scipy.stats import kstest, jarque_bera, shapiro
from sklearn.mixture import GaussianMixture
import sys
import os

# Import parent bayesian modules - use relative imports
try:
    from ..hierarchical_bayesian_model import HierarchicalModelResult, MixedPredictiveEstimation
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hierarchical_bayesian_model import HierarchicalModelResult, MixedPredictiveEstimation

class DistributionType(Enum):
    """分布類型"""
    NORMAL = "normal"
    SKEWED = "skewed"
    HEAVY_TAILED = "heavy_tailed"
    BIMODAL = "bimodal"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"

class LearningEffectiveness(Enum):
    """學習效果"""
    MINIMAL = "minimal"         # < 10% 改變
    MODERATE = "moderate"       # 10-30% 改變
    SUBSTANTIAL = "substantial" # 30-70% 改變
    DRAMATIC = "dramatic"       # > 70% 改變

@dataclass
class DistributionCharacteristics:
    """分布特徵"""
    mean: float
    std: float
    skewness: float
    kurtosis: float
    distribution_type: DistributionType
    normality_p_value: float
    is_normal: bool
    confidence_intervals: Dict[str, Tuple[float, float]]

@dataclass
class PriorPosteriorComparison:
    """先驗後驗比較"""
    parameter_name: str
    prior_characteristics: DistributionCharacteristics
    posterior_characteristics: DistributionCharacteristics
    kl_divergence: float
    wasserstein_distance: float
    learning_effectiveness: LearningEffectiveness
    shift_magnitude: float

@dataclass
class ParameterCorrelation:
    """參數相關性"""
    param1: str
    param2: str
    correlation: float
    mutual_information: float
    rank_correlation: float
    significance_p_value: float

@dataclass
class PosteriorPredictiveCheck:
    """後驗預測檢驗"""
    observed_statistic: float
    predicted_statistics: np.ndarray
    p_value: float
    extreme_probability: float
    check_type: str

@dataclass
class PosteriorAnalysisReport:
    """後驗分析報告"""
    prior_posterior_comparisons: List[PriorPosteriorComparison]
    parameter_correlations: List[ParameterCorrelation]
    posterior_predictive_checks: List[PosteriorPredictiveCheck]
    mixed_predictive_estimation: Dict[str, Any]
    summary_statistics: Dict[str, Any]
    recommendations: List[str]

class PosteriorAnalysisReporter:
    """
    後驗分析報告器
    
    提供後驗分布的全面分析和評估
    """
    
    def __init__(self, 
                 confidence_levels: List[float] = [0.5, 0.8, 0.9, 0.95],
                 n_posterior_predictive_samples: int = 1000,
                 correlation_threshold: float = 0.3):
        """
        初始化後驗分析報告器
        
        Parameters:
        -----------
        confidence_levels : List[float]
            信賴區間水準
        n_posterior_predictive_samples : int
            後驗預測樣本數
        correlation_threshold : float
            相關性顯著性閾值
        """
        self.confidence_levels = confidence_levels
        self.n_pp_samples = n_posterior_predictive_samples
        self.correlation_threshold = correlation_threshold
        
        # 設置圖表樣式
        plt.style.use('default')
        sns.set_style("whitegrid")
        self.colors = sns.color_palette("Set2", 8)
    
    def analyze_posterior_distributions(self, 
                                      posterior_samples: Dict[str, np.ndarray],
                                      prior_specifications: Optional[Dict[str, Dict]] = None,
                                      observed_data: Optional[np.ndarray] = None,
                                      hierarchical_result: Optional[HierarchicalModelResult] = None) -> PosteriorAnalysisReport:
        """
        全面的後驗分布分析
        
        Parameters:
        -----------
        posterior_samples : Dict[str, np.ndarray]
            後驗樣本
        prior_specifications : Dict[str, Dict], optional
            先驗規格 {param: {'distribution': 'normal', 'params': {...}}}
        observed_data : np.ndarray, optional
            觀測資料
        hierarchical_result : HierarchicalModelResult, optional
            階層模型結果
            
        Returns:
        --------
        PosteriorAnalysisReport
            完整的後驗分析報告
        """
        
        print("📊 開始後驗分布分析...")
        
        # 1. 先驗與後驗比較
        print("  🔄 先驗 vs 後驗比較...")
        prior_posterior_comparisons = self._compare_prior_posterior(
            posterior_samples, prior_specifications
        )
        
        # 2. 參數相關性分析
        print("  🔗 參數相關性分析...")
        parameter_correlations = self._analyze_parameter_correlations(posterior_samples)
        
        # 3. 後驗預測檢驗
        print("  ✅ 後驗預測檢驗...")
        posterior_predictive_checks = []
        if observed_data is not None:
            posterior_predictive_checks = self._perform_posterior_predictive_checks(
                posterior_samples, observed_data
            )
        
        # 4. MPE 分析
        print("  🔄 混合預測估計分析...")
        mpe_analysis = self._analyze_mixed_predictive_estimation(
            posterior_samples, hierarchical_result
        )
        
        # 5. 摘要統計
        summary_statistics = self._calculate_posterior_summary_statistics(
            posterior_samples, prior_posterior_comparisons, parameter_correlations
        )
        
        # 6. 生成建議
        recommendations = self._generate_posterior_recommendations(
            prior_posterior_comparisons, parameter_correlations, 
            posterior_predictive_checks, summary_statistics
        )
        
        report = PosteriorAnalysisReport(
            prior_posterior_comparisons=prior_posterior_comparisons,
            parameter_correlations=parameter_correlations,
            posterior_predictive_checks=posterior_predictive_checks,
            mixed_predictive_estimation=mpe_analysis,
            summary_statistics=summary_statistics,
            recommendations=recommendations
        )
        
        print("✅ 後驗分布分析完成")
        return report
    
    def _analyze_distribution_characteristics(self, 
                                            samples: np.ndarray) -> DistributionCharacteristics:
        """分析分布特徵"""
        
        if len(samples) == 0:
            return DistributionCharacteristics(
                mean=0, std=0, skewness=0, kurtosis=0,
                distribution_type=DistributionType.UNKNOWN,
                normality_p_value=0, is_normal=False,
                confidence_intervals={}
            )
        
        # 基本統計量
        mean = np.mean(samples)
        std = np.std(samples, ddof=1)
        skewness = stats.skew(samples)
        kurtosis = stats.kurtosis(samples)
        
        # 常態性檢定
        if len(samples) >= 8:
            try:
                _, normality_p = shapiro(samples) if len(samples) <= 5000 else jarque_bera(samples)[1]
            except:
                normality_p = 0.0
        else:
            normality_p = 0.0
        
        is_normal = normality_p > 0.05
        
        # 分布類型判斷
        distribution_type = self._classify_distribution_type(
            samples, skewness, kurtosis, is_normal
        )
        
        # 信賴區間
        confidence_intervals = {}
        for level in self.confidence_levels:
            alpha = 1 - level
            lower = np.percentile(samples, 100 * alpha / 2)
            upper = np.percentile(samples, 100 * (1 - alpha / 2))
            confidence_intervals[f'{level:.0%}'] = (lower, upper)
        
        return DistributionCharacteristics(
            mean=mean,
            std=std,
            skewness=skewness,
            kurtosis=kurtosis,
            distribution_type=distribution_type,
            normality_p_value=normality_p,
            is_normal=is_normal,
            confidence_intervals=confidence_intervals
        )
    
    def _classify_distribution_type(self, 
                                  samples: np.ndarray, 
                                  skewness: float, 
                                  kurtosis: float, 
                                  is_normal: bool) -> DistributionType:
        """分類分布類型"""
        
        # 檢查多模態
        if len(samples) > 50:
            try:
                # 使用高斯混合模型檢測模態數
                n_components_range = range(1, min(6, len(samples) // 10))
                best_n_components = 1
                best_bic = np.inf
                
                for n_comp in n_components_range:
                    gm = GaussianMixture(n_components=n_comp, random_state=42)
                    gm.fit(samples.reshape(-1, 1))
                    bic = gm.bic(samples.reshape(-1, 1))
                    if bic < best_bic:
                        best_bic = bic
                        best_n_components = n_comp
                
                if best_n_components > 2:
                    return DistributionType.MULTIMODAL
                elif best_n_components == 2:
                    return DistributionType.BIMODAL
            except:
                pass
        
        # 單模態分類 - 確保使用標量值
        skewness_val = float(skewness) if hasattr(skewness, 'item') else skewness
        kurtosis_val = float(kurtosis) if hasattr(kurtosis, 'item') else kurtosis
        
        if is_normal and abs(skewness_val) < 0.5 and abs(kurtosis_val) < 1:
            return DistributionType.NORMAL
        elif abs(skewness_val) > 1:
            return DistributionType.SKEWED
        elif abs(kurtosis_val) > 2:
            return DistributionType.HEAVY_TAILED
        else:
            return DistributionType.NORMAL
    
    def _compare_prior_posterior(self, 
                               posterior_samples: Dict[str, np.ndarray],
                               prior_specifications: Optional[Dict[str, Dict]]) -> List[PriorPosteriorComparison]:
        """比較先驗與後驗分布"""
        
        comparisons = []
        
        for param_name, post_samples in posterior_samples.items():
            if post_samples.ndim == 1 and len(post_samples) > 0:
                # 後驗特徵
                posterior_chars = self._analyze_distribution_characteristics(post_samples)
                
                # 先驗特徵 (如果提供)
                if prior_specifications and param_name in prior_specifications:
                    prior_chars = self._get_prior_characteristics(
                        prior_specifications[param_name], post_samples
                    )
                else:
                    # 使用弱信息先驗作為參考
                    prior_chars = self._create_default_prior_characteristics(post_samples)
                
                # 計算距離測度
                kl_div = self._calculate_kl_divergence(prior_chars, posterior_chars)
                wasserstein_dist = self._calculate_wasserstein_distance(prior_chars, posterior_chars)
                
                # 評估學習效果
                learning_effect = self._assess_learning_effectiveness(prior_chars, posterior_chars)
                
                # 計算變化幅度
                shift_magnitude = abs(posterior_chars.mean - prior_chars.mean) / prior_chars.std if prior_chars.std > 0 else 0
                
                comparison = PriorPosteriorComparison(
                    parameter_name=param_name,
                    prior_characteristics=prior_chars,
                    posterior_characteristics=posterior_chars,
                    kl_divergence=kl_div,
                    wasserstein_distance=wasserstein_dist,
                    learning_effectiveness=learning_effect,
                    shift_magnitude=shift_magnitude
                )
                
                comparisons.append(comparison)
        
        return comparisons
    
    def _get_prior_characteristics(self, 
                                 prior_spec: Dict, 
                                 reference_samples: np.ndarray) -> DistributionCharacteristics:
        """從先驗規格獲取先驗特徵"""
        
        # 生成先驗樣本
        n_samples = len(reference_samples)
        
        if prior_spec['distribution'] == 'normal':
            params = prior_spec['params']
            prior_samples = np.random.normal(
                params['loc'], params['scale'], n_samples
            )
        elif prior_spec['distribution'] == 'gamma':
            params = prior_spec['params']
            prior_samples = np.random.gamma(
                params['a'], params['scale'], n_samples
            )
        elif prior_spec['distribution'] == 'uniform':
            params = prior_spec['params']
            prior_samples = np.random.uniform(
                params['low'], params['high'], n_samples
            )
        else:
            # 回退到弱信息先驗
            return self._create_default_prior_characteristics(reference_samples)
        
        return self._analyze_distribution_characteristics(prior_samples)
    
    def _create_default_prior_characteristics(self, 
                                            reference_samples: np.ndarray) -> DistributionCharacteristics:
        """創建預設先驗特徵 (弱信息)"""
        
        # 使用後驗的範圍創建弱信息先驗
        post_mean = np.mean(reference_samples)
        post_std = np.std(reference_samples)
        
        # 弱信息先驗：均值相同，但標準差大5倍
        prior_std = post_std * 5
        
        return DistributionCharacteristics(
            mean=post_mean,
            std=prior_std,
            skewness=0,  # 假設對稱
            kurtosis=0,  # 假設常態
            distribution_type=DistributionType.NORMAL,
            normality_p_value=1.0,
            is_normal=True,
            confidence_intervals={}
        )
    
    def _calculate_kl_divergence(self, 
                               prior_chars: DistributionCharacteristics,
                               posterior_chars: DistributionCharacteristics) -> float:
        """計算 KL 散度 (簡化版)"""
        
        # 假設常態分布的 KL 散度
        mu1, sig1 = prior_chars.mean, prior_chars.std
        mu2, sig2 = posterior_chars.mean, posterior_chars.std
        
        if sig1 <= 0 or sig2 <= 0:
            return np.inf
        
        kl = np.log(sig2 / sig1) + (sig1**2 + (mu1 - mu2)**2) / (2 * sig2**2) - 0.5
        
        return max(0, kl)
    
    def _calculate_wasserstein_distance(self, 
                                      prior_chars: DistributionCharacteristics,
                                      posterior_chars: DistributionCharacteristics) -> float:
        """計算 Wasserstein 距離 (簡化版)"""
        
        # 對於常態分布的 Wasserstein 距離
        mu1, sig1 = prior_chars.mean, prior_chars.std
        mu2, sig2 = posterior_chars.mean, posterior_chars.std
        
        wasserstein = np.sqrt((mu1 - mu2)**2 + (sig1 - sig2)**2)
        
        return wasserstein
    
    def _assess_learning_effectiveness(self, 
                                     prior_chars: DistributionCharacteristics,
                                     posterior_chars: DistributionCharacteristics) -> LearningEffectiveness:
        """評估學習效果"""
        
        # 基於標準差的減少比例
        if prior_chars.std <= 0:
            return LearningEffectiveness.MINIMAL
        
        variance_reduction = 1 - (posterior_chars.std / prior_chars.std)
        
        if variance_reduction < 0.1:
            return LearningEffectiveness.MINIMAL
        elif variance_reduction < 0.3:
            return LearningEffectiveness.MODERATE
        elif variance_reduction < 0.7:
            return LearningEffectiveness.SUBSTANTIAL
        else:
            return LearningEffectiveness.DRAMATIC
    
    def _analyze_parameter_correlations(self, 
                                      posterior_samples: Dict[str, np.ndarray]) -> List[ParameterCorrelation]:
        """分析參數相關性"""
        
        correlations = []
        param_names = [name for name, samples in posterior_samples.items() 
                      if samples.ndim == 1 and len(samples) > 0]
        
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names[i+1:], i+1):
                samples1 = posterior_samples[param1]
                samples2 = posterior_samples[param2]
                
                # 確保樣本長度相同
                min_len = min(len(samples1), len(samples2))
                samples1 = samples1[:min_len]
                samples2 = samples2[:min_len]
                
                if min_len > 3:
                    # Pearson 相關係數
                    corr_coef, p_value = stats.pearsonr(samples1, samples2)
                    
                    # Spearman 等級相關
                    rank_corr, _ = stats.spearmanr(samples1, samples2)
                    
                    # 互信息 (簡化版)
                    mutual_info = self._calculate_mutual_information(samples1, samples2)
                    
                    correlation = ParameterCorrelation(
                        param1=param1,
                        param2=param2,
                        correlation=corr_coef,
                        mutual_information=mutual_info,
                        rank_correlation=rank_corr,
                        significance_p_value=p_value
                    )
                    
                    correlations.append(correlation)
        
        return correlations
    
    def _calculate_mutual_information(self, 
                                    samples1: np.ndarray, 
                                    samples2: np.ndarray,
                                    bins: int = 20) -> float:
        """計算互信息 (簡化版)"""
        
        try:
            # 離散化
            hist_2d, x_edges, y_edges = np.histogram2d(samples1, samples2, bins=bins)
            hist_1d_x = np.histogram(samples1, bins=x_edges)[0]
            hist_1d_y = np.histogram(samples2, bins=y_edges)[0]
            
            # 歸一化
            p_xy = hist_2d / np.sum(hist_2d)
            p_x = hist_1d_x / np.sum(hist_1d_x)
            p_y = hist_1d_y / np.sum(hist_1d_y)
            
            # 計算互信息
            mi = 0
            for i in range(len(p_x)):
                for j in range(len(p_y)):
                    # 確保所有值都是標量進行比較
                    pxy_val = float(p_xy[i, j]) if hasattr(p_xy[i, j], 'item') else p_xy[i, j]
                    px_val = float(p_x[i]) if hasattr(p_x[i], 'item') else p_x[i]
                    py_val = float(p_y[j]) if hasattr(p_y[j], 'item') else p_y[j]
                    
                    if pxy_val > 0 and px_val > 0 and py_val > 0:
                        mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            return max(0, mi)
            
        except:
            return 0.0
    
    def _perform_posterior_predictive_checks(self, 
                                           posterior_samples: Dict[str, np.ndarray],
                                           observed_data: np.ndarray) -> List[PosteriorPredictiveCheck]:
        """執行後驗預測檢驗"""
        
        checks = []
        
        # 檢驗項目
        check_functions = {
            'mean': lambda x: np.mean(x),
            'std': lambda x: np.std(x),
            'min': lambda x: np.min(x),
            'max': lambda x: np.max(x),
            'skewness': lambda x: stats.skew(x) if len(x) > 2 else 0,
            'kurtosis': lambda x: stats.kurtosis(x) if len(x) > 2 else 0
        }
        
        # 觀測統計量
        observed_stats = {name: func(observed_data) 
                         for name, func in check_functions.items()}
        
        # 生成後驗預測樣本
        predicted_stats = {name: [] for name in check_functions.keys()}
        
        for _ in range(self.n_pp_samples):
            # 簡化的預測資料生成
            if 'mu' in posterior_samples and 'sigma' in posterior_samples:
                # 從後驗樣本中隨機選擇參數
                idx = np.random.randint(len(posterior_samples['mu']))
                mu = posterior_samples['mu'][idx]
                sigma = posterior_samples['sigma'][idx] if 'sigma' in posterior_samples else np.std(observed_data)
                
                # 生成預測資料
                predicted_data = np.random.normal(mu, sigma, len(observed_data))
            else:
                # 回退：使用觀測資料的 bootstrap
                predicted_data = np.random.choice(observed_data, len(observed_data), replace=True)
            
            # 計算預測統計量
            for name, func in check_functions.items():
                try:
                    stat_value = func(predicted_data)
                    predicted_stats[name].append(stat_value)
                except:
                    predicted_stats[name].append(np.nan)
        
        # 創建檢驗結果
        for check_name in check_functions.keys():
            pred_array = np.array(predicted_stats[check_name])
            pred_array = pred_array[~np.isnan(pred_array)]
            
            if len(pred_array) > 0:
                obs_stat = observed_stats[check_name]
                
                # 計算 p 值
                p_value = np.mean(pred_array >= obs_stat) if not np.isnan(obs_stat) else 0.5
                p_value = min(p_value, 1 - p_value) * 2  # 雙邊檢驗
                
                # 極端機率
                extreme_prob = min(np.mean(pred_array <= obs_stat), np.mean(pred_array >= obs_stat))
                
                check = PosteriorPredictiveCheck(
                    observed_statistic=obs_stat,
                    predicted_statistics=pred_array,
                    p_value=p_value,
                    extreme_probability=extreme_prob,
                    check_type=check_name
                )
                
                checks.append(check)
        
        return checks
    
    def _analyze_mixed_predictive_estimation(self, 
                                           posterior_samples: Dict[str, np.ndarray],
                                           hierarchical_result: Optional[HierarchicalModelResult]) -> Dict[str, Any]:
        """分析混合預測估計"""
        
        mpe_analysis = {}
        
        if hierarchical_result and hierarchical_result.mpe_components:
            mpe_analysis['available'] = True
            mpe_analysis['components'] = hierarchical_result.mpe_components
            
            # 分析混合成分
            for param_name, mpe_result in hierarchical_result.mpe_components.items():
                if isinstance(mpe_result, dict) and 'mixture_weights' in mpe_result:
                    weights = mpe_result['mixture_weights']
                    n_components = len(weights)
                    
                    mpe_analysis[f'{param_name}_n_components'] = n_components
                    mpe_analysis[f'{param_name}_weights'] = weights
                    mpe_analysis[f'{param_name}_effective_components'] = np.sum(np.array(weights) > 0.05)
        else:
            # 手動執行 MPE
            mpe_analysis['available'] = False
            mpe_analysis['manual_mpe'] = {}
            
            mpe = MixedPredictiveEstimation(n_components=3)
            
            for param_name, samples in posterior_samples.items():
                if samples.ndim == 1 and len(samples) > 10:
                    try:
                        mpe_result = mpe.fit_mixture(samples, "normal")
                        mpe_analysis['manual_mpe'][param_name] = mpe_result
                    except:
                        pass
        
        return mpe_analysis
    
    def _calculate_posterior_summary_statistics(self, 
                                              posterior_samples: Dict[str, np.ndarray],
                                              prior_posterior_comparisons: List[PriorPosteriorComparison],
                                              parameter_correlations: List[ParameterCorrelation]) -> Dict[str, Any]:
        """計算後驗摘要統計"""
        
        summary = {
            'n_parameters': len([samples for samples in posterior_samples.values() 
                               if samples.ndim == 1]),
            'total_samples': sum(len(samples) for samples in posterior_samples.values() 
                               if samples.ndim == 1),
            'parameter_names': [name for name, samples in posterior_samples.items() 
                              if samples.ndim == 1]
        }
        
        # 學習效果統計
        if prior_posterior_comparisons:
            learning_effects = [comp.learning_effectiveness for comp in prior_posterior_comparisons]
            summary['learning_effects'] = {
                'minimal': sum(1 for le in learning_effects if le == LearningEffectiveness.MINIMAL),
                'moderate': sum(1 for le in learning_effects if le == LearningEffectiveness.MODERATE),
                'substantial': sum(1 for le in learning_effects if le == LearningEffectiveness.SUBSTANTIAL),
                'dramatic': sum(1 for le in learning_effects if le == LearningEffectiveness.DRAMATIC)
            }
            
            summary['mean_kl_divergence'] = np.mean([comp.kl_divergence for comp in prior_posterior_comparisons])
        
        # 相關性統計
        if parameter_correlations:
            high_correlations = [corr for corr in parameter_correlations 
                               if abs(corr.correlation) > self.correlation_threshold]
            
            summary['n_correlations'] = len(parameter_correlations)
            summary['n_high_correlations'] = len(high_correlations)
            summary['max_correlation'] = max(abs(corr.correlation) for corr in parameter_correlations)
        
        return summary
    
    def _generate_posterior_recommendations(self, 
                                          prior_posterior_comparisons: List[PriorPosteriorComparison],
                                          parameter_correlations: List[ParameterCorrelation],
                                          posterior_predictive_checks: List[PosteriorPredictiveCheck],
                                          summary_statistics: Dict[str, Any]) -> List[str]:
        """生成後驗分析建議"""
        
        recommendations = []
        
        # 學習效果建議
        if 'learning_effects' in summary_statistics:
            effects = summary_statistics['learning_effects']
            
            if effects['minimal'] > effects['substantial'] + effects['dramatic']:
                recommendations.append("⚠️ 多數參數學習效果有限，考慮:")
                recommendations.append("• 增加資料量")
                recommendations.append("• 使用更信息性的先驗")
                recommendations.append("• 檢查模型規格")
            
            elif effects['dramatic'] > 0:
                recommendations.append(f"📈 {effects['dramatic']} 個參數有顯著學習效果")
                recommendations.append("• 資料對這些參數提供了豐富信息")
        
        # 參數相關性建議
        if 'n_high_correlations' in summary_statistics and summary_statistics['n_high_correlations'] > 0:
            n_high = summary_statistics['n_high_correlations']
            recommendations.append(f"🔗 發現 {n_high} 組高相關參數")
            
            if summary_statistics['max_correlation'] > 0.8:
                recommendations.append("• 考慮參數重新參數化")
                recommendations.append("• 注意多重共線性問題")
        
        # 後驗預測檢驗建議
        failed_checks = [check for check in posterior_predictive_checks if check.p_value < 0.05]
        
        if len(failed_checks) > len(posterior_predictive_checks) * 0.3:
            recommendations.append("❌ 多項後驗預測檢驗未通過")
            recommendations.append("• 模型可能不適合資料")
            recommendations.append("• 考慮修改模型規格")
        elif len(failed_checks) > 0:
            check_names = [check.check_type for check in failed_checks]
            recommendations.append(f"⚠️ 檢驗項目異常: {', '.join(check_names[:3])}")
        else:
            recommendations.append("✅ 後驗預測檢驗通過")
        
        # 分布類型建議
        non_normal_params = [comp.parameter_name for comp in prior_posterior_comparisons 
                           if comp.posterior_characteristics.distribution_type != DistributionType.NORMAL]
        
        if len(non_normal_params) > 0:
            recommendations.append(f"📊 非常態參數: {', '.join(non_normal_params[:3])}")
            recommendations.append("• 考慮變換或使用非常態模型")
        
        return recommendations
    
    def plot_posterior_analysis(self, 
                              posterior_report: PosteriorAnalysisReport,
                              posterior_samples: Dict[str, np.ndarray],
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """繪製後驗分析圖表"""
        
        n_params = len([name for name, samples in posterior_samples.items() 
                       if samples.ndim == 1])
        
        if n_params == 0:
            print("沒有參數可繪圖")
            return None
        
        # 動態計算子圖數量
        n_plots = min(6, n_params + 2)  # 參數圖 + 相關性 + 預測檢驗
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('後驗分布分析', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        
        # 1. 參數後驗分布 (前4個參數)
        param_names = [name for name, samples in posterior_samples.items() 
                      if samples.ndim == 1][:4]
        
        for i, param_name in enumerate(param_names):
            if plot_idx >= n_plots:
                break
                
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]
            
            samples = posterior_samples[param_name]
            
            # 繪製後驗分布
            ax.hist(samples, bins=50, alpha=0.7, color=self.colors[i], 
                   density=True, label='Posterior')
            
            # 添加統計線
            mean_val = np.mean(samples)
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            
            # 信賴區間
            ci_95 = np.percentile(samples, [2.5, 97.5])
            ax.axvspan(ci_95[0], ci_95[1], alpha=0.2, color=self.colors[i], label='95% CI')
            
            ax.set_title(f'{param_name}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            
            plot_idx += 1
        
        # 2. 參數相關性熱圖
        if plot_idx < n_plots and posterior_report.parameter_correlations:
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]
            
            # 創建相關性矩陣
            unique_params = list(set([corr.param1 for corr in posterior_report.parameter_correlations] +
                                   [corr.param2 for corr in posterior_report.parameter_correlations]))
            
            n_unique = len(unique_params)
            corr_matrix = np.eye(n_unique)
            
            for corr in posterior_report.parameter_correlations:
                i = unique_params.index(corr.param1)
                j = unique_params.index(corr.param2)
                corr_matrix[i, j] = corr.correlation
                corr_matrix[j, i] = corr.correlation
            
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(n_unique))
            ax.set_yticks(range(n_unique))
            ax.set_xticklabels(unique_params, rotation=45)
            ax.set_yticklabels(unique_params)
            ax.set_title('參數相關性')
            
            # 添加數值標註
            for i in range(n_unique):
                for j in range(n_unique):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im, ax=ax, shrink=0.8)
            plot_idx += 1
        
        # 3. 後驗預測檢驗
        if plot_idx < n_plots and posterior_report.posterior_predictive_checks:
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]
            
            check_names = [check.check_type for check in posterior_report.posterior_predictive_checks]
            p_values = [check.p_value for check in posterior_report.posterior_predictive_checks]
            
            bars = ax.bar(range(len(check_names)), p_values, 
                         color=['red' if p < 0.05 else 'green' for p in p_values],
                         alpha=0.7)
            
            ax.axhline(0.05, color='red', linestyle='--', label='α = 0.05')
            ax.set_xlabel('檢驗項目')
            ax.set_ylabel('p-value')
            ax.set_title('後驗預測檢驗')
            ax.set_xticks(range(len(check_names)))
            ax.set_xticklabels(check_names, rotation=45)
            ax.legend()
            
            plot_idx += 1
        
        # 隱藏多餘的子圖
        for i in range(plot_idx, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_posterior_report(self, 
                                posterior_report: PosteriorAnalysisReport,
                                include_details: bool = True) -> str:
        """生成後驗分析報告"""
        
        report = []
        report.append("=" * 80)
        report.append("                    後驗分布分析報告")
        report.append("=" * 80)
        report.append("")
        
        # 整體摘要
        report.append("📊 分析摘要")
        report.append("-" * 40)
        
        summary = posterior_report.summary_statistics
        report.append(f"📈 分析參數數: {summary.get('n_parameters', 0)}")
        report.append(f"📊 總樣本數: {summary.get('total_samples', 0)}")
        
        if 'learning_effects' in summary:
            effects = summary['learning_effects']
            report.append(f"📚 學習效果分布:")
            report.append(f"    顯著: {effects.get('dramatic', 0)} 個")
            report.append(f"    實質: {effects.get('substantial', 0)} 個")
            report.append(f"    中等: {effects.get('moderate', 0)} 個")
            report.append(f"    微弱: {effects.get('minimal', 0)} 個")
        
        report.append("")
        
        # 先驗後驗比較
        if include_details and posterior_report.prior_posterior_comparisons:
            report.append("🔄 先驗 vs 後驗比較")
            report.append("-" * 40)
            
            for comp in posterior_report.prior_posterior_comparisons[:5]:  # 前5個
                learning_icons = {
                    LearningEffectiveness.DRAMATIC: "🚀",
                    LearningEffectiveness.SUBSTANTIAL: "📈",
                    LearningEffectiveness.MODERATE: "📊",
                    LearningEffectiveness.MINIMAL: "📉"
                }
                
                icon = learning_icons.get(comp.learning_effectiveness, "📊")
                report.append(f"{icon} {comp.parameter_name}:")
                report.append(f"    後驗均值: {comp.posterior_characteristics.mean:.3f}")
                report.append(f"    後驗標準差: {comp.posterior_characteristics.std:.3f}")
                report.append(f"    學習效果: {comp.learning_effectiveness.value}")
                report.append(f"    KL 散度: {comp.kl_divergence:.3f}")
                report.append("")
        
        # 參數相關性
        if posterior_report.parameter_correlations:
            report.append("🔗 參數相關性分析")
            report.append("-" * 40)
            
            high_corr = [corr for corr in posterior_report.parameter_correlations 
                        if abs(corr.correlation) > self.correlation_threshold]
            
            report.append(f"📊 高相關參數對: {len(high_corr)}")
            
            if include_details:
                for corr in high_corr[:5]:  # 前5個高相關
                    report.append(f"• {corr.param1} - {corr.param2}: {corr.correlation:.3f}")
            report.append("")
        
        # 後驗預測檢驗
        if posterior_report.posterior_predictive_checks:
            report.append("✅ 後驗預測檢驗")
            report.append("-" * 40)
            
            failed_checks = [check for check in posterior_report.posterior_predictive_checks 
                           if check.p_value < 0.05]
            
            report.append(f"📊 檢驗項目數: {len(posterior_report.posterior_predictive_checks)}")
            report.append(f"❌ 未通過檢驗: {len(failed_checks)}")
            
            if failed_checks and include_details:
                report.append("未通過的檢驗:")
                for check in failed_checks:
                    report.append(f"• {check.check_type}: p = {check.p_value:.3f}")
            report.append("")
        
        # 建議
        report.append("💡 分析建議")
        report.append("-" * 40)
        for recommendation in posterior_report.recommendations:
            report.append(recommendation)
        
        return "\n".join(report)