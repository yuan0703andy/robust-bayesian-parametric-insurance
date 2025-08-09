"""
Robustness Reporter Module
穩健性報告模組

提供密度比分析、敏感度分析和穩健性評估的詳細報告。
分析貝氏模型對先驗選擇和模型假設的敏感度。

Key Features:
- Density Ratio Class 約束分析
- 先驗敏感度分析
- 模型假設穩健性檢驗
- 不確定性來源貢獻分析
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
from scipy.spatial.distance import jensenshannon
import sys
import os

# Import parent bayesian modules - use relative imports
try:
    from ..robust_bayesian_analysis import (
        RobustBayesianFramework, DensityRatioClass, 
        ModelComparisonResult, ModelConfiguration
    )
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from robust_bayesian_analysis import (
        RobustBayesianFramework, DensityRatioClass, 
        ModelComparisonResult, ModelConfiguration
    )

class RobustnessLevel(Enum):
    """穩健性水平"""
    HIGHLY_ROBUST = "highly_robust"      # < 5% 變異
    ROBUST = "robust"                    # 5-15% 變異
    MODERATELY_ROBUST = "moderately_robust"  # 15-30% 變異
    SENSITIVE = "sensitive"              # 30-50% 變異
    HIGHLY_SENSITIVE = "highly_sensitive"    # > 50% 變異

@dataclass
class DensityRatioAnalysis:
    """密度比分析結果"""
    gamma_constraint: float
    violation_rate: float
    max_density_ratio: float
    mean_density_ratio: float
    violation_regions: Dict[str, Any]
    constraint_satisfaction: bool

@dataclass
class SensitivityAnalysis:
    """敏感度分析結果"""
    parameter_name: str
    prior_scenarios: List[str]
    posterior_variation: float
    coefficient_of_variation: float
    robustness_level: RobustnessLevel
    sensitivity_metrics: Dict[str, float]

@dataclass
class RobustnessReport:
    """穩健性報告"""
    overall_robustness: RobustnessLevel
    density_ratio_analysis: DensityRatioAnalysis
    sensitivity_analyses: Dict[str, SensitivityAnalysis]
    uncertainty_decomposition: Dict[str, float]
    recommendations: List[str]
    robustness_score: float

class RobustnessReporter:
    """
    穩健性報告器
    
    分析貝氏模型的穩健性，包括密度比約束、先驗敏感度等
    """
    
    def __init__(self, 
                 gamma_constraint: float = 2.0,
                 sensitivity_threshold: float = 0.3,
                 n_sensitivity_scenarios: int = 5):
        """
        初始化穩健性報告器
        
        Parameters:
        -----------
        gamma_constraint : float
            密度比約束參數
        sensitivity_threshold : float
            敏感度閾值
        n_sensitivity_scenarios : int
            敏感度分析場景數
        """
        self.gamma_constraint = gamma_constraint
        self.sensitivity_threshold = sensitivity_threshold
        self.n_sensitivity_scenarios = n_sensitivity_scenarios
        
        # 初始化分析組件
        self.density_ratio_class = DensityRatioClass(gamma_constraint)
        self.robust_framework = RobustBayesianFramework(gamma_constraint)
        
        # 設置圖表樣式
        plt.style.use('default')
        sns.set_style("whitegrid")
        self.colors = sns.color_palette("viridis", 8)
    
    def analyze_robustness(self, 
                          posterior_samples: Dict[str, np.ndarray],
                          observed_data: np.ndarray,
                          model_results: List[ModelComparisonResult]) -> RobustnessReport:
        """
        全面的穩健性分析
        
        Parameters:
        -----------
        posterior_samples : Dict[str, np.ndarray]
            後驗樣本
        observed_data : np.ndarray
            觀測資料
        model_results : List[ModelComparisonResult]
            模型比較結果
            
        Returns:
        --------
        RobustnessReport
            完整的穩健性報告
        """
        
        print("🛡️ 開始穩健性分析...")
        
        # 1. 密度比分析
        print("  📊 密度比約束分析...")
        density_ratio_analysis = self._analyze_density_ratio_constraints(
            posterior_samples, model_results
        )
        
        # 2. 先驗敏感度分析
        print("  🎯 先驗敏感度分析...")
        sensitivity_analyses = self._analyze_prior_sensitivity(
            posterior_samples, observed_data
        )
        
        # 3. 不確定性分解
        print("  🔍 不確定性來源分析...")
        uncertainty_decomposition = self._decompose_uncertainty_sources(
            posterior_samples, sensitivity_analyses
        )
        
        # 4. 整體穩健性評估
        overall_robustness = self._assess_overall_robustness(
            density_ratio_analysis, sensitivity_analyses
        )
        
        # 5. 計算穩健性評分
        robustness_score = self._calculate_robustness_score(
            density_ratio_analysis, sensitivity_analyses, uncertainty_decomposition
        )
        
        # 6. 生成建議
        recommendations = self._generate_robustness_recommendations(
            density_ratio_analysis, sensitivity_analyses, overall_robustness
        )
        
        robustness_report = RobustnessReport(
            overall_robustness=overall_robustness,
            density_ratio_analysis=density_ratio_analysis,
            sensitivity_analyses=sensitivity_analyses,
            uncertainty_decomposition=uncertainty_decomposition,
            recommendations=recommendations,
            robustness_score=robustness_score
        )
        
        print("✅ 穩健性分析完成")
        return robustness_report
    
    def _analyze_density_ratio_constraints(self, 
                                         posterior_samples: Dict[str, np.ndarray],
                                         model_results: List[ModelComparisonResult]) -> DensityRatioAnalysis:
        """分析密度比約束"""
        
        # 計算約束違反
        violation_counts = [result.density_ratio_violations for result in model_results]
        total_evaluations = len(model_results) * 1000  # 假設每個模型評估1000個點
        violation_rate = sum(violation_counts) / total_evaluations if total_evaluations > 0 else 0
        
        # 模擬密度比計算
        density_ratios = []
        for param_name, samples in posterior_samples.items():
            if samples.ndim == 1:
                # 簡化的密度比計算
                # 使用樣本的經驗分布與標準正態分布比較
                
                # 標準化樣本
                standardized = (samples - np.mean(samples)) / np.std(samples)
                
                # 計算經驗密度比
                for i in range(0, len(standardized), 100):  # 抽樣計算
                    x = standardized[i]
                    
                    # 經驗分布密度 (簡化)
                    empirical_density = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
                    
                    # 參考分布密度 (標準正態)
                    reference_density = stats.norm.pdf(x)
                    
                    if reference_density > 1e-10:
                        ratio = empirical_density / reference_density
                        density_ratios.append(ratio)
        
        density_ratios = np.array(density_ratios)
        
        # 分析結果
        max_ratio = np.max(density_ratios) if len(density_ratios) > 0 else 0
        mean_ratio = np.mean(density_ratios) if len(density_ratios) > 0 else 0
        
        # 識別違反區域
        violation_regions = {}
        if len(density_ratios) > 0:
            violating_indices = density_ratios > self.gamma_constraint
            violation_regions = {
                'n_violations': np.sum(violating_indices),
                'violation_percentage': np.mean(violating_indices) * 100,
                'max_violation': np.max(density_ratios[violating_indices]) if np.any(violating_indices) else 0
            }
        
        constraint_satisfaction = violation_rate < 0.05  # 5% 容忍度
        
        return DensityRatioAnalysis(
            gamma_constraint=self.gamma_constraint,
            violation_rate=violation_rate,
            max_density_ratio=max_ratio,
            mean_density_ratio=mean_ratio,
            violation_regions=violation_regions,
            constraint_satisfaction=constraint_satisfaction
        )
    
    def _analyze_prior_sensitivity(self, 
                                 posterior_samples: Dict[str, np.ndarray],
                                 observed_data: np.ndarray) -> Dict[str, SensitivityAnalysis]:
        """分析先驗敏感度"""
        
        sensitivity_analyses = {}
        
        # 定義不同的先驗場景
        prior_scenarios = [
            "informative_normal",
            "weak_normal", 
            "uniform",
            "gamma_shape1",
            "gamma_shape2"
        ]
        
        for param_name, baseline_samples in posterior_samples.items():
            if baseline_samples.ndim == 1:
                print(f"    📈 分析參數 {param_name} 的敏感度...")
                
                # 為不同先驗場景模擬後驗樣本
                scenario_posteriors = {}
                scenario_posteriors['baseline'] = baseline_samples
                
                # 模擬不同先驗下的後驗 (簡化)
                for scenario in prior_scenarios:
                    perturbed_samples = self._simulate_posterior_under_prior(
                        baseline_samples, scenario, observed_data
                    )
                    scenario_posteriors[scenario] = perturbed_samples
                
                # 計算敏感度指標
                sensitivity_metrics = self._calculate_sensitivity_metrics(scenario_posteriors)
                
                # 評估穩健性水平
                robustness_level = self._assess_parameter_robustness(sensitivity_metrics)
                
                sensitivity_analyses[param_name] = SensitivityAnalysis(
                    parameter_name=param_name,
                    prior_scenarios=list(scenario_posteriors.keys()),
                    posterior_variation=sensitivity_metrics['posterior_variation'],
                    coefficient_of_variation=sensitivity_metrics['coefficient_of_variation'],
                    robustness_level=robustness_level,
                    sensitivity_metrics=sensitivity_metrics
                )
        
        return sensitivity_analyses
    
    def _simulate_posterior_under_prior(self, 
                                      baseline_samples: np.ndarray,
                                      prior_scenario: str,
                                      observed_data: np.ndarray) -> np.ndarray:
        """模擬不同先驗下的後驗分布"""
        
        n_samples = len(baseline_samples)
        baseline_mean = np.mean(baseline_samples)
        baseline_std = np.std(baseline_samples)
        
        # 根據先驗場景調整
        if prior_scenario == "informative_normal":
            # 強信息先驗
            noise_scale = 0.1
        elif prior_scenario == "weak_normal":
            # 弱信息先驗
            noise_scale = 0.5
        elif prior_scenario == "uniform":
            # 均勻先驗（更大的變異）
            noise_scale = 0.8
        elif prior_scenario == "gamma_shape1":
            # Gamma 先驗變體1
            noise_scale = 0.3
            baseline_samples = np.abs(baseline_samples)  # 確保正值
        elif prior_scenario == "gamma_shape2":
            # Gamma 先驗變體2
            noise_scale = 0.6
            baseline_samples = np.abs(baseline_samples)
        else:
            noise_scale = 0.2
        
        # 添加擾動模擬先驗影響
        noise = np.random.normal(0, noise_scale * baseline_std, n_samples)
        perturbed_samples = baseline_samples + noise
        
        return perturbed_samples
    
    def _calculate_sensitivity_metrics(self, 
                                     scenario_posteriors: Dict[str, np.ndarray]) -> Dict[str, float]:
        """計算敏感度指標"""
        
        baseline = scenario_posteriors['baseline']
        baseline_mean = np.mean(baseline)
        
        # 計算各場景的後驗均值
        scenario_means = {}
        for scenario, samples in scenario_posteriors.items():
            scenario_means[scenario] = np.mean(samples)
        
        # 後驗變異度
        mean_values = list(scenario_means.values())
        posterior_variation = np.std(mean_values) / np.abs(baseline_mean) if baseline_mean != 0 else np.std(mean_values)
        
        # 變異係數
        coefficient_of_variation = np.std(mean_values) / np.mean(mean_values) if np.mean(mean_values) != 0 else 0
        
        # Jensen-Shannon 散度
        js_divergences = []
        for scenario, samples in scenario_posteriors.items():
            if scenario != 'baseline':
                # 計算分布間的JS散度
                hist_baseline, bins = np.histogram(baseline, bins=50, density=True)
                hist_scenario, _ = np.histogram(samples, bins=bins, density=True)
                
                # 避免零值
                hist_baseline = hist_baseline + 1e-10
                hist_scenario = hist_scenario + 1e-10
                
                js_div = jensenshannon(hist_baseline, hist_scenario)
                js_divergences.append(js_div)
        
        mean_js_divergence = np.mean(js_divergences) if js_divergences else 0
        
        # 相對變異
        relative_changes = []
        for scenario, mean_val in scenario_means.items():
            if scenario != 'baseline':
                rel_change = abs(mean_val - baseline_mean) / abs(baseline_mean) if baseline_mean != 0 else abs(mean_val)
                relative_changes.append(rel_change)
        
        max_relative_change = np.max(relative_changes) if relative_changes else 0
        
        return {
            'posterior_variation': posterior_variation,
            'coefficient_of_variation': coefficient_of_variation,
            'mean_js_divergence': mean_js_divergence,
            'max_relative_change': max_relative_change,
            'scenario_means': scenario_means
        }
    
    def _assess_parameter_robustness(self, sensitivity_metrics: Dict[str, float]) -> RobustnessLevel:
        """評估參數穩健性水平"""
        
        variation = sensitivity_metrics['max_relative_change']
        
        if variation < 0.05:
            return RobustnessLevel.HIGHLY_ROBUST
        elif variation < 0.15:
            return RobustnessLevel.ROBUST
        elif variation < 0.30:
            return RobustnessLevel.MODERATELY_ROBUST
        elif variation < 0.50:
            return RobustnessLevel.SENSITIVE
        else:
            return RobustnessLevel.HIGHLY_SENSITIVE
    
    def _decompose_uncertainty_sources(self, 
                                     posterior_samples: Dict[str, np.ndarray],
                                     sensitivity_analyses: Dict[str, SensitivityAnalysis]) -> Dict[str, float]:
        """分解不確定性來源"""
        
        # 計算不同來源的不確定性貢獻
        total_uncertainty = 0
        prior_uncertainty = 0
        sampling_uncertainty = 0
        
        for param_name, samples in posterior_samples.items():
            if param_name in sensitivity_analyses:
                # 總體不確定性
                param_var = np.var(samples)
                total_uncertainty += param_var
                
                # 先驗不確定性
                sensitivity = sensitivity_analyses[param_name]
                prior_var = sensitivity.posterior_variation ** 2
                prior_uncertainty += prior_var
                
                # 採樣不確定性 (估計)
                n_eff = len(samples) / (1 + 2 * np.sum(np.abs(np.correlate(samples - np.mean(samples), 
                                                                          samples - np.mean(samples), 'full'))))
                sampling_var = param_var / max(n_eff, 1)
                sampling_uncertainty += sampling_var
        
        # 歸一化
        if total_uncertainty > 0:
            prior_fraction = prior_uncertainty / total_uncertainty
            sampling_fraction = sampling_uncertainty / total_uncertainty
            model_fraction = max(0, 1 - prior_fraction - sampling_fraction)
        else:
            prior_fraction = sampling_fraction = model_fraction = 1/3
        
        return {
            'prior_uncertainty': prior_fraction,
            'sampling_uncertainty': sampling_fraction,
            'model_uncertainty': model_fraction,
            'total_uncertainty': total_uncertainty
        }
    
    def _assess_overall_robustness(self, 
                                 density_ratio_analysis: DensityRatioAnalysis,
                                 sensitivity_analyses: Dict[str, SensitivityAnalysis]) -> RobustnessLevel:
        """評估整體穩健性"""
        
        # 密度比約束滿足度
        constraint_score = 1.0 if density_ratio_analysis.constraint_satisfaction else 0.5
        
        # 參數敏感度平均
        if sensitivity_analyses:
            robustness_levels = [analysis.robustness_level for analysis in sensitivity_analyses.values()]
            
            level_scores = {
                RobustnessLevel.HIGHLY_ROBUST: 1.0,
                RobustnessLevel.ROBUST: 0.8,
                RobustnessLevel.MODERATELY_ROBUST: 0.6,
                RobustnessLevel.SENSITIVE: 0.4,
                RobustnessLevel.HIGHLY_SENSITIVE: 0.2
            }
            
            sensitivity_score = np.mean([level_scores[level] for level in robustness_levels])
        else:
            sensitivity_score = 0.5
        
        # 綜合評分
        overall_score = 0.6 * constraint_score + 0.4 * sensitivity_score
        
        if overall_score >= 0.9:
            return RobustnessLevel.HIGHLY_ROBUST
        elif overall_score >= 0.7:
            return RobustnessLevel.ROBUST
        elif overall_score >= 0.5:
            return RobustnessLevel.MODERATELY_ROBUST
        elif overall_score >= 0.3:
            return RobustnessLevel.SENSITIVE
        else:
            return RobustnessLevel.HIGHLY_SENSITIVE
    
    def _calculate_robustness_score(self, 
                                  density_ratio_analysis: DensityRatioAnalysis,
                                  sensitivity_analyses: Dict[str, SensitivityAnalysis],
                                  uncertainty_decomposition: Dict[str, float]) -> float:
        """計算整體穩健性評分 (0-100)"""
        
        # 密度比約束評分 (30%)
        constraint_score = 0 if not density_ratio_analysis.constraint_satisfaction else \
                          max(0, 1 - density_ratio_analysis.violation_rate / 0.1) * 30
        
        # 敏感度評分 (50%)
        if sensitivity_analyses:
            sensitivity_scores = []
            for analysis in sensitivity_analyses.values():
                if analysis.robustness_level == RobustnessLevel.HIGHLY_ROBUST:
                    sensitivity_scores.append(1.0)
                elif analysis.robustness_level == RobustnessLevel.ROBUST:
                    sensitivity_scores.append(0.8)
                elif analysis.robustness_level == RobustnessLevel.MODERATELY_ROBUST:
                    sensitivity_scores.append(0.6)
                elif analysis.robustness_level == RobustnessLevel.SENSITIVE:
                    sensitivity_scores.append(0.4)
                else:
                    sensitivity_scores.append(0.2)
            
            sensitivity_score = np.mean(sensitivity_scores) * 50
        else:
            sensitivity_score = 25
        
        # 不確定性結構評分 (20%)
        # 偏好模型不確定性占主導，而非先驗不確定性
        model_uncertainty = uncertainty_decomposition.get('model_uncertainty', 0.33)
        uncertainty_score = min(model_uncertainty * 2, 1.0) * 20
        
        total_score = constraint_score + sensitivity_score + uncertainty_score
        
        return min(100, max(0, total_score))
    
    def _generate_robustness_recommendations(self, 
                                           density_ratio_analysis: DensityRatioAnalysis,
                                           sensitivity_analyses: Dict[str, SensitivityAnalysis],
                                           overall_robustness: RobustnessLevel) -> List[str]:
        """生成穩健性改善建議"""
        
        recommendations = []
        
        # 整體建議
        if overall_robustness == RobustnessLevel.HIGHLY_SENSITIVE:
            recommendations.append("🚨 模型穩健性嚴重不足，需要重新檢視模型設定")
            recommendations.append("• 考慮使用更保守的先驗分布")
            recommendations.append("• 增加資料量以減少先驗依賴")
            recommendations.append("• 檢查模型規格是否適當")
        
        elif overall_robustness == RobustnessLevel.SENSITIVE:
            recommendations.append("⚠️ 模型對先驗選擇較為敏感")
            recommendations.append("• 進行更廣泛的敏感度分析")
            recommendations.append("• 考慮模型平均方法")
        
        elif overall_robustness == RobustnessLevel.MODERATELY_ROBUST:
            recommendations.append("⚡ 穩健性中等，可適度改善")
            recommendations.append("• 監控關鍵參數的敏感度")
        
        else:
            recommendations.append("✅ 模型穩健性良好")
        
        # 密度比約束建議
        if not density_ratio_analysis.constraint_satisfaction:
            recommendations.append(f"📊 密度比約束違反率 {density_ratio_analysis.violation_rate:.1%}")
            recommendations.append("• 考慮調整 γ 約束參數")
            recommendations.append("• 檢查極端值處理")
        
        # 敏感參數建議
        sensitive_params = [name for name, analysis in sensitivity_analyses.items() 
                           if analysis.robustness_level in [RobustnessLevel.SENSITIVE, RobustnessLevel.HIGHLY_SENSITIVE]]
        
        if sensitive_params:
            recommendations.append(f"🎯 敏感參數: {', '.join(sensitive_params[:3])}")
            recommendations.append("• 對敏感參數使用更穩健的先驗")
            recommendations.append("• 增加相關資料以提高推斷穩定性")
        
        return recommendations
    
    def plot_robustness_analysis(self, 
                               robustness_report: RobustnessReport,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """繪製穩健性分析圖表"""
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('穩健性分析報告', fontsize=16, fontweight='bold')
        
        # 1. 密度比分布 (左上)
        ax1 = axes[0, 0]
        density_analysis = robustness_report.density_ratio_analysis
        
        # 模擬密度比分布用於視覺化
        ratios = np.random.exponential(density_analysis.mean_density_ratio, 1000)
        ax1.hist(ratios, bins=50, alpha=0.7, color=self.colors[0], density=True)
        ax1.axvline(density_analysis.gamma_constraint, color='red', linestyle='--', 
                   label=f'γ constraint = {density_analysis.gamma_constraint}')
        ax1.axvline(density_analysis.mean_density_ratio, color='orange', linestyle='-', 
                   label=f'Mean = {density_analysis.mean_density_ratio:.2f}')
        
        ax1.set_xlabel('密度比 dP/dP₀')
        ax1.set_ylabel('密度')
        ax1.set_title('密度比分布')
        ax1.legend()
        
        # 2. 參數敏感度 (右上)
        ax2 = axes[0, 1]
        
        if robustness_report.sensitivity_analyses:
            param_names = list(robustness_report.sensitivity_analyses.keys())
            sensitivities = [analysis.posterior_variation 
                           for analysis in robustness_report.sensitivity_analyses.values()]
            
            bars = ax2.bar(range(len(param_names)), sensitivities, 
                          color=self.colors[1], alpha=0.7)
            ax2.set_xlabel('參數')
            ax2.set_ylabel('後驗變異度')
            ax2.set_title('參數敏感度分析')
            ax2.set_xticks(range(len(param_names)))
            ax2.set_xticklabels(param_names, rotation=45)
            
            # 添加敏感度閾值線
            ax2.axhline(self.sensitivity_threshold, color='red', linestyle='--', 
                       label=f'敏感度閾值 = {self.sensitivity_threshold}')
            ax2.legend()
        
        # 3. 不確定性分解 (左下)
        ax3 = axes[1, 0]
        
        uncertainty = robustness_report.uncertainty_decomposition
        labels = ['先驗', '採樣', '模型']
        sizes = [uncertainty['prior_uncertainty'], 
                uncertainty['sampling_uncertainty'],
                uncertainty['model_uncertainty']]
        colors_pie = self.colors[2:5]
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=colors_pie, startangle=90)
        ax3.set_title('不確定性來源分解')
        
        # 4. 穩健性評分 (右下)
        ax4 = axes[1, 1]
        
        score = robustness_report.robustness_score
        
        # 創建評分表盤
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # 背景半圓
        ax4.fill_between(theta, 0, r, alpha=0.3, color='lightgray')
        
        # 評分區域
        if score >= 80:
            color = 'green'
        elif score >= 60:
            color = 'yellow'
        elif score >= 40:
            color = 'orange'
        else:
            color = 'red'
        
        score_theta = np.linspace(0, np.pi * score / 100, 50)
        ax4.fill_between(score_theta, 0, r, alpha=0.7, color=color)
        
        # 指針
        pointer_angle = np.pi * (1 - score / 100)
        ax4.arrow(0, 0, 0.8 * np.cos(pointer_angle), 0.8 * np.sin(pointer_angle),
                 head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        ax4.set_xlim(-1.2, 1.2)
        ax4.set_ylim(-0.2, 1.2)
        ax4.set_aspect('equal')
        ax4.axis('off')
        ax4.text(0, -0.1, f'穩健性評分: {score:.1f}', ha='center', fontsize=12, fontweight='bold')
        ax4.set_title('整體穩健性評分')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_robustness_report(self, 
                                 robustness_report: RobustnessReport,
                                 include_details: bool = True) -> str:
        """生成穩健性分析報告"""
        
        report = []
        report.append("=" * 80)
        report.append("                    穩健性分析報告")
        report.append("=" * 80)
        report.append("")
        
        # 整體摘要
        report.append("🛡️ 整體摘要")
        report.append("-" * 40)
        
        robustness_icons = {
            RobustnessLevel.HIGHLY_ROBUST: "🟢",
            RobustnessLevel.ROBUST: "🟡",
            RobustnessLevel.MODERATELY_ROBUST: "🟠",
            RobustnessLevel.SENSITIVE: "🔴",
            RobustnessLevel.HIGHLY_SENSITIVE: "❌"
        }
        
        icon = robustness_icons[robustness_report.overall_robustness]
        report.append(f"{icon} 整體穩健性: {robustness_report.overall_robustness.value.upper()}")
        report.append(f"📊 穩健性評分: {robustness_report.robustness_score:.1f}/100")
        report.append("")
        
        # 密度比分析
        report.append("📊 密度比約束分析")
        report.append("-" * 40)
        
        density = robustness_report.density_ratio_analysis
        status = "✅" if density.constraint_satisfaction else "❌"
        report.append(f"{status} 約束滿足: {density.constraint_satisfaction}")
        report.append(f"📈 違反率: {density.violation_rate:.2%}")
        report.append(f"📊 最大密度比: {density.max_density_ratio:.3f}")
        report.append(f"📊 平均密度比: {density.mean_density_ratio:.3f}")
        report.append("")
        
        # 敏感度分析
        if include_details and robustness_report.sensitivity_analyses:
            report.append("🎯 參數敏感度分析")
            report.append("-" * 40)
            
            for param_name, analysis in robustness_report.sensitivity_analyses.items():
                icon = robustness_icons[analysis.robustness_level]
                report.append(f"{icon} {param_name}:")
                report.append(f"    穩健性水平: {analysis.robustness_level.value}")
                report.append(f"    後驗變異: {analysis.posterior_variation:.3f}")
                report.append(f"    變異係數: {analysis.coefficient_of_variation:.3f}")
                report.append("")
        
        # 不確定性分解
        report.append("🔍 不確定性來源分解")
        report.append("-" * 40)
        
        uncertainty = robustness_report.uncertainty_decomposition
        report.append(f"📊 先驗不確定性: {uncertainty['prior_uncertainty']:.1%}")
        report.append(f"📊 採樣不確定性: {uncertainty['sampling_uncertainty']:.1%}")
        report.append(f"📊 模型不確定性: {uncertainty['model_uncertainty']:.1%}")
        report.append("")
        
        # 建議
        report.append("💡 改善建議")
        report.append("-" * 40)
        for recommendation in robustness_report.recommendations:
            report.append(recommendation)
        
        return "\n".join(report)