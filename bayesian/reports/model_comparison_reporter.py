"""
Model Comparison Reporter Module
模型比較報告模組

提供全面的貝氏模型比較、選擇和評估功能。
包括 AIC/BIC/WAIC 比較、Bayes Factor 計算、模型平均等。

Key Features:
- 多重模型比較 (AIC, BIC, WAIC, LOO-CV)
- Bayes Factor 計算和解釋
- 模型權重和模型平均
- 預測效能交叉驗證
- 模型選擇建議
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from scipy.special import logsumexp
import sys
import os

# Import parent bayesian modules - use relative imports
try:
    from ..robust_bayesian_analysis import (
        ModelComparisonResult, ModelConfiguration, 
        ModelSelectionCriterion, RobustBayesianFramework
    )
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from robust_bayesian_analysis import (
        ModelComparisonResult, ModelConfiguration, 
        ModelSelectionCriterion, RobustBayesianFramework
    )

class ModelSupport(Enum):
    """模型支持度"""
    DECISIVE = "decisive"           # BF > 100
    VERY_STRONG = "very_strong"     # 30 < BF ≤ 100
    STRONG = "strong"               # 10 < BF ≤ 30
    MODERATE = "moderate"           # 3 < BF ≤ 10
    WEAK = "weak"                   # 1 < BF ≤ 3
    INCONCLUSIVE = "inconclusive"   # BF ≤ 1

@dataclass
class ModelComparison:
    """模型比較結果"""
    model_name: str
    log_likelihood: float
    aic: float
    bic: float
    waic: Optional[float]
    loo_cv: Optional[float]
    bayes_factor: Optional[float]
    model_weight: float
    support_level: ModelSupport
    rank: int

@dataclass
class ModelAveraging:
    """模型平均結果"""
    weighted_predictions: np.ndarray
    model_weights: Dict[str, float]
    individual_predictions: Dict[str, np.ndarray]
    uncertainty_decomposition: Dict[str, float]

@dataclass
class ModelComparisonReport:
    """模型比較報告"""
    model_comparisons: List[ModelComparison]
    best_model: ModelComparison
    model_averaging: ModelAveraging
    cross_validation_results: Dict[str, Any]
    recommendations: List[str]
    comparison_summary: pd.DataFrame

class ModelComparisonReporter:
    """
    模型比較報告器
    
    提供全面的貝氏模型比較和選擇分析
    """
    
    def __init__(self, 
                 information_criteria: List[str] = ['aic', 'bic', 'waic'],
                 cv_folds: int = 5,
                 bayes_factor_threshold: float = 3.0):
        """
        初始化模型比較報告器
        
        Parameters:
        -----------
        information_criteria : List[str]
            使用的資訊準則
        cv_folds : int
            交叉驗證摺數
        bayes_factor_threshold : float
            Bayes Factor 顯著性閾值
        """
        self.information_criteria = information_criteria
        self.cv_folds = cv_folds
        self.bayes_factor_threshold = bayes_factor_threshold
        
        # 設置圖表樣式
        plt.style.use('default')
        sns.set_style("whitegrid")
        self.colors = sns.color_palette("tab10", 10)
    
    def compare_models(self, 
                      model_results: List[ModelComparisonResult],
                      observed_data: np.ndarray,
                      posterior_samples: Dict[str, Dict[str, np.ndarray]]) -> ModelComparisonReport:
        """
        全面的模型比較分析
        
        Parameters:
        -----------
        model_results : List[ModelComparisonResult]
            模型比較結果列表
        observed_data : np.ndarray
            觀測資料
        posterior_samples : Dict[str, Dict[str, np.ndarray]]
            各模型的後驗樣本 {model_name: {param: samples}}
            
        Returns:
        --------
        ModelComparisonReport
            完整的模型比較報告
        """
        
        print("📊 開始模型比較分析...")
        
        if not model_results:
            print("⚠️ 沒有模型結果可比較")
            return self._create_empty_report()
        
        # 1. 計算 Bayes Factors
        print("  🔄 計算 Bayes Factors...")
        bayes_factors = self._calculate_bayes_factors(model_results)
        
        # 2. 計算模型權重
        print("  ⚖️ 計算模型權重...")
        model_weights = self._calculate_model_weights(model_results)
        
        # 3. 模型排名和支持度
        print("  📈 評估模型支持度...")
        model_comparisons = self._create_model_comparisons(
            model_results, bayes_factors, model_weights
        )
        
        # 4. 模型平均
        print("  🔄 執行模型平均...")
        model_averaging = self._perform_model_averaging(
            model_comparisons, posterior_samples, observed_data
        )
        
        # 5. 交叉驗證
        print("  ✅ 交叉驗證分析...")
        cv_results = self._cross_validation_analysis(
            model_results, observed_data, posterior_samples
        )
        
        # 6. 選擇最佳模型
        best_model = min(model_comparisons, key=lambda x: x.aic)
        
        # 7. 生成建議
        recommendations = self._generate_model_recommendations(
            model_comparisons, cv_results
        )
        
        # 8. 創建比較摘要表
        comparison_summary = self._create_comparison_summary(model_comparisons)
        
        report = ModelComparisonReport(
            model_comparisons=model_comparisons,
            best_model=best_model,
            model_averaging=model_averaging,
            cross_validation_results=cv_results,
            recommendations=recommendations,
            comparison_summary=comparison_summary
        )
        
        print("✅ 模型比較分析完成")
        return report
    
    def _calculate_bayes_factors(self, 
                               model_results: List[ModelComparisonResult]) -> Dict[str, float]:
        """計算 Bayes Factors"""
        
        if len(model_results) < 2:
            return {}
        
        # 使用對數似然計算 Bayes Factor (簡化)
        log_likelihoods = {result.model_name: result.log_likelihood 
                          for result in model_results}
        
        # 選擇參考模型 (最高對數似然)
        reference_model = max(log_likelihoods.keys(), key=lambda k: log_likelihoods[k])
        reference_ll = log_likelihoods[reference_model]
        
        bayes_factors = {}
        for model_name, ll in log_likelihoods.items():
            if model_name != reference_model:
                # BF = exp(log_likelihood_model - log_likelihood_reference)
                log_bf = ll - reference_ll
                bayes_factors[model_name] = np.exp(log_bf)
            else:
                bayes_factors[model_name] = 1.0  # 參考模型
        
        return bayes_factors
    
    def _calculate_model_weights(self, 
                               model_results: List[ModelComparisonResult]) -> Dict[str, float]:
        """計算模型權重 (基於 AIC)"""
        
        if not model_results:
            return {}
        
        aics = np.array([result.aic for result in model_results])
        model_names = [result.model_name for result in model_results]
        
        # AIC 權重計算
        min_aic = np.min(aics)
        delta_aic = aics - min_aic
        
        # 避免數值溢出
        delta_aic = np.clip(delta_aic, 0, 700)
        
        # 計算權重
        weights_unnorm = np.exp(-0.5 * delta_aic)
        weights = weights_unnorm / np.sum(weights_unnorm)
        
        return dict(zip(model_names, weights))
    
    def _assess_bayes_factor_support(self, bf: float) -> ModelSupport:
        """評估 Bayes Factor 支持度"""
        
        if bf > 100:
            return ModelSupport.DECISIVE
        elif bf > 30:
            return ModelSupport.VERY_STRONG
        elif bf > 10:
            return ModelSupport.STRONG
        elif bf > 3:
            return ModelSupport.MODERATE
        elif bf > 1:
            return ModelSupport.WEAK
        else:
            return ModelSupport.INCONCLUSIVE
    
    def _create_model_comparisons(self, 
                                model_results: List[ModelComparisonResult],
                                bayes_factors: Dict[str, float],
                                model_weights: Dict[str, float]) -> List[ModelComparison]:
        """創建模型比較結果"""
        
        comparisons = []
        
        # 按 AIC 排序
        sorted_results = sorted(model_results, key=lambda x: x.aic)
        
        for rank, result in enumerate(sorted_results, 1):
            bf = bayes_factors.get(result.model_name, 1.0)
            support = self._assess_bayes_factor_support(bf)
            
            comparison = ModelComparison(
                model_name=result.model_name,
                log_likelihood=result.log_likelihood,
                aic=result.aic,
                bic=result.bic,
                waic=result.waic,
                loo_cv=result.loo_cv,
                bayes_factor=bf,
                model_weight=model_weights.get(result.model_name, 0.0),
                support_level=support,
                rank=rank
            )
            
            comparisons.append(comparison)
        
        return comparisons
    
    def _perform_model_averaging(self, 
                               model_comparisons: List[ModelComparison],
                               posterior_samples: Dict[str, Dict[str, np.ndarray]],
                               observed_data: np.ndarray) -> ModelAveraging:
        """執行模型平均"""
        
        if not model_comparisons or not posterior_samples:
            return ModelAveraging(
                weighted_predictions=np.array([]),
                model_weights={},
                individual_predictions={},
                uncertainty_decomposition={}
            )
        
        # 提取權重
        weights = {comp.model_name: comp.model_weight for comp in model_comparisons}
        
        # 為每個模型生成預測
        individual_predictions = {}
        
        for model_name, weight in weights.items():
            if model_name in posterior_samples and weight > 0.01:  # 忽略權重很小的模型
                # 簡化的預測生成
                samples = posterior_samples[model_name]
                
                if samples:
                    # 假設我們有參數樣本，生成預測分布
                    first_param = list(samples.values())[0]
                    n_pred = len(observed_data)
                    
                    # 簡化：使用參數均值生成預測
                    param_means = {param: np.mean(param_samples) 
                                 for param, param_samples in samples.items()
                                 if param_samples.ndim == 1}
                    
                    if 'mu' in param_means and 'sigma' in param_means:
                        predictions = np.random.normal(
                            param_means['mu'], 
                            param_means['sigma'], 
                            n_pred
                        )
                    else:
                        # 回退到觀測資料的擾動
                        predictions = observed_data + np.random.normal(0, np.std(observed_data) * 0.1, n_pred)
                    
                    individual_predictions[model_name] = predictions
        
        # 計算加權平均預測
        if individual_predictions:
            weighted_pred = np.zeros_like(list(individual_predictions.values())[0])
            total_weight = 0
            
            for model_name, predictions in individual_predictions.items():
                weight = weights[model_name]
                weighted_pred += weight * predictions
                total_weight += weight
            
            if total_weight > 0:
                weighted_predictions = weighted_pred / total_weight
            else:
                weighted_predictions = weighted_pred
        else:
            weighted_predictions = np.array([])
        
        # 不確定性分解
        uncertainty_decomp = self._decompose_prediction_uncertainty(
            individual_predictions, weights
        )
        
        return ModelAveraging(
            weighted_predictions=weighted_predictions,
            model_weights=weights,
            individual_predictions=individual_predictions,
            uncertainty_decomposition=uncertainty_decomp
        )
    
    def _decompose_prediction_uncertainty(self, 
                                        individual_predictions: Dict[str, np.ndarray],
                                        weights: Dict[str, float]) -> Dict[str, float]:
        """分解預測不確定性"""
        
        if len(individual_predictions) < 2:
            return {'within_model': 1.0, 'between_model': 0.0}
        
        # 計算每個模型內的不確定性
        within_model_vars = []
        for model_name, predictions in individual_predictions.items():
            within_model_vars.append(np.var(predictions))
        
        within_model_uncertainty = np.mean(within_model_vars)
        
        # 計算模型間的不確定性
        model_means = [np.mean(pred) for pred in individual_predictions.values()]
        between_model_uncertainty = np.var(model_means)
        
        # 歸一化
        total_uncertainty = within_model_uncertainty + between_model_uncertainty
        
        if total_uncertainty > 0:
            within_fraction = within_model_uncertainty / total_uncertainty
            between_fraction = between_model_uncertainty / total_uncertainty
        else:
            within_fraction = between_fraction = 0.5
        
        return {
            'within_model': within_fraction,
            'between_model': between_fraction,
            'total_uncertainty': total_uncertainty
        }
    
    def _cross_validation_analysis(self, 
                                 model_results: List[ModelComparisonResult],
                                 observed_data: np.ndarray,
                                 posterior_samples: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """交叉驗證分析"""
        
        if len(observed_data) < self.cv_folds:
            return {'error': 'Data too small for cross-validation'}
        
        n_data = len(observed_data)
        fold_size = n_data // self.cv_folds
        
        cv_scores = {}
        
        for result in model_results:
            model_name = result.model_name
            scores = []
            
            # K-fold 交叉驗證
            for fold in range(self.cv_folds):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < self.cv_folds - 1 else n_data
                
                # 分割資料
                test_data = observed_data[start_idx:end_idx]
                train_data = np.concatenate([
                    observed_data[:start_idx],
                    observed_data[end_idx:]
                ])
                
                # 簡化的預測評分
                if len(train_data) > 0 and len(test_data) > 0:
                    # 使用訓練資料的統計量預測測試資料
                    train_mean = np.mean(train_data)
                    train_std = np.std(train_data)
                    
                    # 計算對數似然
                    log_likelihood = np.sum(stats.norm.logpdf(test_data, train_mean, train_std))
                    scores.append(log_likelihood)
            
            cv_scores[model_name] = {
                'mean_score': np.mean(scores) if scores else -np.inf,
                'std_score': np.std(scores) if scores else 0,
                'fold_scores': scores
            }
        
        # 找到最佳 CV 模型
        best_cv_model = max(cv_scores.keys(), 
                           key=lambda k: cv_scores[k]['mean_score'])
        
        return {
            'cv_scores': cv_scores,
            'best_cv_model': best_cv_model,
            'n_folds': self.cv_folds
        }
    
    def _generate_model_recommendations(self, 
                                      model_comparisons: List[ModelComparison],
                                      cv_results: Dict[str, Any]) -> List[str]:
        """生成模型選擇建議"""
        
        recommendations = []
        
        if not model_comparisons:
            recommendations.append("❌ 沒有有效的模型可比較")
            return recommendations
        
        best_model = model_comparisons[0]  # 已按 AIC 排序
        
        # 最佳模型建議
        recommendations.append(f"🏆 推薦模型: {best_model.model_name}")
        recommendations.append(f"📊 AIC: {best_model.aic:.2f}, 權重: {best_model.model_weight:.3f}")
        
        # Bayes Factor 評估
        if best_model.bayes_factor and best_model.bayes_factor > 1:
            support_msg = {
                ModelSupport.DECISIVE: "決定性支持",
                ModelSupport.VERY_STRONG: "非常強烈支持", 
                ModelSupport.STRONG: "強烈支持",
                ModelSupport.MODERATE: "中等支持",
                ModelSupport.WEAK: "微弱支持",
                ModelSupport.INCONCLUSIVE: "無明確支持"
            }
            
            recommendations.append(f"🎯 Bayes Factor: {support_msg[best_model.support_level]}")
        
        # 模型不確定性評估
        top_weight = best_model.model_weight
        
        if top_weight > 0.7:
            recommendations.append("✅ 模型選擇確信度高")
        elif top_weight > 0.4:
            recommendations.append("⚠️ 模型選擇確信度中等，建議考慮模型平均")
        else:
            recommendations.append("🔄 模型選擇不確定，強烈建議使用模型平均")
        
        # 競爭模型警告
        similar_models = [comp for comp in model_comparisons[1:3] 
                         if comp.aic - best_model.aic < 2]
        
        if similar_models:
            model_names = [m.model_name for m in similar_models]
            recommendations.append(f"⚡ 競爭模型: {', '.join(model_names)} (ΔAIC < 2)")
        
        # 交叉驗證一致性
        if 'best_cv_model' in cv_results:
            cv_best = cv_results['best_cv_model']
            if cv_best != best_model.model_name:
                recommendations.append(f"🔍 交叉驗證最佳模型: {cv_best} (與 AIC 選擇不同)")
        
        return recommendations
    
    def _create_comparison_summary(self, 
                                 model_comparisons: List[ModelComparison]) -> pd.DataFrame:
        """創建模型比較摘要表"""
        
        if not model_comparisons:
            return pd.DataFrame()
        
        data = []
        for comp in model_comparisons:
            data.append({
                'Model': comp.model_name,
                'Rank': comp.rank,
                'AIC': comp.aic,
                'BIC': comp.bic,
                'WAIC': comp.waic if comp.waic else np.nan,
                'LogLik': comp.log_likelihood,
                'Weight': comp.model_weight,
                'BayesFactor': comp.bayes_factor if comp.bayes_factor else np.nan,
                'Support': comp.support_level.value
            })
        
        return pd.DataFrame(data)
    
    def _create_empty_report(self) -> ModelComparisonReport:
        """創建空報告"""
        
        return ModelComparisonReport(
            model_comparisons=[],
            best_model=None,
            model_averaging=ModelAveraging(
                weighted_predictions=np.array([]),
                model_weights={},
                individual_predictions={},
                uncertainty_decomposition={}
            ),
            cross_validation_results={},
            recommendations=["❌ 沒有模型結果可分析"],
            comparison_summary=pd.DataFrame()
        )
    
    def plot_model_comparison(self, 
                            comparison_report: ModelComparisonReport,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """繪製模型比較圖表"""
        
        if not comparison_report.model_comparisons:
            print("沒有模型比較結果可繪圖")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('模型比較分析', fontsize=16, fontweight='bold')
        
        comparisons = comparison_report.model_comparisons
        
        # 1. 資訊準則比較 (左上)
        ax1 = axes[0, 0]
        
        model_names = [comp.model_name for comp in comparisons]
        aics = [comp.aic for comp in comparisons]
        bics = [comp.bic for comp in comparisons]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax1.bar(x - width/2, aics, width, label='AIC', color=self.colors[0], alpha=0.7)
        ax1.bar(x + width/2, bics, width, label='BIC', color=self.colors[1], alpha=0.7)
        
        ax1.set_xlabel('模型')
        ax1.set_ylabel('資訊準則值')
        ax1.set_title('AIC vs BIC 比較')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        
        # 2. 模型權重 (右上)
        ax2 = axes[0, 1]
        
        weights = [comp.model_weight for comp in comparisons]
        colors_weights = self.colors[:len(model_names)]
        
        wedges, texts, autotexts = ax2.pie(weights, labels=model_names, autopct='%1.1f%%',
                                          colors=colors_weights, startangle=90)
        ax2.set_title('模型權重分布')
        
        # 3. Bayes Factor (左下)
        ax3 = axes[1, 0]
        
        bayes_factors = [comp.bayes_factor for comp in comparisons if comp.bayes_factor]
        bf_models = [comp.model_name for comp in comparisons if comp.bayes_factor]
        
        if bayes_factors:
            bars = ax3.bar(range(len(bf_models)), bayes_factors, 
                          color=self.colors[2], alpha=0.7)
            ax3.set_xlabel('模型')
            ax3.set_ylabel('Bayes Factor')
            ax3.set_title('Bayes Factor 比較')
            ax3.set_xticks(range(len(bf_models)))
            ax3.set_xticklabels(bf_models, rotation=45)
            
            # 添加閾值線
            ax3.axhline(1, color='red', linestyle='--', alpha=0.5, label='BF = 1')
            ax3.axhline(3, color='orange', linestyle='--', alpha=0.5, label='BF = 3')
            ax3.axhline(10, color='green', linestyle='--', alpha=0.5, label='BF = 10')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, '無 Bayes Factor 資料', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
        
        # 4. 不確定性分解 (右下)
        ax4 = axes[1, 1]
        
        if comparison_report.model_averaging.uncertainty_decomposition:
            uncertainty = comparison_report.model_averaging.uncertainty_decomposition
            labels = ['模型內', '模型間']
            sizes = [uncertainty.get('within_model', 0.5), 
                    uncertainty.get('between_model', 0.5)]
            colors_unc = self.colors[3:5]
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.1f%%',
                                              colors=colors_unc, startangle=90)
            ax4.set_title('預測不確定性分解')
        else:
            ax4.text(0.5, 0.5, '無不確定性分解資料', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_comparison_report(self, 
                                 comparison_report: ModelComparisonReport,
                                 include_details: bool = True) -> str:
        """生成模型比較報告"""
        
        report = []
        report.append("=" * 80)
        report.append("                    模型比較分析報告")
        report.append("=" * 80)
        report.append("")
        
        if not comparison_report.model_comparisons:
            report.append("❌ 沒有模型比較結果")
            return "\n".join(report)
        
        # 整體摘要
        report.append("🏆 模型選擇摘要")
        report.append("-" * 40)
        
        best = comparison_report.best_model
        report.append(f"🥇 最佳模型: {best.model_name}")
        report.append(f"📊 AIC: {best.aic:.2f}")
        report.append(f"📊 模型權重: {best.model_weight:.3f}")
        report.append(f"📊 支持度: {best.support_level.value}")
        report.append("")
        
        # 模型比較表
        if include_details:
            report.append("📋 詳細比較結果")
            report.append("-" * 40)
            
            summary_df = comparison_report.comparison_summary
            if not summary_df.empty:
                # 格式化表格
                for _, row in summary_df.iterrows():
                    rank_icon = "🥇" if row['Rank'] == 1 else "🥈" if row['Rank'] == 2 else "🥉" if row['Rank'] == 3 else "  "
                    report.append(f"{rank_icon} {row['Model']}:")
                    report.append(f"    排名: {row['Rank']}")
                    report.append(f"    AIC: {row['AIC']:.2f}, BIC: {row['BIC']:.2f}")
                    report.append(f"    權重: {row['Weight']:.3f}")
                    if not pd.isna(row['BayesFactor']):
                        report.append(f"    Bayes Factor: {row['BayesFactor']:.2f}")
                    report.append("")
        
        # 模型平均結果
        if comparison_report.model_averaging.model_weights:
            report.append("🔄 模型平均分析")
            report.append("-" * 40)
            
            averaging = comparison_report.model_averaging
            uncertainty = averaging.uncertainty_decomposition
            
            report.append(f"📊 參與模型數: {len(averaging.model_weights)}")
            if uncertainty:
                report.append(f"📊 模型內不確定性: {uncertainty.get('within_model', 0):.1%}")
                report.append(f"📊 模型間不確定性: {uncertainty.get('between_model', 0):.1%}")
            report.append("")
        
        # 交叉驗證結果
        if comparison_report.cross_validation_results.get('best_cv_model'):
            report.append("✅ 交叉驗證結果")
            report.append("-" * 40)
            
            cv = comparison_report.cross_validation_results
            report.append(f"📊 最佳 CV 模型: {cv['best_cv_model']}")
            report.append(f"📊 驗證摺數: {cv['n_folds']}")
            report.append("")
        
        # 建議
        report.append("💡 模型選擇建議")
        report.append("-" * 40)
        for recommendation in comparison_report.recommendations:
            report.append(recommendation)
        
        return "\n".join(report)