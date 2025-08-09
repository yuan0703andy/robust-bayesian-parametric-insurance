"""
Event-Level Analysis for Parametric Insurance Products
事件級參數型保險產品分析

This module provides detailed event-by-event analysis capabilities for parametric insurance products:
- Figure 5 style basis risk analysis and visualization
- Event severity classification and categorization
- Loss vs payout correlation analysis
- Basis risk categorization (loss_only, payout_only, both_triggered, perfect_match)
- Comprehensive event-level reporting and insights

本模組提供參數型保險產品的詳細事件級分析功能：
- Figure 5 風格的基差風險分析和視覺化
- 事件嚴重程度分類和分類
- 損失 vs 賠付相關性分析
- 基差風險分類 (loss_only, payout_only, both_triggered, perfect_match)
- 全面的事件級報告和洞察
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings

# 設置matplotlib中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BasisRiskCategory(Enum):
    """基差風險分類"""
    PERFECT_MATCH = "perfect_match"      # 完全匹配：損失和賠付都觸發
    LOSS_ONLY = "loss_only"             # 僅損失：有損失但無賠付
    PAYOUT_ONLY = "payout_only"         # 僅賠付：有賠付但無顯著損失
    BOTH_TRIGGERED = "both_triggered"    # 雙重觸發：有損失也有賠付但不匹配
    NO_ACTIVITY = "no_activity"         # 無活動：無損失也無賠付


class EventSeverity(Enum):
    """事件嚴重程度"""
    CATASTROPHIC = "catastrophic"       # 災難性 (> $5B)
    MAJOR = "major"                     # 重大 ($1B - $5B)
    MODERATE = "moderate"               # 中等 ($100M - $1B)
    MINOR = "minor"                     # 輕微 ($10M - $100M)
    MINIMAL = "minimal"                 # 極輕微 (< $10M)


@dataclass
class EventAnalysis:
    """單一事件分析結果"""
    event_id: str
    event_name: Optional[str]
    actual_loss: float
    predicted_payout: float
    basis_risk_category: BasisRiskCategory
    severity: EventSeverity
    parametric_indices: Dict[str, float]
    
    # 分析指標
    relative_basis_risk: float  # 相對基差風險 = |loss - payout| / max(loss, payout)
    coverage_ratio: float       # 覆蓋率 = payout / loss
    trigger_efficiency: float   # 觸發效率 = min(payout, loss) / max(payout, loss)
    
    # 元數據
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventLevelResults:
    """事件級分析結果"""
    individual_events: List[EventAnalysis]
    summary_statistics: Dict[str, float]
    basis_risk_distribution: Dict[BasisRiskCategory, int]
    severity_distribution: Dict[EventSeverity, int]
    correlation_metrics: Dict[str, float]
    recommendation_insights: List[str]
    
    # 可視化數據
    visualization_data: Dict[str, Any] = field(default_factory=dict)


class EventLevelAnalyzer:
    """
    事件級分析器
    
    提供全面的事件級參數型保險分析功能：
    - 事件分類和嚴重程度評估
    - 基差風險詳細分析
    - 觸發效率和覆蓋率計算
    - Figure 5 風格的分析和視覺化
    """
    
    def __init__(self, 
                 loss_threshold_minor: float = 1e7,      # $10M
                 loss_threshold_moderate: float = 1e8,   # $100M
                 loss_threshold_major: float = 1e9,      # $1B
                 loss_threshold_catastrophic: float = 5e9, # $5B
                 basis_risk_tolerance: float = 0.2):    # 20% 基差容忍度
        """
        初始化事件級分析器
        
        Parameters:
        -----------
        loss_threshold_minor : float
            輕微事件損失閾值
        loss_threshold_moderate : float
            中等事件損失閾值
        loss_threshold_major : float
            重大事件損失閾值
        loss_threshold_catastrophic : float
            災難性事件損失閾值
        basis_risk_tolerance : float
            基差風險容忍度 (相對誤差)
        """
        self.loss_thresholds = {
            'minimal': 0,
            'minor': loss_threshold_minor,
            'moderate': loss_threshold_moderate,
            'major': loss_threshold_major,
            'catastrophic': loss_threshold_catastrophic
        }
        
        self.basis_risk_tolerance = basis_risk_tolerance
        
        # 分析緩存
        self._analysis_cache = {}
        
        print(f"🔍 初始化事件級分析器")
        print(f"   損失分級閾值:")
        print(f"     輕微: ${loss_threshold_minor/1e6:.0f}M")
        print(f"     中等: ${loss_threshold_moderate/1e6:.0f}M") 
        print(f"     重大: ${loss_threshold_major/1e9:.1f}B")
        print(f"     災難性: ${loss_threshold_catastrophic/1e9:.1f}B")
        print(f"   基差風險容忍度: {basis_risk_tolerance*100:.1f}%")
    
    def analyze_events(self,
                      actual_losses: np.ndarray,
                      predicted_payouts: np.ndarray,
                      parametric_indices: Dict[str, np.ndarray],
                      event_metadata: Optional[pd.DataFrame] = None) -> EventLevelResults:
        """
        完整的事件級分析
        
        Parameters:
        -----------
        actual_losses : np.ndarray
            實際損失數組
        predicted_payouts : np.ndarray
            預測賠付數組
        parametric_indices : Dict[str, np.ndarray]
            參數指標字典
        event_metadata : pd.DataFrame, optional
            事件元數據
            
        Returns:
        --------
        EventLevelResults
            完整的事件級分析結果
        """
        
        print(f"🔄 開始事件級詳細分析...")
        print(f"   分析事件數: {len(actual_losses)}")
        print(f"   參數指標數: {len(parametric_indices)}")
        
        # 驗證輸入
        self._validate_inputs(actual_losses, predicted_payouts, parametric_indices)
        
        # 逐事件分析
        individual_events = []
        
        for i in range(len(actual_losses)):
            event_analysis = self._analyze_single_event(
                event_index=i,
                actual_loss=actual_losses[i],
                predicted_payout=predicted_payouts[i],
                parametric_indices={key: values[i] for key, values in parametric_indices.items()},
                event_metadata=event_metadata.iloc[i] if event_metadata is not None else None
            )
            individual_events.append(event_analysis)
        
        # 計算總體統計
        summary_stats = self._calculate_summary_statistics(individual_events)
        
        # 分布統計
        basis_risk_dist = self._calculate_basis_risk_distribution(individual_events)
        severity_dist = self._calculate_severity_distribution(individual_events)
        
        # 相關性分析
        correlation_metrics = self._calculate_correlation_metrics(
            actual_losses, predicted_payouts, parametric_indices
        )
        
        # 生成洞察和建議
        insights = self._generate_insights(
            individual_events, summary_stats, basis_risk_dist, correlation_metrics
        )
        
        # 準備視覺化數據
        viz_data = self._prepare_visualization_data(
            individual_events, actual_losses, predicted_payouts
        )
        
        results = EventLevelResults(
            individual_events=individual_events,
            summary_statistics=summary_stats,
            basis_risk_distribution=basis_risk_dist,
            severity_distribution=severity_dist,
            correlation_metrics=correlation_metrics,
            recommendation_insights=insights,
            visualization_data=viz_data
        )
        
        print(f"✅ 事件級分析完成！")
        print(f"   分析事件: {len(individual_events)}")
        print(f"   基差風險分類: {len(basis_risk_dist)} 種")
        print(f"   生成洞察: {len(insights)} 條")
        
        return results
    
    def _validate_inputs(self, actual_losses, predicted_payouts, parametric_indices):
        """驗證輸入數據的有效性"""
        n_events = len(actual_losses)
        
        if len(predicted_payouts) != n_events:
            raise ValueError(f"Length mismatch: losses={n_events}, payouts={len(predicted_payouts)}")
        
        for key, values in parametric_indices.items():
            if len(values) != n_events:
                raise ValueError(f"Parametric index '{key}' length mismatch: {len(values)} != {n_events}")
        
        # 檢查數值有效性
        if np.any(actual_losses < 0):
            warnings.warn("Negative losses detected")
        if np.any(predicted_payouts < 0):
            warnings.warn("Negative payouts detected")
    
    def _analyze_single_event(self,
                            event_index: int,
                            actual_loss: float,
                            predicted_payout: float,
                            parametric_indices: Dict[str, float],
                            event_metadata: Optional[pd.Series] = None) -> EventAnalysis:
        """分析單一事件"""
        
        # 基本資訊
        event_id = f"event_{event_index:04d}"
        event_name = None
        
        if event_metadata is not None:
            if 'event_id' in event_metadata:
                event_id = event_metadata['event_id']
            if 'event_name' in event_metadata:
                event_name = event_metadata['event_name']
        
        # 分類
        severity = self._classify_event_severity(actual_loss)
        basis_risk_category = self._classify_basis_risk(actual_loss, predicted_payout)
        
        # 計算風險指標
        relative_basis_risk = self._calculate_relative_basis_risk(actual_loss, predicted_payout)
        coverage_ratio = self._calculate_coverage_ratio(actual_loss, predicted_payout)
        trigger_efficiency = self._calculate_trigger_efficiency(actual_loss, predicted_payout)
        
        # 元數據
        metadata = {}
        if event_metadata is not None:
            metadata = event_metadata.to_dict()
        
        return EventAnalysis(
            event_id=event_id,
            event_name=event_name,
            actual_loss=actual_loss,
            predicted_payout=predicted_payout,
            basis_risk_category=basis_risk_category,
            severity=severity,
            parametric_indices=parametric_indices,
            relative_basis_risk=relative_basis_risk,
            coverage_ratio=coverage_ratio,
            trigger_efficiency=trigger_efficiency,
            metadata=metadata
        )
    
    def _classify_event_severity(self, loss: float) -> EventSeverity:
        """根據損失大小分類事件嚴重程度"""
        if loss >= self.loss_thresholds['catastrophic']:
            return EventSeverity.CATASTROPHIC
        elif loss >= self.loss_thresholds['major']:
            return EventSeverity.MAJOR
        elif loss >= self.loss_thresholds['moderate']:
            return EventSeverity.MODERATE
        elif loss >= self.loss_thresholds['minor']:
            return EventSeverity.MINOR
        else:
            return EventSeverity.MINIMAL
    
    def _classify_basis_risk(self, loss: float, payout: float) -> BasisRiskCategory:
        """分類基差風險類型"""
        loss_significant = loss > self.loss_thresholds['minor']
        payout_significant = payout > self.loss_thresholds['minor']
        
        if not loss_significant and not payout_significant:
            return BasisRiskCategory.NO_ACTIVITY
        elif loss_significant and not payout_significant:
            return BasisRiskCategory.LOSS_ONLY
        elif not loss_significant and payout_significant:
            return BasisRiskCategory.PAYOUT_ONLY
        else:
            # 兩者都有，檢查匹配程度
            relative_error = abs(loss - payout) / max(loss, payout)
            if relative_error <= self.basis_risk_tolerance:
                return BasisRiskCategory.PERFECT_MATCH
            else:
                return BasisRiskCategory.BOTH_TRIGGERED
    
    def _calculate_relative_basis_risk(self, loss: float, payout: float) -> float:
        """計算相對基差風險"""
        if max(loss, payout) == 0:
            return 0.0
        return abs(loss - payout) / max(loss, payout)
    
    def _calculate_coverage_ratio(self, loss: float, payout: float) -> float:
        """計算覆蓋率"""
        if loss == 0:
            return float('inf') if payout > 0 else 1.0
        return payout / loss
    
    def _calculate_trigger_efficiency(self, loss: float, payout: float) -> float:
        """計算觸發效率"""
        if max(loss, payout) == 0:
            return 1.0
        return min(loss, payout) / max(loss, payout)
    
    def _calculate_summary_statistics(self, events: List[EventAnalysis]) -> Dict[str, float]:
        """計算總體統計指標"""
        if not events:
            return {}
        
        # 提取數值
        losses = [e.actual_loss for e in events]
        payouts = [e.predicted_payout for e in events]
        basis_risks = [e.relative_basis_risk for e in events]
        coverage_ratios = [e.coverage_ratio for e in events if not np.isinf(e.coverage_ratio)]
        trigger_efficiencies = [e.trigger_efficiency for e in events]
        
        return {
            'total_events': len(events),
            'total_actual_loss': np.sum(losses),
            'total_predicted_payout': np.sum(payouts),
            'mean_actual_loss': np.mean(losses),
            'mean_predicted_payout': np.mean(payouts),
            'median_actual_loss': np.median(losses),
            'median_predicted_payout': np.median(payouts),
            'std_actual_loss': np.std(losses),
            'std_predicted_payout': np.std(payouts),
            'mean_relative_basis_risk': np.mean(basis_risks),
            'median_relative_basis_risk': np.median(basis_risks),
            'mean_coverage_ratio': np.mean(coverage_ratios) if coverage_ratios else 0,
            'mean_trigger_efficiency': np.mean(trigger_efficiencies),
            'max_loss_event_loss': np.max(losses),
            'max_payout_event_payout': np.max(payouts),
            'correlation_loss_payout': pearsonr(losses, payouts)[0] if len(losses) > 1 else 0
        }
    
    def _calculate_basis_risk_distribution(self, events: List[EventAnalysis]) -> Dict[BasisRiskCategory, int]:
        """計算基差風險分布"""
        distribution = {category: 0 for category in BasisRiskCategory}
        
        for event in events:
            distribution[event.basis_risk_category] += 1
        
        return distribution
    
    def _calculate_severity_distribution(self, events: List[EventAnalysis]) -> Dict[EventSeverity, int]:
        """計算嚴重程度分布"""
        distribution = {severity: 0 for severity in EventSeverity}
        
        for event in events:
            distribution[event.severity] += 1
        
        return distribution
    
    def _calculate_correlation_metrics(self,
                                     actual_losses: np.ndarray,
                                     predicted_payouts: np.ndarray,
                                     parametric_indices: Dict[str, np.ndarray]) -> Dict[str, float]:
        """計算相關性指標"""
        metrics = {}
        
        # 損失與賠付相關性
        if len(actual_losses) > 1:
            pearson_r, pearson_p = pearsonr(actual_losses, predicted_payouts)
            spearman_r, spearman_p = spearmanr(actual_losses, predicted_payouts)
            
            metrics['loss_payout_pearson'] = pearson_r
            metrics['loss_payout_pearson_pvalue'] = pearson_p
            metrics['loss_payout_spearman'] = spearman_r
            metrics['loss_payout_spearman_pvalue'] = spearman_p
        
        # 參數指標與損失的相關性
        for index_name, index_values in parametric_indices.items():
            if len(index_values) > 1 and np.std(index_values) > 0:
                try:
                    pearson_r, _ = pearsonr(index_values, actual_losses)
                    metrics[f'{index_name}_loss_correlation'] = pearson_r
                except:
                    metrics[f'{index_name}_loss_correlation'] = 0.0
        
        # 參數指標與賠付的相關性
        for index_name, index_values in parametric_indices.items():
            if len(index_values) > 1 and np.std(index_values) > 0:
                try:
                    pearson_r, _ = pearsonr(index_values, predicted_payouts)
                    metrics[f'{index_name}_payout_correlation'] = pearson_r
                except:
                    metrics[f'{index_name}_payout_correlation'] = 0.0
        
        return metrics
    
    def _generate_insights(self,
                          events: List[EventAnalysis],
                          summary_stats: Dict[str, float],
                          basis_risk_dist: Dict[BasisRiskCategory, int],
                          correlation_metrics: Dict[str, float]) -> List[str]:
        """生成分析洞察和建議"""
        insights = []
        
        total_events = len(events)
        if total_events == 0:
            return ["無可分析的事件數據"]
        
        # 基差風險分析
        perfect_match_rate = basis_risk_dist[BasisRiskCategory.PERFECT_MATCH] / total_events
        loss_only_rate = basis_risk_dist[BasisRiskCategory.LOSS_ONLY] / total_events
        payout_only_rate = basis_risk_dist[BasisRiskCategory.PAYOUT_ONLY] / total_events
        
        if perfect_match_rate > 0.7:
            insights.append(f"✅ 產品表現優秀：{perfect_match_rate*100:.1f}% 的事件實現完美匹配")
        elif perfect_match_rate > 0.5:
            insights.append(f"🔄 產品表現良好：{perfect_match_rate*100:.1f}% 的事件實現完美匹配，有改善空間")
        else:
            insights.append(f"⚠️ 產品需要優化：僅 {perfect_match_rate*100:.1f}% 的事件實現完美匹配")
        
        if loss_only_rate > 0.3:
            insights.append(f"📉 基差風險警告：{loss_only_rate*100:.1f}% 的損失事件未獲得賠付，建議降低觸發閾值")
        
        if payout_only_rate > 0.2:
            insights.append(f"💰 過度賠付風險：{payout_only_rate*100:.1f}% 的賠付發生在無重大損失時，建議提高觸發閾值")
        
        # 相關性分析
        loss_payout_corr = correlation_metrics.get('loss_payout_pearson', 0)
        if loss_payout_corr > 0.8:
            insights.append(f"🎯 強相關性：損失與賠付相關係數 {loss_payout_corr:.3f}，產品設計非常有效")
        elif loss_payout_corr > 0.6:
            insights.append(f"👍 中等相關性：損失與賠付相關係數 {loss_payout_corr:.3f}，產品設計基本有效")
        else:
            insights.append(f"⚠️ 弱相關性：損失與賠付相關係數 {loss_payout_corr:.3f}，需要重新設計參數指標")
        
        # 覆蓋率分析
        mean_coverage = summary_stats.get('mean_coverage_ratio', 0)
        if mean_coverage > 1.5:
            insights.append(f"📈 過度覆蓋：平均覆蓋率 {mean_coverage:.2f}，可能存在道德風險")
        elif mean_coverage < 0.5:
            insights.append(f"📉 覆蓋不足：平均覆蓋率 {mean_coverage:.2f}，無法有效轉移風險")
        else:
            insights.append(f"✅ 適當覆蓋：平均覆蓋率 {mean_coverage:.2f}，風險轉移效果良好")
        
        # 最佳參數指標識別
        best_corr = 0
        best_index = None
        for key, value in correlation_metrics.items():
            if '_loss_correlation' in key and abs(value) > abs(best_corr):
                best_corr = value
                best_index = key.replace('_loss_correlation', '')
        
        if best_index and abs(best_corr) > 0.7:
            insights.append(f"🏆 最佳參數指標：{best_index} (相關係數 {best_corr:.3f})")
        
        return insights
    
    def _prepare_visualization_data(self,
                                  events: List[EventAnalysis],
                                  actual_losses: np.ndarray,
                                  predicted_payouts: np.ndarray) -> Dict[str, Any]:
        """準備視覺化數據"""
        return {
            'scatter_data': {
                'actual_losses': actual_losses,
                'predicted_payouts': predicted_payouts,
                'basis_risk_categories': [e.basis_risk_category.value for e in events],
                'severities': [e.severity.value for e in events],
                'event_ids': [e.event_id for e in events]
            },
            'distribution_data': {
                'relative_basis_risks': [e.relative_basis_risk for e in events],
                'coverage_ratios': [e.coverage_ratio for e in events if not np.isinf(e.coverage_ratio)],
                'trigger_efficiencies': [e.trigger_efficiency for e in events]
            }
        }
    
    def create_loss_vs_payout_analysis(self, results: EventLevelResults,
                                     figsize: Tuple[int, int] = (15, 10),
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        創建 Figure 5 風格的損失 vs 賠付分析圖
        
        Parameters:
        -----------
        results : EventLevelResults
            事件級分析結果
        figsize : Tuple[int, int]
            圖表大小
        save_path : str, optional
            保存路徑
            
        Returns:
        --------
        plt.Figure
            圖表對象
        """
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('事件級損失 vs 賠付分析 (Figure 5 風格)', fontsize=16, fontweight='bold')
        
        scatter_data = results.visualization_data['scatter_data']
        dist_data = results.visualization_data['distribution_data']
        
        # 子圖1: 損失 vs 賠付散點圖 (按基差風險分類)
        ax1 = axes[0, 0]
        
        # 基差風險類別顏色映射
        color_map = {
            'perfect_match': '#2E8B57',    # 海綠色
            'both_triggered': '#FFD700',   # 金色
            'loss_only': '#DC143C',        # 深紅色
            'payout_only': '#4169E1',      # 皇家藍
            'no_activity': '#708090'       # 石板灰
        }
        
        for category in set(scatter_data['basis_risk_categories']):
            mask = np.array(scatter_data['basis_risk_categories']) == category
            ax1.scatter(
                np.array(scatter_data['actual_losses'])[mask] / 1e9,
                np.array(scatter_data['predicted_payouts'])[mask] / 1e9,
                c=color_map.get(category, '#000000'),
                label=category.replace('_', ' ').title(),
                alpha=0.7,
                s=50
            )
        
        # 添加對角線 (完美匹配線)
        max_val = max(
            np.max(scatter_data['actual_losses']) / 1e9,
            np.max(scatter_data['predicted_payouts']) / 1e9
        )
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='完美匹配線')
        
        ax1.set_xlabel('實際損失 (十億美元)')
        ax1.set_ylabel('預測賠付 (十億美元)')
        ax1.set_title('損失 vs 賠付 (按基差風險分類)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子圖2: 損失 vs 賠付散點圖 (按事件嚴重程度)
        ax2 = axes[0, 1]
        
        severity_colors = {
            'catastrophic': '#8B0000',    # 深紅
            'major': '#FF4500',           # 橙紅
            'moderate': '#FFA500',        # 橙色
            'minor': '#FFD700',           # 金色
            'minimal': '#90EE90'          # 淺綠
        }
        
        for severity in set(scatter_data['severities']):
            mask = np.array(scatter_data['severities']) == severity
            ax2.scatter(
                np.array(scatter_data['actual_losses'])[mask] / 1e9,
                np.array(scatter_data['predicted_payouts'])[mask] / 1e9,
                c=severity_colors.get(severity, '#000000'),
                label=severity.title(),
                alpha=0.7,
                s=50
            )
        
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        ax2.set_xlabel('實際損失 (十億美元)')
        ax2.set_ylabel('預測賠付 (十億美元)')
        ax2.set_title('損失 vs 賠付 (按事件嚴重程度)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子圖3: 相對基差風險分布
        ax3 = axes[1, 0]
        
        basis_risks = dist_data['relative_basis_risks']
        ax3.hist(basis_risks, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(basis_risks), color='red', linestyle='--', label=f'平均值: {np.mean(basis_risks):.3f}')
        ax3.axvline(np.median(basis_risks), color='orange', linestyle='--', label=f'中位數: {np.median(basis_risks):.3f}')
        
        ax3.set_xlabel('相對基差風險')
        ax3.set_ylabel('事件數量')
        ax3.set_title('相對基差風險分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子圖4: 覆蓋率分布
        ax4 = axes[1, 1]
        
        coverage_ratios = [cr for cr in dist_data['coverage_ratios'] if cr <= 5]  # 限制異常值
        if coverage_ratios:
            ax4.hist(coverage_ratios, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            ax4.axvline(np.mean(coverage_ratios), color='red', linestyle='--', 
                       label=f'平均值: {np.mean(coverage_ratios):.2f}')
            ax4.axvline(1.0, color='green', linestyle='-', linewidth=2, label='完美覆蓋')
        
        ax4.set_xlabel('覆蓋率 (賠付/損失)')
        ax4.set_ylabel('事件數量')
        ax4.set_title('覆蓋率分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 圖表已保存到: {save_path}")
        
        return fig
    
    def generate_event_analysis_report(self, results: EventLevelResults) -> str:
        """
        生成全面的事件級分析報告
        
        Parameters:
        -----------
        results : EventLevelResults
            事件級分析結果
            
        Returns:
        --------
        str
            格式化的分析報告
        """
        
        report = []
        report.append("=" * 80)
        report.append("                    事件級參數型保險分析報告")
        report.append("=" * 80)
        report.append("")
        
        # 總體統計
        stats = results.summary_statistics
        report.append("📊 總體統計")
        report.append("-" * 40)
        report.append(f"分析事件總數: {int(stats['total_events'])}")
        report.append(f"總實際損失: ${stats['total_actual_loss']/1e9:.2f}B")
        report.append(f"總預測賠付: ${stats['total_predicted_payout']/1e9:.2f}B")
        report.append(f"平均事件損失: ${stats['mean_actual_loss']/1e6:.1f}M")
        report.append(f"平均事件賠付: ${stats['mean_predicted_payout']/1e6:.1f}M")
        report.append(f"損失-賠付相關係數: {stats['correlation_loss_payout']:.3f}")
        report.append("")
        
        # 基差風險分布
        report.append("🎯 基差風險分析")
        report.append("-" * 40)
        total_events = stats['total_events']
        for category, count in results.basis_risk_distribution.items():
            percentage = (count / total_events) * 100
            report.append(f"{category.value.replace('_', ' ').title():<20}: {count:>3} ({percentage:>5.1f}%)")
        
        report.append(f"平均相對基差風險: {stats['mean_relative_basis_risk']:.3f}")
        report.append("")
        
        # 嚴重程度分布
        report.append("⚡ 事件嚴重程度分布")
        report.append("-" * 40)
        for severity, count in results.severity_distribution.items():
            percentage = (count / total_events) * 100
            report.append(f"{severity.value.title():<20}: {count:>3} ({percentage:>5.1f}%)")
        report.append("")
        
        # 關鍵洞察
        report.append("💡 關鍵洞察與建議")
        report.append("-" * 40)
        for i, insight in enumerate(results.recommendation_insights, 1):
            report.append(f"{i:>2}. {insight}")
        report.append("")
        
        # 頂級事件分析
        report.append("🏆 關鍵事件分析")
        report.append("-" * 40)
        
        # 找出最大損失事件
        max_loss_event = max(results.individual_events, key=lambda e: e.actual_loss)
        report.append(f"最大損失事件: {max_loss_event.event_id}")
        report.append(f"  實際損失: ${max_loss_event.actual_loss/1e9:.2f}B")
        report.append(f"  預測賠付: ${max_loss_event.predicted_payout/1e9:.2f}B")
        report.append(f"  覆蓋率: {max_loss_event.coverage_ratio:.2f}")
        report.append(f"  基差風險類別: {max_loss_event.basis_risk_category.value}")
        
        # 找出最大基差風險事件
        max_basis_risk_event = max(results.individual_events, key=lambda e: e.relative_basis_risk)
        report.append(f"最大基差風險事件: {max_basis_risk_event.event_id}")
        report.append(f"  相對基差風險: {max_basis_risk_event.relative_basis_risk:.3f}")
        report.append(f"  實際損失: ${max_basis_risk_event.actual_loss/1e6:.1f}M")
        report.append(f"  預測賠付: ${max_basis_risk_event.predicted_payout/1e6:.1f}M")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_detailed_results(self, results: EventLevelResults, 
                              output_path: str,
                              include_visualization: bool = True) -> None:
        """
        導出詳細的事件級分析結果
        
        Parameters:
        -----------
        results : EventLevelResults
            分析結果
        output_path : str
            輸出路徑 (不含副檔名)
        include_visualization : bool
            是否包含視覺化
        """
        
        # 1. 導出事件詳情為 CSV
        events_data = []
        for event in results.individual_events:
            event_dict = {
                'event_id': event.event_id,
                'event_name': event.event_name,
                'actual_loss': event.actual_loss,
                'predicted_payout': event.predicted_payout,
                'basis_risk_category': event.basis_risk_category.value,
                'severity': event.severity.value,
                'relative_basis_risk': event.relative_basis_risk,
                'coverage_ratio': event.coverage_ratio,
                'trigger_efficiency': event.trigger_efficiency
            }
            
            # 添加參數指標
            for key, value in event.parametric_indices.items():
                event_dict[f'param_{key}'] = value
            
            events_data.append(event_dict)
        
        events_df = pd.DataFrame(events_data)
        events_df.to_csv(f"{output_path}_events_detail.csv", index=False, encoding='utf-8-sig')
        
        # 2. 導出分析報告為文本文件
        report = self.generate_event_analysis_report(results)
        with open(f"{output_path}_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 3. 導出摘要統計為 JSON
        import json
        summary = {
            'summary_statistics': results.summary_statistics,
            'basis_risk_distribution': {k.value: v for k, v in results.basis_risk_distribution.items()},
            'severity_distribution': {k.value: v for k, v in results.severity_distribution.items()},
            'correlation_metrics': results.correlation_metrics,
            'recommendation_insights': results.recommendation_insights
        }
        
        with open(f"{output_path}_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 4. 創建並保存視覺化
        if include_visualization:
            fig = self.create_loss_vs_payout_analysis(results, save_path=f"{output_path}_visualization.png")
            plt.close(fig)
        
        print(f"✅ 事件級分析結果已導出到: {output_path}_*")
        print(f"   - 事件詳情: {output_path}_events_detail.csv")
        print(f"   - 分析報告: {output_path}_analysis_report.txt")
        print(f"   - 摘要統計: {output_path}_summary.json")
        if include_visualization:
            print(f"   - 視覺化圖表: {output_path}_visualization.png")


# 便利函數

def quick_event_analysis(actual_losses: np.ndarray,
                        predicted_payouts: np.ndarray,
                        parametric_indices: Dict[str, np.ndarray],
                        event_metadata: Optional[pd.DataFrame] = None,
                        output_prefix: Optional[str] = None) -> EventLevelResults:
    """
    快速事件級分析
    
    Parameters:
    -----------
    actual_losses : np.ndarray
        實際損失
    predicted_payouts : np.ndarray
        預測賠付
    parametric_indices : Dict[str, np.ndarray]
        參數指標
    event_metadata : pd.DataFrame, optional
        事件元數據
    output_prefix : str, optional
        輸出文件前綴
        
    Returns:
    --------
    EventLevelResults
        分析結果
    """
    
    analyzer = EventLevelAnalyzer()
    results = analyzer.analyze_events(
        actual_losses, predicted_payouts, parametric_indices, event_metadata
    )
    
    # 輸出報告
    print(analyzer.generate_event_analysis_report(results))
    
    # 保存結果
    if output_prefix:
        analyzer.export_detailed_results(results, output_prefix)
    
    return results


def compare_product_event_performance(products_results: Dict[str, EventLevelResults]) -> pd.DataFrame:
    """
    比較多個產品的事件級表現
    
    Parameters:
    -----------
    products_results : Dict[str, EventLevelResults]
        產品結果字典 {product_name: results}
        
    Returns:
    --------
    pd.DataFrame
        比較表
    """
    
    comparison_data = []
    
    for product_name, results in products_results.items():
        stats = results.summary_statistics
        basis_risk_dist = results.basis_risk_distribution
        
        # 計算關鍵指標
        total_events = stats['total_events']
        perfect_match_rate = basis_risk_dist[BasisRiskCategory.PERFECT_MATCH] / total_events
        loss_only_rate = basis_risk_dist[BasisRiskCategory.LOSS_ONLY] / total_events
        
        comparison_data.append({
            'product_name': product_name,
            'total_events': int(total_events),
            'perfect_match_rate': perfect_match_rate,
            'loss_only_rate': loss_only_rate,
            'mean_basis_risk': stats['mean_relative_basis_risk'],
            'loss_payout_correlation': stats['correlation_loss_payout'],
            'mean_coverage_ratio': stats.get('mean_coverage_ratio', 0),
            'total_loss_billion': stats['total_actual_loss'] / 1e9,
            'total_payout_billion': stats['total_predicted_payout'] / 1e9
        })
    
    return pd.DataFrame(comparison_data)