"""
Bayesian Visualization Module
Bayesian Analysis Visualization Module

This module provides comprehensive visualization capabilities for Robust Bayesian Analysis results,
including probabilistic basis risk distributions, robustness analysis charts, and sensitivity plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings

# 使用絕對導入避免相對導入問題
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Updated to use new bayesian module architecture
    from bayesian import (
        RobustBayesianAnalyzer,
        HierarchicalBayesianModel,
        ProbabilisticLossDistributionGenerator
    )
    HAS_BAYESIAN = True
except ImportError:
    HAS_BAYESIAN = False
    # 提供基本的類型定義
    class RobustBayesianAnalyzer:
        pass
    class HierarchicalBayesianModel:
        pass
    class ProbabilisticLossDistributionGenerator:
        pass

# Import skill scores for visualization
try:
    from skill_scores import (
        calculate_crps, calculate_edi, calculate_tss,
        calculate_rmse, calculate_mae
    )
    HAS_SKILL_SCORES = True
except ImportError:
    HAS_SKILL_SCORES = False
    warnings.warn("skill_scores module not available for visualization")

# Define legacy types for backward compatibility
class ProbabilisticLossDistribution:
    pass
class BayesianAnalysisResult:
    pass
from enum import Enum
class PriorScenario(Enum):
    NORMAL = "normal"
    GAMMA = "gamma"
class LikelihoodType(Enum):
    NORMAL = "normal"
    GAMMA = "gamma"

class BayesianVisualization:
    """
    Bayesian Analysis Visualizer
    
    Provides comprehensive visualization capabilities for Robust Bayesian Analysis results,
    including probabilistic basis risk, robustness analysis charts, and sensitivity plots.
    """
    
    def __init__(self, style: str = "whitegrid", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        style : str
            Chart style
        figsize : Tuple[int, int]
            Default figure size
        """
        self.style = style
        self.default_figsize = figsize
        
        # Set visualization style
        plt.style.use('default')
        sns.set_style(style)
        
        # Color configuration
        self.colors = {
            'neutral': '#2E86AB',
            'optimistic': '#A23B72', 
            'pessimistic': '#F18F01',
            'conservative': '#C73E1D',
            'robust': '#4CAF50',
            'non_robust': '#FF5722'
        }
    
    def plot_probabilistic_basis_risk(self,
                                    basis_risk_distributions: Dict[str, np.ndarray],
                                    product_names: List[str] = None,
                                    save_path: str = None,
                                    figsize: Tuple[int, int] = None) -> plt.Figure:
        """
        Presentation Method 1: Probabilistic Basis Risk Distribution
        
        Plot the complete probability distribution of basis risk, clearly showing
        probabilities of under-payment and over-payment.
        
        Parameters:
        -----------
        basis_risk_distributions : Dict[str, np.ndarray]
            Product basis risk distributions
        product_names : List[str], optional
            List of product names
        save_path : str, optional
            Save path
        figsize : Tuple[int, int], optional
            Figure size
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        
        figsize = figsize or self.default_figsize
        n_products = len(basis_risk_distributions)
        
        # 創建子圖
        if n_products <= 3:
            fig, axes = plt.subplots(1, n_products, figsize=(figsize[0], figsize[1]//2))
        else:
            n_cols = 3
            n_rows = (n_products + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows // 2))
        
        if n_products == 1:
            axes = [axes]
        elif n_products > 1:
            axes = axes.flatten()
        
        for i, (product_id, basis_risks) in enumerate(basis_risk_distributions.items()):
            ax = axes[i] if i < len(axes) else axes[0]
            
            # 計算統計量
            under_payment_prob = np.mean(basis_risks > 0) * 100  # 理賠不足機率
            over_payment_prob = np.mean(basis_risks < 0) * 100   # 過度理賠機率
            perfect_payment_prob = np.mean(basis_risks == 0) * 100  # 完美理賠機率
            
            # 繪製直方圖
            ax.hist(basis_risks / 1e9, bins=50, alpha=0.7, density=True, 
                   color=self.colors['neutral'], edgecolor='white', linewidth=0.5)
            
            # 添加垂直線標示零點
            ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='完美理賠')
            
            # 添加統計信息
            ax.text(0.05, 0.95, f'理賠不足: {under_payment_prob:.1f}%\n過度理賠: {over_payment_prob:.1f}%', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 設置標籤
            product_name = product_names[i] if product_names and i < len(product_names) else product_id
            ax.set_title(f'{product_name[:30]}...', fontsize=10, fontweight='bold')
            ax.set_xlabel('基差風險 (十億美元)', fontsize=9)
            ax.set_ylabel('機率密度', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 隱藏多餘的子圖
        for i in range(n_products, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 添加總標題
        fig.suptitle('機率化基差風險分佈 - 理賠不足 vs 過度理賠分析', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已保存至: {save_path}")
        
        return fig
    
    def plot_robustness_ranking(self,
                              robustness_results: Dict[str, Any],
                              top_n: int = 10,
                              save_path: str = None,
                              figsize: Tuple[int, int] = None) -> plt.Figure:
        """
        呈現方式二：優化結果的穩健性分析
        
        繪製產品在不同模型假設下的績效排名，識別穩健產品。
        
        Parameters:
        -----------
        robustness_results : Dict[str, Any]
            穩健性分析結果
        top_n : int
            顯示前N個產品
        save_path : str, optional
            Save path
        figsize : Tuple[int, int], optional
            Figure size
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        
        figsize = figsize or self.default_figsize
        
        # 提取數據
        performance_matrix = robustness_results['product_performance_matrix']
        robustness_metrics = robustness_results['robustness_metrics']
        robust_products = set(robustness_results.get('robust_products', []))
        
        # 選擇前N個產品（按平均CRPS排序）
        sorted_products = sorted(robustness_metrics.items(), 
                               key=lambda x: x[1]['mean_crps'])[:top_n]
        
        product_ids = [pid for pid, _ in sorted_products]
        scenarios = list(next(iter(performance_matrix.values())).keys())
        
        # 創建排名矩陣
        ranking_matrix = np.zeros((len(product_ids), len(scenarios)))
        
        for j, scenario in enumerate(scenarios):
            scenario_scores = [(pid, performance_matrix[pid][scenario]) for pid in product_ids]
            scenario_scores.sort(key=lambda x: x[1])  # 按CRPS分數排序
            
            for rank, (pid, _) in enumerate(scenario_scores):
                i = product_ids.index(pid)
                ranking_matrix[i, j] = rank + 1
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 左圖：排名矩陣熱圖
        colors_for_heatmap = ['green' if pid in robust_products else 'lightgray' 
                             for pid in product_ids]
        
        im = ax1.imshow(ranking_matrix, cmap='RdYlGn_r', aspect='auto')
        
        # 設置刻度
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45, ha='right')
        ax1.set_yticks(range(len(product_ids)))
        ax1.set_yticklabels([f"{pid[:15]}..." for pid in product_ids])
        
        # 標記穩健產品
        for i, pid in enumerate(product_ids):
            if pid in robust_products:
                ax1.text(-0.5, i, '★', fontsize=12, color='gold', 
                        ha='center', va='center', fontweight='bold')
        
        # 添加數值標籤
        for i in range(len(product_ids)):
            for j in range(len(scenarios)):
                ax1.text(j, i, f'{int(ranking_matrix[i, j])}', 
                        ha='center', va='center', fontweight='bold')
        
        ax1.set_title('產品排名穩定性分析\n(★ = 穩健產品)', fontweight='bold')
        ax1.set_xlabel('模型情境')
        ax1.set_ylabel('產品 (按平均CRPS排序)')
        
        # 右圖：穩健性指標散點圖
        mean_crps = [robustness_metrics[pid]['mean_crps'] for pid in product_ids]
        cv_crps = [robustness_metrics[pid]['coefficient_of_variation'] for pid in product_ids]
        
        colors = [self.colors['robust'] if pid in robust_products 
                 else self.colors['non_robust'] for pid in product_ids]
        
        ax2.scatter(mean_crps, cv_crps, c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # 標記產品名稱
        for i, pid in enumerate(product_ids):
            if pid in robust_products:
                ax2.annotate(f'{pid[:10]}...', (mean_crps[i], cv_crps[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('平均 CRPS 分數')
        ax2.set_ylabel('CRPS 變異係數')
        ax2.set_title('穩健性指標分析\n(左下角 = 最穩健)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加圖例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=self.colors['robust'], label='穩健產品'),
                          Patch(facecolor=self.colors['non_robust'], label='非穩健產品')]
        ax2.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已保存至: {save_path}")
        
        return fig
    
    def plot_sensitivity_analysis(self,
                                sensitivity_results: Dict[str, Any],
                                select_products: List[str] = None,
                                save_path: str = None,
                                figsize: Tuple[int, int] = None) -> plt.Figure:
        """
        敏感度分析圖表
        
        展示產品表現對模型參數變化的敏感度。
        
        Parameters:
        -----------
        sensitivity_results : Dict[str, Any]
            敏感度分析結果
        select_products : List[str], optional
            選擇特定產品顯示
        save_path : str, optional
            Save path
        figsize : Tuple[int, int], optional
            Figure size
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        
        figsize = figsize or (15, 8)
        n_params = len(sensitivity_results)
        
        fig, axes = plt.subplots(1, n_params, figsize=figsize)
        if n_params == 1:
            axes = [axes]
        
        for i, (param_name, param_data) in enumerate(sensitivity_results.items()):
            ax = axes[i]
            
            # 獲取參數值和產品列表
            param_values = list(param_data.keys())
            all_products = set()
            for pdata in param_data.values():
                all_products.update(pdata.keys())
            
            # 選擇要顯示的產品
            if select_products:
                products_to_plot = [p for p in select_products if p in all_products]
            else:
                # 選擇前5個表現最佳的產品
                avg_scores = {}
                for product in all_products:
                    scores = []
                    for pval in param_values:
                        if product in param_data[pval]:
                            scores.append(param_data[pval][product])
                    if scores:
                        avg_scores[product] = np.mean(scores)
                
                products_to_plot = sorted(avg_scores.keys(), key=lambda x: avg_scores[x])[:5]
            
            # 繪製敏感度曲線
            for product in products_to_plot:
                param_vals_numeric = []
                scores = []
                
                for pval in param_values:
                    if product in param_data[pval]:
                        try:
                            param_vals_numeric.append(float(pval))
                            scores.append(param_data[pval][product])
                        except ValueError:
                            continue
                
                if len(param_vals_numeric) > 1:
                    ax.plot(param_vals_numeric, scores, marker='o', linewidth=2, 
                           label=f'{product[:15]}...', alpha=0.8)
            
            ax.set_xlabel(param_name.replace('_', ' ').title())
            ax.set_ylabel('CRPS 分數')
            ax.set_title(f'{param_name.replace("_", " ").title()} 敏感度分析')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已保存至: {save_path}")
        
        return fig
    
    def plot_loss_distribution_comparison(self,
                                        loss_distributions: Dict[str, Dict[str, ProbabilisticLossDistribution]],
                                        select_events: List[str] = None,
                                        save_path: str = None,
                                        figsize: Tuple[int, int] = None) -> plt.Figure:
        """
        損失分佈比較圖
        
        比較不同情境下的損失後驗預測分佈。
        
        Parameters:
        -----------
        loss_distributions : Dict[str, Dict[str, ProbabilisticLossDistribution]]
            損失分佈數據
        select_events : List[str], optional
            選擇特定事件
        save_path : str, optional
            Save path
        figsize : Tuple[int, int], optional
            Figure size
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        
        figsize = figsize or (15, 10)
        
        # 選擇要顯示的事件
        all_events = list(loss_distributions.keys())
        if select_events:
            events_to_plot = [e for e in select_events if e in all_events]
        else:
            events_to_plot = all_events[:6]  # 最多6個事件
        
        n_events = len(events_to_plot)
        n_cols = min(3, n_events)
        n_rows = (n_events + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_events == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # 獲取所有情境
        all_scenarios = set()
        for event_dists in loss_distributions.values():
            all_scenarios.update(event_dists.keys())
        scenarios = sorted(list(all_scenarios))
        
        for i, event_id in enumerate(events_to_plot):
            ax = axes[i] if i < len(axes) else axes[0]
            event_dists = loss_distributions[event_id]
            
            for scenario_key in scenarios:
                if scenario_key in event_dists:
                    dist = event_dists[scenario_key]
                    
                    # 繪製密度曲線
                    ax.hist(dist.samples / 1e9, bins=30, alpha=0.3, density=True, 
                           label=scenario_key.replace('_', ' ').title())
                    
                    # 添加統計線
                    ax.axvline(dist.mean / 1e9, linestyle='--', alpha=0.8)
            
            ax.set_title(f'事件 {event_id}', fontweight='bold')
            ax.set_xlabel('損失 (十億美元)')
            ax.set_ylabel('機率密度')
            ax.grid(True, alpha=0.3)
            
            if i == 0:  # 只在第一個子圖顯示圖例
                ax.legend()
        
        # 隱藏多餘的子圖
        for i in range(n_events, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        fig.suptitle('不同情境下的損失後驗預測分佈比較', fontsize=14, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已保存至: {save_path}")
        
        return fig
    
    def create_comprehensive_dashboard(self,
                                     bayesian_results: Dict[str, BayesianAnalysisResult],
                                     robustness_results: Dict[str, Any],
                                     save_path: str = None) -> plt.Figure:
        """
        創建綜合儀表板
        
        結合所有主要視覺化結果在一個綜合儀表板中。
        
        Parameters:
        -----------
        bayesian_results : Dict[str, BayesianAnalysisResult]
            貝氏分析結果
        robustness_results : Dict[str, Any]
            穩健性分析結果
        save_path : str, optional
            Save path
            
        Returns:
        --------
        plt.Figure
            儀表板圖表
        """
        
        fig = plt.figure(figsize=(20, 15))
        
        # 創建網格布局
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 頂部：CRPS分數比較 (跨兩列)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_crps_comparison(bayesian_results, ax1)
        
        # 2. 頂部右側：穩健產品識別
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_robust_products_summary(robustness_results, ax2)
        
        # 3. 中間左：基差風險分佈（小提琴圖）
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_basis_risk_violin(robustness_results, ax3)
        
        # 4. 中間右：排名穩定性
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_ranking_stability(robustness_results, ax4)
        
        # 5. 底部：敏感度分析摘要
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_sensitivity_summary(robustness_results, ax5)
        
        # 添加總標題
        fig.suptitle('穩健貝氏保險分析 - 綜合儀表板', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"綜合儀表板已保存至: {save_path}")
        
        return fig
    
    # ========== 私有輔助方法 ==========
    
    def _plot_crps_comparison(self, bayesian_results: Dict[str, BayesianAnalysisResult], ax):
        """繪製CRPS分數比較"""
        scenarios = list(bayesian_results.keys())
        all_products = set()
        
        for result in bayesian_results.values():
            all_products.update(result.crps_scores.keys())
        
        # 選擇前10個產品
        avg_scores = {}
        for product in all_products:
            scores = []
            for result in bayesian_results.values():
                if product in result.crps_scores:
                    scores.append(result.crps_scores[product])
            if scores:
                avg_scores[product] = np.mean(scores)
        
        top_products = sorted(avg_scores.keys(), key=lambda x: avg_scores[x])[:10]
        
        # 創建堆疊柱狀圖
        x = np.arange(len(top_products))
        width = 0.8 / len(scenarios)
        
        for i, scenario in enumerate(scenarios):
            scores = [bayesian_results[scenario].crps_scores.get(p, 0) for p in top_products]
            ax.bar(x + i * width, scores, width, label=scenario.replace('_', ' ').title())
        
        ax.set_xlabel('產品')
        ax.set_ylabel('CRPS 分數')
        ax.set_title('各情境下的 CRPS 分數比較')
        ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
        ax.set_xticklabels([p[:10] + '...' for p in top_products], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_robust_products_summary(self, robustness_results: Dict[str, Any], ax):
        """繪製穩健產品摘要"""
        robust_products = robustness_results.get('robust_products', [])
        recommendations = robustness_results.get('recommendations', [])
        
        # 創建文字摘要
        ax.text(0.1, 0.9, f'識別出 {len(robust_products)} 個穩健產品', 
               transform=ax.transAxes, fontsize=14, fontweight='bold')
        
        y_pos = 0.75
        for i, product in enumerate(robust_products[:5]):  # 顯示前5個
            ax.text(0.1, y_pos - i * 0.1, f'• {product[:30]}...', 
                   transform=ax.transAxes, fontsize=10)
        
        # 添加建議
        if recommendations:
            ax.text(0.1, 0.4, '主要建議:', transform=ax.transAxes, 
                   fontsize=12, fontweight='bold')
            
            for i, rec in enumerate(recommendations[:3]):
                ax.text(0.1, 0.3 - i * 0.08, f'• {rec}', 
                       transform=ax.transAxes, fontsize=9)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('穩健產品識別結果')
    
    def _plot_basis_risk_violin(self, robustness_results: Dict[str, Any], ax):
        """繪製基差風險小提琴圖"""
        # 這裡需要從 robustness_results 中提取基差風險數據
        # 簡化實現
        ax.text(0.5, 0.5, '基差風險分佈\n(小提琴圖)', 
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('基差風險機率分佈')
    
    def _plot_ranking_stability(self, robustness_results: Dict[str, Any], ax):
        """繪製排名穩定性"""
        robustness_metrics = robustness_results.get('robustness_metrics', {})
        
        if robustness_metrics:
            products = list(robustness_metrics.keys())[:10]
            cv_values = [robustness_metrics[p]['coefficient_of_variation'] for p in products]
            
            ax.barh(range(len(products)), cv_values)
            ax.set_yticks(range(len(products)))
            ax.set_yticklabels([p[:15] + '...' for p in products])
            ax.set_xlabel('變異係數')
            ax.set_title('產品排名穩定性')
            ax.grid(True, alpha=0.3)
        
    def _plot_sensitivity_summary(self, robustness_results: Dict[str, Any], ax):
        """繪製敏感度分析摘要"""
        sensitivity_data = robustness_results.get('sensitivity_analysis', {})
        
        if sensitivity_data:
            # 簡化的敏感度摘要
            ax.text(0.5, 0.5, f'敏感度分析結果\n分析了 {len(sensitivity_data)} 個參數的敏感度', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, '敏感度分析\n(數據準備中)', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
        
        ax.set_title('敏感度分析摘要')
        ax.axis('off')