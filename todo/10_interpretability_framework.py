#!/usr/bin/env python3
"""
可詮釋性框架 (Interpretability Framework)
為參數保險優化結果提供全面的可詮釋性分析

本模組回應問題4：框架的「可詮釋性 (Interpretability)」改進
- 解釋為什麼某個產品被選為「最佳」
- 分析預測變數的重要性和貢獻
- 視覺化基差風險與產品參數的關係曲面
- 提供模型決策的透明化解釋

核心功能：
1. 變數重要性分析 (Feature Importance Analysis)
2. 關係曲面視覺化 (Response Surface Visualization)
3. 決策路徑追蹤 (Decision Path Tracking)
4. 敏感性梯度分析 (Sensitivity Gradient Analysis)
5. 反事實分析 (Counterfactual Analysis)

Author: Research Team
Date: 2025-01-10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import warnings
from pathlib import Path
from scipy import stats
from scipy.interpolate import griddata
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import shap

# 導入基礎模組
from skill_scores.basis_risk_functions import (
    BasisRiskCalculator, 
    BasisRiskType, 
    create_basis_risk_function
)

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

@dataclass
class ProductCandidate:
    """產品候選者資料結構"""
    trigger_threshold: float
    payout_amount: float
    max_payout: float
    basis_risk: float
    trigger_rate: float
    expected_payout: float
    market_acceptability: float
    technical_premium: float
    product_id: str = ""

@dataclass
class InterpretabilityConfig:
    """可詮釋性分析配置"""
    surface_resolution: int = 50          # 關係曲面解析度
    importance_method: str = "permutation" # 重要性計算方法
    shap_samples: int = 100               # SHAP 樣本數
    gradient_epsilon: float = 0.01        # 梯度計算的擾動量
    counterfactual_samples: int = 20      # 反事實分析樣本數
    output_dir: str = "results/interpretability"

class VariableImportanceAnalyzer:
    """變數重要性分析器"""
    
    def __init__(self, config: InterpretabilityConfig):
        """
        初始化變數重要性分析器
        
        Parameters:
        -----------
        config : InterpretabilityConfig
            配置參數
        """
        self.config = config
        self.basis_risk_calc = BasisRiskCalculator()
    
    def calculate_permutation_importance(self,
                                       products: List[ProductCandidate],
                                       target_metric: str = "basis_risk") -> Dict[str, float]:
        """
        計算排列重要性 (Permutation Importance)
        
        通過隨機打亂每個特徵來測量其對目標指標的影響
        
        Parameters:
        -----------
        products : List[ProductCandidate]
            產品候選清單
        target_metric : str
            目標指標名稱
            
        Returns:
        --------
        Dict[str, float]
            各特徵的重要性分數
        """
        
        print("🔍 計算排列重要性...")
        
        if not products:
            return {}
        
        # 準備特徵矩陣和目標向量
        feature_names = ['trigger_threshold', 'payout_amount', 'max_payout', 
                        'trigger_rate', 'expected_payout', 'market_acceptability']
        
        X = np.array([[
            getattr(p, feat) for feat in feature_names
        ] for p in products])
        
        y = np.array([getattr(p, target_metric) for p in products])
        
        # 建立基準模型 (隨機森林)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        baseline_score = rf.score(X, y)
        
        importance_scores = {}
        
        # 對每個特徵進行排列測試
        for i, feature_name in enumerate(feature_names):
            X_permuted = X.copy()
            np.random.seed(42)
            np.random.shuffle(X_permuted[:, i])  # 打亂第i個特徵
            
            rf_permuted = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_permuted.fit(X_permuted, y)
            permuted_score = rf_permuted.score(X_permuted, y)
            
            # 重要性 = 基準分數 - 打亂後分數
            importance_scores[feature_name] = baseline_score - permuted_score
        
        print(f"✅ 排列重要性計算完成")
        return importance_scores
    
    def calculate_gradient_importance(self,
                                    objective_function: Callable,
                                    optimal_product: ProductCandidate,
                                    feature_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        計算梯度重要性 (基於數值微分)
        
        Parameters:
        -----------
        objective_function : Callable
            目標函數 (產品參數 -> 基差風險)
        optimal_product : ProductCandidate
            最佳產品
        feature_ranges : Dict[str, Tuple[float, float]]
            各特徵的取值範圍
            
        Returns:
        --------
        Dict[str, float]
            各特徵的梯度重要性
        """
        
        print("📈 計算梯度重要性...")
        
        epsilon = self.config.gradient_epsilon
        base_params = {
            'trigger_threshold': optimal_product.trigger_threshold,
            'payout_amount': optimal_product.payout_amount,
            'max_payout': optimal_product.max_payout
        }
        
        base_value = objective_function(**base_params)
        gradients = {}
        
        for param_name, base_val in base_params.items():
            if param_name in feature_ranges:
                param_range = feature_ranges[param_name][1] - feature_ranges[param_name][0]
                step = epsilon * param_range
                
                # 計算數值梯度
                params_plus = base_params.copy()
                params_plus[param_name] = base_val + step
                
                params_minus = base_params.copy()
                params_minus[param_name] = base_val - step
                
                try:
                    value_plus = objective_function(**params_plus)
                    value_minus = objective_function(**params_minus)
                    gradient = (value_plus - value_minus) / (2 * step)
                    gradients[param_name] = abs(gradient)  # 取絕對值作為重要性
                except:
                    gradients[param_name] = 0.0
        
        print(f"✅ 梯度重要性計算完成")
        return gradients
    
    def analyze_correlation_importance(self,
                                     products: List[ProductCandidate],
                                     target_metric: str = "basis_risk") -> Dict[str, float]:
        """
        基於相關性分析計算特徵重要性
        
        Parameters:
        -----------
        products : List[ProductCandidate]
            產品候選清單
        target_metric : str
            目標指標
            
        Returns:
        --------
        Dict[str, float]
            各特徵與目標的相關性
        """
        
        print("🔗 分析相關性重要性...")
        
        if not products:
            return {}
        
        feature_names = ['trigger_threshold', 'payout_amount', 'max_payout', 
                        'trigger_rate', 'expected_payout', 'market_acceptability']
        
        correlations = {}
        target_values = [getattr(p, target_metric) for p in products]
        
        for feature_name in feature_names:
            try:
                feature_values = [getattr(p, feature_name) for p in products]
                correlation = abs(np.corrcoef(feature_values, target_values)[0, 1])
                correlations[feature_name] = correlation if not np.isnan(correlation) else 0.0
            except:
                correlations[feature_name] = 0.0
        
        print(f"✅ 相關性分析完成")
        return correlations

class ResponseSurfaceVisualizer:
    """關係曲面視覺化器"""
    
    def __init__(self, config: InterpretabilityConfig):
        """
        初始化關係曲面視覺化器
        
        Parameters:
        -----------
        config : InterpretabilityConfig
            配置參數
        """
        self.config = config
        self.basis_risk_calc = BasisRiskCalculator()
    
    def create_2d_response_surface(self,
                                 objective_function: Callable,
                                 feature1_name: str,
                                 feature1_range: Tuple[float, float],
                                 feature2_name: str, 
                                 feature2_range: Tuple[float, float],
                                 fixed_params: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        創建2D響應曲面
        
        Parameters:
        -----------
        objective_function : Callable
            目標函數
        feature1_name, feature2_name : str
            兩個變化的特徵名稱
        feature1_range, feature2_range : Tuple[float, float]
            特徵取值範圍
        fixed_params : Dict[str, float]
            固定的其他參數
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            X座標網格, Y座標網格, Z值網格
        """
        
        resolution = self.config.surface_resolution
        
        # 創建網格
        x = np.linspace(feature1_range[0], feature1_range[1], resolution)
        y = np.linspace(feature2_range[0], feature2_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # 計算每個網格點的目標函數值
        Z = np.zeros_like(X)
        
        base_params = fixed_params or {}
        
        for i in range(resolution):
            for j in range(resolution):
                params = base_params.copy()
                params[feature1_name] = X[i, j]
                params[feature2_name] = Y[i, j]
                
                try:
                    Z[i, j] = objective_function(**params)
                except:
                    Z[i, j] = np.nan
        
        return X, Y, Z
    
    def visualize_basis_risk_surface(self,
                                   actual_losses: np.ndarray,
                                   hazard_indices: np.ndarray,
                                   optimal_product: ProductCandidate) -> None:
        """
        視覺化基差風險曲面
        
        Parameters:
        -----------
        actual_losses : np.ndarray
            實際損失數據
        hazard_indices : np.ndarray
            災害指標數據
        optimal_product : ProductCandidate
            最佳產品 (用作固定參數的參考)
        """
        
        print("🎨 生成基差風險響應曲面...")
        
        # 定義目標函數
        def basis_risk_function(trigger_threshold: float, payout_amount: float, **kwargs) -> float:
            # 計算賠付
            payouts = np.where(hazard_indices >= trigger_threshold, payout_amount, 0)
            
            # 計算基差風險
            risks = []
            for loss, payout in zip(actual_losses, payouts):
                risk = self.basis_risk_calc.calculate_weighted_asymmetric_basis_risk(
                    loss, payout, w_under=2.0, w_over=0.5
                )
                risks.append(risk)
            
            return np.mean(risks)
        
        # 定義特徵範圍
        trigger_range = (
            np.percentile(hazard_indices, 60),
            np.percentile(hazard_indices, 95)
        )
        
        payout_range = (
            optimal_product.payout_amount * 0.5,
            optimal_product.payout_amount * 2.0
        )
        
        # 創建響應曲面
        X, Y, Z = self.create_2d_response_surface(
            basis_risk_function,
            'trigger_threshold', trigger_range,
            'payout_amount', payout_range
        )
        
        # 創建視覺化
        fig = plt.figure(figsize=(20, 6))
        
        # 3D 曲面圖
        ax1 = fig.add_subplot(131, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        # 標記最佳點
        ax1.scatter([optimal_product.trigger_threshold], 
                   [optimal_product.payout_amount], 
                   [optimal_product.basis_risk],
                   color='red', s=100, label='最佳產品')
        
        ax1.set_xlabel('觸發閾值')
        ax1.set_ylabel('賠付金額')
        ax1.set_zlabel('基差風險')
        ax1.set_title('基差風險 3D 響應曲面')
        fig.colorbar(surf, ax=ax1, shrink=0.5)
        
        # 2D 等高線圖
        ax2 = fig.add_subplot(132)
        contour = ax2.contour(X, Y, Z, levels=20)
        ax2.clabel(contour, inline=True, fontsize=8)
        filled_contour = ax2.contourf(X, Y, Z, levels=50, alpha=0.6, cmap='viridis')
        
        # 標記最佳點
        ax2.scatter([optimal_product.trigger_threshold], 
                   [optimal_product.payout_amount],
                   color='red', s=100, marker='*', label='最佳產品')
        
        ax2.set_xlabel('觸發閾值')
        ax2.set_ylabel('賠付金額')
        ax2.set_title('基差風險等高線圖')
        ax2.legend()
        fig.colorbar(filled_contour, ax=ax2)
        
        # 切片分析
        ax3 = fig.add_subplot(133)
        
        # 固定賠付金額，變化觸發閾值
        trigger_slice = np.linspace(trigger_range[0], trigger_range[1], 100)
        risk_slice = []
        
        for trigger in trigger_slice:
            risk = basis_risk_function(trigger, optimal_product.payout_amount)
            risk_slice.append(risk)
        
        ax3.plot(trigger_slice, risk_slice, 'b-', linewidth=2, label='固定賠付金額')
        ax3.axvline(optimal_product.trigger_threshold, color='red', linestyle='--', 
                   label=f'最佳觸發閾值={optimal_product.trigger_threshold:.2f}')
        ax3.axhline(optimal_product.basis_risk, color='red', linestyle='--', alpha=0.5)
        
        ax3.set_xlabel('觸發閾值')
        ax3.set_ylabel('基差風險')
        ax3.set_title('基差風險切片分析')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = Path(self.config.output_dir) / "basis_risk_response_surface.png"
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 響應曲面已保存: {output_file}")
    
    def create_partial_dependence_plots(self,
                                      products: List[ProductCandidate],
                                      target_metric: str = "basis_risk") -> None:
        """
        創建部分依賴圖 (Partial Dependence Plots)
        
        Parameters:
        -----------
        products : List[ProductCandidate]
            產品候選清單
        target_metric : str
            目標指標
        """
        
        print("📊 生成部分依賴圖...")
        
        if not products:
            return
        
        # 準備數據
        feature_names = ['trigger_threshold', 'payout_amount', 'max_payout', 
                        'trigger_rate', 'expected_payout', 'market_acceptability']
        
        X = np.array([[getattr(p, feat) for feat in feature_names] for p in products])
        y = np.array([getattr(p, target_metric) for p in products])
        
        # 訓練模型
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # 創建部分依賴圖
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'部分依賴分析 - {target_metric}', fontsize=16, fontweight='bold')
        
        axes = axes.ravel()
        
        for i, feature_name in enumerate(feature_names):
            display = PartialDependenceDisplay.from_estimator(
                rf, X, [i], feature_names=[feature_name], ax=axes[i]
            )
            axes[i].set_title(f'{feature_name} 的部分依賴')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = Path(self.config.output_dir) / "partial_dependence_plots.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 部分依賴圖已保存: {output_file}")

class DecisionPathTracker:
    """決策路徑追蹤器"""
    
    def __init__(self, config: InterpretabilityConfig):
        """
        初始化決策路徑追蹤器
        
        Parameters:
        -----------
        config : InterpretabilityConfig
            配置參數
        """
        self.config = config
    
    def trace_optimization_path(self,
                              optimization_history: List[Dict[str, Any]],
                              final_optimal: ProductCandidate) -> Dict[str, Any]:
        """
        追蹤優化路徑
        
        Parameters:
        -----------
        optimization_history : List[Dict[str, Any]]
            優化過程的歷史記錄
        final_optimal : ProductCandidate
            最終最佳產品
            
        Returns:
        --------
        Dict[str, Any]
            決策路徑分析結果
        """
        
        print("🛤️ 追蹤決策路徑...")
        
        if not optimization_history:
            return {"error": "No optimization history provided"}
        
        path_analysis = {
            'total_iterations': len(optimization_history),
            'convergence_pattern': [],
            'parameter_evolution': {
                'trigger_threshold': [],
                'payout_amount': [],
                'basis_risk': []
            },
            'improvement_steps': [],
            'plateau_detection': []
        }
        
        # 分析參數演化
        for i, step in enumerate(optimization_history):
            path_analysis['parameter_evolution']['trigger_threshold'].append(
                step.get('trigger_threshold', 0)
            )
            path_analysis['parameter_evolution']['payout_amount'].append(
                step.get('payout_amount', 0)
            )
            path_analysis['parameter_evolution']['basis_risk'].append(
                step.get('basis_risk', 0)
            )
            
            # 檢測改進步驟
            if i > 0:
                prev_risk = optimization_history[i-1].get('basis_risk', float('inf'))
                curr_risk = step.get('basis_risk', float('inf'))
                if curr_risk < prev_risk:
                    improvement = (prev_risk - curr_risk) / prev_risk
                    path_analysis['improvement_steps'].append({
                        'iteration': i,
                        'improvement_rate': improvement,
                        'from_risk': prev_risk,
                        'to_risk': curr_risk
                    })
        
        # 檢測收斂平台
        risks = path_analysis['parameter_evolution']['basis_risk']
        if len(risks) > 10:
            # 滑動窗口檢測平台
            window_size = min(10, len(risks) // 4)
            for i in range(window_size, len(risks)):
                window = risks[i-window_size:i]
                if len(set(window)) == 1 or (max(window) - min(window)) / np.mean(window) < 0.001:
                    path_analysis['plateau_detection'].append({
                        'start_iteration': i - window_size,
                        'end_iteration': i,
                        'plateau_value': np.mean(window)
                    })
        
        print(f"✅ 決策路徑追蹤完成")
        return path_analysis
    
    def explain_optimal_selection(self,
                                optimal_product: ProductCandidate,
                                all_candidates: List[ProductCandidate],
                                selection_criteria: Dict[str, float]) -> Dict[str, Any]:
        """
        解釋最佳產品選擇的原因
        
        Parameters:
        -----------
        optimal_product : ProductCandidate
            最佳產品
        all_candidates : List[ProductCandidate]
            所有候選產品
        selection_criteria : Dict[str, float]
            選擇標準權重
            
        Returns:
        --------
        Dict[str, Any]
            選擇原因解釋
        """
        
        print("💡 解釋最佳產品選擇...")
        
        explanation = {
            'optimal_product_id': optimal_product.product_id,
            'selection_reasons': [],
            'comparative_analysis': {},
            'decisive_factors': {},
            'trade_offs': {}
        }
        
        # 計算最佳產品在各指標上的排名
        metrics = ['basis_risk', 'trigger_rate', 'expected_payout', 'market_acceptability', 'technical_premium']
        rankings = {}
        
        for metric in metrics:
            if hasattr(optimal_product, metric):
                optimal_value = getattr(optimal_product, metric)
                all_values = [getattr(p, metric, 0) for p in all_candidates]
                
                # 對於風險類指標，排名越小越好
                if metric in ['basis_risk']:
                    rank = sum(1 for v in all_values if v < optimal_value) + 1
                else:
                    rank = sum(1 for v in all_values if v > optimal_value) + 1
                
                rankings[metric] = {
                    'rank': rank,
                    'total_candidates': len(all_candidates),
                    'percentile': (1 - rank / len(all_candidates)) * 100,
                    'value': optimal_value
                }
        
        # 生成選擇原因
        for metric, ranking_info in rankings.items():
            if ranking_info['percentile'] >= 80:  # 前20%
                explanation['selection_reasons'].append(
                    f"{metric} 表現優異 (第{ranking_info['rank']}名, 前{100-ranking_info['percentile']:.1f}%)"
                )
            elif ranking_info['percentile'] >= 50:  # 前50%
                explanation['selection_reasons'].append(
                    f"{metric} 表現良好 (第{ranking_info['rank']}名, 前{100-ranking_info['percentile']:.1f}%)"
                )
        
        # 決定性因素分析
        if 'basis_risk' in rankings:
            explanation['decisive_factors']['primary'] = {
                'factor': 'basis_risk',
                'rank': rankings['basis_risk']['rank'],
                'reasoning': "基差風險是主要優化目標，此產品在該指標上表現最佳"
            }
        
        # 權衡分析
        for metric1 in metrics[:3]:  # 只分析前3個指標的權衡
            for metric2 in metrics[:3]:
                if metric1 != metric2 and metric1 in rankings and metric2 in rankings:
                    rank1 = rankings[metric1]['rank']
                    rank2 = rankings[metric2]['rank']
                    
                    if abs(rank1 - rank2) > len(all_candidates) * 0.2:  # 排名差異大於20%
                        if rank1 < rank2:
                            explanation['trade_offs'][f"{metric1}_vs_{metric2}"] = {
                                'description': f"在{metric1}上表現更好，{metric2}上略有妥協",
                                'rank_difference': rank2 - rank1
                            }
        
        print(f"✅ 最佳產品選擇解釋完成")
        return explanation

class CounterfactualAnalyzer:
    """反事實分析器"""
    
    def __init__(self, config: InterpretabilityConfig):
        """
        初始化反事實分析器
        
        Parameters:
        -----------
        config : InterpretabilityConfig
            配置參數
        """
        self.config = config
        self.basis_risk_calc = BasisRiskCalculator()
    
    def generate_counterfactual_scenarios(self,
                                        optimal_product: ProductCandidate,
                                        actual_losses: np.ndarray,
                                        hazard_indices: np.ndarray) -> Dict[str, Any]:
        """
        生成反事實情境分析
        
        "如果改變某個參數，結果會如何？"
        
        Parameters:
        -----------
        optimal_product : ProductCandidate
            最佳產品
        actual_losses : np.ndarray
            實際損失數據
        hazard_indices : np.ndarray
            災害指標數據
            
        Returns:
        --------
        Dict[str, Any]
            反事實分析結果
        """
        
        print("🔮 生成反事實情境分析...")
        
        base_params = {
            'trigger_threshold': optimal_product.trigger_threshold,
            'payout_amount': optimal_product.payout_amount,
            'max_payout': optimal_product.max_payout
        }
        
        base_risk = optimal_product.basis_risk
        
        counterfactuals = {
            'base_scenario': {
                'parameters': base_params.copy(),
                'basis_risk': base_risk,
                'description': "當前最佳產品"
            },
            'what_if_scenarios': []
        }
        
        # 情境1: 如果提高觸發閾值
        cf_params1 = base_params.copy()
        cf_params1['trigger_threshold'] *= 1.1  # 提高10%
        cf_risk1 = self._calculate_basis_risk(cf_params1, actual_losses, hazard_indices)
        
        counterfactuals['what_if_scenarios'].append({
            'scenario': '提高觸發閾值10%',
            'parameters': cf_params1,
            'basis_risk': cf_risk1,
            'risk_change': cf_risk1 - base_risk,
            'relative_change': (cf_risk1 - base_risk) / base_risk,
            'interpretation': "更保守的觸發條件" + (
                "，基差風險增加" if cf_risk1 > base_risk else "，基差風險減少"
            )
        })
        
        # 情境2: 如果降低觸發閾值
        cf_params2 = base_params.copy()
        cf_params2['trigger_threshold'] *= 0.9  # 降低10%
        cf_risk2 = self._calculate_basis_risk(cf_params2, actual_losses, hazard_indices)
        
        counterfactuals['what_if_scenarios'].append({
            'scenario': '降低觸發閾值10%',
            'parameters': cf_params2,
            'basis_risk': cf_risk2,
            'risk_change': cf_risk2 - base_risk,
            'relative_change': (cf_risk2 - base_risk) / base_risk,
            'interpretation': "更寬鬆的觸發條件" + (
                "，基差風險增加" if cf_risk2 > base_risk else "，基差風險減少"
            )
        })
        
        # 情境3: 如果增加賠付金額
        cf_params3 = base_params.copy()
        cf_params3['payout_amount'] *= 1.2  # 增加20%
        cf_risk3 = self._calculate_basis_risk(cf_params3, actual_losses, hazard_indices)
        
        counterfactuals['what_if_scenarios'].append({
            'scenario': '增加賠付金額20%',
            'parameters': cf_params3,
            'basis_risk': cf_risk3,
            'risk_change': cf_risk3 - base_risk,
            'relative_change': (cf_risk3 - base_risk) / base_risk,
            'interpretation': "更高的賠付水平" + (
                "，基差風險增加" if cf_risk3 > base_risk else "，基差風險減少"
            )
        })
        
        # 情境4: 如果減少賠付金額
        cf_params4 = base_params.copy()
        cf_params4['payout_amount'] *= 0.8  # 減少20%
        cf_risk4 = self._calculate_basis_risk(cf_params4, actual_losses, hazard_indices)
        
        counterfactuals['what_if_scenarios'].append({
            'scenario': '減少賠付金額20%',
            'parameters': cf_params4,
            'basis_risk': cf_risk4,
            'risk_change': cf_risk4 - base_risk,
            'relative_change': (cf_risk4 - base_risk) / base_risk,
            'interpretation': "更低的賠付水平" + (
                "，基差風險增加" if cf_risk4 > base_risk else "，基差風險減少"
            )
        })
        
        # 找出最佳的反事實情境
        best_cf = min(counterfactuals['what_if_scenarios'], 
                     key=lambda x: x['basis_risk'])
        
        counterfactuals['best_alternative'] = {
            'scenario': best_cf['scenario'],
            'improvement_potential': base_risk - best_cf['basis_risk'],
            'parameters': best_cf['parameters'],
            'recommendation': f"考慮{best_cf['scenario']}可能進一步改善基差風險"
        }
        
        print(f"✅ 反事實分析完成")
        return counterfactuals
    
    def _calculate_basis_risk(self,
                            params: Dict[str, float],
                            actual_losses: np.ndarray,
                            hazard_indices: np.ndarray) -> float:
        """計算給定參數下的基差風險"""
        
        payouts = np.where(hazard_indices >= params['trigger_threshold'],
                          params['payout_amount'], 0)
        
        risks = []
        for loss, payout in zip(actual_losses, payouts):
            risk = self.basis_risk_calc.calculate_weighted_asymmetric_basis_risk(
                loss, payout, w_under=2.0, w_over=0.5
            )
            risks.append(risk)
        
        return np.mean(risks)

class InterpretabilityFramework:
    """整合可詮釋性框架"""
    
    def __init__(self, config: InterpretabilityConfig = None):
        """
        初始化可詮釋性框架
        
        Parameters:
        -----------
        config : InterpretabilityConfig
            配置參數
        """
        self.config = config or InterpretabilityConfig()
        
        # 初始化子模組
        self.importance_analyzer = VariableImportanceAnalyzer(self.config)
        self.surface_visualizer = ResponseSurfaceVisualizer(self.config)
        self.decision_tracker = DecisionPathTracker(self.config)
        self.counterfactual_analyzer = CounterfactualAnalyzer(self.config)
        
        # 創建輸出目錄
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def run_comprehensive_interpretability_analysis(self,
                                                  optimal_product: ProductCandidate,
                                                  all_candidates: List[ProductCandidate],
                                                  actual_losses: np.ndarray,
                                                  hazard_indices: np.ndarray,
                                                  optimization_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        執行完整的可詮釋性分析
        
        Parameters:
        -----------
        optimal_product : ProductCandidate
            最佳產品
        all_candidates : List[ProductCandidate]
            所有候選產品
        actual_losses : np.ndarray
            實際損失數據
        hazard_indices : np.ndarray
            災害指標數據
        optimization_history : List[Dict[str, Any]]
            優化歷史記錄
            
        Returns:
        --------
        Dict[str, Any]
            完整的可詮釋性分析結果
        """
        
        print("🔍 執行完整可詮釋性分析...")
        print("=" * 60)
        
        comprehensive_results = {
            'analysis_metadata': {
                'optimal_product_id': optimal_product.product_id,
                'total_candidates': len(all_candidates),
                'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # 1. 變數重要性分析
        print("📊 1. 變數重要性分析...")
        importance_results = {}
        
        # 排列重要性
        importance_results['permutation_importance'] = \
            self.importance_analyzer.calculate_permutation_importance(all_candidates)
        
        # 相關性重要性
        importance_results['correlation_importance'] = \
            self.importance_analyzer.analyze_correlation_importance(all_candidates)
        
        # 梯度重要性 (如果可能)
        feature_ranges = {
            'trigger_threshold': (
                min(p.trigger_threshold for p in all_candidates),
                max(p.trigger_threshold for p in all_candidates)
            ),
            'payout_amount': (
                min(p.payout_amount for p in all_candidates),
                max(p.payout_amount for p in all_candidates)
            ),
            'max_payout': (
                min(p.max_payout for p in all_candidates),
                max(p.max_payout for p in all_candidates)
            )
        }
        
        def objective_func(**params):
            return self.counterfactual_analyzer._calculate_basis_risk(
                params, actual_losses, hazard_indices
            )
        
        importance_results['gradient_importance'] = \
            self.importance_analyzer.calculate_gradient_importance(
                objective_func, optimal_product, feature_ranges
            )
        
        comprehensive_results['variable_importance'] = importance_results
        
        # 2. 響應曲面視覺化
        print("📈 2. 響應曲面視覺化...")
        self.surface_visualizer.visualize_basis_risk_surface(
            actual_losses, hazard_indices, optimal_product
        )
        
        self.surface_visualizer.create_partial_dependence_plots(all_candidates)
        
        # 3. 決策路徑追蹤
        print("🛤️ 3. 決策路徑分析...")
        if optimization_history:
            path_analysis = self.decision_tracker.trace_optimization_path(
                optimization_history, optimal_product
            )
            comprehensive_results['decision_path'] = path_analysis
        
        # 最佳產品選擇解釋
        selection_explanation = self.decision_tracker.explain_optimal_selection(
            optimal_product, all_candidates, {'basis_risk': 1.0}
        )
        comprehensive_results['selection_explanation'] = selection_explanation
        
        # 4. 反事實分析
        print("🔮 4. 反事實情境分析...")
        counterfactual_results = self.counterfactual_analyzer.generate_counterfactual_scenarios(
            optimal_product, actual_losses, hazard_indices
        )
        comprehensive_results['counterfactual_analysis'] = counterfactual_results
        
        # 5. 生成綜合解釋報告
        comprehensive_results['interpretability_summary'] = self._generate_interpretability_summary(
            comprehensive_results, optimal_product
        )
        
        # 保存結果
        self._save_interpretability_results(comprehensive_results)
        
        print("✅ 完整可詮釋性分析完成！")
        return comprehensive_results
    
    def _generate_interpretability_summary(self,
                                         results: Dict[str, Any],
                                         optimal_product: ProductCandidate) -> Dict[str, Any]:
        """生成可詮釋性總結"""
        
        summary = {
            'key_insights': [],
            'most_important_variables': [],
            'decision_rationale': [],
            'improvement_suggestions': []
        }
        
        # 分析最重要的變數
        importance_data = results.get('variable_importance', {})
        
        if 'permutation_importance' in importance_data:
            perm_imp = importance_data['permutation_importance']
            most_important = max(perm_imp.items(), key=lambda x: x[1])
            summary['most_important_variables'].append({
                'method': 'permutation',
                'variable': most_important[0],
                'importance': most_important[1],
                'interpretation': f"{most_important[0]} 對模型預測影響最大"
            })
        
        # 決策理由
        selection_exp = results.get('selection_explanation', {})
        if 'selection_reasons' in selection_exp:
            summary['decision_rationale'] = selection_exp['selection_reasons']
        
        # 改進建議
        cf_analysis = results.get('counterfactual_analysis', {})
        if 'best_alternative' in cf_analysis:
            best_alt = cf_analysis['best_alternative']
            if best_alt['improvement_potential'] > 0:
                summary['improvement_suggestions'].append({
                    'suggestion': best_alt['scenario'],
                    'potential_improvement': best_alt['improvement_potential'],
                    'recommendation': best_alt['recommendation']
                })
        
        # 關鍵洞察
        summary['key_insights'] = [
            f"最佳產品的基差風險為 {optimal_product.basis_risk:.2e}",
            f"觸發率為 {optimal_product.trigger_rate:.1%}",
            f"市場接受度評分為 {optimal_product.market_acceptability:.3f}"
        ]
        
        if 'most_important_variables' in summary and summary['most_important_variables']:
            top_var = summary['most_important_variables'][0]['variable']
            summary['key_insights'].append(f"{top_var} 是影響產品性能的關鍵因素")
        
        return summary
    
    def _save_interpretability_results(self, results: Dict[str, Any]) -> None:
        """保存可詮釋性結果"""
        
        import json
        
        # 保存完整結果
        output_file = Path(self.config.output_dir) / "interpretability_analysis_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📁 可詮釋性分析結果已保存: {output_file}")

def main():
    """主函數示例"""
    
    print("🔍 可詮釋性框架示例...")
    print("=" * 60)
    
    # 創建示例數據
    np.random.seed(42)
    
    # 生成模擬損失和災害數據
    n_scenarios = 500
    actual_losses = np.random.lognormal(np.log(5e7), 0.8, n_scenarios)
    hazard_indices = np.random.gamma(2, 25, n_scenarios)
    
    # 生成候選產品
    candidates = []
    for i in range(50):
        trigger = np.random.uniform(40, 80)
        payout = np.random.uniform(2e7, 1e8)
        max_payout = payout * np.random.uniform(1.5, 3.0)
        
        # 計算基差風險 (簡化)
        payouts = np.where(hazard_indices >= trigger, payout, 0)
        basis_risk_calc = BasisRiskCalculator()
        
        risks = [
            basis_risk_calc.calculate_weighted_asymmetric_basis_risk(
                loss, pay, w_under=2.0, w_over=0.5
            ) for loss, pay in zip(actual_losses, payouts)
        ]
        
        candidate = ProductCandidate(
            trigger_threshold=trigger,
            payout_amount=payout,
            max_payout=max_payout,
            basis_risk=np.mean(risks),
            trigger_rate=np.mean(payouts > 0),
            expected_payout=np.mean(payouts),
            market_acceptability=np.random.uniform(0.5, 1.0),
            technical_premium=np.mean(payouts) * 1.5,
            product_id=f"Product_{i+1:02d}"
        )
        
        candidates.append(candidate)
    
    # 找出最佳產品
    optimal_product = min(candidates, key=lambda p: p.basis_risk)
    
    print(f"最佳產品: {optimal_product.product_id}")
    print(f"基差風險: {optimal_product.basis_risk:.2e}")
    print(f"觸發閾值: {optimal_product.trigger_threshold:.2f}")
    print(f"賠付金額: {optimal_product.payout_amount:.2e}")
    
    # 執行可詮釋性分析
    config = InterpretabilityConfig()
    framework = InterpretabilityFramework(config)
    
    results = framework.run_comprehensive_interpretability_analysis(
        optimal_product=optimal_product,
        all_candidates=candidates,
        actual_losses=actual_losses,
        hazard_indices=hazard_indices
    )
    
    # 輸出關鍵結果
    print("\n" + "=" * 60)
    print("📋 可詮釋性分析關鍵結果:")
    print("=" * 60)
    
    summary = results.get('interpretability_summary', {})
    
    if 'key_insights' in summary:
        print("💡 關鍵洞察:")
        for insight in summary['key_insights']:
            print(f"  • {insight}")
    
    if 'decision_rationale' in summary:
        print(f"\n🎯 決策理由:")
        for reason in summary['decision_rationale']:
            print(f"  • {reason}")
    
    if 'improvement_suggestions' in summary:
        print(f"\n📈 改進建議:")
        for suggestion in summary['improvement_suggestions']:
            print(f"  • {suggestion['suggestion']}")
            print(f"    潛在改進: {suggestion['potential_improvement']:.2e}")
    
    print(f"\n📁 詳細結果已保存在: {config.output_dir}")
    print("🎉 可詮釋性分析完成！")

if __name__ == "__main__":
    main()