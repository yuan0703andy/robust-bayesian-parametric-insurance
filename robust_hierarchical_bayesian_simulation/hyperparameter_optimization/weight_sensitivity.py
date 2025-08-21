"""
Weight Sensitivity Analyzer
權重敏感性分析器

本模組為 bayesian/ 框架提供權重敏感性分析功能，回應學術考量點一:
「懲罰權重的選擇與合理性 (The Choice of Weights)」

核心功能:
- 系統性測試不同 (w_under, w_over) 權重組合
- 分析權重選擇對最佳產品參數的影響  
- 提供相關性分析和穩健性評估
- 與 RobustBayesianAnalyzer 無縫整合

整合設計:
- 可作為 RobustBayesianAnalyzer 的擴展功能
- 支持現有的 basis_risk_type 和產品優化流程
- 提供獨立的敏感性分析介面
- 結果可與其他 bayesian/ 組件共享

Author: Robust Bayesian Analysis Team
Date: 2025-01-10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import warnings
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor
import time

# 導入 bayesian 模組的相關組件
try:
    from .robust_bayesian_analyzer import RobustBayesianAnalyzer
    HAS_ROBUST_ANALYZER = True
except ImportError:
    HAS_ROBUST_ANALYZER = False
    warnings.warn("RobustBayesianAnalyzer not available for integration")

# 導入 skill_scores 的基差風險函數
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from skill_scores.basis_risk_functions import (
        BasisRiskCalculator, 
        BasisRiskType, 
        create_basis_risk_function
    )
    HAS_BASIS_RISK_FUNCTIONS = True
except ImportError:
    HAS_BASIS_RISK_FUNCTIONS = False
    warnings.warn("skill_scores.basis_risk_functions not available")

# 設定中文字體支持
try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    pass

@dataclass
class WeightSensitivityConfig:
    """權重敏感性分析配置"""
    
    # 權重組合設定
    weight_combinations: List[Tuple[float, float]] = field(default_factory=lambda: [
        (2.0, 0.5),   # 基準組合 (當前使用)
        (1.0, 1.0),   # 相等權重
        (3.0, 1.0),   # 3:1 比率
        (4.0, 1.0),   # 4:1 比率  
        (5.0, 1.0),   # 5:1 比率
        (10.0, 1.0),  # 10:1 比率
        (0.5, 2.0),   # 反向權重 (更關心過度賠付)
        (1.0, 2.0),   # 1:2 比率
        (1.5, 1.0),   # 溫和權重
        (2.0, 1.0),   # 2:1 比率
        (5.0, 0.1),   # 極度懲罰不足覆蓋
        (0.1, 5.0)    # 極度懲罰過度覆蓋
    ])
    
    # 分析設定
    basis_risk_type: str = "weighted_asymmetric"  # 基差風險類型
    product_search_resolution: int = 20           # 產品搜索解析度
    use_parallel_processing: bool = True          # 是否使用並行處理
    n_workers: int = 4                           # 工作進程數
    
    # 輸出設定
    output_dir: str = "results/weight_sensitivity"
    save_detailed_results: bool = True
    generate_plots: bool = True
    plot_dpi: int = 300

@dataclass  
class WeightSensitivityResult:
    """權重敏感性分析結果"""
    
    # 基本資訊
    weight_combination: Tuple[float, float]
    weight_ratio: float
    
    # 最佳產品參數
    optimal_trigger_threshold: float
    optimal_payout_amount: float
    optimal_basis_risk: float
    optimal_trigger_rate: float
    
    # 產品性能指標
    expected_payout: float
    coverage_efficiency: float
    risk_stability_score: float
    
    # 與基準的比較
    improvement_vs_baseline: float
    rank_among_combinations: int

@dataclass
class WeightSensitivityAnalysis:
    """完整的權重敏感性分析結果"""
    
    # 分析元數據
    analysis_config: WeightSensitivityConfig
    analysis_timestamp: str
    data_summary: Dict[str, Any]
    
    # 所有權重組合的結果
    weight_results: List[WeightSensitivityResult]
    
    # 總體分析
    best_weight_combination: Tuple[float, float]
    worst_weight_combination: Tuple[float, float]
    sensitivity_level: str  # "High", "Medium", "Low"
    correlation_analysis: Dict[str, float]
    
    # 業務洞察
    key_insights: List[str]
    business_recommendations: List[str]

class WeightSensitivityAnalyzer:
    """
    權重敏感性分析器
    
    與 RobustBayesianAnalyzer 整合，提供權重選擇的系統性分析
    """
    
    def __init__(self, 
                 config: Optional[WeightSensitivityConfig] = None,
                 robust_analyzer: Optional['RobustBayesianAnalyzer'] = None):
        """
        初始化權重敏感性分析器
        
        Parameters:
        -----------
        config : WeightSensitivityConfig, optional
            敏感性分析配置
        robust_analyzer : RobustBayesianAnalyzer, optional
            已配置的貝葉斯分析器，可重用其設定
        """
        self.config = config or WeightSensitivityConfig()
        self.robust_analyzer = robust_analyzer
        
        # 初始化基差風險計算器
        if HAS_BASIS_RISK_FUNCTIONS:
            self.basis_risk_calc = BasisRiskCalculator()
        else:
            self.basis_risk_calc = None
            warnings.warn("BasisRiskCalculator not available, using simplified calculations")
        
        # 創建輸出目錄
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 結果緩存
        self._analysis_cache = {}
    
    def analyze_weight_sensitivity(self,
                                 observations: np.ndarray,
                                 validation_data: np.ndarray,
                                 hazard_indices: np.ndarray,
                                 actual_losses: np.ndarray,
                                 product_bounds: Dict[str, Tuple[float, float]]) -> WeightSensitivityAnalysis:
        """
        執行完整的權重敏感性分析
        
        Parameters:
        -----------
        observations : np.ndarray
            訓練數據
        validation_data : np.ndarray
            驗證數據
        hazard_indices : np.ndarray
            災害指標
        actual_losses : np.ndarray
            真實損失
        product_bounds : Dict[str, Tuple[float, float]]
            產品參數邊界
            
        Returns:
        --------
        WeightSensitivityAnalysis
            完整的敏感性分析結果
        """
        
        print("🔍 執行權重敏感性分析...")
        print("=" * 60)
        print(f"分析 {len(self.config.weight_combinations)} 個權重組合")
        
        start_time = time.time()
        
        # 準備數據摘要
        data_summary = {
            'n_observations': len(observations),
            'n_validation': len(validation_data),
            'n_hazard_events': len(hazard_indices),
            'n_loss_scenarios': len(actual_losses),
            'hazard_range': (float(hazard_indices.min()), float(hazard_indices.max())),
            'loss_range': (float(actual_losses.min()), float(actual_losses.max()))
        }
        
        # 分析每個權重組合
        if self.config.use_parallel_processing:
            weight_results = self._analyze_weights_parallel(
                observations, validation_data, hazard_indices, actual_losses, product_bounds
            )
        else:
            weight_results = self._analyze_weights_sequential(
                observations, validation_data, hazard_indices, actual_losses, product_bounds
            )
        
        # 執行總體分析
        analysis_results = self._perform_overall_analysis(weight_results, data_summary)
        
        execution_time = time.time() - start_time
        print(f"✅ 權重敏感性分析完成 (執行時間: {execution_time:.2f} 秒)")
        
        return analysis_results
    
    def _analyze_single_weight_combination(self,
                                         w_under: float,
                                         w_over: float,
                                         observations: np.ndarray,
                                         validation_data: np.ndarray,
                                         hazard_indices: np.ndarray,
                                         actual_losses: np.ndarray,
                                         product_bounds: Dict[str, Tuple[float, float]]) -> WeightSensitivityResult:
        """分析單一權重組合"""
        
        try:
            # 如果有可用的 RobustBayesianAnalyzer，使用整合方法
            if self.robust_analyzer and HAS_ROBUST_ANALYZER:
                results = self._analyze_with_robust_analyzer(
                    w_under, w_over, observations, validation_data, 
                    hazard_indices, actual_losses, product_bounds
                )
            else:
                # 使用簡化的獨立分析方法
                results = self._analyze_with_simple_optimization(
                    w_under, w_over, hazard_indices, actual_losses, product_bounds
                )
                
            return results
            
        except Exception as e:
            print(f"⚠️ 權重組合 ({w_under}, {w_over}) 分析失敗: {e}")
            # 返回默認結果
            return self._create_default_result(w_under, w_over)
    
    def _analyze_with_robust_analyzer(self,
                                    w_under: float,
                                    w_over: float,
                                    observations: np.ndarray,
                                    validation_data: np.ndarray,
                                    hazard_indices: np.ndarray,
                                    actual_losses: np.ndarray,
                                    product_bounds: Dict[str, Tuple[float, float]]) -> WeightSensitivityResult:
        """使用 RobustBayesianAnalyzer 進行整合分析"""
        
        # 執行整合貝葉斯優化，使用指定的權重
        bayesian_results = self.robust_analyzer.integrated_bayesian_optimization(
            observations=observations,
            validation_data=validation_data,
            hazard_indices=hazard_indices,
            actual_losses=actual_losses,
            product_bounds=product_bounds,
            w_under=w_under,
            w_over=w_over,
            configure_pymc=False  # 避免重複配置
        )
        
        # 提取最佳產品參數
        optimal_product = bayesian_results['phase_2_decision_optimization']['optimal_product']
        
        # 計算額外的性能指標
        trigger_threshold = optimal_product['trigger_threshold']
        payout_amount = optimal_product['payout_amount']
        basis_risk = bayesian_results['phase_2_decision_optimization']['expected_basis_risk']
        
        # 計算觸發率和期望賠付
        payouts = np.where(hazard_indices >= trigger_threshold, payout_amount, 0)
        trigger_rate = np.mean(payouts > 0)
        expected_payout = np.mean(payouts)
        
        # 計算覆蓋效率
        total_losses = np.sum(actual_losses)
        total_payouts = np.sum(payouts)
        coverage_efficiency = 1.0 - abs(total_losses - total_payouts) / max(total_losses, 1)
        
        return WeightSensitivityResult(
            weight_combination=(w_under, w_over),
            weight_ratio=w_under / w_over if w_over > 0 else float('inf'),
            optimal_trigger_threshold=trigger_threshold,
            optimal_payout_amount=payout_amount,
            optimal_basis_risk=basis_risk,
            optimal_trigger_rate=trigger_rate,
            expected_payout=expected_payout,
            coverage_efficiency=coverage_efficiency,
            risk_stability_score=0.0,  # 將在後續計算
            improvement_vs_baseline=0.0,  # 將在後續計算
            rank_among_combinations=0  # 將在後續計算
        )
    
    def _analyze_with_simple_optimization(self,
                                        w_under: float,
                                        w_over: float,
                                        hazard_indices: np.ndarray,
                                        actual_losses: np.ndarray,
                                        product_bounds: Dict[str, Tuple[float, float]]) -> WeightSensitivityResult:
        """使用簡化的優化方法"""
        
        if not self.basis_risk_calc:
            return self._create_default_result(w_under, w_over)
        
        # 定義搜索空間
        trigger_min, trigger_max = product_bounds.get('trigger_threshold', 
                                                     (np.percentile(hazard_indices, 60), np.percentile(hazard_indices, 95)))
        payout_min, payout_max = product_bounds.get('payout_amount',
                                                   (np.percentile(actual_losses[actual_losses > 0], 20),
                                                    np.percentile(actual_losses[actual_losses > 0], 80)))
        
        trigger_range = np.linspace(trigger_min, trigger_max, self.config.product_search_resolution)
        payout_range = np.linspace(payout_min, payout_max, self.config.product_search_resolution)
        
        best_risk = float('inf')
        best_trigger = trigger_min
        best_payout = payout_min
        
        # 網格搜索
        for trigger in trigger_range:
            for payout in payout_range:
                payouts = np.where(hazard_indices >= trigger, payout, 0)
                
                # 計算基差風險
                risks = []
                for loss, pay in zip(actual_losses, payouts):
                    risk = self.basis_risk_calc.calculate_weighted_asymmetric_basis_risk(
                        loss, pay, w_under=w_under, w_over=w_over
                    )
                    risks.append(risk)
                
                mean_risk = np.mean(risks)
                if mean_risk < best_risk:
                    best_risk = mean_risk
                    best_trigger = trigger
                    best_payout = payout
        
        # 計算最佳產品的性能指標
        best_payouts = np.where(hazard_indices >= best_trigger, best_payout, 0)
        trigger_rate = np.mean(best_payouts > 0)
        expected_payout = np.mean(best_payouts)
        
        total_losses = np.sum(actual_losses)
        total_payouts = np.sum(best_payouts)
        coverage_efficiency = 1.0 - abs(total_losses - total_payouts) / max(total_losses, 1)
        
        return WeightSensitivityResult(
            weight_combination=(w_under, w_over),
            weight_ratio=w_under / w_over if w_over > 0 else float('inf'),
            optimal_trigger_threshold=best_trigger,
            optimal_payout_amount=best_payout,
            optimal_basis_risk=best_risk,
            optimal_trigger_rate=trigger_rate,
            expected_payout=expected_payout,
            coverage_efficiency=coverage_efficiency,
            risk_stability_score=0.0,
            improvement_vs_baseline=0.0,
            rank_among_combinations=0
        )
    
    def _create_default_result(self, w_under: float, w_over: float) -> WeightSensitivityResult:
        """創建默認結果(當分析失敗時)"""
        return WeightSensitivityResult(
            weight_combination=(w_under, w_over),
            weight_ratio=w_under / w_over if w_over > 0 else float('inf'),
            optimal_trigger_threshold=50.0,
            optimal_payout_amount=1e8,
            optimal_basis_risk=1e9,
            optimal_trigger_rate=0.2,
            expected_payout=2e7,
            coverage_efficiency=0.5,
            risk_stability_score=0.0,
            improvement_vs_baseline=0.0,
            rank_among_combinations=999
        )
    
    def _analyze_weights_parallel(self,
                                observations: np.ndarray,
                                validation_data: np.ndarray,
                                hazard_indices: np.ndarray,
                                actual_losses: np.ndarray,
                                product_bounds: Dict[str, Tuple[float, float]]) -> List[WeightSensitivityResult]:
        """並行分析權重組合"""
        
        print("⚡ 使用並行處理分析權重組合...")
        
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = []
            for w_under, w_over in self.config.weight_combinations:
                future = executor.submit(
                    self._analyze_single_weight_combination,
                    w_under, w_over, observations, validation_data,
                    hazard_indices, actual_losses, product_bounds
                )
                futures.append(future)
            
            results = [future.result() for future in futures]
        
        return results
    
    def _analyze_weights_sequential(self,
                                  observations: np.ndarray,
                                  validation_data: np.ndarray,
                                  hazard_indices: np.ndarray,
                                  actual_losses: np.ndarray,
                                  product_bounds: Dict[str, Tuple[float, float]]) -> List[WeightSensitivityResult]:
        """順序分析權重組合"""
        
        print("🔄 使用順序處理分析權重組合...")
        
        results = []
        for i, (w_under, w_over) in enumerate(self.config.weight_combinations):
            print(f"  分析權重組合 {i+1}/{len(self.config.weight_combinations)}: ({w_under:.1f}, {w_over:.1f})")
            
            result = self._analyze_single_weight_combination(
                w_under, w_over, observations, validation_data,
                hazard_indices, actual_losses, product_bounds
            )
            results.append(result)
        
        return results
    
    def _perform_overall_analysis(self,
                                weight_results: List[WeightSensitivityResult],
                                data_summary: Dict[str, Any]) -> WeightSensitivityAnalysis:
        """執行總體分析和洞察生成"""
        
        print("📊 執行總體敏感性分析...")
        
        # 排序和排名
        sorted_results = sorted(weight_results, key=lambda x: x.optimal_basis_risk)
        for i, result in enumerate(sorted_results):
            result.rank_among_combinations = i + 1
        
        # 計算與基準的改進
        baseline_risk = next((r.optimal_basis_risk for r in weight_results 
                            if r.weight_combination == (2.0, 0.5)), 
                           weight_results[0].optimal_basis_risk)
        
        for result in weight_results:
            result.improvement_vs_baseline = (baseline_risk - result.optimal_basis_risk) / baseline_risk
        
        # 計算相關性
        weight_ratios = [r.weight_ratio for r in weight_results if np.isfinite(r.weight_ratio)]
        basis_risks = [r.optimal_basis_risk for r in weight_results if np.isfinite(r.weight_ratio)]
        
        correlation = np.corrcoef(weight_ratios, basis_risks)[0, 1] if len(weight_ratios) > 1 else 0.0
        
        # 確定敏感性級別
        risk_range = max(basis_risks) - min(basis_risks)
        relative_range = risk_range / np.mean(basis_risks) if np.mean(basis_risks) > 0 else 0
        
        if relative_range > 0.3:
            sensitivity_level = "High"
        elif relative_range > 0.1:
            sensitivity_level = "Medium"  
        else:
            sensitivity_level = "Low"
        
        # 生成洞察和建議
        key_insights = self._generate_key_insights(weight_results, correlation, sensitivity_level)
        business_recommendations = self._generate_business_recommendations(sorted_results, sensitivity_level)
        
        return WeightSensitivityAnalysis(
            analysis_config=self.config,
            analysis_timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            data_summary=data_summary,
            weight_results=weight_results,
            best_weight_combination=sorted_results[0].weight_combination,
            worst_weight_combination=sorted_results[-1].weight_combination,
            sensitivity_level=sensitivity_level,
            correlation_analysis={'weight_ratio_vs_basis_risk': correlation},
            key_insights=key_insights,
            business_recommendations=business_recommendations
        )
    
    def _generate_key_insights(self,
                             weight_results: List[WeightSensitivityResult],
                             correlation: float,
                             sensitivity_level: str) -> List[str]:
        """生成關鍵洞察"""
        
        insights = []
        
        # 敏感性水平洞察
        insights.append(f"權重敏感性等級: {sensitivity_level}")
        
        # 相關性洞察
        if abs(correlation) > 0.7:
            insights.append(f"權重比率與基差風險呈強相關性 (r={correlation:.3f})")
        elif abs(correlation) > 0.3:
            insights.append(f"權重比率與基差風險呈中度相關性 (r={correlation:.3f})")
        else:
            insights.append(f"權重比率與基差風險相關性較低 (r={correlation:.3f})")
        
        # 性能範圍洞察
        risks = [r.optimal_basis_risk for r in weight_results]
        min_risk, max_risk = min(risks), max(risks)
        performance_ratio = max_risk / min_risk if min_risk > 0 else 1.0
        
        insights.append(f"最佳與最差權重組合的性能差異為 {performance_ratio:.2f} 倍")
        
        # 最佳權重洞察
        best_result = min(weight_results, key=lambda x: x.optimal_basis_risk)
        best_ratio = best_result.weight_ratio
        
        if best_ratio > 5:
            insights.append(f"最佳權重比率為 {best_ratio:.1f}:1，偏向懲罰不足覆蓋")
        elif best_ratio < 2:
            insights.append(f"最佳權重比率為 {best_ratio:.1f}:1，相對均衡")
        else:
            insights.append(f"最佳權重比率為 {best_ratio:.1f}:1，適中偏好")
        
        return insights
    
    def _generate_business_recommendations(self,
                                         sorted_results: List[WeightSensitivityResult],
                                         sensitivity_level: str) -> List[str]:
        """生成業務建議"""
        
        recommendations = []
        
        best_result = sorted_results[0]
        
        # 基於敏感性級別的建議
        if sensitivity_level == "High":
            recommendations.append("權重選擇對產品性能有重大影響，建議進行詳細的權重調校分析")
            recommendations.append("建議與業務部門討論風險偏好，確定適當的權重設定")
        elif sensitivity_level == "Medium":
            recommendations.append("權重選擇有中等程度影響，建議進行適度的權重優化")
        else:
            recommendations.append("產品對權重變化相對穩健，可使用標準權重設定")
        
        # 最佳權重建議
        w_under, w_over = best_result.weight_combination
        recommendations.append(f"建議使用權重組合 (w_under={w_under:.1f}, w_over={w_over:.1f})")
        
        # 風險控制建議
        if best_result.optimal_trigger_rate < 0.1:
            recommendations.append("觸發率較低，考慮適當降低觸發閾值以提高覆蓋度")
        elif best_result.optimal_trigger_rate > 0.4:
            recommendations.append("觸發率較高，考慮適當提高觸發閾值以控制賠付頻率")
        
        return recommendations
    
    def visualize_sensitivity_results(self, analysis: WeightSensitivityAnalysis) -> None:
        """視覺化敏感性分析結果"""
        
        if not self.config.generate_plots:
            return
        
        print("📊 生成權重敏感性視覺化...")
        
        # 準備數據
        results = analysis.weight_results
        weight_ratios = [r.weight_ratio for r in results if np.isfinite(r.weight_ratio)]
        basis_risks = [r.optimal_basis_risk for r in results if np.isfinite(r.weight_ratio)]
        trigger_rates = [r.optimal_trigger_rate for r in results if np.isfinite(r.weight_ratio)]
        
        # 創建視覺化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('權重敏感性分析結果', fontsize=16, fontweight='bold')
        
        # 1. 權重比率 vs 基差風險
        ax1 = axes[0, 0]
        scatter = ax1.scatter(weight_ratios, basis_risks, c=trigger_rates, 
                            cmap='viridis', s=100, alpha=0.7)
        ax1.set_xlabel('權重比率 (w_under / w_over)')
        ax1.set_ylabel('最佳基差風險')
        ax1.set_title('權重比率對基差風險的影響')
        ax1.set_xscale('log')
        plt.colorbar(scatter, ax=ax1, label='觸發率')
        ax1.grid(True, alpha=0.3)
        
        # 2. 權重組合排名
        ax2 = axes[0, 1]
        sorted_results = sorted(results, key=lambda x: x.optimal_basis_risk)
        ranks = range(1, len(sorted_results) + 1)
        risks = [r.optimal_basis_risk for r in sorted_results]
        
        bars = ax2.bar(ranks, risks, alpha=0.7)
        ax2.set_xlabel('排名')
        ax2.set_ylabel('基差風險')
        ax2.set_title('權重組合性能排名')
        
        # 標記最佳組合
        best_combo = sorted_results[0].weight_combination
        ax2.text(1, risks[0], f'最佳: ({best_combo[0]:.1f}, {best_combo[1]:.1f})',
                ha='center', va='bottom', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 權重敏感性熱圖
        ax3 = axes[1, 0]
        
        # 創建權重網格數據
        w_under_values = sorted(set(r.weight_combination[0] for r in results))
        w_over_values = sorted(set(r.weight_combination[1] for r in results))
        
        if len(w_under_values) > 1 and len(w_over_values) > 1:
            risk_matrix = np.full((len(w_over_values), len(w_under_values)), np.nan)
            
            for result in results:
                w_u, w_o = result.weight_combination
                if w_u in w_under_values and w_o in w_over_values:
                    i = w_over_values.index(w_o)
                    j = w_under_values.index(w_u)
                    risk_matrix[i, j] = result.optimal_basis_risk
            
            im = ax3.imshow(risk_matrix, cmap='RdYlBu_r', aspect='auto')
            ax3.set_xticks(range(len(w_under_values)))
            ax3.set_yticks(range(len(w_over_values)))
            ax3.set_xticklabels([f'{w:.1f}' for w in w_under_values])
            ax3.set_yticklabels([f'{w:.1f}' for w in w_over_values])
            ax3.set_xlabel('w_under')
            ax3.set_ylabel('w_over')
            ax3.set_title('權重敏感性熱圖')
            plt.colorbar(im, ax=ax3, label='基差風險')
        else:
            ax3.text(0.5, 0.5, '權重組合不足以生成熱圖', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('權重敏感性熱圖 (數據不足)')
        
        # 4. 改進效果分析
        ax4 = axes[1, 1]
        improvements = [r.improvement_vs_baseline for r in results]
        weight_labels = [f'({r.weight_combination[0]:.1f},{r.weight_combination[1]:.1f})' 
                        for r in results]
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax4.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
        ax4.set_xlabel('權重組合')
        ax4.set_ylabel('相對基準的改進率')
        ax4.set_title('相對基準組合 (2.0, 0.5) 的改進')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xticks(range(len(improvements)))
        ax4.set_xticklabels(weight_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = Path(self.config.output_dir) / "weight_sensitivity_analysis.png"
        plt.savefig(output_file, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 視覺化已保存: {output_file}")
    
    def save_analysis_results(self, analysis: WeightSensitivityAnalysis) -> None:
        """保存分析結果"""
        
        if not self.config.save_detailed_results:
            return
        
        print("💾 保存分析結果...")
        
        # 保存 JSON 結果
        results_dict = {
            'analysis_metadata': {
                'timestamp': analysis.analysis_timestamp,
                'config': {
                    'weight_combinations': analysis.analysis_config.weight_combinations,
                    'basis_risk_type': analysis.analysis_config.basis_risk_type
                },
                'data_summary': analysis.data_summary
            },
            'sensitivity_summary': {
                'best_weight_combination': analysis.best_weight_combination,
                'worst_weight_combination': analysis.worst_weight_combination,
                'sensitivity_level': analysis.sensitivity_level,
                'correlation_analysis': analysis.correlation_analysis
            },
            'detailed_results': [
                {
                    'weight_combination': r.weight_combination,
                    'weight_ratio': r.weight_ratio,
                    'optimal_basis_risk': r.optimal_basis_risk,
                    'optimal_trigger_threshold': r.optimal_trigger_threshold,
                    'optimal_payout_amount': r.optimal_payout_amount,
                    'trigger_rate': r.optimal_trigger_rate,
                    'improvement_vs_baseline': r.improvement_vs_baseline,
                    'rank': r.rank_among_combinations
                } for r in analysis.weight_results
            ],
            'insights': analysis.key_insights,
            'recommendations': analysis.business_recommendations
        }
        
        json_file = Path(self.config.output_dir) / "weight_sensitivity_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存 CSV 結果
        results_df = pd.DataFrame([
            {
                'w_under': r.weight_combination[0],
                'w_over': r.weight_combination[1],
                'weight_ratio': r.weight_ratio,
                'basis_risk': r.optimal_basis_risk,
                'trigger_threshold': r.optimal_trigger_threshold,
                'payout_amount': r.optimal_payout_amount,
                'trigger_rate': r.optimal_trigger_rate,
                'expected_payout': r.expected_payout,
                'coverage_efficiency': r.coverage_efficiency,
                'improvement_vs_baseline': r.improvement_vs_baseline,
                'rank': r.rank_among_combinations
            } for r in analysis.weight_results
        ])
        
        csv_file = Path(self.config.output_dir) / "weight_sensitivity_results.csv"
        results_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"✅ 結果已保存:")
        print(f"   JSON: {json_file}")
        print(f"   CSV: {csv_file}")

# =============================================================================
# 整合介面：與 RobustBayesianAnalyzer 的擴展
# =============================================================================

def extend_robust_analyzer_with_weight_sensitivity(analyzer: 'RobustBayesianAnalyzer') -> 'RobustBayesianAnalyzer':
    """
    為 RobustBayesianAnalyzer 添加權重敏感性分析功能
    
    Parameters:
    -----------
    analyzer : RobustBayesianAnalyzer
        要擴展的分析器
        
    Returns:
    --------
    RobustBayesianAnalyzer
        添加了權重敏感性功能的分析器
    """
    
    def analyze_weight_sensitivity_integrated(
        observations: np.ndarray,
        validation_data: np.ndarray,
        hazard_indices: np.ndarray,
        actual_losses: np.ndarray,
        product_bounds: Dict[str, Tuple[float, float]],
        weight_sensitivity_config: Optional[WeightSensitivityConfig] = None
    ) -> WeightSensitivityAnalysis:
        """整合的權重敏感性分析方法"""
        
        sensitivity_analyzer = WeightSensitivityAnalyzer(
            config=weight_sensitivity_config,
            robust_analyzer=analyzer
        )
        
        return sensitivity_analyzer.analyze_weight_sensitivity(
            observations, validation_data, hazard_indices, 
            actual_losses, product_bounds
        )
    
    # 動態添加方法到分析器
    analyzer.analyze_weight_sensitivity = analyze_weight_sensitivity_integrated
    
    return analyzer

# =============================================================================
# 便利函數
# =============================================================================

def create_weight_sensitivity_analyzer(config: Optional[WeightSensitivityConfig] = None) -> WeightSensitivityAnalyzer:
    """
    創建權重敏感性分析器的便利函數
    
    Parameters:
    -----------
    config : WeightSensitivityConfig, optional
        配置參數
        
    Returns:
    --------
    WeightSensitivityAnalyzer
        權重敏感性分析器
    """
    return WeightSensitivityAnalyzer(config=config)

def quick_weight_sensitivity_analysis(
    observations: np.ndarray,
    validation_data: np.ndarray,
    hazard_indices: np.ndarray,
    actual_losses: np.ndarray,
    product_bounds: Dict[str, Tuple[float, float]],
    output_dir: str = "results/weight_sensitivity"
) -> WeightSensitivityAnalysis:
    """
    快速權重敏感性分析
    
    使用默認配置執行完整的權重敏感性分析
    """
    
    config = WeightSensitivityConfig(output_dir=output_dir)
    analyzer = WeightSensitivityAnalyzer(config=config)
    
    analysis = analyzer.analyze_weight_sensitivity(
        observations, validation_data, hazard_indices, 
        actual_losses, product_bounds
    )
    
    analyzer.visualize_sensitivity_results(analysis)
    analyzer.save_analysis_results(analysis)
    
    return analysis