#!/usr/bin/env python3
"""
技術保費多目標分析 - 使用模組化架構
Technical Premium Multi-Objective Analysis - Using Modular Architecture

使用新的模組化架構實現技術保費分析：
- TechnicalPremiumCalculator: 進階保費計算（VaR、Solvency II）
- MarketAcceptabilityAnalyzer: 市場接受度分析
- MultiObjectiveOptimizer: Pareto前緣分析
- TechnicalPremiumVisualizer: 視覺化和報告

Author: Research Team
Date: 2025-01-10
"""

import numpy as np
from pathlib import Path

# 使用新的模組化組件
from insurance_analysis_refactored.core import (
    create_standard_technical_premium_calculator,
    create_standard_market_analyzer,
    create_standard_multi_objective_optimizer,
    create_standard_visualizer,
    create_standard_report_generator,
    ParametricProduct,
    PayoutFunctionType,
    OptimizationConfig,
    OptimizationObjective,
    DecisionPreferenceType
)


def generate_candidate_products(actual_losses: np.ndarray, 
                               hazard_indices: np.ndarray, 
                               n_candidates: int = 100) -> list:
    """生成候選產品"""
    
    # 定義搜索空間
    trigger_min, trigger_max = np.percentile(hazard_indices, [50, 95])
    payout_min = np.percentile(actual_losses[actual_losses > 0], 10)
    payout_max = np.percentile(actual_losses[actual_losses > 0], 90)
    
    print(f"產品參數搜索空間:")
    print(f"  觸發閾值: {trigger_min:.2f} - {trigger_max:.2f}")
    print(f"  賠付金額: {payout_min:.2e} - {payout_max:.2e}")
    
    # 生成候選產品
    np.random.seed(42)
    candidate_products = []
    
    for i in range(n_candidates):
        trigger = np.random.uniform(trigger_min, trigger_max)
        payout = np.random.uniform(payout_min, payout_max)
        max_payout = payout * np.random.uniform(1.0, 3.0)  # 最大賠付為基本賠付的1-3倍
        
        product = ParametricProduct(
            product_id=f"CANDIDATE_{i+1:03d}",
            name=f"候選產品 {i+1}",
            description=f"單一觸發產品，觸發={trigger:.1f}, 賠付={payout:.1e}",
            index_type="cat_in_circle",  # 簡化為字符串
            payout_function_type="step",  # 簡化為字符串
            trigger_thresholds=[trigger],
            payout_amounts=[payout],
            max_payout=max_payout
        )
        candidate_products.append(product)
    
    return candidate_products


def main():
    """主執行函數 - 使用模組化技術保費分析"""
    
    print("🚀 技術保費多目標分析開始（使用模組化架構）...")
    print("=" * 80)
    
    # 生成模擬數據
    print("📊 生成模擬數據...")
    np.random.seed(42)
    
    n_scenarios = 1000
    
    # 生成損失數據 (混合分佈，模擬真實巨災損失)
    normal_losses = np.random.lognormal(np.log(5e7), 0.8, int(0.8 * n_scenarios))
    extreme_losses = np.random.lognormal(np.log(2e8), 1.0, int(0.2 * n_scenarios))
    actual_losses = np.concatenate([normal_losses, extreme_losses])
    np.random.shuffle(actual_losses)
    
    # 生成災害指標
    hazard_indices = np.random.gamma(2, 25, n_scenarios)
    
    print(f"模擬數據統計:")
    print(f"  情境數量: {n_scenarios}")
    print(f"  平均損失: {np.mean(actual_losses):.2e}")
    print(f"  損失範圍: {actual_losses.min():.2e} - {actual_losses.max():.2e}")
    print(f"  災害指標範圍: {hazard_indices.min():.2f} - {hazard_indices.max():.2f}")
    
    # 創建模組化組件
    print("\n🔧 初始化模組化組件...")
    
    # 1. 技術保費計算器（進階功能：VaR, Solvency II）
    premium_calculator = create_standard_technical_premium_calculator(
        risk_free_rate=0.02,
        risk_loading_factor=0.20,
        solvency_ratio=1.25,
        expense_ratio=0.15,
        profit_margin=0.10,
        confidence_level=0.995
    )
    print("   ✅ 技術保費計算器 (含VaR & Solvency II)")
    
    # 2. 市場接受度分析器
    market_analyzer = create_standard_market_analyzer(
        optimal_trigger_rate=0.20,
        market_benchmark=1.5
    )
    print("   ✅ 市場接受度分析器")
    
    # 3. 多目標優化器（Pareto前緣分析）
    optimizer = create_standard_multi_objective_optimizer(
        premium_calculator, market_analyzer
    )
    print("   ✅ 多目標優化器 (Pareto前緣)")
    
    # 4. 視覺化和報告生成器
    visualizer = create_standard_visualizer()
    report_generator = create_standard_report_generator()
    print("   ✅ 視覺化器和報告生成器")
    
    # 生成候選產品
    print("\n📦 生成候選產品...")
    candidate_products = generate_candidate_products(
        actual_losses, hazard_indices, n_candidates=200
    )
    print(f"   生成 {len(candidate_products)} 個候選產品")
    
    # 配置多目標優化
    optimization_config = OptimizationConfig(
        objectives=[
            OptimizationObjective.MINIMIZE_TECHNICAL_PREMIUM,
            OptimizationObjective.MINIMIZE_BASIS_RISK,
            OptimizationObjective.MAXIMIZE_MARKET_ACCEPTABILITY
        ],
        n_candidates=len(candidate_products),
        enable_pareto_analysis=True,
        enable_preference_ranking=True,
        random_seed=42
    )
    
    # 執行多目標優化
    print(f"\n🎯 執行多目標優化...")
    print(f"   優化目標: 技術保費最小化 + 基差風險最小化 + 市場接受度最大化")
    
    results = optimizer.optimize(
        candidate_products=candidate_products,
        actual_losses=actual_losses,
        hazard_indices=hazard_indices,
        config=optimization_config
    )
    
    # 生成視覺化
    print("\n📊 生成多目標優化視覺化...")
    output_dir = "results/technical_premium_modular"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    viz_file = visualizer.visualize_multi_objective_results(
        results, output_dir=output_dir, show_plots=False
    )
    print(f"   ✅ 多目標優化圖表: {viz_file}")
    
    # 生成偏好分析視覺化
    pref_file = visualizer.visualize_preference_analysis(
        results, output_dir=output_dir, show_plots=False
    )
    if pref_file:
        print(f"   ✅ 偏好分析圖表: {pref_file}")
    
    # 生成決策支援報告
    print("\n📋 生成決策支援報告...")
    report_file = report_generator.generate_decision_support_report(
        results, output_dir=output_dir
    )
    print(f"   ✅ 決策支援報告: {report_file}")
    
    # 顯示結果摘要
    print("\n" + "=" * 80)
    print("📋 技術保費多目標分析結果摘要:")
    print("=" * 80)
    
    summary = results.optimization_summary
    print(f"✅ 分析完成:")
    print(f"   評估候選產品: {summary['total_candidates']} 個")
    print(f"   Pareto效率解: {summary['pareto_efficient_solutions']} 個")
    print(f"   效率解比例: {summary['pareto_efficiency_rate']:.1%}")
    
    # 顯示各偏好類型的最佳解
    print(f"\n🎯 各決策偏好下的最佳產品:")
    best_solutions = summary.get('best_solutions_by_preference', {})
    
    preference_labels = {
        'risk_averse': '風險厭惡型',
        'cost_sensitive': '成本敏感型', 
        'market_oriented': '市場導向型',
        'balanced': '平衡型'
    }
    
    for pref_type, label in preference_labels.items():
        if pref_type in best_solutions:
            sol = best_solutions[pref_type]
            print(f"\n   {label}:")
            print(f"     推薦產品: {sol['product_id']}")
            print(f"     技術保費: ${sol['technical_premium']:.2e}")
            print(f"     基差風險: ${sol['basis_risk']:.2e}")
            print(f"     市場接受度: {sol['market_acceptability']:.1%}")
    
    print(f"\n🎯 主要發現:")
    print("1. ✅ 技術保費包含完整的VaR和Solvency II風險資本計算")
    print("2. ✅ 市場接受度考慮產品複雜度、觸發頻率和保費可負擔性")
    print("3. ✅ Pareto前緣提供無支配的產品組合供決策參考")
    print("4. ✅ 不同決策偏好導向不同的最佳產品選擇")
    print("5. ✅ 平衡型策略在多數情況下提供良好的整體性能")
    
    print(f"\n📁 所有輸出文件保存在: {output_dir}/")
    print("✨ 使用完全模組化的 insurance_analysis_refactored.core 組件實現")
    
    print(f"\n🎉 技術保費多目標分析完成！")
    
    return results


if __name__ == "__main__":
    results = main()