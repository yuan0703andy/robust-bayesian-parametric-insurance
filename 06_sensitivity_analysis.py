#!/usr/bin/env python3
"""
權重敏感性分析 - 使用模組化架構
Weight Sensitivity Analysis - Using Modular Architecture

使用新的模組化 bayesian.WeightSensitivityAnalyzer 實現權重敏感性分析
針對懲罰權重 (w_under, w_over) 的敏感性進行全面分析

Author: Research Team  
Date: 2025-01-10
"""

import numpy as np
from pathlib import Path

# 使用新的模組化組件
from bayesian import WeightSensitivityAnalyzer


def main():
    """主執行函數 - 使用模組化權重敏感性分析器"""
    
    print("🚀 權重敏感性分析開始（使用模組化架構）...")
    print("=" * 60)
    
    # 生成測試數據
    print("📊 生成模擬數據...")
    
    n_scenarios = 1000
    np.random.seed(42)
    
    # 生成損失情境 (對數正態分佈，更符合巨災損失特徵)
    loss_mean = 1e8  # 1億平均損失
    loss_std = 5e7   # 0.5億標準差
    
    log_mean = np.log(loss_mean) - 0.5 * np.log(1 + (loss_std / loss_mean) ** 2)
    log_std = np.sqrt(np.log(1 + (loss_std / loss_mean) ** 2))
    
    actual_losses = np.random.lognormal(log_mean, log_std, n_scenarios)
    
    # 加入30%的零損失情境
    zero_loss_indices = np.random.choice(n_scenarios, size=int(0.3 * n_scenarios), replace=False)
    actual_losses[zero_loss_indices] = 0
    
    # 生成災害指標 (Gamma分佈，更符合風災指標)
    hazard_indices = np.random.gamma(2, 20, n_scenarios)
    
    print(f"   損失範圍: {actual_losses.min():.2e} - {actual_losses.max():.2e}")
    print(f"   災害指標範圍: {hazard_indices.min():.2f} - {hazard_indices.max():.2f}")
    
    # 創建權重敏感性分析器（使用模組化組件）
    print("\n🔧 初始化權重敏感性分析器...")
    
    # 定義權重組合進行測試
    weight_combinations = [
        # 基準組合
        (2.0, 0.5),  # 當前使用 (4:1 比率)
        
        # 對稱組合  
        (1.0, 1.0),  # 相等權重
        
        # 不同比率測試
        (2.0, 1.0),  # 2:1 比率
        (3.0, 1.0),  # 3:1 比率
        (4.0, 1.0),  # 4:1 比率
        (5.0, 1.0),  # 5:1 比率
        (10.0, 1.0), # 10:1 比率
        
        # 反向權重 (更關心過度賠付)
        (0.5, 2.0),  # 1:4 比率
        (1.0, 2.0),  # 1:2 比率
        
        # 極端情況
        (5.0, 0.1),  # 極度懲罰不足覆蓋
        (0.1, 5.0),  # 極度懲罰過度覆蓋
        
        # 溫和權重
        (1.5, 1.0),  # 1.5:1 比率
        (1.0, 0.7),  # 1:0.7 比率
    ]
    
    # 使用模組化組件配置
    from bayesian.weight_sensitivity_analyzer import WeightSensitivityConfig
    
    config = WeightSensitivityConfig(
        weight_combinations=weight_combinations,
        output_dir="results/sensitivity_analysis_modular"
    )
    
    analyzer = WeightSensitivityAnalyzer(config=config)
    
    # 執行權重敏感性分析
    print(f"\n🔍 執行權重敏感性分析 ({len(weight_combinations)} 個權重組合)...")
    
    # 定義產品參數搜索範圍
    product_bounds = {
        'trigger_threshold': (np.percentile(hazard_indices, 50), np.percentile(hazard_indices, 95)),
        'payout_amount': (np.percentile(actual_losses[actual_losses > 0], 10), 
                         np.percentile(actual_losses[actual_losses > 0], 90))
    }
    
    print(f"   產品搜索範圍:")
    print(f"     觸發閾值: {product_bounds['trigger_threshold'][0]:.2f} - {product_bounds['trigger_threshold'][1]:.2f}")
    print(f"     賠付金額: {product_bounds['payout_amount'][0]:.2e} - {product_bounds['payout_amount'][1]:.2e}")
    
    results = analyzer.analyze_weight_sensitivity(
        observations=actual_losses,      # 訓練數據
        validation_data=actual_losses,   # 驗證數據（在這個示例中使用相同數據）
        hazard_indices=hazard_indices,
        actual_losses=actual_losses,
        product_bounds=product_bounds
    )
    
    # 顯示結果摘要
    print("\n" + "=" * 60)
    print("📋 權重敏感性分析結果摘要:")
    print("=" * 60)
    
    if hasattr(results, 'weight_combinations_analysis') and results.weight_combinations_analysis:
        analysis_data = results.weight_combinations_analysis
        
        # 找到最佳和最差權重組合
        best_idx = min(range(len(analysis_data)), key=lambda i: analysis_data[i]['optimal_expected_loss'])
        worst_idx = max(range(len(analysis_data)), key=lambda i: analysis_data[i]['optimal_expected_loss'])
        
        best_combo = analysis_data[best_idx]
        worst_combo = analysis_data[worst_idx]
        
        print(f"✅ 最佳權重組合:")
        print(f"   w_under={best_combo['w_under']:.1f}, w_over={best_combo['w_over']:.1f}")
        print(f"   權重比率: {best_combo['w_under']/best_combo['w_over']:.1f}:1")
        print(f"   最小期望損失: {best_combo['optimal_expected_loss']:.2e}")
        
        print(f"\n❌ 最差權重組合:")
        print(f"   w_under={worst_combo['w_under']:.1f}, w_over={worst_combo['w_over']:.1f}")
        print(f"   權重比率: {worst_combo['w_under']/worst_combo['w_over']:.1f}:1")
        print(f"   最大期望損失: {worst_combo['optimal_expected_loss']:.2e}")
        
        print(f"\n📈 敏感性統計:")
        risks = [x['optimal_expected_loss'] for x in analysis_data]
        print(f"   期望損失變異係數: {np.std(risks)/np.mean(risks):.3f}")
        print(f"   性能差異倍數: {worst_combo['optimal_expected_loss']/best_combo['optimal_expected_loss']:.2f}")
    else:
        print("⚠️ 分析結果格式未知或為空")
        if hasattr(results, '__dict__'):
            print(f"   結果屬性: {list(results.__dict__.keys())}")
    
    # 輸出檔案位置
    print(f"\n📁 結果已保存至:")
    output_dir = Path("results/sensitivity_analysis_modular")
    if output_dir.exists():
        for file in output_dir.glob("*"):
            print(f"   • {file}")
    
    # 如果有總結報告，顯示關鍵洞察
    if hasattr(results, 'summary_report'):
        print(f"\n💡 關鍵洞察:")
        summary = results.summary_report
        if 'sensitivity_insights' in summary:
            insights = summary['sensitivity_insights']
            if 'correlation_analysis' in insights:
                correlation = insights['correlation_analysis']
                print(f"   權重-性能相關性: {correlation.get('correlation_coefficient', 'N/A')}")
            if 'robustness_assessment' in insights:
                robustness = insights['robustness_assessment']
                print(f"   穩健性評分: {robustness.get('stability_score', 'N/A')}")
            if 'recommendations' in insights:
                print(f"   建議: {insights['recommendations']}")
    
    print(f"\n🎉 權重敏感性分析完成！")
    print("✨ 使用模組化 bayesian.WeightSensitivityAnalyzer 實現")
    
    return results


if __name__ == "__main__":
    results = main()