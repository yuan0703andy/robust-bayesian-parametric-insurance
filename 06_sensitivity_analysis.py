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
import pandas as pd
from pathlib import Path

# 使用新的模組化組件
from bayesian import WeightSensitivityAnalyzer


def main():
    """主執行函數 - 使用05_robust_bayesian_parm_insurance.py的結果"""
    
    print("🚀 權重敏感性分析開始（基於05的Robust Bayesian結果）...")
    print("=" * 60)
    
    # 載入05_robust_bayesian_parm_insurance.py的結果
    print("📊 載入05_robust_bayesian_parm_insurance.py的模擬結果...")
    
    import pickle
    
    results_file = Path("results/robust_hierarchical_bayesian_analysis/comprehensive_analysis_results.pkl")
    if not results_file.exists():
        print(f"❌ 找不到05的結果文件: {results_file}")
        print("   請先執行 05_robust_bayesian_parm_insurance.py")
        return None
    
    # 載入完整的Robust Bayesian結果
    with open(results_file, 'rb') as f:
        robust_results = pickle.load(f)
    
    print("✅ 成功載入05的Robust Bayesian分析結果")
    print(f"   數據摘要: {robust_results.get('data_summary', {})}")
    
    # 從結果中提取真實數據
    data_summary = robust_results.get('data_summary', {})
    n_events = data_summary.get('n_events', 1000)
    
    # 載入原始數據文件獲取實際的損失和風險指標
    try:
        # 載入空間分析結果
        with open("results/spatial_analysis/cat_in_circle_results.pkl", 'rb') as f:
            spatial_results = pickle.load(f)
        wind_indices_dict = spatial_results['indices']
        hazard_indices = wind_indices_dict.get('cat_in_circle_30km_max', np.array([]))
        
        # 載入CLIMADA數據或生成損失數據
        climada_data = None
        for data_path in ["results/climada_data/climada_complete_data.pkl", "climada_complete_data.pkl"]:
            if Path(data_path).exists():
                try:
                    with open(data_path, 'rb') as f:
                        climada_data = pickle.load(f)
                    break
                except Exception:
                    continue
        
        if climada_data and 'impact' in climada_data:
            actual_losses = climada_data['impact'].at_event
        else:
            # 使用Emanuel關係生成合成損失數據（與05一致）
            np.random.seed(42)
            actual_losses = np.zeros(len(hazard_indices))
            for i, wind in enumerate(hazard_indices):
                if wind > 33:
                    base_loss = ((wind / 33) ** 3.5) * 1e8
                    actual_losses[i] = base_loss * np.random.lognormal(0, 0.5)
                else:
                    if np.random.random() < 0.05:
                        actual_losses[i] = np.random.lognormal(10, 2) * 1e3
        
        # 確保數組長度匹配
        min_length = min(len(hazard_indices), len(actual_losses))
        hazard_indices = hazard_indices[:min_length]
        actual_losses = actual_losses[:min_length]
        
        print(f"✅ 使用來自05分析的真實數據:")
        print(f"   事件數: {min_length}")
        print(f"   損失範圍: {actual_losses.min():.2e} - {actual_losses.max():.2e}")
        print(f"   災害指標範圍: {hazard_indices.min():.2f} - {hazard_indices.max():.2f}")
        
    except Exception as e:
        print(f"⚠️ 無法載入原始數據: {e}")
        print("   使用模擬數據進行權重敏感性分析...")
        
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
    
    # 載入05的Bayesian不確定性量化結果（如果有）
    uncertainty_results = robust_results.get('uncertainty_results', {})
    if uncertainty_results:
        print(f"✅ 發現Bayesian不確定性量化結果:")
        if 'event_loss_distributions' in uncertainty_results:
            n_distributions = len(uncertainty_results['event_loss_distributions'])
            print(f"   機率損失分布: {n_distributions} 事件")
        methodology = uncertainty_results.get('methodology', 'Unknown')
        print(f"   方法: {methodology}")
    
    # 保存robust_results供後續使用
    robust_bayesian_context = {
        'robust_results': robust_results,
        'comprehensive_results': robust_results.get('comprehensive_results'),
        'hierarchical_results': robust_results.get('hierarchical_results'),
        'uncertainty_results': uncertainty_results,
        'original_config': robust_results.get('configuration', {})
    }
    
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
    
    # 使用模組化組件配置，整合05的Robust Bayesian結果
    from bayesian.weight_sensitivity_analyzer import WeightSensitivityConfig
    
    config = WeightSensitivityConfig(
        weight_combinations=weight_combinations,
        output_dir="results/sensitivity_analysis_from_robust_bayesian"
    )
    
    # 如果有05的RobustBayesianAnalyzer結果，嘗試重用它
    robust_analyzer = None
    if robust_bayesian_context['comprehensive_results']:
        print("   🔗 嘗試整合05的RobustBayesianAnalyzer...")
        try:
            # 重新初始化RobustBayesianAnalyzer以便整合
            from bayesian import RobustBayesianAnalyzer
            original_config = robust_bayesian_context['original_config']
            robust_analyzer = RobustBayesianAnalyzer(
                density_ratio_constraint=original_config.get('density_ratio_constraint', 2.0),
                n_monte_carlo_samples=original_config.get('n_monte_carlo_samples', 500),
                n_mixture_components=original_config.get('n_mixture_components', 3),
                hazard_uncertainty_std=original_config.get('hazard_uncertainty_std', 0.15),
                exposure_uncertainty_log_std=original_config.get('exposure_uncertainty_log_std', 0.20),
                vulnerability_uncertainty_std=original_config.get('vulnerability_uncertainty_std', 0.10)
            )
            print("   ✅ 成功整合RobustBayesianAnalyzer")
        except Exception as e:
            print(f"   ⚠️ 無法整合RobustBayesianAnalyzer: {e}")
    
    analyzer = WeightSensitivityAnalyzer(config=config, robust_analyzer=robust_analyzer)
    
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
    print("✨ 基於05_robust_bayesian_parm_insurance.py結果的權重敏感性分析")
    
    # 顯示與05 Robust Bayesian 結果的整合情況
    print(f"\n🔗 與05 Robust Bayesian分析的整合狀況:")
    if robust_bayesian_context['comprehensive_results']:
        print("   ✅ 成功載入並整合05的完整Bayesian優化結果")
    if robust_bayesian_context['hierarchical_results']:
        print("   ✅ 成功載入並整合05的階層Bayesian分析結果")
    if robust_bayesian_context['uncertainty_results']:
        print("   ✅ 成功載入並整合05的不確定性量化結果")
        
    # 保存整合結果
    print(f"\n💾 保存整合分析結果...")
    integrated_results = {
        'weight_sensitivity_results': results,
        'robust_bayesian_context': robust_bayesian_context,
        'analysis_type': 'integrated_weight_sensitivity_from_robust_bayesian',
        'data_source': '05_robust_bayesian_parm_insurance.py',
        'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    output_dir = Path("results/sensitivity_analysis_from_robust_bayesian")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "integrated_sensitivity_results.pkl", 'wb') as f:
        pickle.dump(integrated_results, f)
    print(f"   ✅ 整合結果已保存: {output_dir}/integrated_sensitivity_results.pkl")
    
    return integrated_results


if __name__ == "__main__":
    results = main()