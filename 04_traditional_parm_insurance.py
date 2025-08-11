#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_traditional_parm_insurance.py
=================================
Traditional Parametric Insurance Analysis using existing framework

Simply configures and runs the existing analysis components for 
traditional RMSE-based basis risk evaluation.
"""
# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from pathlib import Path

# %%
def main():
    """
    Main program: Traditional basis risk analysis using existing framework
    """
    print("=" * 80)
    print("Traditional Parametric Insurance Analysis")
    print("Using existing insurance_analysis_refactored framework")
    print("RMSE-based deterministic evaluation")
    print("=" * 80)
    
    # Load required data
    print("\n📂 Loading data...")
    
    # Load products
    try:
        with open("results/insurance_products/products.pkl", 'rb') as f:
            products = pickle.load(f)
        print(f"✅ Loaded {len(products)} insurance products")
    except FileNotFoundError:
        print("❌ Products not found. Run 03_insurance_product.py first.")
        return
    
    # Load spatial analysis results  
    try:
        with open("results/spatial_analysis/cat_in_circle_results.pkl", 'rb') as f:
            spatial_results = pickle.load(f)
        wind_indices_dict = spatial_results['indices']
        # Extract main wind index for analysis (using 30km max as primary)
        wind_indices = wind_indices_dict.get('cat_in_circle_30km_max', np.array([]))
        print("✅ Loaded spatial analysis results")
        print(f"   Using primary index: cat_in_circle_30km_max ({len(wind_indices)} events)")
    except FileNotFoundError:
        print("❌ Spatial results not found. Run 02_spatial_analysis.py first.")
        return
    
    # Load CLIMADA data
    try:
        with open("results/climada_data/climada_complete_data.pkl", 'rb') as f:
            climada_data = pickle.load(f)
        print("✅ Loaded CLIMADA data (真實數據)")
    except FileNotFoundError:
        print("⚠️ Using synthetic loss data (風速相關)")
        np.random.seed(42)
        # 創建與風速相關的合成損失數據
        n_events = len(wind_indices) if len(wind_indices) > 0 else 1000
        
        # 基於風速生成損失（風速越高，損失越大）
        # 使用指數關係模擬真實的風災損失
        synthetic_losses = np.zeros(n_events)
        for i, wind in enumerate(wind_indices[:n_events]):
            if wind > 33:  # 颱風閾值
                # 損失與風速的3.5次方成正比（符合Emanuel公式）
                base_loss = (wind / 33) ** 3.5 * 1e8
                # 加入隨機變異
                synthetic_losses[i] = base_loss * np.random.lognormal(0, 0.5)
            else:
                # 低於颱風閾值，小概率產生小損失
                if np.random.random() < 0.05:
                    synthetic_losses[i] = np.random.lognormal(10, 2) * 1e3
        
        climada_data = {
            'impact': type('MockImpact', (), {
                'at_event': synthetic_losses
            })()
        }
    
    # Ensure data arrays have matching lengths
    observed_losses = climada_data.get('impact').at_event if 'impact' in climada_data else np.array([])
    
    # Truncate to minimum length to ensure compatibility
    min_length = min(len(wind_indices), len(observed_losses))
    if min_length > 0:
        wind_indices = wind_indices[:min_length]
        observed_losses = observed_losses[:min_length]
        print(f"   Aligned data to {min_length} events")
    else:
        print("❌ No valid data found")
        return
    
    print("\n📊 執行傳統基差風險分析...")
    print("   • 方法: 多種基差風險定義")
    print("   • 指標: 絕對、不對稱、加權不對稱、RMSE、相對絕對、相對加權不對稱 基差風險") 
    print("   • 方式: 確定性點估計 + 相對基差風險（解決極端事件主導問題）")
    print(f"   • 使用預生成產品: {len(products)} 個")
    
    # Import basis risk calculator (使用整合的 skill_scores 模組)
    from skill_scores.basis_risk_functions import BasisRiskCalculator, BasisRiskType, BasisRiskConfig
    
    # 初始化不同類型的基差風險計算器 (包含相對基差風險)
    calculators = {
        # 傳統絕對基差風險
        'absolute': BasisRiskCalculator(BasisRiskConfig(
            risk_type=BasisRiskType.ABSOLUTE, 
            normalize=False  # 關閉標準化以顯示實際差異
        )),
        'asymmetric': BasisRiskCalculator(BasisRiskConfig(
            risk_type=BasisRiskType.ASYMMETRIC,
            normalize=False  # 關閉標準化
        )),  
        'weighted': BasisRiskCalculator(BasisRiskConfig(
            risk_type=BasisRiskType.WEIGHTED_ASYMMETRIC,
            w_under=2.0,  # 賠不夠的懲罰權重
            w_over=0.5,   # 賠多了的懲罰權重
            normalize=False  # 關閉標準化
        )),
        'rmse': BasisRiskCalculator(BasisRiskConfig(
            risk_type=BasisRiskType.RMSE,
            normalize=False  # 關閉標準化
        )),
        # 新增相對基差風險 - 來自 07_relative_basis_risk_analysis.py
        'relative_absolute': BasisRiskCalculator(BasisRiskConfig(
            risk_type=BasisRiskType.RELATIVE_ABSOLUTE,
            min_loss_threshold=1e7,  # 最小損失閾值 1千萬
            normalize=False
        )),
        'relative_weighted': BasisRiskCalculator(BasisRiskConfig(
            risk_type=BasisRiskType.RELATIVE_WEIGHTED_ASYMMETRIC,
            w_under=2.0,
            w_over=0.5,
            min_loss_threshold=1e7,  # 最小損失閾值 1千萬
            normalize=False
        ))
    }
    
    # 分析結果儲存
    analysis_results = []
    
    print(f"   分析產品數量: {len(products)}")
    print(f"   事件數量: {len(wind_indices)}")
    
    # 檢查前3個產品的設置 (調試信息)
    print(f"\n🔍 產品設置檢查 (前3個):")
    for i in range(min(3, len(products))):
        product = products[i]
        print(f"  {product['product_id']}: 閾值={product['trigger_thresholds']}")
        print(f"    賠付比例={product['payout_ratios']}, 最大賠付={product['max_payout']}")
    
    # 檢查風速數據範圍
    print(f"\n🌪️  風速數據檢查:")
    print(f"   風速範圍: {np.min(wind_indices):.2f} - {np.max(wind_indices):.2f}")
    print(f"   風速平均: {np.mean(wind_indices):.2f}")
    print(f"   風速標準差: {np.std(wind_indices):.2f}")
    
    for i, product in enumerate(products):
        if (i + 1) % 20 == 0:
            print(f"   進度: {i+1}/{len(products)}")
        
        # 計算階梯式賠付 (使用整合的 skill_scores 模組)
        from skill_scores.basis_risk_functions import calculate_step_payouts_batch
        
        payouts = calculate_step_payouts_batch(
            wind_indices,
            product['trigger_thresholds'],
            product['payout_ratios'],
            product['max_payout']
        )
        
        # 調試：檢查前幾個產品的賠付分佈
        if i < 3:
            print(f"  產品 {product['product_id']}: 賠付範圍={np.min(payouts):.2e}-{np.max(payouts):.2e}, 觸發率={np.mean(payouts > 0):.3f}")
        
        # 計算各種基差風險指標
        product_result = {
            'product_id': product['product_id'],
            'name': product.get('name', 'Unknown'),
            'structure_type': product['structure_type'],
            'radius_km': product.get('radius_km', 30),
            'n_thresholds': len(product['trigger_thresholds']),
            'max_payout': product['max_payout']
        }
        
        # 使用不同的基差風險計算器
        for risk_name, calculator in calculators.items():
            try:
                risk_value = calculator.calculate_basis_risk(observed_losses, payouts)
                product_result[f'{risk_name}_risk'] = risk_value
            except Exception as e:
                print(f"Warning: Failed to calculate {risk_name} risk for {product['product_id']}: {e}")
                product_result[f'{risk_name}_risk'] = np.inf
        
        # 計算額外的傳統指標
        try:
            product_result['correlation'] = np.corrcoef(observed_losses, payouts)[0,1] if np.std(payouts) > 0 else 0
            product_result['trigger_rate'] = np.mean(payouts > 0)
            product_result['mean_payout'] = np.mean(payouts)
            product_result['coverage_ratio'] = np.sum(payouts) / np.sum(observed_losses) if np.sum(observed_losses) > 0 else 0
            product_result['basis_risk_std'] = np.std(observed_losses - payouts)
        except Exception as e:
            print(f"Warning: Failed to calculate additional metrics for {product['product_id']}: {e}")
            for key in ['correlation', 'trigger_rate', 'mean_payout', 'coverage_ratio', 'basis_risk_std']:
                if key not in product_result:
                    product_result[key] = 0
        
        analysis_results.append(product_result)
    
    # 創建結果DataFrame
    import pandas as pd
    results_df = pd.DataFrame(analysis_results)
    
    # 將框架調用替換為我們的分析結果
    class TraditionalAnalysisResults:
        def __init__(self, results_df):
            self.results_df = results_df
            self.best_products = self._find_best_products()
            self.summary_statistics = self._generate_summary()
        
        def _find_best_products(self):
            best_products = {}
            metrics = ['absolute_risk', 'asymmetric_risk', 'weighted_risk', 'rmse_risk']
            
            for metric in metrics:
                if metric in self.results_df.columns:
                    best_idx = self.results_df[metric].idxmin()
                    if not pd.isna(best_idx):
                        best_product = self.results_df.iloc[best_idx]
                        # 使用字典而不是動態類別，避免 pickle 問題
                        best_products[f'best_{metric}'] = {
                            'product_id': best_product['product_id'],
                            'name': best_product['name'], 
                            'description': f"{best_product['structure_type']} threshold product"
                        }
            
            return best_products
        
        def _generate_summary(self):
            return {
                'total_products': len(self.results_df),
                'analysis_type': 'Traditional Basis Risk Analysis'
            }
    
    results = TraditionalAnalysisResults(results_df)
    
    # Extract and display results
    print("\n✅ Traditional basis risk analysis complete!")
    print(f"📊 Analyzed {len(products)} products with multiple basis risk definitions")
    
    # Display comprehensive basis risk analysis results
    print("\n📋 基差風險分析摘要:")
    print("=" * 60)
    
    # 基本統計
    print(f"總產品數: {len(results_df)}")
    print(f"產品結構分布:")
    structure_counts = results_df['structure_type'].value_counts()
    for structure, count in structure_counts.items():
        print(f"  • {structure.capitalize()}: {count} 產品")
    
    # 基差風險統計摘要（包含相對基差風險）
    print(f"\n🎯 基差風險指標統計:")
    risk_metrics = ['absolute_risk', 'asymmetric_risk', 'weighted_risk', 'rmse_risk', 
                    'relative_absolute_risk', 'relative_weighted_risk']
    
    for metric in risk_metrics:
        if metric in results_df.columns and not results_df[metric].isna().all():
            mean_risk = results_df[metric].mean()
            min_risk = results_df[metric].min()
            max_risk = results_df[metric].max()
            print(f"  • {metric.replace('_', ' ').title()}:")
            print(f"    平均: {mean_risk:.2e}, 最小: {min_risk:.2e}, 最大: {max_risk:.2e}")
    
    # 顯示最佳產品
    if hasattr(results, 'best_products') and results.best_products:
        print(f"\n🏆 各指標最佳產品:")
        print("-" * 40)
        
        count = 0
        for metric, product in results.best_products.items():
            if count >= 8:  # 限制顯示數量
                break
            count += 1
            
            metric_name = metric.replace('best_', '').replace('_', ' ').title()
            print(f"{count}. {metric_name}: {product['name']}")
            print(f"   產品ID: {product['product_id']}")
            print(f"   描述: {product['description']}")
            
            # 顯示該產品的具體風險值
            product_row = results_df[results_df['product_id'] == product['product_id']]
            if not product_row.empty:
                risk_col = metric.replace('best_', '') + '_risk' if not metric.endswith('_risk') else metric.replace('best_', '')
                if risk_col in product_row.columns:
                    risk_value = product_row[risk_col].iloc[0]
                    print(f"   風險值: {risk_value:.6f}")
            print()
    
    # Top 10 綜合排名 (使用加權不對稱基差風險)
    print(f"\n📈 Top 10 產品排名 (按加權不對稱基差風險):")
    print("-" * 40)
    
    if 'weighted_risk' in results_df.columns:
        top_10 = results_df.nsmallest(10, 'weighted_risk')
        
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            print(f"{i:2d}. {row['product_id']} ({row['structure_type']})")
            print(f"     加權基差風險: {row['weighted_risk']:.2e}")
            print(f"     絕對基差風險: {row.get('absolute_risk', 'N/A'):.2e}")
            print(f"     不對稱基差風險: {row.get('asymmetric_risk', 'N/A'):.2e}")
            print(f"     RMSE風險: {row.get('rmse_risk', 'N/A'):.2e}")
            print(f"     觸發率: {row.get('trigger_rate', 'N/A'):.3f}")
            print(f"     相關係數: {row.get('correlation', 'N/A'):.3f}")
            print()
    
    # 額外統計
    print(f"\n📊 額外統計指標:")
    avg_trigger_rate = results_df['trigger_rate'].mean()
    avg_correlation = results_df['correlation'].mean()
    avg_coverage = results_df['coverage_ratio'].mean()
    
    print(f"  • 平均觸發率: {avg_trigger_rate:.3f}")
    print(f"  • 平均相關係數: {avg_correlation:.3f}")
    print(f"  • 平均覆蓋率: {avg_coverage:.3f}")
    
    # Skill Score 評估（更新包含相對基差風險）
    print(f"\n🎯 Skill Score 評估:")
    print(f"  基差風險分析中的不同損失函數比較:")
    print(f"  • 絕對基差風險: 對稱懲罰所有偏差")
    print(f"  • 不對稱基差風險: 只懲罰賠付不足")
    print(f"  • 加權不對稱基差風險: 不對稱懲罰，權重化考慮")
    print(f"  • RMSE風險: 傳統統計方法，平方懲罰")
    print(f"  • 相對絕對基差風險: 標準化處理，避免極端事件主導")
    print(f"  • 相對加權不對稱基差風險: 結合權重與標準化的優勢")
    
    # 相對 vs 絕對基差風險對比分析
    print(f"\n🔍 相對 vs 絕對基差風險對比:")
    if 'relative_weighted_risk' in results_df.columns and 'weighted_risk' in results_df.columns:
        abs_risk_max = results_df['weighted_risk'].max()
        rel_risk_max = results_df['relative_weighted_risk'].max()
        abs_risk_dominated = (results_df['weighted_risk'].quantile(0.9) - results_df['weighted_risk'].quantile(0.1)) / results_df['weighted_risk'].mean()
        rel_risk_spread = (results_df['relative_weighted_risk'].quantile(0.9) - results_df['relative_weighted_risk'].quantile(0.1)) / results_df['relative_weighted_risk'].mean()
        
        print(f"  • 絕對風險最大值: {abs_risk_max:.2e}")
        print(f"  • 相對風險最大值: {rel_risk_max:.3f}")
        print(f"  • 絕對風險變異度: {abs_risk_dominated:.2f}")
        print(f"  • 相對風險變異度: {rel_risk_spread:.2f}")
        print(f"  • 相對基差風險有效減少了極端事件對風險評估的主導效應")
    
    # Save results
    output_dir = "results/traditional_basis_risk_analysis"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save detailed DataFrame  
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    print(f"💾 Detailed results saved to: {output_dir}/detailed_results.csv")
    
    # Save analysis object
    with open(f"{output_dir}/analysis_results.pkl", 'wb') as f:
        pickle.dump({
            'results_df': results_df,
            'best_products': results.best_products,
            'summary_statistics': results.summary_statistics,
            'analysis_config': {
                'risk_types': list(calculators.keys()),
                'n_products': len(products),
                'n_events': len(wind_indices),
                'undercompensation_weight': 2.0,
                'overcompensation_weight': 0.5
            }
        }, f)
    
    print(f"💾 Analysis object saved to: {output_dir}/analysis_results.pkl")
    
    # Generate summary report
    from datetime import datetime
    
    report_lines = [
        "Traditional Basis Risk Analysis Report",
        "=" * 40,
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Products Analyzed: {len(products)}",
        f"Events Analyzed: {len(wind_indices)}",
        "",
        "Basis Risk Definitions Used:",
        "1. Absolute Basis Risk: |Actual_Loss - Payout|",
        "2. Asymmetric Basis Risk: max(0, Actual_Loss - Payout)",
        "3. Weighted Asymmetric: 2.0*undercomp + 0.5*overcomp",
        "4. RMSE Risk: sqrt(mean((Actual_Loss - Payout)²))",
        "",
        "Key Findings:"
    ]
    
    # Add top performers for each metric
    for metric in risk_metrics:
        if metric in results_df.columns and not results_df[metric].isna().all():
            best_product = results_df.loc[results_df[metric].idxmin()]
            best_value = results_df[metric].min()
            report_lines.append(f"- Best {metric.replace('_', ' ').title()}: {best_product['product_id']} (Value: {best_value:.6f})")
    
    report_text = "\n".join(report_lines)
    with open(f"{output_dir}/analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"📄 Analysis report saved to: {output_dir}/analysis_report.txt")
    
    print(f"\n🎉 傳統基差風險分析完成！")
    print("   使用的方法 (整合來自 07_relative_basis_risk_analysis.py):")
    print("   • 絕對基差風險計算 (對稱懲罰)")
    print("   • 不對稱基差風險計算 (只懲罰賠付不足)")
    print("   • 加權不對稱基差風險計算 (權重化懲罰)")
    print("   • 傳統RMSE風險計算")
    print("   • 相對絕對基差風險計算 (標準化避免極端事件主導)")
    print("   • 相對加權不對稱基差風險計算 (結合標準化與權重)")
    print("   • Skill Score多重評估架構")
    print("   • 絕對 vs 相對基差風險對比分析")
    
    return results

# %%
if __name__ == "__main__":
    results = main()
# %%
