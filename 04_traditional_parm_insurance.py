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

# Import hospital-based configuration
from config.hospital_based_payout_config import HospitalPayoutConfig, create_hospital_based_config

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
    with open("results/insurance_products/products.pkl", 'rb') as f:
        products = pickle.load(f)
    print(f"✅ Loaded {len(products)} insurance products")
    
    # 配置基於醫院的賠付
    print("\n🏥 Configuring hospital-based payouts...")
    hospital_config = create_hospital_based_config(
        n_hospitals=20,  # 預設20家醫院
        base_value_per_hospital=1e7  # 每家醫院$10M USD
    )
    
    # 根據產品類型更新最大賠付
    total_exposure = hospital_config.calculate_total_exposure()
    print(f"   💰 總曝險值: ${total_exposure:,.0f}")
    
    # 獲取不同產品類型的最大賠付（使用50km標準半徑）
    max_payouts = hospital_config.get_max_payout_amounts(total_exposure, radius_km=50)
    print(f"   📊 最大賠付配置:")
    for ptype, amount in max_payouts.items():
        print(f"      - {ptype}: ${amount:,.0f}")
    
    # 更新產品的最大賠付值
    for product in products:
        # 根據產品結構類型設定最大賠付
        structure_type = product.get('structure_type', 'single')
        if structure_type in max_payouts:
            original_payout = product['max_payout']
            product['max_payout'] = max_payouts[structure_type]
            # 調整賠付比例以保持相對關係
            if original_payout > 0:
                scale_factor = max_payouts[structure_type] / original_payout
                # 如果需要，也可以調整賠付比例
                # product['payout_ratios'] = [r * scale_factor for r in product['payout_ratios']]
    
    print(f"   ✅ 已更新 {len(products)} 個產品的最大賠付值")
    
    # Load spatial analysis results  
    with open("results/spatial_analysis/cat_in_circle_results.pkl", 'rb') as f:
        spatial_results = pickle.load(f)
    wind_indices_dict = spatial_results['indices']
    print("✅ Loaded spatial analysis results")
    print("🌪️  Available Cat-in-Circle indices:")
    for key in wind_indices_dict.keys():
        print(f"   • {key}: {len(wind_indices_dict[key])} events")
    
    # 注意：產品已經包含不同半徑的配置，需要根據每個產品的半徑選擇對應的風速數據
    # 這裡我們先載入所有半徑的數據以供後續使用
    print("\n📐 準備多半徑風速數據...")
    radius_wind_indices = {}
    for radius in [15, 30, 50, 75, 100]:
        key = f'cat_in_circle_{radius}km_max'
        if key in wind_indices_dict:
            radius_wind_indices[radius] = wind_indices_dict[key]
            print(f"   ✅ {radius}km半徑: {len(wind_indices_dict[key])} events")
        else:
            print(f"   ⚠️ {radius}km半徑數據不可用")
    
    # 使用50km作為預設（用於數據對齊檢查）
    default_wind_indices = radius_wind_indices.get(50, 
                          radius_wind_indices.get(30, 
                          list(radius_wind_indices.values())[0] if radius_wind_indices else np.array([])))
    
    # 載入CLIMADA數據
    print("📂 Loading CLIMADA data...")
    
    # 直接載入CLIMADA數據
    with open("results/climada_data/climada_complete_data.pkl", 'rb') as f:
        climada_data = pickle.load(f)
    
    print("✅ Successfully loaded CLIMADA data")
    
    # 提取impact數據
    impact_obj = climada_data['impact']
    observed_losses = impact_obj.at_event
    print(f"   ✅ CLIMADA損失數據: {len(observed_losses)} events")
    print(f"   損失範圍: ${np.min(observed_losses):,.0f} - ${np.max(observed_losses):,.0f}")
    print(f"   平均損失: ${np.mean(observed_losses):,.0f}")
    
    # 將CLIMADA損失解釋為醫院聚合損失
    print("🏥 將CLIMADA損失解釋為醫院聚合損失...")
    
    # 使用與02相同的方法獲取醫院數據（用於計數和配置）
    from exposure_modeling.hospital_osm_extraction import get_nc_hospitals
    
    # 使用模擬醫院數據（與02保持一致）
    gdf_hospitals_calc, _ = get_nc_hospitals(
        use_mock=True,  # 與02_spatial_analysis一致使用mock數據
        create_exposures=False,
        visualize=False
    )
    print(f"   ✅ 醫院數量: {len(gdf_hospitals_calc)}")
    
    # 檢查是否有曝險數據可用於空間分配
    if 'exposures' in climada_data and hasattr(climada_data['exposures'], 'gdf'):
        exposure_gdf = climada_data['exposures'].gdf
        print(f"   ✅ 曝險點數量: {len(exposure_gdf)}")
    
    # CLIMADA損失本身就代表區域內所有資產（包括醫院）的總損失
    print(f"   ✅ CLIMADA損失代表 {len(gdf_hospitals_calc)} 家醫院的聚合損失")
    
    # Ensure data arrays have matching lengths
    observed_losses = climada_data.get('impact').at_event if 'impact' in climada_data else np.array([])
    
    # 使用預設風速數據檢查長度對齊
    min_length = min(len(default_wind_indices), len(observed_losses))
    if min_length > 0:
        # 對所有半徑的風速數據進行截斷以確保一致性
        for radius in radius_wind_indices:
            radius_wind_indices[radius] = radius_wind_indices[radius][:min_length]
        observed_losses = observed_losses[:min_length]
        print(f"   Aligned all radius data to {min_length} events")
    else:
        print("❌ No valid data found")
        return
    
    print("\n🏥 執行醫院導向的基差風險分析...")
    print("   • 目標: 將觸發器賠付與醫院總損失匹配")
    print("   • 損失計算: 每家醫院個別損失的總和")
    print("   • 方法: 多種基差風險定義 + 多水平最大賠付測試")
    print(f"   • 使用預生成產品: {len(products)} 個")
    print(f"   • 測試最大賠付水平: 25%, 50%, 75%, 100% 總曝險")
    
    # 多水平最大賠付測試
    total_exposure = hospital_config.calculate_total_exposure()
    payout_levels = [0.25, 0.50, 0.75, 1.00]  # 25%, 50%, 75%, 100% 總曝險
    
    print(f"\n🔍 最大賠付水平測試:")
    for level in payout_levels:
        max_payout_value = total_exposure * level
        print(f"   - {level*100:3.0f}% 總曝險: ${max_payout_value:,.0f}")
    
    print(f"\n📊 開始分析...")
    print(f"   分析產品數量: {len(products)} (70個閾值函數 × 5個半徑)")
    print(f"   事件數量: {min_length}")
    print(f"   最大賠付水平: {len(payout_levels)} 個")
    print(f"   總分析組合: {len(products) * len(payout_levels)}")
    
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
    all_analysis_results = []
    
    # 檢查前3個產品的設置 (調試信息)
    print(f"\n🔍 產品設置檢查 (前3個):")
    for i in range(min(3, len(products))):
        product = products[i]
        print(f"  {product['product_id']}: 閾值={product['trigger_thresholds']}")
        print(f"    賠付比例={product['payout_ratios']}, 最大賠付=${product['max_payout']:,.0f}")
    
    # 檢查各半徑風速數據範圍
    print(f"\n🌪️  各半徑風速數據檢查:")
    for radius, wind_data in radius_wind_indices.items():
        print(f"   {radius}km半徑:")
        print(f"      範圍: {np.min(wind_data):.2f} - {np.max(wind_data):.2f} mph")
        print(f"      平均: {np.mean(wind_data):.2f}, 標準差: {np.std(wind_data):.2f}")
    
    # 為每個產品測試多個最大賠付水平
    total_combinations = len(products) * len(payout_levels)
    combination_count = 0
    
    for i, product in enumerate(products):
        # 根據產品的半徑選擇對應的風速數據
        product_radius = product.get('radius_km', 50)  # 預設50km
        if product_radius not in radius_wind_indices:
            print(f"   ⚠️ 跳過產品 {product['product_id']}: 半徑 {product_radius}km 數據不可用")
            continue
        
        # 使用該產品對應半徑的風速數據
        wind_indices = radius_wind_indices[product_radius]
        
        for payout_level in payout_levels:
            combination_count += 1
            if combination_count % 50 == 0:
                print(f"   進度: {combination_count}/{total_combinations}")
            
            # 為這個組合設定最大賠付
            current_max_payout = total_exposure * payout_level
        
            # 計算階梯式賠付 (使用整合的 skill_scores 模組)
            from skill_scores.basis_risk_functions import calculate_step_payouts_batch
            
            payouts = calculate_step_payouts_batch(
                wind_indices,  # 現在使用對應半徑的風速數據
                product['trigger_thresholds'],
                product['payout_ratios'],
                current_max_payout  # 使用當前水平的最大賠付
            )
        
            # 調試：檢查前幾個組合的賠付分佈
            if combination_count <= 6:  # 只顯示前6個組合
                print(f"    產品 {product['product_id']}, 水平{payout_level*100:.0f}%: 賠付範圍={np.min(payouts):.2e}-{np.max(payouts):.2e}, 觸發率={np.mean(payouts > 0):.3f}")
            
            # 計算各種基差風險指標
            product_result = {
                'product_id': f"{product['product_id']}_L{payout_level*100:.0f}",  # 添加水平標識
                'base_product_id': product['product_id'],
                'name': product.get('name', 'Unknown'),
                'structure_type': product['structure_type'],
                'radius_km': product.get('radius_km', 30),
                'n_thresholds': len(product['trigger_thresholds']),
                'max_payout': current_max_payout,
                'payout_level': payout_level,
                'payout_level_pct': f"{payout_level*100:.0f}%"
            }
        
            # 使用不同的基差風險計算器
            for risk_name, calculator in calculators.items():
                risk_value = calculator.calculate_basis_risk(observed_losses, payouts)
                product_result[f'{risk_name}_risk'] = risk_value
        
            # 計算額外的傳統指標
            product_result['correlation'] = np.corrcoef(observed_losses, payouts)[0,1] if np.std(payouts) > 0 else 0
            product_result['trigger_rate'] = np.mean(payouts > 0)
            product_result['mean_payout'] = np.mean(payouts)
            product_result['coverage_ratio'] = np.sum(payouts) / np.sum(observed_losses) if np.sum(observed_losses) > 0 else 0
            product_result['basis_risk_std'] = np.std(observed_losses - payouts)
            # 醫院導向指標
            product_result['hospital_match_score'] = 1 / (1 + product_result.get('weighted_risk', np.inf))  # 轉為匹配分數
            
            all_analysis_results.append(product_result)
    
    # 創建結果DataFrame
    import pandas as pd
    results_df = pd.DataFrame(all_analysis_results)
    
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
    print("\n✅ 醫院導向的基差風險分析完成！")
    print(f"📊 分析了 {len(products)} 個產品 × {len(payout_levels)} 個賠付水平 = {len(results_df)} 個組合")
    
    # Display comprehensive basis risk analysis results
    print("\n📋 基差風險分析摘要:")
    print("=" * 60)
    
    # 基本統計
    print(f"總組合數: {len(results_df)}")
    print(f"原始產品數: {len(products)}")
    print(f"測試賠付水平: {len(payout_levels)} 個 ({', '.join([f'{l*100:.0f}%' for l in payout_levels])})")
    
    print(f"\n產品結構分布:")
    structure_counts = results_df['structure_type'].value_counts()
    for structure, count in structure_counts.items():
        print(f"  • {structure.capitalize()}: {count} 組合")
    
    print(f"\n賠付水平分布:")
    payout_counts = results_df['payout_level_pct'].value_counts()
    for level, count in payout_counts.items():
        print(f"  • {level} 總曝險: {count} 組合")
    
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
    
    # 醫院導向的最佳產品建議
    print(f"\n🏥 醫院保護最佳Cat-in-Circle產品建議:")
    print("=" * 50)
    
    if 'weighted_risk' in results_df.columns:
        # 找到最佳的醫院匹配產品
        best_overall = results_df.loc[results_df['weighted_risk'].idxmin()]
        
        print(f"🏆 總體最佳產品:")
        print(f"   產品: {best_overall['base_product_id']} ({best_overall['structure_type']})")
        print(f"   最佳賠付水平: {best_overall['payout_level_pct']} 總曝險 (${best_overall['max_payout']:,.0f})")
        print(f"   加權基差風險: {best_overall['weighted_risk']:.2e}")
        print(f"   觸發率: {best_overall['trigger_rate']:.3f}")
        print(f"   覆蓋率: {best_overall['coverage_ratio']:.3f}")
        print(f"   醫院匹配分數: {best_overall['hospital_match_score']:.6f}")
        
        # 按賠付水平分組找最佳
        print(f"\n💰 各賠付水平最佳產品:")
        for level in sorted(results_df['payout_level'].unique()):
            level_data = results_df[results_df['payout_level'] == level]
            best_in_level = level_data.loc[level_data['weighted_risk'].idxmin()]
            
            print(f"   • {level*100:.0f}% 總曝險水平: {best_in_level['base_product_id']}")
            print(f"     風險: {best_in_level['weighted_risk']:.2e}, 觸發率: {best_in_level['trigger_rate']:.3f}")
        
        # 按產品結構分組找最佳
        print(f"\n🔧 各結構類型最佳產品:")
        for structure in sorted(results_df['structure_type'].unique()):
            structure_data = results_df[results_df['structure_type'] == structure]
            best_in_structure = structure_data.loc[structure_data['weighted_risk'].idxmin()]
            
            print(f"   • {structure.capitalize()}: {best_in_structure['product_id']}")
            print(f"     風險: {best_in_structure['weighted_risk']:.2e}, 賠付水平: {best_in_structure['payout_level_pct']}")
    
    # Top 10 綜合排名 (使用加權不對稱基差風險)
    print(f"\n📈 Top 10 組合排名 (按醫院匹配效果):")
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
                'overcompensation_weight': 0.5,
                'hospital_config': {
                    'n_hospitals': hospital_config.n_hospitals,
                    'base_hospital_value': hospital_config.base_hospital_value,
                    'total_exposure': total_exposure,
                    'max_payouts': max_payouts
                }
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
        f"Hospital-Based Configuration:",
        f"  - Hospitals: {hospital_config.n_hospitals}",
        f"  - Base Value per Hospital: ${hospital_config.base_hospital_value:,.0f}",
        f"  - Total Exposure: ${total_exposure:,.0f}",
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
    print("\n   📐 多半徑測試配置:")
    print(f"   • 測試半徑: 15km, 30km, 50km, 75km, 100km")
    print(f"   • 每個產品使用其對應半徑的Cat-in-Circle風速數據")
    print(f"   • Steinmann 2023標準: 70個閾值函數 × 5個半徑 = 350個產品")
    print("\n   🏥 基於醫院的賠付配置:")
    print(f"   • 醫院數量: {hospital_config.n_hospitals}")
    print(f"   • 總曝險值: ${total_exposure:,.0f}")
    print(f"   • 最大賠付已根據醫院曝險調整")
    
    return results

# %%
if __name__ == "__main__":
    results = main()
# %%
