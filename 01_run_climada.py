#!/usr/bin/env python3
"""
最小化CLIMADA數據生成
Minimal CLIMADA Data Generation

從成功的nc_tc_comprehensive_functional.py提取最核心的數據生成代碼
直接執行，不做複雜的錯誤處理
"""

print("🚀 最小化CLIMADA數據生成")

# %% 
import os
import sys
import numpy as np
from datetime import datetime

# 設置路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
insurance_dir = os.path.join(current_dir, 'insurance_analysis_refactored')

for path in [insurance_dir, current_dir]:
    if path not in sys.path:
        sys.path.append(path)

print(f"✅ 路徑設置: {current_dir}")

# %% 直接導入
print("📦 導入模組...")
from config.settings import NC_BOUNDS, YEAR_RANGE, RESOLUTION
from data_processing.track_processing import get_regional_tracks
from hazard_modeling.tc_hazard import create_tc_hazard
from exposure_modeling.litpop_processing import process_litpop_exposures
from exposure_modeling.hospital_osm_extraction import get_nc_hospitals
from impact_analysis.impact_calculation import calculate_tc_impact

print("✅ 模組導入完成")

# %% 直接複製數據生成代碼
print("\n🌪️ 準備CLIMADA真實數據...")

print("正在準備CLIMADA真實數據...")
# 獲取軌跡數據
print(f"   🌀 目標區域: North Carolina {NC_BOUNDS}")
print(f"   📅 分析期間: {YEAR_RANGE[0]}-{YEAR_RANGE[1]}")

tracks = get_regional_tracks(NC_BOUNDS, YEAR_RANGE, nb_synth=3)
print(f"   ✅ 獲取 {len(tracks.data)} 條軌跡")

# 創建災害場
from hazard_modeling.centroids import create_hazard_centroids
centroids_lat, centroids_lon = create_hazard_centroids(NC_BOUNDS, RESOLUTION)
tc_hazard = create_tc_hazard(tracks, centroids_lat, centroids_lon)
print(f"   ✅ 災害場: {tc_hazard.size} 個事件")

# 創建曝險 - 使用完整年份範圍
exposure_dict, successful_years = process_litpop_exposures(
    country_iso="USA", state_name="North Carolina", years=range(2019, 2025)
)

if successful_years:
    print(f"   ✅ 成功處理 {len(successful_years)} 年曝險數據: {successful_years}")
    
    # 計算所有年份的影響
    yearly_impacts = {}
    yearly_exposures_summary = {}
    
    for year in successful_years:
        exposure = exposure_dict[year]
        print(f"\n   📊 {year}年曝險數據:")
        print(f"      資產點數: {len(exposure.gdf):,}")
        print(f"      總曝險值: ${exposure.value.sum()/1e9:.2f}B")
        
        # 計算該年份影響
        impact, impact_func_set = calculate_tc_impact(tc_hazard, exposure)
        yearly_impacts[year] = impact
        
        yearly_exposures_summary[year] = {
            'asset_count': len(exposure.gdf),
            'total_value': exposure.value.sum(),
            'annual_average_impact': impact.aai_agg
        }
        
        print(f"      年均損失: ${impact.aai_agg/1e9:.2f}B")
    
    # 使用最新年份作為主要曝險數據
    latest_year = max(successful_years)
    exposure_main = exposure_dict[latest_year]
    impact = yearly_impacts[latest_year]
    print(f"\n   🎯 使用 {latest_year} 年作為主要曝險數據")
    
    # 輸出詳細結果統計
    print(f"\n💥 災害影響分析結果:")
    print(f"   年均總損失 (AAI): ${impact.aai_agg/1e9:.2f}B")
    print(f"   總事件損失: ${impact.at_event.sum()/1e9:.2f}B")
    print(f"   最大單次事件損失: ${impact.at_event.max()/1e9:.2f}B")
    print(f"   受影響事件數: {(impact.at_event > 0).sum()}")
    
    # 計算回歸期損失
    freq_curve = impact.calc_freq_curve()
    rp_losses = {
        10: freq_curve.impact[np.argmin(np.abs(freq_curve.return_per - 10))],
        50: freq_curve.impact[np.argmin(np.abs(freq_curve.return_per - 50))], 
        100: freq_curve.impact[np.argmin(np.abs(freq_curve.return_per - 100))],
        500: freq_curve.impact[np.argmin(np.abs(freq_curve.return_per - 500))]
    }
    
    print(f"\n📊 回歸期損失估計:")
    for rp, loss in rp_losses.items():
        print(f"   {rp}年回歸期: ${loss/1e9:.2f}B")
    
    print(f"📈 災害影響計算完成")
    
    # 準備完整多年份數據
    climada_complete_data = {
        'tc_hazard': tc_hazard,
        'exposure_main': exposure_main,  # 主要曝險數據（最新年份）
        'impact_main': impact,  # 主要影響數據（最新年份）
        'impact_func_set': impact_func_set,
        
        # 多年份數據
        'exposure_dict': exposure_dict,  # 所有年份曝險數據
        'yearly_impacts': yearly_impacts,  # 所有年份影響數據
        'successful_years': successful_years,  # 成功處理的年份
        'yearly_exposures_summary': yearly_exposures_summary,  # 年份摘要統計
        
        # 向後兼容
        'exposure': exposure_main,
        'impact': impact,
        'event_losses': impact.at_event,
        'exposure_locations': [(lat, lon) for lat, lon in zip(exposure_main.latitude, exposure_main.longitude)],
        
        'metadata': {
            'n_events': tc_hazard.size,
            'total_exposure_latest': exposure_main.value.sum(),
            'annual_average_impact_latest': impact.aai_agg,
            'latest_year': latest_year,
            'successful_years': successful_years,
            'n_years_processed': len(successful_years),
            'generation_time': datetime.now().isoformat()
        }
    }
    print("   🎉 CLIMADA真實數據準備完成！")
    
    # 保存到pickle
    import pickle
    with open('climada_complete_data.pkl', 'wb') as f:
        pickle.dump(climada_complete_data, f)
    print("💾 數據已保存到 climada_complete_data.pkl")
    
else:
    print("   ❌ 無法創建曝險數據")
    raise ValueError("無法創建曝險數據")

print("\n🎊 完成！CLIMADA完整多年份數據對象已生成")
print("📋 數據結構說明：")
print("  - exposure_main: 主要曝險數據（最新年份）")
print("  - exposure_dict: 所有年份曝險數據字典")
print("  - yearly_impacts: 所有年份影響計算結果")
print("  - successful_years: 成功處理的年份列表")
print("\n💻 使用方式：")
print("  import pickle")
print("  with open('climada_complete_data.pkl', 'rb') as f:")
print("      data = pickle.load(f)")
print("  # 訪問特定年份: data['exposure_dict'][2023]")
print("  # 查看所有年份: data['successful_years']")
# %%
