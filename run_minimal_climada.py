#!/usr/bin/env python3
"""
最小化CLIMADA數據生成
Minimal CLIMADA Data Generation

從成功的nc_tc_comprehensive_functional.py提取最核心的數據生成代碼
直接執行，不做複雜的錯誤處理
"""

print("🚀 最小化CLIMADA數據生成")

# %% 直接複製成功腳本的導入和設置
import os
import sys
import numpy as np
from datetime import datetime

# 設置路徑 (複製自成功腳本)
current_dir = os.path.dirname(os.path.abspath(__file__))
insurance_dir = os.path.join(current_dir, 'insurance_analysis_refactored')

for path in [insurance_dir, current_dir]:
    if path not in sys.path:
        sys.path.append(path)

print(f"✅ 路徑設置: {current_dir}")

# %% 直接導入 (複製自成功腳本)
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

# 創建曝險
exposure_dict, successful_years = process_litpop_exposures(
    country_iso="USA", state_name="North Carolina", years=[2020, 2019]
)

if successful_years:
    exposure_main = exposure_dict[successful_years[0]]
    print(f"   ✅ 曝險數據: {len(exposure_main.gdf)} 個點")
    print(f"   💰 總曝險值: ${exposure_main.value.sum()/1e9:.2f}B")
    
    # 計算影響
    impact, impact_func_set = calculate_tc_impact(tc_hazard, exposure_main)
    
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
    
    # 準備數據
    climada_complete_data = {
        'tc_hazard': tc_hazard,
        'exposure': exposure_main,
        'impact': impact,
        'impact_func_set': impact_func_set,  # 添加影響函數集
        'event_losses': impact.at_event,
        'exposure_locations': [(lat, lon) for lat, lon in zip(exposure_main.latitude, exposure_main.longitude)],
        'metadata': {
            'n_events': tc_hazard.size,
            'total_exposure': exposure_main.value.sum(),
            'annual_average_impact': impact.aai_agg,
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

print("\n🎊 完成！CLIMADA完整數據對象已生成")
print("可通過以下方式使用：")
print("  import pickle")
print("  with open('climada_complete_data.pkl', 'rb') as f:")
print("      data = pickle.load(f)")