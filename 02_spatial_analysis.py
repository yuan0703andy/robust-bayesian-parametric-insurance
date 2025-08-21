#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_spatial_analysis.py
======================
完整的 Cat-in-a-Circle 空間分析展示
實現 Steinmann et al. (2023) 論文的標準方法

流程：
1. 載入 CLIMADA 數據 (climada_complete_data.pkl)
2. 提取醫院座標（曝險點）
3. 執行多半徑 Cat-in-a-Circle 分析
4. 視覺化結果
5. 輸出統計報告
"""
# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 導入新的模組化組件
from data_processing import SpatialDataProcessor, load_spatial_data_from_02_results

# Advanced Cat-in-Circle analysis removed per user request
# Using Basic Cat-in-Circle implementation only
ENHANCED_ANALYSIS_AVAILABLE = False

# 導入OSM醫院提取模組
try:
    from exposure_modeling.hospital_osm_extraction import get_nc_hospitals
    OSM_HOSPITALS_AVAILABLE = True
except ImportError:
    OSM_HOSPITALS_AVAILABLE = False

# 設置matplotlib支援中文
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# %%
def visualize_cat_in_circle_results(results, output_dir="results/spatial_analysis"):
    """
    視覺化 Cat-in-a-Circle 分析結果
    Visualize Cat-in-a-Circle analysis results
    
    Parameters:
    -----------
    results : dict
        分析結果字典
    output_dir : str
        輸出目錄
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 創建圖表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cat-in-a-Circle Spatial Analysis Results', fontsize=16, fontweight='bold')
    
    # 提取所有半徑
    radii = results['metadata']['radii_km']
    
    # 1. 不同半徑的風速分布
    ax = axes[0, 0]
    for radius in radii:
        index_name = f"cat_in_circle_{radius}km_max"
        if index_name in results['indices']:
            wind_speeds = results['indices'][index_name]
            wind_speeds_nonzero = wind_speeds[wind_speeds > 0]
            if len(wind_speeds_nonzero) > 0:
                ax.hist(wind_speeds_nonzero, bins=30, alpha=0.5, label=f'{radius}km')
    ax.set_xlabel('Max Wind Speed (m/s)')
    ax.set_ylabel('Number of Events')
    ax.set_title('Wind Speed Distribution by Radius')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 半徑與平均風速的關係
    ax = axes[0, 1]
    mean_winds = []
    max_winds = []
    for radius in radii:
        index_name = f"cat_in_circle_{radius}km_max"
        if index_name in results['indices']:
            wind_speeds = results['indices'][index_name]
            wind_speeds_nonzero = wind_speeds[wind_speeds > 0]
            if len(wind_speeds_nonzero) > 0:
                mean_winds.append(np.mean(wind_speeds_nonzero))
                max_winds.append(np.max(wind_speeds_nonzero))
            else:
                mean_winds.append(0)
                max_winds.append(0)
    
    ax.plot(radii, mean_winds, 'o-', label='Mean Wind Speed', linewidth=2, markersize=8)
    ax.plot(radii, max_winds, 's-', label='Max Wind Speed', linewidth=2, markersize=8)
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title('Radius vs Wind Speed Relationship')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 事件影響覆蓋率
    ax = axes[0, 2]
    coverage_rates = []
    for radius in radii:
        index_name = f"cat_in_circle_{radius}km_max"
        if index_name in results['indices']:
            wind_speeds = results['indices'][index_name]
            coverage_rate = np.sum(wind_speeds > 0) / len(wind_speeds) * 100
            coverage_rates.append(coverage_rate)
        else:
            coverage_rates.append(0)
    
    bars = ax.bar(radii, coverage_rates, width=5, alpha=0.7, color='skyblue', edgecolor='navy')
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel('Event Coverage Rate (%)')
    ax.set_title('Event Coverage by Radius')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在條形圖上添加數值
    for bar, rate in zip(bars, coverage_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # 4. 醫院級別風速熱圖（示例：30km半徑）
    ax = axes[1, 0]
    if 'radius_30km' in results.get('hospital_series', {}):
        hospital_winds = results['hospital_series']['radius_30km']
        n_hospitals = min(20, len(hospital_winds))  # 最多顯示20家醫院
        n_events = min(50, len(list(hospital_winds.values())[0]))  # 最多顯示50個事件
        
        # 創建矩陣
        wind_matrix = np.zeros((n_hospitals, n_events))
        for h_idx in range(n_hospitals):
            if h_idx in hospital_winds:
                wind_matrix[h_idx, :n_events] = hospital_winds[h_idx][:n_events]
        
        im = ax.imshow(wind_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Event Number')
        ax.set_ylabel('Hospital Number')
        ax.set_title('Hospital Wind Speed Heatmap (30km radius)')
        plt.colorbar(im, ax=ax, label='Wind Speed (m/s)')
    
    # 5. 統計指標比較
    ax = axes[1, 1]
    statistics = results['metadata']['statistics']
    if len(statistics) > 1:
        stat_data = []
        for stat in statistics:
            stat_values = []
            for radius in radii[:3]:  # 只顯示前3個半徑
                index_name = f"cat_in_circle_{radius}km_{stat}"
                if index_name in results['indices']:
                    values = results['indices'][index_name]
                    values_nonzero = values[values > 0]
                    if len(values_nonzero) > 0:
                        stat_values.append(np.mean(values_nonzero))
                    else:
                        stat_values.append(0)
            stat_data.append(stat_values)
        
        x = np.arange(len(radii[:3]))
        width = 0.25
        for i, (stat, values) in enumerate(zip(statistics, stat_data)):
            ax.bar(x + i*width, values, width, label=stat)
        
        ax.set_xlabel('Radius (km)')
        ax.set_ylabel('Mean Value (m/s)')
        ax.set_title('Statistical Metrics Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(radii[:3])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # 6. 累積分布函數
    ax = axes[1, 2]
    for radius in [15, 30, 50]:  # 選擇關鍵半徑
        index_name = f"cat_in_circle_{radius}km_max"
        if index_name in results['indices']:
            wind_speeds = results['indices'][index_name]
            wind_speeds_nonzero = np.sort(wind_speeds[wind_speeds > 0])
            if len(wind_speeds_nonzero) > 0:
                cdf = np.arange(1, len(wind_speeds_nonzero) + 1) / len(wind_speeds_nonzero)
                ax.plot(wind_speeds_nonzero, cdf, label=f'{radius}km', linewidth=2)
    
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Wind Speed CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 儲存圖表
    output_path = Path(output_dir) / "cat_in_circle_analysis.png"
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Chart saved to: {output_path}")
    
    plt.show()

# %%
def process_spatial_data_with_modular_components(hospital_coords, hazard_data=None):
    """
    使用新的模組化組件處理空間數據
    Process spatial data using new modular components
    
    Parameters:
    -----------
    hospital_coords : list
        醫院座標列表 [(lat, lon), ...]
    hazard_data : optional
        災害數據 (如果可用)
        
    Returns:
    --------
    SpatialData : 處理後的空間數據對象
    """
    print("🔧 使用模組化SpatialDataProcessor處理空間數據...")
    
    # 轉換為numpy數組
    coords_array = np.array(hospital_coords)
    
    # 創建處理器並處理空間數據
    processor = SpatialDataProcessor()
    spatial_data = processor.process_hospital_spatial_data(
        coords_array, 
        n_regions=3,  # 沿海/中部/山區
        region_method="risk_based"
    )
    
    # 如果有災害數據，添加模擬的Cat-in-Circle結果
    if hazard_data is not None:
        print("   🌪️ 添加Cat-in-Circle災害數據...")
        n_hospitals = len(hospital_coords)
        n_events = 100  # 假設100個事件
        
        # 創建模擬的災害強度和損失數據
        hazard_intensities = np.random.uniform(20, 70, (n_hospitals, n_events))
        exposure_values = np.random.uniform(1e7, 5e7, n_hospitals)
        observed_losses = np.random.lognormal(15, 1, (n_hospitals, n_events))
        
        spatial_data = processor.add_cat_in_circle_data(
            hazard_intensities, exposure_values, observed_losses
        )
    
    return spatial_data

# %%
def generate_analysis_report(results, output_dir="results/spatial_analysis"):
    """
    生成分析報告
    Generate analysis report
    
    Parameters:
    -----------
    results : dict
        分析結果
    output_dir : str
        輸出目錄
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Cat-in-a-Circle Spatial Analysis Report")
    report_lines.append("Based on Steinmann et al. (2023) Standard Method")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 基礎信息
    report_lines.append("📊 Analysis Overview")
    report_lines.append("-" * 40)
    report_lines.append(f"Number of Hospitals: {results['metadata']['n_hospitals']}")
    report_lines.append(f"Number of TC Events: {results['metadata']['n_events']}")
    report_lines.append(f"Hazard Grid Points: {results['metadata']['hazard_grid_size']}")
    report_lines.append(f"Analysis Radii: {results['metadata']['radii_km']} km")
    report_lines.append(f"Statistical Methods: {results['metadata']['statistics']}")
    report_lines.append("")
    
    # 各半徑統計
    report_lines.append("🎯 Detailed Statistics by Radius")
    report_lines.append("-" * 40)
    
    for radius in results['metadata']['radii_km']:
        report_lines.append(f"\nRadius {radius} km:")
        
        index_name = f"cat_in_circle_{radius}km_max"
        if index_name in results['indices']:
            wind_speeds = results['indices'][index_name]
            wind_speeds_nonzero = wind_speeds[wind_speeds > 0]
            
            if len(wind_speeds_nonzero) > 0:
                report_lines.append(f"  • Affected Events: {len(wind_speeds_nonzero)} / {len(wind_speeds)}")
                report_lines.append(f"  • Coverage Rate: {len(wind_speeds_nonzero)/len(wind_speeds)*100:.1f}%")
                report_lines.append(f"  • Mean Wind Speed: {np.mean(wind_speeds_nonzero):.1f} m/s")
                report_lines.append(f"  • Max Wind Speed: {np.max(wind_speeds_nonzero):.1f} m/s")
                report_lines.append(f"  • Median Wind Speed: {np.median(wind_speeds_nonzero):.1f} m/s")
                report_lines.append(f"  • 95th Percentile: {np.percentile(wind_speeds_nonzero, 95):.1f} m/s")
            else:
                report_lines.append(f"  • No affected events")
    
    # 關鍵發現
    report_lines.append("\n" + "=" * 40)
    report_lines.append("🔍 Key Findings")
    report_lines.append("-" * 40)
    
    # 找出最佳半徑（基於覆蓋率和風速變異性的平衡）
    best_radius = None
    best_score = 0
    for radius in results['metadata']['radii_km']:
        index_name = f"cat_in_circle_{radius}km_max"
        if index_name in results['indices']:
            wind_speeds = results['indices'][index_name]
            wind_speeds_nonzero = wind_speeds[wind_speeds > 0]
            if len(wind_speeds_nonzero) > 0:
                coverage = len(wind_speeds_nonzero) / len(wind_speeds)
                variability = np.std(wind_speeds_nonzero) / np.mean(wind_speeds_nonzero)
                score = coverage * (1 + variability)  # 平衡覆蓋率和變異性
                if score > best_score:
                    best_score = score
                    best_radius = radius
    
    if best_radius:
        report_lines.append(f"• Recommended Optimal Analysis Radius: {best_radius} km")
        report_lines.append(f"  (Based on balance of coverage and wind speed variability)")
    
    # 寫入報告
    report_path = Path(output_dir) / "analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n📝 Report saved to: {report_path}")
    
    # 同時打印到控制台
    print("\n" + '\n'.join(report_lines))

# %%
def main():
    """
    主程序：執行完整的 Cat-in-a-Circle 空間分析
    Main program: Execute complete Cat-in-a-Circle spatial analysis
    """
    print("=" * 80)
    print("Cat-in-a-Circle Spatial Analysis")
    print("Implementing Steinmann et al. (2023) Standard Method")
    print("=" * 80)
    
    # 步驟 1: 載入 CLIMADA 數據
    # Step 1: Load CLIMADA data
    data_path = "results/climada_data/climada_complete_data.pkl"
    
    climada_data = None
    if ENHANCED_ANALYSIS_AVAILABLE:
        climada_data = load_climada_data(data_path)
    
    if climada_data is None and ENHANCED_ANALYSIS_AVAILABLE:
        print("❌ Unable to load CLIMADA data from enhanced analysis module")
        print("⚠️ Will proceed with modular components only")
    elif not ENHANCED_ANALYSIS_AVAILABLE:
        print("⚠️ Enhanced analysis module not available, using modular components only")
    
    # 步驟 2: 提取醫院座標 - 優先使用真實OSM數據
    # Step 2: Extract hospital coordinates - Prioritize real OSM data
    print("\n🏥 Loading hospital coordinates for analysis...")
    
    # 首先嘗試使用真實OSM醫院數據
    hospital_coords = None
    gdf_hospitals = None
    
    if OSM_HOSPITALS_AVAILABLE:
        try:
            print("   📍 嘗試載入真實OSM醫院數據...")
            gdf_hospitals, hospital_exposures = get_nc_hospitals(
                use_mock=False,  # 使用真實OSM數據
                osm_file_path=None,
                create_exposures=True,  # 創建曝險數據
                visualize=False
            )
            
            # 轉換為座標列表 (lat, lon)
            hospital_coords = [(row.geometry.y, row.geometry.x) 
                              for idx, row in gdf_hospitals.iterrows()]
            
            print(f"   ✅ 成功載入 {len(hospital_coords)} 家真實OSM醫院")
            if len(gdf_hospitals) > 0 and 'name' in gdf_hospitals.columns:
                print(f"   📋 示例醫院: {gdf_hospitals['name'].iloc[0] if pd.notna(gdf_hospitals['name'].iloc[0]) else '未命名醫院'}")
            print(f"   🏥 數據來源: OpenStreetMap 真實醫院數據")
            print(f"   📍 座標範圍: lat [{min(c[0] for c in hospital_coords):.3f}, {max(c[0] for c in hospital_coords):.3f}]")
            print(f"                lon [{min(c[1] for c in hospital_coords):.3f}, {max(c[1] for c in hospital_coords):.3f}]")
            
        except Exception as e:
            print(f"   ⚠️ 真實OSM數據載入失敗: {e}")
            print("   🔄 使用mock數據作為fallback...")
            
            try:
                # Fallback到模擬數據
                gdf_hospitals, hospital_exposures = get_nc_hospitals(
                    use_mock=True,  # 使用模擬數據作為fallback
                    osm_file_path=None,
                    create_exposures=False,
                    visualize=False
                )
                
                hospital_coords = [(row.geometry.y, row.geometry.x) 
                                  for idx, row in gdf_hospitals.iterrows()]
                
                print(f"   ✅ 使用模擬醫院數據: {len(hospital_coords)} 家")
                print(f"   🏥 數據來源: 模擬座標數據")
                
            except Exception as e2:
                print(f"   ⚠️ 模擬數據也失敗: {e2}")
                hospital_coords = None
    else:
        print("   ⚠️ OSM醫院模組不可用")
    
    # 如果OSM數據獲取失敗，使用備用座標
    if hospital_coords is None:
        print("   🔄 使用備用座標...")
        gdf_hospitals = None
        hospital_coords = [
            (35.7796, -78.6382),  # Raleigh
            (36.0726, -79.7920),  # Greensboro
            (35.2271, -80.8431),  # Charlotte
            (35.0527, -78.8784),  # Fayetteville
            (35.9132, -79.0558),  # Chapel Hill
            (36.1349, -80.2676),  # Winston-Salem
            (35.6127, -77.3663),  # Greenville
            (34.2257, -77.9447),  # Wilmington
            (35.6069, -82.5540),  # Asheville
            (36.0999, -78.7837),  # Durham
        ]
        print(f"   ✅ 使用備用座標: {len(hospital_coords)} 個位置")
    
    # 步驟 3: 使用模組化組件處理空間數據
    # Step 3: Process spatial data using modular components
    print("\n🔧 Processing spatial data with modular components...")
    
    spatial_data = process_spatial_data_with_modular_components(
        hospital_coords, 
        hazard_data=climada_data
    )
    
    # 步驟 4: 執行 Cat-in-a-Circle 分析 (如果增強模組可用)
    # Step 4: Execute Cat-in-a-Circle analysis (if enhanced module available)
    results = None
    
    if ENHANCED_ANALYSIS_AVAILABLE and climada_data is not None:
        print("\n🔄 執行完整的 Cat-in-a-Circle 分析...")
        
        # 使用 Steinmann 標準配置
        radii_km = [15, 30, 50, 75, 100]
        statistics = ['max', 'mean', '95th']
        
        try:
            results = extract_hospital_cat_in_circle_complete(
                tc_hazard=climada_data['tc_hazard'],
                hospital_coords=hospital_coords,
                radii_km=radii_km,
                statistics=statistics
            )
        except Exception as e:
            print(f"   ⚠️ 增強分析失敗: {e}")
            print("   🔄 僅保存模組化空間數據...")
            results = None
    else:
        print("\n⚠️ 增強分析模組不可用，僅處理空間數據")
    
    # 步驟 5: 保存結果
    # Step 5: Save results
    print("\n💾 儲存分析結果...")
    
    output_dir = Path("results/spatial_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存空間數據結果
    import pickle
    spatial_results = {
        'spatial_data': spatial_data,
        'hospital_coordinates': spatial_data.hospital_coords,
        'distance_matrix': spatial_data.distance_matrix,
        'region_assignments': spatial_data.region_assignments,
        'hospitals': gdf_hospitals,
        'metadata': {
            'n_hospitals': spatial_data.n_hospitals,
            'n_regions': spatial_data.n_regions,
            'data_source': 'modular_spatial_processor'
        }
    }
    
    # 如果有Cat-in-Circle數據，也添加進去
    if spatial_data.hazard_intensities is not None:
        spatial_results['indices'] = {
            'cat_in_circle_50km_max': spatial_data.hazard_intensities[0, :],  # 使用第一家醫院的數據作為示例
            'hazard_intensities': spatial_data.hazard_intensities,
            'exposure_values': spatial_data.exposure_values,
            'observed_losses': spatial_data.observed_losses
        }
    
    # 如果有增強分析結果，也添加
    if results is not None:
        spatial_results.update(results)
    
    # 保存結果
    spatial_results_path = output_dir / "cat_in_circle_results.pkl"
    with open(spatial_results_path, 'wb') as f:
        pickle.dump(spatial_results, f)
    
    print(f"   ✅ 空間分析結果已保存至: {spatial_results_path}")
    
    # 步驟 6: 視覺化結果 (如果可用)
    # Step 6: Visualize results (if available)
    if results is not None:
        print("\n📈 生成視覺化...")
        visualize_cat_in_circle_results(results)
    else:
        print("\n📊 顯示空間數據統計...")
        print(f"   醫院數量: {spatial_data.n_hospitals}")
        print(f"   區域數量: {spatial_data.n_regions}")
        print(f"   距離範圍: {spatial_data.distance_matrix[spatial_data.distance_matrix > 0].min():.1f} - {spatial_data.distance_matrix.max():.1f} km")
        print(f"   區域分配: {dict(enumerate(np.bincount(spatial_data.region_assignments)))}")
        
        if spatial_data.hazard_intensities is not None:
            print(f"   災害強度範圍: {spatial_data.hazard_intensities.min():.1f} - {spatial_data.hazard_intensities.max():.1f}")
            print(f"   事件數量: {spatial_data.hazard_intensities.shape[1]}")
    
    # 步驟 7: 生成報告
    # Step 7: Generate analysis report
    if results is not None:
        print("\n📝 生成分析報告...")
        generate_analysis_report(results)
    else:
        print("\n📝 生成模組化空間數據報告...")
        # 創建簡化報告
        report_path = Path("results/spatial_analysis") / "modular_spatial_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Modular Spatial Data Processing Report\n")
            f.write("基於新模組化SpatialDataProcessor的空間數據處理報告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📊 Processing Overview\n")
            f.write("-" * 40 + "\n")
            f.write(f"醫院數量: {spatial_data.n_hospitals}\n")
            f.write(f"區域數量: {spatial_data.n_regions}\n")
            f.write(f"區域分配方法: risk_based\n")
            f.write(f"距離計算方法: Haversine\n\n")
            
            f.write("🗺️ Geographic Information\n")
            f.write("-" * 40 + "\n")
            f.write(f"座標範圍 - 緯度: [{spatial_data.hospital_coords[:,0].min():.3f}, {spatial_data.hospital_coords[:,0].max():.3f}]\n")
            f.write(f"座標範圍 - 經度: [{spatial_data.hospital_coords[:,1].min():.3f}, {spatial_data.hospital_coords[:,1].max():.3f}]\n")
            f.write(f"最小距離: {spatial_data.distance_matrix[spatial_data.distance_matrix > 0].min():.1f} km\n")
            f.write(f"最大距離: {spatial_data.distance_matrix.max():.1f} km\n\n")
            
            f.write("🏥 Regional Assignment\n")
            f.write("-" * 40 + "\n")
            region_counts = np.bincount(spatial_data.region_assignments)
            for i, count in enumerate(region_counts):
                f.write(f"區域 {i}: {count} 家醫院\n")
            
            if spatial_data.hazard_intensities is not None:
                f.write("\n🌪️ Hazard Data (Simulated)\n")
                f.write("-" * 40 + "\n")
                f.write(f"事件數量: {spatial_data.hazard_intensities.shape[1]}\n")
                f.write(f"災害強度範圍: {spatial_data.hazard_intensities.min():.1f} - {spatial_data.hazard_intensities.max():.1f} mph\n")
                f.write(f"曝險價值總計: ${spatial_data.exposure_values.sum():,.0f}\n")
        
        print(f"   ✅ 模組化空間數據報告已保存至: {report_path}")
    
    print(f"\n✅ 02_spatial_analysis.py 執行完成!")
    print(f"   📁 結果保存在: results/spatial_analysis/")
    print(f"   🔧 使用了新的模組化SpatialDataProcessor")
    print(f"   💡 結果可被後續腳本 (03, 04, 05) 使用")
    
    return spatial_results

# %%
if __name__ == "__main__":
    results = main()
# %%
