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

# 導入增強的空間分析模組
from insurance_analysis_refactored.core.enhanced_spatial_analysis import (
    extract_hospital_cat_in_circle_complete,
    load_climada_data,
    extract_hospitals_from_exposure,
    EnhancedCatInCircleAnalyzer,
    create_standard_steinmann_config
)

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
    data_path = "climada_complete_data.pkl"
    climada_data = load_climada_data(data_path)
    
    if climada_data is None:
        print("❌ Unable to load data, please confirm file exists")
        return
    
    # 步驟 2: 提取醫院座標
    # Step 2: Extract hospital coordinates
    print("\n🏥 Extracting hospital coordinates...")
    
    # 方法1: 從曝險數據中提取高價值點作為醫院代理
    # Method 1: Extract high-value points from exposure as hospital proxies
    if 'exposure' in climada_data:
        hospital_coords = extract_hospitals_from_exposure(climada_data['exposure'])
    else:
        # 方法2: 如果有預存的醫院座標
        # Method 2: Use pre-stored hospital coordinates
        print("   ⚠️ Using example hospital coordinates")
        hospital_coords = [
            (35.7796, -78.6382),  # Raleigh
            (36.0726, -79.7920),  # Greensboro
            (35.2271, -80.8431),  # Charlotte
            (35.0527, -78.8784),  # Fayetteville
            (35.9132, -79.0558),  # Chapel Hill
        ]
    
    # 步驟 3: 執行完整的 Cat-in-a-Circle 分析
    # Step 3: Execute complete Cat-in-a-Circle analysis
    print("\n🔄 Executing Cat-in-a-Circle analysis...")
    
    # 使用 Steinmann 標準配置
    # Use Steinmann standard configuration
    radii_km = [15, 30, 50, 75, 100]
    statistics = ['max', 'mean', '95th']
    
    results = extract_hospital_cat_in_circle_complete(
        tc_hazard=climada_data['tc_hazard'],
        hospital_coords=hospital_coords,
        radii_km=radii_km,
        statistics=statistics
    )
    
    # 步驟 4: 視覺化結果
    # Step 4: Visualize results
    print("\n📈 Generating visualizations...")
    visualize_cat_in_circle_results(results)
    
    # 步驟 5: 生成報告
    # Step 5: Generate report
    print("\n📝 Generating analysis report...")
    generate_analysis_report(results)
    
    # 步驟 6: 儲存結果供後續使用
    # Step 6: Save results for later use
    import pickle
    results_path = "results/spatial_analysis/cat_in_circle_results.pkl"
    Path("results/spatial_analysis").mkdir(parents=True, exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Results saved to: {results_path}")
    
    print("\n✅ Analysis complete!")
    
    # 顯示摘要
    # Display summary
    print("\n" + "=" * 40)
    print("📊 Analysis Summary")
    print("-" * 40)
    print(f"• Analyzed {len(hospital_coords)} hospital locations")
    print(f"• Used {len(radii_km)} different radii")
    print(f"• Generated {len(results['indices'])} parametric indices")
    print(f"• Covered {results['metadata']['n_events']} TC events")
    
    return results

# %%
if __name__ == "__main__":
    results = main()
# %%
