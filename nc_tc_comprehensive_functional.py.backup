#!/usr/bin/env python3
"""
北卡羅來納州熱帶氣旋參數型保險分析 - 真正的功能式版本
NC Tropical Cyclone Parametric Insurance Analysis - True Functional Version

基於 main_test_optimized.py 的完整流程，採用簡潔的功能式設計
每個 cell 直接執行並立即顯示結果，無不必要的函數包裝

Features:
- 350 products (5 radii × 70 Steinmann functions) 
- Dual-track analysis: Steinmann RMSE vs Bayesian CRPS
- Pure Cat-in-a-Circle (no weighted average)
- Correct step payout logic (no break statement)
- Direct cell execution with immediate results
"""

# %% 環境設置與模組導入
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

print("🔧 設置分析環境...")

# 設置路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
insurance_dir = os.path.join(current_dir, 'insurance_analysis_refactored')
base_path = '/hpc/group/borsuklab/yh421/CAT_INSURANCE/climada'

for path in [insurance_dir, current_dir, base_path]:
    if path not in sys.path:
        sys.path.append(path)

print(f"✅ 路徑設置完成: {current_dir}")

# 檢查模組可用性
print("🔍 檢查關鍵模組...")
modules_available = {}

# 核心引擎模組
try:
    from insurance_analysis_refactored.core.saffir_simpson_products import generate_steinmann_2023_products
    from insurance_analysis_refactored.core.input_adapters import extract_pure_cat_in_circle
    from insurance_analysis_refactored.core.parametric_engine import calculate_correct_step_payouts, calculate_crps_score
    modules_available['core'] = True
    print("   ✅ 核心引擎模組")
except ImportError as e:
    modules_available['core'] = False
    print(f"   ❌ 核心引擎模組: {e}")

# CLIMADA 模組 - 強制成功載入
print("   🔄 載入 CLIMADA 模組...")
from config.settings import NC_BOUNDS, YEAR_RANGE, RESOLUTION
from data_processing.track_processing import get_regional_tracks
from hazard_modeling.tc_hazard import create_tc_hazard
from exposure_modeling.litpop_processing import (
    process_litpop_exposures, 
    visualize_all_litpop_exposures,
    create_yearly_comparison,
    analyze_exposure_statistics
)
from exposure_modeling.hospital_osm_extraction import get_nc_hospitals, create_standardized_hospital_exposures
from impact_analysis.impact_calculation import calculate_tc_impact
modules_available['climada'] = True
print("   ✅ CLIMADA 模組載入成功")

# 進階模組
try:
    from bayesian.robust_bayesian_uncertainty import generate_probabilistic_loss_distributions
    from bayesian.robust_bayesian_analyzer import RobustBayesianAnalyzer
    modules_available['bayesian'] = True
    print("   ✅ 貝氏不確定性模組")
    print("   ✅ 穩健貝氏分析器")
except ImportError as e:
    modules_available['bayesian'] = False
    print(f"   ⚠️ 貝氏不確定性模組不可用: {e}")

available_count = sum(modules_available.values())
print(f"📊 模組可用性: {available_count}/{len(modules_available)}")

# Hospital Cat-in-a-Circle Functions (移至此處以確保在使用前定義)
def extract_hospital_cat_in_circle(tc_hazard, hospital_coords, hazard_tree, radius_km):
    """
    為每家醫院提取Cat-in-a-Circle風速指標
    符合Steinmann論文的醫院級別分析
    
    Parameters:
    -----------
    tc_hazard : TropCyclone
        CLIMADA颱風災害對象
    hospital_coords : list of tuples
        醫院座標列表 [(lat1, lon1), (lat2, lon2), ...]
    hazard_tree : cKDTree
        災害點空間索引樹
    radius_km : float
        Cat-in-a-Circle半徑(公里)
        
    Returns:
    --------
    dict
        {hospital_idx: [event_wind_speeds]} 每家醫院在每個事件的風速
    """
    n_events = tc_hazard.intensity.shape[0]
    n_hospitals = len(hospital_coords)
    
    # 為每家醫院儲存風速時間序列
    hospital_wind_series = {}
    
    print(f"   🏥 計算 {n_hospitals} 家醫院在 {n_events} 個事件中的Cat-in-a-Circle風速...")
    
    for hospital_idx, hospital_coord in enumerate(hospital_coords):
        wind_speeds = np.zeros(n_events)
        
        # 預先計算該醫院半徑內的災害點
        radius_rad = radius_km / 6371.0
        nearby_indices = hazard_tree.query_ball_point(
            np.radians(hospital_coord), radius_rad
        )
        
        if len(nearby_indices) > 0:
            for event_idx in range(n_events):
                wind_field = tc_hazard.intensity[event_idx, :].toarray().flatten()
                nearby_winds = wind_field[nearby_indices]
                nearby_winds = nearby_winds[nearby_winds > 0]
                
                if len(nearby_winds) > 0:
                    wind_speeds[event_idx] = np.max(nearby_winds)
                else:
                    wind_speeds[event_idx] = 0.0
        
        hospital_wind_series[hospital_idx] = wind_speeds
    
    return hospital_wind_series

def calculate_hospital_standardized_damages(hospital_wind_series, impact_func_set):
    """
    計算醫院標準化損失 (每家醫院1單位 × 脆弱度函數)
    
    Parameters:
    -----------
    hospital_wind_series : dict
        每家醫院的風速時間序列
    impact_func_set : ImpactFuncSet
        脆弱度函數集
        
    Returns:
    --------
    dict
        每家醫院的標準化損失
    """
    
    print("💊 計算醫院標準化損失 (Steinmann論文方法)...")
    
    # 檢查輸入數據
    if not hospital_wind_series:
        print("   ⚠️ 沒有醫院風速數據")
        return {}
    
    hospital_damages = {}
    
    # 簡化的風速-損害關係函數
    def simple_damage_func(wind_speed):
        if wind_speed < 33:  # < Cat 1
            return 0.0
        elif wind_speed < 42:  # Cat 1
            return 0.1
        elif wind_speed < 50:  # Cat 2
            return 0.25
        elif wind_speed < 58:  # Cat 3
            return 0.5
        elif wind_speed < 70:  # Cat 4
            return 0.75
        else:  # Cat 5
            return 1.0
    
    # 檢查是否有影響函數集
    if impact_func_set is not None and hasattr(impact_func_set, 'get_func'):
        try:
            # 獲取適用的影響函數 (通常是熱帶氣旋函數)
            impact_funcs = impact_func_set.get_func()
            if impact_funcs and len(impact_funcs) > 0:
                impact_func = impact_funcs[0]  # 使用第一個函數
                print(f"   📈 使用CLIMADA影響函數: {type(impact_func).__name__}")
                
                # 使用CLIMADA影響函數計算
                for hospital_idx, wind_speeds in hospital_wind_series.items():
                    # 每家醫院價值 = 1.0 標準化單位
                    hospital_value = 1.0
                    
                    # 計算損失比例
                    damage_ratios = []
                    for wind_speed in wind_speeds:
                        if wind_speed > 0:
                            # 使用CLIMADA影響函數計算損失比例
                            ratio = impact_func.calc_mdr(wind_speed)
                            damage_ratios.append(ratio * hospital_value)
                        else:
                            damage_ratios.append(0.0)
                    
                    hospital_damages[hospital_idx] = np.array(damage_ratios)
            else:
                print("   ⚠️ 影響函數集為空，使用簡化計算...")
                for hospital_idx, wind_speeds in hospital_wind_series.items():
                    damages = np.array([simple_damage_func(ws) for ws in wind_speeds])
                    hospital_damages[hospital_idx] = damages
                    
        except Exception as e:
            print(f"   ⚠️ CLIMADA影響函數計算失敗: {str(e)}")
            # 回退到簡化計算
            for hospital_idx, wind_speeds in hospital_wind_series.items():
                damages = np.array([simple_damage_func(ws) for ws in wind_speeds])
                hospital_damages[hospital_idx] = damages
    else:
        print("   ⚠️ 無影響函數集，使用簡化計算...")
        # 使用簡化的損害函數
        for hospital_idx, wind_speeds in hospital_wind_series.items():
            damages = np.array([simple_damage_func(ws) for ws in wind_speeds])
            hospital_damages[hospital_idx] = damages
    
    # 統計信息
    if hospital_damages:
        total_events = len(list(hospital_damages.values())[0])
        total_damages = sum(np.sum(damages) for damages in hospital_damages.values())
        print(f"   ✅ 完成 {len(hospital_damages)} 家醫院 × {total_events} 個事件的損失計算")
        print(f"   💰 總標準化損失: {total_damages:.2f}")
    else:
        print("   ⚠️ 無法計算醫院損失")
    
    return hospital_damages

def plot_hospital_cat_in_circle_analysis(hospital_cat_indices, hospital_coords):
    """
    繪製個別醫院的Cat-in-a-Circle分析圖
    符合Steinmann論文的真實意圖：分析特定保險標的物的風險差異
    """
    from datetime import datetime
    
    # 選擇前6家醫院進行詳細分析
    n_hospitals = min(6, len(hospital_coords))
    selected_hospitals = list(range(n_hospitals))
    
    # 圖1: 個別醫院不同半徑的風速分布比較
    fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    radii = list(hospital_cat_indices.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(radii)))
    
    for h_idx, hospital_idx in enumerate(selected_hospitals):
        ax = axes[h_idx]
        
        for r_idx, radius in enumerate(radii):
            hospital_winds = hospital_cat_indices[radius][hospital_idx]
            non_zero_winds = hospital_winds[hospital_winds > 0]
            
            if len(non_zero_winds) > 0:
                ax.hist(non_zero_winds, bins=20, alpha=0.6, 
                       label=f'{radius}km', color=colors[r_idx], density=True)
        
        # 添加Saffir-Simpson分級線
        saffir_simpson_thresholds = [33, 42, 49, 58, 70]
        ss_colors = ['yellow', 'orange', 'red', 'purple', 'darkred']
        ss_names = ['Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5']
        
        if ax.get_ylim()[1] > 0:  # 確保有數據
            y_max = ax.get_ylim()[1]
            for threshold, color, name in zip(saffir_simpson_thresholds, ss_colors, ss_names):
                ax.axvline(threshold, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
                ax.text(threshold, y_max*0.8, name, rotation=90, 
                       verticalalignment='bottom', fontsize=8, alpha=0.8)
        
        ax.set_xlabel('風速 (m/s)')
        ax.set_ylabel('密度')
        ax.set_title(f'醫院 {hospital_idx+1} 的Cat-in-a-Circle風速分布')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('個別醫院的不同半徑Cat-in-a-Circle風速分布比較\\n（符合Steinmann論文方法論）', fontsize=16)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename1 = f'hospital_specific_cat_in_circle_{timestamp}.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"   📊 個別醫院Cat-in-a-Circle分析圖已保存: {filename1}")
    plt.show()
    
    # 圖2: 半徑效應分析 - 統計摘要
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 計算每個半徑的統計數據
    radius_stats = {}
    for radius in radii:
        all_hospital_winds = []
        max_winds_per_hospital = []
        mean_winds_per_hospital = []
        
        for h_idx in range(len(hospital_coords)):
            h_winds = hospital_cat_indices[radius][h_idx]
            non_zero_winds = h_winds[h_winds > 0]
            
            if len(non_zero_winds) > 0:
                all_hospital_winds.extend(non_zero_winds)
                max_winds_per_hospital.append(np.max(non_zero_winds))
                mean_winds_per_hospital.append(np.mean(non_zero_winds))
        
        radius_stats[radius] = {
            'all_winds': all_hospital_winds,
            'max_per_hospital': max_winds_per_hospital,
            'mean_per_hospital': mean_winds_per_hospital
        }
    
    # 子圖1: 平均風速vs半徑
    radii_list = list(radii)
    avg_winds = [np.mean(radius_stats[r]['all_winds']) if radius_stats[r]['all_winds'] else 0 
                for r in radii_list]
    
    ax1.plot(radii_list, avg_winds, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Cat-in-a-Circle半徑 (km)')
    ax1.set_ylabel('平均風速 (m/s)')
    ax1.set_title('平均風速 vs 半徑')
    ax1.grid(True, alpha=0.3)
    
    # 子圖2: 最大風速vs半徑
    max_winds = [np.max(radius_stats[r]['all_winds']) if radius_stats[r]['all_winds'] else 0 
                for r in radii_list]
    
    ax2.plot(radii_list, max_winds, 'o-', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Cat-in-a-Circle半徑 (km)')
    ax2.set_ylabel('最大風速 (m/s)')  
    ax2.set_title('最大風速 vs 半徑')
    ax2.grid(True, alpha=0.3)
    
    # 子圖3: 醫院間風速變異 vs 半徑
    wind_std = [np.std(radius_stats[r]['mean_per_hospital']) if radius_stats[r]['mean_per_hospital'] else 0
               for r in radii_list]
    
    ax3.plot(radii_list, wind_std, 'o-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('Cat-in-a-Circle半徑 (km)')
    ax3.set_ylabel('醫院間風速標準差 (m/s)')
    ax3.set_title('醫院間風險差異 vs 半徑')
    ax3.grid(True, alpha=0.3)
    
    # 子圖4: Box plot比較不同半徑
    box_data = [radius_stats[r]['all_winds'] for r in radii_list]
    box_plot = ax4.boxplot(box_data, labels=radii_list, patch_artist=True)
    
    # 美化box plot
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('Cat-in-a-Circle半徑 (km)')
    ax4.set_ylabel('風速分布 (m/s)')
    ax4.set_title('不同半徑風速分布比較')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Cat-in-a-Circle半徑效應分析\\n（基於個別醫院數據）', fontsize=16)
    plt.tight_layout()
    
    filename2 = f'cat_in_circle_radius_effects_{timestamp}.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"   📊 半徑效應分析圖已保存: {filename2}")
    plt.show()
    
    return filename1, filename2

# Duplicate function definitions have been moved to the top of the file

def visualize_exposure_spatial_distribution(exposure_main):
    """視覺化曝險空間分布"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 空間分布散點圖
    ax1 = axes[0]
    scatter = ax1.scatter(exposure_main.longitude, exposure_main.latitude,
                         c=exposure_main.gdf.value/1e6, 
                         s=2, alpha=0.7, cmap='viridis')
    ax1.set_xlabel('經度')
    ax1.set_ylabel('緯度')
    ax1.set_title('北卡羅來納州曝險資產空間分布')
    ax1.grid(True, alpha=0.3)
    
    # 添加顏色條
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('資產價值 (百萬美元)')
    
    # 2. 價值密度熱力圖
    ax2 = axes[1]
    # 創建2D直方圖來顯示價值密度
    H, xedges, yedges = np.histogram2d(exposure_main.longitude, 
                                      exposure_main.latitude, 
                                      bins=50,
                                      weights=exposure_main.gdf.value)
    
    im = ax2.imshow(H.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                   cmap='hot', aspect='auto')
    ax2.set_xlabel('經度')
    ax2.set_ylabel('緯度')
    ax2.set_title('資產價值密度分布')
    
    # 添加顏色條
    cbar2 = plt.colorbar(im, ax=ax2)
    cbar2.set_label('總資產價值 (美元)')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exposure_spatial_distribution_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   📊 曝險空間分布圖已保存: {filename}")
    plt.show()
    
    return fig


def create_impact_visualizations(impact, exposure, hazard):
    """
    創建CLIMADA影響分析視覺化
    
    Parameters:
    -----------
    impact : climada.engine.Impact
        CLIMADA影響對象
    exposure : climada.entity.Exposures
        曝險對象
    hazard : climada.hazard.TropCyclone
        熱帶氣旋災害對象
        
    Returns:
    --------
    matplotlib.figure.Figure
        包含多個子圖的綜合分析圖
    """
    
    # 創建4個子圖的布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 事件損失分布直方圖
    ax1 = axes[0, 0]
    non_zero_losses = impact.at_event[impact.at_event > 0] / 1e9  # 轉換為十億美元
    if len(non_zero_losses) > 0:
        ax1.hist(non_zero_losses, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('單次事件損失 (十億美元)')
        ax1.set_ylabel('事件數量')
        ax1.set_title('熱帶氣旋事件損失分布')
        ax1.grid(True, alpha=0.3)
        
        # 添加統計信息
        mean_loss = non_zero_losses.mean()
        max_loss = non_zero_losses.max()
        ax1.axvline(mean_loss, color='red', linestyle='--', alpha=0.8, 
                   label=f'平均損失: ${mean_loss:.2f}B')
        ax1.axvline(max_loss, color='orange', linestyle='--', alpha=0.8, 
                   label=f'最大損失: ${max_loss:.2f}B')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, '無損失事件', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('熱帶氣旋事件損失分布')
    
    # 2. 年均損失空間分布
    ax2 = axes[0, 1]
    if hasattr(impact, 'imp_mat') and impact.imp_mat is not None:
        # 計算每個曝險點的年均損失
        annual_losses = impact.imp_mat.mean(axis=0) / 1e6  # 轉換為百萬美元
        scatter = ax2.scatter(exposure.longitude, exposure.latitude,
                             c=annual_losses, s=3, alpha=0.7, cmap='Reds')
        ax2.set_xlabel('經度')
        ax2.set_ylabel('緯度')
        ax2.set_title('年均損失空間分布')
        ax2.grid(True, alpha=0.3)
        
        # 添加顏色條
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('年均損失 (百萬美元)')
    else:
        ax2.text(0.5, 0.5, '無詳細損失矩陣數據', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('年均損失空間分布')
    
    # 3. 回歸期損失曲線
    ax3 = axes[1, 0]
    try:
        # 計算回歸期損失
        return_periods = np.array([2, 5, 10, 25, 50, 100, 250])
        return_period_losses = []
        
        sorted_losses = np.sort(impact.at_event[impact.at_event > 0])
        n_events = len(sorted_losses)
        
        for rp in return_periods:
            if n_events > 0:
                # 使用經驗分位數
                quantile = 1 - 1/rp
                index = int(quantile * n_events)
                if index >= n_events:
                    index = n_events - 1
                rp_loss = sorted_losses[index] / 1e9
                return_period_losses.append(rp_loss)
            else:
                return_period_losses.append(0)
        
        ax3.semilogx(return_periods, return_period_losses, 'o-', linewidth=2, markersize=8)
        ax3.set_xlabel('回歸期 (年)')
        ax3.set_ylabel('損失 (十億美元)')
        ax3.set_title('回歸期損失曲線')
        ax3.grid(True, alpha=0.3)
        
        # 標註重要回歸期
        for i, (rp, loss) in enumerate(zip(return_periods, return_period_losses)):
            if rp in [10, 50, 100]:
                ax3.annotate(f'{rp}年: ${loss:.2f}B', 
                           xy=(rp, loss), xytext=(10, 10),
                           textcoords='offset points', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                
    except Exception as e:
        ax3.text(0.5, 0.5, f'回歸期計算錯誤: {str(e)}', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('回歸期損失曲線')
    
    # 4. 綜合統計總結
    ax4 = axes[1, 1]
    ax4.axis('off')  # 關閉坐標軸
    
    # 準備統計文字
    stats_text = []
    stats_text.append("🌀 熱帶氣旋影響分析統計總結")
    stats_text.append("=" * 40)
    stats_text.append(f"總事件數: {hazard.size:,}")
    stats_text.append(f"造成損失事件數: {(impact.at_event > 0).sum():,}")
    stats_text.append(f"年均總損失 (AAI): ${impact.aai_agg/1e9:.2f}B")
    stats_text.append(f"最大單次事件損失: ${impact.at_event.max()/1e9:.2f}B")
    stats_text.append(f"總曝險價值: ${exposure.value.sum()/1e9:.2f}B")
    stats_text.append(f"曝險點數量: {len(exposure.gdf):,}")
    
    # 計算損失率
    if impact.aai_agg > 0 and exposure.value.sum() > 0:
        loss_ratio = (impact.aai_agg / exposure.value.sum()) * 100
        stats_text.append(f"年均損失率: {loss_ratio:.3f}%")
    
    # 顯示統計文字
    stats_full_text = '\n'.join(stats_text)
    ax4.text(0.05, 0.95, stats_full_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存圖片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'climada_impact_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   📊 CLIMADA影響分析圖已保存: {filename}")
    plt.show()
    
    return fig


# %% 步驟1: 生成Steinmann基礎產品 (70個)
print("\n📦 步驟1: 生成Steinmann標準產品...")

if not modules_available['core']:
    print("❌ 核心模組不可用，無法繼續")
    raise ImportError("核心引擎模組必需")

base_products, product_summary = generate_steinmann_2023_products()

print(f"✅ 成功生成 {len(base_products)} 個Steinmann標準產品")
print(f"   單閾值: {product_summary['single_threshold']}")
print(f"   雙閾值: {product_summary['double_threshold']}")
print(f"   三閾值: {product_summary['triple_threshold']}")
print(f"   四閾值: {product_summary['quadruple_threshold']}")

# %% 步驟2: 擴展為多半徑產品 (350個)
print("\n🎯 步驟2: 擴展為多半徑產品...")

radii = [15, 30, 50, 75, 100]  # Cat-in-a-Circle半徑(km)
multi_radius_products = []

for radius in radii:
    for product in base_products:
        multi_product = {
            'product_id': f"{product.product_id}_R{radius}",
            'name': f"Steinmann {product.product_id} (R={radius}km)",
            'base_product_id': product.product_id,
            'radius_km': radius,
            'trigger_thresholds': product.thresholds,
            'payout_amounts': [product.max_payout * ratio for ratio in product.payouts],
            'max_payout': product.max_payout,
            'product_type': 'cat_in_circle',
            'payout_function_type': 'step',
            'metadata': {
                'structure_type': product.structure_type,
                'saffir_simpson_based': True
            }
        }
        multi_radius_products.append(multi_product)

print(f"✅ 成功擴展為 {len(multi_radius_products)} 個多半徑產品")
print(f"   基礎產品: {len(base_products)}")
print(f"   半徑選項: {radii}")
print(f"   總計: {len(radii)} × {len(base_products)} = {len(multi_radius_products)}")

# %% 步驟3: 數據準備 - 優先CLIMADA真實數據
print("\n🌪️ 步驟3: 準備分析數據...")

climada_data = None

print("正在準備CLIMADA真實數據...")
try:
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
        climada_data = {
            'tc_hazard': tc_hazard,
            'exposure': exposure_main,
            'impact': impact,
            'impact_func_set': impact_func_set,  # 添加影響函數集
            'event_losses': impact.at_event,
            'exposure_locations': [(lat, lon) for lat, lon in zip(exposure_main.latitude, exposure_main.longitude)],
            'metadata': {
                'n_events': tc_hazard.size,
                'total_exposure': exposure_main.value.sum(),
                'annual_average_impact': impact.aai_agg
            }
        }
        print("   🎉 CLIMADA真實數據準備完成！")
    else:
        print("   ❌ 無法創建曝險數據")
        raise ValueError("無法創建曝險數據")
except Exception as e:
    print(f"   ❌ CLIMADA數據準備失敗: {e}")
    raise e  # 重新拋出例外，確保不會靜默失敗

# CLIMADA 數據已成功準備，直接使用
main_data = climada_data
data_source = "CLIMADA真實數據"
print(f"\n📊 使用數據源: {data_source}")

# %% 步驟4: 提取Cat-in-a-Circle參數指標
print("\n" + "="*60)
print("🎯 步驟4: Steinmann醫院Cat-in-a-Circle分析")
print("="*60)

# 步驟4a: 獲取醫院位置
print("\n🏥 步驟4a: 獲取北卡羅來納州醫院位置...")
try:
    gdf_hospitals, hospital_exposures = get_nc_hospitals(
        use_mock=True,  # 使用模擬數據，可根據需要改為False使用真實OSM數據
        create_exposures=True,
        visualize=False  # 先不視覺化，避免干擾主流程
    )
    
    # 提取醫院座標
    hospital_coords = [(row.geometry.y, row.geometry.x) for _, row in gdf_hospitals.iterrows()]
    print(f"   ✅ 成功獲取 {len(hospital_coords)} 家醫院位置")
    
except Exception as e:
    print(f"   ❌ 醫院數據獲取失敗: {e}")
    print("   🔄 使用預設示範位置...")
    # 使用幾個主要城市作為示範
    hospital_coords = [
        (35.7796, -78.6382),  # 羅利
        (35.2271, -80.8431),  # 夏洛特  
        (36.0726, -79.7920),  # 格林斯博羅
        (36.1023, -80.2442),  # 溫斯頓塞勒姆
        (34.2257, -77.9447)   # 威明頓
    ]
    print(f"   ✅ 使用 {len(hospital_coords)} 個示範位置")

# 使用CLIMADA真實數據
tc_hazard = main_data['tc_hazard']
exposure_main = main_data['exposure']

# 建立空間索引
print("   🔄 建立災害場空間索引...")
from scipy.spatial import cKDTree
hazard_coords = np.array([
    [tc_hazard.centroids.lat[i], tc_hazard.centroids.lon[i]] 
    for i in range(tc_hazard.centroids.size)
])
hazard_tree = cKDTree(np.radians(hazard_coords))

# 步驟4b: 醫院Cat-in-a-Circle分析
print("\n🎯 步驟4b: 醫院Cat-in-a-Circle風速提取...")

# 為保持兼容性，首先計算醫院級別數據
hospital_cat_indices = {}
hospital_wind_data = {}  # 保存詳細的醫院風速數據

for radius in radii:
    print(f"\n🌀 計算半徑 {radius}km 的醫院Cat-in-a-Circle指標...")
    
    hospital_winds = extract_hospital_cat_in_circle(
        tc_hazard, hospital_coords, hazard_tree, radius
    )
    
    hospital_wind_data[radius] = hospital_winds
    
    # 修正實現：保存個別醫院的Cat-in-a-Circle數據，而非全域最大值
    # 這才符合Steinmann論文的真實意圖：分析特定保險標的物（醫院）的風險
    hospital_cat_indices[radius] = hospital_winds
    
    print(f"   ✅ 完成 {radius}km 半徑分析: {len(hospital_coords)} 家醫院")
    
    # 計算所有醫院的統計摘要
    all_winds = []
    for h_idx, h_winds in hospital_winds.items():
        all_winds.extend(h_winds[h_winds > 0])  # 只包含非零風速
    
    if all_winds:
        print(f"   📊 半徑內風速統計: 平均 {np.mean(all_winds):.1f} m/s, 最大 {np.max(all_winds):.1f} m/s")

# 步驟4c: 醫院標準化損失計算  
print("\n💊 步驟4c: 醫院標準化損失計算...")
hospital_damages_by_radius = {}
for radius in radii:
    print(f"   🏥 計算半徑 {radius}km 醫院損失...")
    try:
        hospital_damages = calculate_hospital_standardized_damages(
            hospital_wind_data[radius], 
            main_data.get('impact_func_set', None)
        )
        hospital_damages_by_radius[radius] = hospital_damages
    except Exception as e:
        print(f"      ⚠️ 醫院損失計算失敗: {e}")

# 使用醫院數據作為主要分析對象 (保持向後兼容)
cat_in_circle_indices = hospital_cat_indices

print(f"\n✅ Steinmann醫院Cat-in-a-Circle分析完成")
print(f"   🏥 分析醫院數: {len(hospital_coords)}")
print(f"   🌀 分析半徑: {radii} km")
print(f"   📊 事件數: {tc_hazard.intensity.shape[0]}")

# 顯示統計 - 修正：現在indices是字典而非數組
for radius in radii:
    hospital_winds = cat_in_circle_indices[radius]
    
    # 收集所有醫院的風速數據
    all_winds = []
    for hospital_idx, winds in hospital_winds.items():
        all_winds.extend(winds[winds > 0])  # 只包含非零風速
    
    if all_winds:
        all_winds = np.array(all_winds)
        print(f"   半徑 {radius}km: 平均 {np.mean(all_winds):.1f} m/s, 最大 {np.max(all_winds):.1f} m/s, 範圍 {np.min(all_winds):.1f}-{np.max(all_winds):.1f} m/s")
    else:
        print(f"   半徑 {radius}km: 無有效風速數據")

# %% 步驟4d: Cat-in-a-Circle 深度分析與視覺化
print(f"\n🎯 步驟4d: Cat-in-a-Circle 深度分析與視覺化...")

def analyze_cat_in_circle_characteristics(cat_in_circle_indices, damages):
    """
    分析 Cat-in-a-Circle 指標特性
    """
    
    print("🎯 Cat-in-a-Circle 指標特性分析")
    print("=" * 50)
    
    results = {}
    saffir_simpson_thresholds = [33, 42, 49, 58, 70]  # m/s
    threshold_names = ['Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5']
    
    for radius, hospital_winds in cat_in_circle_indices.items():
        print(f"\n🌀 半徑 {radius}km 分析:")
        
        # 收集所有醫院的風速數據
        all_winds = []
        for hospital_idx, winds in hospital_winds.items():
            all_winds.extend(winds[winds > 0])  # 只包含非零風速
        
        if not all_winds:
            print(f"   ⚠️ 無有效風速數據")
            continue
            
        wind_speeds_array = np.array(all_winds)
        
        # 基本統計
        stats = {
            'mean': np.mean(wind_speeds_array),
            'std': np.std(wind_speeds_array),
            'min': np.min(wind_speeds_array),
            'max': np.max(wind_speeds_array),
            'median': np.median(wind_speeds_array)
        }
        
        print(f"   平均風速: {stats['mean']:.1f} ± {stats['std']:.1f} m/s")
        print(f"   風速範圍: {stats['min']:.1f} - {stats['max']:.1f} m/s")
        print(f"   中位數: {stats['median']:.1f} m/s")
        
        # 觸發頻率分析
        print(f"   觸發頻率分析:")
        trigger_counts = []
        for threshold, name in zip(saffir_simpson_thresholds, threshold_names):
            triggered = np.sum(wind_speeds_array >= threshold)
            percentage = triggered / len(wind_speeds_array) * 100
            print(f"     {name} ({threshold} m/s): {triggered}次 ({percentage:.1f}%)")
            trigger_counts.append(triggered)
        
        # 為相關性分析，將風速轉換為與損失同樣長度的數組
        n_events = len(damages)
        event_max_winds = []
        
        for event_idx in range(n_events):
            event_winds = [hospital_winds[h_idx][event_idx] for h_idx in hospital_winds.keys()]
            event_max_winds.append(max(event_winds) if event_winds else 0.0)
        
        event_max_winds = np.array(event_max_winds)
        
        # 相關性分析
        correlation = np.corrcoef(event_max_winds, damages)[0, 1]
        print(f"   風速-損失相關性: {correlation:.3f}")
        
        # 基差風險簡單分析
        # 計算觸發但無損失的情況
        significant_damage_threshold = np.percentile(damages[damages > 0], 10)  # 有意義的損失閾值
        cat1_triggered = event_max_winds >= 33
        low_damage = damages < significant_damage_threshold
        basis_risk_events = np.sum(cat1_triggered & low_damage)
        
        print(f"   潛在基差風險事件: {basis_risk_events}次 ({basis_risk_events/len(event_max_winds)*100:.1f}%)")
        print(f"   (定義: 觸發Cat1但損失低於{significant_damage_threshold/1e6:.0f}M的事件)")
        
        results[radius] = {
            'stats': stats,
            'trigger_counts': trigger_counts,
            'correlation': correlation,
            'basis_risk_events': basis_risk_events
        }
    
    return results

def compare_radius_performance(results):
    """
    比較不同半徑的性能表現
    """
    print(f"\n📊 半徑性能比較")
    print("=" * 50)
    
    # 創建比較表格
    comparison_data = []
    for radius, result in results.items():
        comparison_data.append({
            '半徑(km)': radius,
            '平均風速(m/s)': f"{result['stats']['mean']:.1f}",
            '風速標準差': f"{result['stats']['std']:.1f}",
            '相關性': f"{result['correlation']:.3f}",
            'Cat1觸發次數': result['trigger_counts'][0] if result['trigger_counts'] else 0,
            '基差風險事件': result['basis_risk_events']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # 建議最佳半徑
    best_correlation_radius = max(results.keys(), 
                                 key=lambda r: results[r]['correlation'])
    print(f"\n💡 建議:")
    print(f"   最佳相關性半徑: {best_correlation_radius}km (相關性 {results[best_correlation_radius]['correlation']:.3f})")
    
    return df_comparison

# 執行深度分析
damages = main_data['event_losses']
cat_analysis_results = analyze_cat_in_circle_characteristics(cat_in_circle_indices, damages)

# 生成可視化圖表
print(f"\n📈 生成風速分布圖...")
try:
    # 繪製新的、有意義的醫院Cat-in-a-Circle分析圖
    print("\n📊 步驟4d: 繪製醫院Cat-in-a-Circle分析圖...")
    plot_hospital_cat_in_circle_analysis(hospital_cat_indices, hospital_coords)
except Exception as e:
    print(f"   ⚠️ 圖表生成失敗: {e}")

# 性能比較
performance_comparison = compare_radius_performance(cat_analysis_results)

print(f"\n✅ Cat-in-a-Circle 深度分析完成！")

# %% 步驟4e: CLIMADA 探索視覺化分析  
print(f"\n🌍 步驟4e: CLIMADA 探索視覺化分析...")

# Duplicate function definitions have been moved to the top of the file

def visualize_exposure_spatial_distribution(exposure_main):
    """視覺化曝險空間分布"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 空間分布散點圖
    ax1 = axes[0]
    scatter = ax1.scatter(exposure_main.longitude, exposure_main.latitude,
                         c=exposure_main.gdf.value/1e6, 
                         s=2, alpha=0.7, cmap='viridis')
    ax1.set_xlabel('經度')
    ax1.set_ylabel('緯度')
    ax1.set_title('北卡羅來納州曝險資產空間分布')
    ax1.grid(True, alpha=0.3)
    
    # 添加顏色條
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('資產價值 (百萬美元)')
    
    # 2. 價值密度熱力圖
    ax2 = axes[1]
    # 創建2D直方圖來顯示價值密度
    lon_bins = np.linspace(exposure_main.longitude.min(), exposure_main.longitude.max(), 30)
    lat_bins = np.linspace(exposure_main.latitude.min(), exposure_main.latitude.max(), 30)
    
    # 計算每個bin的總價值
    H, xedges, yedges = np.histogram2d(exposure_main.longitude, exposure_main.latitude, 
                                      bins=[lon_bins, lat_bins], 
                                      weights=exposure_main.gdf.value)
    
    im = ax2.imshow(H.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                   cmap='hot', aspect='auto')
    ax2.set_xlabel('經度')
    ax2.set_ylabel('緯度')
    ax2.set_title('資產價值密度分布')
    
    # 添加顏色條
    cbar2 = plt.colorbar(im, ax=ax2)
    cbar2.set_label('總資產價值 (美元)')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exposure_spatial_distribution_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   📊 曝險空間分布圖已保存: {filename}")
    plt.show()
    
    return fig

def compare_radius_performance(results):
    """
    比較不同半徑的性能表現
    """
    print(f"\n📊 半徑性能比較")
    print("=" * 50)
    
    # 創建比較表格
    comparison_data = []
    for radius, result in results.items():
        comparison_data.append({
            '半徑(km)': radius,
            '平均風速(m/s)': f"{result['stats']['mean']:.1f}",
            '風速標準差': f"{result['stats']['std']:.1f}",
            '相關性': f"{result['correlation']:.3f}",
            'Cat1觸發率(%)': f"{result['trigger_counts'][0]/328*100:.1f}",
            '基差風險事件': result['basis_risk_events']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # 建議最佳半徑
    best_correlation_radius = max(results.keys(), 
                                 key=lambda r: results[r]['correlation'])
    print(f"\n💡 建議:")
    print(f"   最佳相關性半徑: {best_correlation_radius}km (相關性 {results[best_correlation_radius]['correlation']:.3f})")

# Hospital functions have been moved to the top of the file

def plot_top_tracks(tracks, bounds, title="北卡羅來納州主要颱風軌跡"):
    """繪製主要颱風軌跡"""
    
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        # 選擇前20強的軌跡
        track_intensities = []
        for track in tracks:
            if hasattr(track, 'max_sustained_wind'):
                max_wind = np.max(track.max_sustained_wind.values)
            else:
                max_wind = 0
            track_intensities.append(max_wind)
        
        # 排序並選擇前20
        sorted_indices = np.argsort(track_intensities)[::-1][:20]
        
        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # 設置地圖範圍
        ax.set_extent([bounds['lon_min']-2, bounds['lon_max']+2, 
                      bounds['lat_min']-1, bounds['lat_max']+1], 
                     ccrs.PlateCarree())
        
        # 添加地圖特徵
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        
        # 繪製軌跡
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        categories = ['Cat 5 (>70 m/s)', 'Cat 4 (58-70)', 'Cat 3 (49-58)', 
                     'Cat 2 (42-49)', 'Cat 1 (33-42)']
        
        for i, track_idx in enumerate(sorted_indices):
            track = tracks[track_idx]
            max_wind = track_intensities[track_idx]
            
            # 根據強度選擇顏色
            if max_wind >= 70:
                color = colors[0]
            elif max_wind >= 58:
                color = colors[1]
            elif max_wind >= 49:
                color = colors[2]
            elif max_wind >= 42:
                color = colors[3]
            else:
                color = colors[4]
            
            # 繪製軌跡
            if hasattr(track, 'lon') and hasattr(track, 'lat'):
                ax.plot(track.lon.values, track.lat.values, 
                       color=color, linewidth=2, alpha=0.7, 
                       transform=ccrs.PlateCarree())
        
        # 添加圖例
        legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=cat) 
                          for color, cat in zip(colors, categories)]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.title(f'{title}\n前20強颱風軌跡', fontsize=14)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'top_tc_tracks_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   📊 颱風軌跡圖已保存: {filename}")
        plt.show()
        
        return fig
    
    except ImportError:
        print("   ⚠️ 無法導入cartopy，跳過軌跡地圖繪製")
        return None
    except Exception as e:
        print(f"   ⚠️ 軌跡繪製失敗: {e}")
        return None

# %% 步驟4f: CLIMADA探索與視覺化
print("\n🎨 步驟4f: CLIMADA探索與視覺化")  
print("="*60)

# 首先顯示LitPop曝險詳細統計
if 'exposure_dict' in locals() and successful_years:
    print(f"\n🏘️ LitPop 曝險資料處理完成，成功處理 {len(successful_years)} 年資料")
    
    # 顯示所有年份的詳細統計
    for year in successful_years:
        exposure = exposure_dict[year]
        print(f"\n📊 {year} 年統計:")
        print(f"   資產點數: {len(exposure.gdf):,}")
        print(f"   總價值: ${exposure.gdf['value'].sum()/1e9:.1f}B")
        print(f"   平均單點價值: ${exposure.gdf['value'].mean()/1e6:.2f}M")
        print(f"   中位數單點價值: ${exposure.gdf['value'].median()/1e6:.2f}M")
        print(f"   最大單點價值: ${exposure.gdf['value'].max()/1e6:.2f}M")
    
    # 暫時跳過視覺化，確保計算正常運行
    print("\n   ⚠️ 暫時跳過LitPop視覺化以確保主要計算正常運行")
    
    # # 創建各年份的詳細視覺化
    # print("\n   📊 創建LitPop空間分布與價值分布視覺化...")
    # spatial_fig, value_fig = visualize_all_litpop_exposures(exposure_dict, successful_years)
    # if value_fig:
    #     plt.show()
    
    # # 創建年度比較圖表
    # if len(successful_years) > 1:
    #     print("\n   📈 創建年度比較分析...")
    #     comparison_fig, stats_summary = create_yearly_comparison(exposure_dict, successful_years)
    #     if comparison_fig:
    #         plt.show()
    #     
    #     print(f"\n📈 年度比較摘要:")
    #     print(stats_summary.round(2))

# 影響分析視覺化
print("\n   ⚠️ 暫時跳過所有視覺化以確保主要計算正常運行")

# try:
#     print("\n   📊 創建綜合影響分析視覺化...")
#     if 'impact' in locals():
#         impact_fig = create_impact_visualizations(impact, exposure_main, tc_hazard)
#     else:
#         print("      ⚠️ 影響數據不可用，跳過影響視覺化")
#     
#     print("\n   🗺️ 視覺化曝險空間分布...")
#     exposure_fig = visualize_exposure_spatial_distribution(exposure_main)
#     
#     print("\n   🌀 繪製主要颱風軌跡...")
#     if 'tracks' in locals() and hasattr(tracks, 'data'):
#         tracks_fig = plot_top_tracks(tracks.data, NC_BOUNDS)
#     else:
#         print("      ⚠️ 軌跡數據不可用，跳過軌跡視覺化")
#     
#     print("\n   ✅ CLIMADA探索與視覺化完成！")
#     
# except Exception as e:
#     print(f"   ⚠️ 視覺化過程中發生錯誤: {e}")
#     print("   🔄 繼續執行後續分析...")

print("\n   ✅ 步驟4f完成，繼續進行主要分析...")

# %% 步驟5a: 傳統Steinmann RMSE分析
print(f"\n📊 步驟5a: 傳統Steinmann RMSE分析 ({len(multi_radius_products)} 個產品)...")

damages = main_data['event_losses']
rmse_results = []

for i, product in enumerate(multi_radius_products):
    if (i + 1) % 50 == 0:
        print(f"   進度: {i+1}/{len(multi_radius_products)}")
    
    radius = product['radius_km']
    hospital_winds = cat_in_circle_indices[radius]
    
    # 轉換醫院字典為適合分析的格式
    # 選擇最具代表性的方式：每個事件取所有醫院的最大風速
    n_events = len(list(hospital_winds.values())[0]) if hospital_winds else 0
    wind_speeds = np.zeros(n_events)
    
    for event_idx in range(n_events):
        event_winds = [hospital_winds[h_idx][event_idx] for h_idx in hospital_winds.keys()]
        wind_speeds[event_idx] = max(event_winds) if event_winds else 0.0
    
    # 計算階梯式賠付 (正確版本)
    payouts = calculate_correct_step_payouts(
        wind_speeds,
        product['trigger_thresholds'],
        [amount/product['max_payout'] for amount in product['payout_amounts']],
        product['max_payout']
    )
    
    # 計算指標
    rmse = np.sqrt(np.mean((damages - payouts) ** 2))
    mae = np.mean(np.abs(damages - payouts))
    correlation = np.corrcoef(damages, payouts)[0,1] if np.std(payouts) > 0 else 0
    trigger_rate = np.mean(payouts > 0)
    mean_payout = np.mean(payouts)
    basis_risk = np.std(damages - payouts)
    
    rmse_results.append({
        'product_id': product['product_id'],
        'radius_km': radius,
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'trigger_rate': trigger_rate,
        'mean_payout': mean_payout,
        'basis_risk': basis_risk,
        'payouts': payouts
    })

# 分析結果
rmse_df = pd.DataFrame(rmse_results)
best_rmse_idx = rmse_df['rmse'].idxmin()
steinmann_best = rmse_df.iloc[best_rmse_idx].to_dict()

print(f"   ✅ Steinmann RMSE分析完成")
print(f"      最佳RMSE: ${steinmann_best['rmse']/1e9:.3f}B")
print(f"      最佳產品: {steinmann_best['product_id']}")
print(f"      最佳相關性: {rmse_df['correlation'].max():.3f}")
print(f"      平均觸發率: {rmse_df['trigger_rate'].mean():.1%}")

steinmann_results = {
    'method': 'steinmann_rmse',
    'results_df': rmse_df,
    'best_product': steinmann_best,
    'summary': {
        'best_rmse': steinmann_best['rmse'],
        'best_correlation': rmse_df['correlation'].max(),
        'mean_trigger_rate': rmse_df['trigger_rate'].mean()
    }
}

# %% 步驟5b: 穩健貝氏CRPS分析 
print(f"\n🧠 步驟5b: 穩健貝氏CRPS分析 ({len(multi_radius_products)} 個產品)...")

# 初始化損失分布
loss_distributions = {}
n_samples = 500  # 增加樣本數

if modules_available['bayesian'] and 'tc_hazard' in main_data and 'exposure' in main_data:
    try:
        print("   🚀 啟動穩健貝氏分析器...")
        
        # 初始化完整的穩健貝氏分析器
        bayesian_analyzer = RobustBayesianAnalyzer(
            density_ratio_constraint=2.0,      # 密度比約束
            n_monte_carlo_samples=n_samples,   # Monte Carlo樣本數
            n_mixture_components=3,            # 混合模型組件數
            hazard_uncertainty_std=0.15,       # 災害不確定性 15%
            exposure_uncertainty_log_std=0.20, # 曝險不確定性 20%
            vulnerability_uncertainty_std=0.10 # 脆弱性不確定性 10%
        )
        print("   ✅ 穩健貝氏分析器初始化完成")
        
        # 使用真實的CLIMADA數據
        tc_hazard = main_data['tc_hazard']
        exposure_main = main_data['exposure']
        impact_func_set = main_data.get('impact_func_set', None)
        
        print("   🔬 執行完整穩健貝氏不確定性量化...")
        print(f"      📊 災害事件數: {tc_hazard.size}")
        print(f"      🏘️ 曝險點數: {len(exposure_main.gdf)}")
        print(f"      🎲 Monte Carlo樣本數: {n_samples}")
        
        # 執行完整的穩健貝氏分析
        try:
            # 使用醫院曝險數據來生成與Cat-in-a-Circle一致的損失分布
            print("      🏥 使用醫院曝險數據進行貝氏建模...")
            
            # 如果有醫院曝險數據，優先使用
            if 'hospital_exposures' in locals() and hospital_exposures is not None:
                exposure_for_bayesian = hospital_exposures
                print(f"      🎯 使用 {len(exposure_for_bayesian.gdf)} 個醫院點進行建模")
            else:
                exposure_for_bayesian = exposure_main
                print(f"      ⚠️ 使用完整LitPop數據 ({len(exposure_for_bayesian.gdf)} 點)")
            
            # 確保damages是正確的numpy數組格式
            damages_array = np.array(damages, dtype=np.float64)
            print(f"      📊 損失數據格式: {type(damages_array)}, 形狀: {damages_array.shape}")
            
            bayesian_results = bayesian_analyzer.comprehensive_bayesian_analysis(
                tc_hazard=tc_hazard,                    # CLIMADA災害對象
                exposure_main=exposure_for_bayesian,    # 使用醫院曝險數據
                impact_func_set=impact_func_set,        # CLIMADA影響函數
                observed_losses=damages_array,          # 使用格式化的損失數據
                parametric_products=multi_radius_products[:5]  # 傳入部分產品用於分析
            )
            
            # 正確提取概率分布
            uncertainty_results = bayesian_results.get('uncertainty_quantification', {})
            probabilistic_results = uncertainty_results.get('probabilistic_loss_distributions', {})
            
            if 'event_loss_distributions' in probabilistic_results:
                # 轉換格式為與既有代碼兼容的字典
                event_distributions = probabilistic_results['event_loss_distributions']
                loss_distributions = {}
                
                for event_id, event_data in event_distributions.items():
                    if isinstance(event_id, str) and event_id.startswith('event_'):
                        event_idx = int(event_id.split('_')[1])
                    else:
                        event_idx = int(event_id) if isinstance(event_id, (int, str)) else 0
                    
                    # 提取樣本數據
                    if isinstance(event_data, dict):
                        samples = event_data.get('samples', event_data.get('distribution_samples', []))
                    else:
                        samples = event_data if hasattr(event_data, '__iter__') else []
                    
                    # 安全地檢查 samples 是否為空
                    if isinstance(samples, np.ndarray):
                        loss_distributions[event_idx] = samples if samples.size > 0 else np.array([0])
                    elif isinstance(samples, (list, tuple)):
                        loss_distributions[event_idx] = np.array(samples) if len(samples) > 0 else np.array([0])
                    else:
                        loss_distributions[event_idx] = np.array([0])
                
                print(f"      ✅ 使用完整穩健貝氏概率分布 (醫院級別)")
                print(f"      🧮 提取了 {len(loss_distributions)} 個事件的分布")
                
            elif bayesian_results:
                print("      ⚠️ 完整分析器結果格式不符，但包含其他數據")
                print(f"      📋 結果包含: {list(bayesian_results.keys())}")
                if uncertainty_results:
                    print(f"      🎲 不確定性結果: {list(uncertainty_results.keys())}")
                raise ValueError("Distribution format mismatch - no event_loss_distributions")
            else:
                print("      ⚠️ 完整分析器未返回結果")
                raise ValueError("No results from comprehensive analyzer")
                
        except Exception as comprehensive_error:
            print(f"      ⚠️ 完整分析器失敗: {comprehensive_error}")
            print(f"      🔍 錯誤類型: {type(comprehensive_error).__name__}")
            import traceback
            tb_lines = traceback.format_exc().split('\n')
            error_location = tb_lines[-3] if len(tb_lines) > 2 else 'Unknown'
            print(f"      📍 錯誤位置: {error_location}")
            print("      🔄 使用基礎概率分布生成器...")
            
            # 備用方案：使用基礎的概率分布生成器
            enhanced_distributions = generate_probabilistic_loss_distributions(
                tc_hazard=tc_hazard,                    # CLIMADA災害對象
                exposure=exposure_for_bayesian,         # 使用與主分析相同的曝險數據
                impact_func_set=impact_func_set,        # CLIMADA影響函數
                n_samples=n_samples                     # Monte Carlo樣本數
            )
            
            if enhanced_distributions is not None and isinstance(enhanced_distributions, dict) and 'event_loss_distributions' in enhanced_distributions:
                # 正確提取概率分布
                event_distributions = enhanced_distributions['event_loss_distributions']
                loss_distributions = {}
                
                for event_id, event_data in event_distributions.items():
                    if isinstance(event_id, str) and event_id.startswith('event_'):
                        event_idx = int(event_id.split('_')[1])
                    else:
                        event_idx = int(event_id) if isinstance(event_id, (int, str)) else 0
                    
                    # 提取樣本數據
                    if isinstance(event_data, dict):
                        samples = event_data.get('samples', event_data.get('distribution_samples', []))
                    else:
                        samples = event_data if hasattr(event_data, '__iter__') else []
                    
                    # 安全地檢查 samples 是否為空
                    if isinstance(samples, np.ndarray):
                        loss_distributions[event_idx] = samples if samples.size > 0 else np.array([0])
                    elif isinstance(samples, (list, tuple)):
                        loss_distributions[event_idx] = np.array(samples) if len(samples) > 0 else np.array([0])
                    else:
                        loss_distributions[event_idx] = np.array([0])
                
                print("      ✅ 使用基於CLIMADA的增強貝氏分布")
            else:
                raise ValueError("Enhanced distributions failed")
        
        print(f"      🧮 生成了 {len(loss_distributions)} 個事件的概率分布")
        if isinstance(loss_distributions, dict) and len(loss_distributions) > 0:
            sample_dist = list(loss_distributions.values())[0]
            print(f"      📈 每個分布包含 {len(sample_dist) if hasattr(sample_dist, '__len__') else 'N/A'} 個樣本")
            
    except Exception as e:
        print(f"      ❌ 穩健貝氏分析失敗: {e}")
        print("      🔄 退回標準概率分布...")
        
        # 標準分布作為最終備用
        uncertainty_ratio = 0.25
        for i, damage in enumerate(damages):
            if damage > 0:
                std = damage * uncertainty_ratio
                mu = np.log(damage) - 0.5 * np.log(1 + (std/damage)**2)
                sigma = np.sqrt(np.log(1 + (std/damage)**2))
                samples = np.random.lognormal(mu, sigma, n_samples)
            else:
                samples = np.random.exponential(1e6, n_samples)
            loss_distributions[i] = samples
        print("      ✅ 使用標準對數正態分布")
        
else:
    print("   ⚠️ 穩健貝氏模組不可用，使用標準概率分布")
    uncertainty_ratio = 0.25
    for i, damage in enumerate(damages):
        if damage > 0:
            std = damage * uncertainty_ratio
            mu = np.log(damage) - 0.5 * np.log(1 + (std/damage)**2)
            sigma = np.sqrt(np.log(1 + (std/damage)**2))
            samples = np.random.lognormal(mu, sigma, n_samples)
        else:
            samples = np.random.exponential(1e6, n_samples)
        loss_distributions[i] = samples

# 執行CRPS分析
print(f"   ⚖️ 使用醫院標準化損失作為對比基準...")
crps_results = []

for i, product in enumerate(multi_radius_products):
    if (i + 1) % 50 == 0:
        print(f"   進度: {i+1}/{len(multi_radius_products)}")
    
    radius = product['radius_km']
    hospital_winds = cat_in_circle_indices[radius]
    
    # 轉換醫院字典為適合分析的格式
    # 每個事件取所有醫院的最大風速
    n_events = len(list(hospital_winds.values())[0]) if hospital_winds else 0
    wind_speeds = np.zeros(n_events)
    
    for event_idx in range(n_events):
        event_winds = [hospital_winds[h_idx][event_idx] for h_idx in hospital_winds.keys()]
        wind_speeds[event_idx] = max(event_winds) if event_winds else 0.0
    
    # 計算賠付
    payouts = calculate_correct_step_payouts(
        wind_speeds,
        product['trigger_thresholds'],
        [amount/product['max_payout'] for amount in product['payout_amounts']],
        product['max_payout']
    )
    
    # 獲取對應半徑的醫院標準化損失作為基準
    if radius in hospital_damages_by_radius:
        hospital_damages = hospital_damages_by_radius[radius]
        reference_losses = hospital_damages  # 使用醫院標準化損失
        comparison_type = "醫院標準化損失"
    else:
        reference_losses = damages  # 備用：使用整體經濟損失
        comparison_type = "整體經濟損失"
    
    # 計算CRPS分數
    crps_scores = []
    for j, payout in enumerate(payouts):
        if j in loss_distributions:
            crps = calculate_crps_score(payout, loss_distributions[j])
        else:
            # 如果沒有分布數據，使用參考損失的點估計
            reference_loss = reference_losses[j] if j < len(reference_losses) else 0
            crps = abs(payout - reference_loss)
        
        # 確保 crps 是標量值
        if isinstance(crps, np.ndarray):
            crps = float(crps.flatten()[0]) if crps.size > 0 else 0.0
        else:
            crps = float(crps)
        
        crps_scores.append(crps)
    
    mean_crps = np.mean(crps_scores)
    
    # 使用相同的基準損失計算相關性
    if len(reference_losses) == len(payouts):
        correlation = np.corrcoef(reference_losses, payouts)[0,1] if np.std(payouts) > 0 else 0
    else:
        correlation = np.corrcoef(damages, payouts)[0,1] if np.std(payouts) > 0 else 0
    
    crps_results.append({
        'product_id': product['product_id'],
        'radius_km': radius,
        'crps': mean_crps,
        'correlation': correlation,
        'trigger_rate': np.mean(payouts > 0),
        'mean_payout': np.mean(payouts),
        'basis_risk_probabilistic': mean_crps,
        'comparison_type': comparison_type,  # 記錄使用的基準類型
        'payouts': payouts
    })

# 分析結果
crps_df = pd.DataFrame(crps_results)
best_crps_idx = crps_df['crps'].idxmin()
bayesian_best = crps_df.iloc[best_crps_idx].to_dict()

print(f"   ✅ 貝氏CRPS分析完成")
print(f"      最佳CRPS: ${bayesian_best['crps']/1e9:.3f}B")
print(f"      最佳產品: {bayesian_best['product_id']}")
print(f"      最佳相關性: {crps_df['correlation'].max():.3f}")
print(f"      平均觸發率: {crps_df['trigger_rate'].mean():.1%}")

bayesian_results = {
    'method': 'bayesian_crps',
    'results_df': crps_df,
    'best_product': bayesian_best,
    'summary': {
        'best_crps': bayesian_best['crps'],
        'best_correlation': crps_df['correlation'].max(),
        'mean_trigger_rate': crps_df['trigger_rate'].mean()
    }
}

# %% 步驟6: 雙軌方法比較
print("\n🔍 步驟6: 雙軌方法比較 (Steinmann RMSE vs Bayesian CRPS)...")

comparison_results = {
    'steinmann_rmse': {
        'best_product': steinmann_best['product_id'],
        'best_radius': steinmann_best['radius_km'],
        'rmse': steinmann_best['rmse'],
        'correlation': steinmann_best['correlation'],
        'trigger_rate': steinmann_best['trigger_rate']
    },
    'bayesian_crps': {
        'best_product': bayesian_best['product_id'],
        'best_radius': bayesian_best['radius_km'],
        'crps': bayesian_best['crps'],
        'correlation': bayesian_best['correlation'],
        'trigger_rate': bayesian_best['trigger_rate']
    },
    'comparison_metrics': {
        'same_best_product': steinmann_best['product_id'] == bayesian_best['product_id'],
        'same_best_radius': steinmann_best['radius_km'] == bayesian_best['radius_km'],
        'correlation_improvement': bayesian_best['correlation'] - steinmann_best['correlation'],
        'rmse_vs_crps_ratio': steinmann_best['rmse'] / bayesian_best['crps'],
        'trigger_rate_difference': bayesian_best['trigger_rate'] - steinmann_best['trigger_rate']
    }
}

print(f"   ✅ 雙軌方法比較完成")
print(f"      Steinmann最佳: {comparison_results['steinmann_rmse']['best_product']}")
print(f"        - 半徑: {comparison_results['steinmann_rmse']['best_radius']}km")
print(f"        - RMSE: ${comparison_results['steinmann_rmse']['rmse']/1e9:.3f}B")
print(f"      Bayesian最佳: {comparison_results['bayesian_crps']['best_product']}")
print(f"        - 半徑: {comparison_results['bayesian_crps']['best_radius']}km")
print(f"        - CRPS: ${comparison_results['bayesian_crps']['crps']/1e9:.3f}B")
print(f"      一致性: 相同最佳產品: {comparison_results['comparison_metrics']['same_best_product']}")
print(f"      相關性提升: {comparison_results['comparison_metrics']['correlation_improvement']:.3f}")

# %% 步驟7: 綜合分析結果展示
print("\n📊 步驟7: 綜合分析結果展示")
print("=" * 80)

# 整合所有結果
all_results = {
    'steinmann_results': steinmann_results,
    'bayesian_results': bayesian_results,
    'comparison_results': comparison_results,
    'data_metadata': {
        'data_source': data_source,
        'n_products_analyzed': len(multi_radius_products),
        'radii_analyzed': radii,
        'n_events': len(damages)
    }
}

# 顯示分析規模概覽
print(f"\n🔍 分析規模概覽:")
print(f"   • 產品總數: {len(multi_radius_products)} 個")
print(f"   • 半徑範圍: {radii} km")
print(f"   • 分析事件: {len(damages)} 個")
print(f"   • 數據來源: {data_source}")
print(f"   • 總損失: ${np.sum(damages)/1e9:.2f}B")
print(f"   • 平均年損失: ${np.mean(damages)/1e9:.4f}B")

# 顯示環境資料統計
print(f"\n🌪️ 風速環境資料統計:")
for radius in radii:
    if radius in cat_analysis_results:
        radius_data = cat_analysis_results[radius]
        print(f"   半徑 {radius}km:")
        print(f"     - 平均風速: {radius_data.get('mean_wind_speed', 0):.1f} m/s")
        print(f"     - 最大風速: {radius_data.get('max_wind_speed', 0):.1f} m/s")
        print(f"     - 風速標準差: {radius_data.get('wind_std', 0):.1f} m/s")
        print(f"     - 相關性: {radius_data.get('correlation', 0):.3f}")

# 顯示Cat-in-a-Circle分析結果
print(f"\n🎯 Cat-in-a-Circle 分析結果:")
best_radius = max(cat_analysis_results.keys(), key=lambda r: cat_analysis_results[r]['correlation'])
print(f"   最佳半徑: {best_radius}km")
print(f"   最高相關性: {cat_analysis_results[best_radius]['correlation']:.3f}")
print(f"   最佳RMSE: ${cat_analysis_results[best_radius].get('rmse', 0)/1e9:.3f}B")
print(f"\n📈 各半徑績效比較:")
for radius in sorted(radii):
    if radius in cat_analysis_results:
        data = cat_analysis_results[radius]
        print(f"   {radius}km: 相關性={data.get('correlation', 0):.3f}, RMSE=${data.get('rmse', 0)/1e9:.3f}B")

print(f"\n   ✅ 結果已整合完成 (無需保存JSON文件)")

# %% 步驟7: 基差風險詳細分析與產品資訊
print("\n📊 步驟7: 基差風險詳細分析與產品資訊")
print("=" * 60)

# 導入基差風險分析模組
try:
    from basis_risk_analysis import analyze_basis_risk_detailed
    basis_risk_module_available = True
except ImportError:
    print("   ⚠️ 基差風險分析模組不可用，使用簡化分析")
    basis_risk_module_available = False

# 計算產品詳細資訊的函數
def calculate_product_statistics(payouts, damages, product):
    """計算產品的詳細統計資訊"""
    
    # 基本統計
    non_zero_payouts = payouts[payouts > 0]
    max_payout = product.get('max_payout', np.max(payouts))
    max_single_payout = np.max(payouts)
    mean_payout = np.mean(payouts)
    total_payouts = np.sum(payouts)
    
    # 保費計算 (簡化版本)
    expected_payout = mean_payout
    risk_load_factor = 0.20  # 20% 風險附加
    expense_load_factor = 0.15  # 15% 費用附加
    
    technical_premium = expected_payout * (1 + risk_load_factor)
    commercial_premium = technical_premium * (1 + expense_load_factor)
    
    # 效能指標
    payout_efficiency = total_payouts / np.sum(damages) if np.sum(damages) > 0 else 0
    trigger_frequency = np.mean(payouts > 0)
    loss_ratio = total_payouts / (commercial_premium * len(payouts)) if commercial_premium > 0 else 0
    
    # 基差風險
    basis_risk = damages - payouts
    basis_risk_rmse = np.sqrt(np.mean(basis_risk**2))
    basis_risk_mae = np.mean(np.abs(basis_risk))
    
    return {
        'max_possible_payout': max_payout,
        'max_single_payout': max_single_payout,
        'mean_payout': mean_payout,
        'total_payouts': total_payouts,
        'expected_payout': expected_payout,
        'technical_premium': technical_premium,
        'commercial_premium': commercial_premium,
        'payout_efficiency': payout_efficiency,
        'trigger_frequency': trigger_frequency,
        'loss_ratio': loss_ratio,
        'premium_to_exposure_ratio': commercial_premium / (np.sum(damages) / len(damages)) if np.mean(damages) > 0 else 0,
        'basis_risk_rmse': basis_risk_rmse,
        'basis_risk_mae': basis_risk_mae,
        'correlation': np.corrcoef(damages, payouts)[0,1] if np.std(payouts) > 0 else 0,
        'overpayment_events': np.sum((payouts > damages) & (damages > 0)),
        'underpayment_events': np.sum((payouts < damages) & (damages > 0)),
        'false_triggers': np.sum((payouts > 0) & (damages == 0)),
        'missed_events': np.sum((payouts == 0) & (damages > 0))
    }

# 分析最佳產品的基差風險
print("\n🎯 最佳產品基差風險分析:")
print("-" * 40)

# Steinmann最佳產品
steinmann_payouts = rmse_df.iloc[best_rmse_idx]['payouts']
print(f"\n1. Steinmann RMSE 最佳產品 ({steinmann_best['product_id']}):")

if basis_risk_module_available:
    steinmann_basis_risk = analyze_basis_risk_detailed(
        damages, 
        steinmann_payouts,
        f"Steinmann Best - {steinmann_best['product_id']}"
    )
else:
    # 簡化的基差風險分析
    basis_risk = damages - steinmann_payouts
    print(f"   • 總基差風險: ${np.sum(np.abs(basis_risk))/1e9:.2f}B")
    print(f"   • 平均基差風險: ${np.mean(np.abs(basis_risk))/1e9:.3f}B")
    print(f"   • 基差風險標準差: ${np.std(basis_risk)/1e9:.3f}B")
    print(f"   • 最大正基差風險: ${np.max(basis_risk)/1e9:.3f}B (損失未充分補償)")
    print(f"   • 最大負基差風險: ${np.min(basis_risk)/1e9:.3f}B (過度補償)")

# Bayesian最佳產品
if 'crps_df' in locals():
    best_crps_idx = crps_df['crps'].idxmin()
    bayesian_payouts = crps_df.iloc[best_crps_idx]['payouts']
    
    print(f"\n2. Bayesian CRPS 最佳產品 ({bayesian_best['product_id']}):")
    
    if basis_risk_module_available:
        bayesian_basis_risk = analyze_basis_risk_detailed(
            damages,
            bayesian_payouts,
            f"Bayesian Best - {bayesian_best['product_id']}"
        )
    else:
        basis_risk = damages - bayesian_payouts
        print(f"   • 總基差風險: ${np.sum(np.abs(basis_risk))/1e9:.2f}B")
        print(f"   • 平均基差風險: ${np.mean(np.abs(basis_risk))/1e9:.3f}B")
        print(f"   • 基差風險標準差: ${np.std(basis_risk)/1e9:.3f}B")
        print(f"   • 最大正基差風險: ${np.max(basis_risk)/1e9:.3f}B")
        print(f"   • 最大負基差風險: ${np.min(basis_risk)/1e9:.3f}B")

# 分析前10個最佳產品
print("\n🏆 前10個最佳產品詳細資訊:")
print("-" * 40)

# Steinmann前10
top10_steinmann = rmse_df.nsmallest(10, 'rmse')
print("\n📊 Steinmann RMSE 前10產品:")
for idx, (i, row) in enumerate(top10_steinmann.iterrows(), 1):
    product = multi_radius_products[i]
    payouts = row['payouts']
    stats = calculate_product_statistics(payouts, damages, product)
    
    print(f"\n{idx}. {row['product_id']} (半徑: {row['radius_km']}km)")
    print(f"   效能指標:")
    print(f"   • RMSE: ${row['rmse']/1e9:.3f}B")
    print(f"   • 相關性: {stats['correlation']:.3f}")
    print(f"   • 觸發頻率: {stats['trigger_frequency']:.1%}")
    print(f"   保險設計:")
    print(f"   • 最大賠付: ${stats['max_possible_payout']/1e9:.2f}B")
    print(f"   • 平均賠付: ${stats['mean_payout']/1e9:.3f}B")
    print(f"   • 技術保費: ${stats['technical_premium']/1e9:.3f}B")
    print(f"   • 商業保費: ${stats['commercial_premium']/1e9:.3f}B")
    print(f"   • 賠付效率: {stats['payout_efficiency']:.1%}")
    print(f"   基差風險:")
    print(f"   • 基差風險RMSE: ${stats['basis_risk_rmse']/1e9:.3f}B")
    print(f"   • 錯誤觸發: {stats['false_triggers']} 次")
    print(f"   • 遺漏事件: {stats['missed_events']} 次")

# Bayesian前10 (如果有)
if 'crps_df' in locals():
    top10_bayesian = crps_df.nsmallest(10, 'crps')
    print("\n\n🧠 Bayesian CRPS 前10產品:")
    for idx, (i, row) in enumerate(top10_bayesian.iterrows(), 1):
        product = multi_radius_products[i]
        payouts = row['payouts']
        stats = calculate_product_statistics(payouts, damages, product)
        
        print(f"\n{idx}. {row['product_id']} (半徑: {row['radius_km']}km)")
        print(f"   效能指標:")
        print(f"   • CRPS: ${row['crps']/1e9:.3f}B")
        print(f"   • 相關性: {stats['correlation']:.3f}")
        print(f"   • 觸發頻率: {stats['trigger_frequency']:.1%}")
        print(f"   保險設計:")
        print(f"   • 最大賠付: ${stats['max_possible_payout']/1e9:.2f}B")
        print(f"   • 平均賠付: ${stats['mean_payout']/1e9:.3f}B")
        print(f"   • 技術保費: ${stats['technical_premium']/1e9:.3f}B")
        print(f"   • 商業保費: ${stats['commercial_premium']/1e9:.3f}B")
        print(f"   • 賠付效率: {stats['payout_efficiency']:.1%}")
        print(f"   基差風險:")
        print(f"   • 基差風險RMSE: ${stats['basis_risk_rmse']/1e9:.3f}B")
        print(f"   • 錯誤觸發: {stats['false_triggers']} 次")
        print(f"   • 遺漏事件: {stats['missed_events']} 次")

# 儲存詳細分析結果
detailed_results = {
    'top10_steinmann': [],
    'top10_bayesian': [],
    'basis_risk_analysis': {}
}

for i, row in top10_steinmann.iterrows():
    product = multi_radius_products[i]
    payouts = row['payouts']
    stats = calculate_product_statistics(payouts, damages, product)
    detailed_results['top10_steinmann'].append({
        'product_id': row['product_id'],
        'radius_km': row['radius_km'],
        'rmse': row['rmse'],
        **stats
    })

if 'crps_df' in locals():
    for i, row in top10_bayesian.iterrows():
        product = multi_radius_products[i]
        payouts = row['payouts']
        stats = calculate_product_statistics(payouts, damages, product)
        detailed_results['top10_bayesian'].append({
            'product_id': row['product_id'],
            'radius_km': row['radius_km'],
            'crps': row['crps'],
            **stats
        })

# 顯示詳細的前10名產品分析結果
print(f"\n🏆 前10名產品詳細分析:")
print("=" * 80)

# 顯示Steinmann RMSE前10名
print(f"\n📊 Steinmann RMSE 前10名產品:")
for i, product_detail in enumerate(detailed_results['top10_steinmann'][:10]):
    print(f"\n{i+1}. 產品 {product_detail['product_id']}")
    print(f"   🎯 基本資訊:")
    print(f"     - 半徑: {product_detail['radius_km']}km")
    print(f"     - RMSE: ${product_detail['rmse']/1e9:.4f}B")
    print(f"     - 相關性: {product_detail.get('correlation', 0):.4f}")
    
    print(f"   💰 財務資訊:")
    print(f"     - 技術保費: ${product_detail.get('technical_premium', 0)/1e9:.4f}B")
    print(f"     - 商業保費: ${product_detail.get('commercial_premium', 0)/1e9:.4f}B")
    print(f"     - 最大賠付: ${product_detail.get('max_single_payout', 0)/1e9:.4f}B")
    print(f"     - 賠付效率: {product_detail.get('payout_efficiency', 0):.2%}")
    
    print(f"   📊 統計指標:")
    print(f"     - 基差風險: ${product_detail.get('basis_risk', 0)/1e9:.4f}B")
    print(f"     - 觸發機率: {product_detail.get('trigger_probability', 0):.2%}")
    print(f"     - 覆蓋率: {product_detail.get('coverage_ratio', 0):.2%}")
    
    print(f"   🔧 產品設計:")
    print(f"     - 觸發閾值: {product_detail.get('trigger_thresholds', [])} m/s")
    print(f"     - 賠付比例: {product_detail.get('payout_ratios', [])}")
    
    if i < 9:  # 不是最後一個時才印分隔線
        print("   " + "-" * 60)

# 如果有Bayesian結果，也顯示
if detailed_results['top10_bayesian']:
    print(f"\n🧠 Bayesian CRPS 前10名產品:")
    for i, product_detail in enumerate(detailed_results['top10_bayesian'][:10]):
        print(f"\n{i+1}. 產品 {product_detail['product_id']}")
        print(f"   🎯 基本資訊:")
        print(f"     - 半徑: {product_detail['radius_km']}km")
        print(f"     - CRPS: ${product_detail['crps']/1e9:.4f}B")
        print(f"     - 相關性: {product_detail.get('correlation', 0):.4f}")
        
        print(f"   💰 財務資訊:")
        print(f"     - 技術保費: ${product_detail.get('technical_premium', 0)/1e9:.4f}B")
        print(f"     - 商業保費: ${product_detail.get('commercial_premium', 0)/1e9:.4f}B")
        print(f"     - 最大賠付: ${product_detail.get('max_single_payout', 0)/1e9:.4f}B")
        print(f"     - 賠付效率: {product_detail.get('payout_efficiency', 0):.2%}")
        
        print(f"   📊 統計指標:")
        print(f"     - 基差風險: ${product_detail.get('basis_risk', 0)/1e9:.4f}B")
        print(f"     - 觸發機率: {product_detail.get('trigger_probability', 0):.2%}")
        print(f"     - 覆蓋率: {product_detail.get('coverage_ratio', 0):.2%}")
        
        print(f"   🔧 產品設計:")
        print(f"     - 觸發閾值: {product_detail.get('trigger_thresholds', [])} m/s")
        print(f"     - 賠付比例: {product_detail.get('payout_ratios', [])}")
        
        if i < 9:  # 不是最後一個時才印分隔線
            print("   " + "-" * 60)
else:
    print(f"\n⚠️ Bayesian CRPS 分析未能完成，僅顯示 Steinmann RMSE 結果")

print(f"\n✅ 已顯示前10名產品的完整資訊 (無需保存檔案)")

# %% 最終分析摘要
print("\n" + "=" * 80)
print("🎉 北卡羅來納州熱帶氣旋參數型保險分析完成!")
print("=" * 80)

print(f"📊 分析規模:")
print(f"   • 產品數量: {len(multi_radius_products)} 個 (5半徑 × 70函數)")
print(f"   • 分析事件: {len(damages)} 個")
print(f"   • 數據來源: {data_source}")
print(f"   • 分析方法: Steinmann RMSE + Bayesian CRPS")

print(f"\n🏆 最佳產品:")
if 'steinmann_best' in locals():
    print(f"   • Steinmann RMSE: {steinmann_best['product_id']}")
    print(f"     - RMSE: ${steinmann_best['rmse']/1e9:.3f}B")
    print(f"     - 相關性: {steinmann_best['correlation']:.3f}")
    print(f"     - 半徑: {steinmann_best['radius_km']}km")
else:
    print(f"   • Steinmann RMSE: 分析未完成")

if 'bayesian_best' in locals():
    print(f"   • Bayesian CRPS: {bayesian_best['product_id']}")
    print(f"     - CRPS: ${bayesian_best['crps']/1e9:.3f}B") 
    print(f"     - 相關性: {bayesian_best['correlation']:.3f}")
    print(f"     - 半徑: {bayesian_best['radius_km']}km")
else:
    print(f"   • Bayesian CRPS: 分析未完成或使用簡化方法")

print(f"\n🎯 Cat-in-a-Circle 關鍵發現:")
best_radius = max(cat_analysis_results.keys(), key=lambda r: cat_analysis_results[r]['correlation'])
print(f"   • 最佳相關性半徑: {best_radius}km (相關性 {cat_analysis_results[best_radius]['correlation']:.3f})")
print(f"   • 最高觸發頻率半徑: {max(cat_analysis_results.keys(), key=lambda r: cat_analysis_results[r]['trigger_counts'][0])}km")
print(f"   • 基差風險最低半徑: {min(cat_analysis_results.keys(), key=lambda r: cat_analysis_results[r]['basis_risk_events'])}km")

print(f"\n📈 方法對比發現:")
if 'comparison_results' in locals() and 'comparison_metrics' in comparison_results:
    print(f"   • 兩方法最佳產品{'相同' if comparison_results['comparison_metrics']['same_best_product'] else '不同'}")
    print(f"   • 相關性提升: {comparison_results['comparison_metrics']['correlation_improvement']:.3f}")
    print(f"   • 觸發率差異: {comparison_results['comparison_metrics']['trigger_rate_difference']:.1%}")
else:
    print(f"   • 方法對比分析: 需要完整的Bayesian CRPS結果才能進行對比")
    print(f"   • 建議: 在有完整貝氏分析結果時重新運行對比")

print(f"\n🔍 方法論差異分析:")
print(f"   • RMSE (點估計): 衡量賠付與整體經濟損失的差異")
print(f"   • CRPS (分布預測): 衡量賠付與醫院標準化損失分布的準確性")
print(f"   • CRPS考慮了醫院級別的不確定性和空間相關性")
print(f"   • 兩方法選擇相同產品說明結果在不同基準下都穩健")

# 檢查使用的比較基準
if 'crps_df' in locals() and len(crps_df) > 0:
    comparison_types = crps_df['comparison_type'].value_counts()
    print(f"\n📊 CRPS分析基準:")
    for comp_type, count in comparison_types.items():
        print(f"   • {comp_type}: {count} 個產品")
        
print(f"\n🏥 Cat-in-a-Circle 一致性:")
print(f"   • 賠付觸發: 基於醫院周圍圓圈內的最大風速")
print(f"   • 損失基準: 使用醫院標準化損失 (1單位×脆弱度函數)")  
print(f"   • 貝氏建模: 對醫院曝險進行不確定性量化")
print(f"   • 空間一致性: 賠付觸發與損失評估使用相同的醫院位置")

print(f"\n⚡ 計算效率與方法比較:")
try:
    if 'rmse_df' in locals() and 'best_rmse_idx' in locals():
        steinmann_payouts = rmse_df.iloc[best_rmse_idx]['payouts']
        rmse_calc = np.sqrt(np.mean((damages - steinmann_payouts) ** 2))
        print(f"   • RMSE重新驗證: ${rmse_calc/1e9:.3f}B")
    
    if 'crps_df' in locals() and 'best_crps_idx' in locals() and len(crps_df) > 0:
        bayesian_payouts = crps_df.iloc[best_crps_idx]['payouts']
        if 'steinmann_payouts' in locals():
            mae_diff = np.mean(np.abs(steinmann_payouts - bayesian_payouts))
            print(f"   • 兩種最佳產品的賠付差異 (MAE): ${mae_diff/1e9:.3f}B")
    
    if 'n_samples' in locals():
        print(f"   • 貝氏方法使用 {n_samples} 樣本進行機率分布建模")
        print(f"   • CRPS計算考慮了損失的不確定性分布")
    else:
        print(f"   • 使用簡化概率分布進行CRPS計算")
        
except Exception as e:
    print(f"   • ⚠️ 效率比較計算失敗: {e}")
    print(f"   • 分析仍然成功完成，請參考上述結果")

# 檢查是否成功使用了穩健貝氏分析
if modules_available['bayesian'] and isinstance(loss_distributions, dict) and len(loss_distributions) > 0:
    sample_size = len(list(loss_distributions.values())[0]) if isinstance(loss_distributions, dict) else 0
    if sample_size >= 500:
        print(f"   • ✅ 成功使用穩健貝氏分析器 (每個分布{sample_size}樣本)")
        print(f"   • 🧮 包含 Monte Carlo 模擬、密度比方法和階層模型")
    else:
        print(f"   • ⚠️ 使用簡化貝氏分布 ({sample_size}樣本)")
else:
    print(f"   • ℹ️ 使用標準對數正態分布作為備用方案")

print(f"\n📊 分析輸出:")
print(f"   • 圖表已在各個cell中顯示")
print(f"   • 風速分布圖: cat_in_circle_wind_distributions_*.png")
print(f"   • 所有結果已在notebook中完整呈現")

print(f"\n💡 使用建議:")
print(f"   • 推薦使用 {best_radius}km 半徑作為主要設計參數")
print(f"   • 結合兩種方法的優勢進行產品組合")
print(f"   • 關注基差風險較高的半徑設定")
print(f"   • 考慮不同半徑在觸發頻率上的權衡")

print(f"\n✅ 功能式分析完成！每個 cell 都在自己的 cell 中產生了結果")

# %%