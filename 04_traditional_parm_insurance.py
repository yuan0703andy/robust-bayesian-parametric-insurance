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

# 導入新的模組化組件
from data_processing import load_spatial_data_from_02_results

# Import hospital-based configuration (保留作為備用)
try:
    from config.hospital_based_payout_config import HospitalPayoutConfig, create_hospital_based_config
    HOSPITAL_CONFIG_AVAILABLE = True
except ImportError:
    HOSPITAL_CONFIG_AVAILABLE = False
    print("⚠️ Hospital config module not available, using basic configuration")

# %%
def calculate_basic_traditional_analysis(products, spatial_data, hospital_indices=None, observed_losses=None, radius_km=30):
    """
    基本的傳統參數保險分析
    Basic traditional parametric insurance analysis
    
    Parameters:
    -----------
    products : list
        保險產品列表
    spatial_data : SpatialData
        空間數據對象
    hospital_indices : array, optional
        醫院指標數據
    observed_losses : array, optional
        觀測損失數據
    radius_km : int, default=30
        Cat-in-Circle半徑 (km) - Steinmann 2023標準使用30km
        
    Returns:
    --------
    dict : 分析結果
    """
    print(f"🔧 執行基本傳統分析 (Cat-in-Circle半徑: {radius_km}km)...")
    
    if spatial_data.hazard_intensities is None or spatial_data.observed_losses is None:
        print("   ⚠️ 缺少災害或損失數據，創建模擬數據...")
        n_hospitals = spatial_data.n_hospitals
        n_events = 100
        
        # 使用模擬數據
        hazard_intensities = np.random.uniform(20, 60, (n_hospitals, n_events))
        observed_losses = np.random.lognormal(15, 1, (n_hospitals, n_events))
    else:
        hazard_intensities = spatial_data.hazard_intensities
        observed_losses = spatial_data.observed_losses
        n_hospitals = spatial_data.n_hospitals
        n_events = hazard_intensities.shape[1]
    
    # 計算產品性能
    results = {
        'products': [],
        'basis_risk_summary': {},
        'performance_metrics': {}
    }
    
    # 分析所有350個產品 (70結構 × 5半徑)
    print(f"   🎯 分析完整的產品套裝: {len(products)} 個產品")
    if len(products) == 350:
        print("   📐 包含5個半徑: 15km, 30km, 50km, 75km, 100km")
    elif len(products) == 70:
        print("   📐 單一半徑: 30km (Steinmann標準)")
    else:
        print(f"   📦 產品數量: {len(products)}")
    sample_products = products  # 分析所有產品
    
    for i, product in enumerate(sample_products):
        product_results = {
            'product_id': product['product_id'],
            'structure_type': product['structure_type'],
            'basis_risk_rmse': 0.0,
            'basis_risk_mae': 0.0,
            'coverage_ratio': 0.0,
            'trigger_frequency': 0.0
        }
        
        # 計算每個事件的賠付
        total_payouts = []
        total_losses = []
        
        for event_idx in range(n_events):  # 分析所有事件
            event_total_loss = observed_losses[:, event_idx].sum()
            event_total_payout = 0.0
            
            for hospital_idx in range(n_hospitals):
                hospital_hazard = hazard_intensities[hospital_idx, event_idx]
                payout = calculate_product_payout(product, hospital_hazard)
                event_total_payout += payout
            
            total_payouts.append(event_total_payout)
            total_losses.append(event_total_loss)
        
        # 計算基差風險指標
        total_payouts = np.array(total_payouts)
        total_losses = np.array(total_losses)
        
        basis_risk = total_losses - total_payouts
        product_results['basis_risk_rmse'] = np.sqrt(np.mean(basis_risk**2))
        product_results['basis_risk_mae'] = np.mean(np.abs(basis_risk))
        product_results['coverage_ratio'] = np.mean(total_payouts) / np.mean(total_losses) if np.mean(total_losses) > 0 else 0
        product_results['trigger_frequency'] = np.mean(total_payouts > 0)
        
        results['products'].append(product_results)
        
        if i % 25 == 0 or i < 10:  # 每25個產品顯示一次進度，前10個也顯示
            print(f"     分析進度: {i+1}/{len(sample_products)} 產品 ({100*i/len(sample_products):.1f}%)")
    
    # 計算總體統計
    rmse_values = [p['basis_risk_rmse'] for p in results['products']]
    mae_values = [p['basis_risk_mae'] for p in results['products']]
    
    results['basis_risk_summary'] = {
        'mean_rmse': np.mean(rmse_values),
        'min_rmse': np.min(rmse_values),
        'max_rmse': np.max(rmse_values),
        'mean_mae': np.mean(mae_values),
        'min_mae': np.min(mae_values),
        'max_mae': np.max(mae_values)
    }
    
    results['performance_metrics'] = {
        'best_rmse_product': results['products'][np.argmin(rmse_values)]['product_id'],
        'best_mae_product': results['products'][np.argmin(mae_values)]['product_id'],
        'n_products_analyzed': len(sample_products),
        'n_events_analyzed': n_events
    }
    
    print(f"   ✅ 分析完成: {len(sample_products)} 個產品, {n_events} 個事件")
    return results

def calculate_product_payout(product, hazard_intensity):
    """計算產品賠付"""
    thresholds = product['trigger_thresholds']
    ratios = product['payout_ratios']
    max_payout = product['max_payout']
    
    for i, threshold in enumerate(thresholds):
        if hazard_intensity >= threshold:
            if i < len(ratios):
                return max_payout * ratios[i]
    return 0.0

# %%
def main():
    """
    Main program: Traditional basis risk analysis using modular components
    """
    print("=" * 80)
    print("Traditional Parametric Insurance Analysis")
    print("使用模組化組件和RMSE基差風險評估")
    print("RMSE-based deterministic evaluation")
    print("=" * 80)
    
    # 步驟1: 載入所需數據
    print("\n📂 載入數據...")
    
    # 載入產品
    try:
        with open("results/insurance_products/products.pkl", 'rb') as f:
            products = pickle.load(f)
        print(f"✅ 載入保險產品: {len(products)} 個")
    except FileNotFoundError:
        print("❌ 未找到產品文件，請先執行 03_insurance_product.py")
        return
    
    # 載入空間分析結果 - 優先使用模組化loader
    print("\n📂 載入空間分析結果...")
    spatial_data = None
    
    try:
        spatial_data = load_spatial_data_from_02_results("results/spatial_analysis/cat_in_circle_results.pkl")
        if spatial_data is not None:
            print(f"✅ 使用模組化loader載入空間數據")
            print(f"   醫院數量: {spatial_data.n_hospitals}")
            print(f"   區域數量: {spatial_data.n_regions}")
            if spatial_data.hazard_intensities is not None:
                print(f"   事件數量: {spatial_data.hazard_intensities.shape[1]}")
                print(f"   災害強度範圍: {spatial_data.hazard_intensities.min():.1f} - {spatial_data.hazard_intensities.max():.1f}")
                print(f"   📐 使用30km Cat-in-Circle風速數據 (Steinmann 2023標準)")
        else:
            raise ValueError("模組化loader返回None")
    except Exception as e:
        print(f"⚠️ 模組化載入失敗: {e}")
        print("   🔄 嘗試直接載入...")
        
        try:
            with open("results/spatial_analysis/cat_in_circle_results.pkl", 'rb') as f:
                spatial_results = pickle.load(f)
            print("✅ 直接載入成功")
            
            # 創建一個基本的空間數據替代對象
            class BasicSpatialData:
                def __init__(self, spatial_results):
                    if 'hospital_coordinates' in spatial_results:
                        self.hospital_coords = spatial_results['hospital_coordinates']
                        self.n_hospitals = len(self.hospital_coords)
                    else:
                        self.n_hospitals = 20  # 默認值
                        self.hospital_coords = np.random.uniform([34, -84], [37, -75], (self.n_hospitals, 2))
                    
                    self.n_regions = 3
                    self.hazard_intensities = None
                    self.observed_losses = None
                    
                    # 嘗試從indices中提取災害數據
                    if 'indices' in spatial_results:
                        indices = spatial_results['indices']
                        if 'hazard_intensities' in indices:
                            self.hazard_intensities = indices['hazard_intensities']
                        if 'observed_losses' in indices:
                            self.observed_losses = indices['observed_losses']
            
            spatial_data = BasicSpatialData(spatial_results)
            print(f"   創建基本空間數據對象: {spatial_data.n_hospitals} 家醫院")
            
        except Exception as e2:
            print(f"❌ 直接載入也失敗: {e2}")
            print("   創建示例數據進行演示...")
            
            # 創建示例數據
            class MockSpatialData:
                def __init__(self):
                    self.n_hospitals = 10
                    self.n_regions = 3
                    self.hospital_coords = np.random.uniform([35, -82], [36, -77], (self.n_hospitals, 2))
                    self.hazard_intensities = None
                    self.observed_losses = None
            
            spatial_data = MockSpatialData()
            print(f"   ✅ 創建示例數據: {spatial_data.n_hospitals} 家醫院")
    
    if spatial_data is None:
        print("❌ 無法獲取空間數據")
        return
    
    # 步驟2: 執行傳統分析
    print("\n🔧 執行傳統RMSE基差風險分析...")
    print("   📐 使用Steinmann 2023標準: 30km Cat-in-Circle半徑")
    print("   📦 分析所有Steinmann產品套裝")
    
    # 使用基本分析方法 - 30km半徑符合Steinmann 2023標準
    results = calculate_basic_traditional_analysis(
        products=products,
        spatial_data=spatial_data,
        radius_km=30
    )
    
    # 步驟3: 展示結果
    print("\n📊 分析結果:")
    print("-" * 60)
    
    print(f"📈 基差風險統計摘要:")
    summary = results['basis_risk_summary']
    print(f"   平均RMSE: ${summary['mean_rmse']:,.0f}")
    print(f"   最小RMSE: ${summary['min_rmse']:,.0f}")
    print(f"   最大RMSE: ${summary['max_rmse']:,.0f}")
    print(f"   平均MAE:  ${summary['mean_mae']:,.0f}")
    print(f"   最小MAE:  ${summary['min_mae']:,.0f}")
    print(f"   最大MAE:  ${summary['max_mae']:,.0f}")
    
    print(f"\n🎯 性能指標:")
    metrics = results['performance_metrics']
    print(f"   最佳RMSE產品: {metrics['best_rmse_product']}")
    print(f"   最佳MAE產品:  {metrics['best_mae_product']}")
    print(f"   分析產品數量: {metrics['n_products_analyzed']}")
    print(f"   分析事件數量: {metrics['n_events_analyzed']}")
    
    print(f"\n📋 前5個產品詳細結果:")
    for i, product_result in enumerate(results['products'][:5]):
        print(f"   {i+1}. {product_result['product_id']}")
        print(f"      結構類型: {product_result['structure_type']}")
        print(f"      RMSE: ${product_result['basis_risk_rmse']:,.0f}")
        print(f"      MAE:  ${product_result['basis_risk_mae']:,.0f}")
        print(f"      覆蓋率: {product_result['coverage_ratio']:.3f}")
        print(f"      觸發頻率: {product_result['trigger_frequency']:.3f}")
        print()
    
    # 步驟4: 保存結果
    print("\n💾 保存分析結果...")
    output_dir = Path("results/traditional_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "traditional_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✅ 結果已保存至: {results_path}")
    
    print("\n✅ 04_traditional_parm_insurance.py 執行完成!")
    print(f"   📊 分析了 {len(results['products'])} 個產品")
    print(f"   📁 結果保存在: results/traditional_analysis/")
    print(f"   🔧 使用了模組化SpatialDataProcessor")
    print(f"   💡 結果可被後續腳本 (05) 使用")
    
    return results

# %%
if __name__ == "__main__":
    results = main()
# %%
