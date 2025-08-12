#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_hospital_based_payout_analysis.py
=====================================
Hospital-Based Maximum Payout Configuration and Analysis
基於醫院的最大賠付配置與分析

This script demonstrates how to configure maximum payouts based on hospital exposure values
本腳本展示如何基於醫院曝險值配置最大賠付
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import hospital-based configuration 匯入醫院基礎配置
from config.hospital_based_payout_config import (
    HospitalPayoutConfig,
    create_hospital_based_config
)

# Import hospital extraction 匯入醫院提取
from exposure_modeling.hospital_osm_extraction import get_nc_hospitals

# Import insurance framework 匯入保險框架
from insurance_analysis_refactored.core.saffir_simpson_products import (
    SaffirSimpsonProductGenerator,
    SteinmannProductConfig
)

print("🏥 Hospital-Based Maximum Payout Configuration")
print("   基於醫院的最大賠付配置分析")
print("=" * 100)

# %%
# Step 1: Configure Hospital Values 配置醫院價值
print("\n📊 Step 1: Configure Hospital Values 配置醫院價值")
print("-" * 50)

# 創建醫院價值配置
hospital_value_config = {
    'base_value': 1e7,  # $10M USD per hospital 每家醫院1000萬美元
    'type_multipliers': {
        'general': 1.0,        # 一般醫院
        'emergency': 2.0,      # 急救中心 (2x value)
        'specialty': 1.5,      # 專科醫院 (1.5x value)
        'regional': 2.5,       # 區域醫療中心 (2.5x value)
        'university': 3.0,     # 大學醫院 (3x value)
        'community': 0.8       # 社區醫院 (0.8x value)
    },
    'use_real_values': False  # 若為True，將根據醫院類型分配不同價值
}

print(f"基礎醫院價值: ${hospital_value_config['base_value']:,.0f} USD")
print("醫院類型乘數:")
for htype, multiplier in hospital_value_config['type_multipliers'].items():
    value = hospital_value_config['base_value'] * multiplier
    print(f"  - {htype}: {multiplier}x (${value:,.0f})")

# %%
# Step 2: Extract Hospital Data with Value Config 提取醫院數據
print("\n🏥 Step 2: Extract Hospital Data 提取醫院數據")
print("-" * 50)

# 獲取醫院數據 (使用模擬數據進行示範)
gdf_hospitals, hospital_exposures = get_nc_hospitals(
    use_mock=True,
    create_exposures=True,
    visualize=False,
    value_config=hospital_value_config
)

n_hospitals = len(gdf_hospitals)
print(f"✅ 提取到 {n_hospitals} 家醫院")

# 計算總曝險值
if hospital_exposures and hasattr(hospital_exposures, 'value'):
    total_exposure = hospital_exposures.value.sum() * hospital_value_config['base_value']
else:
    total_exposure = n_hospitals * hospital_value_config['base_value']

print(f"💰 總曝險值: ${total_exposure:,.0f} USD")

# %%
# Step 3: Create Hospital-Based Payout Configuration 創建基於醫院的賠付配置
print("\n💰 Step 3: Configure Hospital-Based Payouts 配置基於醫院的賠付")
print("-" * 50)

# 創建賠付配置
payout_config = HospitalPayoutConfig(
    n_hospitals=n_hospitals,
    base_hospital_value=hospital_value_config['base_value'],
    # 設定不同產品類型的覆蓋比例
    coverage_ratios={
        'single': 0.25,      # 單閾值: 覆蓋25%總曝險
        'double': 0.40,      # 雙閾值: 覆蓋40%總曝險
        'triple': 0.60,      # 三閾值: 覆蓋60%總曝險
        'quadruple': 0.80    # 四閾值: 覆蓋80%總曝險
    },
    # 基於分析半徑的調整因子
    radius_multipliers={
        15: 1.5,   # 15km: 局部高密度賠付
        30: 1.2,   # 30km: 標準密度
        50: 1.0,   # 50km: 基準
        75: 0.9,   # 75km: 較低密度
        100: 0.8   # 100km: 區域性低密度
    }
)

# 獲取不同半徑的最大賠付金額
radii = [15, 30, 50, 75, 100]
print("\n🎯 不同分析半徑的最大賠付金額:")
print("Radius | Single    | Double    | Triple    | Quadruple")
print("-" * 60)

for radius in radii:
    max_payouts = payout_config.get_max_payout_amounts(total_exposure, radius)
    print(f"{radius:3d}km | ", end="")
    for ptype in ['single', 'double', 'triple', 'quadruple']:
        amount = max_payouts[ptype]
        print(f"${amount/1e6:7.1f}M | ", end="")
    print()

# %%
# Step 4: Generate Steinmann Products with Hospital-Based Payouts
print("\n🔧 Step 4: Generate Products with Hospital-Based Payouts")
print("-" * 50)

# 選擇一個標準半徑 (50km)
selected_radius = 50
print(f"使用分析半徑: {selected_radius}km")

# 獲取Steinmann配置與醫院賠付整合
steinmann_config = payout_config.get_steinmann_config_with_hospital_payouts(
    hospital_df=gdf_hospitals if hasattr(gdf_hospitals, 'iterrows') else None,
    radius_km=selected_radius
)

# 生成產品
generator = SaffirSimpsonProductGenerator(steinmann_config)
products = generator.generate_all_steinmann_products()

print(f"\n✅ 成功生成 {len(products)} 個產品")

# %%
# Step 5: Display Product Summary 顯示產品摘要
print("\n📋 Step 5: Product Summary 產品摘要")
print("-" * 50)

# 創建產品摘要
product_summary = []
for product in products[:10]:  # 顯示前10個產品
    product_summary.append({
        'Product ID': product.product_id,
        'Type': product.structure_type,
        'Thresholds': len(product.thresholds),
        'First Threshold': f"{product.thresholds[0]:.1f} m/s" if product.thresholds else "N/A",
        'Max Payout': f"${product.max_payout/1e6:.1f}M",
        'Payout Steps': ', '.join([f"{p*100:.0f}%" for p in product.payouts])
    })

df_summary = pd.DataFrame(product_summary)
print(df_summary.to_string(index=False))

# %%
# Step 6: Dynamic Payout Example 動態賠付範例
print("\n🔄 Step 6: Dynamic Payout Calculation Example")
print("-" * 50)

# 模擬受影響醫院
affected_hospitals = [
    {'name': 'Hospital A', 'hospital_type': 'emergency'},
    {'name': 'Hospital B', 'hospital_type': 'general'},
    {'name': 'Hospital C', 'hospital_type': 'university'},
]

all_hospitals = [{'hospital_type': 'general'}] * n_hospitals  # 簡化示例

# 計算動態賠付
base_payout = 50e6  # $50M base payout
dynamic_payout = payout_config.calculate_dynamic_payout(
    affected_hospitals=affected_hospitals,
    total_hospitals=all_hospitals,
    base_payout=base_payout
)

print(f"受影響醫院: {len(affected_hospitals)}/{len(all_hospitals)}")
print(f"基礎賠付: ${base_payout/1e6:.1f}M")
print(f"動態調整後賠付: ${dynamic_payout/1e6:.1f}M")
print(f"調整因子: {dynamic_payout/base_payout:.2%}")

# %%
# Step 7: Save Configuration 保存配置
print("\n💾 Step 7: Save Configuration 保存配置")
print("-" * 50)

# 保存配置到文件
config_output = {
    'hospital_configuration': {
        'n_hospitals': n_hospitals,
        'base_value_per_hospital': hospital_value_config['base_value'],
        'total_exposure': total_exposure,
        'value_config': hospital_value_config
    },
    'payout_configuration': {
        'coverage_ratios': payout_config.coverage_ratios,
        'radius_multipliers': payout_config.radius_multipliers,
        'selected_radius': selected_radius
    },
    'max_payout_amounts': {
        radius: payout_config.get_max_payout_amounts(total_exposure, radius)
        for radius in radii
    },
    'products_generated': len(products)
}

# 保存為pickle
import pickle
output_dir = Path('results/hospital_based_payouts')
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / 'hospital_payout_config.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(config_output, f)

print(f"✅ 配置已保存至: {output_file}")

# %%
# Summary 總結
print("\n" + "=" * 100)
print("🎯 Hospital-Based Payout Configuration Complete!")
print("=" * 100)
print("\n關鍵結果:")
print(f"• 醫院數量: {n_hospitals}")
print(f"• 每家醫院基礎價值: ${hospital_value_config['base_value']/1e6:.1f}M")
print(f"• 總曝險值: ${total_exposure/1e6:.1f}M")
print(f"• 生成產品數: {len(products)}")
print(f"• 最大賠付範圍: ${min([p.max_payout for p in products])/1e6:.1f}M - ${max([p.max_payout for p in products])/1e6:.1f}M")

print("\n💡 建議:")
print("1. 根據實際醫院重要性調整 type_multipliers")
print("2. 基於歷史損失數據優化 coverage_ratios")
print("3. 考慮地理分布調整 radius_multipliers")
print("4. 定期更新醫院數據和價值評估")

print("\n✅ Analysis Complete! 分析完成！")