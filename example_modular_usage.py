#!/usr/bin/env python3
"""
Example: How to Use the Modular Hierarchical Bayesian Framework
示例：如何使用模組化階層貝葉斯框架

展示如何在現有分析中使用新的模組化組件：
1. SpatialDataProcessor - 處理醫院空間數據和Cat-in-Circle結果  
2. build_hierarchical_model - 構建正確的4層階層模型
3. PortfolioOptimizer - 多尺度投資組合優化

這個文件可以被現有的05_complete_integrated_framework_v3_cellbased.py導入使用
"""

import numpy as np
import pymc as pm
import pickle
from pathlib import Path


def load_and_process_spatial_data():
    """
    載入並處理空間數據
    替代05_complete_integrated_framework_v3_cellbased.py中的硬編碼部分
    
    Returns:
    --------
    spatial_data : SpatialData or None
        處理後的空間數據
    """
    try:
        from robust_hierarchical_bayesian_simulation.spatial_data_processor import (
            SpatialDataProcessor, load_spatial_data_from_02_results
        )
        
        print("🔧 使用模組化空間數據處理...")
        
        # 方法1: 從02_spatial_analysis.py結果載入
        spatial_analysis_path = "results/spatial_analysis/cat_in_circle_results.pkl"
        if Path(spatial_analysis_path).exists():
            spatial_data = load_spatial_data_from_02_results(spatial_analysis_path)
            if spatial_data is not None:
                return spatial_data
        
        # 方法2: 手動處理空間數據
        print("   📍 手動處理空間數據...")
        processor = SpatialDataProcessor()
        
        # 示例醫院座標 (北卡羅萊納州)
        hospital_coords = np.array([
            [35.7796, -78.6382],  # Raleigh
            [36.0726, -79.7920],  # Greensboro  
            [35.2271, -80.8431],  # Charlotte
            [35.0527, -78.8784],  # Fayetteville
            [35.9132, -79.0558],  # Chapel Hill
            [36.1349, -80.2676],  # Winston-Salem
            [35.6127, -77.3663],  # Greenville
            [34.2257, -77.9447],  # Wilmington
            [35.6069, -82.5540],  # Asheville
            [36.0999, -78.7837],  # Durham
        ])
        
        spatial_data = processor.process_hospital_spatial_data(
            hospital_coords, 
            n_regions=3,
            region_method="risk_based"
        )
        
        # 添加模擬的Cat-in-Circle數據
        n_hospitals = spatial_data.n_hospitals
        n_events = 100
        
        # 模擬災害強度 (mph)
        hazard_intensities = np.random.uniform(20, 70, (n_hospitals, n_events))
        
        # 模擬曝險價值 (根據醫院重要性)
        base_values = np.array([5e7, 3e7, 8e7, 2e7, 4e7, 3e7, 2.5e7, 2e7, 2.5e7, 4.5e7])[:n_hospitals]
        exposure_values = base_values * np.random.uniform(0.8, 1.2, n_hospitals)
        
        # 模擬觀測損失
        observed_losses = np.random.lognormal(15, 1.5, (n_hospitals, n_events))
        
        spatial_data = processor.add_cat_in_circle_data(
            hazard_intensities, exposure_values, observed_losses
        )
        
        return spatial_data
        
    except ImportError as e:
        print(f"⚠️ 模組載入失敗: {e}")
        return None


def build_correct_hierarchical_model(spatial_data, contamination_epsilon=0.05):
    """
    構建正確的階層模型
    替代visualize_bayesian_model.py中的硬編碼問題
    
    Parameters:
    -----------
    spatial_data : SpatialData
        空間數據
    contamination_epsilon : float
        ε-contamination參數
        
    Returns:
    --------
    pm.Model : PyMC階層模型
    """
    try:
        from robust_hierarchical_bayesian_simulation.hierarchical_model_builder import (
            build_hierarchical_model, validate_model_inputs
        )
        
        print("🏗️ 構建正確的4層階層模型...")
        
        # 驗證輸入
        if not validate_model_inputs(spatial_data):
            print("❌ 模型輸入驗證失敗")
            return None
        
        # 構建模型
        model = build_hierarchical_model(
            spatial_data, 
            contamination_epsilon=contamination_epsilon,
            model_name="corrected_hierarchical_model"
        )
        
        return model
        
    except ImportError as e:
        print(f"⚠️ 階層模型構建器載入失敗: {e}")
        return None


def run_portfolio_optimization(spatial_data, insurance_products):
    """
    運行多尺度投資組合優化
    實現醫院級建模 + 投資組合級優化
    
    Parameters:
    -----------
    spatial_data : SpatialData
        空間數據
    insurance_products : List[Dict]
        保險產品列表
        
    Returns:
    --------
    ProductAllocation : 最優分配結果
    """
    try:
        from robust_hierarchical_bayesian_simulation.portfolio_optimizer import PortfolioOptimizer
        
        print("🎯 運行多尺度投資組合優化...")
        
        optimizer = PortfolioOptimizer(
            spatial_data, 
            insurance_products,
            loss_function="weighted_asymmetric"
        )
        
        # 執行優化
        optimal_allocation = optimizer.optimize_portfolio_allocation(
            method="discrete_search",
            n_monte_carlo=500
        )
        
        # 分析空間相關性影響
        # 需要提取產品索引，而不是醫院索引
        hospital_product_indices = []
        for hospital_idx in range(spatial_data.n_hospitals):
            product_id = optimal_allocation.hospital_products.get(hospital_idx, 'product_0')
            # 找到產品在產品列表中的索引
            product_idx = 0
            for i, product in enumerate(insurance_products):
                if product.get('product_id', f'product_{i}') == product_id:
                    product_idx = i
                    break
            hospital_product_indices.append(product_idx)
        
        spatial_analysis = optimizer.analyze_spatial_correlation_impact(
            np.array(hospital_product_indices)
        )
        
        return optimal_allocation, spatial_analysis
        
    except ImportError as e:
        print(f"⚠️ 投資組合優化器載入失敗: {e}")
        return None, None


def demonstrate_complete_workflow():
    """
    完整工作流程演示
    展示如何將所有模組整合使用
    """
    print("🚀 模組化階層貝葉斯框架完整演示")
    print("=" * 60)
    
    # 步驟1: 處理空間數據
    spatial_data = load_and_process_spatial_data()
    if spatial_data is None:
        print("❌ 空間數據處理失敗")
        return
    
    # 步驟2: 構建階層模型  
    hierarchical_model = build_correct_hierarchical_model(spatial_data, contamination_epsilon=0.08)
    if hierarchical_model is None:
        print("❌ 階層模型構建失敗")
        return
        
    print(f"✅ 階層模型構建成功: {len(hierarchical_model.free_RVs)} 個參數")
    
    # 步驟3: 載入保險產品
    try:
        with open("results/insurance_products/products.pkl", 'rb') as f:
            insurance_products = pickle.load(f)
        print(f"✅ 載入保險產品: {len(insurance_products)} 個")
    except:
        # 創建示例產品
        insurance_products = create_example_products()
        print(f"✅ 創建示例產品: {len(insurance_products)} 個")
    
    # 步驟4: 投資組合優化
    optimal_allocation, spatial_analysis = run_portfolio_optimization(spatial_data, insurance_products)
    
    if optimal_allocation is not None:
        print(f"\n🎯 投資組合優化結果:")
        print(f"   投資組合基差風險: {optimal_allocation.portfolio_basis_risk:.6f}")
        print(f"   覆蓋率: {optimal_allocation.coverage_ratio:.3f}")
        print(f"   總期望賠付: ${optimal_allocation.total_expected_payout:,.0f}")
        print(f"   總期望損失: ${optimal_allocation.total_expected_loss:,.0f}")
        print(f"   醫院產品分配: {optimal_allocation.hospital_products}")
        
        if spatial_analysis is not None:
            clustering_effect = spatial_analysis.get('spatial_clustering_effect', 0)
            print(f"   空間集群效應: {clustering_effect:.3f}")
    
    # 步驟5: 模型擬合演示 (小樣本測試)
    print(f"\n🔄 MCMC採樣演示...")
    try:
        with hierarchical_model:
            trace = pm.sample(draws=100, tune=100, chains=2, return_inferencedata=True)
        
        from robust_hierarchical_bayesian_simulation.hierarchical_model_builder import get_portfolio_loss_predictions
        
        portfolio_predictions = get_portfolio_loss_predictions(trace, spatial_data, [0, 1, 2])
        
        print(f"✅ MCMC採樣完成")
        print(f"   事件0投資組合損失: ${portfolio_predictions['event_0']['mean']:,.0f} ± ${portfolio_predictions['event_0']['std']:,.0f}")
        
    except Exception as e:
        print(f"⚠️ MCMC採樣演示跳過: {e}")
    
    print(f"\n✅ 完整工作流程演示完成!")
    print(f"💡 現在可以在05_complete_integrated_framework_v3_cellbased.py中導入使用這些函數")


def create_example_products():
    """創建示例保險產品"""
    products = []
    
    # Steinmann 2023 標準產品示例
    radii = [15, 30, 50, 75, 100]
    thresholds_sets = [
        ([40], [1.0]),          # Single threshold
        ([35, 50], [0.5, 1.0]), # Dual threshold  
        ([30, 45, 60], [0.33, 0.67, 1.0]), # Triple threshold
    ]
    
    for radius in radii:
        for i, (thresholds, ratios) in enumerate(thresholds_sets):
            products.append({
                "product_id": f"R{radius}_T{len(thresholds)}_{i+1}",
                "name": f"{radius}km {len(thresholds)}-threshold product {i+1}",
                "trigger_thresholds": thresholds,
                "payout_ratios": ratios,
                "max_payout": 1e8,  # $100M
                "radius_km": radius,
                "structure_type": {1: "single", 2: "dual", 3: "triple"}[len(thresholds)]
            })
    
    return products


# 使用範例
if __name__ == "__main__":
    demonstrate_complete_workflow()