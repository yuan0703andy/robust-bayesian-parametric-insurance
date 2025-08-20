#!/usr/bin/env python3
"""
可視化貝氏模型結構
Visualize Bayesian Model Structure using pm.model_to_graphviz

使用範例：
python visualize_bayesian_model.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bayesian.parametric_bayesian_hierarchy import (
    VulnerabilityData, ModelSpec, MCMCConfig, ParametricHierarchicalModel,
    LikelihoodFamily, PriorScenario, VulnerabilityFunctionType
)

try:
    import pymc as pm
    import graphviz
    HAS_GRAPHVIZ = True
    print("✅ PyMC 和 graphviz 可用")
except ImportError as e:
    HAS_GRAPHVIZ = False
    print(f"❌ 缺少依賴: {e}")
    print("請安裝: pip install graphviz")

def visualize_spatial_bayesian_model():
    """可視化空間階層貝氏模型"""
    
    if not HAS_GRAPHVIZ:
        print("❌ 無法可視化，請安裝 graphviz")
        return
    
    print("🗺️ 創建空間階層貝氏模型用於可視化...")
    
    # 創建模擬數據
    np.random.seed(42)
    n_events = 20  # 較小規模用於展示
    n_hospitals = 5
    
    # 模擬醫院座標 (NC)
    hospital_coords = np.array([
        [36.0153, -78.9384],  # Duke
        [35.9049, -79.0469],  # UNC
        [35.8043, -78.6569],  # Rex
        [35.7520, -78.6037],  # WakeMed
        [35.2045, -80.8395],  # Carolinas
    ])
    
    # 模擬災害數據
    wind_speeds = np.random.uniform(25, 80, n_events)
    building_values = np.random.uniform(1e6, 1e8, n_events)
    
    # 簡單脆弱度關係
    vulnerability = 0.001 * np.maximum(wind_speeds - 25, 0)**2
    true_losses = building_values * vulnerability
    observed_losses = np.maximum(true_losses * (1 + np.random.normal(0, 0.1, n_events)), 0)
    
    # 創建脆弱度數據
    vulnerability_data = VulnerabilityData(
        hazard_intensities=wind_speeds,
        exposure_values=building_values,
        observed_losses=observed_losses,
        hospital_coordinates=hospital_coords,
        hospital_names=[f"Hospital_{i+1}" for i in range(n_hospitals)]
    )
    
    # 配置空間階層模型
    model_spec = ModelSpec(
        likelihood_family=LikelihoodFamily.LOGNORMAL,
        prior_scenario=PriorScenario.WEAK_INFORMATIVE,
        vulnerability_type=VulnerabilityFunctionType.EMANUEL,
        include_spatial_effects=True,      # 啟用空間效應
        include_region_effects=True,       # 啟用區域效應
        spatial_covariance_function="exponential"
    )
    
    mcmc_config = MCMCConfig(
        n_samples=100,  # 小樣本用於快速演示
        n_warmup=50,
        n_chains=1
    )
    
    print("🔧 構建模型結構（不執行MCMC）...")
    
    # 修改的模型創建函數，只構建不採樣
    def create_model_for_visualization():
        """創建模型結構用於可視化"""
        
        hazard = vulnerability_data.hazard_intensities
        exposure = vulnerability_data.exposure_values
        losses = vulnerability_data.observed_losses
        coords = vulnerability_data.hospital_coordinates
        n_hospitals = len(coords)
        
        # 計算距離矩陣（簡化）
        from scipy.spatial.distance import pdist, squareform
        
        def haversine_distance(coord1, coord2):
            R = 6371  # 地球半徑 km
            lat1, lon1 = np.radians(coord1)
            lat2, lon2 = np.radians(coord2)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            return 2 * R * np.arcsin(np.sqrt(a))
        
        # 計算醫院間距離
        distance_matrix = np.zeros((n_hospitals, n_hospitals))
        for i in range(n_hospitals):
            for j in range(n_hospitals):
                if i != j:
                    distance_matrix[i, j] = haversine_distance(coords[i], coords[j])
        
        with pm.Model() as spatial_model:
            print("   🏗️ 構建空間階層結構...")
            
            # Level 4: 全域超參數
            alpha_global = pm.Normal("alpha_global", mu=0, sigma=2)
            
            # 空間參數
            rho_spatial = pm.Gamma("rho_spatial", alpha=2, beta=0.1)
            sigma2_spatial = pm.Gamma("sigma2_spatial", alpha=2, beta=1)
            nugget = pm.Uniform("nugget", lower=0.01, upper=0.5)
            
            # Level 3: 區域效應 α_r(i)
            n_regions = 3
            alpha_region = pm.Normal("alpha_region", mu=alpha_global, sigma=0.5, shape=n_regions)
            
            # TODO: 區域分配應該基於真實地理或風險區域
            # 當前為示例，實際使用時應傳入真實的region_assignments
            region_mapping = np.array([0, 0, 1, 1, 2])  # 5個醫院分配到3個區域
            hospital_region_effects = alpha_region[region_mapping]
            
            # Level 2: 空間隨機效應 δ_i（核心！）
            # TODO: distance_matrix應該來自真實醫院座標計算
            # 當前為示例，實際使用時應傳入真實的distance_matrix
            cov_matrix = sigma2_spatial * pm.math.exp(-distance_matrix / rho_spatial)
            cov_matrix_stable = cov_matrix + nugget * np.eye(n_hospitals)
            
            delta_spatial = pm.MvNormal("delta_spatial", mu=0, cov=cov_matrix_stable, shape=n_hospitals)
            
            # Level 1: 個體醫院效應 γ_i
            gamma_individual = pm.Normal("gamma_individual", mu=0, sigma=0.2, shape=n_hospitals)
            
            # 組合脆弱度參數：β_i = α_r(i) + δ_i + γ_i
            beta_vulnerability = pm.Deterministic("beta_vulnerability", 
                                                hospital_region_effects + delta_spatial + gamma_individual)
            
            # Emanuel脆弱度函數
            H_threshold = 25.7
            vulnerability_power = pm.Gamma("vulnerability_power", alpha=2, beta=0.5)
            
            # 簡化：只使用第一個醫院的beta值（為了可視化）
            expected_losses = pm.math.switch(
                hazard > H_threshold,
                exposure * pm.math.exp(beta_vulnerability[0]) * pm.math.power(hazard - H_threshold, vulnerability_power),
                0.0
            )
            
            # 觀測模型
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=1e6)
            expected_losses_positive = pm.math.maximum(expected_losses, 1.0)
            
            y_obs = pm.LogNormal("y_obs", 
                               mu=pm.math.log(expected_losses_positive), 
                               sigma=sigma_obs/expected_losses_positive, 
                               observed=losses)
            
            print("   ✅ 模型結構構建完成")
            return spatial_model
    
    # 創建模型
    model = create_model_for_visualization()
    
    print("📊 生成模型圖形...")
    
    try:
        # 使用 pm.model_to_graphviz 生成圖形
        graph = pm.model_to_graphviz(model)
        
        # 保存圖形
        output_file = "spatial_bayesian_model_structure"
        graph.render(output_file, format='png', cleanup=True)
        
        print(f"✅ 模型結構圖已保存: {output_file}.png")
        
        # 也保存為 PDF
        graph.render(output_file + "_pdf", format='pdf', cleanup=True)
        print(f"✅ 模型結構圖已保存: {output_file}_pdf.pdf")
        
        # 顯示圖形信息
        print(f"\n📈 模型結構摘要:")
        print(f"   節點數量: {len(graph.body)}")
        print(f"   包含空間效應: ✅ delta_spatial")
        print(f"   包含區域效應: ✅ alpha_region") 
        print(f"   包含個體效應: ✅ gamma_individual")
        print(f"   脆弱度組合: ✅ beta_vulnerability = α_r(i) + δ_i + γ_i")
        
        # 打印源碼（可選）
        print(f"\n🔍 Graphviz 源碼:")
        print(graph.source[:500] + "..." if len(graph.source) > 500 else graph.source)
        
        return graph
        
    except Exception as e:
        print(f"❌ 圖形生成失敗: {e}")
        print("可能需要安裝系統級 graphviz:")
        print("   macOS: brew install graphviz")
        print("   Ubuntu: sudo apt-get install graphviz")
        print("   Windows: 從 https://graphviz.org/download/ 下載")
        return None

def main():
    """主函數"""
    print("🎨 PyMC 模型可視化工具")
    print("=" * 40)
    
    print("🗺️ 可視化空間階層貝氏模型...")
    graph = visualize_spatial_bayesian_model()
    
    if graph:
        print("\n🎉 可視化完成！")
        print("✅ 已生成您的 β_i = α_r(i) + δ_i + γ_i 階層結構圖")
        print("📁 檢查當前目錄的 .png 和 .pdf 文件")
        
        print(f"\n💡 模型解釋:")
        print("   🌐 alpha_global: 全域均值")
        print("   🏠 alpha_region: 區域效應（東部/中部/山區）")
        print("   🗺️ delta_spatial: 空間相關隨機效應") 
        print("   🏥 gamma_individual: 個體醫院效應")
        print("   🧬 beta_vulnerability: 組合脆弱度參數")
        print("   📊 y_obs: 觀測損失")
    else:
        print("❌ 可視化失敗，請檢查依賴安裝")

if __name__ == "__main__":
    main()