#!/usr/bin/env python3
"""
Hierarchical Model Builder
階層模型構建器

修正現有visualize_bayesian_model.py中的問題：
1. 硬編碼的region_mapping
2. 未定義的distance_matrix  
3. 沒有整合Cat-in-Circle數據作為H_ij

提供正確的4層階層結構實現，可被現有代碼import使用

用法：
from robust_hierarchical_bayesian_simulation.hierarchical_model_builder import build_hierarchical_model
from robust_hierarchical_bayesian_simulation.spatial_data_processor import SpatialDataProcessor

spatial_data = processor.process_hospital_spatial_data(coords)
model = build_hierarchical_model(spatial_data, contamination_epsilon=0.05)
"""

import numpy as np
import pymc as pm
from typing import Dict, Optional, Tuple
from .spatial_data_processor import SpatialData


def build_hierarchical_model(spatial_data: SpatialData,
                           contamination_epsilon: float = 0.0,
                           emanuel_threshold: float = 25.7,
                           model_name: str = "hierarchical_model") -> pm.Model:
    """
    構建正確的4層位置特定階層模型
    
    Parameters:
    -----------
    spatial_data : SpatialData
        空間數據 (包含醫院座標、距離矩陣、區域分配、Cat-in-Circle數據)
    contamination_epsilon : float
        ε-contamination參數
    emanuel_threshold : float
        Emanuel脆弱度函數閾值 (mph)
    model_name : str
        模型名稱
        
    Returns:
    --------
    pm.Model : PyMC階層模型
    """
    if spatial_data.hazard_intensities is None:
        raise ValueError("spatial_data缺少hazard_intensities，請調用add_cat_in_circle_data()")
    
    n_hospitals = spatial_data.n_hospitals
    n_regions = spatial_data.n_regions
    n_events = spatial_data.hazard_intensities.shape[1]
    
    print(f"🏗️ 構建{model_name}: {n_hospitals}醫院, {n_events}事件, {n_regions}區域")
    
    with pm.Model(name=model_name) as model:
        
        # =================================================================
        # Level 4: 超參數層 (Hyperparameter Level)
        # =================================================================
        print("   Level 4: 超參數層")
        
        # 變異數參數
        σ_obs = pm.HalfCauchy("σ_obs", beta=2.5)
        σ_α = pm.HalfCauchy("σ_α", beta=2.5)
        σ_γ = pm.HalfCauchy("σ_γ", beta=2.5)
        σ_δ = pm.HalfCauchy("σ_δ", beta=2.5)
        
        # 空間相關範圍參數
        ρ_spatial = pm.Lognormal("ρ_spatial", mu=np.log(50), sigma=0.5)
        
        # Emanuel脆弱度函數參數
        vulnerability_power = pm.Gamma("vulnerability_power", alpha=2, beta=0.5)
        
        # =================================================================
        # Level 3: 參數層 (Parameter Level) 
        # =================================================================
        print("   Level 3: 參數層")
        
        # 區域平均效應 α_r
        α_region = pm.Normal("α_region", mu=0, sigma=σ_α, shape=n_regions)
        
        # 非結構化個體隨機效應 γ_i
        γ_individual = pm.Normal("γ_individual", mu=0, sigma=σ_γ, shape=n_hospitals)
        
        # 空間結構化隨機效應 δ_i
        # 使用真實距離矩陣構建協方差
        cov_matrix = σ_δ**2 * pm.math.exp(-spatial_data.distance_matrix / ρ_spatial)
        nugget = 0.01
        cov_matrix_stable = cov_matrix + nugget * np.eye(n_hospitals)
        
        δ_spatial = pm.MvNormal("δ_spatial", mu=0, cov=cov_matrix_stable, shape=n_hospitals)
        
        # =================================================================
        # Level 2: 過程層 (Process Level) - 位置特定脆弱度參數
        # =================================================================
        print("   Level 2: 過程層")
        
        # 使用真實區域分配而非硬編碼
        region_effects = α_region[spatial_data.region_assignments]
        
        # β_i = α_{r(i)} + δ_i + γ_i
        β_vulnerability = pm.Deterministic("β_vulnerability", 
                                         region_effects + δ_spatial + γ_individual)
        
        # =================================================================
        # Level 1: 觀測層 (Likelihood) - 災害風險核心公式
        # =================================================================
        print("   Level 1: 觀測層")
        
        # 使用真實Cat-in-Circle數據作為災害強度H_ij
        H_ij = spatial_data.hazard_intensities  # (n_hospitals, n_events)
        E_i = spatial_data.exposure_values      # (n_hospitals,)
        
        # Emanuel脆弱度函數: V(H_ij; β_i) = exp(β_i) * max(H_ij - threshold, 0)^power
        hazard_excess = pm.math.maximum(H_ij - emanuel_threshold, 0.0)
        
        # 廣播處理
        β_broadcast = β_vulnerability[:, None]      # (n_hospitals, 1)
        E_broadcast = E_i[:, None]                  # (n_hospitals, 1)
        
        # 期望損失: μ_ij = E_i × V(H_ij; β_i)
        expected_losses = pm.Deterministic("expected_losses",
            E_broadcast * pm.math.exp(β_broadcast) * 
            (hazard_excess ** vulnerability_power))
        
        # ε-contamination處理
        if contamination_epsilon > 0:
            print(f"   🛡️ 應用ε-contamination (ε={contamination_epsilon})")
            contamination_factor = pm.Lognormal("contamination_factor", mu=0, sigma=1.5)
            
            expected_losses_contaminated = pm.Deterministic("expected_losses_contaminated",
                (1 - contamination_epsilon) * expected_losses + 
                contamination_epsilon * expected_losses * contamination_factor)
            
            likelihood_mean = expected_losses_contaminated
        else:
            likelihood_mean = expected_losses
        
        # 數值穩定性處理
        likelihood_stable = pm.math.maximum(likelihood_mean, 1.0)
        
        # 觀測損失: L_ij ~ LogNormal(log(μ_ij), σ_obs²)
        observed_flat = spatial_data.observed_losses.flatten()
        likelihood_flat = likelihood_stable.flatten()
        
        L_obs = pm.Lognormal("L_obs",
                           mu=pm.math.log(likelihood_flat),
                           sigma=σ_obs,
                           observed=observed_flat)
    
    print(f"✅ 階層模型構建完成")
    return model


def get_portfolio_loss_predictions(trace, spatial_data: SpatialData, 
                                 event_indices: Optional[list] = None) -> Dict:
    """
    獲取投資組合級損失預測
    
    Parameters:
    -----------
    trace : az.InferenceData
        MCMC後驗樣本
    spatial_data : SpatialData
        空間數據
    event_indices : list, optional
        要分析的事件索引，默認分析所有事件
        
    Returns:
    --------
    Dict : 投資組合損失統計
    """
    if event_indices is None:
        event_indices = list(range(spatial_data.hazard_intensities.shape[1]))
    
    expected_losses = trace.posterior["expected_losses"]  # (chains, draws, hospitals, events)
    
    portfolio_results = {}
    
    for event_idx in event_indices:
        # 選擇特定事件的損失
        event_losses = expected_losses[:, :, :, event_idx]  # (chains, draws, hospitals)
        
        # 投資組合總損失 = 所有醫院損失之和
        portfolio_losses = event_losses.sum(axis=2)  # (chains, draws)
        portfolio_flat = portfolio_losses.values.flatten()
        
        portfolio_results[f"event_{event_idx}"] = {
            "total_loss_samples": portfolio_flat,
            "mean": np.mean(portfolio_flat),
            "std": np.std(portfolio_flat),
            "percentiles": {
                "5%": np.percentile(portfolio_flat, 5),
                "25%": np.percentile(portfolio_flat, 25),
                "50%": np.percentile(portfolio_flat, 50),
                "75%": np.percentile(portfolio_flat, 75),
                "95%": np.percentile(portfolio_flat, 95)
            },
            "individual_means": event_losses.mean(axis=(0,1)),  # 每家醫院的平均損失
            "spatial_correlation": np.corrcoef(event_losses.mean(axis=(0,1)))
        }
    
    # 整體統計
    all_events_losses = expected_losses.sum(axis=2)  # (chains, draws, events)
    portfolio_results["summary"] = {
        "mean_loss_per_event": np.mean(all_events_losses.values, axis=(0,1)),
        "total_expected_loss": np.sum(all_events_losses.values.mean(axis=(0,1))),
        "portfolio_volatility": np.std(all_events_losses.values.sum(axis=2))
    }
    
    return portfolio_results


def validate_model_inputs(spatial_data: SpatialData) -> bool:
    """
    驗證模型輸入數據的完整性
    
    Parameters:
    -----------
    spatial_data : SpatialData
        要驗證的空間數據
        
    Returns:
    --------
    bool : 是否通過驗證
    """
    issues = []
    
    # 檢查基本數據
    if spatial_data.hospital_coords is None:
        issues.append("缺少hospital_coords")
    
    if spatial_data.distance_matrix is None:
        issues.append("缺少distance_matrix")
    
    if spatial_data.region_assignments is None:
        issues.append("缺少region_assignments")
    
    # 檢查Cat-in-Circle數據
    if spatial_data.hazard_intensities is None:
        issues.append("缺少hazard_intensities (Cat-in-Circle數據)")
    
    if spatial_data.exposure_values is None:
        issues.append("缺少exposure_values")
    
    if spatial_data.observed_losses is None:
        issues.append("缺少observed_losses")
    
    # 檢查維度一致性
    if spatial_data.hazard_intensities is not None and spatial_data.exposure_values is not None:
        n_hospitals_h = spatial_data.hazard_intensities.shape[0]
        n_hospitals_e = len(spatial_data.exposure_values)
        if n_hospitals_h != n_hospitals_e:
            issues.append(f"醫院數量不一致: hazard {n_hospitals_h} vs exposure {n_hospitals_e}")
    
    if spatial_data.distance_matrix is not None:
        n_hospitals_d = spatial_data.distance_matrix.shape[0]
        if n_hospitals_d != spatial_data.n_hospitals:
            issues.append(f"距離矩陣維度不匹配: {n_hospitals_d} vs {spatial_data.n_hospitals}")
    
    if issues:
        print(f"❌ 模型輸入驗證失敗:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    else:
        print(f"✅ 模型輸入驗證通過")
        return True


# 使用範例
if __name__ == "__main__":
    from .spatial_data_processor import SpatialDataProcessor
    
    # 創建測試數據
    np.random.seed(42)
    hospital_coords = np.random.uniform([35.0, -84.0], [36.5, -75.5], (5, 2))
    
    processor = SpatialDataProcessor()
    spatial_data = processor.process_hospital_spatial_data(hospital_coords, n_regions=3)
    
    # 添加Cat-in-Circle數據
    n_hospitals, n_events = 5, 20
    hazard_intensities = np.random.uniform(20, 60, (n_hospitals, n_events))
    exposure_values = np.random.uniform(1e7, 5e7, n_hospitals)
    observed_losses = np.random.lognormal(15, 1, (n_hospitals, n_events))
    
    spatial_data = processor.add_cat_in_circle_data(
        hazard_intensities, exposure_values, observed_losses
    )
    
    # 驗證並構建模型
    if validate_model_inputs(spatial_data):
        model = build_hierarchical_model(spatial_data, contamination_epsilon=0.05)
        print(f"模型變量數量: {len(model.free_RVs)}")
    else:
        print("模型輸入驗證失敗")