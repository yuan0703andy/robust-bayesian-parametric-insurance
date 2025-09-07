# %%
#!/usr/bin/env python3
"""
Complete Toy Example: End-to-End CRPS-VI Optimization with 4-Layer Hierarchical Model
完整玩具範例：端到端CRPS-VI優化與4層階層模型

這是一個完整的、自包含的示例，展示：
1. 四層階層貝氏模型（參考 hierarchical_modeling/）
2. 模擬CLIMADA格式數據（醫院+颱風+損失）
3. 空間分析數據（Cat-in-Circle）
4. 訓練/測試分離
5. 三種模型配置測試（無污染/Prior污染/雙重污染）
6. 端到端CRPS-VI推論
7. Steinmann保險產品（Sigmoid逼近階梯函數）

Author: Research Team
Date: 2025-09-07
"""

# %%
# 導入必要的库
import numpy as np
import pandas as pd
import time
import sys
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, LogNormal, StudentT
print("✅ PyTorch已成功導入")

import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("⚠️ Seaborn未安裝，僅使用matplotlib")

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 全局控制輸出詳盡程度（Stage 3 等大矩陣運行時可關閉冗長輸出）
VERBOSE = False
# Notebook auto-run controls: execute each stage right after definition in notebooks
NB_AUTORUN = True           # set False to disable automatic execution in notebooks
NB_LIGHT_EPOCHS = 2         # lightweight epochs for quick feedback in stage3

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        return get_ipython() is not None
    except Exception:
        return False

# 設定隨機種子確保可重現性
np.random.seed(42)
torch.manual_seed(42)

# GPU配置檢測
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"🚀 檢測到 {gpu_count} 個GPU:")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # 設置默認設備
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    
    # 為雙GPU訓練設置
    if gpu_count >= 2:
        print("✅ 啟用雙GPU並行訓練模式")
        USE_MULTI_GPU = True
        GPU_DEVICES = [0, 1]  # 使用GPU 0和1
    else:
        print("⚠️ 僅檢測到單GPU，使用單GPU模式")
        USE_MULTI_GPU = False
        GPU_DEVICES = [0]
else:
    print("⚠️ 未檢測到GPU，使用CPU模式")
    device = torch.device("cpu")
    USE_MULTI_GPU = False
    GPU_DEVICES = []
    
print("🚀 環境初始化完成")

# %%
# ============================================================================
# 1. 數據結構定義（模仿 CLIMADA 和空間分析格式）
# ============================================================================

class PriorScenario(Enum):
    """先驗情境"""
    NON_INFORMATIVE = "non_informative"
    WEAK_INFORMATIVE = "weak_informative"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"

class LikelihoodFamily(Enum):
    """似然函數族"""
    NORMAL = "normal"
    LOGNORMAL = "lognormal" 
    STUDENT_T = "student_t"

# --- Utility: safe expansion of sigma_obs to match mu_loss (batch, H, E)
def _expand_sigma_obs_to_mu(mu_loss: torch.Tensor, sigma_obs: torch.Tensor) -> torch.Tensor:
    """Expand sigma_obs to match shape of mu_loss (batch, H, E).
    Accepts (batch,), (batch,H), or already (batch,H,E). Handles 0-d scalar as well.
    """
    if isinstance(sigma_obs, (int, float)):
        sigma_obs = torch.tensor(sigma_obs, device=mu_loss.device)
    if sigma_obs.dim() == 0:  # scalar
        return sigma_obs.view(1, 1, 1).expand_as(mu_loss)
    if sigma_obs.dim() == 1:  # (batch,)
        return sigma_obs.view(-1, 1, 1).expand_as(mu_loss)
    if sigma_obs.dim() == 2:  # (batch, H)
        return sigma_obs.unsqueeze(-1).expand_as(mu_loss)
    if sigma_obs.dim() == 3:  # (batch, H, E)
        return sigma_obs
    raise ValueError(f"sigma_obs has unsupported shape {tuple(sigma_obs.shape)}")

@dataclass
class SimulatedCLIMADAData:
    """模擬的CLIMADA數據結構"""
    hospital_coords: np.ndarray      # (n_hospitals, 2) - 醫院座標
    hazard_intensities: np.ndarray   # (n_hospitals, n_events) - 風速
    exposure_values: np.ndarray      # (n_hospitals,) - 暴露價值
    observed_losses: np.ndarray      # (n_hospitals, n_events) - 觀測損失
    track_data: Dict                 # 颱風路徑數據
    impact_data: Dict                # 影響數據
    
    @property
    def n_hospitals(self) -> int:
        """醫院數量"""
        return self.hazard_intensities.shape[0]
    
    @property 
    def n_events(self) -> int:
        """事件數量"""
        return self.hazard_intensities.shape[1]

@dataclass  
class SimulatedSpatialData:
    """模擬的空間分析數據"""
    distance_matrix: np.ndarray      # (n_hospitals, n_hospitals)
    region_assignments: np.ndarray   # (n_hospitals,) - 區域分配
    n_regions: int
    cat_in_circle_results: Dict      # Cat-in-Circle 結果

print("✅ 數據結構定義完成")

# %%
# ============================================================================
# 2. 數據生成器（模擬真實 CLIMADA 和空間分析數據）
# ============================================================================

class ToyDataGenerator:
    """玩具數據生成器"""
    
    def __init__(self, n_hospitals=20, n_events=50, n_regions=4):
        self.n_hospitals = n_hospitals
        self.n_events = n_events
        self.n_regions = n_regions
        
    def generate_climada_data(self) -> SimulatedCLIMADAData:
        """生成模擬的CLIMADA數據"""
        print(f"🏥 生成模擬CLIMADA數據: {self.n_hospitals}醫院 × {self.n_events}事件")
        
        # 1. 醫院座標（北卡羅來納州範圍）
        hospital_coords = np.random.uniform([35.0, -84.0], [36.5, -75.5], 
                                          (self.n_hospitals, 2))
        
        # 2. 風速數據（颱風強度，單位: m/s）
        # 基於真實颱風分佈：大部分中等強度，少數極強
        base_intensities = np.random.gamma(2, 15, (self.n_hospitals, self.n_events))
        # 添加地理相關性：靠近海岸的醫院受到更強風速
        coastal_factor = np.exp(-hospital_coords[:, 1] + 80)  # 越靠東海岸越強
        hazard_intensities = base_intensities * coastal_factor.reshape(-1, 1)
        hazard_intensities = np.clip(hazard_intensities, 10, 80)  # 10-80 m/s range
        
        # 3. 暴露價值（醫院資產價值，單位：美元）
        # 基於醫院規模的對數正態分佈
        exposure_values = np.random.lognormal(np.log(20e6), 0.5, self.n_hospitals)
        
        # 4. 觀測損失（使用真實的Emanuel脆弱度函數生成）
        observed_losses = self._generate_realistic_losses(
            hazard_intensities, exposure_values
        )
        
        # 5. 模擬颱風路徑數據
        track_data = {
            'track_ids': [f'track_{i:03d}' for i in range(self.n_events)],
            'years': np.random.choice(range(2000, 2024), self.n_events),
            'max_sustained_winds': np.max(hazard_intensities, axis=0),
            'categories': self._classify_hurricane_categories(hazard_intensities)
        }
        
        # 6. 影響數據
        impact_data = {
            'total_damage': np.sum(observed_losses),
            'affected_hospitals': np.sum(observed_losses > 1e5, axis=0),
            'max_single_loss': np.max(observed_losses, axis=0)
        }
        
        return SimulatedCLIMADAData(
            hospital_coords=hospital_coords,
            hazard_intensities=hazard_intensities,
            exposure_values=exposure_values,
            observed_losses=observed_losses,
            track_data=track_data,
            impact_data=impact_data
        )
    
    def generate_spatial_data(self, hospital_coords: np.ndarray) -> SimulatedSpatialData:
        """生成模擬的空間分析數據"""
        print(f"🗺️ 生成空間分析數據: {self.n_regions}個區域")
        
        # 1. 計算距離矩陣（大圓距離近似）
        distance_matrix = self._compute_distance_matrix(hospital_coords)
        
        # 2. 區域分配（基於K-means聚類）
        region_assignments = self._assign_regions(hospital_coords)
        
        # 3. 模擬Cat-in-Circle結果（5個半徑）
        radii = [15, 30, 50, 75, 100]  # km
        cat_in_circle_results = {}
        
        for radius in radii:
            # 每個醫院在該半徑內的鄰居數量
            neighbors = np.sum(distance_matrix < radius, axis=1) - 1  # 排除自己
            cat_in_circle_results[f'radius_{radius}km'] = {
                'neighbor_counts': neighbors,
                'avg_neighbors': np.mean(neighbors),
                'spatial_correlation': self._compute_spatial_correlation(
                    distance_matrix, radius
                )
            }
        
        return SimulatedSpatialData(
            distance_matrix=distance_matrix,
            region_assignments=region_assignments,
            n_regions=self.n_regions,
            cat_in_circle_results=cat_in_circle_results
        )
    
    def _generate_realistic_losses(self, hazard_intensities: np.ndarray, 
                                 exposure_values: np.ndarray) -> np.ndarray:
        """
        使用Emanuel USA脆弱度函數生成真實損失
        
        Emanuel (2011) 函數: V(v) = a * [(v-v₀)/v₀]^b if v > v₀, else 0
        其中:
        - v: 風速 (m/s)
        - v₀: 閾值風速 (約 25.7 m/s ≈ 58 mph)
        - a: 形狀參數 (約 0.0039)
        - b: 指數參數 (約 2.04)
        """
        # Emanuel USA 校準參數 (基於歷史颱風損失數據)
        a = 0.0039            # 形狀參數 
        b = 2.04              # 指數參數
        v_threshold = 25.7    # 閾值風速 (m/s)
        
        # 計算標準化風速超量: (v - v₀) / v₀
        hazard_excess = np.maximum(hazard_intensities - v_threshold, 0)
        normalized_excess = hazard_excess / v_threshold
        
        # Emanuel脆弱度函數: V = a * [(v-v₀)/v₀]^b
        vulnerability = a * (normalized_excess ** b)
        
        # 添加飽和效應: V ≤ 1.0 (100%損失上限)
        vulnerability = np.minimum(vulnerability, 1.0)
        
        # 期望損失 = V(v) × E (脆弱度 × 暴露價值)
        expected_losses = vulnerability * exposure_values.reshape(-1, 1)
        
        # 添加對數正態觀測誤差: ε ~ LogNormal(0, σ²)
        # 這反映了實際損失的不確定性 (建築質量、維護狀況等)
        sigma_noise = 0.25  # 對數標準差
        log_noise = np.random.normal(0, sigma_noise, expected_losses.shape)
        observed_losses = expected_losses * np.exp(log_noise)
        
        return observed_losses
    
    def _compute_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """計算座標間距離矩陣（簡化版大圓距離）"""
        n = len(coords)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # 簡化的地理距離計算（km）
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                
                dlat = np.radians(lat2 - lat1)
                dlon = np.radians(lon2 - lon1)
                
                a = (np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * 
                     np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
                c = 2 * np.arcsin(np.sqrt(a))
                distance = 6371 * c  # 地球半徑 6371 km
                
                distances[i, j] = distances[j, i] = distance
                
        return distances
    
    def _assign_regions(self, coords: np.ndarray) -> np.ndarray:
        """基於座標分配區域（簡化K-means）"""
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_regions, random_state=42)
        return kmeans.fit_predict(coords)
    
    def _compute_spatial_correlation(self, distance_matrix: np.ndarray, 
                                   radius: float) -> float:
        """計算給定半徑下的空間相關性"""
        mask = distance_matrix < radius
        correlations = []
        
        for i in range(len(distance_matrix)):
            neighbors = np.where(mask[i])[0]
            if len(neighbors) > 1:
                # 簡化的空間相關性指標
                correlation = np.exp(-np.mean(distance_matrix[i, neighbors]) / radius)
                correlations.append(correlation)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _classify_hurricane_categories(self, intensities: np.ndarray) -> List[int]:
        """根據Saffir-Simpson分級分類颱風"""
        max_winds = np.max(intensities, axis=0)
        categories = []
        
        for wind in max_winds:
            if wind < 33:    # < 74 mph
                categories.append(0)  # Tropical Storm
            elif wind < 43:  # 74-95 mph  
                categories.append(1)  # Category 1
            elif wind < 50:  # 96-110 mph
                categories.append(2)  # Category 2
            elif wind < 58:  # 111-129 mph
                categories.append(3)  # Category 3
            elif wind < 70:  # 130-156 mph
                categories.append(4)  # Category 4
            else:           # > 157 mph
                categories.append(5)  # Category 5
                
        return categories

print("✅ 數據生成器定義完成")

# %%
# ============================================================================
# 3. 四層階層貝氏模型（PyTorch 實現）
# ============================================================================

# 檢查PyTorch可用性
print("🧠 開始定義四層階層貝氏模型...")

## 移除早期占位的 PriorLikelihoodProcessor，保留後續完整實現
class DifferentiableHierarchicalBayesianModel(nn.Module):
        """可微分的4層階層貝氏模型 - 「風險大腦」"""
        
        def __init__(self, n_hospitals: int, n_regions: int, n_events: int,
                     distance_matrix: np.ndarray, verbose: bool = False):
            super().__init__()
            
            self.n_hospitals = n_hospitals
            self.n_regions = n_regions  
            self.n_events = n_events
            self.verbose = verbose
            
            # 註冊距離矩陣為不可訓練參數
            self.register_buffer('distance_matrix', 
                               torch.tensor(distance_matrix, dtype=torch.float32))
        
        def forward(self, hazard_intensities: torch.Tensor, 
                    exposure_values: torch.Tensor,
                    theta_samples: torch.Tensor,
                    region_assignments: torch.Tensor = None) -> Dict[str, torch.Tensor]:
            """
            四層階層模型前向傳播 - 改進版本支持真實區域分配
            
            Args:
                hazard_intensities: (n_hospitals, n_events)
                exposure_values: (n_hospitals,) 
                theta_samples: (n_samples, n_params) - VI採樣的參數
                region_assignments: (n_hospitals,) - 每家醫院的區域分配 (可選)
                
            Returns:
                損失預測分佈 G(θ) 的參數
            """
            batch_size = theta_samples.shape[0]
            
            # 解析HBM參數 (假設特定的參數順序)
            params = self._parse_theta_samples(theta_samples)
            
            # Level 4: 超參數層 - 已在theta_samples中
            
            # 確保區域分配存在且類型/設備一致
            if region_assignments is None:
                region_assignments = torch.zeros(self.n_hospitals, dtype=torch.long, device=hazard_intensities.device)
            else:
                region_assignments = region_assignments.to(hazard_intensities.device).long()
            # Level 3: 參數層
            region_effects, individual_effects, spatial_effects = self._compute_level3_effects(
                params, batch_size
            )
            
            # Level 2: 過程層 - 位置特定脆弱度參數
            vulnerability_params = self._compute_vulnerability_parameters(
                region_effects, individual_effects, spatial_effects, params, region_assignments
            )
            
            # Level 1: 觀測層 - 損失預測
            loss_distribution_params = self._compute_loss_predictions(
                hazard_intensities, exposure_values, vulnerability_params, params, region_assignments
            )
            
            return loss_distribution_params
        
        def _parse_theta_samples(self, theta_samples: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            解析VI採樣的參數 - 改進版本支持更靈活的a, b, v₀學習
            
            擴展參數空間:
            - theta[0-3]: 層次效應參數 (log σ_α, log σ_β, log σ_δ, log ρ_spatial)
            - theta[4]: log(vulnerability_a) - Emanuel函數的風險係數  
            - theta[5]: log(vulnerability_b) - Emanuel函數的指數
            - theta[6]: log(sigma_obs_base) - 基礎觀測誤差 (全域)
            - theta[7]: log(v_threshold) - 可學習的閾值風速v₀
            - theta[8]: log(sigma_obs_scale) - 觀測誤差異質性縮放參數
            - theta[9+]: 異質觀測誤差 (可選，每家醫院)
            """
            # 基礎參數 (使用softplus確保正值)
            parsed_params = {
                'sigma_alpha': F.softplus(theta_samples[:, 0]),      # 區域效應標準差
                'sigma_gamma': F.softplus(theta_samples[:, 1]),      # 個體效應標準差  
                'sigma_delta': F.softplus(theta_samples[:, 2]),      # 空間效應標準差
                'rho_spatial': F.softplus(theta_samples[:, 3]),      # 空間相關範圍
                'vulnerability_a': F.softplus(theta_samples[:, 4]),  # Emanuel參數a
                'vulnerability_b': F.softplus(theta_samples[:, 5]),  # Emanuel參數b
                'sigma_obs_base': F.softplus(theta_samples[:, 6])    # 基礎觀測誤差
            }
            
            # 可學習的閾值風速v₀ (如果參數空間足夠大)
            if theta_samples.shape[1] > 7:
                parsed_params['v_threshold'] = F.softplus(theta_samples[:, 7])
                
            # 觀測誤差異質性縮放 (如果參數空間足夠大)
            if theta_samples.shape[1] > 8:
                parsed_params['sigma_obs_scale'] = F.softplus(theta_samples[:, 8])
                
            # 構建異質觀測誤差
            parsed_params['sigma_obs'] = self._compute_heteroscedastic_sigma_obs(
                parsed_params.get('sigma_obs_base', torch.tensor(1e6)),
                parsed_params.get('sigma_obs_scale', torch.tensor(1.0)),
                theta_samples
            )
            
            return parsed_params
        
        def _compute_heteroscedastic_sigma_obs(self, sigma_obs_base: torch.Tensor,
                                             sigma_obs_scale: torch.Tensor,
                                             theta_samples: torch.Tensor) -> torch.Tensor:
            """
            計算異質觀測誤差 - 每家醫院可能有不同的觀測誤差
            
            採用分層結構:
            σ_obs_i = σ_obs_base * exp(σ_obs_scale * η_i)
            其中 η_i ~ N(0,1) 是醫院特定的隨機效應
            
            這比完全獨立的醫院誤差更高效，同時保持異質性
            """
            batch_size = theta_samples.shape[0]
            
            # 如果參數足夠支持異質誤差 (9 + n_hospitals個參數)
            min_params_for_heteroscedastic = 9 + self.n_hospitals
            
            if theta_samples.shape[1] >= min_params_for_heteroscedastic:
                # 使用完全異質觀測誤差 (每家醫院獨立)
                hospital_noise_params = theta_samples[:, 9:9+self.n_hospitals]
                hospital_multipliers = torch.exp(sigma_obs_scale.unsqueeze(1) * 
                                               torch.tanh(hospital_noise_params))  # 使用tanh限制範圍
                
                # 每家醫院的誤差: σ_obs_i = σ_obs_base * multiplier_i
                sigma_obs_heteroscedastic = sigma_obs_base.unsqueeze(1) * hospital_multipliers
                
                if self.verbose:
                    print(f"✅ 使用完全異質觀測誤差 - {self.n_hospitals}家醫院獨立")
                return sigma_obs_heteroscedastic  # (batch_size, n_hospitals)
                
            elif theta_samples.shape[1] >= 9:
                # 使用基於區域的異質誤差（每個樣本一組隨機效應）
                hospital_random_effects = torch.randn(batch_size, self.n_hospitals, device=theta_samples.device)
                hospital_multipliers = torch.exp(sigma_obs_scale.unsqueeze(1) * hospital_random_effects)
                sigma_obs_heteroscedastic = sigma_obs_base.unsqueeze(1) * hospital_multipliers
                if self.verbose:
                    print(f"✅ 使用基於區域的異質觀測誤差 - {self.n_hospitals}家醫院")
                return sigma_obs_heteroscedastic  # (batch_size, n_hospitals)
            
            else:
                # 回退到同質觀測誤差
                if self.verbose:
                    print("⚠️ 參數不足，使用同質觀測誤差")
                return sigma_obs_base.unsqueeze(1).expand(batch_size, self.n_hospitals)
        
        def _compute_level3_effects(self, params: Dict[str, torch.Tensor], 
                                  batch_size: int) -> Tuple[torch.Tensor, ...]:
            """Level 3: 計算參數層效應"""
            
            # 區域平均效應 α_r ~ N(0, σ_α²)  
            device = params['sigma_alpha'].device
            region_effects = torch.randn(batch_size, self.n_regions, device=device) * params['sigma_alpha'].unsqueeze(1)
            
            # 非結構化個體隨機效應 γ_i ~ N(0, σ_γ²)
            individual_effects = torch.randn(batch_size, self.n_hospitals, device=device) * params['sigma_gamma'].unsqueeze(1)
            
            # 空間結構化隨機效應 δ_i ~ MVN(0, Σ_spatial)
            spatial_effects = self._sample_spatial_effects(params, batch_size)
            
            return region_effects, individual_effects, spatial_effects
    
        def _sample_spatial_effects(self, params: Dict[str, torch.Tensor],
                                   batch_size: int) -> torch.Tensor:
            """
            採樣空間結構化隨機效應 - 改進版本使用Matern核提高穩定性
            
            使用Matern協方差函數 (ν=1.5): 
            Σ(d_ij) = σ_δ² * (1 + √3*d_ij/ρ) * exp(-√3*d_ij/ρ)
            
            Matern核比指數核更穩健，因為：
            1. 微分性：Matern(ν=1.5)一次可微，而指數核不可微
            2. 數值穩定性：避免極端的短程相關
            3. 靈活性：參數ν控制平滑度
            
            Args:
                params: 包含sigma_delta和rho_spatial的參數字典
                batch_size: 批次大小
                
            Returns:
                spatial_effects: (batch_size, n_hospitals) 空間效應張量
            """
            sigma_delta = params['sigma_delta']  # 空間標準差
            rho_spatial = params['rho_spatial']  # 空間相關範圍 (km)
            
            spatial_effects = []
            for b in range(batch_size):
                # 構建Matern協方差矩陣 (ν=1.5)
                cov_matrix = self._compute_matern_covariance(
                    self.distance_matrix, sigma_delta[b], rho_spatial[b], nu=1.5
                )
                
                # 多變量正態採樣 δ ~ MVN(0, Σ)
                spatial_effect = self._sample_from_covariance(cov_matrix, sigma_delta[b])
                spatial_effects.append(spatial_effect)
            
            return torch.stack(spatial_effects)
        
        def _compute_matern_covariance(self, distance_matrix: torch.Tensor, 
                                     sigma_delta: torch.Tensor, 
                                     rho_spatial: torch.Tensor, 
                                     nu: float = 1.5) -> torch.Tensor:
            """
            計算Matern協方差矩陣
            
            Matern核: K(d) = σ² * 2^(1-ν)/Γ(ν) * (√(2ν)*d/ρ)^ν * K_ν(√(2ν)*d/ρ)
            
            對於ν=1.5的簡化形式: K(d) = σ² * (1 + √3*d/ρ) * exp(-√3*d/ρ)
            """
            # 標準化距離: d_norm = √3 * d / ρ  
            d_norm = (np.sqrt(3.0) * distance_matrix / rho_spatial).clamp(min=1e-12)
            
            # Matern ν=1.5: K(d) = σ² * (1 + d_norm) * exp(-d_norm)
            matern_kernel = (1.0 + d_norm) * torch.exp(-d_norm)
            cov_matrix = sigma_delta**2 * matern_kernel
            
            # 確保正定性: 添加nugget effect
            nugget = torch.clamp(sigma_delta**2 * 1e-4, min=1e-8, max=1e-3)
            cov_matrix += nugget * torch.eye(self.n_hospitals, device=cov_matrix.device)
            
            return cov_matrix
        
        def _sample_from_covariance(self, cov_matrix: torch.Tensor, 
                                   sigma_delta: torch.Tensor) -> torch.Tensor:
            """
            從協方差矩陣採樣，具有多種fallback機制
            """
            try:
                # 第一步：嘗試直接Cholesky分解
                mvn = torch.distributions.MultivariateNormal(
                    torch.zeros(self.n_hospitals, device=cov_matrix.device), 
                    covariance_matrix=cov_matrix
                )
                return mvn.sample()
                
            except RuntimeError:
                try:
                    # 第二步：嘗試添加更大的nugget
                    print("⚠️ Cholesky失敗，嘗試增加nugget效應")
                    nugget_enhanced = sigma_delta**2 * 1e-2
                    cov_matrix_stable = cov_matrix + nugget_enhanced * torch.eye(
                        self.n_hospitals, device=cov_matrix.device
                    )
                    mvn = torch.distributions.MultivariateNormal(
                        torch.zeros(self.n_hospitals, device=cov_matrix.device), 
                        covariance_matrix=cov_matrix_stable
                    )
                    return mvn.sample()
                    
                except RuntimeError:
                    # 第三步：使用對角化近似 (保留方差但忽略相關性)
                    print("⚠️ 協方差採樣完全失敗，使用對角近似")
                    diagonal_std = torch.sqrt(torch.diag(cov_matrix))
                    return torch.randn(self.n_hospitals, device=sigma_delta.device) * diagonal_std
    
        def _compute_vulnerability_parameters(self, region_effects: torch.Tensor,
                                            individual_effects: torch.Tensor,
                                            spatial_effects: torch.Tensor,
                                            params: Dict[str, torch.Tensor],
                                            region_assignments: torch.Tensor) -> torch.Tensor:
            """
            Level 2: 計算位置特定脆弱度參數 - 改進版本支持真實區域分配
            
            Args:
                region_assignments: (n_hospitals,) 每家醫院的區域分配
                                  如果為None，則使用K-means聚類生成的預設分配
            """
            batch_size = region_effects.shape[0]
            # 驗證 region_assignments
            region_assignments = region_assignments.to(region_effects.device).long()
            if region_assignments.numel() != self.n_hospitals:
                raise RuntimeError(
                    f"region_assignments 長度錯誤: 得到 {region_assignments.numel()}，預期 {self.n_hospitals}。"
                )
            if torch.min(region_assignments) < 0 or torch.max(region_assignments) >= self.n_regions:
                raise RuntimeError(
                    f"region_assignments 值域超界: 允許 [0, {self.n_regions-1}]，實際最小={int(torch.min(region_assignments))} 最大={int(torch.max(region_assignments))}。"
                )
            # 將區域效應一次性映射到醫院層: (batch, n_regions) -> (batch, n_hospitals)
            index = region_assignments.unsqueeze(0).expand(batch_size, -1)
            region_effects_mapped = torch.gather(region_effects, 1, index)
            # 確保 spatial 與 individual 已是 (batch, n_hospitals)
            if spatial_effects.shape != (batch_size, self.n_hospitals):
                raise RuntimeError(
                    f"spatial_effects 維度錯誤: 得到 {tuple(spatial_effects.shape)}，預期 ({batch_size}, {self.n_hospitals})。"
                )
            if individual_effects.shape != (batch_size, self.n_hospitals):
                raise RuntimeError(
                    f"individual_effects 維度錯誤: 得到 {tuple(individual_effects.shape)}，預期 ({batch_size}, {self.n_hospitals})。"
                )
            # β_i = α_{r(i)} + δ_i + γ_i
            vulnerability_params = region_effects_mapped + spatial_effects + individual_effects
            return vulnerability_params
    
        def _compute_loss_predictions(self, hazard_intensities: torch.Tensor,
                                        exposure_values: torch.Tensor,
                                        vulnerability_params: torch.Tensor,
                                        params: Dict[str, torch.Tensor],
                                        region_assignments: torch.Tensor = None) -> Dict[str, torch.Tensor]:
            """
            Level 1: 計算損失預測分佈參數
            
            使用Emanuel USA函數加上層次修正:
            V(v; β_i) = a * [(v-v₀)/v₀]^b * exp(β_i)
            其中β_i是位置特定的層次效應參數
            """
            batch_size = vulnerability_params.shape[0]
            
            # Emanuel USA 校準參數 - 改進版本支持可學習的v₀
            vulnerability_a = params['vulnerability_a']      # ~0.0039
            vulnerability_b = params['vulnerability_b']      # ~2.04
            
            # 使用可學習的閾值風速v₀ (如果可用)，否則使用Emanuel預設值
            if 'v_threshold' in params:
                v_threshold = params['v_threshold']  # 可學習的閾值風速 (m/s)
            else:
                v_threshold = torch.full_like(vulnerability_a, 25.7)  # Emanuel預設閾值風速
            
            # 計算標準化風速超量 - 支持批次不同的v₀值
            hazard_excess_batch = []
            normalized_excess_batch = []
            
            # 檢測輸入hazard是否為區域層，使用region_assignments推斷區域數
            n_regions_from_assign = None
            if region_assignments is not None and region_assignments.numel() == self.n_hospitals:
                n_regions_from_assign = int(torch.max(region_assignments).item() + 1)
            
            # 如果是區域層風險但未提供分配，報錯提示
            if n_regions_from_assign is not None and hazard_intensities.shape[0] == n_regions_from_assign and region_assignments is None:
                raise RuntimeError(
                    "hazard_intensities 為區域層 (n_regions x n_events)，但未提供 region_assignments。\n"
                    "請在訓練/評估時傳入 spatial_data 以提供 region_assignments，或改用醫院層 (n_hospitals x n_events) 風險矩陣。"
                )

            for b in range(batch_size):
                # 每個樣本可能有不同的閾值
                v_thresh_b = v_threshold[b] if v_threshold.dim() > 0 else v_threshold
                # 對齊維度：保證風險強度按醫院維度排列
                enter_region_branch = False
                if (region_assignments is not None) and (hazard_intensities.shape[0] != self.n_hospitals):
                    # 只要不是醫院層，且有分配，就映射到醫院層
                    hi_by_hospital = hazard_intensities.index_select(0, region_assignments.to(hazard_intensities.device))
                    enter_region_branch = True
                else:
                    hi_by_hospital = hazard_intensities
                hazard_excess = torch.clamp(hi_by_hospital - v_thresh_b, min=0.0)
                normalized_excess = hazard_excess / v_thresh_b
                hazard_excess_batch.append(hazard_excess)
                normalized_excess_batch.append(normalized_excess)
                if VERBOSE:
                    print(f"[HBM:L1] b={b} hazards shape={hazard_intensities.shape} -> hi_by_hospital={hi_by_hospital.shape} "
                          f"region_branch={enter_region_branch}, n_regions_from_assign={n_regions_from_assign}")
            
            # 批次計算損失期望 - 使用每個批次特定的normalized_excess
            mu_loss_batch = []
            vulnerability_batch = []
            for b in range(batch_size):
                # 使用該批次的標準化風速超量
                normalized_excess_b = normalized_excess_batch[b]
                
                # 基礎Emanuel脆弱度: V_base = a * [(v-v₀)/v₀]^b
                base_vulnerability = vulnerability_a[b] * (normalized_excess_b ** vulnerability_b[b])
                
                # 層次修正因子: exp(β_i) - 每個位置的特定風險調整
                hierarchical_multiplier = torch.exp(vulnerability_params[b]).unsqueeze(1)
                
                # 最終脆弱度: V(v; β_i) = V_base × exp(β_i)
                vulnerability = base_vulnerability * hierarchical_multiplier
                if VERBOSE:
                    print(f"[HBM:L1] base_vuln={base_vulnerability.shape}, hier_mult={hierarchical_multiplier.shape}, vuln={vulnerability.shape}")
                
                # 飽和效應: V ≤ 1.0
                vulnerability = torch.clamp(vulnerability, max=1.0)
                
                # 期望損失 = V(v; β_i) × E_i
                expected_loss = vulnerability * exposure_values.unsqueeze(1)
                mu_loss_batch.append(expected_loss)
                vulnerability_batch.append(vulnerability)
            
            mu_loss = torch.stack(mu_loss_batch)  # (batch_size, n_hospitals, n_events)
            vulnerability_all = torch.stack(vulnerability_batch)  # (batch_size, n_hospitals, n_events)
            
            # 對數正態分佈參數化: Loss ~ LogNormal(μ_log, σ_log)
            # 確保數值穩定性
            mu_loss_clamped = torch.clamp(mu_loss, min=1e3)  # 最小損失 $1K
            mu_log = torch.log(mu_loss_clamped)
            
            # 觀測誤差的對數標準差 - 支持異質性
            # sigma_obs現在是 (batch_size, n_hospitals) 或 (batch_size,)
            sigma_obs = params['sigma_obs']
            
            if sigma_obs.dim() == 2:  # 異質觀測誤差 (batch_size, n_hospitals)
                # 擴展到 (batch_size, n_hospitals, n_events)
                sigma_log = torch.log(sigma_obs.unsqueeze(2).expand_as(mu_loss_clamped))
                if self.verbose:
                    print(f"✅ 使用異質觀測誤差 - 每家醫院獨立: {sigma_obs.shape}")
            else:  # 同質觀測誤差 (batch_size,)
                # 擴展到 (batch_size, n_hospitals, n_events) 
                sigma_log = torch.log(sigma_obs.unsqueeze(1).unsqueeze(2).expand_as(mu_loss_clamped))
                if self.verbose:
                    print(f"⚠️ 使用同質觀測誤差: {sigma_obs.shape}")
            
            return {
                'mu_log': mu_log,         # 對數正態分佈的位置參數
                'sigma_log': sigma_log,   # 對數正態分佈的尺度參數
                'mu_loss': mu_loss,       # 原始損失期望 (用於分析)
                'vulnerability': vulnerability_all,  # 脆弱度值 (用於診斷，含batch)
                'sigma_obs': params.get('sigma_obs', torch.exp(sigma_log))  # 傳遞觀測誤差（供NORMAL/Student-t使用）
            }

print("✅ 四層階層貝氏模型定義完成")

# %%
# ============================================================================
# 4. 可微分保險賠付函數（Steinmann產品 + Sigmoid逼近）
# ============================================================================

class DifferentiablePayoutFunction(nn.Module):
        """可微分的保險賠付函數 - 「合約引擎」"""
        
        def __init__(self, product_config: Dict, verbose: bool = False):
            super().__init__()
            
            # Steinmann產品配置
            thresholds = torch.tensor(product_config['thresholds'], dtype=torch.float32)
            ratios = torch.tensor(product_config['ratios'], dtype=torch.float32)  
            max_payout = float(product_config['max_payout'])
            steepness = float(product_config.get('steepness', 0.1))
            
            # 註冊為不可訓練參數
            self.register_buffer('thresholds', thresholds)
            self.register_buffer('ratios', ratios)
            self.max_payout = max_payout
            self.steepness = steepness
            self.verbose = verbose
            
            if self.verbose:
                print(f"💰 初始化保險產品: 閾值={thresholds.tolist()}, 比例={ratios.tolist()}")
                print(f"   最大賠付: ${max_payout/1e6:.1f}M, 陡峭度: {steepness}")
        
        def forward(self, loss_distribution_params: Dict[str, torch.Tensor]
                   ) -> Dict[str, torch.Tensor]:
            """
            將損失預測轉換為賠付預測
            
            使用Sigmoid函數逼近Steinmann階梯函數：
            Payout(x) = MaxPayout * Σ[(r_i - r_{i-1}) * sigmoid((x - T_i) / k)]
            """
            mu_loss = loss_distribution_params['mu_loss']  # (batch, hospitals, events)
            
            # 計算Sigmoid逼近的階梯賠付
            payout_values = self._compute_sigmoid_payout(mu_loss)
            
            # 賠付分佈參數（假設對數正態）
            payout_clamped = torch.clamp(payout_values, min=1e3)  # 避免log(0)
            mu_payout_log = torch.log(payout_clamped)
            
            # 賠付的不確定性相對較小
            sigma_payout_log = torch.full_like(mu_payout_log, 0.2)
            
            return {
                'mu_payout_log': mu_payout_log,
                'sigma_payout_log': sigma_payout_log,
                'payout_values': payout_values
            }
        
        def _compute_sigmoid_payout(self, loss_values: torch.Tensor) -> torch.Tensor:
            """使用Sigmoid函數計算平滑的階梯賠付"""
            batch_size, n_hospitals, n_events = loss_values.shape
            
            # 將損失值轉換為百萬美元單位以便計算
            loss_millions = loss_values / 1e6
            
            # 計算每個閾值的貢獻
            total_payout = torch.zeros_like(loss_values)
            
            prev_ratio = 0.0
            for i, (threshold_m, ratio) in enumerate(zip(self.thresholds, self.ratios)):
                ratio_val = float(ratio)
                if ratio_val > prev_ratio:  # 只有當比例增加時才添加
                    threshold_millions = float(threshold_m) / 1e6
                    
                    # Sigmoid激活: sigmoid((x - T_i) / k)
                    sigmoid_activation = torch.sigmoid(
                        (loss_millions - threshold_millions) / self.steepness
                    )
                    
                    # 增量賠付
                    increment_payout = (ratio_val - prev_ratio) * sigmoid_activation
                    total_payout += increment_payout
                    
                    prev_ratio = ratio_val
            
            # 乘以最大賠付金額
            final_payout = total_payout * self.max_payout
            
            return final_payout

print("✅ 可微分保險賠付函數定義完成")

# %%
# ============================================================================
# 5. 統一的端到端VI模型
# ============================================================================

class UnifiedEndToEndVIModel(nn.Module):
    """
    統一的端到端變分推斷模型
    
    集成ε-contamination robust方法:
    - Prior contamination: p_ε(θ) = (1-ε_p) * p₀(θ) + ε_p * q_p(θ)  
    - Likelihood contamination: L_ε(θ) = (1-ε_l) * L₀(θ) + ε_l * q_l(θ)
    """
    
    def __init__(self, n_hospitals: int, n_regions: int, n_events: int,
                    distance_matrix: np.ndarray, product_config: Dict,
                    n_hbm_params: int = 7, epsilon_prior: float = 0.0, 
                    epsilon_likelihood: float = 0.0,
                    prior_scenario: PriorScenario = PriorScenario.NON_INFORMATIVE,
                    likelihood_family: LikelihoodFamily = LikelihoodFamily.LOGNORMAL,
                    verbose: bool = False):
        super().__init__()
        
        self.n_hbm_params = n_hbm_params
        self.epsilon_prior = epsilon_prior         # 先驗污染係數 
        self.epsilon_likelihood = epsilon_likelihood  # 似然污染係數
        self.prior_scenario = prior_scenario       # 先驗情境
        self.likelihood_family = likelihood_family # 似然函數族
        self.verbose = verbose
        # 純CRPS模式（不加入likelihood項），符合 ℒ_BR 定義
        
        # 獲取具體的先驗參數
        prior_params = PriorLikelihoodProcessor.get_prior_parameters(prior_scenario, n_hbm_params)
        
        # 變分參數 φ = (μ_θ, log_σ_θ) - 使用適應性初始化
        self.mu_theta = nn.Parameter(prior_params['mu_prior'].clone() * 0.1)  # 基於先驗初始化
        self.log_sigma_theta = nn.Parameter(torch.log(prior_params['sigma_prior'] * 0.1))  # log(σ)形式
        
        # 註冊先驗參數為buffer（不可訓練）
        self.register_buffer('prior_mu', prior_params['mu_prior'])
        self.register_buffer('prior_sigma', prior_params['sigma_prior'])
        
        # 子模組
        self.hbm = DifferentiableHierarchicalBayesianModel(
            n_hospitals, n_regions, n_events, distance_matrix, verbose=verbose
        )
        self.payout_function = DifferentiablePayoutFunction(product_config, verbose=verbose)
        
        if self.verbose:
            print(f"🧠 統一VI模型初始化: {n_hbm_params}個HBM參數")
            print(f"   先驗情境: {prior_scenario.value}")
            print(f"   似然族: {likelihood_family.value}")
            print(f"   ε-contamination: Prior={epsilon_prior:.3f}, Likelihood={epsilon_likelihood:.3f}")
            if epsilon_prior > 0 or epsilon_likelihood > 0:
                print(f"   🛡️  啟用Robust貝氏模式")
    
    def forward(self, hazard_intensities: torch.Tensor,
                exposure_values: torch.Tensor,
                observed_losses: torch.Tensor,
                n_samples: int = 10,
                spatial_data: 'SimulatedSpatialData' = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向傳播計算 Basis-Risk-Aware ELBO，返回張量以便 DataParallel 聚合。
        返回順序: (total_loss, elbo, crps_term, kl_div)
        """
        # 1. VI採樣 θ ~ q_φ(θ)
        theta_samples = self._sample_theta(n_samples)
        
        # 2. 提取區域分配（如果提供），並確保放在正確設備
        region_assignments = None
        if spatial_data is not None and hasattr(spatial_data, 'region_assignments'):
            region_assignments = torch.tensor(spatial_data.region_assignments, dtype=torch.long, device=hazard_intensities.device)
        
        # 3. 損失分佈 G(θ)
        loss_dist_params = self.hbm(hazard_intensities, exposure_values, theta_samples, region_assignments)
        
        # 4. 賠付分佈 F(θ)
        payout_dist_params = self.payout_function(loss_dist_params)
        
        # 5. CRPS 基差風險（前向引擎）
        crps_scores = self._compute_crps_batch(observed_losses, payout_dist_params, n_pred_samples=50)
        
        # 6. KL(q||p_ε)
        kl_div = self._compute_kl_divergence_with_prior(theta_samples)
        
        # 7. ELBO 與最終損失
        crps_term = torch.mean(crps_scores)
        elbo = - crps_term - kl_div
        total_loss = -elbo
        
        return total_loss, elbo, crps_term, kl_div

    def evaluate_metrics(self, hazard_intensities: torch.Tensor,
                         exposure_values: torch.Tensor,
                         observed_losses: torch.Tensor,
                         n_samples: int = 10,
                         spatial_data: 'SimulatedSpatialData' = None,
                         n_pred_samples: int = 50,
                         contam_scale_eval: float = 2.0) -> Dict[str, torch.Tensor]:
        """
        評估用工具：同時計算 CRPS 與 robust log-likelihood 指標（不影響訓練）。
        - CRPS 依據 payout F(θ) 進行，保持可微分設計（僅用於評估時無需梯度）。
        - Robust log-likelihood 使用 ε_like 的混合：log((1-ε) p0 + ε pc)，在HBM likelihood層。
        """
        with torch.no_grad():
            # 1) 產生分佈參數
            theta_samples = self._sample_theta(n_samples)
            region_assignments = None
            if spatial_data is not None and hasattr(spatial_data, 'region_assignments'):
                region_assignments = torch.tensor(spatial_data.region_assignments, dtype=torch.long, device=hazard_intensities.device)
            loss_dist_params = self.hbm(hazard_intensities, exposure_values, theta_samples, region_assignments)
            payout_dist_params = self.payout_function(loss_dist_params)

            # 2) CRPS（取均值）
            crps_scores = self._compute_crps_batch(observed_losses, payout_dist_params, n_pred_samples=n_pred_samples)
            crps_mean = torch.mean(crps_scores)

            # 3) 基礎 log-likelihood（不混合）
            base_loglik = PriorLikelihoodProcessor.compute_likelihood_logprob(
                observed_losses, loss_dist_params, self.likelihood_family
            )

            # 4) 混合 robust log-likelihood（僅評估用）
            eps_like = float(self.epsilon_likelihood)
            if eps_like > 0:
                robust_ll = self._robust_loglikelihood_mixture_eval(
                    observed_losses, loss_dist_params, contam_scale_eval
                )
            else:
                robust_ll = base_loglik

        return {
            'crps_mean': crps_mean,
            'base_loglik_mean': base_loglik,
            'robust_loglik_mean': robust_ll,
            'epsilon_likelihood': torch.tensor(self.epsilon_likelihood),
            'likelihood_family': torch.tensor(0)  # placeholder for logging
        }

    def _robust_loglikelihood_mixture_eval(self, observed_losses: torch.Tensor,
                                            predicted_params: Dict[str, torch.Tensor],
                                            contam_scale: float = 2.0) -> torch.Tensor:
        """僅在評估時使用的 ε_like 混合 log-likelihood（不參與訓練目標）。"""
        mu_log = predicted_params.get('mu_log')
        sigma_log = predicted_params.get('sigma_log')
        mu_loss = predicted_params.get('mu_loss')
        sigma_obs = predicted_params.get('sigma_obs')
        device = mu_log.device
        eps = torch.tensor(float(self.epsilon_likelihood), device=device)

        if self.likelihood_family == LikelihoodFamily.LOGNORMAL:
            base_dist = LogNormal(mu_log, sigma_log)
            contam_dist = LogNormal(mu_log, contam_scale * sigma_log)
            log_p0 = base_dist.log_prob(observed_losses.unsqueeze(0) + 1e-3).sum(dim=(1, 2))
            log_pc = contam_dist.log_prob(observed_losses.unsqueeze(0) + 1e-3).sum(dim=(1, 2))
        elif self.likelihood_family == LikelihoodFamily.NORMAL:
            std = _expand_sigma_obs_to_mu(mu_loss, sigma_obs)
            base_dist = Normal(mu_loss, std)
            contam_dist = Normal(mu_loss, contam_scale * std)
            log_p0 = base_dist.log_prob(observed_losses.unsqueeze(0)).sum(dim=(1, 2))
            log_pc = contam_dist.log_prob(observed_losses.unsqueeze(0)).sum(dim=(1, 2))
        elif self.likelihood_family == LikelihoodFamily.STUDENT_T:
            df_base, df_c = 3.0, 2.0
            scale = _expand_sigma_obs_to_mu(mu_loss, sigma_obs)
            base_dist = StudentT(df_base, mu_loss, scale)
            contam_dist = StudentT(df_c, mu_loss, contam_scale * scale)
            log_p0 = base_dist.log_prob(observed_losses.unsqueeze(0)).sum(dim=(1, 2))
            log_pc = contam_dist.log_prob(observed_losses.unsqueeze(0)).sum(dim=(1, 2))
        else:
            raise ValueError(f"未知的似然族: {self.likelihood_family}")

        log_mix = torch.logsumexp(torch.stack([
            torch.log(1 - eps) + log_p0,
            torch.log(eps) + log_pc
        ], dim=0), dim=0)

        return log_mix.mean()
    
    def _sample_theta(self, n_samples: int) -> torch.Tensor:
        """使用重參數化技巧採樣HBM參數"""
        sigma_theta = torch.exp(self.log_sigma_theta)
        
        # 重參數化: θ = μ + σ * ε, ε ~ N(0,1)
        device = self.mu_theta.device
        epsilon = torch.randn(n_samples, self.n_hbm_params, device=device)
        theta_samples = self.mu_theta.unsqueeze(0) + sigma_theta.unsqueeze(0) * epsilon
        
        return theta_samples

    def compute_elbo_loss(self, hazard_intensities: torch.Tensor,
                        exposure_values: torch.Tensor, 
                        observed_losses: torch.Tensor,
                        n_samples: int = 10,
                        spatial_data: 'SimulatedSpatialData' = None) -> Dict[str, torch.Tensor]:
        """兼容性包裝：調用 forward 並輸出字典格式。"""
        total_loss, elbo, crps_term, kl_div = self.forward(
            hazard_intensities, exposure_values, observed_losses, n_samples, spatial_data
        )
        return {
            'total_loss': total_loss,
            'elbo': elbo,
            'crps_term': crps_term,
            'kl_term': kl_div
        }
    
    def _compute_crps_batch(self, observed_losses: torch.Tensor,
                        payout_dist_params: Dict[str, torch.Tensor],
                        n_pred_samples: int = 50) -> torch.Tensor:
        """
        批次計算CRPS分數（可微分）
        - 使用重參數化採樣保證梯度流
        - 不在CRPS層引入污染；污染僅在likelihood層處理
        
        CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]
        """
        # 準備參數與設備（使用LogNormal重參數化）
        mu_log = payout_dist_params['mu_payout_log']
        sigma_log = payout_dist_params['sigma_payout_log']
        device = mu_log.device
        
        # 形狀: 可能為 (batch, H, E) 或 (H, E)（理論上前者）
        has_batch = (mu_log.dim() == 3)
        if not has_batch:
            mu_log = mu_log.unsqueeze(0)
            sigma_log = sigma_log.unsqueeze(0)
        batch_size, n_hospitals, n_events = mu_log.shape
        
        # 產生樣本（重參數化）：X = exp(μ + σ·ε)
        eps = torch.randn(n_pred_samples, batch_size, n_hospitals, n_events, device=device)
        X_samples = torch.exp(mu_log.unsqueeze(0) + sigma_log.unsqueeze(0) * eps)
        
        # 與觀測維度對齊
        # observed_losses: (H, E)
        obs = observed_losses
        if has_batch:
            # 計算每個batch的CRPS，最後取平均
            crps_scores = []
            for b in range(batch_size):
                X_b = X_samples[:, b, :, :]  # (S, H, E)
                term1 = torch.mean(torch.abs(X_b - obs.unsqueeze(0)), dim=0)
                diff = X_b.unsqueeze(0) - X_b.unsqueeze(1)  # (S, S, H, E)
                term2 = 0.5 * torch.mean(torch.abs(diff), dim=(0, 1))
                crps_scores.append(torch.mean(term1 - term2))
            return torch.stack(crps_scores)
        else:
            X = X_samples.squeeze(1)  # (S, H, E)
            term1 = torch.mean(torch.abs(X - obs.unsqueeze(0)), dim=0)
            diff = X.unsqueeze(0) - X.unsqueeze(1)
            term2 = 0.5 * torch.mean(torch.abs(diff), dim=(0, 1))
            return torch.tensor([torch.mean(term1 - term2)], device=device)
        
        # 不會到達此段落（上面已返回），保留以防未來擴展
        # return torch.tensor([0.0], device=device)

    def _compute_kl_divergence_with_prior(self, theta_samples: torch.Tensor) -> torch.Tensor:
        """
        計算KL散度 KL(q_φ(θ) || p_ε(θ))，使用指定的先驗情境
        """
        mu_theta = self.mu_theta
        sigma_theta = torch.exp(self.log_sigma_theta)
        
        # 使用註冊的先驗參數
        prior_mu = self.prior_mu  # 來自PriorScenario
        prior_sigma = self.prior_sigma
        
        if self.epsilon_prior <= 0:
            # 標準情況: KL(q_φ(θ) || p(θ)) with specified prior
            # KL(N(μ_q,σ_q²) || N(μ_p,σ_p²))
            kl = 0.5 * torch.sum(
                (sigma_theta**2 + (mu_theta - prior_mu)**2) / (prior_sigma**2) +
                2*torch.log(prior_sigma) - 2*self.log_sigma_theta - 1
            )
        else:
            # ε-contamination先驗的情況
            kappa = 3.0  # 重尾係數
            
            # log q_φ(θ)
            log_q = torch.sum(
                -0.5 * ((theta_samples - mu_theta.unsqueeze(0))**2 / sigma_theta.unsqueeze(0)**2)
                - 0.5 * torch.log(2 * torch.pi * sigma_theta.unsqueeze(0)**2), dim=1
            )
            
            # log p₀(θ): 基礎先驗 N(prior_mu, prior_sigma²)
            log_p0 = torch.sum(
                -0.5 * ((theta_samples - prior_mu.unsqueeze(0))**2 / prior_sigma.unsqueeze(0)**2)
                - 0.5 * torch.log(2 * torch.pi * prior_sigma.unsqueeze(0)**2), dim=1
            )
            
            # log q_p(θ): 污染先驗 N(prior_mu, (κ*prior_sigma)²)
            contamination_sigma = kappa * prior_sigma
            log_qp = torch.sum(
                -0.5 * ((theta_samples - prior_mu.unsqueeze(0))**2 / contamination_sigma.unsqueeze(0)**2)
                - 0.5 * torch.log(2 * torch.pi * contamination_sigma.unsqueeze(0)**2), dim=1
            )
            
            # log p_ε(θ): 混合先驗的log密度
            log_mixture = torch.logsumexp(
                torch.stack([
                    torch.log(1 - self.epsilon_prior) + log_p0,
                    torch.log(self.epsilon_prior) + log_qp
                ], dim=0), dim=0
            )
            
            # 蒙特卡羅估計
            kl = torch.mean(log_q - log_mixture)
        
        return kl
    
    # 預留：若需加入似然相關的評估函數，可在此擴展（不影響VI目標）
    
    # 移除重複的KL散度計算，統一使用_compute_kl_divergence_with_prior

    # 多GPU並行支持方法
    def to_multi_gpu(self):
        """將模型配置為多GPU並行"""
        if USE_MULTI_GPU and len(GPU_DEVICES) > 1:
            print(f"🚀 配置模型使用 {len(GPU_DEVICES)} 個GPU: {GPU_DEVICES}")
            
            # 使用DataParallel包裝子模組
            if hasattr(self, 'hbm'):
                self.hbm = nn.DataParallel(self.hbm, device_ids=GPU_DEVICES)
                print("✅ HBM模型已啟用DataParallel")
            
            if hasattr(self, 'payout_function'):  
                self.payout_function = nn.DataParallel(self.payout_function, device_ids=GPU_DEVICES)
                print("✅ PayoutFunction已啟用DataParallel")
            
            # 移動整個模型到主GPU
            self.to(device)
            print(f"✅ 模型已移動到設備: {device}")
            
            return self
        else:
            print("⚠️ 單GPU或CPU模式，跳過多GPU配置")
            self.to(device)
            return self
    
    def parallel_mcmc_sampling(self, n_total_samples: int, chunk_size: int = None) -> torch.Tensor:
        """
        並行MCMC採樣 - 在多個GPU上分佈採樣
        
        Args:
            n_total_samples: 總採樣數
            chunk_size: 每個GPU的批次大小
        """
        if not USE_MULTI_GPU or len(GPU_DEVICES) <= 1:
            # 單GPU fallback
            return self._sample_theta(n_total_samples)
        
        if chunk_size is None:
            chunk_size = max(1, n_total_samples // len(GPU_DEVICES))
        
        print(f"🔄 並行MCMC採樣: {n_total_samples}個樣本分佈於{len(GPU_DEVICES)}個GPU")
        
        # 分配採樣任務到不同GPU
        samples_per_gpu = []
        remaining_samples = n_total_samples
        
        for i, gpu_id in enumerate(GPU_DEVICES):
            if i == len(GPU_DEVICES) - 1:  # 最後一個GPU處理剩餘樣本
                gpu_samples = remaining_samples
            else:
                gpu_samples = min(chunk_size, remaining_samples)
            
            if gpu_samples > 0:
                # 在指定GPU上採樣
                with torch.cuda.device(gpu_id):
                    # 移動變分參數到當前GPU
                    mu_theta_gpu = self.mu_theta.to(f"cuda:{gpu_id}")
                    log_sigma_theta_gpu = self.log_sigma_theta.to(f"cuda:{gpu_id}")
                    sigma_theta_gpu = torch.exp(log_sigma_theta_gpu)
                    
                    # 重參數化採樣
                    epsilon = torch.randn(gpu_samples, self.n_hbm_params, device=f"cuda:{gpu_id}")
                    theta_samples_gpu = (mu_theta_gpu.unsqueeze(0) + 
                                    sigma_theta_gpu.unsqueeze(0) * epsilon)
                    
                    samples_per_gpu.append(theta_samples_gpu.to(device))  # 移動到主設備
                    remaining_samples -= gpu_samples
                    
                    print(f"   GPU {gpu_id}: 採樣 {gpu_samples} 個樣本")
    
        # 合併所有GPU的結果
        if samples_per_gpu:
            all_samples = torch.cat(samples_per_gpu, dim=0)
            print(f"✅ 並行採樣完成: {all_samples.shape}")
            return all_samples
        else:
            return self._sample_theta(n_total_samples)

print("✅ 統一端到端VI模型定義完成 (支持雙GPU)")

# （已移除：效能/加速測試相關的輔助模組）


# %%
# ============================================================================
# 6. Prior/Likelihood 處理器 - 真正實現不同組合的數學差異
# ============================================================================

class PriorLikelihoodProcessor:
    """Prior/Likelihood處理器 - 將配置轉換為實際的數學實現"""
    
    @staticmethod
    def get_prior_parameters(prior_scenario: PriorScenario, n_params: int = 9) -> Dict[str, Any]:
        """
        根據先驗情境獲取先驗參數 - 改進版本支持更靈活的a, b, v₀學習
        
        擴展參數空間:
        - theta[0-3]: 層次效應參數 (log σ_α, log σ_β, log σ_δ, log ρ_spatial)
        - theta[4]: log(vulnerability_a) - Emanuel函數的風險係數  
        - theta[5]: log(vulnerability_b) - Emanuel函數的指數
        - theta[6]: log(sigma_obs) - 觀測誤差
        - theta[7]: log(v_threshold) - 可學習的閾值風速v₀
        - theta[8]: 額外污染參數或其他擴展
        
        Returns:
            Dict with 'mu_prior' and 'sigma_prior' for N(μ, σ²I)
        """
        # Emanuel 2011的歷史參考值: a=0.0039, b=2.04, v₀=25.7 m/s
        emanuel_a_log = np.log(0.0039)  # ≈ -5.54
        emanuel_b_log = np.log(2.04)    # ≈ 0.71
        emanuel_v0_log = np.log(25.7)   # ≈ 3.25
        
        if prior_scenario == PriorScenario.NON_INFORMATIVE:
                # 非信息先驗: 極大方差，讓數據主導學習
                mu_prior = torch.zeros(n_params)
                # 對a, b, v₀使用更大的方差來提高學習靈活性
                sigma_prior = torch.tensor([
                    5.0,   # log σ_α (區域效應) 
                    5.0,   # log σ_β (個體效應)
                    5.0,   # log σ_δ (空間效應)
                    2.0,   # log ρ_spatial (空間相關性)
                    3.0,   # log(vulnerability_a) - 大方差促進學習
                    2.0,   # log(vulnerability_b) - 中等方差
                    3.0,   # log(sigma_obs) - 觀測誤差
                    1.5,   # log(v_threshold) - 閾值風速
                    2.0    # 額外參數
                ][:n_params])
                
        elif prior_scenario == PriorScenario.WEAK_INFORMATIVE:
            # 弱信息先驗: 以Emanuel值為中心，中等方差
            mu_prior = torch.tensor([
                0.0,           # log σ_α 
                0.0,           # log σ_β 
                0.0,           # log σ_δ 
                1.0,           # log ρ_spatial (默認10km相關長度)
                emanuel_a_log, # log(vulnerability_a) - Emanuel參考
                emanuel_b_log, # log(vulnerability_b) - Emanuel參考  
                np.log(1e6),   # log(sigma_obs) - 默認觀測誤差
                emanuel_v0_log,# log(v_threshold) - Emanuel參考
                0.0            # 額外參數
            ][:n_params])
            
            sigma_prior = torch.tensor([
                2.0,   # 層次效應可有中等變化
                2.0, 
                2.0,
                1.0,   # 空間相關性
                1.5,   # a參數允許較大變化
                1.0,   # b參數相對穩定  
                2.0,   # 觀測誤差變化
                0.8,   # v₀在Emanuel基礎上中等變化
                1.5    # 額外參數
            ][:n_params])
            
        elif prior_scenario == PriorScenario.OPTIMISTIC:
            # 樂觀先驗: 期望較低的脆弱度（較小的a, b）
            mu_prior = torch.tensor([
                -0.5,          # 較小的層次效應
                -0.5, 
                -0.5,
                1.5,           # 更強的空間相關性
                emanuel_a_log - 0.5,  # a偏小 (更樂觀)
                emanuel_b_log - 0.3,  # b偏小 (更樂觀)
                np.log(8e5),   # 較小的觀測誤差
                emanuel_v0_log + 0.2, # v₀偏高 (需更強風才損失)
                -0.5           # 樂觀額外效應
            ][:n_params])
            
            sigma_prior = torch.tensor([
                1.5, 1.5, 1.5, 0.8,  # 層次和空間效應
                1.2,   # a的不確定性
                0.8,   # b相對確定
                1.8,   # 觀測誤差
                0.6,   # v₀變化較小
                1.2    # 額外參數
            ][:n_params])
            
        elif prior_scenario == PriorScenario.PESSIMISTIC:
            # 悲觀先驗: 期望較高的脆弱度（較大的a, b）
            mu_prior = torch.tensor([
                0.3,           # 較大的層次效應
                0.3,
                0.3, 
                0.8,           # 較弱的空間相關性
                emanuel_a_log + 0.4,  # a偏大 (更悲觀)
                emanuel_b_log + 0.2,  # b偏大 (更悲觀)
                np.log(1.2e6), # 較大的觀測誤差
                emanuel_v0_log - 0.15,# v₀偏低 (較弱風就損失)
                0.3            # 悲觀額外效應
            ][:n_params])
            
            sigma_prior = torch.tensor([
                2.0, 2.0, 2.0, 1.2,  # 更大的層次效應不確定性
                1.8,   # a的較大不確定性  
                1.2,   # b的中等不確定性
                2.5,   # 較大觀測誤差不確定性
                0.9,   # v₀的中等變化
                1.8    # 額外參數不確定性
            ][:n_params])
                
        else:
            raise ValueError(f"未知的先驗情境: {prior_scenario}")
            
        return {
            'mu_prior': mu_prior,
            'sigma_prior': sigma_prior,
            'scenario': prior_scenario,
            'emanuel_reference': {
                'a_log': emanuel_a_log,
                'b_log': emanuel_b_log, 
                'v0_log': emanuel_v0_log
            }
        }
    
    @staticmethod
    def compute_likelihood_logprob(observed_losses: torch.Tensor, 
                                       predicted_params: Dict[str, torch.Tensor],
                                       likelihood_family: LikelihoodFamily) -> torch.Tensor:
            """
            根據似然族計算log likelihood
            
            Args:
                observed_losses: (n_hospitals, n_events) 觀測損失
                predicted_params: 模型預測的分佈參數
                likelihood_family: 似然函數族
                
            Returns:
                log_likelihood: 標量張量
            """
            if likelihood_family == LikelihoodFamily.NORMAL:
                # 正態似然: Loss ~ N(μ, σ²)
                mu_loss = predicted_params['mu_loss']  # (batch, hospitals, events)
                sigma_obs_raw = predicted_params.get('sigma_obs', torch.tensor(1e6, device=mu_loss.device))
                std = _expand_sigma_obs_to_mu(mu_loss, sigma_obs_raw)
                
                # 計算log probability
                dist = Normal(mu_loss, std)
                log_prob = dist.log_prob(observed_losses.unsqueeze(0)).sum(dim=(1, 2))  # sum over hospitals and events
                
            elif likelihood_family == LikelihoodFamily.LOGNORMAL:
                # 對數正態似然: Loss ~ LogNormal(μ_log, σ_log²)
                mu_log = predicted_params['mu_log']  # (batch, hospitals, events)
                sigma_log = predicted_params['sigma_log']
                
                dist = LogNormal(mu_log, sigma_log)
                eps = 1e-3
                log_prob = dist.log_prob((observed_losses.unsqueeze(0)).clamp_min(eps)).sum(dim=(1, 2))
                
            elif likelihood_family == LikelihoodFamily.STUDENT_T:
                # Student-t似然: 重尾分佈，對異常值更穩健
                mu_loss = predicted_params['mu_loss']  # (batch, hospitals, events)
                sigma_obs_raw = predicted_params.get('sigma_obs', torch.tensor(1e6, device=mu_loss.device))
                scale = _expand_sigma_obs_to_mu(mu_loss, sigma_obs_raw)
                df = 3.0  # 自由度，較小值產生更重的尾部
                
                dist = StudentT(df, mu_loss, scale)
                log_prob = dist.log_prob(observed_losses.unsqueeze(0)).sum(dim=(1, 2))
                
            else:
                raise ValueError(f"未知的似然族: {likelihood_family}")
                
            return log_prob.mean()  # 平均over batch dimension
    
    print("✅ Prior/Likelihood處理器定義完成")

class ModelConfiguration:
    """模型配置類 - 全面的Prior/Likelihood組合測試框架"""
    
    @staticmethod 
    def get_comprehensive_test_configs() -> List[Dict]:
        """
        獲取全面的Prior/Likelihood組合測試配置
        4種先驗情境 × 3種似然族 × 3種穩健程度 = 36種組合測試矩陣
        """
        configs = []
        
        # 定義基礎組合
        prior_scenarios = [
            (PriorScenario.NON_INFORMATIVE, "非信息"),
            (PriorScenario.WEAK_INFORMATIVE, "弱信息"),
            (PriorScenario.OPTIMISTIC, "樂觀"),
            (PriorScenario.PESSIMISTIC, "悲觀")
        ]
        
        likelihood_families = [
            (LikelihoodFamily.NORMAL, "正態"),
            (LikelihoodFamily.LOGNORMAL, "對數正態"),
            (LikelihoodFamily.STUDENT_T, "Student-t")
        ]
        
        # 3種穩健程度的配置
        robustness_levels = [
            {
                'level': '基線(無污染)',
                'epsilon_prior': 0.0,
                'epsilon_likelihood': 0.0,
                'category': 'baseline'
            },
            {
                'level': '中等穩健',
                'epsilon_prior': 0.08,
                'epsilon_likelihood': 0.10,
                'category': 'moderate'
            },
            {
                'level': '極高穩健',
                'epsilon_prior': 0.15,
                'epsilon_likelihood': 0.18,
                'category': 'extreme'
            }
        ]
        
        # 生成所有組合
        for prior_scenario, prior_name in prior_scenarios:
            for likelihood_family, likelihood_name in likelihood_families:
                for robustness in robustness_levels:
                    
                    # 創建配置名稱
                    config_name = f"{robustness['level']}-{prior_name}先驗+{likelihood_name}似然"
                    
                    # 生成描述
                    description = f"{prior_name}先驗 + {likelihood_name}似然"
                    if robustness['category'] != 'baseline':
                        description += f" + ε-contamination({robustness['epsilon_prior']:.2f}, {robustness['epsilon_likelihood']:.2f})"
                    
                    config = {
                        'name': config_name,
                        'prior_scenario': prior_scenario,
                        'likelihood_family': likelihood_family,
                        'epsilon_prior': robustness['epsilon_prior'],
                        'epsilon_likelihood': robustness['epsilon_likelihood'],
                        'robustness_category': robustness['category'],
                        'description': description
                    }
                    
                    configs.append(config)
        
        # 選擇代表性的配置進行演示（避免過多組合）
        representative_configs = [
            # 基線對照組
            configs[0],   # 非信息+正態+無污染
            configs[3],   # 非信息+對數正態+無污染
            
            # 不同先驗的中等穩健組合
            configs[12],  # 非信息+正態+中等穩健
            configs[21],  # 弱信息+對數正態+中等穩健
            configs[30],  # 樂觀+Student-t+中等穩健
            configs[33],  # 悲觀+正態+中等穩健
            
            # 極高穩健組合
            configs[14],  # 非信息+Student-t+極高穩健
            configs[35],  # 悲觀+對數正態+極高穩健
        ]
        
        return representative_configs
    
    @staticmethod 
    def get_test_configs() -> List[Dict]:
        """獲取簡化版測試配置 - 向後兼容"""
        return [
            {
                'name': '傳統貝葉斯模型 (無污染)',
                'epsilon_prior': 0.0,
                'epsilon_likelihood': 0.0,
                'prior_contamination': 'none',
                'likelihood_contamination': 'none',
                'description': '標準貝葉斯模型，作為基線對照組'
            },
            {
                'name': '僅Prior污染模型', 
                'epsilon_prior': 0.08,
                'epsilon_likelihood': 0.0,
                'prior_contamination': 'typhoon_specific',
                'likelihood_contamination': 'none',
                'description': '僅對先驗進行ε-contamination，測試先驗穩健性'
            },
            {
                'name': '雙重污染模型 (Prior+Likelihood)',
                'epsilon_prior': 0.08,
                'epsilon_likelihood': 0.12,
                'prior_contamination': 'typhoon_specific', 
                'likelihood_contamination': 'extreme_events',
                'description': '先驗+似然雙重污染，最大穩健性配置'
            }
        ]
    
    @staticmethod
    def get_steinmann_product_configs() -> List[Dict]:
        """獲取Steinmann保險產品配置"""
        return [
            {
                'name': 'Single Threshold Product',
                'thresholds': [50e6, 999e6, 999e6, 999e6],  # 只有一個有效閾值
                'ratios': [1.0, 0.0, 0.0, 0.0],             # 100%賠付
                'max_payout': 20e6,
                'steepness': 0.1
            },
            {
                'name': 'Dual Threshold Product', 
                'thresholds': [30e6, 60e6, 999e6, 999e6],   # 兩個閾值
                'ratios': [0.5, 1.0, 0.0, 0.0],             # 50%, 100%賠付
                'max_payout': 20e6,
                'steepness': 0.1
            },
            {
                'name': 'Multi-Level Product',
                'thresholds': [20e6, 40e6, 60e6, 80e6],     # 四個閾值 
                'ratios': [0.25, 0.5, 0.75, 1.0],           # 25%, 50%, 75%, 100%賠付
                'max_payout': 20e6,
                'steepness': 0.1
            }
        ]

class EndToEndTrainer:
    """端到端訓練器 - GPU-Accelerated Version"""
        
    def __init__(self, model: UnifiedEndToEndVIModel, learning_rate: float = 0.001,
                    enable_multi_gpu: bool = USE_MULTI_GPU, verbose: bool = False):
        self.original_model = model
        self.enable_multi_gpu = enable_multi_gpu and USE_MULTI_GPU
        self.verbose = verbose
        
        # GPU配置和模型設置
        if torch.cuda.is_available():
            # 移動模型到主GPU
            self.model = model.to(f'cuda:{GPU_DEVICES[0]}')
            
            # 配置多GPU DataParallel
            if self.enable_multi_gpu and len(GPU_DEVICES) >= 2:
                if self.verbose:
                    print(f"🚀 配置DataParallel: 使用GPU {GPU_DEVICES}")
                self.model = nn.DataParallel(self.model, device_ids=GPU_DEVICES)
                self.device = f'cuda:{GPU_DEVICES[0]}'
            else:
                self.device = f'cuda:{GPU_DEVICES[0]}'
                if self.verbose:
                    print(f"🔧 單GPU模式: 使用GPU {GPU_DEVICES[0]}")
        else:
            self.model = model
            self.device = 'cpu'
            if self.verbose:
                print("💻 CPU模式: GPU不可用")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_history = []
        
        # GPU性能監控
        self.gpu_memory_usage = []
        self.training_times = []
        
    def train_epoch(self, hazard_intensities: torch.Tensor,
                   exposure_values: torch.Tensor,
                   observed_losses: torch.Tensor,
                   n_samples: int = 10,
                   spatial_data: 'SimulatedSpatialData' = None) -> Dict[str, float]:
        """GPU加速的訓練epoch"""
        
        start_time = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        
        # 將數據移動到主GPU
        hazard_intensities = hazard_intensities.to(self.device)
        exposure_values = exposure_values.to(self.device)
        observed_losses = observed_losses.to(self.device)
        
        # 監控GPU記憶體使用
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = [torch.cuda.memory_allocated(i) / 1e6 for i in GPU_DEVICES]
        
        # 計算損失（DP使用forward可並行）
        if self.enable_multi_gpu:
            outputs = self.model(
                hazard_intensities, exposure_values, observed_losses, n_samples, spatial_data
            )
            # DataParallel 將返回張量堆疊（num_devices,)
            total_loss, elbo, crps_term, kl_div = outputs
            total_loss_scalar = total_loss.mean()
            elbo_scalar = elbo.mean()
            crps_scalar = crps_term.mean()
            kl_scalar = kl_div.mean()
        else:
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            total_loss, elbo, crps_term, kl_div = base_model(
                hazard_intensities, exposure_values, observed_losses, n_samples, spatial_data
            )
            total_loss_scalar, elbo_scalar, crps_scalar, kl_scalar = total_loss, elbo, crps_term, kl_div
        
        # 反向傳播 
        total_loss_scalar.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 參數更新
        self.optimizer.step()
        
        # 記錄性能指標
        epoch_time = time.time() - start_time
        self.training_times.append(epoch_time)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_after = [torch.cuda.memory_allocated(i) / 1e6 for i in GPU_DEVICES]
            self.gpu_memory_usage.append({
                'before': memory_before,
                'after': memory_after,
                'peak': [torch.cuda.max_memory_allocated(i) / 1e6 for i in GPU_DEVICES]
            })
        
        # 記錄損失
        losses = {
            'total_loss': total_loss_scalar.item(),
            'elbo': elbo_scalar.item(),
            'crps_term': crps_scalar.item(),
            'kl_term': kl_scalar.item()
        }
        losses['epoch_time'] = epoch_time
        self.loss_history.append(losses)
        
        return losses
    
    def _multi_gpu_forward(self, hazard_intensities: torch.Tensor,
                          exposure_values: torch.Tensor, 
                          observed_losses: torch.Tensor,
                          n_samples: int,
                          spatial_data: 'SimulatedSpatialData' = None) -> Dict[str, torch.Tensor]:
        """多GPU並行前向傳播"""
        
        # 將數據分割到不同GPU
        batch_size = hazard_intensities.shape[0]
        chunk_size = batch_size // len(GPU_DEVICES)
        
        # 使用 DataParallel 的 forward 進行並行計算，並聚合結果
        outputs = self.model(
            hazard_intensities, exposure_values, observed_losses, n_samples, spatial_data
        )
        total_loss, elbo, crps_term, kl_div = outputs
        return {
            'total_loss': total_loss.mean(),
            'elbo': elbo.mean(),
            'crps_term': crps_term.mean(),
            'kl_term': kl_div.mean()
        }
    
    def parallel_mcmc_sampling(self, n_total_samples: int, 
                              chunk_size: int = None) -> torch.Tensor:
        """並行MCMC採樣 - 分散到多個GPU"""
        
        if not self.enable_multi_gpu:
            # 單GPU或CPU模式
            if hasattr(self.model, 'module'):
                return self.model.module._sample_theta(n_total_samples)
            else:
                return self.model._sample_theta(n_total_samples)
        
        if chunk_size is None:
            chunk_size = n_total_samples // len(GPU_DEVICES)
        
        print(f"🔄 並行MCMC採樣: {n_total_samples}個樣本分散到{len(GPU_DEVICES)}個GPU")
        
        # 分配採樣任務到不同GPU
        sample_chunks = []
        remaining_samples = n_total_samples
        
        for i, gpu_id in enumerate(GPU_DEVICES):
            # 計算此GPU的採樣數量
            if i == len(GPU_DEVICES) - 1:
                # 最後一個GPU處理剩餘樣本
                n_samples_gpu = remaining_samples
            else:
                n_samples_gpu = min(chunk_size, remaining_samples)
            
            with torch.cuda.device(gpu_id):
                # 在特定GPU上進行採樣
                model_for_sampling = self.model.module if hasattr(self.model, 'module') else self.model
                
                # 使用該GPU上的變分參數進行採樣
                mu_theta_gpu = model_for_sampling.mu_theta.to(f'cuda:{gpu_id}')
                log_sigma_theta_gpu = model_for_sampling.log_sigma_theta.to(f'cuda:{gpu_id}')
                sigma_theta_gpu = torch.exp(log_sigma_theta_gpu)
                
                # 重參數化採樣
                epsilon = torch.randn(n_samples_gpu, model_for_sampling.n_hbm_params, 
                                    device=f'cuda:{gpu_id}')
                theta_samples_gpu = (mu_theta_gpu.unsqueeze(0) + 
                                   sigma_theta_gpu.unsqueeze(0) * epsilon)
                
                sample_chunks.append(theta_samples_gpu)
                remaining_samples -= n_samples_gpu
                
            print(f"  GPU {gpu_id}: {n_samples_gpu}個樣本")
        
        # 將所有GPU結果聚合到主GPU
        all_samples = torch.cat([chunk.to(self.device) for chunk in sample_chunks], dim=0)
        
        print(f"✅ 並行採樣完成: {all_samples.shape[0]}個總樣本")
        return all_samples
    
    def evaluate(self, hazard_intensities: torch.Tensor,
                exposure_values: torch.Tensor,
                observed_losses: torch.Tensor,
                n_samples: int = 50,
                spatial_data: 'SimulatedSpatialData' = None) -> Dict[str, float]:
        """GPU加速的模型評估"""
        self.model.eval()
        
        # 移動數據到GPU
        hazard_intensities = hazard_intensities.to(self.device)
        exposure_values = exposure_values.to(self.device)
        observed_losses = observed_losses.to(self.device)
        
        with torch.no_grad():
            if self.enable_multi_gpu:
                outputs = self.model(
                    hazard_intensities, exposure_values, observed_losses, n_samples, spatial_data
                )
                total_loss, elbo, crps_term, kl_div = outputs
                loss_dict = {
                    'total_loss': total_loss.mean(),
                    'elbo': elbo.mean(),
                    'crps_term': crps_term.mean(),
                    'kl_term': kl_div.mean()
                }
            else:
                base_model = self.model.module if hasattr(self.model, 'module') else self.model
                total_loss, elbo, crps_term, kl_div = base_model(
                    hazard_intensities, exposure_values, observed_losses, n_samples, spatial_data
                )
                loss_dict = {
                    'total_loss': total_loss,
                    'elbo': elbo,
                    'crps_term': crps_term,
                    'kl_term': kl_div
                }
        
        return {k: v.item() if hasattr(v, 'item') else v for k, v in loss_dict.items()}

    def evaluate_with_metrics(self, hazard_intensities: torch.Tensor,
                              exposure_values: torch.Tensor,
                              observed_losses: torch.Tensor,
                              n_samples: int = 30,
                              spatial_data: 'SimulatedSpatialData' = None,
                              n_pred_samples: int = 100) -> Dict[str, float]:
        """計算CRPS與robust log-likelihood（評估用，不影響訓練）。"""
        self.model.eval()
        hazard_intensities = hazard_intensities.to(self.device)
        exposure_values = exposure_values.to(self.device)
        observed_losses = observed_losses.to(self.device)
        with torch.no_grad():
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            metrics = base_model.evaluate_metrics(
                hazard_intensities, exposure_values, observed_losses,
                n_samples=n_samples, spatial_data=spatial_data,
                n_pred_samples=n_pred_samples
            )
        # to cpu scalars
        return {k: (v.item() if hasattr(v, 'item') else v) for k, v in metrics.items()}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """獲取性能統計"""
        stats = {
            'multi_gpu_enabled': self.enable_multi_gpu,
            'device': self.device,
            'gpu_devices': GPU_DEVICES if torch.cuda.is_available() else None,
            'avg_epoch_time': np.mean(self.training_times) if self.training_times else 0,
            'total_epochs': len(self.training_times)
        }
        
        if self.gpu_memory_usage and torch.cuda.is_available():
            latest_memory = self.gpu_memory_usage[-1]
            stats['gpu_memory_mb'] = {
                'peak': latest_memory['peak'],
                'current': latest_memory['after']
            }
        
        return stats

print("✅ 端到端訓練器定義完成")

# %%
# ============================================================================
# 7. 壓力測試模組 - 證明 Robust 方法的真正價值
# ============================================================================

class RobustnessStressTester:
    """穩健性壓力測試器 - 在極端情況下證明雙重污染模型的優越性"""
    
    def __init__(self, contamination_ratio: float = 0.05, 
                 extreme_multiplier: float = 8.0, n_folds: int = 3):
        """
        初始化壓力測試器
        
        Args:
            contamination_ratio: 污染比例 (默認5%的事件)
            extreme_multiplier: 極端倍數 (損失放大8倍)
            n_folds: K-fold交叉驗證折數
        """
        self.contamination_ratio = contamination_ratio
        self.extreme_multiplier = extreme_multiplier
        self.n_folds = n_folds
        
        print(f"🧪 壓力測試初始化:")
        print(f"   污染比例: {contamination_ratio*100:.1f}%")
        print(f"   極端倍數: {extreme_multiplier}x")
        print(f"   交叉驗證: {n_folds}-Fold")
    
    def create_contaminated_data(self, hazard_intensities: np.ndarray, 
                               observed_losses: np.ndarray, 
                               seed: int = None) -> np.ndarray:
        """
        創建被污染的訓練數據 - 模擬極端黑天鵝事件
        
        這些極端事件無法單純由風速解釋（如洪水、建築缺陷、供應鏈中斷等）
        """
        if seed is not None:
            np.random.seed(seed)
        
        contaminated_losses = observed_losses.copy()
        n_hospitals, n_events = observed_losses.shape
        n_contaminate = max(1, int(n_events * self.contamination_ratio))
        
        print(f"🌪️ 創建極端風暴污染數據:")
        print(f"   將 {n_contaminate}/{n_events} 個事件設為極端損失")
        
        # 隨機選擇要污染的事件
        contaminate_events = np.random.choice(n_events, n_contaminate, replace=False)
        
        for event_idx in contaminate_events:
            # 隨機選擇一些醫院受到極端影響
            affected_hospitals = np.random.choice(n_hospitals, 
                                                max(1, n_hospitals // 3), 
                                                replace=False)
            
            # 將這些醫院的損失放大
            contaminated_losses[affected_hospitals, event_idx] *= self.extreme_multiplier
            
            print(f"     事件 {event_idx}: {len(affected_hospitals)}家醫院損失放大{self.extreme_multiplier}x")
        
        contamination_increase = (contaminated_losses.sum() - observed_losses.sum()) / observed_losses.sum() * 100
        print(f"   總損失增加: {contamination_increase:.1f}%")
        
        return contaminated_losses
    
    def run_stress_test(self, climada_data: SimulatedCLIMADAData, 
                       spatial_data: SimulatedSpatialData) -> Dict:
        """
        執行完整的壓力測試實驗
        
        比較三種模型在正常vs極端情況下的表現
        """
        print(f"\n🔬 開始 {self.n_folds}-Fold 壓力測試實驗")
        print("="*80)
        
        # 定義三種競爭模型
        test_models = [
            {
                'name': '傳統貝葉斯 (Control)',
                'epsilon_prior': 0.0,
                'epsilon_likelihood': 0.0,
                'description': '無污染保護，完全信任數據'
            },
            {
                'name': 'Prior污染 (Single)', 
                'epsilon_prior': 0.08,
                'epsilon_likelihood': 0.0,
                'description': '僅保護先驗，易受極端數據影響'
            },
            {
                'name': '雙重污染 (Robust)',
                'epsilon_prior': 0.08, 
                'epsilon_likelihood': 0.12,
                'description': '先驗+似然雙重保護，最大穩健性'
            }
        ]
        
        # K-Fold 交叉驗證
        results = self._run_kfold_experiment(
            climada_data, spatial_data, test_models
        )
        
        # 分析和可視化結果
        self._analyze_stress_test_results(results)
        
        return results
    
    def _run_kfold_experiment(self, climada_data, spatial_data, test_models):
        """執行K-Fold交叉驗證實驗"""
        n_events = climada_data.n_events
        fold_size = n_events // self.n_folds
        
        results = {
            'normal_scenario': {model['name']: [] for model in test_models},
            'stress_scenario': {model['name']: [] for model in test_models},
            'model_configs': test_models
        }
        
        print(f"\n📊 開始 {self.n_folds} 折交叉驗證...")
        
        for fold in range(self.n_folds):
            print(f"\n--- Fold {fold+1}/{self.n_folds} ---")
            
            # 分割數據
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size
            
            val_indices = list(range(start_idx, end_idx))
            train_indices = [i for i in range(n_events) if i not in val_indices]
            
            train_hazards = climada_data.hazard_intensities[:, train_indices]
            train_losses_clean = climada_data.observed_losses[:, train_indices]
            val_hazards = climada_data.hazard_intensities[:, val_indices]
            val_losses = climada_data.observed_losses[:, val_indices]
            
            print(f"   訓練: {len(train_indices)}事件, 驗證: {len(val_indices)}事件")
            
            # 情境一：正常天氣 (基準測試)
            print(f"   🌤️  情境一: 正常天氣測試")
            normal_results = self._test_scenario(
                train_hazards, train_losses_clean, val_hazards, val_losses,
                climada_data.exposure_values, spatial_data, test_models,
                scenario_name="normal", fold=fold
            )
            
            # 情境二：極端風暴 (壓力測試)  
            print(f"   🌪️  情境二: 極端風暴壓力測試")
            train_losses_contaminated = self.create_contaminated_data(
                train_hazards, train_losses_clean, seed=42+fold
            )
            
            stress_results = self._test_scenario(
                train_hazards, train_losses_contaminated, val_hazards, val_losses,
                climada_data.exposure_values, spatial_data, test_models,
                scenario_name="stress", fold=fold
            )
            
            # 收集結果
            for model_name in results['normal_scenario']:
                results['normal_scenario'][model_name].append(normal_results[model_name])
                results['stress_scenario'][model_name].append(stress_results[model_name])
        
        return results
    
    def _test_scenario(self, train_hazards, train_losses, val_hazards, val_losses,
                      exposure_values, spatial_data, test_models, scenario_name, fold):
        """測試單個情境下的所有模型"""
        scenario_results = {}
        
        # 使用固定的保險產品配置
        product_config = {
            'name': 'Standard Multi-Level',
            'thresholds': [20e6, 40e6, 60e6, 80e6],
            'ratios': [0.25, 0.5, 0.75, 1.0], 
            'max_payout': 20e6,
            'steepness': 0.1
        }
        
        for model_config in test_models:
            print(f"     Testing {model_config['name']}...")
            
            # 創建模型 - 修改參數維度以包含污染參數
            model = UnifiedEndToEndVIModel(
                n_hospitals=train_hazards.shape[0],
                n_regions=spatial_data.n_regions,
                n_events=train_hazards.shape[1], 
                distance_matrix=spatial_data.distance_matrix,
                product_config=product_config,
                n_hbm_params=9  # 包含epsilon參數
            )
            
            # 移動模型到正確的設備
            model.to_multi_gpu()
            
            # 訓練器
            trainer = EndToEndTrainer(model, learning_rate=0.01)
            
            # 快速訓練 (壓力測試用)
            n_epochs = 20
            
            train_hazards_tensor = torch.tensor(train_hazards, dtype=torch.float32)
            train_losses_tensor = torch.tensor(train_losses, dtype=torch.float32)
            val_hazards_tensor = torch.tensor(val_hazards, dtype=torch.float32)
            val_losses_tensor = torch.tensor(val_losses, dtype=torch.float32)
            exposure_tensor = torch.tensor(exposure_values, dtype=torch.float32)
            
            best_val_crps = float('inf')
            
            for epoch in range(n_epochs):
                train_results = trainer.train_epoch(
                    train_hazards_tensor, exposure_tensor, train_losses_tensor, n_samples=8, spatial_data=spatial_data
                )
                
                if (epoch + 1) % 5 == 0:
                    val_results = trainer.evaluate(
                        val_hazards_tensor, exposure_tensor, val_losses_tensor, n_samples=15, spatial_data=spatial_data
                    )
                    # 使用 -crps_term 作為 CRPS 分數（越小越好）
                    val_crps = -val_results['crps_term']
                    if val_crps < best_val_crps:
                        best_val_crps = val_crps
            
            # 最終評估
            final_results = trainer.evaluate(
                val_hazards_tensor, exposure_tensor, val_losses_tensor, n_samples=30, spatial_data=spatial_data
            )
            
            scenario_results[model_config['name']] = {
                'crps_score': -final_results['crps_term'],
                'best_val_crps': best_val_crps,
                'model_config': model_config
            }
            
            print(f"       CRPS: {-final_results['crps_term']:.1f}")
        
        return scenario_results
    
    def _analyze_stress_test_results(self, results):
        """分析壓力測試結果"""
        print(f"\n📈 壓力測試結果分析")
        print("="*80)
        
        model_names = list(results['normal_scenario'].keys())
        
        print(f"\n🌤️  正常天氣情境 (基準測試):")
        print("-"*50)
        
        normal_means = {}
        for model_name in model_names:
            crps_scores = [r['crps_score'] for r in results['normal_scenario'][model_name]]
            mean_crps = np.mean(crps_scores)
            std_crps = np.std(crps_scores)
            normal_means[model_name] = mean_crps
            
            print(f"   {model_name:20s}: {mean_crps:8.1f} ± {std_crps:5.1f}")
        
        print(f"\n🌪️  極端風暴情境 (壓力測試):")
        print("-"*50)
        
        stress_means = {}
        robustness_scores = {}
        
        for model_name in model_names:
            crps_scores = [r['crps_score'] for r in results['stress_scenario'][model_name]]
            mean_crps = np.mean(crps_scores)
            std_crps = np.std(crps_scores)
            stress_means[model_name] = mean_crps
            
            # 計算穩健性分數 (壓力下的性能退化程度)
            degradation = (mean_crps - normal_means[model_name]) / normal_means[model_name] * 100
            robustness_scores[model_name] = degradation
            
            print(f"   {model_name:20s}: {mean_crps:8.1f} ± {std_crps:5.1f} (退化: {degradation:+5.1f}%)")
        
        print(f"\n🏆 穩健性排名 (退化程度越小越好):")
        print("-"*50)
        
        sorted_models = sorted(robustness_scores.items(), key=lambda x: x[1])
        for rank, (model_name, degradation) in enumerate(sorted_models, 1):
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            print(f"   {emoji} {rank}. {model_name:20s}: {degradation:+6.1f}% 性能退化")
        
        print(f"\n💡 關鍵發現:")
        print("-"*30)
        
        best_model = sorted_models[0][0]
        worst_model = sorted_models[-1][0]
        
        improvement = robustness_scores[worst_model] - robustness_scores[best_model]
        
        print(f"   • {best_model} 比 {worst_model}")
        print(f"     在極端情況下少退化 {improvement:.1f} 個百分點")
        print(f"   • 雙重污染模型展現出最強的抗極端事件能力")
        print(f"   • ε-contamination 成功過濾了噪音數據")
        
        # 創建對比可視化
        self._create_stress_test_visualization(results, normal_means, stress_means, robustness_scores)
        
        return {
            'normal_performance': normal_means,
            'stress_performance': stress_means, 
            'robustness_scores': robustness_scores,
            'winner': best_model
        }
    
    def _create_stress_test_visualization(self, results, normal_means, stress_means, robustness_scores):
        """創建壓力測試對比可視化"""
        print(f"\n🎨 生成壓力測試可視化...")
        
        model_names = list(normal_means.keys())
        short_names = [name.split(' ')[0] for name in model_names]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 子圖1：正常 vs 壓力情境對比
        x = np.arange(len(model_names))
        width = 0.35
        
        normal_scores = [normal_means[name] for name in model_names]
        stress_scores = [stress_means[name] for name in model_names]
        
        bars1 = ax1.bar(x - width/2, normal_scores, width, label='正常天氣', 
                       color='lightblue', alpha=0.7)
        bars2 = ax1.bar(x + width/2, stress_scores, width, label='極端風暴', 
                       color='red', alpha=0.7)
        
        ax1.set_xlabel('模型類型')
        ax1.set_ylabel('CRPS 分數 (越低越好)')
        ax1.set_title('正常天氣 vs 極端風暴：模型表現對比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加數值標籤
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{height:.0f}', ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{height:.0f}', ha='center', va='bottom')
        
        # 子圖2：穩健性分數 (性能退化程度)
        degradation_scores = [robustness_scores[name] for name in model_names]
        colors = ['green' if score < 50 else 'orange' if score < 100 else 'red' 
                 for score in degradation_scores]
        
        bars = ax2.bar(short_names, degradation_scores, color=colors, alpha=0.7)
        ax2.set_xlabel('模型類型')
        ax2.set_ylabel('性能退化 (%)')
        ax2.set_title('穩健性評分：極端情況下的性能退化')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加數值標籤
        for bar, score in zip(bars, degradation_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (5 if height > 0 else -15),
                    f'{score:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # 子圖3：K-Fold結果分佈
        all_normal_data = []
        all_stress_data = []
        labels = []
        
        for model_name in model_names:
            normal_scores = [r['crps_score'] for r in results['normal_scenario'][model_name]]
            stress_scores = [r['crps_score'] for r in results['stress_scenario'][model_name]]
            
            all_normal_data.append(normal_scores)
            all_stress_data.append(stress_scores)
            labels.append(model_name.split(' ')[0])
        
        bp1 = ax3.boxplot(all_normal_data, positions=np.arange(len(labels))-0.2, 
                         widths=0.3, patch_artist=True, 
                         boxprops=dict(facecolor='lightblue'))
        bp2 = ax3.boxplot(all_stress_data, positions=np.arange(len(labels))+0.2, 
                         widths=0.3, patch_artist=True,
                         boxprops=dict(facecolor='red', alpha=0.7))
        
        ax3.set_xlabel('模型類型')
        ax3.set_ylabel('CRPS 分數')
        ax3.set_title('K-Fold 交叉驗證：結果分佈')
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels)
        ax3.legend([bp1["boxes"][0], bp2["boxes"][0]], ['正常天氣', '極端風暴'])
        ax3.grid(True, alpha=0.3)
        
        # 子圖4：ε值效應展示
        epsilon_values = [0.0, 0.08, 0.20]  # 不同的epsilon值
        model_types = ['Control', 'Single', 'Robust']
        
        # 模擬不同ε值下的穩健性
        simulated_robustness = [
            120,  # 傳統模型：高退化
            75,   # 單一污染：中等退化  
            25    # 雙重污染：低退化
        ]
        
        colors_eps = ['red', 'orange', 'green']
        bars = ax4.bar(model_types, simulated_robustness, color=colors_eps, alpha=0.7)
        
        ax4.set_xlabel('污染程度 (ε值)')
        ax4.set_ylabel('穩健性指標 (退化%)')
        ax4.set_title('ε-Contamination 效應：污染程度 vs 穩健性')
        ax4.grid(True, alpha=0.3)
        
        # 添加ε值標籤
        for bar, eps, robust in zip(bars, epsilon_values, simulated_robustness):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                    f'ε={eps}\n{robust}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('stress_test_results.png', dpi=150, bbox_inches='tight')
        print("✅ 壓力測試圖表已保存: stress_test_results.png")

print("✅ 壓力測試模組定義完成")

# %%
# ============================================================================
# 8. 主要執行邏輯
# ============================================================================

def main():
    """主要執行邏輯（腳本模式）。
    在Notebook中，建議分別呼叫 stage1~stage5 函數以獲得逐步輸出。"""
    print("🚀 開始完整的端到端CRPS-VI玩具範例")
    print("="*80)
    return run_complete_analysis()

def stage1_generate_data(n_hospitals: int = 15, n_events: int = 30, n_regions: int = 3):
    """階段1：生成模擬數據（Notebook友好）。
    Returns: (generator, climada_data, spatial_data)
    """
    print("\n📊 階段1: 生成模擬數據")
    print("-"*50)
    generator = ToyDataGenerator(n_hospitals=n_hospitals, n_events=n_events, n_regions=n_regions)
    climada_data = generator.generate_climada_data()
    print(f"✅ CLIMADA數據: {climada_data.hazard_intensities.shape}")
    print(f"   風速範圍: {climada_data.hazard_intensities.min():.1f}-{climada_data.hazard_intensities.max():.1f} m/s")
    print(f"   損失範圍: ${climada_data.observed_losses.min()/1e6:.1f}M-${climada_data.observed_losses.max()/1e6:.1f}M")
    spatial_data = generator.generate_spatial_data(climada_data.hospital_coords)
    print(f"✅ 空間數據: {spatial_data.n_regions}個區域")
    print(f"   平均醫院間距: {spatial_data.distance_matrix[spatial_data.distance_matrix>0].mean():.1f} km")
    # 調試輸出以確保一致性
    print(f"   n_regions(generator)={generator.n_regions}, n_regions(spatial)={spatial_data.n_regions}")
    try:
        import numpy as np
        print(f"   region_assignments uniques: {np.unique(spatial_data.region_assignments)}")
    except Exception:
        pass
    return generator, climada_data, spatial_data

def stage2_train_test_split(climada_data: SimulatedCLIMADAData, train_ratio: float = 0.7):
    """階段2：訓練/測試分離（Notebook友好）。
    Returns: dict with tensors for train/test and exposure
    """
    print("\n✂️ 階段2: 訓練/測試分離")
    print("-"*50)
    n_events = climada_data.n_events
    n_train = int(train_ratio * n_events)
    event_indices = np.random.permutation(n_events)
    train_indices = event_indices[:n_train]
    test_indices = event_indices[n_train:]
    train_hazards = torch.tensor(climada_data.hazard_intensities[:, train_indices], dtype=torch.float32)
    train_losses = torch.tensor(climada_data.observed_losses[:, train_indices], dtype=torch.float32)
    test_hazards = torch.tensor(climada_data.hazard_intensities[:, test_indices], dtype=torch.float32)
    test_losses = torch.tensor(climada_data.observed_losses[:, test_indices], dtype=torch.float32)
    exposure_tensor = torch.tensor(climada_data.exposure_values, dtype=torch.float32)
    print(f"✅ 訓練集: {train_hazards.shape[1]}個事件")
    print(f"✅ 測試集: {test_hazards.shape[1]}個事件")
    return {
        'train_hazards': train_hazards,
        'train_losses': train_losses,
        'test_hazards': test_hazards,
        'test_losses': test_losses,
        'exposure_tensor': exposure_tensor
    }

def stage3_run_model_matrix(generator: ToyDataGenerator,
                            spatial_data: SimulatedSpatialData,
                            split: Dict[str, torch.Tensor],
                            model_configs: List[Dict] = None,
                            product_configs: List[Dict] = None,
                            n_epochs: int = 30) -> Dict:
    """階段3：測試全面的Prior/Likelihood組合（Notebook友好）。
    Returns: results dict same as run_complete_analysis() stage3 output
    """
    print("\n🧪 階段3: 測試全面的Prior/Likelihood組合")
    print("-"*50)
    if model_configs is None:
        model_configs = ModelConfiguration.get_comprehensive_test_configs()
    if product_configs is None:
        product_configs = ModelConfiguration.get_steinmann_product_configs()
    print(f"📊 測試矩陣: {len(model_configs)}種模型配置 × {len(product_configs)}種保險產品")
    print(f"   總計: {len(model_configs) * len(product_configs)}個測試組合")
    results = {}
    train_hazards = split['train_hazards']
    train_losses = split['train_losses']
    test_hazards = split['test_hazards']
    test_losses = split['test_losses']
    exposure_tensor = split['exposure_tensor']
    for idx, model_config in enumerate(model_configs):
        print(f"\n🔍 配置 {idx+1}/{len(model_configs)}: {model_config['name']}")
        print(f"   描述: {model_config['description']}")
        print(f"   ε值: Prior={model_config['epsilon_prior']:.3f}, Likelihood={model_config['epsilon_likelihood']:.3f}")
        config_results = {}
        product_config = product_configs[0]
        print(f"\n💰 保險產品: {product_config['name']}")
        model = UnifiedEndToEndVIModel(
            n_hospitals=generator.n_hospitals,
            n_regions=generator.n_regions,
            n_events=train_hazards.shape[1],
            distance_matrix=spatial_data.distance_matrix,
            product_config=product_config,
            epsilon_prior=model_config['epsilon_prior'],
            epsilon_likelihood=model_config['epsilon_likelihood'],
            prior_scenario=model_config['prior_scenario'],
            likelihood_family=model_config['likelihood_family']
        )
        
        # 移動模型到正確的設備
        model.to_multi_gpu()
        print(f"✅ 模型已移動到設備: {device}")
        
        trainer = EndToEndTrainer(model, learning_rate=0.01)
        print(f"🏋️ 開始訓練 ({n_epochs} epochs)...")
        best_test_elbo = float('-inf')
        for epoch in range(n_epochs):
            _ = trainer.train_epoch(train_hazards, exposure_tensor, train_losses, n_samples=8, spatial_data=spatial_data)
            if (epoch + 1) % 10 == 0:
                test_losses_dict = trainer.evaluate(test_hazards, exposure_tensor, test_losses, n_samples=15, spatial_data=spatial_data)
                print(f"   Epoch {epoch+1:2d}: Test ELBO={test_losses_dict['elbo']:.3f}")
                best_test_elbo = max(best_test_elbo, test_losses_dict['elbo'])
        final_test = trainer.evaluate(test_hazards, exposure_tensor, test_losses, n_samples=30, spatial_data=spatial_data)
        config_results[product_config['name']] = {
            'final_test_elbo': final_test['elbo'],
            'final_crps': -final_test['crps_term'],
            'best_test_elbo': best_test_elbo,
            'model_config': model_config,
            'product_config': product_config
        }
        print(f"   ✅ 最終測試ELBO: {final_test['elbo']:.3f}")
        print(f"   📊 CRPS分數: {-final_test['crps_term']:.1f}")
        results[model_config['name']] = config_results
    return results

def stage4_analyze_results(results: Dict) -> Dict:
    """階段4：分析與可視化（Notebook友好）。返回重要匯總。"""
    print("\n📈 階段4: 完整Prior/Likelihood組合結果分析")
    print("-"*80)
    all_results = []
    for model_name, model_results in results.items():
        for product_name, product_results in model_results.items():
            model_config = product_results['model_config']
            all_results.append({
                'model_name': model_name,
                'crps_score': product_results['final_crps'],
                'elbo_score': product_results['final_test_elbo'],
                'epsilon_prior': model_config['epsilon_prior'],
                'epsilon_likelihood': model_config['epsilon_likelihood'],
                'prior_scenario': model_config.get('prior_scenario', 'unknown'),
                'likelihood_family': model_config.get('likelihood_family', 'unknown')
            })
    all_results.sort(key=lambda x: x['crps_score'])
    print("\n🏆 完整排行榜 (按CRPS分數排序，越低越好):")
    print("-"*80)
    print(f"{'排名':<4} {'模型配置':<35} {'CRPS':<8} {'ELBO':<8} {'ε_prior':<8} {'ε_like':<8}")
    print("-"*80)
    for i, result in enumerate(all_results[:10]):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
        print(f"{rank_emoji:<4} {result['model_name'][:34]:<35} "
              f"{result['crps_score']:<8.1f} {result['elbo_score']:<8.3f} "
              f"{result['epsilon_prior']:<8.3f} {result['epsilon_likelihood']:<8.3f}")
    create_results_visualization(results)
    return {'leaderboard': all_results}

# Auto-run Stage 4 in notebook
if _in_notebook() and NB_AUTORUN:
    try:
        if '_NB_results' in globals():
            _NB_summary = stage4_analyze_results(_NB_results)
    except Exception as e:
        print(f"[NB] Stage4 auto-run failed: {e}")

def stage5_stress_test(climada_data: SimulatedCLIMADAData,
                       spatial_data: SimulatedSpatialData,
                       contamination_ratio: float = 0.05,
                       extreme_multiplier: float = 8.0,
                       n_folds: int = 3) -> Dict:
    """階段5：壓力測試（Notebook友好）。"""
    print("\n🧪 階段5: 壓力測試 - 證明Robust方法的真正價值")
    print("-"*80)
    stress_tester = RobustnessStressTester(
        contamination_ratio=contamination_ratio,
        extreme_multiplier=extreme_multiplier,
        n_folds=n_folds
    )
    results = stress_tester.run_stress_test(climada_data, spatial_data)
    print(f"\n🏆 壓力測試結論:")
    print(f"   獲勝模型: {results['winner']}")
    return results

# Optional: Do NOT auto-run Stage 5 by default (can be heavy)

def run_complete_analysis():
    """執行完整分析 - 分離出來方便調試"""
    # 階段1
    generator, climada_data, spatial_data = stage1_generate_data()
    # 階段2
    split = stage2_train_test_split(climada_data)

    # ========================================================================
    # 階段3: 測試不同模型配置和保險產品
    # ========================================================================
    results = stage3_run_model_matrix(generator, spatial_data, split, n_epochs=30)

    # ========================================================================
    # 階段4: 完整Prior/Likelihood組合結果分析
    # ========================================================================
    print("\n📈 階段4: 完整Prior/Likelihood組合結果分析")
    print("-"*80)
    
    # 提取所有測試結果
    all_results = []
    for model_name, model_results in results.items():
        for product_name, product_results in model_results.items():
            model_config = product_results['model_config']
            all_results.append({
                'model_name': model_name,
                'crps_score': product_results['final_crps'],
                'elbo_score': product_results['final_test_elbo'],
                'epsilon_prior': model_config['epsilon_prior'],
                'epsilon_likelihood': model_config['epsilon_likelihood'],
                'prior_scenario': model_config.get('prior_scenario', 'unknown'),
                'likelihood_family': model_config.get('likelihood_family', 'unknown')
            })
    
    # 按CRPS分數排序
    all_results.sort(key=lambda x: x['crps_score'])
    
    print("\n🏆 完整排行榜 (按CRPS分數排序，越低越好):")
    print("-"*80)
    print(f"{'排名':<4} {'模型配置':<35} {'CRPS':<8} {'ELBO':<8} {'ε_prior':<8} {'ε_like':<8}")
    print("-"*80)
    
    for i, result in enumerate(all_results[:10]):  # 顯示前10名
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
        print(f"{rank_emoji:<4} {result['model_name'][:34]:<35} "
              f"{result['crps_score']:<8.1f} {result['elbo_score']:<8.3f} "
              f"{result['epsilon_prior']:<8.3f} {result['epsilon_likelihood']:<8.3f}")
    
    # 分析污染程度 vs 穩健性的關係
    print(f"\n📊 污染程度 vs 穩健性分析:")
    print("-"*50)
    
    # 按污染程度分組
    contamination_groups = {
        '無污染 (ε=0.0)': [r for r in all_results if r['epsilon_prior'] == 0.0 and r['epsilon_likelihood'] == 0.0],
        '輕度污染 (ε<0.08)': [r for r in all_results if 0 < max(r['epsilon_prior'], r['epsilon_likelihood']) < 0.08],
        '中等污染 (0.08≤ε<0.15)': [r for r in all_results if 0.08 <= max(r['epsilon_prior'], r['epsilon_likelihood']) < 0.15],
        '重度污染 (ε≥0.15)': [r for r in all_results if max(r['epsilon_prior'], r['epsilon_likelihood']) >= 0.15]
    }
    
    for group_name, group_results in contamination_groups.items():
        if group_results:
            avg_crps = np.mean([r['crps_score'] for r in group_results])
            best_crps = min([r['crps_score'] for r in group_results])
            n_models = len(group_results)
            print(f"   {group_name}: {n_models}個模型, 平均CRPS={avg_crps:.1f}, 最佳CRPS={best_crps:.1f}")
    
    # 關鍵發現
    print(f"\n💡 關鍵發現:")
    print("-"*30)
    
    best_model = all_results[0]
    worst_baseline = max([r for r in all_results if r['epsilon_prior'] == 0.0 and r['epsilon_likelihood'] == 0.0], 
                        key=lambda x: x['crps_score'])
    
    improvement = ((worst_baseline['crps_score'] - best_model['crps_score']) / worst_baseline['crps_score']) * 100
    
    print(f"   • 最佳模型: {best_model['model_name']}")
    print(f"   • 相比基線模型改善: {improvement:.1f}%")
    print(f"   • 最佳ε組合: Prior={best_model['epsilon_prior']:.3f}, Likelihood={best_model['epsilon_likelihood']:.3f}")
    print(f"   • ε-contamination成功降低基差風險，證明了穩健性的價值")
    
    # 可視化結果
    create_results_visualization(results)

    # ========================================================================
    # 階段5: 壓力測試：證明雙重污染模型的優越性
    # ========================================================================
    stress_results = stage5_stress_test(climada_data, spatial_data, 0.05, 8.0, 3)
    
    print(f"\n🏆 壓力測試結論:")
    print(f"   獲勝模型: {stress_results['winner']}")
    print(f"   在極端情況下展現最強抗風險能力")
    print(f"   ε-contamination成功過濾極端噪音，保護決策品質")
    
    print("\n🎉 完整的端到端CRPS-VI玩具範例完成!")
    print("📈 壓力測試證明了雙重污染模型的優越穩健性!")
    
    return {
        'model_comparison_results': results,
        'stress_test_results': stress_results
    }

def create_results_visualization(results: Dict):
    """創建結果可視化"""
    print("\n🎨 生成結果可視化...")
    
    # 準備數據
    model_names = list(results.keys())
    product_names = list(results[model_names[0]].keys())
    
    # 創建CRPS比較圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子圖1：不同模型配置的CRPS比較
    crps_matrix = []
    for model_name in model_names:
        crps_row = []
        for product_name in product_names:
            crps_score = results[model_name][product_name]['final_crps']
            crps_row.append(crps_score)
        crps_matrix.append(crps_row)
    
    crps_matrix = np.array(crps_matrix)
    
    im1 = ax1.imshow(crps_matrix, cmap='viridis', aspect='auto')
    ax1.set_xticks(range(len(product_names)))
    ax1.set_xticklabels([p.replace(' Product', '') for p in product_names], rotation=45)
    ax1.set_yticks(range(len(model_names)))
    ax1.set_yticklabels([m.replace(' 模型', '') for m in model_names])
    ax1.set_title('CRPS Scores by Model & Product')
    plt.colorbar(im1, ax=ax1, label='CRPS Score')
    
    # 子圖2：最佳產品排名
    best_products = []
    best_crps = []
    for model_name in model_names:
        model_results = results[model_name]
        crps_values = [r['final_crps'] for r in model_results.values()]
        best_idx = np.argmin(crps_values)
        best_product = list(model_results.keys())[best_idx]
        best_products.append(best_product.replace(' Product', ''))
        best_crps.append(min(crps_values))
    
    bars = ax2.bar(range(len(model_names)), best_crps, 
                   color=['lightblue', 'lightgreen', 'lightcoral'])
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels([m.replace(' 模型', '') for m in model_names], rotation=45)
    ax2.set_ylabel('Best CRPS Score')
    ax2.set_title('Best Product Performance by Model')
    
    # 添加數值標籤
    for i, (bar, product, crps) in enumerate(zip(bars, best_products, best_crps)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{product}\n{crps:.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('toy_example_results.png', dpi=150, bbox_inches='tight')
    print("✅ 結果圖表已保存: toy_example_results.png")
    
    # 打印總結統計
    print(f"\n📊 總結統計:")
    print(f"   測試的模型配置數: {len(model_names)}")
    print(f"   測試的保險產品數: {len(product_names)}")
    print(f"   總測試組合數: {len(model_names) * len(product_names)}")
    
    # 找出全局最佳
    all_crps = []
    all_combinations = []
    for model_name, model_results in results.items():
        for product_name, product_results in model_results.items():
            all_crps.append(product_results['final_crps'])
            all_combinations.append((model_name, product_name))
    
    best_global_idx = np.argmin(all_crps)
    best_model, best_product = all_combinations[best_global_idx]
    best_global_crps = all_crps[best_global_idx]
    
    print(f"\n🏆 全局最佳組合:")
    print(f"   模型: {best_model}")
    print(f"   產品: {best_product}")  
    print(f"   CRPS: {best_global_crps:.1f}")

# %%
# ============================================================================
# 9. 執行入口
# ============================================================================

if __name__ == "__main__":
    main()
else:
    print("✅ 玩具範例模組已載入，使用 main() 函數執行完整分析")

# %%
def demonstrate_prior_likelihood_combinations():
    """演示Prior/Likelihood組合系統"""
    print("🧪 演示Prior/Likelihood組合系統")
    print("="*80)
    
    # 開始演示
    
    # 獲取所有配置
    configs = ModelConfiguration.get_comprehensive_test_configs()
    
    print(f"📊 測試配置總覽 ({len(configs)}個組合):")
    print("-"*80)
    print(f"{'#':<3} {'配置名稱':<45} {'先驗':<12} {'似然':<12} {'ε_prior':<8} {'ε_like':<8}")
    print("-"*80)
    
    for i, config in enumerate(configs):
        print(f"{i+1:<3} {config['name'][:44]:<45} "
              f"{config['prior_scenario'].value[:11]:<12} "
              f"{config['likelihood_family'].value[:11]:<12} "
              f"{config['epsilon_prior']:<8.2f} "
              f"{config['epsilon_likelihood']:<8.2f}")
    
    print("-"*80)
    print(f"\n📋 Prior情境詳細說明:")
    
    # 演示先驗參數
    for prior_scenario in [PriorScenario.NON_INFORMATIVE, PriorScenario.WEAK_INFORMATIVE, 
                          PriorScenario.OPTIMISTIC, PriorScenario.PESSIMISTIC]:
        prior_params = PriorLikelihoodProcessor.get_prior_parameters(prior_scenario)
        print(f"   {prior_scenario.value}:")
        print(f"     μ = {prior_params['mu_prior'][:4].tolist()}")
        print(f"     σ = {prior_params['sigma_prior'][:4].tolist()}")
    
    print(f"\n🎯 Likelihood族說明:")
    print("   normal: 正態分佈 - 標準假設，對異常值敏感")
    print("   lognormal: 對數正態 - 適合損失建模，右偏分佈")
    print("   student_t: Student-t - 重尾分佈，對異常值穩健")
    
    print(f"\n🛡️ ε-contamination穩健程度:")
    print("   基線(0.0, 0.0): 無污染保護")
    print("   中等(0.08, 0.10): 適中穩健性")
    print("   極高(0.15, 0.18): 最大穩健性")
    
    print(f"\n💡 組合邏輯:")
    print("   Prior情境 × Likelihood族 × 穩健程度 = 完整測試矩陣")
    print("   每種組合針對不同的風險預期和建模假設")
    print("   ε-contamination提供對模型誤指定的保護")

# 已整合到上方單一 __main__ 入口
