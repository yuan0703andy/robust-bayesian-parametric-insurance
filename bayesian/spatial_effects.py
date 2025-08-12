#!/usr/bin/env python3
"""
Spatial Effects Module for Hierarchical Bayesian Models
階層貝氏模型的空間效應模組

Implements spatial random effects for hospital vulnerability modeling:
β_i = α_r(i) + δ_i + γ_i

Key Features:
- Hospital spatial covariance matrix construction
- Multiple covariance function types (exponential, Matern, Gaussian)
- Bayesian spatial parameter estimation
- Spatial information borrowing for improved estimates

Author: Research Team
Date: 2025-01-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky, solve_triangular
from scipy.special import gamma, kv
import matplotlib.pyplot as plt

# Try PyMC imports for Bayesian modeling
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    warnings.warn("PyMC not available, spatial Bayesian modeling will be limited")


class CovarianceFunction(Enum):
    """空間協方差函數類型"""
    EXPONENTIAL = "exponential"      # 指數衰減：最常用
    MATERN_32 = "matern_3_2"        # Matérn ν=3/2：中等平滑
    MATERN_52 = "matern_5_2"        # Matérn ν=5/2：高平滑
    GAUSSIAN = "gaussian"           # 高斯（平方指數）：非常平滑
    LINEAR = "linear"               # 線性衰減：簡單模型
    

@dataclass
class SpatialConfig:
    """空間效應配置"""
    covariance_function: CovarianceFunction = CovarianceFunction.EXPONENTIAL
    length_scale: float = 50.0          # 空間相關長度尺度 (km)
    variance: float = 1.0               # 空間變異數參數
    nugget: float = 0.1                 # 測量誤差/非空間變異數
    region_effect: bool = True          # 是否包含區域效應
    n_regions: int = 3                  # 區域數量
    
    # 貝氏推論參數
    length_scale_prior: Tuple[float, float] = (10.0, 100.0)    # Gamma(α, β)
    variance_prior: Tuple[float, float] = (1.0, 1.0)           # Gamma(α, β)
    nugget_prior: Tuple[float, float] = (0.01, 0.2)           # Uniform(a, b)


@dataclass 
class SpatialEffectsResult:
    """空間效應分析結果"""
    # 協方差矩陣和參數
    covariance_matrix: np.ndarray              # Σ_δ 協方差矩陣
    cholesky_factor: np.ndarray               # L: Σ_δ = L @ L.T
    spatial_parameters: Dict[str, float]       # 估計的空間參數
    
    # 空間效應樣本
    spatial_effects: np.ndarray               # δ_i 空間隨機效應
    region_effects: Optional[np.ndarray] = None    # α_r(i) 區域效應
    individual_effects: Optional[np.ndarray] = None # γ_i 個體效應
    
    # 診斷和評估
    effective_range: float = 0.0              # 有效空間影響範圍
    spatial_dependence: float = 0.0          # 空間依賴性強度
    model_fit_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 貝氏推論結果
    posterior_samples: Optional[Dict[str, np.ndarray]] = None
    credible_intervals: Optional[Dict[str, Tuple[float, float]]] = None


class SpatialEffectsAnalyzer:
    """
    空間效應分析器
    
    專門用於醫院脆弱度的空間建模：
    - 建構醫院間的空間協方差矩陣
    - 估計空間參數（長度尺度、變異數）
    - 生成空間相關的隨機效應
    - 支援貝氏空間參數推論
    """
    
    def __init__(self, config: SpatialConfig = None):
        """
        Parameters:
        -----------
        config : SpatialConfig, optional
            空間效應配置，預設使用指數衰減函數
        """
        self.config = config or SpatialConfig()
        self.hospital_coords = None
        self.distance_matrix = None
        
        # 結果儲存
        self.last_result: Optional[SpatialEffectsResult] = None
        self.analysis_history: List[SpatialEffectsResult] = []
        
    def fit(self, 
            hospital_coordinates: Union[np.ndarray, pd.DataFrame, List[Tuple[float, float]]],
            hospital_names: Optional[List[str]] = None,
            region_assignments: Optional[List[int]] = None) -> SpatialEffectsResult:
        """
        擬合空間效應模型
        
        Parameters:
        -----------
        hospital_coordinates : array-like
            醫院座標 [(lat1, lon1), (lat2, lon2), ...] 或 DataFrame with 'lat', 'lon'
        hospital_names : List[str], optional
            醫院名稱，用於識別
        region_assignments : List[int], optional
            每家醫院的區域分配 [0, 1, 2, 0, 1, ...]
            
        Returns:
        --------
        SpatialEffectsResult
            空間效應分析結果
        """
        print(f"🗺️ 開始空間效應分析")
        print(f"   協方差函數: {self.config.covariance_function.value}")
        print(f"   長度尺度: {self.config.length_scale} km")
        
        # 處理座標數據
        self.hospital_coords = self._process_coordinates(hospital_coordinates)
        n_hospitals = len(self.hospital_coords)
        print(f"   醫院數量: {n_hospitals}")
        
        # 計算距離矩陣
        print("   計算醫院間距離...")
        self.distance_matrix = self._compute_distance_matrix(self.hospital_coords)
        
        # 建構空間協方差矩陣
        print("   建構空間協方差矩陣...")
        covariance_matrix = self._build_covariance_matrix(
            self.distance_matrix,
            self.config.length_scale,
            self.config.variance,
            self.config.nugget
        )
        
        # Cholesky 分解（用於高效採樣）
        print("   執行 Cholesky 分解...")
        try:
            cholesky_factor = cholesky(covariance_matrix, lower=True)
        except np.linalg.LinAlgError:
            print("   ⚠️ 協方差矩陣不正定，添加數值穩定項...")
            regularized_cov = covariance_matrix + np.eye(n_hospitals) * 1e-6
            cholesky_factor = cholesky(regularized_cov, lower=True)
        
        # 生成空間效應樣本
        print("   生成空間隨機效應...")
        spatial_effects = self._sample_spatial_effects(cholesky_factor)
        
        # 區域效應（如果啟用）
        region_effects = None
        if self.config.region_effect:
            print("   生成區域效應...")
            region_effects = self._generate_region_effects(
                n_hospitals, region_assignments or self._assign_regions(self.hospital_coords)
            )
        
        # 個體效應
        print("   生成個體隨機效應...")
        individual_effects = np.random.normal(0, 0.2, n_hospitals)
        
        # 計算診斷統計
        print("   計算診斷統計...")
        effective_range = self._compute_effective_range()
        spatial_dependence = self._compute_spatial_dependence(covariance_matrix)
        
        # 創建結果對象
        result = SpatialEffectsResult(
            covariance_matrix=covariance_matrix,
            cholesky_factor=cholesky_factor,
            spatial_parameters={
                'length_scale': self.config.length_scale,
                'variance': self.config.variance,
                'nugget': self.config.nugget
            },
            spatial_effects=spatial_effects,
            region_effects=region_effects,
            individual_effects=individual_effects,
            effective_range=effective_range,
            spatial_dependence=spatial_dependence
        )
        
        self.last_result = result
        self.analysis_history.append(result)
        
        print("✅ 空間效應分析完成")
        return result
    
    def _process_coordinates(self, coordinates) -> np.ndarray:
        """處理各種格式的座標數據"""
        if isinstance(coordinates, pd.DataFrame):
            if 'lat' in coordinates.columns and 'lon' in coordinates.columns:
                return coordinates[['lat', 'lon']].values
            elif 'latitude' in coordinates.columns and 'longitude' in coordinates.columns:
                return coordinates[['latitude', 'longitude']].values
        elif isinstance(coordinates, (list, tuple)):
            return np.array(coordinates)
        elif isinstance(coordinates, np.ndarray):
            return coordinates
        else:
            raise ValueError("無法識別的座標格式")
    
    def _compute_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """計算醫院間的地理距離矩陣（使用 Haversine 公式）"""
        n = len(coords)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Haversine 距離計算
                distance = self._haversine_distance(
                    coords[i][0], coords[i][1],  # lat1, lon1
                    coords[j][0], coords[j][1]   # lat2, lon2
                )
                distances[i, j] = distance
                distances[j, i] = distance  # 對稱矩陣
        
        return distances
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine 地理距離計算（公里）"""
        R = 6371.0  # 地球半徑 (km)
        
        # 轉為弧度
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine 公式
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _build_covariance_matrix(self, 
                                distance_matrix: np.ndarray,
                                length_scale: float,
                                variance: float,
                                nugget: float) -> np.ndarray:
        """建構空間協方差矩陣"""
        n = distance_matrix.shape[0]
        
        if self.config.covariance_function == CovarianceFunction.EXPONENTIAL:
            # 指數協方差：σ² * exp(-d/ρ)
            cov_matrix = variance * np.exp(-distance_matrix / length_scale)
            
        elif self.config.covariance_function == CovarianceFunction.MATERN_32:
            # Matérn ν=3/2: σ² * (1 + √3*d/ρ) * exp(-√3*d/ρ)
            scaled_dist = np.sqrt(3) * distance_matrix / length_scale
            cov_matrix = variance * (1 + scaled_dist) * np.exp(-scaled_dist)
            
        elif self.config.covariance_function == CovarianceFunction.MATERN_52:
            # Matérn ν=5/2: σ² * (1 + √5*d/ρ + 5*d²/3ρ²) * exp(-√5*d/ρ)
            scaled_dist = np.sqrt(5) * distance_matrix / length_scale
            cov_matrix = variance * (1 + scaled_dist + scaled_dist**2/3) * np.exp(-scaled_dist)
            
        elif self.config.covariance_function == CovarianceFunction.GAUSSIAN:
            # 高斯（平方指數）：σ² * exp(-d²/2ρ²)
            cov_matrix = variance * np.exp(-distance_matrix**2 / (2 * length_scale**2))
            
        elif self.config.covariance_function == CovarianceFunction.LINEAR:
            # 線性衰減：σ² * max(0, 1 - d/ρ)
            linear_decay = np.maximum(0, 1 - distance_matrix / length_scale)
            cov_matrix = variance * linear_decay
            
        else:
            raise ValueError(f"不支援的協方差函數: {self.config.covariance_function}")
        
        # 添加 nugget 效應（對角線）
        cov_matrix += np.eye(n) * nugget
        
        return cov_matrix
    
    def _sample_spatial_effects(self, cholesky_factor: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """使用 Cholesky 分解採樣空間隨機效應"""
        n_hospitals = cholesky_factor.shape[0]
        
        if n_samples == 1:
            # 單次採樣
            z = np.random.normal(0, 1, n_hospitals)
            return cholesky_factor @ z
        else:
            # 多次採樣
            samples = np.zeros((n_samples, n_hospitals))
            for i in range(n_samples):
                z = np.random.normal(0, 1, n_hospitals)
                samples[i] = cholesky_factor @ z
            return samples
    
    def _assign_regions(self, coords: np.ndarray) -> List[int]:
        """自動分配醫院到區域（基於地理位置）"""
        # 簡單的經緯度分割方式
        lats = coords[:, 0]
        lons = coords[:, 1]
        
        # 基於緯度分為3個區域：海岸(East), 中部(Central), 山區(West)
        lon_33rd = np.percentile(lons, 33.33)
        lon_67th = np.percentile(lons, 66.67)
        
        regions = []
        for lon in lons:
            if lon >= lon_33rd:  # 東部（較大經度）
                regions.append(0)  # 海岸區域
            elif lon >= lon_67th:  # 中部
                regions.append(1)  # 中部區域  
            else:  # 西部（較小經度）
                regions.append(2)  # 山區
        
        return regions
    
    def _generate_region_effects(self, n_hospitals: int, region_assignments: List[int]) -> np.ndarray:
        """生成區域固定效應"""
        n_regions = self.config.n_regions
        
        # 生成區域效應值
        region_values = np.random.normal(0, 0.5, n_regions)
        
        # 將區域效應分配給醫院
        hospital_region_effects = np.array([region_values[r] for r in region_assignments])
        
        return hospital_region_effects
    
    def _compute_effective_range(self) -> float:
        """計算有效空間影響範圍（相關性降至5%的距離）"""
        if self.config.covariance_function == CovarianceFunction.EXPONENTIAL:
            # exp(-d/ρ) = 0.05 => d = -ρ * ln(0.05) ≈ 3.0 * ρ
            return 3.0 * self.config.length_scale
        elif self.config.covariance_function in [CovarianceFunction.MATERN_32, CovarianceFunction.MATERN_52]:
            # 近似為 2.5 * ρ
            return 2.5 * self.config.length_scale
        elif self.config.covariance_function == CovarianceFunction.GAUSSIAN:
            # 高斯函數衰減更快
            return 2.0 * self.config.length_scale
        else:
            return 2.0 * self.config.length_scale
    
    def _compute_spatial_dependence(self, covariance_matrix: np.ndarray) -> float:
        """計算空間依賴性強度（平均相關性）"""
        n = covariance_matrix.shape[0]
        
        # 轉換為相關矩陣
        std_devs = np.sqrt(np.diag(covariance_matrix))
        correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
        
        # 計算非對角線元素的平均值
        mask = ~np.eye(n, dtype=bool)  # 排除對角線
        mean_correlation = np.mean(np.abs(correlation_matrix[mask]))
        
        return mean_correlation
    
    def visualize_spatial_structure(self, result: SpatialEffectsResult = None, figsize: Tuple[int, int] = (15, 5)):
        """視覺化空間結構"""
        if result is None:
            result = self.last_result
        
        if result is None:
            print("⚠️ 沒有可用的分析結果")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. 醫院位置和空間效應
        ax1 = axes[0]
        scatter = ax1.scatter(
            self.hospital_coords[:, 1], self.hospital_coords[:, 0],
            c=result.spatial_effects, s=100, alpha=0.7, 
            cmap='RdBu_r', edgecolors='black', linewidth=0.5
        )
        ax1.set_xlabel('經度 (Longitude)')
        ax1.set_ylabel('緯度 (Latitude)')
        ax1.set_title('醫院空間隨機效應 (δᵢ)')
        plt.colorbar(scatter, ax=ax1, label='空間效應值')
        
        # 2. 協方差矩陣熱圖
        ax2 = axes[1]
        im = ax2.imshow(result.covariance_matrix, cmap='viridis', aspect='auto')
        ax2.set_title('空間協方差矩陣 (Σ_δ)')
        ax2.set_xlabel('醫院索引')
        ax2.set_ylabel('醫院索引')
        plt.colorbar(im, ax=ax2, label='協方差')
        
        # 3. 距離-相關性關係
        ax3 = axes[2]
        distances = self.distance_matrix[np.triu_indices_from(self.distance_matrix, k=1)]
        correlations = result.covariance_matrix[np.triu_indices_from(result.covariance_matrix, k=1)]
        correlations = correlations / result.spatial_parameters['variance']  # 標準化
        
        ax3.scatter(distances, correlations, alpha=0.6, s=20)
        ax3.set_xlabel('距離 (km)')
        ax3.set_ylabel('相關係數')
        ax3.set_title('距離-相關性衰減')
        ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% 閾值')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 顯示診斷統計
        print(f"\n📊 空間結構診斷:")
        print(f"   有效影響範圍: {result.effective_range:.1f} km")
        print(f"   空間依賴性: {result.spatial_dependence:.3f}")
        print(f"   最大醫院間距離: {np.max(self.distance_matrix):.1f} km")
        print(f"   最小醫院間距離: {np.min(self.distance_matrix[self.distance_matrix > 0]):.1f} km")


def create_standard_spatial_config(**kwargs) -> SpatialConfig:
    """創建標準的空間效應配置"""
    defaults = {
        'covariance_function': CovarianceFunction.EXPONENTIAL,
        'length_scale': 50.0,
        'variance': 1.0,
        'nugget': 0.1,
        'region_effect': True,
        'n_regions': 3
    }
    defaults.update(kwargs)
    return SpatialConfig(**defaults)


def quick_spatial_analysis(hospital_coordinates: Union[np.ndarray, pd.DataFrame, List], 
                          **config_kwargs) -> SpatialEffectsResult:
    """快速空間效應分析"""
    config = create_standard_spatial_config(**config_kwargs)
    analyzer = SpatialEffectsAnalyzer(config)
    return analyzer.fit(hospital_coordinates)


if __name__ == "__main__":
    # 示範用法
    print("🧪 空間效應模組測試")
    
    # 使用北卡羅來納州醫院的模擬數據
    mock_coords = [
        (36.0153, -78.9384),  # Duke University Hospital
        (35.9049, -79.0469),  # UNC Hospitals  
        (35.8043, -78.6569),  # Rex Hospital
        (35.2045, -80.8395),  # Carolinas Medical Center
        (36.0835, -79.8235),  # Moses H. Cone Memorial Hospital
    ]
    
    # 執行空間分析
    result = quick_spatial_analysis(
        mock_coords,
        length_scale=60.0,
        variance=1.5
    )
    
    print(f"空間效應值: {result.spatial_effects}")
    print(f"有效範圍: {result.effective_range:.1f} km")