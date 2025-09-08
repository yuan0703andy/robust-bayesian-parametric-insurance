


"""Data structures and generator moved from toy_example_complete.py."""
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

# %%
# ============================================================================
# 1. 數據結構定義（模仿 CLIMADA 和空間分析格式）
# ============================================================================

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

# ============================================================================
# 2. 數據生成器（模擬真實 CLIMADA 和空間分析數據）
# ============================================================================

class ToyDataGenerator:
    """玩具數據生成器（支援極端事件）"""
    
    def __init__(self, n_hospitals=20, n_events=120, n_regions=4,
                 extreme_event_ratio: float = 0.10,
                 extreme_hazard_multiplier: float = 2.5,
                 extreme_event_hospital_fraction: float = 0.7,
                 max_wind_speed: float = 120.0):
        self.n_hospitals = n_hospitals
        self.n_events = n_events
        self.n_regions = n_regions
        # 極端事件控制
        self.extreme_event_ratio = float(extreme_event_ratio)
        self.extreme_hazard_multiplier = float(extreme_hazard_multiplier)
        self.extreme_event_hospital_fraction = float(extreme_event_hospital_fraction)
        self.max_wind_speed = float(max_wind_speed)
        
    def generate_climada_data(self) -> SimulatedCLIMADAData:
        """生成模擬的CLIMADA數據"""
        print(f"🏥 生成模擬CLIMADA數據: {self.n_hospitals}醫院 × {self.n_events}事件")
        
        # 1. 醫院座標（北卡羅來納州範圍）
        hospital_coords = np.random.uniform([35.0, -84.0], [36.5, -75.5], 
                                          (self.n_hospitals, 2))
        
        # 2. 風速數據（颱風強度，單位: m/s）
        # 基於真實颱風分佈：大部分中等強度，少數極強
        base_intensities = np.random.gamma(2, 15, (self.n_hospitals, self.n_events))
        # # 添加地理相關性：靠近海岸的醫院受到更強風速
        # coastal_factor = np.exp(-hospital_coords[:, 1] + 80)  # 越靠東海岸越強
        # hazard_intensities = base_intensities * coastal_factor.reshape(-1, 1)
        # hazard_intensities = np.clip(hazard_intensities, 10, 80)  # 10-80 m/s range
        # 修正海岸相關性：使用 0~1 的「海岸性」避免指數爆炸；靠東(經度大)風更強
        lon = hospital_coords[:, 1]
        min_lon, max_lon = -84.0, -75.5
        coastiness = np.clip((lon - min_lon) / (max_lon - min_lon), 0.0, 1.0)  # 0(西)→1(東)
        coastal_factor = 1.0 + 0.6 * coastiness  # 1.0~1.6，線性放大，不會把事件全夾到80
        hazard_intensities = base_intensities * coastiness.reshape(-1, 1) * 0 + base_intensities * coastal_factor.reshape(-1, 1)
        hazard_intensities = np.clip(hazard_intensities, 10, 80)  # 10–80 m/s 合理範圍（常規）
        
        # 2.1 注入極端事件：放大部分事件的風速並以更高上限截斷
        n_extreme_events = max(1, int(self.n_events * self.extreme_event_ratio))
        extreme_event_indices = np.random.choice(self.n_events, n_extreme_events, replace=False)
        n_affected = max(1, int(self.extreme_event_hospital_fraction * self.n_hospitals))
        for e_idx in extreme_event_indices:
            affected = np.random.choice(self.n_hospitals, n_affected, replace=False)
            hazard_intensities[affected, e_idx] *= self.extreme_hazard_multiplier
        # 允許極端更高風速
        hazard_intensities = np.clip(hazard_intensities, 10, self.max_wind_speed)
        
        # 3. 暴露價值（醫院資產價值，單位：美元）
        # 醫院級資產：中位 ~ 2.5e8，離散度稍大（更貼近大型綜合醫院/醫學中心）
        exposure_values = np.random.lognormal(mean=np.log(10e6), sigma=0.4, size=self.n_hospitals)

        # 4. 觀測損失（使用真實的Emanuel脆弱度函數生成）
        observed_losses = self._generate_realistic_losses(
            hazard_intensities, exposure_values
        )
        
        # 5. 模擬颱風路徑數據
        track_data = {
            'track_ids': [f'track_{i:03d}' for i in range(self.n_events)],
            'years': np.random.choice(range(2000, 2024), self.n_events),
            'max_sustained_winds': np.max(hazard_intensities, axis=0),
            'categories': self._classify_hurricane_categories(hazard_intensities),
            'extreme_event_indices': extreme_event_indices.tolist(),
            'extreme_event_ratio': self.extreme_event_ratio,
            'extreme_hazard_multiplier': self.extreme_hazard_multiplier
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
        """基於座標分配區域（修正版）- 確保返回長度等於座標數"""
        n_coords = len(coords)
        print(f"   [_assign_regions] 輸入座標數: {n_coords}, 目標區域數: {self.n_regions}")
        try:
            from sklearn.cluster import KMeans
            actual_n_regions = min(self.n_regions, n_coords)
            if actual_n_regions < self.n_regions:
                print(f"   ⚠️ 座標數 {n_coords} < 區域數 {self.n_regions}, 調整為 {actual_n_regions} 個區域")
            kmeans = KMeans(n_clusters=actual_n_regions, random_state=42, n_init=10)
            assignments = kmeans.fit_predict(coords)
            print(f"   ✅ K-means 完成: 返回 {len(assignments)} 個分配")
        except Exception as e:
            print(f"   ⚠️ K-means 失敗 ({e})，使用循環分配")
            assignments = np.array([i % self.n_regions for i in range(n_coords)])
        assert len(assignments) == n_coords, f"分配長度錯誤: {len(assignments)} != {n_coords}"
        print(f"   ✅ 區域分配: {n_coords}家醫院 → {len(np.unique(assignments))}個區域")
        try:
            counts = np.bincount(assignments)
            print(f"   每個區域的醫院數: {counts.tolist()}")
        except Exception:
            pass
        return assignments
    
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

__all__ = [
    'SimulatedCLIMADAData',
    'SimulatedSpatialData', 
    'ToyDataGenerator'
]