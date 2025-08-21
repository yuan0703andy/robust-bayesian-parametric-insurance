#!/usr/bin/env python3
"""
Spatial Data Processor for Hierarchical Bayesian Model
空間數據處理模組

專門處理：
1. 真實醫院座標 → 距離矩陣計算
2. 地理區域自動分配
3. Cat-in-Circle數據預處理
4. 為階層模型提供正確的空間結構數據

用法：
from robust_hierarchical_bayesian_simulation.spatial_data_processor import SpatialDataProcessor
processor = SpatialDataProcessor()
spatial_data = processor.process_hospital_spatial_data(hospital_coords, cat_in_circle_data)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
import warnings

@dataclass
class SpatialData:
    """空間數據結構"""
    hospital_coords: np.ndarray          # (n_hospitals, 2) lat,lon
    distance_matrix: np.ndarray          # (n_hospitals, n_hospitals) km
    region_assignments: np.ndarray       # (n_hospitals,) region indices
    n_hospitals: int
    n_regions: int
    
    # Cat-in-Circle數據
    hazard_intensities: Optional[np.ndarray] = None  # (n_hospitals, n_events) H_ij
    exposure_values: Optional[np.ndarray] = None     # (n_hospitals,) E_i
    observed_losses: Optional[np.ndarray] = None     # (n_hospitals, n_events) L_ij
    

class SpatialDataProcessor:
    """空間數據處理器"""
    
    def __init__(self):
        self.spatial_data = None
        
    def process_hospital_spatial_data(self,
                                    hospital_coords: np.ndarray,
                                    n_regions: int = 3,
                                    region_method: str = "geographic") -> SpatialData:
        """
        處理醫院空間數據
        
        Parameters:
        -----------
        hospital_coords : np.ndarray, shape (n_hospitals, 2)
            醫院座標 [lat, lon]
        n_regions : int
            區域數量，預設3 (沿海/內陸/山區)
        region_method : str
            區域分配方法 "geographic" 或 "risk_based"
            
        Returns:
        --------
        SpatialData : 處理後的空間數據
        """
        n_hospitals = len(hospital_coords)
        
        print(f"🗺️ 處理醫院空間數據...")
        print(f"   醫院數量: {n_hospitals}")
        print(f"   座標範圍: lat [{hospital_coords[:,0].min():.3f}, {hospital_coords[:,0].max():.3f}]")
        print(f"              lon [{hospital_coords[:,1].min():.3f}, {hospital_coords[:,1].max():.3f}]")
        
        # 計算距離矩陣
        distance_matrix = self._calculate_haversine_distance_matrix(hospital_coords)
        print(f"   距離範圍: {distance_matrix[distance_matrix > 0].min():.1f} - {distance_matrix.max():.1f} km")
        
        # 區域分配
        if region_method == "geographic":
            region_assignments = self._assign_geographic_regions(hospital_coords, n_regions)
        elif region_method == "risk_based":
            region_assignments = self._assign_risk_based_regions(hospital_coords, n_regions)
        else:
            raise ValueError(f"未知的區域分配方法: {region_method}")
            
        region_counts = np.bincount(region_assignments)
        print(f"   區域分配: {dict(enumerate(region_counts))}")
        
        self.spatial_data = SpatialData(
            hospital_coords=hospital_coords,
            distance_matrix=distance_matrix,
            region_assignments=region_assignments,
            n_hospitals=n_hospitals,
            n_regions=n_regions
        )
        
        return self.spatial_data
    
    def add_cat_in_circle_data(self,
                              hazard_intensities: np.ndarray,
                              exposure_values: np.ndarray,
                              observed_losses: np.ndarray) -> SpatialData:
        """
        添加Cat-in-Circle數據
        
        Parameters:
        -----------
        hazard_intensities : np.ndarray, shape (n_hospitals, n_events)
            災害強度 H_ij (來自Cat-in-Circle分析)
        exposure_values : np.ndarray, shape (n_hospitals,)
            曝險價值 E_i
        observed_losses : np.ndarray, shape (n_hospitals, n_events)
            觀測損失 L_ij
        """
        if self.spatial_data is None:
            raise ValueError("請先調用process_hospital_spatial_data()")
            
        n_hospitals, n_events = hazard_intensities.shape
        
        if n_hospitals != self.spatial_data.n_hospitals:
            raise ValueError(f"醫院數量不匹配: {n_hospitals} vs {self.spatial_data.n_hospitals}")
            
        print(f"🌪️ 添加Cat-in-Circle數據...")
        print(f"   事件數量: {n_events}")
        print(f"   災害強度範圍: {hazard_intensities.min():.1f} - {hazard_intensities.max():.1f}")
        print(f"   曝險價值總和: ${exposure_values.sum():,.0f}")
        print(f"   損失範圍: ${observed_losses.min():,.0f} - ${observed_losses.max():,.0f}")
        
        self.spatial_data.hazard_intensities = hazard_intensities
        self.spatial_data.exposure_values = exposure_values  
        self.spatial_data.observed_losses = observed_losses
        
        return self.spatial_data
    
    def get_spatial_correlation_matrix(self, 
                                     spatial_range: float = 50.0,
                                     nugget: float = 0.01) -> np.ndarray:
        """
        計算空間相關矩陣
        
        Parameters:
        -----------
        spatial_range : float
            空間相關範圍 (km)
        nugget : float
            nugget效應 (最小相關值)
            
        Returns:
        --------
        np.ndarray : 空間相關矩陣 (n_hospitals, n_hospitals)
        """
        if self.spatial_data is None:
            raise ValueError("請先處理空間數據")
            
        # 指數衰減相關: exp(-distance / range)
        correlation_matrix = np.exp(-self.spatial_data.distance_matrix / spatial_range)
        
        # 添加nugget效應
        np.fill_diagonal(correlation_matrix, 1.0 + nugget)
        
        return correlation_matrix
    
    def _calculate_haversine_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """計算Haversine距離矩陣 (km)"""
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371.0  # 地球半徑 km
            
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        
        n = len(coords)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = haversine_distance(coords[i,0], coords[i,1], 
                                        coords[j,0], coords[j,1])
                distance_matrix[i,j] = distance_matrix[j,i] = dist
        
        return distance_matrix
    
    def _assign_geographic_regions(self, coords: np.ndarray, n_regions: int) -> np.ndarray:
        """基於地理位置的區域分配 (K-means聚類)"""
        kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
        region_assignments = kmeans.fit_predict(coords)
        
        return region_assignments
    
    def _assign_risk_based_regions(self, coords: np.ndarray, n_regions: int) -> np.ndarray:
        """基於風險的區域分配 (例如：沿海距離)"""
        # 簡化實現：基於經度分配 (東部沿海 vs 西部內陸)
        lons = coords[:, 1]
        
        if n_regions == 2:
            # 沿海 vs 內陸
            coastal_threshold = np.percentile(lons, 40)  # 東部40%為沿海
            regions = (lons > coastal_threshold).astype(int)
        elif n_regions == 3:
            # 沿海 / 中部 / 山區
            coastal_threshold = np.percentile(lons, 33)
            mountain_threshold = np.percentile(lons, 67)
            regions = np.zeros(len(lons), dtype=int)
            regions[lons > coastal_threshold] = 1  # 沿海
            regions[lons < mountain_threshold] = 2  # 山區
        else:
            # 回到地理聚類
            return self._assign_geographic_regions(coords, n_regions)
            
        return regions


def load_spatial_data_from_02_results(spatial_analysis_path: str) -> Optional[SpatialData]:
    """
    從02_spatial_analysis.py的結果載入空間數據
    
    Parameters:
    -----------
    spatial_analysis_path : str
        空間分析結果路徑 (通常是 "results/spatial_analysis/cat_in_circle_results.pkl")
        
    Returns:
    --------
    Optional[SpatialData] : 空間數據，如果載入失敗則返回None
    """
    try:
        import pickle
        with open(spatial_analysis_path, 'rb') as f:
            spatial_results = pickle.load(f)
        
        print(f"📂 載入02空間分析結果...")
        
        if 'hospital_coordinates' in spatial_results and 'hospitals' in spatial_results:
            processor = SpatialDataProcessor()
            
            # 處理醫院空間數據
            spatial_data = processor.process_hospital_spatial_data(
                hospital_coords=spatial_results['hospital_coordinates'],
                n_regions=3
            )
            
            # 如果有Cat-in-Circle數據也加入
            if 'indices' in spatial_results:
                indices = spatial_results['indices']
                # 選擇50km作為主要指標
                if 'cat_in_circle_50km_max' in indices:
                    n_events = len(indices['cat_in_circle_50km_max'])
                    n_hospitals = len(spatial_results['hospital_coordinates'])
                    
                    # 創建災害強度矩陣 (所有醫院使用相同的風速序列)
                    hazard_intensities = np.tile(
                        indices['cat_in_circle_50km_max'], 
                        (n_hospitals, 1)
                    )
                    
                    # 模擬曝險價值和觀測損失 (實際應用中這些來自CLIMADA)
                    exposure_values = np.random.uniform(1e7, 5e7, n_hospitals)
                    observed_losses = np.random.lognormal(15, 1, (n_hospitals, n_events))
                    
                    spatial_data = processor.add_cat_in_circle_data(
                        hazard_intensities, exposure_values, observed_losses
                    )
            
            print(f"✅ 成功載入空間數據: {spatial_data.n_hospitals}家醫院")
            return spatial_data
            
        else:
            print(f"⚠️ 02結果中缺少醫院數據")
            return None
            
    except Exception as e:
        print(f"❌ 載入02結果失敗: {e}")
        return None


# 使用範例
if __name__ == "__main__":
    # 測試空間數據處理
    np.random.seed(42)
    
    # 模擬北卡醫院座標
    hospital_coords = np.random.uniform([35.0, -84.0], [36.5, -75.5], (10, 2))
    
    # 處理空間數據
    processor = SpatialDataProcessor()
    spatial_data = processor.process_hospital_spatial_data(hospital_coords, n_regions=3)
    
    # 添加Cat-in-Circle數據
    n_hospitals, n_events = 10, 50
    hazard_intensities = np.random.uniform(20, 60, (n_hospitals, n_events))
    exposure_values = np.random.uniform(1e7, 5e7, n_hospitals)
    observed_losses = np.random.lognormal(15, 1, (n_hospitals, n_events))
    
    spatial_data = processor.add_cat_in_circle_data(
        hazard_intensities, exposure_values, observed_losses
    )
    
    # 計算空間相關矩陣
    correlation_matrix = processor.get_spatial_correlation_matrix()
    print(f"\n空間相關範圍: {correlation_matrix.min():.3f} - {correlation_matrix.max():.3f}")