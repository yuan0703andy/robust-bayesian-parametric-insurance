"""
Input Adapters for Unified Product Design Engine
統一產品設計引擎的輸入適配器

This module provides adapters to integrate different input sources:
- CLIMADAInputAdapter: Traditional CLIMADA hazard objects
- BayesianInputAdapter: Bayesian simulation results with probabilistic distributions

本模組提供適配器以整合不同的輸入來源：
- CLIMADA適配器：傳統的CLIMADA災害物件
- 貝氏適配器：帶有機率分布的貝氏模擬結果
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple, Optional
from dataclasses import dataclass
import warnings

# Import CLIMADA if available
try:
    from climada.engine import ImpactCalc
    from climada.entity import ImpactFuncSet
    CLIMADA_AVAILABLE = True
except ImportError:
    CLIMADA_AVAILABLE = False
    warnings.warn("CLIMADA not available - CLIMADAInputAdapter will have limited functionality")

# Import spatial analysis tools
from scipy.spatial import cKDTree
from scipy import stats


@dataclass
class EventMetadata:
    """事件元數據結構"""
    event_id: str
    event_name: Optional[str] = None
    year: Optional[int] = None
    category: Optional[str] = None
    max_wind_speed: Optional[float] = None
    central_pressure: Optional[float] = None
    affected_exposure: Optional[float] = None


class InputAdapter(ABC):
    """
    統一輸入適配器基類
    
    所有輸入適配器都必須實現這些方法，以提供統一的介面給產品設計引擎
    """
    
    @abstractmethod
    def extract_parametric_indices(self) -> Dict[str, np.ndarray]:
        """
        提取參數指標
        
        Returns:
        --------
        Dict[str, np.ndarray]
            參數指標字典，格式為 {index_type: values}
            例如：{'cat_in_circle_30km_max': array, 'cat_in_circle_50km_mean': array}
        """
        pass
    
    @abstractmethod
    def get_loss_data(self) -> np.ndarray:
        """
        獲取損失數據
        
        Returns:
        --------
        np.ndarray
            損失數據數組，每個元素對應一個事件的損失
        """
        pass
    
    @abstractmethod
    def get_event_metadata(self) -> pd.DataFrame:
        """
        獲取事件元數據
        
        Returns:
        --------
        pd.DataFrame
            事件元數據表，包含事件ID、名稱、年份等資訊
        """
        pass
    
    @abstractmethod
    def get_input_type(self) -> str:
        """
        獲取輸入類型標識
        
        Returns:
        --------
        str
            輸入類型 ('traditional' 或 'probabilistic')
        """
        pass


class CLIMADAInputAdapter(InputAdapter):
    """
    CLIMADA 物件適配器
    
    將 CLIMADA 的 hazard, exposure, impact_func_set 轉換為統一格式
    """
    
    def __init__(self, tc_hazard, exposure_main, impact_func_set, 
                 cat_in_circle_radii: list = [15, 30, 50],
                 statistics: list = ['max', 'mean', '95th']):
        """
        初始化 CLIMADA 適配器
        
        Parameters:
        -----------
        tc_hazard : climada.hazard.TropCyclone
            颱風災害物件
        exposure_main : climada.entity.Exposures
            曝險物件
        impact_func_set : climada.entity.ImpactFuncSet
            影響函數集
        cat_in_circle_radii : list
            Cat-in-a-Circle 分析半徑 (km)
        statistics : list
            要計算的統計指標
        """
        self.tc_hazard = tc_hazard
        self.exposure_main = exposure_main
        self.impact_func_set = impact_func_set
        self.cat_in_circle_radii = cat_in_circle_radii
        self.statistics = statistics
        
        # 緩存計算結果
        self._cached_indices = None
        self._cached_losses = None
        self._cached_metadata = None
        
        # 驗證輸入
        self._validate_inputs()
    
    def _validate_inputs(self):
        """驗證輸入的有效性"""
        if not CLIMADA_AVAILABLE:
            raise ImportError("CLIMADA is required for CLIMADAInputAdapter but not available")
        
        # 基本驗證
        if self.tc_hazard is None:
            raise ValueError("tc_hazard cannot be None")
        if self.exposure_main is None:
            raise ValueError("exposure_main cannot be None")
        if self.impact_func_set is None:
            raise ValueError("impact_func_set cannot be None")
    
    def extract_parametric_indices(self) -> Dict[str, np.ndarray]:
        """
        從 CLIMADA 物件提取 Cat-in-a-Circle 參數指標
        
        根據 Steinmann et al. (2023)，Cat-in-a-Circle 是每個保險標的物（曝險點）
        周圍一定半徑圓圈內的最大持續風速
        """
        if self._cached_indices is not None:
            return self._cached_indices
        
        print("🔄 從 CLIMADA 物件提取 Cat-in-a-Circle 參數指標...")
        
        # 獲取曝險點座標 - 支援多種欄位名稱格式
        exposure_gdf = self.exposure_main.gdf
        
        # 嘗試不同的座標欄位名稱
        if 'latitude' in exposure_gdf.columns and 'longitude' in exposure_gdf.columns:
            lat_col, lon_col = 'latitude', 'longitude'
        elif 'lat' in exposure_gdf.columns and 'lon' in exposure_gdf.columns:
            lat_col, lon_col = 'lat', 'lon'
        elif hasattr(self.exposure_main, 'coord') and self.exposure_main.coord.shape[1] >= 2:
            # 如果有 coord 屬性，直接使用
            exposure_coords = self.exposure_main.coord
        elif 'geometry' in exposure_gdf.columns:
            # 從 geometry 欄位提取座標
            exposure_coords = np.column_stack([
                exposure_gdf.geometry.y.values,  # latitude
                exposure_gdf.geometry.x.values   # longitude
            ])
        else:
            raise ValueError("Cannot find coordinate columns in exposure data. Expected 'latitude'/'longitude', 'lat'/'lon', or 'geometry' column")
        
        # 如果找到了命名欄位，提取座標
        if 'lat_col' in locals():
            exposure_coords = np.column_stack([
                exposure_gdf[lat_col].values,
                exposure_gdf[lon_col].values
            ])
        
        # 獲取災害網格點座標
        hazard_coords = np.column_stack([
            self.tc_hazard.centroids.lat,
            self.tc_hazard.centroids.lon
        ])
        
        # 建立空間索引
        from scipy.spatial import cKDTree
        hazard_tree = cKDTree(np.radians(hazard_coords))
        
        indices_dict = {}
        
        # 對每個半徑和統計指標組合提取 Cat-in-a-Circle 指標
        for radius_km in self.cat_in_circle_radii:
            for stat in self.statistics:
                index_name = f"cat_in_circle_{radius_km}km_{stat}"
                
                # 為每個事件計算指標
                event_indices = []
                
                # 獲取事件數量
                if hasattr(self.tc_hazard, 'size'):
                    if hasattr(self.tc_hazard.size, '__getitem__'):
                        n_events = self.tc_hazard.size[0]
                    else:
                        n_events = self.tc_hazard.intensity.shape[0]
                else:
                    n_events = self.tc_hazard.intensity.shape[0]
                
                for event_idx in range(n_events):
                    # 獲取該事件的風速場
                    wind_field = self.tc_hazard.intensity[event_idx, :].toarray().flatten()
                    
                    # 計算每個曝險點的 Cat-in-a-Circle 指標
                    exposure_indices = []
                    
                    for exp_coord in exposure_coords:
                        # 找到半徑範圍內的災害點 (Cat-in-a-Circle)
                        radius_rad = radius_km / 6371.0  # 轉換為弧度 (地球半徑約6371km)
                        nearby_indices = hazard_tree.query_ball_point(
                            np.radians(exp_coord), radius_rad
                        )
                        
                        if len(nearby_indices) > 0:
                            nearby_winds = wind_field[nearby_indices]
                            nearby_winds = nearby_winds[nearby_winds > 0]  # 排除零值
                            
                            if len(nearby_winds) > 0:
                                if stat == 'max':
                                    exposure_indices.append(np.max(nearby_winds))
                                elif stat == 'mean':
                                    exposure_indices.append(np.mean(nearby_winds))
                                elif stat == '95th':
                                    exposure_indices.append(np.percentile(nearby_winds, 95))
                                else:
                                    exposure_indices.append(np.mean(nearby_winds))
                            else:
                                exposure_indices.append(0.0)
                        else:
                            exposure_indices.append(0.0)
                    
                    # 使用純Cat-in-a-Circle最大風速（符合Steinmann論文定義）
                    if len(exposure_indices) > 0:
                        if stat == 'max':
                            # 對於最大值統計，取所有曝險點的最大值作為該事件的Cat-in-a-Circle指標
                            event_max_index = np.max(exposure_indices)
                        else:
                            # 對於其他統計（mean, 95th），使用簡單平均
                            event_max_index = np.mean(exposure_indices)
                        event_indices.append(event_max_index)
                    else:
                        event_indices.append(0.0)
                
                indices_dict[index_name] = np.array(event_indices)
        
        self._cached_indices = indices_dict
        print(f"✅ 提取了 {len(indices_dict)} 個 Cat-in-a-Circle 參數指標")
        print(f"   半徑範圍: {self.cat_in_circle_radii} km")
        print(f"   統計指標: {self.statistics}")
        print(f"   曝險點數量: {len(exposure_coords)}")
        print(f"   📍 使用純Cat-in-a-Circle最大風速（無加權平均）")
        
        return indices_dict
    
    def get_loss_data(self) -> np.ndarray:
        """
        計算確定性損失數據
        """
        if self._cached_losses is not None:
            return self._cached_losses
        
        print("🔄 計算 CLIMADA 影響損失...")
        
        # 使用 CLIMADA 計算影響 - 根據你的範例使用正確的 API
        impact_calc = ImpactCalc(self.exposure_main, self.impact_func_set, self.tc_hazard)
        impact = impact_calc.impact(save_mat=False)
        
        # 提取事件損失
        losses = impact.at_event
        self._cached_losses = losses
        
        print(f"✅ 計算了 {len(losses)} 個事件的損失")
        print(f"   總損失: ${np.sum(losses)/1e9:.2f}B")
        print(f"   最大事件損失: ${np.max(losses)/1e9:.2f}B")
        
        return losses
    
    def get_event_metadata(self) -> pd.DataFrame:
        """
        從 CLIMADA 物件提取事件元數據
        """
        if self._cached_metadata is not None:
            return self._cached_metadata
        
        # 獲取事件數量 - 支援不同的 size 屬性格式
        if hasattr(self.tc_hazard, 'size'):
            if hasattr(self.tc_hazard.size, '__getitem__'):
                # size 是 tuple 或 array
                n_events = self.tc_hazard.size[0]
            else:
                # size 是單一整數，通常表示總元素數
                # 對於 CLIMADA hazard，事件數是 intensity 矩陣的第一維
                n_events = self.tc_hazard.intensity.shape[0]
        else:
            # 直接從 intensity 矩陣獲取事件數
            n_events = self.tc_hazard.intensity.shape[0]
        
        metadata_list = []
        for i in range(n_events):
            # 嘗試從 hazard 中提取事件資訊
            event_id = f"event_{i:04d}"
            event_name = None
            year = None
            category = None
            max_wind = None
            
            # 如果 hazard 有事件名稱資訊
            if hasattr(self.tc_hazard, 'event_name') and len(self.tc_hazard.event_name) > i:
                event_name = self.tc_hazard.event_name[i]
            
            # 如果有年份資訊
            if hasattr(self.tc_hazard, 'date') and len(self.tc_hazard.date) > i:
                year = pd.to_datetime(self.tc_hazard.date[i], unit='s').year
            
            # 計算最大風速
            wind_field = self.tc_hazard.intensity[i, :].toarray().flatten()
            if len(wind_field) > 0:
                max_wind = np.max(wind_field)
                
                # 根據風速估算颶風等級
                if max_wind >= 70:  # 137+ kt (Cat 5)
                    category = "Cat_5"
                elif max_wind >= 58:  # 113-136 kt (Cat 4)
                    category = "Cat_4"
                elif max_wind >= 50:  # 96-112 kt (Cat 3)
                    category = "Cat_3"
                elif max_wind >= 43:  # 83-95 kt (Cat 2)
                    category = "Cat_2"
                elif max_wind >= 33:  # 64-82 kt (Cat 1)
                    category = "Cat_1"
                else:
                    category = "Tropical Storm"
            
            metadata_list.append(EventMetadata(
                event_id=event_id,
                event_name=event_name,
                year=year,
                category=category,
                max_wind_speed=max_wind,
                affected_exposure=np.sum(self.exposure_main.gdf['value'].values)
            ))
        
        # 轉換為 DataFrame
        metadata_df = pd.DataFrame([
            {
                'event_id': meta.event_id,
                'event_name': meta.event_name,
                'year': meta.year,
                'category': meta.category,
                'max_wind_speed': meta.max_wind_speed,
                'affected_exposure': meta.affected_exposure
            }
            for meta in metadata_list
        ])
        
        self._cached_metadata = metadata_df
        return metadata_df
    
    def get_input_type(self) -> str:
        """返回輸入類型"""
        return "traditional"


class BayesianInputAdapter(InputAdapter):
    """
    貝氏模擬結果適配器
    
    將貝氏分析的機率性結果轉換為統一格式
    """
    
    def __init__(self, bayesian_results: Dict[str, Any]):
        """
        初始化貝氏適配器
        
        Parameters:
        -----------
        bayesian_results : Dict[str, Any]
            貝氏分析結果，應包含：
            - uncertainty_quantification: 不確定性量化結果
            - event_loss_distributions: 事件損失分布
            - parametric_indices: 參數指標 (可選)
        """
        self.bayesian_results = bayesian_results
        
        # 緩存計算結果
        self._cached_indices = None
        self._cached_losses = None
        self._cached_metadata = None
        
        # 驗證輸入
        self._validate_bayesian_results()
    
    def _validate_bayesian_results(self):
        """驗證貝氏結果的有效性"""
        required_keys = ['uncertainty_quantification']
        
        for key in required_keys:
            if key not in self.bayesian_results:
                raise ValueError(f"Bayesian results missing required key: {key}")
        
        # 檢查不確定性量化結果
        uncertainty_data = self.bayesian_results['uncertainty_quantification']
        if 'event_loss_distributions' not in uncertainty_data:
            raise ValueError("uncertainty_quantification missing event_loss_distributions")
    
    def extract_parametric_indices(self) -> Dict[str, np.ndarray]:
        """
        從貝氏結果提取參數指標
        """
        if self._cached_indices is not None:
            return self._cached_indices
        
        print("🔄 從貝氏結果提取參數指標...")
        
        # 檢查是否有預計算的參數指標
        if 'parametric_indices' in self.bayesian_results:
            indices = self.bayesian_results['parametric_indices']
            self._cached_indices = indices
            print(f"✅ 使用預計算的 {len(indices)} 個參數指標")
            return indices
        
        # 否則從損失分布中計算
        uncertainty_data = self.bayesian_results['uncertainty_quantification']
        event_distributions = uncertainty_data['event_loss_distributions']
        
        indices_dict = {}
        
        # 基於損失分布特徵作為參數指標
        indices_dict['loss_mean'] = np.array([
            event_data['mean'] for event_data in event_distributions.values()
        ])
        
        indices_dict['loss_std'] = np.array([
            event_data['std'] for event_data in event_distributions.values()
        ])
        
        indices_dict['loss_95th'] = np.array([
            np.percentile(event_data['samples'], 95) 
            for event_data in event_distributions.values()
        ])
        
        # 如果有風速資訊，使用風速作為參數指標
        if 'hazard_intensities' in uncertainty_data:
            hazard_intensities = uncertainty_data['hazard_intensities']
            indices_dict['wind_speed_max'] = np.array([
                np.max(intensity) if hasattr(intensity, '__iter__') else intensity
                for intensity in hazard_intensities
            ])
        
        self._cached_indices = indices_dict
        print(f"✅ 計算了 {len(indices_dict)} 個參數指標類型")
        
        return indices_dict
    
    def get_loss_data(self) -> np.ndarray:
        """
        提取機率性損失的期望值作為代表性損失
        """
        if self._cached_losses is not None:
            return self._cached_losses
        
        print("🔄 從貝氏分布提取代表性損失...")
        
        uncertainty_data = self.bayesian_results['uncertainty_quantification']
        event_distributions = uncertainty_data['event_loss_distributions']
        
        # 使用分布的期望值作為代表性損失
        losses = np.array([
            event_data['mean'] for event_data in event_distributions.values()
        ])
        
        self._cached_losses = losses
        
        print(f"✅ 提取了 {len(losses)} 個事件的代表性損失")
        print(f"   總期望損失: ${np.sum(losses)/1e9:.2f}B")
        print(f"   最大期望損失: ${np.max(losses)/1e9:.2f}B")
        
        return losses
    
    def get_probabilistic_distributions(self) -> Dict[str, Any]:
        """
        貝氏專用：獲取完整的機率分布
        
        Returns:
        --------
        Dict[str, Any]
            完整的不確定性量化結果
        """
        return self.bayesian_results['uncertainty_quantification']
    
    def get_event_metadata(self) -> pd.DataFrame:
        """
        從貝氏結果提取事件元數據
        """
        if self._cached_metadata is not None:
            return self._cached_metadata
        
        uncertainty_data = self.bayesian_results['uncertainty_quantification']
        event_distributions = uncertainty_data['event_loss_distributions']
        
        metadata_list = []
        
        for i, (event_id, event_data) in enumerate(event_distributions.items()):
            # 基本元數據
            event_name = event_data.get('event_name', f"bayesian_event_{i:04d}")
            year = event_data.get('year')
            
            # 從損失分布推斷颶風強度
            loss_mean = event_data['mean']
            loss_std = event_data['std']
            
            # 簡化的颶風等級推斷 (基於損失大小)
            if loss_mean > 5e9:  # > $5B
                category = "Cat_4_5"
            elif loss_mean > 1e9:  # > $1B
                category = "Cat_3"
            elif loss_mean > 2e8:  # > $200M
                category = "Cat_2"
            elif loss_mean > 5e7:  # > $50M
                category = "Cat_1"
            else:
                category = "Tropical Storm"
            
            # 估算風速 (如果有的話)
            max_wind = None
            if 'max_wind_speed' in event_data:
                max_wind = event_data['max_wind_speed']
            elif 'hazard_intensity' in event_data:
                max_wind = event_data['hazard_intensity']
            
            metadata_list.append({
                'event_id': event_id,
                'event_name': event_name,
                'year': year,
                'category': category,
                'max_wind_speed': max_wind,
                'loss_mean': loss_mean,
                'loss_std': loss_std,
                'uncertainty_ratio': loss_std / loss_mean if loss_mean > 0 else 0
            })
        
        metadata_df = pd.DataFrame(metadata_list)
        self._cached_metadata = metadata_df
        
        return metadata_df
    
    def get_input_type(self) -> str:
        """返回輸入類型"""
        return "probabilistic"


class HybridInputAdapter(InputAdapter):
    """
    混合輸入適配器
    
    結合 CLIMADA 和貝氏結果的混合適配器，用於對比分析
    """
    
    def __init__(self, climada_adapter: CLIMADAInputAdapter, 
                 bayesian_adapter: BayesianInputAdapter,
                 blending_weight: float = 0.5):
        """
        初始化混合適配器
        
        Parameters:
        -----------
        climada_adapter : CLIMADAInputAdapter
            CLIMADA 適配器
        bayesian_adapter : BayesianInputAdapter
            貝氏適配器
        blending_weight : float
            混合權重 (0=純CLIMADA, 1=純貝氏)
        """
        self.climada_adapter = climada_adapter
        self.bayesian_adapter = bayesian_adapter
        self.blending_weight = blending_weight
        
        # 驗證事件數量匹配
        climada_losses = climada_adapter.get_loss_data()
        bayesian_losses = bayesian_adapter.get_loss_data()
        
        if len(climada_losses) != len(bayesian_losses):
            raise ValueError(
                f"Event count mismatch: CLIMADA={len(climada_losses)}, "
                f"Bayesian={len(bayesian_losses)}"
            )
    
    def extract_parametric_indices(self) -> Dict[str, np.ndarray]:
        """混合參數指標提取"""
        climada_indices = self.climada_adapter.extract_parametric_indices()
        bayesian_indices = self.bayesian_adapter.extract_parametric_indices()
        
        # 結合兩種指標
        hybrid_indices = {}
        
        # 添加 CLIMADA 指標 (帶前綴)
        for key, values in climada_indices.items():
            hybrid_indices[f"climada_{key}"] = values
        
        # 添加貝氏指標 (帶前綴)
        for key, values in bayesian_indices.items():
            hybrid_indices[f"bayesian_{key}"] = values
        
        return hybrid_indices
    
    def get_loss_data(self) -> np.ndarray:
        """混合損失數據"""
        climada_losses = self.climada_adapter.get_loss_data()
        bayesian_losses = self.bayesian_adapter.get_loss_data()
        
        # 加權混合
        hybrid_losses = (
            (1 - self.blending_weight) * climada_losses + 
            self.blending_weight * bayesian_losses
        )
        
        return hybrid_losses
    
    def get_event_metadata(self) -> pd.DataFrame:
        """混合事件元數據"""
        climada_meta = self.climada_adapter.get_event_metadata()
        bayesian_meta = self.bayesian_adapter.get_event_metadata()
        
        # 合併元數據
        hybrid_meta = climada_meta.copy()
        
        # 添加貝氏相關列
        if 'loss_mean' in bayesian_meta.columns:
            hybrid_meta['bayesian_loss_mean'] = bayesian_meta['loss_mean']
        if 'loss_std' in bayesian_meta.columns:
            hybrid_meta['bayesian_loss_std'] = bayesian_meta['loss_std']
        if 'uncertainty_ratio' in bayesian_meta.columns:
            hybrid_meta['uncertainty_ratio'] = bayesian_meta['uncertainty_ratio']
        
        hybrid_meta['adapter_type'] = 'hybrid'
        hybrid_meta['blending_weight'] = self.blending_weight
        
        return hybrid_meta
    
    def get_input_type(self) -> str:
        """返回輸入類型"""
        return "hybrid"


def create_adapter_from_data(data_source: Union[tuple, dict], adapter_type: str = "auto") -> InputAdapter:
    """
    工廠函數：根據數據來源自動創建適當的適配器
    
    Parameters:
    -----------
    data_source : Union[tuple, dict]
        數據來源，可以是：
        - tuple: (tc_hazard, exposure_main, impact_func_set) 用於 CLIMADA
        - dict: bayesian_results 用於貝氏
    adapter_type : str
        適配器類型 ("auto", "climada", "bayesian", "hybrid")
        
    Returns:
    --------
    InputAdapter
        適當的輸入適配器實例
    """
    
    if adapter_type == "auto":
        # 自動檢測數據類型
        if isinstance(data_source, tuple) and len(data_source) == 3:
            adapter_type = "climada"
        elif isinstance(data_source, dict):
            adapter_type = "bayesian"
        else:
            raise ValueError("Cannot auto-detect adapter type from data_source")
    
    if adapter_type == "climada":
        if not isinstance(data_source, tuple) or len(data_source) != 3:
            raise ValueError("CLIMADA adapter requires tuple of (tc_hazard, exposure_main, impact_func_set)")
        tc_hazard, exposure_main, impact_func_set = data_source
        return CLIMADAInputAdapter(tc_hazard, exposure_main, impact_func_set)
    
    elif adapter_type == "bayesian":
        if not isinstance(data_source, dict):
            raise ValueError("Bayesian adapter requires dict of bayesian_results")
        return BayesianInputAdapter(data_source)
    
    else:
        raise ValueError(f"Unsupported adapter_type: {adapter_type}")


# 便利函數
def validate_adapter_compatibility(adapter1: InputAdapter, adapter2: InputAdapter) -> bool:
    """
    驗證兩個適配器的兼容性（事件數量等）
    
    Parameters:
    -----------
    adapter1, adapter2 : InputAdapter
        要檢查的適配器
        
    Returns:
    --------
    bool
        是否兼容
    """
    try:
        losses1 = adapter1.get_loss_data()
        losses2 = adapter2.get_loss_data()
        
        return len(losses1) == len(losses2)
    except Exception:
        return False


def extract_pure_cat_in_circle(tc_hazard, exposure_coords, hazard_tree, radius_km):
    """提取純 Cat-in-a-Circle 最大風速（無加權平均）"""
    n_events = tc_hazard.intensity.shape[0]
    max_winds = np.zeros(n_events)
    
    for event_idx in range(n_events):
        wind_field = tc_hazard.intensity[event_idx, :].toarray().flatten()
        event_max_wind = 0.0
        
        for exp_coord in exposure_coords:
            # Cat-in-a-Circle: 半徑內的最大風速
            radius_rad = radius_km / 6371.0
            nearby_indices = hazard_tree.query_ball_point(
                np.radians(exp_coord), radius_rad
            )
            
            if len(nearby_indices) > 0:
                nearby_winds = wind_field[nearby_indices]
                nearby_winds = nearby_winds[nearby_winds > 0]
                
                if len(nearby_winds) > 0:
                    local_max = np.max(nearby_winds)
                    event_max_wind = max(event_max_wind, local_max)
        
        max_winds[event_idx] = event_max_wind
    
    return max_winds


# Bayesian integration helper functions

def create_vulnerability_data_from_climada_adapter(climada_adapter: CLIMADAInputAdapter):
    """
    從 CLIMADAInputAdapter 創建 VulnerabilityData 物件
    
    這個函數將 CLIMADA 的災害、暴險和損失數據轉換為 @bayesian/ 模組
    所需的 VulnerabilityData 格式，支援脆弱度關係建模。
    
    Parameters:
    -----------
    climada_adapter : CLIMADAInputAdapter
        已初始化的 CLIMADA 輸入適配器
        
    Returns:
    --------
    VulnerabilityData
        格式化的脆弱度數據物件，包含：
        - hazard_intensities: 災害強度數據 (H_ij)
        - exposure_values: 暴險值數據 (E_i) 
        - observed_losses: 觀測損失數據 (L_ij)
    """
    try:
        # Import VulnerabilityData from bayesian module
        from bayesian.parametric_bayesian_hierarchy import VulnerabilityData
    except ImportError:
        raise ImportError("無法導入 VulnerabilityData，請檢查 @bayesian/ 模組")
    
    print("🔄 從 CLIMADA 適配器提取脆弱度數據...")
    
    # 1. 提取災害強度 (H_ij) - 使用 Cat-in-a-Circle 參數指標
    parametric_indices = climada_adapter.extract_parametric_indices()
    
    # 選擇最大風速作為主要災害強度指標
    # 優先選擇 'max' 統計的 Cat-in-a-Circle 指標
    hazard_intensities = None
    for key, values in parametric_indices.items():
        if 'max' in key.lower():
            hazard_intensities = values
            print(f"   使用災害指標: {key}")
            break
    
    # 如果沒有找到 'max' 指標，使用第一個可用的指標
    if hazard_intensities is None and parametric_indices:
        key, hazard_intensities = next(iter(parametric_indices.items()))
        print(f"   回退使用災害指標: {key}")
    
    if hazard_intensities is None:
        raise ValueError("無法從 CLIMADA 適配器中提取災害強度數據")
    
    # 2. 提取暴險值 (E_i)
    try:
        exposure_gdf = climada_adapter.exposure_main.gdf
        if 'value' in exposure_gdf.columns:
            # 取暴險總值作為代表性暴險值
            total_exposure = np.sum(exposure_gdf['value'].values)
            # 為每個事件重複相同的暴險值
            exposure_values = np.full(len(hazard_intensities), total_exposure)
            print(f"   暴險總值: ${total_exposure/1e9:.2f}B")
        else:
            # 使用單位暴險值
            exposure_values = np.ones(len(hazard_intensities))
            print("   使用單位暴險值")
    except Exception as e:
        print(f"   ⚠️ 暴險值提取失敗: {e}")
        exposure_values = np.ones(len(hazard_intensities))
    
    # 3. 提取觀測損失 (L_ij)
    observed_losses = climada_adapter.get_loss_data()
    
    # 確保數組長度一致
    min_length = min(len(hazard_intensities), len(exposure_values), len(observed_losses))
    
    if min_length == 0:
        raise ValueError("數據數組為空，無法創建 VulnerabilityData")
    
    hazard_intensities = hazard_intensities[:min_length]
    exposure_values = exposure_values[:min_length]  
    observed_losses = observed_losses[:min_length]
    
    print(f"   ✅ 數據對齊完成: {min_length} 個事件")
    print(f"   災害強度範圍: [{np.min(hazard_intensities):.1f}, {np.max(hazard_intensities):.1f}] m/s")
    print(f"   損失範圍: [${np.min(observed_losses)/1e6:.1f}M, ${np.max(observed_losses)/1e6:.1f}M]")
    
    # 創建 VulnerabilityData 物件
    vulnerability_data = VulnerabilityData(
        hazard_intensities=hazard_intensities,
        exposure_values=exposure_values,
        observed_losses=observed_losses
    )
    
    print("✅ VulnerabilityData 創建完成")
    
    return vulnerability_data


def compare_adapters_summary(adapters: Dict[str, InputAdapter]) -> pd.DataFrame:
    """
    比較多個適配器的摘要統計
    
    Parameters:
    -----------
    adapters : Dict[str, InputAdapter]
        適配器字典 {name: adapter}
        
    Returns:
    --------
    pd.DataFrame
        比較摘要表
    """
    summary_data = []
    
    for name, adapter in adapters.items():
        try:
            losses = adapter.get_loss_data()
            indices = adapter.extract_parametric_indices()
            metadata = adapter.get_event_metadata()
            
            summary_data.append({
                'adapter_name': name,
                'input_type': adapter.get_input_type(),
                'n_events': len(losses),
                'total_loss_billion': np.sum(losses) / 1e9,
                'max_loss_billion': np.max(losses) / 1e9,
                'mean_loss_million': np.mean(losses) / 1e6,
                'n_parametric_indices': len(indices),
                'parametric_index_types': list(indices.keys()),
                'data_availability': 'Complete'
            })
        except Exception as e:
            summary_data.append({
                'adapter_name': name,
                'input_type': 'ERROR',
                'error': str(e),
                'data_availability': 'Failed'
            })
    
    return pd.DataFrame(summary_data)