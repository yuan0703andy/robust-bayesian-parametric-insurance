"""
OSM Hospital Extraction for Cat-in-a-Circle Analysis
從OSM數據提取醫院位置，實現Steinmann論文的標準化單位方法
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Try to import geopandas, but provide fallback if not available
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    warnings.warn("geopandas not available - will use simplified data structures")

# OSM-flex imports with error handling
try:
    import osm_flex
    import osm_flex.download
    import osm_flex.extract
    import osm_flex.clip as cp
    OSM_FLEX_AVAILABLE = True
except ImportError:
    OSM_FLEX_AVAILABLE = False
    warnings.warn("osm_flex not available - hospital extraction will use mock data")

# Try CLIMADA-petals OSM module as alternative
try:
    from climada_petals.entity.exposures.osm_dataloader import OSMApiQuery
    CLIMADA_OSM_AVAILABLE = True
except ImportError:
    CLIMADA_OSM_AVAILABLE = False

# CLIMADA imports
try:
    from climada.entity import Exposures
    import climada.util.lines_polys_handler as u_lp
    CLIMADA_AVAILABLE = True
except ImportError:
    CLIMADA_AVAILABLE = False
    warnings.warn("CLIMADA not available - using simplified hospital processing")


def extract_nc_hospitals_from_osm(osm_file_path: str = None, use_mock: bool = False, use_overpass_api: bool = False):
    """
    從OSM數據提取北卡羅來納州醫院
    
    Parameters:
    -----------
    osm_file_path : str, optional
        OSM PBF文件路徑，如果未提供將使用預設路徑
    use_mock : bool
        是否使用模擬醫院數據（用於測試）
    use_overpass_api : bool
        是否使用Overpass API直接下載（需要網絡連接）
        
    Returns:
    --------
    geopandas.GeoDataFrame
        醫院位置數據
    """
    
    # 檢查是否應該使用模擬數據
    if use_mock:
        print("📍 使用模擬醫院數據進行測試...")
        return create_mock_hospitals()
    
    # 嘗試使用Overpass API（如果可用且被要求）
    if use_overpass_api and CLIMADA_OSM_AVAILABLE and GEOPANDAS_AVAILABLE:
        print("🌐 嘗試使用Overpass API下載醫院數據...")
        try:
            return extract_hospitals_via_overpass()
        except Exception as e:
            print(f"   ⚠️ Overpass API失敗: {e}")
            print("   🔄 回退到其他方法...")
    
    # 檢查依賴是否可用
    if not OSM_FLEX_AVAILABLE or not GEOPANDAS_AVAILABLE:
        if not OSM_FLEX_AVAILABLE:
            print("   ⚠️ osm_flex 不可用")
        if not GEOPANDAS_AVAILABLE:
            print("   ⚠️ geopandas 不可用")
        print("📍 使用模擬醫院數據進行測試...")
        return create_mock_hospitals()
    
    # 使用實際OSM數據
    print("🏥 從OSM數據提取醫院...")
    
    if osm_file_path is None:
        osm_file_path = '/Users/andyhou/osm/osm_bpf/nc.osm.pbf'
    
    try:
        # 檢查文件是否存在
        osm_path = Path(osm_file_path)
        if not osm_path.exists():
            print(f"   ⚠️ OSM文件不存在: {osm_file_path}")
            # 檢查可能的替代路徑
            alternative_paths = [
                '/Users/andyhou/osm/nc.osm.pbf',
                './osm_data/nc.osm.pbf',
                './data/nc.osm.pbf',
                'nc.osm.pbf'
            ]
            
            for alt_path in alternative_paths:
                if Path(alt_path).exists():
                    print(f"   🔍 找到替代OSM文件: {alt_path}")
                    osm_file_path = alt_path
                    break
            else:
                print("   🔄 未找到OSM文件，使用模擬數據替代...")
                return create_mock_hospitals()
        
        print(f"   📂 讀取OSM文件: {osm_file_path}")
        
        # 定義要保留的屬性
        keys_to_keep = ['amenity', 'name', 'addr:city', 'healthcare', 'emergency']
        
        # 定義醫院篩選條件
        filter_query = "amenity='hospital'"
        
        # 提取點位型醫院
        print("   🔍 提取點位型醫院...")
        gdf_hospitals_points = None
        
        # 嘗試使用osm_flex.extract的不同方法
        try:
            # 方法1: 使用extract_cis (infrastructure extraction)
            if hasattr(osm_flex.extract, 'extract_cis'):
                print("      嘗試使用extract_cis方法...")
                gdf_hospitals_points = osm_flex.extract.extract_cis(osm_file_path, 'health')
                if gdf_hospitals_points is not None and len(gdf_hospitals_points) > 0:
                    # 過濾只保留醫院
                    gdf_hospitals_points = gdf_hospitals_points[
                        gdf_hospitals_points.get('amenity', '') == 'hospital'
                    ] if 'amenity' in gdf_hospitals_points.columns else gdf_hospitals_points
        except Exception as e:
            print(f"      extract_cis失敗: {e}")
        
        # 如果方法1失敗，嘗試原始方法
        if gdf_hospitals_points is None or len(gdf_hospitals_points) == 0:
            try:
                gdf_hospitals_points = osm_flex.extract.extract(
                    osm_file_path,
                    'points',
                    keys_to_keep,
                    filter_query
                )
            except (TypeError, AttributeError, StopIteration) as e:
                print(f"      ⚠️ 點位提取失敗: {e}")
        
        # 確保有有效的DataFrame
        if gdf_hospitals_points is None:
            if GEOPANDAS_AVAILABLE:
                gdf_hospitals_points = gpd.GeoDataFrame(columns=['amenity', 'name'], crs='EPSG:4326')
            else:
                gdf_hospitals_points = pd.DataFrame(columns=['amenity', 'name', 'geometry'])
        
        print(f"      找到 {len(gdf_hospitals_points)} 個點位型醫院")
        
        # 提取多邊形型醫院
        print("   🔍 提取多邊形型醫院...")
        gdf_hospitals_polygons = None
        
        try:
            gdf_hospitals_polygons = osm_flex.extract.extract(
                osm_file_path,
                'multipolygons',
                keys_to_keep,
                filter_query
            )
        except (TypeError, AttributeError, StopIteration) as e:
            print(f"      ⚠️ 多邊形提取失敗: {e}")
        
        # 確保有有效的DataFrame
        if gdf_hospitals_polygons is None:
            if GEOPANDAS_AVAILABLE:
                gdf_hospitals_polygons = gpd.GeoDataFrame(columns=['amenity', 'name'], crs='EPSG:4326')
            else:
                gdf_hospitals_polygons = pd.DataFrame(columns=['amenity', 'name', 'geometry'])
            
        print(f"      找到 {len(gdf_hospitals_polygons)} 個多邊形型醫院")
        
        # 合併結果
        gdf_hospitals = pd.concat(
            [gdf_hospitals_points, gdf_hospitals_polygons], 
            ignore_index=True
        )
        
        if len(gdf_hospitals) == 0:
            print("   ⚠️ 未找到醫院數據，使用模擬數據...")
            return create_mock_hospitals()
        
        print(f"   ✅ 總計提取 {len(gdf_hospitals)} 家醫院")
        
        # 處理多邊形醫院，轉換為中心點
        gdf_hospitals = process_hospital_geometries(gdf_hospitals)
        
        return gdf_hospitals
        
    except Exception as e:
        print(f"   ❌ OSM提取失敗: {e}")
        print("   🔄 使用模擬數據...")
        return create_mock_hospitals()


def extract_hospitals_via_overpass():
    """
    使用Overpass API直接下載北卡羅來納州醫院數據
    
    Returns:
    --------
    geopandas.GeoDataFrame
        醫院位置數據
    """
    print("   📍 定義北卡羅來納州邊界...")
    # North Carolina bounding box (approximate)
    nc_bbox = (33.8, -84.5, 36.6, -75.5)  # (lat_min, lon_min, lat_max, lon_max)
    
    # Define hospital query conditions
    condition_hospital = '["amenity"="hospital"]'
    
    print("   🔍 查詢醫院數據...")
    try:
        from climada_petals.entity.exposures.osm_dataloader import OSMApiQuery
        
        # Create query
        hospital_query = OSMApiQuery.from_bounding_box(nc_bbox, condition_hospital)
        
        # Get data from Overpass API
        gdf_hospitals = hospital_query.get_data_overpass()
        
        if gdf_hospitals is None or len(gdf_hospitals) == 0:
            print("   ⚠️ 未找到醫院數據")
            return create_mock_hospitals()
            
        print(f"   ✅ 成功下載 {len(gdf_hospitals)} 家醫院")
        return gdf_hospitals
        
    except Exception as e:
        print(f"   ❌ Overpass API提取失敗: {e}")
        return create_mock_hospitals()


def create_mock_hospitals():
    """
    創建模擬醫院數據用於測試和演示
    基於北卡羅來納州主要城市的典型醫院分布
    """
    
    print("   🏗️ 創建模擬醫院數據...")
    
    # 北卡羅來納州主要城市的醫院
    mock_hospitals_data = [
        # 羅利-達勒姆地區 (Research Triangle)
        {"name": "Duke University Hospital", "lat": 36.0153, "lon": -78.9384, "city": "Durham"},
        {"name": "UNC Hospitals", "lat": 35.9049, "lon": -79.0469, "city": "Chapel Hill"},
        {"name": "Rex Hospital", "lat": 35.8043, "lon": -78.6569, "city": "Raleigh"},
        {"name": "WakeMed Raleigh Campus", "lat": 35.7520, "lon": -78.6037, "city": "Raleigh"},
        {"name": "Duke Regional Hospital", "lat": 36.0726, "lon": -78.8278, "city": "Durham"},
        
        # 夏洛特地區
        {"name": "Carolinas Medical Center", "lat": 35.2045, "lon": -80.8395, "city": "Charlotte"},
        {"name": "Presbyterian Hospital", "lat": 35.2515, "lon": -80.8294, "city": "Charlotte"},
        {"name": "Mercy Hospital", "lat": 35.1968, "lon": -80.8414, "city": "Charlotte"},
        
        # 格林斯博羅地區
        {"name": "Moses H. Cone Memorial Hospital", "lat": 36.0835, "lon": -79.8235, "city": "Greensboro"},
        {"name": "Wesley Long Community Hospital", "lat": 36.0627, "lon": -79.7877, "city": "Greensboro"},
        
        # 溫斯頓-塞勒姆地區
        {"name": "Wake Forest Baptist Medical Center", "lat": 36.1123, "lon": -80.2779, "city": "Winston-Salem"},
        {"name": "Novant Health Forsyth Medical Center", "lat": 36.0998, "lon": -80.2442, "city": "Winston-Salem"},
        
        # 沿海地區
        {"name": "New Hanover Regional Medical Center", "lat": 34.2257, "lon": -77.9447, "city": "Wilmington"},
        {"name": "Vidant Medical Center", "lat": 35.6212, "lon": -77.3663, "city": "Greenville"},
        {"name": "CarolinaEast Medical Center", "lat": 35.1084, "lon": -77.0444, "city": "New Bern"},
        
        # 山區
        {"name": "Mission Hospital", "lat": 35.5731, "lon": -82.5515, "city": "Asheville"},
        {"name": "Appalachian Regional Healthcare", "lat": 36.2168, "lon": -81.6746, "city": "Boone"},
        
        # 其他地區
        {"name": "Cape Fear Valley Medical Center", "lat": 35.0879, "lon": -78.9582, "city": "Fayetteville"},
        {"name": "Cone Health MedCenter", "lat": 35.2524, "lon": -79.8009, "city": "Burlington"},
        {"name": "FirstHealth Moore Regional Hospital", "lat": 35.1779, "lon": -79.4608, "city": "Pinehurst"}
    ]
    
    # 轉換為DataFrame
    df = pd.DataFrame(mock_hospitals_data)
    
    if GEOPANDAS_AVAILABLE:
        # 創建Point geometry
        try:
            from shapely.geometry import Point
            geometry = [Point(lon, lat) for lat, lon in zip(df['lat'], df['lon'])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        except ImportError:
            # If shapely is not available, create a simple structure
            print("   ⚠️ Shapely not available, using simplified structure")
            gdf = df.copy()
            # Add a mock geometry column
            gdf['geometry'] = [type('Point', (), {'x': lon, 'y': lat})() 
                              for lat, lon in zip(df['lat'], df['lon'])]
    else:
        # No geopandas available, use simple DataFrame with geometry-like objects
        gdf = df.copy()
        # Create simple mock geometry objects
        gdf['geometry'] = [type('Point', (), {'x': lon, 'y': lat})() 
                          for lat, lon in zip(df['lat'], df['lon'])]
    
    # 添加OSM標準屬性
    gdf['amenity'] = 'hospital'
    gdf['healthcare'] = 'hospital'
    
    print(f"   ✅ 創建了 {len(gdf)} 家模擬醫院")
    
    return gdf


def process_hospital_geometries(gdf_hospitals):
    """
    處理醫院幾何形狀，將多邊形轉換為中心點
    
    Parameters:
    -----------
    gdf_hospitals : geopandas.GeoDataFrame
        原始醫院數據
        
    Returns:
    --------
    geopandas.GeoDataFrame
        處理後的醫院點位數據
    """
    
    print("   🔄 處理醫院幾何形狀...")
    
    processed_hospitals = []
    
    for idx, hospital in gdf_hospitals.iterrows():
        if hospital.geometry.geom_type == 'Point':
            # 已經是點位，直接保留
            processed_hospitals.append(hospital)
        else:
            # 多邊形或其他形狀，轉換為中心點
            hospital_copy = hospital.copy()
            hospital_copy.geometry = hospital.geometry.centroid
            processed_hospitals.append(hospital_copy)
    
    gdf_processed = gpd.GeoDataFrame(processed_hospitals, crs=gdf_hospitals.crs)
    
    print(f"   ✅ 處理完成，所有 {len(gdf_processed)} 家醫院已轉換為點位")
    
    return gdf_processed


def create_standardized_hospital_exposures(gdf_hospitals, value_config=None):
    """
    創建符合Steinmann論文的標準化醫院曝險數據
    支援基於醫院類型的差異化價值分配
    
    Parameters:
    -----------
    gdf_hospitals : geopandas.GeoDataFrame
        醫院位置數據
    value_config : dict, optional
        價值配置，包含:
        - base_value: 基礎價值 (預設 1.0)
        - type_multipliers: 醫院類型乘數字典
        - use_real_values: 是否使用真實價值 (預設 False)
        
    Returns:
    --------
    climada.entity.Exposures or dict
        標準化醫院曝險對象
    """
    
    print("🏥 創建標準化醫院曝險數據...")
    
    # 預設配置
    if value_config is None:
        value_config = {
            'base_value': 1.0,
            'type_multipliers': {
                'general': 1.0,
                'emergency': 2.0,
                'specialty': 1.5,
                'regional': 2.5,
                'university': 3.0,
                'community': 0.8
            },
            'use_real_values': False
        }
    
    base_value = value_config.get('base_value', 1.0)
    print(f"   📋 基礎價值: {base_value} 標準化單位")
    
    if not CLIMADA_AVAILABLE:
        print("   ⚠️ CLIMADA不可用，返回基本數據結構...")
        return {
            'hospitals': gdf_hospitals,
            'base_value': base_value,
            'total_hospitals': len(gdf_hospitals),
            'value_config': value_config
        }
    
    try:
        # 準備曝險數據
        gdf_exposure = gdf_hospitals.copy()
        
        # 設定醫院價值
        type_multipliers = value_config.get('type_multipliers', {})
        use_real_values = value_config.get('use_real_values', False)
        
        if use_real_values and 'hospital_type' in gdf_exposure.columns:
            # 基於醫院類型分配差異化價值
            gdf_exposure['value'] = gdf_exposure['hospital_type'].map(
                lambda t: base_value * type_multipliers.get(t, 1.0)
            )
            print(f"   💰 使用差異化價值: 類型乘數已應用")
        else:
            # 設定標準化價值：每家醫院 = base_value 單位
            gdf_exposure['value'] = base_value
        
        # 添加必要的CLIMADA屬性
        gdf_exposure['region_id'] = 840  # USA
        gdf_exposure['if_'] = 1  # 影響函數ID
        
        # 確保座標格式正確
        if 'lat' not in gdf_exposure.columns:
            gdf_exposure['latitude'] = gdf_exposure.geometry.y
            gdf_exposure['longitude'] = gdf_exposure.geometry.x
        
        # 創建CLIMADA Exposures對象
        hospital_exposures = Exposures(gdf_exposure)
        hospital_exposures.check()
        
        print(f"   ✅ 創建標準化醫院曝險: {len(hospital_exposures.gdf)} 家醫院")
        print(f"   💰 價值範圍: {hospital_exposures.value.min():.2f} - {hospital_exposures.value.max():.2f} 單位")
        print(f"   🏥 總計價值: {hospital_exposures.value.sum():.2f} 標準化單位")
        
        return hospital_exposures
        
    except Exception as e:
        print(f"   ❌ CLIMADA曝險創建失敗: {e}")
        return {
            'hospitals': gdf_hospitals,
            'standardized_value': 1.0,
            'total_hospitals': len(gdf_hospitals),
            'error': str(e)
        }


def visualize_hospitals(gdf_hospitals, save_plot=True):
    """
    視覺化醫院分布
    
    Parameters:
    -----------
    gdf_hospitals : geopandas.GeoDataFrame
        醫院數據
    save_plot : bool
        是否保存圖片
    """
    
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    print("📊 視覺化醫院分布...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 繪製醫院位置
    gdf_hospitals.plot(ax=ax, color='red', markersize=100, alpha=0.7, 
                      marker='+', linewidth=3, label='醫院位置')
    
    # 添加醫院名稱（如果有的話）
    if 'name' in gdf_hospitals.columns:
        for idx, hospital in gdf_hospitals.iterrows():
            if pd.notna(hospital['name']) and len(str(hospital['name'])) < 30:
                ax.annotate(hospital['name'], 
                          (hospital.geometry.x, hospital.geometry.y),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8)
    
    # 設定地圖範圍 (北卡羅來納州)
    ax.set_xlim(-84.5, -75.0)
    ax.set_ylim(33.5, 37.0)
    
    ax.set_xlabel('經度')
    ax.set_ylabel('緯度') 
    ax.set_title(f'北卡羅來納州醫院分布\n總計: {len(gdf_hospitals)} 家醫院')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_plot:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'nc_hospitals_distribution_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   📊 醫院分布圖已保存: {filename}")
    
    plt.show()
    
    return fig


# 便利函數
def get_nc_hospitals(use_mock=True, osm_file_path=None, create_exposures=True, visualize=True, value_config=None):
    """
    一站式獲取北卡羅來納州醫院數據
    
    Parameters:
    -----------
    use_mock : bool
        是否使用模擬數據
    osm_file_path : str, optional
        OSM文件路徑
    create_exposures : bool
        是否創建CLIMADA曝險對象
    visualize : bool
        是否顯示視覺化
    value_config : dict, optional
        醫院價值配置
        
    Returns:
    --------
    tuple
        (醫院GeoDataFrame, 標準化曝險對象)
    """
    
    print("🏥 開始獲取北卡羅來納州醫院數據...")
    
    # 提取醫院位置
    gdf_hospitals = extract_nc_hospitals_from_osm(osm_file_path, use_mock)
    
    # 視覺化
    if visualize and len(gdf_hospitals) > 0:
        visualize_hospitals(gdf_hospitals)
    
    # 創建標準化曝險
    hospital_exposures = None
    if create_exposures:
        hospital_exposures = create_standardized_hospital_exposures(gdf_hospitals, value_config)
    
    print(f"✅ 醫院數據獲取完成: {len(gdf_hospitals)} 家醫院")
    
    return gdf_hospitals, hospital_exposures


if __name__ == "__main__":
    # 測試運行
    print("🧪 測試醫院數據提取...")
    gdf_hospitals, hospital_exposures = get_nc_hospitals(use_mock=True)
    
    print(f"\n📊 結果摘要:")
    print(f"醫院總數: {len(gdf_hospitals)}")
    if hospital_exposures and hasattr(hospital_exposures, 'value'):
        print(f"標準化總價值: {hospital_exposures.value.sum()}")
    print("✅ 測試完成！")