"""
Tropical cyclone track data processing functions
"""

import numpy as np
from climada.hazard import TCTracks


def get_regional_tracks(bounds, year_range, nb_synth=10):
    """
    獲取並篩選區域颱風軌跡
    
    Parameters:
    -----------
    bounds : dict
        Geographic bounds with lon_min, lon_max, lat_min, lat_max
    year_range : tuple
        Start and end years for data retrieval
    nb_synth : int
        Number of synthetic tracks to generate
        
    Returns:
    --------
    TCTracks
        Filtered and perturbed tropical cyclone tracks
    """
    
    # 獲取北大西洋數據
    print(f"正在下載北大西洋颱風軌跡資料 ({year_range[0]}-{year_range[1]})...")
    na_tracks = TCTracks.from_ibtracs_netcdf(
        provider="usa",
        basin="NA", 
        year_range=year_range
    )
    print(f"   原始軌跡數: {na_tracks.size}")
    
    # 篩選穿過目標區域的軌跡
    print("正在篩選影響北卡的颱風軌跡...")
    regional_tracks = TCTracks()
    tracks_added = set()
    
    for track in na_tracks.data:
        if track.sid in tracks_added:
            continue
            
        # 檢查是否穿過北卡
        for lon, lat in zip(track.lon, track.lat):
            if (bounds['lon_min'] <= lon <= bounds['lon_max'] and 
                bounds['lat_min'] <= lat <= bounds['lat_max']):
                regional_tracks.append(track)
                tracks_added.add(track.sid)
                break
    
    print(f"   篩選後軌跡數: {regional_tracks.size}")
    
    # 標準化時間步長並生成擾動軌跡
    print("正在生成合成軌跡...")
    regional_tracks.equal_timestep()
    regional_tracks.calc_perturbed_trajectories(nb_synth_tracks=nb_synth)
    print(f"   擾動後總軌跡數: {regional_tracks.size}")
    
    return regional_tracks