"""
Hazard centroids generation and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from climada.util.coordinates import coord_on_land


def create_hazard_centroids(bounds, resolution=0.1):
    """
    Create grid centroids for hazard calculation
    
    Parameters:
    -----------
    bounds : dict
        Geographic bounds with lon_min, lon_max, lat_min, lat_max
    resolution : float
        Grid resolution in degrees
        
    Returns:
    --------
    tuple
        (centroids_lat_land, centroids_lon_land) - Arrays of land-based grid points
    """
    
    # Generate grid
    lat_range = np.arange(bounds['lat_min'], bounds['lat_max'], resolution)
    lon_range = np.arange(bounds['lon_min'], bounds['lon_max'], resolution)
    
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    centroids_lat = lat_grid.flatten()
    centroids_lon = lon_grid.flatten()
    
    print(f"Total grid points generated: {len(centroids_lat)}")
    
    # Filter land points
    print("Filtering land-based grid points...")
    land_mask = coord_on_land(centroids_lat, centroids_lon)
    centroids_lat_land = centroids_lat[land_mask]
    centroids_lon_land = centroids_lon[land_mask]
    
    print(f"Number of land grid points: {len(centroids_lat_land)}")
    
    return centroids_lat_land, centroids_lon_land


def visualize_centroids(centroids_lat, centroids_lon, bounds, resolution):
    """
    Visualize grid centroid distribution
    
    Parameters:
    -----------
    centroids_lat : np.array
        Latitude coordinates of centroids
    centroids_lon : np.array
        Longitude coordinates of centroids
    bounds : dict
        Geographic bounds
    resolution : float
        Grid resolution in degrees
        
    Returns:
    --------
    matplotlib.figure.Figure
        Generated plot figure
    """
    
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([bounds['lon_min'], bounds['lon_max'], 
                   bounds['lat_min'], bounds['lat_max']])
    
    # Geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.LAND, alpha=0.3, color='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
    
    # Grid points
    scatter = ax.scatter(centroids_lon, centroids_lat, 
                        c='red', s=0.5, alpha=0.6, transform=ccrs.PlateCarree())
    
    # Grid lines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.title(f'Hazard Calculation Grid Centroids\n({len(centroids_lat)} land points, resolution: {resolution}°)')
    plt.tight_layout()
    return fig