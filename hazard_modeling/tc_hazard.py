"""
Tropical cyclone hazard modeling functions
"""

import numpy as np
import matplotlib.pyplot as plt
from climada.hazard import TropCyclone
from climada.hazard.centroids import Centroids


def create_tc_hazard(tracks, centroids_lat, centroids_lon):
    """
    Create tropical cyclone hazard object
    
    Parameters:
    -----------
    tracks : TCTracks
        Tropical cyclone tracks
    centroids_lat : np.array
        Latitude coordinates of centroids
    centroids_lon : np.array
        Longitude coordinates of centroids
        
    Returns:
    --------
    TropCyclone
        CLIMADA tropical cyclone hazard object
    """
    
    print("Creating CLIMADA Centroids object...")
    centroids = Centroids.from_lat_lon(centroids_lat, centroids_lon)
    print(f"   Number of centroids: {centroids.size}")
    
    print("Calculating tropical cyclone hazard intensity...")
    tc_hazard = TropCyclone.from_tracks(tracks, centroids=centroids)
    
    # Check hazard object
    tc_hazard.check()
    
    return tc_hazard


def analyze_hazard_statistics(tc_hazard):
    """
    Analyze and display hazard statistics
    
    Parameters:
    -----------
    tc_hazard : TropCyclone
        CLIMADA tropical cyclone hazard object
    """
    
    print(f"\n📈 Tropical Cyclone Hazard Statistics:")
    print(f"   Number of hazard events: {tc_hazard.size}")
    print(f"   Number of calculation centroids: {tc_hazard.centroids.size}")
    print(f"   Hazard type: {tc_hazard.haz_type}")
    print(f"   Intensity units: {tc_hazard.units}")
    print(f"   Maximum wind speed: {tc_hazard.intensity.max():.1f} {tc_hazard.units}")
    print(f"   Average wind speed: {tc_hazard.intensity.mean():.1f} {tc_hazard.units}")


def visualize_hazard_intensity(tc_hazard):
    """
    Visualize hazard intensity distribution
    
    Parameters:
    -----------
    tc_hazard : TropCyclone
        CLIMADA tropical cyclone hazard object
        
    Returns:
    --------
    matplotlib.figure.Figure
        Generated plot figure
    """
    
    if tc_hazard.size == 0:
        print("No hazard data to visualize")
        return None
        
    max_intensity_per_event = tc_hazard.intensity.max(axis=1).toarray().flatten()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(max_intensity_per_event[max_intensity_per_event > 0], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Maximum Wind Speed (m/s)')
    plt.ylabel('Number of Events')
    plt.title('Maximum Wind Speed Distribution by Event')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    event_intensities = tc_hazard.intensity.toarray().flatten()
    event_intensities = event_intensities[event_intensities > 0]
    plt.hist(event_intensities, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Number of Grid Points')
    plt.title('Wind Speed Distribution Across All Grid Points')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()