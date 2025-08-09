"""
LitPop exposure data processing and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import climada.util.coordinates as u_coord
import climada.entity.exposures.litpop as lp
from climada.entity import LitPop


def process_litpop_exposures(country_iso="USA", state_name="North Carolina", years=range(2020, 2025)):
    """
    Process LitPop exposure data
    
    Parameters:
    -----------
    country_iso : str
        Country ISO code
    state_name : str
        State name
    years : range or list
        Years to process
        
    Returns:
    --------
    tuple
        (exposures_dict, successful_years) - Dictionary of exposures and list of successful years
    """
    
    exposures_dict = {}
    
    # Get state geographic boundaries
    print(f"Getting geographic boundaries for {state_name}...")
    admin1_info, admin1_shapes = u_coord.get_admin1_info(country_iso)
    admin1_info = admin1_info[country_iso]
    admin1_shapes = admin1_shapes[country_iso]
    
    # Find state index
    state_index = -1
    for idx, record in enumerate(admin1_info):
        if record["name"] == state_name:
            state_index = idx
            break
    
    if state_index == -1:
        raise ValueError(f"State not found: {state_name}")
    
    print(f"Found {state_name}, index: {state_index}")
    
    # Process year by year
    successful_years = []
    for year in years:
        print(f"\nProcessing {year} data...")
        start_time = time.time()
        
        try:
            # Estimate state total value (North Carolina is about 3.2% of US GDP)
            total_value_usa = lp._get_total_value_per_country(country_iso, "pc", year)
            total_value_state = 0.032 * total_value_usa
            
            # Create LitPop exposure
            exp = LitPop.from_shape(
                admin1_shapes[state_index],
                total_value_state,
                res_arcsec=600,  # About 600m resolution
                reference_year=year
            )
            
            exposures_dict[year] = exp
            successful_years.append(year)
            
            processing_time = time.time() - start_time
            print(f"   ✅ {year} completed - Asset points: {len(exp.gdf)}")
            print(f"   📊 Total value: ${total_value_state/1e9:.1f}B")
            print(f"   ⏱️ Processing time: {processing_time:.1f} seconds")
            
        except Exception as e:
            print(f"   ❌ {year} processing failed: {e}")
    
    return exposures_dict, successful_years


def visualize_all_litpop_exposures(exposures_dict, successful_years):
    """
    Create detailed visualizations for each successful year
    
    Parameters:
    -----------
    exposures_dict : dict
        Dictionary of exposure objects by year
    successful_years : list
        List of successfully processed years
        
    Returns:
    --------
    tuple
        (spatial_fig, value_fig) - Two matplotlib figures
    """
    
    n_years = len(successful_years)
    if n_years == 0:
        print("❌ No LitPop data available for visualization")
        return None, None
    
    # Calculate subplot layout
    n_cols = min(3, n_years)  # Maximum 3 columns
    n_rows = (n_years + n_cols - 1) // n_cols  # Round up
    
    # Create spatial distribution plots
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_years == 1:
        axes1 = [axes1]
    elif n_rows == 1:
        axes1 = axes1.reshape(1, -1)
    
    # Create value distribution histograms
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_years == 1:
        axes2 = [axes2]
    elif n_rows == 1:
        axes2 = axes2.reshape(1, -1)
    
    for i, year in enumerate(successful_years):
        row = i // n_cols
        col = i % n_cols
        
        exposure = exposures_dict[year]
        
        # Spatial distribution plot
        if n_years == 1:
            ax1 = axes1[0]  # When only one year, axes1 is a list with one element
        elif n_rows > 1:
            ax1 = axes1[row, col]
        else:
            ax1 = axes1[col]
        
        # Check longitude/latitude column names
        if 'longitude' in exposure.gdf.columns:
            lon_col, lat_col = 'longitude', 'latitude'
        elif 'lon' in exposure.gdf.columns:
            lon_col, lat_col = 'lon', 'lat'
        else:
            # If neither exists, use geometry coordinates
            lon_col = exposure.gdf.geometry.x
            lat_col = exposure.gdf.geometry.y
        
        if isinstance(lon_col, str):
            scatter = ax1.scatter(exposure.gdf[lon_col], exposure.gdf[lat_col], 
                                 c=exposure.gdf.value/1e6, 
                                 s=1, alpha=0.6, cmap='viridis')
        else:
            scatter = ax1.scatter(lon_col, lat_col, 
                                 c=exposure.gdf.value/1e6, 
                                 s=1, alpha=0.6, cmap='viridis')
        ax1.set_title(f'{year} LitPop Exposure Distribution')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Asset Value (Million USD)')
        
        # Value distribution histogram
        if n_years == 1:
            ax2 = axes2[0]  # When only one year, axes2 is a list with one element
        elif n_rows > 1:
            ax2 = axes2[row, col]
        else:
            ax2 = axes2[col]
        values_millions = exposure.gdf['value'] / 1e6
        
        n_bins = min(50, len(values_millions) // 10)  # Dynamically adjust bin count
        ax2.hist(values_millions, bins=n_bins, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Asset Value (Million USD)')
        ax2.set_ylabel('Number of Asset Points')
        ax2.set_title(f'{year} Asset Value Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add statistical information
        mean_val = values_millions.mean()
        median_val = values_millions.median()
        ax2.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: ${mean_val:.2f}M')
        ax2.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: ${median_val:.2f}M')
        ax2.legend()
    
    # Hide extra subplots
    if n_years > 1:  # Only hide extra subplots when there are multiple years
        for i in range(n_years, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes1[row, col].set_visible(False)
                axes2[row, col].set_visible(False)
            else:
                axes1[col].set_visible(False)
                axes2[col].set_visible(False)
    
    fig1.suptitle('LitPop Exposure Spatial Distribution by Year', fontsize=16)
    fig1.tight_layout()
    
    fig2.suptitle('Asset Value Distribution Statistics by Year', fontsize=16)
    fig2.tight_layout()
    
    return fig1, fig2


def create_yearly_comparison(exposures_dict, successful_years):
    """
    Create yearly comparison charts
    
    Parameters:
    -----------
    exposures_dict : dict
        Dictionary of exposure objects by year
    successful_years : list
        List of successfully processed years
        
    Returns:
    --------
    tuple
        (comparison_fig, stats_df) - Comparison figure and statistics dataframe
    """
    
    if len(successful_years) < 2:
        print("Need at least 2 years of data for comparison")
        return None, None
    
    # Collect yearly statistics
    stats_data = []
    for year in successful_years:
        exposure = exposures_dict[year]
        stats_data.append({
            'year': year,
            'total_value_b': exposure.gdf['value'].sum() / 1e9,
            'mean_value_m': exposure.gdf['value'].mean() / 1e6,
            'median_value_m': exposure.gdf['value'].median() / 1e6,
            'asset_count': len(exposure.gdf),
            'max_value_m': exposure.gdf['value'].max() / 1e6
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create comparison charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total value trend
    axes[0, 0].plot(stats_df['year'], stats_df['total_value_b'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Total Asset Value Trend')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Total Value (Billion USD)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Average single point value trend
    axes[0, 1].plot(stats_df['year'], stats_df['mean_value_m'], 'o-', color='orange', linewidth=2, markersize=8)
    axes[0, 1].set_title('Average Asset Point Value Trend')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Average Value (Million USD)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Asset point count trend
    axes[1, 0].plot(stats_df['year'], stats_df['asset_count'], 'o-', color='green', linewidth=2, markersize=8)
    axes[1, 0].set_title('Asset Point Count Trend')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Number of Asset Points')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Value statistics comparison (mean vs median)
    x_pos = np.arange(len(successful_years))
    width = 0.35
    
    axes[1, 1].bar(x_pos - width/2, stats_df['mean_value_m'], width, label='Mean', alpha=0.8)
    axes[1, 1].bar(x_pos + width/2, stats_df['median_value_m'], width, label='Median', alpha=0.8)
    axes[1, 1].set_title('Mean vs Median Comparison by Year')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Value (Million USD)')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(successful_years)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, stats_df


def analyze_exposure_statistics(exposures_dict, successful_years):
    """
    Analyze and display exposure statistics
    
    Parameters:
    -----------
    exposures_dict : dict
        Dictionary of exposure objects by year
    successful_years : list
        List of successfully processed years
    """
    
    if not successful_years:
        print("❌ No LitPop data available")
        return
        
    print(f"\n🏘️ LitPop exposure data processing completed, successfully processed {len(successful_years)} years")
    
    # Display detailed statistics for all years
    for year in successful_years:
        exposure = exposures_dict[year]
        print(f"\n📊 {year} Statistics:")
        print(f"   Asset points: {len(exposure.gdf):,}")
        print(f"   Total value: ${exposure.gdf['value'].sum()/1e9:.1f}B")
        print(f"   Average point value: ${exposure.gdf['value'].mean()/1e6:.2f}M")
        print(f"   Median point value: ${exposure.gdf['value'].median()/1e6:.2f}M")
        print(f"   Maximum point value: ${exposure.gdf['value'].max()/1e6:.2f}M")