"""
Configuration settings for North Carolina Tropical Cyclone Analysis
"""

import warnings
warnings.filterwarnings('ignore')

# Geographic bounds for North Carolina
NC_BOUNDS = {
    'lon_min': -84.5, 'lon_max': -75.5,
    'lat_min': 33.8, 'lat_max': 36.6
}

# Time period for analysis
YEAR_RANGE = (1980, 2024)

# Spatial resolution in degrees
RESOLUTION = 0.1

# Matplotlib settings for Chinese font
MATPLOTLIB_CONFIG = {
    'font.family': 'Heiti TC'
}

# Track analysis parameters
TRACK_PARAMS = {
    'nb_synth': 5,  # Number of synthetic tracks to generate
    'provider': "usa",
    'basin': "NA"
}

# Color mappings for storm categories
STORM_COLORS = {
    'TD': '#74a9cf',     # Tropical Depression
    'TS': '#2b8cbe',     # Tropical Storm  
    'H1': '#fee391',     # Hurricane Category 1
    'H2': '#fec44f',     # Hurricane Category 2
    'H3': '#fe9929',     # Hurricane Category 3
    'H4': '#d95f0e',     # Hurricane Category 4
    'H5': '#993404'      # Hurricane Category 5
}

# Impact function parameters
IMPACT_FUNC_PARAMS = {
    'impf_id': 1,
    'intensity_unit': 'm/s'
}

# Exposure parameters
EXPOSURE_PARAMS = {
    'res_arcsec': 300,  # Resolution in arc seconds (5 arcmin)
    'total_value': 200000000,  # Total exposure value in USD
    'reference_year': 2020
}