"""
CLIMADA Data Loader for VI+MCMC Framework
CLIMADA 數據載入器
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class CLIMADADataLoader:
    """Load and prepare CLIMADA data for Bayesian analysis"""
    
    def __init__(self, base_path: Path = None):
        """Initialize data loader"""
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent
        self.base_path = base_path
        self.results_path = base_path / 'results'
        
    def load_climada_data(self) -> Optional[Dict]:
        """Load CLIMADA data from pickle file"""
        
        climada_path = self.results_path / 'climada_data' / 'climada_complete_data.pkl'
        
        if not climada_path.exists():
            print(f"⚠️ CLIMADA data not found at {climada_path}")
            return None
            
        try:
            with open(climada_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Loaded CLIMADA data from {climada_path}")
            return data
        except Exception as e:
            print(f"❌ Error loading CLIMADA data: {e}")
            return None
    
    def load_spatial_analysis_data(self) -> Optional[Dict]:
        """Load spatial analysis results"""
        
        spatial_path = self.results_path / 'spatial_analysis' / 'cat_in_circle_results.pkl'
        
        if not spatial_path.exists():
            print(f"⚠️ Spatial data not found at {spatial_path}")
            return None
            
        try:
            with open(spatial_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Loaded spatial analysis data from {spatial_path}")
            return data
        except Exception as e:
            print(f"❌ Error loading spatial data: {e}")
            return None
    
    def load_insurance_products(self) -> Optional[Dict]:
        """Load insurance products data"""
        
        products_path = self.results_path / 'insurance_products' / 'products.pkl'
        
        if not products_path.exists():
            print(f"⚠️ Insurance products not found at {products_path}")
            return None
            
        try:
            with open(products_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Loaded insurance products from {products_path}")
            return data
        except Exception as e:
            print(f"❌ Error loading insurance products: {e}")
            return None
    
    def prepare_regression_data(self, data_dict: Dict) -> Dict:
        """
        Prepare data for Bayesian regression from CLIMADA results
        
        Args:
            data_dict: Dictionary containing CLIMADA data
            
        Returns:
            Dictionary with X (features) and y (losses) for regression
        """
        print("\n📊 Preparing regression data from CLIMADA...")
        
        # Try different data structures
        if 'impact_results' in data_dict:
            # Use impact calculation results
            impact_data = data_dict['impact_results']
            
            if hasattr(impact_data, 'event_id') and hasattr(impact_data, 'eai_exp'):
                # Extract features from impact data
                event_damages = impact_data.at_event
                
                # Create features from CLIMADA impact data
                features = []
                losses = []
                
                for i, damage in enumerate(event_damages):
                    if damage > 0:  # Only include events with damage
                        # Basic features: event index, damage magnitude
                        features.append([i, np.log(damage + 1)])
                        losses.append(damage)
                
                X = np.array(features)
                y = np.array(losses)
                
            else:
                # Fallback: use any available numerical data
                X, y = self._extract_features_fallback(impact_data)
                
        elif 'yearset_results' in data_dict:
            # Use yearset results
            X, y = self._prepare_from_yearset(data_dict['yearset_results'])
            
        elif 'damages_fixed' in data_dict:
            # Use fixed damages
            X, y = self._prepare_from_damages(data_dict['damages_fixed'])
            
        else:
            # Generate features from any available data
            X, y = self._extract_features_fallback(data_dict)
        
        # Ensure we have valid data
        if X.size == 0 or y.size == 0:
            print("⚠️ No valid features extracted, creating minimal synthetic data")
            X, y = self._create_minimal_data()
        
        print(f"   ✅ Features shape: {X.shape}")
        print(f"   ✅ Losses shape: {y.shape}")
        print(f"   📈 Loss statistics: mean={np.mean(y):.2e}, std={np.std(y):.2e}")
        
        return {
            'X': X,
            'y': y,
            'features': X,
            'losses': y,
            'n_samples': len(y),
            'n_features': X.shape[1] if X.ndim > 1 else 1
        }
    
    def _prepare_from_yearset(self, yearset_data) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data from yearset results"""
        
        if hasattr(yearset_data, 'index') and hasattr(yearset_data, 'values'):
            # Pandas DataFrame/Series
            years = np.array(yearset_data.index)
            damages = np.array(yearset_data.values)
        elif isinstance(yearset_data, dict):
            years = np.array(list(yearset_data.keys()))
            damages = np.array(list(yearset_data.values()))
        else:
            # Array-like
            damages = np.array(yearset_data)
            years = np.arange(len(damages))
        
        # Create features: year, log(year), trend
        X = np.column_stack([
            years,
            np.log(years - years.min() + 1),
            np.arange(len(years))  # trend
        ])
        
        return X, damages
    
    def _prepare_from_damages(self, damages_data) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data from damages"""
        
        if hasattr(damages_data, 'values'):
            damages = damages_data.values
        else:
            damages = np.array(damages_data)
        
        # Simple features: index, squared index, log index
        n = len(damages)
        indices = np.arange(n)
        
        X = np.column_stack([
            indices,
            indices**2,
            np.log(indices + 1)
        ])
        
        return X, damages
    
    def _extract_features_fallback(self, data) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback feature extraction"""
        
        # Try to find any numerical arrays in the data
        arrays = []
        
        def find_arrays(obj, path=""):
            if isinstance(obj, np.ndarray) and obj.size > 0:
                arrays.append((path, obj.flatten()))
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    find_arrays(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    find_arrays(v, f"{path}[{i}]")
        
        find_arrays(data)
        
        if len(arrays) >= 2:
            # Use first array as features, second as target
            _, X_flat = arrays[0]
            _, y_flat = arrays[1]
            
            # Reshape to compatible sizes
            min_len = min(len(X_flat), len(y_flat))
            X = X_flat[:min_len].reshape(-1, 1)
            y = y_flat[:min_len]
            
            return X, y
        elif len(arrays) == 1:
            # Use single array as target, create features
            _, y = arrays[0]
            X = np.arange(len(y)).reshape(-1, 1)
            return X, y
        else:
            return self._create_minimal_data()
    
    def _create_minimal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create minimal synthetic data as last resort"""
        print("   ⚠️ Creating minimal synthetic data as fallback")
        
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)  # 3 features
        y = X @ [1.0, -0.5, 0.2] + np.random.exponential(1, n)  # Positive losses
        
        return X, y
    
    def load_for_bayesian_analysis(self) -> Dict:
        """
        Load all available data and prepare for Bayesian analysis
        
        Returns:
            Dictionary with regression data and metadata
        """
        print("\n🔍 Loading CLIMADA data for Bayesian analysis...")
        
        # Try to load different data sources
        climada_data = self.load_climada_data()
        spatial_data = self.load_spatial_analysis_data()
        insurance_data = self.load_insurance_products()
        
        # Choose best available data source
        if climada_data:
            print("   📊 Using CLIMADA complete data")
            regression_data = self.prepare_regression_data(climada_data)
            metadata = {
                'data_source': 'climada_complete',
                'original_data': climada_data
            }
        elif spatial_data:
            print("   📍 Using spatial analysis data")
            regression_data = self.prepare_regression_data(spatial_data)
            metadata = {
                'data_source': 'spatial_analysis',
                'original_data': spatial_data
            }
        elif insurance_data:
            print("   💼 Using insurance products data")
            regression_data = self.prepare_regression_data(insurance_data)
            metadata = {
                'data_source': 'insurance_products',
                'original_data': insurance_data
            }
        else:
            print("   ⚠️ No CLIMADA data found, using minimal synthetic data")
            X, y = self._create_minimal_data()
            regression_data = {
                'X': X, 'y': y, 'features': X, 'losses': y,
                'n_samples': len(y), 'n_features': X.shape[1]
            }
            metadata = {
                'data_source': 'synthetic_fallback',
                'original_data': None
            }
        
        # Add metadata to regression data
        regression_data.update(metadata)
        
        return regression_data