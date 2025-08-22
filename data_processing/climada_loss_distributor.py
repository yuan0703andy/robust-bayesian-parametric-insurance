#!/usr/bin/env python3
"""
CLIMADA Loss Distribution Module
CLIMADA損失分解模組

Distributes CLIMADA's event-level losses to hospital-level spatial resolution
for spatial analysis and parametric insurance evaluation.

將CLIMADA事件級別的損失分解到醫院級別的空間分辨率，
用於空間分析和參數保險評估。

Author: Research Team
Date: 2025-01-21
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class CLIMADALossDistributor:
    """
    Distributes CLIMADA event-level total losses to hospital-level spatial resolution.
    
    This class solves the fundamental problem of converting:
    - CLIMADA total losses per event: (n_events,) 
    - To hospital-level losses: (n_hospitals, n_events)
    
    Using physically meaningful weighting based on wind exposure and asset values.
    """
    
    def __init__(self, method: str = 'wind_exposure_weighted'):
        """
        Initialize the loss distributor.
        
        Args:
            method: Distribution method
                - 'wind_exposure_weighted': Weight by wind speed and exposure (recommended)
                - 'exposure_only': Weight by exposure values only
                - 'wind_only': Weight by wind speeds only
                - 'equal': Equal distribution across hospitals
        """
        self.method = method
        self.distribution_stats = {}
        
    def distribute_losses(self,
                         climada_impact,
                         hospital_coords: np.ndarray,
                         exposure_values: np.ndarray, 
                         hazard_intensities: np.ndarray,
                         wind_threshold: float = 25.7,
                         exposure_power: float = 1.0,
                         wind_power: float = 1.5) -> Tuple[np.ndarray, Dict]:
        """
        Distribute CLIMADA total losses to hospital-level resolution.
        
        Args:
            climada_impact: CLIMADA Impact object with at_event losses
            hospital_coords: Hospital coordinates (n_hospitals, 2) 
            exposure_values: Hospital exposure values (n_hospitals,)
            hazard_intensities: Wind speeds at hospitals (n_hospitals, n_events)
            wind_threshold: Minimum wind speed for damage (m/s)
            exposure_power: Power for exposure weighting
            wind_power: Power for wind speed weighting
            
        Returns:
            Tuple of:
            - observed_losses: Hospital-level losses (n_hospitals, n_events)
            - distribution_stats: Distribution statistics and validation
        """
        n_hospitals, n_events = hazard_intensities.shape
        
        # Get CLIMADA real total losses per event
        total_losses = climada_impact.at_event  # (n_events,)
        
        print(f"🎯 Distributing CLIMADA real losses to {n_hospitals} hospitals...")
        print(f"   Total CLIMADA losses: ${np.sum(total_losses)/1e9:.1f}B across {n_events} events")
        print(f"   Method: {self.method}")
        
        # Initialize hospital-level loss matrix
        observed_losses = np.zeros((n_hospitals, n_events))
        
        # Track distribution statistics
        events_with_losses = 0
        total_distributed = 0
        zero_weight_events = 0
        
        for event_idx in range(n_events):
            total_event_loss = total_losses[event_idx]
            
            if total_event_loss > 0:
                events_with_losses += 1
                
                # Calculate weights for each hospital
                weights = self._calculate_hospital_weights(
                    event_idx, hazard_intensities, exposure_values,
                    wind_threshold, exposure_power, wind_power
                )
                
                total_weight = np.sum(weights)
                
                if total_weight > 0:
                    # Distribute total loss proportionally
                    for h_idx in range(n_hospitals):
                        hospital_share = weights[h_idx] / total_weight
                        observed_losses[h_idx, event_idx] = total_event_loss * hospital_share
                    
                    total_distributed += total_event_loss
                else:
                    # No hospitals affected - distribute equally among exposed hospitals
                    exposed_hospitals = exposure_values > 0
                    n_exposed = np.sum(exposed_hospitals)
                    if n_exposed > 0:
                        loss_per_hospital = total_event_loss / n_exposed
                        observed_losses[exposed_hospitals, event_idx] = loss_per_hospital
                        total_distributed += total_event_loss
                    zero_weight_events += 1
        
        # Calculate distribution statistics
        distribution_stats = {
            'method': self.method,
            'total_original_loss': np.sum(total_losses),
            'total_distributed_loss': total_distributed,
            'conservation_ratio': total_distributed / np.sum(total_losses) if np.sum(total_losses) > 0 else 0,
            'events_with_losses': events_with_losses,
            'zero_weight_events': zero_weight_events,
            'hospital_loss_stats': {
                'mean_per_hospital': np.mean(np.sum(observed_losses, axis=1)),
                'max_single_event': np.max(observed_losses),
                'hospitals_with_losses': np.sum(np.any(observed_losses > 0, axis=1)),
                'loss_concentration': self._calculate_loss_concentration(observed_losses)
            }
        }
        
        self._print_distribution_summary(distribution_stats)
        
        return observed_losses, distribution_stats
    
    def _calculate_hospital_weights(self,
                                   event_idx: int,
                                   hazard_intensities: np.ndarray,
                                   exposure_values: np.ndarray,
                                   wind_threshold: float,
                                   exposure_power: float,
                                   wind_power: float) -> np.ndarray:
        """Calculate distribution weights for each hospital for a given event."""
        n_hospitals = hazard_intensities.shape[0]
        weights = np.zeros(n_hospitals)
        
        for h_idx in range(n_hospitals):
            wind_speed = hazard_intensities[h_idx, event_idx]
            exposure = exposure_values[h_idx]
            
            if self.method == 'wind_exposure_weighted':
                if wind_speed > wind_threshold and exposure > 0:
                    # Weight by both wind speed and exposure
                    wind_factor = ((wind_speed - wind_threshold) / 50) ** wind_power
                    exposure_factor = (exposure / 1e9) ** exposure_power  # Normalize by $1B
                    weights[h_idx] = wind_factor * exposure_factor
                    
            elif self.method == 'exposure_only':
                if exposure > 0:
                    weights[h_idx] = (exposure / 1e9) ** exposure_power
                    
            elif self.method == 'wind_only':
                if wind_speed > wind_threshold:
                    weights[h_idx] = ((wind_speed - wind_threshold) / 50) ** wind_power
                    
            elif self.method == 'equal':
                if wind_speed > wind_threshold and exposure > 0:
                    weights[h_idx] = 1.0
        
        return weights
    
    def _calculate_loss_concentration(self, observed_losses: np.ndarray) -> Dict:
        """Calculate loss concentration metrics (like Gini coefficient)."""
        hospital_totals = np.sum(observed_losses, axis=1)
        hospital_totals_sorted = np.sort(hospital_totals)
        
        n = len(hospital_totals_sorted)
        cumsum = np.cumsum(hospital_totals_sorted)
        
        # Gini coefficient
        if np.sum(hospital_totals) > 0:
            gini = (2 * np.sum((np.arange(1, n+1)) * hospital_totals_sorted)) / (n * np.sum(hospital_totals)) - (n + 1) / n
        else:
            gini = 0
        
        return {
            'gini_coefficient': gini,
            'top_10pct_share': np.sum(hospital_totals_sorted[-int(n*0.1):]) / np.sum(hospital_totals) if np.sum(hospital_totals) > 0 else 0,
            'bottom_50pct_share': np.sum(hospital_totals_sorted[:int(n*0.5)]) / np.sum(hospital_totals) if np.sum(hospital_totals) > 0 else 0
        }
    
    def _print_distribution_summary(self, stats: Dict):
        """Print summary of loss distribution results."""
        print(f"\n📊 CLIMADA Loss Distribution Results:")
        print(f"   Original total: ${stats['total_original_loss']/1e9:.2f}B")
        print(f"   Distributed total: ${stats['total_distributed_loss']/1e9:.2f}B")
        print(f"   Conservation ratio: {stats['conservation_ratio']:.1%}")
        print(f"   Events with losses: {stats['events_with_losses']:,}")
        print(f"   Zero-weight events: {stats['zero_weight_events']:,}")
        
        hospital_stats = stats['hospital_loss_stats']
        print(f"\n🏥 Hospital-Level Statistics:")
        print(f"   Mean loss per hospital: ${hospital_stats['mean_per_hospital']/1e6:.1f}M")
        print(f"   Max single event loss: ${hospital_stats['max_single_event']/1e6:.1f}M")
        print(f"   Hospitals with losses: {hospital_stats['hospitals_with_losses']}")
        
        concentration = hospital_stats['loss_concentration']
        print(f"   Loss concentration (Gini): {concentration['gini_coefficient']:.3f}")
        print(f"   Top 10% hospitals share: {concentration['top_10pct_share']:.1%}")
    
    def validate_distribution(self,
                            original_losses: np.ndarray,
                            distributed_losses: np.ndarray,
                            tolerance: float = 0.01) -> Dict:
        """
        Validate that the distribution conserves total losses and makes sense.
        
        Args:
            original_losses: CLIMADA event-level losses (n_events,)
            distributed_losses: Hospital-level losses (n_hospitals, n_events)
            tolerance: Acceptable relative error for conservation
            
        Returns:
            Validation results dictionary
        """
        # Check conservation of total losses
        original_total = np.sum(original_losses)
        distributed_total = np.sum(distributed_losses)
        conservation_error = abs(distributed_total - original_total) / original_total if original_total > 0 else 0
        
        # Check event-by-event conservation
        event_errors = []
        for event_idx in range(len(original_losses)):
            if original_losses[event_idx] > 0:
                event_distributed = np.sum(distributed_losses[:, event_idx])
                event_error = abs(event_distributed - original_losses[event_idx]) / original_losses[event_idx]
                event_errors.append(event_error)
        
        # Check for negative losses
        negative_losses = np.sum(distributed_losses < 0)
        
        # Check for unrealistic concentrations
        max_single_loss = np.max(distributed_losses)
        mean_event_loss = np.mean(original_losses[original_losses > 0]) if np.any(original_losses > 0) else 0
        
        validation_results = {
            'conservation_error': conservation_error,
            'conservation_pass': conservation_error < tolerance,
            'mean_event_error': np.mean(event_errors) if event_errors else 0,
            'max_event_error': np.max(event_errors) if event_errors else 0,
            'negative_losses': negative_losses,
            'max_single_loss': max_single_loss,
            'loss_realism_ratio': max_single_loss / mean_event_loss if mean_event_loss > 0 else 0,
            'overall_pass': (conservation_error < tolerance and 
                           negative_losses == 0 and 
                           (max_single_loss / mean_event_loss < 10 if mean_event_loss > 0 else True))
        }
        
        print(f"\n✅ Distribution Validation:")
        print(f"   Conservation error: {conservation_error:.1%} (pass: {validation_results['conservation_pass']})")
        print(f"   No negative losses: {negative_losses == 0}")
        print(f"   Overall validation: {'PASS' if validation_results['overall_pass'] else 'FAIL'}")
        
        return validation_results


# Convenience function for direct import
def distribute_climada_losses_to_hospitals(climada_impact,
                                         hospital_coords: np.ndarray,
                                         exposure_values: np.ndarray,
                                         hazard_intensities: np.ndarray,
                                         method: str = 'wind_exposure_weighted') -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to distribute CLIMADA losses to hospitals.
    
    Args:
        climada_impact: CLIMADA Impact object
        hospital_coords: Hospital coordinates 
        exposure_values: Hospital exposure values
        hazard_intensities: Wind speeds at hospitals
        method: Distribution method
        
    Returns:
        Tuple of (hospital_losses, distribution_stats)
    """
    distributor = CLIMADALossDistributor(method=method)
    return distributor.distribute_losses(
        climada_impact, hospital_coords, exposure_values, hazard_intensities
    )


if __name__ == '__main__':
    # Test with synthetic data
    print("🧪 Testing CLIMADA Loss Distribution")
    print("=" * 50)
    
    # Create synthetic test data
    n_hospitals = 20
    n_events = 100
    
    # Synthetic CLIMADA-like impact data
    class MockImpact:
        def __init__(self):
            # Create realistic event losses (heavy-tailed distribution)
            self.at_event = np.random.lognormal(mean=15, sigma=2, size=n_events)
            # Add some zero-loss events
            zero_events = np.random.choice(n_events, size=n_events//3, replace=False)
            self.at_event[zero_events] = 0
    
    mock_impact = MockImpact()
    
    # Synthetic hospital data
    hospital_coords = np.random.uniform(-82, -78, (n_hospitals, 2))
    exposure_values = np.random.lognormal(mean=20, sigma=1, size=n_hospitals)
    hazard_intensities = np.random.gamma(shape=2, scale=15, size=(n_hospitals, n_events))
    
    print(f"📊 Test data:")
    print(f"   Mock CLIMADA total: ${np.sum(mock_impact.at_event)/1e9:.1f}B")
    print(f"   Hospital exposures: ${np.sum(exposure_values)/1e9:.1f}B")
    print(f"   Wind speed range: {np.min(hazard_intensities):.1f} - {np.max(hazard_intensities):.1f} m/s")
    
    # Test the distribution
    hospital_losses, stats = distribute_climada_losses_to_hospitals(
        mock_impact, hospital_coords, exposure_values, hazard_intensities
    )
    
    # Validate results
    distributor = CLIMADALossDistributor()
    validation = distributor.validate_distribution(
        mock_impact.at_event, hospital_losses
    )
    
    print(f"\n🎯 Test Results:")
    print(f"   Hospital losses shape: {hospital_losses.shape}")
    print(f"   Non-zero hospital losses: {np.count_nonzero(hospital_losses):,}")
    print(f"   Distribution method: {stats['method']}")
    print(f"   Validation passed: {validation['overall_pass']}")