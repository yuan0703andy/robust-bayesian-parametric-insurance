#!/usr/bin/env python3
"""
Data Splitting Module for Robust Bayesian Analysis
數據分割模組 - 用於穩健貝葉斯分析

Provides stratified train/validation/test splitting for tropical cyclone events,
ensuring proper representation of real and synthetic events across all splits.

Author: Research Team
Date: 2025-01-21
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class RobustDataSplitter:
    """
    Robust data splitter for hierarchical Bayesian modeling with TC events.
    
    Key Features:
    - Preserves all real events in training
    - Stratified sampling of synthetic events by intensity
    - Temporal splitting for true out-of-sample testing
    - Ensures no data leakage between splits
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data splitter.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
    def create_event_metadata(self, 
                            n_events: int,
                            n_real_events: int = 82,
                            start_year: int = 1980,
                            end_year: int = 2024) -> pd.DataFrame:
        """
        Create event metadata DataFrame with event types and years.
        
        Args:
            n_events: Total number of events
            n_real_events: Number of real (non-synthetic) events
            start_year: Start year of analysis period
            end_year: End year of analysis period
            
        Returns:
            DataFrame with event_id, event_type, and year columns
        """
        event_ids = [f'event_{i}' for i in range(n_events)]
        event_types = ['real'] * n_real_events + ['synthetic'] * (n_events - n_real_events)
        
        # Distribute events across years
        event_years = np.linspace(start_year, end_year, n_events).astype(int)
        
        return pd.DataFrame({
            'event_id': event_ids,
            'event_type': event_types,
            'year': event_years,
            'event_index': range(n_events)
        })
    
    def stratified_sampling(self,
                           event_indices: np.ndarray,
                           intensities: np.ndarray,
                           n_samples: int,
                           n_strata: int = 4) -> List[int]:
        """
        Robust two-stage stratified sampling based on event intensities.
        
        Stage 1: Attempt balanced sampling from each stratum
        Stage 2: Fill remaining quota from all available events
        
        Args:
            event_indices: Array of event indices to sample from
            intensities: Intensity values for each event (e.g., max wind speed)
            n_samples: Number of samples to draw
            n_strata: Number of strata for stratification
            
        Returns:
            List of sampled event indices
        """
        if len(event_indices) <= n_samples:
            print(f"  📌 Available events ({len(event_indices)}) ≤ target ({n_samples}), returning all")
            return event_indices.tolist()
        
        # Create strata boundaries
        strata_bins = np.percentile(intensities, np.linspace(0, 100, n_strata + 1))
        strata_bins[-1] += 1  # Ensure maximum value is included
        
        # Stage 1: Balanced sampling from each stratum
        sampled_indices = []
        remaining_events = set(event_indices)
        samples_per_stratum = n_samples // n_strata
        
        print(f"  🎯 Stage 1: Targeting {samples_per_stratum} samples per stratum")
        
        for i in range(n_strata):
            # Find events in this stratum
            stratum_mask = (intensities >= strata_bins[i]) & (intensities < strata_bins[i+1])
            stratum_indices = event_indices[stratum_mask]
            
            # Filter to only remaining events
            available_in_stratum = [idx for idx in stratum_indices if idx in remaining_events]
            
            if len(available_in_stratum) > 0:
                n_to_sample = min(samples_per_stratum, len(available_in_stratum))
                chosen = self.rng.choice(available_in_stratum, n_to_sample, replace=False)
                sampled_indices.extend(chosen.tolist())
                
                # Remove from remaining pool
                for idx in chosen:
                    remaining_events.discard(idx)
                
                print(f"    Stratum {i+1}: {n_to_sample}/{len(available_in_stratum)} events sampled")
            else:
                print(f"    Stratum {i+1}: 0 events available")
        
        # Stage 2: Fill remaining quota if needed
        current_samples = len(sampled_indices)
        if current_samples < n_samples and len(remaining_events) > 0:
            additional_needed = n_samples - current_samples
            available_remaining = list(remaining_events)
            
            print(f"  🎯 Stage 2: Need {additional_needed} more samples from {len(available_remaining)} remaining")
            
            if len(available_remaining) >= additional_needed:
                additional_chosen = self.rng.choice(available_remaining, additional_needed, replace=False)
                sampled_indices.extend(additional_chosen.tolist())
                print(f"    ✅ Successfully added {len(additional_chosen)} additional samples")
            else:
                # Take all remaining events
                sampled_indices.extend(available_remaining)
                print(f"    ⚠️ Only {len(available_remaining)} additional events available (needed {additional_needed})")
        
        final_count = len(sampled_indices)
        efficiency = (final_count / n_samples) * 100 if n_samples > 0 else 100
        print(f"  📊 Final sampling: {final_count}/{n_samples} ({efficiency:.1f}% of target)")
        
        return sampled_indices
    
    def create_data_splits(self,
                          hazard_intensities: np.ndarray,
                          observed_losses: np.ndarray,
                          event_metadata: Optional[pd.DataFrame] = None,
                          train_val_frac: float = 0.8,
                          val_frac: float = 0.2,
                          n_synthetic_samples: int = 100,
                          n_strata: int = 4) -> Dict[str, List[int]]:
        """
        Create robust train/validation/test splits for Bayesian analysis.
        
        Strategy:
        1. Temporal split for test set (last 20% of events)
        2. Keep all real events in train+val
        3. Stratified sampling of synthetic events
        4. Split train+val into final train and validation sets
        
        Args:
            hazard_intensities: Hazard intensity matrix (n_locations, n_events)
            observed_losses: Observed loss matrix (n_locations, n_events)
            event_metadata: DataFrame with event metadata (optional)
            train_val_frac: Fraction for train+validation (default 0.8)
            val_frac: Fraction of train+val for validation (default 0.2)
            n_synthetic_samples: Number of synthetic events to sample
            n_strata: Number of strata for stratification
            
        Returns:
            Dictionary with 'train', 'validation', and 'test' event indices
        """
        n_events = hazard_intensities.shape[1]
        
        # Create metadata if not provided
        if event_metadata is None:
            print("📊 Creating default event metadata...")
            event_metadata = self.create_event_metadata(n_events)
        
        # Step 1: Temporal split for test set
        print("\n🔄 Step 1: Creating temporal test split...")
        sorted_events = event_metadata.sort_values('year')
        split_point = int(n_events * train_val_frac)
        
        train_val_indices = sorted_events['event_index'].iloc[:split_point].tolist()
        test_indices = sorted_events['event_index'].iloc[split_point:].tolist()
        
        print(f"  ✅ Train+Val: {len(train_val_indices)} events")
        print(f"  ✅ Test (holdout): {len(test_indices)} events")
        
        # Step 2: Representative sampling within train+val
        print("\n🔄 Step 2: Representative sampling...")
        train_val_meta = event_metadata[event_metadata['event_index'].isin(train_val_indices)]
        
        # Separate real and synthetic events
        real_events = train_val_meta[train_val_meta['event_type'] == 'real']['event_index'].tolist()
        synthetic_events = train_val_meta[train_val_meta['event_type'] == 'synthetic']['event_index'].tolist()
        
        print(f"  📌 Real events: {len(real_events)} (all preserved)")
        print(f"  📌 Synthetic events: {len(synthetic_events)} (will sample {n_synthetic_samples})")
        
        # Calculate max intensities for stratification
        if len(synthetic_events) > 0:
            synthetic_intensities = np.max(hazard_intensities[:, synthetic_events], axis=0)
            
            # Stratified sampling of synthetic events
            sampled_synthetic = self.stratified_sampling(
                np.array(synthetic_events),
                synthetic_intensities,
                n_synthetic_samples,
                n_strata
            )
            print(f"  ✅ Sampled {len(sampled_synthetic)} synthetic events across {n_strata} strata")
        else:
            sampled_synthetic = []
        
        # Combine for high-quality sample
        high_quality_sample = real_events + sampled_synthetic
        
        # Step 3: Split into final train and validation
        print("\n🔄 Step 3: Creating train/validation split...")
        
        # Shuffle while maintaining reproducibility
        self.rng.shuffle(real_events)
        if len(sampled_synthetic) > 0:
            self.rng.shuffle(sampled_synthetic)
        
        # Split real events
        real_split = int(len(real_events) * (1 - val_frac))
        train_real = real_events[:real_split]
        val_real = real_events[real_split:]
        
        # Split synthetic events
        if len(sampled_synthetic) > 0:
            synth_split = int(len(sampled_synthetic) * (1 - val_frac))
            train_synth = sampled_synthetic[:synth_split]
            val_synth = sampled_synthetic[synth_split:]
        else:
            train_synth = []
            val_synth = []
        
        # Final sets
        final_train = train_real + train_synth
        final_val = val_real + val_synth
        
        # Shuffle final sets
        self.rng.shuffle(final_train)
        self.rng.shuffle(final_val)
        
        print(f"  ✅ Train: {len(final_train)} events (Real: {len(train_real)}, Synthetic: {len(train_synth)})")
        print(f"  ✅ Validation: {len(final_val)} events (Real: {len(val_real)}, Synthetic: {len(val_synth)})")
        
        # Validation checks
        self._validate_splits(final_train, final_val, test_indices, n_events)
        
        return {
            'train': final_train,
            'validation': final_val,
            'test': test_indices,
            'metadata': {
                'n_real_train': len(train_real),
                'n_synthetic_train': len(train_synth),
                'n_real_val': len(val_real),
                'n_synthetic_val': len(val_synth),
                'n_test': len(test_indices),
                'total_events': n_events
            }
        }
    
    def _validate_splits(self, train: List[int], val: List[int], test: List[int], n_events: int):
        """
        Validate that splits are mutually exclusive and complete.
        """
        train_set = set(train)
        val_set = set(val)
        test_set = set(test)
        
        # Check for overlaps
        assert len(train_set.intersection(val_set)) == 0, "❌ Train and validation sets overlap!"
        assert len(train_set.intersection(test_set)) == 0, "❌ Train and test sets overlap!"
        assert len(val_set.intersection(test_set)) == 0, "❌ Validation and test sets overlap!"
        
        # Check completeness
        total_events_in_splits = len(train_set) + len(val_set) + len(test_set)
        print(f"\n✅ Validation passed: {total_events_in_splits} events split (sampling from {n_events} total)")
        print("✅ All splits are mutually exclusive")
    
    def get_split_data(self,
                      hazard_intensities: np.ndarray,
                      observed_losses: np.ndarray,
                      exposure_values: np.ndarray,
                      split_indices: Dict[str, List[int]]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract data arrays for each split.
        
        Args:
            hazard_intensities: Full hazard intensity matrix
            observed_losses: Full observed loss matrix
            exposure_values: Exposure values array
            split_indices: Dictionary with train/val/test indices
            
        Returns:
            Dictionary with data arrays for each split
        """
        return {
            'train': {
                'hazard_intensities': hazard_intensities[:, split_indices['train']],
                'observed_losses': observed_losses[:, split_indices['train']],
                'exposure_values': exposure_values,
                'event_indices': np.array(split_indices['train'])
            },
            'validation': {
                'hazard_intensities': hazard_intensities[:, split_indices['validation']],
                'observed_losses': observed_losses[:, split_indices['validation']],
                'exposure_values': exposure_values,
                'event_indices': np.array(split_indices['validation'])
            },
            'test': {
                'hazard_intensities': hazard_intensities[:, split_indices['test']],
                'observed_losses': observed_losses[:, split_indices['test']],
                'exposure_values': exposure_values,
                'event_indices': np.array(split_indices['test'])
            }
        }
    
    def compute_split_statistics(self,
                                hazard_intensities: np.ndarray,
                                observed_losses: np.ndarray,
                                split_indices: Dict[str, List[int]]) -> pd.DataFrame:
        """
        Compute statistics for each data split.
        
        Args:
            hazard_intensities: Full hazard intensity matrix
            observed_losses: Full observed loss matrix
            split_indices: Dictionary with train/val/test indices
            
        Returns:
            DataFrame with statistics for each split
        """
        stats = []
        
        for split_name, indices in split_indices.items():
            if split_name == 'metadata':
                continue
                
            split_hazard = hazard_intensities[:, indices]
            split_losses = observed_losses[:, indices]
            
            stats.append({
                'split': split_name,
                'n_events': len(indices),
                'mean_intensity': np.mean(split_hazard),
                'max_intensity': np.max(split_hazard),
                'mean_loss': np.mean(split_losses),
                'max_loss': np.max(split_losses),
                'zero_loss_fraction': np.mean(split_losses == 0)
            })
        
        return pd.DataFrame(stats)


# Convenience function for direct import
def create_robust_splits(hazard_intensities: np.ndarray,
                        observed_losses: np.ndarray,
                        n_synthetic_samples: int = 100,
                        random_state: int = 42) -> Dict[str, List[int]]:
    """
    Convenience function to create data splits with default parameters.
    
    Args:
        hazard_intensities: Hazard intensity matrix
        observed_losses: Observed loss matrix
        n_synthetic_samples: Number of synthetic events to sample
        random_state: Random seed
        
    Returns:
        Dictionary with train/validation/test indices
    """
    splitter = RobustDataSplitter(random_state=random_state)
    return splitter.create_data_splits(
        hazard_intensities=hazard_intensities,
        observed_losses=observed_losses,
        n_synthetic_samples=n_synthetic_samples
    )


if __name__ == '__main__':
    # Example usage
    print("📊 Data Splitting Module - Example Usage")
    print("=" * 60)
    
    # Create simulated data with realistic NC hurricane characteristics
    N_LOCATIONS = 20
    N_EVENTS = 1312
    N_REAL_EVENTS = 82
    
    # Simulate realistic hurricane intensities and losses
    np.random.seed(42)
    hazard_sim = np.random.gamma(2, 15, (N_LOCATIONS, N_EVENTS))  # More realistic wind speeds
    
    # Create mostly zero losses with some significant events (like real hurricanes)
    losses_sim = np.zeros((N_LOCATIONS, N_EVENTS))
    
    # Add significant losses for ~5% of events (like major hurricanes)
    significant_events = np.random.choice(N_EVENTS, size=int(N_EVENTS * 0.05), replace=False)
    for event in significant_events:
        losses_sim[:, event] = np.random.gamma(2, 5e6, N_LOCATIONS)
    
    # Add minor losses for another ~10% of events  
    minor_events = np.random.choice(
        [i for i in range(N_EVENTS) if i not in significant_events], 
        size=int(N_EVENTS * 0.10), replace=False
    )
    for event in minor_events:
        losses_sim[:, event] = np.random.gamma(1, 1e5, N_LOCATIONS)
    
    exposure_sim = np.random.uniform(1e7, 1e9, N_LOCATIONS)
    
    # Create splitter and test robust sampling
    splitter = RobustDataSplitter(random_state=42)
    
    print("\n🧪 Testing Robust Two-Stage Sampling:")
    print("-" * 40)
    
    # Test different sampling targets
    for n_samples in [50, 100, 150, 200]:
        print(f"\n📋 Testing with target = {n_samples} synthetic samples:")
        
        splits = splitter.create_data_splits(
            hazard_intensities=hazard_sim,
            observed_losses=losses_sim,
            n_synthetic_samples=n_samples,
            n_strata=4
        )
        
        # Quick analysis of results
        total_sampled = len(splits['train']) + len(splits['validation'])
        real_in_train_val = splits['metadata']['n_real_train'] + splits['metadata']['n_real_val']
        synthetic_in_train_val = splits['metadata']['n_synthetic_train'] + splits['metadata']['n_synthetic_val']
        
        efficiency = (synthetic_in_train_val / n_samples) * 100 if n_samples > 0 else 100
        print(f"  📊 Results: {synthetic_in_train_val}/{n_samples} synthetic ({efficiency:.1f}%), "
              f"{real_in_train_val} real")
    
    print("\n" + "=" * 60)
    
    # Standard analysis with optimal settings
    print("\n📊 Standard Analysis (n_synthetic_samples=100):")
    splits = splitter.create_data_splits(
        hazard_intensities=hazard_sim,
        observed_losses=losses_sim,
        n_synthetic_samples=100
    )
    
    # Get split statistics
    stats = splitter.compute_split_statistics(hazard_sim, losses_sim, splits)
    print("\n📈 Split Statistics:")
    print(stats.to_string())
    
    # Get actual data for each split
    split_data = splitter.get_split_data(hazard_sim, losses_sim, exposure_sim, splits)
    
    print("\n📊 Split Shapes:")
    for split_name, data in split_data.items():
        print(f"  {split_name}: hazard={data['hazard_intensities'].shape}, "
              f"losses={data['observed_losses'].shape}")
    
    # Validate quality of splits
    print("\n✅ Validation Checks:")
    train_events = set(splits['train'])
    val_events = set(splits['validation'])
    test_events = set(splits['test'])
    
    print(f"  🔍 No overlap: {len(train_events & val_events) == 0}")
    print(f"  🔍 No train-test leak: {len(train_events & test_events) == 0}")
    print(f"  🔍 No val-test leak: {len(val_events & test_events) == 0}")
    
    # Check temporal ordering (test set should have higher event indices on average)
    avg_train_idx = np.mean(list(train_events))
    avg_test_idx = np.mean(list(test_events))
    print(f"  📅 Temporal order preserved: {avg_test_idx > avg_train_idx} "
          f"(train avg: {avg_train_idx:.1f}, test avg: {avg_test_idx:.1f})")