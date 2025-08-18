"""
MCMC Validator Module
MCMC 驗證模組
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import numpy as np
import pymc as pm
import arviz as az
from typing import Dict, Optional, Any

from .model_factory import ModelFactory, ModelConfig


class MCMCValidator:
    """MCMC validation for selected models"""
    
    def __init__(self, chains: int = 4, draws: int = 1000, tune: int = 1000):
        """
        Initialize MCMC Validator
        
        Args:
            chains: Number of MCMC chains
            draws: Number of samples per chain
            tune: Number of tuning samples
        """
        self.chains = chains
        self.draws = draws
        self.tune = tune
        self.results = {}
        
    def validate_model(self, config: ModelConfig, X: np.ndarray, y: np.ndarray,
                       approx: Optional[Any] = None) -> Dict:
        """
        Run MCMC validation for a model
        
        Args:
            config: Model configuration
            X: Feature matrix
            y: Target variable
            approx: VI approximation for initialization
            
        Returns:
            Dictionary with MCMC results
        """
        print(f"\n🔥 MCMC Validation: {config.name}")
        print(f"   Chains: {self.chains}, Draws: {self.draws}, Tune: {self.tune}")
        
        start_time = time.time()
        
        # Create model
        model = ModelFactory.create_model(config, X, y)
        
        with model:
            # Initialize from VI if available
            init_vals = None
            if approx is not None:
                try:
                    init_points = approx.sample(self.chains)
                    init_vals = [
                        {k: v[i] for k, v in init_points.items()}
                        for i in range(self.chains)
                    ]
                    print("   📍 Using VI initialization")
                except Exception as e:
                    print(f"   ⚠️ VI init failed: {e}, using default")
                    init_vals = None
            
            # Run MCMC with divergence monitoring
            # First run a short pilot to detect obvious problems
            pilot_draws = min(100, self.draws // 10)
            pilot_tune = min(200, self.tune // 5)
            
            print("   🔍 Running pilot chain to detect issues...")
            try:
                pilot_trace = pm.sample(
                    draws=pilot_draws,
                    tune=pilot_tune,
                    chains=1,
                    cores=1,
                    initvals=init_vals[:1] if init_vals else None,
                    target_accept=0.9,
                    return_inferencedata=True,
                    progressbar=False,
                    discard_tuned_samples=False
                )
                
                # Check pilot for issues
                pilot_divergences = pilot_trace.sample_stats.diverging.sum().item()
                pilot_rhat = az.summary(pilot_trace)['r_hat'].max()
                
                if pilot_divergences > pilot_draws * 0.1:  # More than 10% divergences
                    print(f"   ⚠️ High divergence rate in pilot: {pilot_divergences}/{pilot_draws}")
                    print(f"   ⏭️ Skipping full MCMC for {config.name}")
                    trace = pilot_trace  # Use pilot as final
                    early_stopped = True
                elif pilot_rhat > 1.5:  # Poor mixing
                    print(f"   ⚠️ Poor mixing detected (R̂={pilot_rhat:.2f})")
                    print(f"   ⏭️ Skipping full MCMC for {config.name}")
                    trace = pilot_trace
                    early_stopped = True
                else:
                    print(f"   ✅ Pilot successful, running full MCMC...")
                    early_stopped = False
                    
                    # Run full MCMC
                    trace = pm.sample(
                        draws=self.draws,
                        tune=self.tune,
                        chains=self.chains,
                        cores=min(self.chains, 4),
                        initvals=init_vals,
                        target_accept=0.95,
                        return_inferencedata=True,
                        progressbar=True
                    )
                    
            except Exception as pilot_error:
                print(f"   ❌ Pilot failed: {pilot_error}")
                print(f"   ⏭️ Skipping {config.name}")
                raise pilot_error
            
            # Calculate diagnostics
            summary = az.summary(trace)
            
            # Model comparison metrics
            try:
                waic = az.waic(trace)
                loo = az.loo(trace)
            except Exception as e:
                print(f"   ⚠️ WAIC/LOO calculation failed: {e}")
                waic = loo = None
            
            # Extract diagnostics
            divergences = trace.sample_stats.diverging.sum().item()
            mean_ess = summary['ess_mean'].mean()
            mean_rhat = summary['r_hat'].mean()
            
            elapsed = time.time() - start_time
            
            result = {
                'config': config,
                'trace': trace,
                'summary': summary,
                'waic': waic.elpd_waic if waic else None,
                'loo': loo.elpd_loo if loo else None,
                'divergences': divergences,
                'mean_ess': mean_ess,
                'mean_rhat': mean_rhat,
                'time': elapsed,
                'early_stopped': early_stopped,
                'actual_draws': pilot_draws if early_stopped else self.draws,
                'actual_chains': 1 if early_stopped else self.chains
            }
            
            # Store result
            self.results[config.name] = result
            
            status = "⚠️ Early stopped" if early_stopped else "✅ Complete"
            print(f"   {status} in {elapsed/60:.1f} minutes")
            print(f"      Divergences: {divergences}")
            print(f"      Mean R̂: {mean_rhat:.3f}")
            print(f"      Mean ESS: {mean_ess:.0f}")
            if early_stopped:
                print(f"      ⚠️ Used pilot only: {pilot_draws} draws, 1 chain")
            if waic:
                print(f"      WAIC: {waic.elpd_waic:.2f}")
            
        return result
    
    def validate_multiple(self, vi_results: list, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Validate multiple models from VI results
        
        Args:
            vi_results: List of VI screening results
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of MCMC results
        """
        print(f"\n{'='*60}")
        print(f"🎯 MCMC Validation for {len(vi_results)} models")
        print(f"{'='*60}")
        
        for vi_result in vi_results:
            if not vi_result['success']:
                print(f"   ⏭️ Skipping {vi_result['config'].name} (VI failed)")
                continue
            
            mcmc_result = self.validate_model(
                vi_result['config'],
                X, y,
                vi_result.get('approx')
            )
            
        return self.results
    
    def get_best_model(self, metric: str = 'waic') -> Optional[Dict]:
        """
        Get best model by metric
        
        Args:
            metric: Metric to use ('waic', 'loo', 'divergences')
            
        Returns:
            Best model result
        """
        if not self.results:
            return None
        
        if metric == 'divergences':
            # Lower is better
            return min(self.results.values(), 
                      key=lambda x: x.get('divergences', float('inf')))
        else:
            # Higher is better for WAIC/LOO
            valid_results = [r for r in self.results.values() 
                           if r.get(metric) is not None]
            if not valid_results:
                return None
            return max(valid_results, key=lambda x: x[metric])