"""
VI Screener Module
變分推論篩選模組
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import numpy as np
import pandas as pd
import pymc as pm
from typing import Dict, List, Optional, Any
from joblib import Parallel, delayed
import multiprocessing as mp

from .model_factory import ModelFactory, ModelConfig


class VIScreener:
    """Variational Inference screening for fast model selection"""
    
    def __init__(self, n_jobs: Optional[int] = None):
        """
        Initialize VI Screener
        
        Args:
            n_jobs: Number of parallel jobs (default: CPU count - 2)
        """
        self.n_jobs = n_jobs or min(mp.cpu_count() - 2, 30)
        self.results = []
        
    def run_single_vi(self, config: ModelConfig, X: np.ndarray, y: np.ndarray,
                     n_iterations: int = 30000) -> Dict:
        """
        Run VI for a single model configuration
        
        Args:
            config: Model configuration
            X: Feature matrix
            y: Target variable
            n_iterations: Number of VI iterations
            
        Returns:
            Dictionary with VI results
        """
        print(f"  🔄 VI screening: {config.name}")
        start_time = time.time()
        
        try:
            # Create model
            model = ModelFactory.create_model(config, X, y)
            
            with model:
                # Run ADVI
                approx = pm.fit(
                    n=n_iterations,
                    method='advi',
                    callbacks=[
                        pm.callbacks.CheckParametersConvergence(
                            diff='relative',
                            tolerance=0.01
                        )
                    ],
                    progressbar=False
                )
                
                # Extract ELBO
                elbo_history = -np.array(approx.hist)
                final_elbo = elbo_history[-1]
                
                # Check convergence
                if len(elbo_history) > 1000:
                    converged = np.std(elbo_history[-1000:]) < 1.0
                else:
                    converged = False
                
                # Sample from approximate posterior
                vi_samples = approx.sample(1000)
                
                elapsed = time.time() - start_time
                
                result = {
                    'config': config,
                    'elbo': final_elbo,
                    'converged': converged,
                    'time': elapsed,
                    'approx': approx,
                    'vi_samples': vi_samples,
                    'elbo_history': elbo_history,
                    'success': True
                }
                
                print(f"  ✅ {config.name}: ELBO={final_elbo:.2f}, Time={elapsed:.1f}s")
                
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                'config': config,
                'elbo': -np.inf,
                'converged': False,
                'time': elapsed,
                'error': str(e),
                'success': False
            }
            print(f"  ❌ {config.name}: {e}")
        
        return result
    
    def screen_models(self, configs: List[ModelConfig], X: np.ndarray, y: np.ndarray,
                     n_iterations: int = 30000) -> pd.DataFrame:
        """
        Screen multiple models in parallel
        
        Args:
            configs: List of model configurations
            X: Feature matrix
            y: Target variable
            n_iterations: Number of VI iterations
            
        Returns:
            DataFrame with screening results
        """
        print(f"\n📊 VI Screening: {len(configs)} models with {self.n_jobs} parallel jobs")
        print("=" * 60)
        
        # Run in parallel
        results = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(self.run_single_vi)(config, X, y, n_iterations)
            for config in configs
        )
        
        # Store full results
        self.results = results
        
        # Create summary DataFrame
        df = pd.DataFrame([
            {
                'name': r['config'].name,
                'type': r['config'].model_type,
                'epsilon': r['config'].epsilon,
                'nu': r['config'].nu,
                'elbo': r['elbo'],
                'converged': r['converged'],
                'time': r['time'],
                'success': r['success']
            }
            for r in results
        ])
        
        # Sort by ELBO
        df = df.sort_values('elbo', ascending=False)
        
        return df
    
    def get_top_models(self, k: int = 5) -> List[Dict]:
        """
        Get top k models by ELBO
        
        Args:
            k: Number of top models to return
            
        Returns:
            List of top model results
        """
        if not self.results:
            raise ValueError("No screening results available. Run screen_models first.")
        
        # Sort by ELBO
        sorted_results = sorted(self.results, 
                               key=lambda x: x['elbo'], 
                               reverse=True)
        
        # Filter successful models
        successful = [r for r in sorted_results if r['success']]
        
        return successful[:k]
    
    def get_model_by_type(self, model_type: str) -> Optional[Dict]:
        """
        Get best model of a specific type
        
        Args:
            model_type: Type of model ('epsilon', 'student_t', etc.)
            
        Returns:
            Best model result of that type
        """
        type_results = [r for r in self.results 
                       if r['config'].model_type == model_type and r['success']]
        
        if not type_results:
            return None
        
        return max(type_results, key=lambda x: x['elbo'])