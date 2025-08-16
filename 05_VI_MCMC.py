#!/usr/bin/env python3
"""
05. VI+ELBO Screening → MCMC Validation Framework
變分推論篩選 → MCMC 驗證框架

Two-stage approach: Fast VI screening of 60+ models, then detailed MCMC on top performers.
兩階段方法：快速 VI 篩選 60+ 模型，然後對表現最佳者進行詳細 MCMC。

Optimized for 32-core CPU with 32GB RAM.

Author: Research Team
Date: 2025-01-17
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import multiprocessing as mp
from functools import partial
from joblib import Parallel, delayed
import json

# System setup for 32 cores
N_CORES = min(32, mp.cpu_count())
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent nested parallelism
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# PyMC settings
os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64,openmp=False'
os.environ['PYTHONUNBUFFERED'] = '1'

print("=" * 80)
print("🚀 VI+ELBO → MCMC Two-Stage Framework")
print(f"💻 System: {N_CORES} cores, 32GB RAM")
print("=" * 80)

# Import libraries
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    from scipy import stats
    from properscoring import crps_ensemble
    print("✅ Core libraries loaded")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')
az.style.use("arviz-darkgrid")

# ============================================================================
# Model Configuration Classes
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for each model variant"""
    name: str
    robust_type: str  # 'standard', 'epsilon', 'student_t', 'huber', 'mixture'
    epsilon: Optional[float] = None
    nu: Optional[float] = None  # For Student-t
    delta: Optional[float] = None  # For Huber
    prior_type: str = 'weakly_informative'  # 'informative', 'weakly_informative', 'horseshoe'
    likelihood_complexity: str = 'simple'  # 'simple', 'zero_inflated', 'compound'
    feature_engineering: str = 'base'  # 'base', 'polynomial', 'interaction'
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}

# ============================================================================
# Model Building Functions
# ============================================================================

def build_model(config: ModelConfig, data: Dict) -> pm.Model:
    """Build PyMC model based on configuration"""
    
    X = data['X']
    y = data['y']
    n_samples, n_features = X.shape
    
    with pm.Model() as model:
        # ===== Prior Selection =====
        if config.prior_type == 'informative':
            beta = pm.Normal('beta', mu=data.get('prior_mean', 0), 
                           sigma=data.get('prior_std', 1), shape=n_features)
            sigma = pm.HalfNormal('sigma', sigma=data.get('prior_sigma', 10))
            
        elif config.prior_type == 'weakly_informative':
            beta = pm.Normal('beta', mu=0, sigma=10, shape=n_features)
            sigma = pm.HalfNormal('sigma', sigma=10)
            
        elif config.prior_type == 'horseshoe':
            # Horseshoe prior for sparsity
            tau = pm.HalfCauchy('tau', beta=1)
            lam = pm.HalfCauchy('lam', beta=1, shape=n_features)
            beta = pm.Normal('beta', mu=0, sigma=tau * lam, shape=n_features)
            sigma = pm.HalfNormal('sigma', sigma=10)
        
        # ===== Feature Engineering =====
        if config.feature_engineering == 'base':
            mu = pm.math.dot(X, beta)
        elif config.feature_engineering == 'polynomial':
            # Add quadratic terms
            X_poly = np.column_stack([X, X**2])
            beta_poly = pm.Normal('beta_poly', mu=0, sigma=10, shape=X_poly.shape[1])
            mu = pm.math.dot(X_poly, beta_poly)
        elif config.feature_engineering == 'interaction':
            # Add interaction terms
            X_int = X[:, 0:1] * X[:, 1:2] if n_features >= 2 else X
            beta_int = pm.Normal('beta_int', mu=0, sigma=10, shape=X_int.shape[1])
            mu = pm.math.dot(X, beta) + pm.math.dot(X_int, beta_int)
        
        # ===== Likelihood Selection =====
        if config.robust_type == 'standard':
            if config.likelihood_complexity == 'simple':
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            elif config.likelihood_complexity == 'zero_inflated':
                psi = pm.Beta('psi', alpha=1, beta=1)
                y_obs = pm.ZeroInflatedNormal('y_obs', psi=psi, mu=mu, sigma=sigma, observed=y)
                
        elif config.robust_type == 'epsilon':
            # ε-contamination model
            epsilon = config.epsilon or 0.1
            
            # Main component
            main_dist = pm.Normal.dist(mu=mu, sigma=sigma)
            
            # Contamination component (heavy-tailed)
            contam_dist = pm.StudentT.dist(nu=3, mu=mu, sigma=sigma*5)
            
            # Mixture
            y_obs = pm.Mixture('y_obs', 
                              w=[1-epsilon, epsilon],
                              comp_dists=[main_dist, contam_dist],
                              observed=y)
                              
        elif config.robust_type == 'student_t':
            nu = config.nu or 5
            y_obs = pm.StudentT('y_obs', nu=nu, mu=mu, sigma=sigma, observed=y)
            
        elif config.robust_type == 'huber':
            # Approximate Huber with mixture
            delta = config.delta or 1.5
            # This is a simplification - true Huber needs custom implementation
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
        elif config.robust_type == 'mixture':
            # Two-component Gaussian mixture
            w = pm.Dirichlet('w', a=np.ones(2))
            mu2 = pm.Normal('mu2', mu=0, sigma=10)
            sigma2 = pm.HalfNormal('sigma2', sigma=10)
            
            components = [
                pm.Normal.dist(mu=mu, sigma=sigma),
                pm.Normal.dist(mu=mu2, sigma=sigma2)
            ]
            y_obs = pm.Mixture('y_obs', w=w, comp_dists=components, observed=y)
    
    return model

# ============================================================================
# VI Screening Functions
# ============================================================================

def run_vi_single(config: ModelConfig, data: Dict, n_iterations: int = 30000) -> Dict:
    """Run VI for a single model configuration"""
    
    start_time = time.time()
    print(f"  🔄 Starting VI for {config.name}...")
    
    try:
        with build_model(config, data) as model:
            # Run ADVI
            approx = pm.fit(
                n=n_iterations,
                method='advi',
                callbacks=[pm.callbacks.CheckParametersConvergence(diff='relative', tolerance=0.01)],
                progressbar=False
            )
            
            # Calculate ELBO
            elbo_trace = -np.array(approx.hist)
            final_elbo = elbo_trace[-1]
            converged = np.std(elbo_trace[-1000:]) < 1.0 if len(elbo_trace) > 1000 else False
            
            # Sample from VI posterior for diagnostics
            vi_trace = approx.sample(1000)
            
            # Calculate additional metrics
            mean_params = {var: approx.mean[var].eval() for var in approx.mean}
            std_params = {var: approx.std[var].eval() for var in approx.std}
            
            elapsed = time.time() - start_time
            
            result = {
                'config': config,
                'elbo': final_elbo,
                'converged': converged,
                'elapsed_time': elapsed,
                'approx': approx,
                'vi_trace': vi_trace,
                'mean_params': mean_params,
                'std_params': std_params,
                'elbo_history': elbo_trace
            }
            
            print(f"  ✅ {config.name}: ELBO={final_elbo:.2f}, Time={elapsed:.1f}s, Converged={converged}")
            
        return result
        
    except Exception as e:
        print(f"  ❌ {config.name} failed: {e}")
        return {
            'config': config,
            'elbo': -np.inf,
            'converged': False,
            'elapsed_time': time.time() - start_time,
            'error': str(e)
        }

def parallel_vi_screening(configs: List[ModelConfig], data: Dict, 
                         n_jobs: int = None) -> pd.DataFrame:
    """Run VI screening in parallel for all model configurations"""
    
    if n_jobs is None:
        n_jobs = min(len(configs), N_CORES - 2)  # Leave 2 cores free
    
    print(f"\n🚀 Running VI screening for {len(configs)} models using {n_jobs} cores")
    print("=" * 60)
    
    # Run in parallel
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(run_vi_single)(config, data, n_iterations=30000)
        for config in configs
    )
    
    # Convert to DataFrame
    df_results = pd.DataFrame([
        {
            'model_name': r['config'].name,
            'robust_type': r['config'].robust_type,
            'epsilon': r['config'].epsilon,
            'nu': r['config'].nu,
            'prior_type': r['config'].prior_type,
            'likelihood': r['config'].likelihood_complexity,
            'features': r['config'].feature_engineering,
            'elbo': r['elbo'],
            'converged': r['converged'],
            'time_seconds': r['elapsed_time'],
            'error': r.get('error', None)
        }
        for r in results
    ])
    
    # Sort by ELBO
    df_results = df_results.sort_values('elbo', ascending=False)
    
    # Store full results for later use
    df_results.attrs['full_results'] = results
    
    return df_results

# ============================================================================
# MCMC Validation Functions
# ============================================================================

def run_mcmc_single(config: ModelConfig, data: Dict, 
                    approx: Optional[Any] = None,
                    draws: int = 1000, chains: int = 4) -> Dict:
    """Run MCMC for a single model"""
    
    start_time = time.time()
    print(f"\n🔥 Running MCMC for {config.name} ({chains} chains, {draws} draws each)")
    
    with build_model(config, data) as model:
        # Use VI initialization if available
        if approx is not None:
            try:
                init_points = approx.sample(chains)
                init_vals = [{k: v[i] for k, v in init_points.items()} 
                            for i in range(chains)]
                print(f"  📍 Using VI initialization")
            except:
                init_vals = None
                print(f"  📍 Using default initialization")
        else:
            init_vals = None
        
        # Sample with NUTS
        trace = pm.sample(
            draws=draws,
            tune=1000,
            chains=chains,
            cores=min(chains, 4),  # Use 4 cores per model max
            init='adapt_diag' if init_vals is None else None,
            initvals=init_vals,
            target_accept=0.95,
            return_inferencedata=True,
            progressbar=True
        )
        
        # Calculate diagnostics
        summary = az.summary(trace)
        
        # WAIC and LOO
        waic = az.waic(trace)
        loo = az.loo(trace)
        
        elapsed = time.time() - start_time
        
        result = {
            'config': config,
            'trace': trace,
            'summary': summary,
            'waic': waic.elpd_waic,
            'loo': loo.elpd_loo,
            'elapsed_time': elapsed,
            'n_divergences': trace.sample_stats.diverging.sum().item(),
            'mean_ess': summary['ess_mean'].mean(),
            'mean_rhat': summary['r_hat'].mean()
        }
        
        print(f"  ✅ {config.name} complete: Time={elapsed/60:.1f}min, "
              f"Divergences={result['n_divergences']}, "
              f"Mean R̂={result['mean_rhat']:.3f}")
        
        return result

def run_mcmc_top_models(vi_results: pd.DataFrame, data: Dict, 
                        top_k: int = 5) -> Dict:
    """Run MCMC for top K models from VI screening"""
    
    print(f"\n🎯 Running MCMC for top {top_k} models")
    print("=" * 60)
    
    # Get top models
    top_models = vi_results.head(top_k)
    
    # Get full results with approx objects
    full_results = vi_results.attrs.get('full_results', [])
    
    mcmc_results = {}
    
    for idx, row in top_models.iterrows():
        # Find corresponding full result
        full_result = next((r for r in full_results 
                           if r['config'].name == row['model_name']), None)
        
        if full_result and not full_result.get('error'):
            config = full_result['config']
            approx = full_result.get('approx')
            
            # Run MCMC
            mcmc_result = run_mcmc_single(
                config, data, approx,
                draws=1000, chains=4
            )
            
            mcmc_results[config.name] = mcmc_result
    
    return mcmc_results

# ============================================================================
# Model Comparison and Analysis
# ============================================================================

def comprehensive_model_comparison(vi_results: pd.DataFrame, 
                                  mcmc_results: Dict) -> pd.DataFrame:
    """Create comprehensive comparison table"""
    
    comparison_data = []
    
    for model_name, mcmc_result in mcmc_results.items():
        # Get VI results
        vi_row = vi_results[vi_results['model_name'] == model_name].iloc[0]
        
        comparison_data.append({
            'model': model_name,
            'robust_type': vi_row['robust_type'],
            # VI metrics
            'vi_elbo': vi_row['elbo'],
            'vi_time': vi_row['time_seconds'],
            'vi_converged': vi_row['converged'],
            # MCMC metrics
            'mcmc_waic': mcmc_result['waic'],
            'mcmc_loo': mcmc_result['loo'],
            'mcmc_time': mcmc_result['elapsed_time'],
            'mcmc_divergences': mcmc_result['n_divergences'],
            'mcmc_ess': mcmc_result['mean_ess'],
            'mcmc_rhat': mcmc_result['mean_rhat'],
            # Combined score
            'combined_score': vi_row['elbo'] + mcmc_result['waic']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('combined_score', ascending=False)
    
    return df_comparison

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_vi_screening_results(vi_results: pd.DataFrame, save_path: Path):
    """Plot VI screening results"""
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # ELBO by robust type
    ax = axes[0, 0]
    vi_results.boxplot(column='elbo', by='robust_type', ax=ax)
    ax.set_title('ELBO by Robust Type')
    ax.set_xlabel('Robust Type')
    ax.set_ylabel('ELBO')
    
    # Convergence rate
    ax = axes[0, 1]
    conv_rate = vi_results.groupby('robust_type')['converged'].mean()
    conv_rate.plot(kind='bar', ax=ax)
    ax.set_title('Convergence Rate by Robust Type')
    ax.set_ylabel('Convergence Rate')
    
    # Time comparison
    ax = axes[1, 0]
    vi_results.boxplot(column='time_seconds', by='robust_type', ax=ax)
    ax.set_title('VI Time by Robust Type')
    ax.set_xlabel('Robust Type')
    ax.set_ylabel('Time (seconds)')
    
    # Top 10 models
    ax = axes[1, 1]
    top_10 = vi_results.head(10)
    ax.barh(range(len(top_10)), top_10['elbo'])
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10['model_name'])
    ax.set_xlabel('ELBO')
    ax.set_title('Top 10 Models by ELBO')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path / 'vi_screening_results.png', dpi=150)
    plt.close()

def plot_mcmc_comparison(comparison_df: pd.DataFrame, save_path: Path):
    """Plot MCMC comparison results"""
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # VI ELBO vs MCMC WAIC
    ax = axes[0, 0]
    ax.scatter(comparison_df['vi_elbo'], comparison_df['mcmc_waic'])
    ax.set_xlabel('VI ELBO')
    ax.set_ylabel('MCMC WAIC')
    ax.set_title('VI ELBO vs MCMC WAIC')
    for i, row in comparison_df.iterrows():
        ax.annotate(row['model'][:10], (row['vi_elbo'], row['mcmc_waic']), fontsize=8)
    
    # Time comparison
    ax = axes[0, 1]
    x = range(len(comparison_df))
    width = 0.35
    ax.bar([i - width/2 for i in x], comparison_df['vi_time'], width, label='VI')
    ax.bar([i + width/2 for i in x], comparison_df['mcmc_time'], width, label='MCMC')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['model'], rotation=45, ha='right')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computation Time Comparison')
    ax.legend()
    
    # Diagnostic metrics
    ax = axes[1, 0]
    ax.scatter(comparison_df['mcmc_ess'], comparison_df['mcmc_rhat'])
    ax.axhline(y=1.01, color='r', linestyle='--', alpha=0.5, label='R̂ threshold')
    ax.set_xlabel('Mean ESS')
    ax.set_ylabel('Mean R̂')
    ax.set_title('MCMC Diagnostics')
    for i, row in comparison_df.iterrows():
        ax.annotate(row['model'][:10], (row['mcmc_ess'], row['mcmc_rhat']), fontsize=8)
    ax.legend()
    
    # Combined ranking
    ax = axes[1, 1]
    ax.barh(range(len(comparison_df)), comparison_df['combined_score'])
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df['model'])
    ax.set_xlabel('Combined Score (ELBO + WAIC)')
    ax.set_title('Final Model Ranking')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path / 'mcmc_comparison.png', dpi=150)
    plt.close()

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_analysis_data() -> Dict:
    """Load the analysis data"""
    
    print("\n📂 Loading data...")
    
    # Try to load from pickle first
    pickle_path = Path('data/processed_data.pkl')
    if pickle_path.exists():
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  ✅ Loaded data from {pickle_path}")
        return data
    
    # Otherwise create synthetic data for testing
    print("  ⚠️ No data found, creating synthetic data for demonstration")
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Create synthetic data with outliers
    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([2.0, -1.5, 0.8, -0.3, 1.2])
    
    # Clean signal
    y_clean = X @ true_beta + np.random.randn(n_samples) * 2
    
    # Add outliers (10%)
    n_outliers = int(0.1 * n_samples)
    outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)
    y = y_clean.copy()
    y[outlier_idx] += np.random.randn(n_outliers) * 10
    
    data = {
        'X': X,
        'y': y,
        'true_beta': true_beta,
        'outlier_idx': outlier_idx,
        'n_samples': n_samples,
        'n_features': n_features
    }
    
    return data

# ============================================================================
# Model Configuration Generation
# ============================================================================

def generate_model_configs() -> List[ModelConfig]:
    """Generate all model configurations for screening"""
    
    configs = []
    
    # 1. Robust type variations
    robust_variations = [
        ('standard_bayes', 'standard', None, None),
        ('epsilon_005', 'epsilon', 0.05, None),
        ('epsilon_010', 'epsilon', 0.10, None),
        ('epsilon_015', 'epsilon', 0.15, None),
        ('epsilon_020', 'epsilon', 0.20, None),
        ('student_t3', 'student_t', None, 3),
        ('student_t5', 'student_t', None, 5),
        ('student_t10', 'student_t', None, 10),
        ('huber_10', 'huber', None, None),
        ('huber_15', 'huber', None, None),
        ('mixture_2', 'mixture', None, None),
    ]
    
    # 2. Prior variations
    prior_types = ['weakly_informative', 'informative', 'horseshoe']
    
    # 3. Likelihood complexity
    likelihood_types = ['simple', 'zero_inflated']
    
    # 4. Feature engineering
    feature_types = ['base', 'polynomial', 'interaction']
    
    # Generate combinations (selective, not all)
    for robust_name, robust_type, epsilon, nu in robust_variations:
        for prior_type in prior_types:
            for likelihood in likelihood_types:
                # Skip some invalid combinations
                if robust_type == 'mixture' and likelihood == 'zero_inflated':
                    continue
                    
                # Add base features version
                name = f"{robust_name}_{prior_type[:4]}_{likelihood[:4]}_base"
                configs.append(ModelConfig(
                    name=name,
                    robust_type=robust_type,
                    epsilon=epsilon,
                    nu=nu,
                    prior_type=prior_type,
                    likelihood_complexity=likelihood,
                    feature_engineering='base'
                ))
                
                # Add one enhanced feature version for promising models
                if robust_type in ['epsilon', 'student_t'] and prior_type == 'weakly_informative':
                    name_poly = f"{robust_name}_{prior_type[:4]}_{likelihood[:4]}_poly"
                    configs.append(ModelConfig(
                        name=name_poly,
                        robust_type=robust_type,
                        epsilon=epsilon,
                        nu=nu,
                        prior_type=prior_type,
                        likelihood_complexity=likelihood,
                        feature_engineering='polynomial'
                    ))
    
    print(f"\n📋 Generated {len(configs)} model configurations")
    
    return configs

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    
    start_time = time.time()
    
    # Setup paths
    results_dir = Path('results/vi_mcmc_framework')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_analysis_data()
    print(f"  📊 Data shape: X={data['X'].shape}, y={data['y'].shape}")
    
    # Generate model configurations
    configs = generate_model_configs()
    
    # ========== Phase 1: VI Screening ==========
    print("\n" + "="*60)
    print("📊 PHASE 1: Variational Inference Screening")
    print("="*60)
    
    vi_results = parallel_vi_screening(
        configs, 
        data, 
        n_jobs=min(len(configs), N_CORES-2)  # Use almost all cores
    )
    
    # Save VI results
    vi_results.to_csv(results_dir / 'vi_screening_results.csv', index=False)
    
    # Display top models
    print("\n🏆 Top 10 Models by ELBO:")
    print(vi_results[['model_name', 'robust_type', 'elbo', 'converged', 'time_seconds']].head(10))
    
    # Analyze results by type
    print("\n📈 Average ELBO by Robust Type:")
    elbo_by_type = vi_results.groupby('robust_type')['elbo'].agg(['mean', 'std', 'max'])
    print(elbo_by_type.sort_values('mean', ascending=False))
    
    # ========== Phase 2: MCMC Validation ==========
    print("\n" + "="*60)
    print("🔥 PHASE 2: MCMC Validation of Top Models")
    print("="*60)
    
    # Select top models for MCMC
    top_k = 5
    
    # Ensure we have at least one of each promising type
    top_models_idx = []
    
    # Get best epsilon model
    best_epsilon = vi_results[vi_results['robust_type'] == 'epsilon'].head(1).index
    if len(best_epsilon) > 0:
        top_models_idx.append(best_epsilon[0])
    
    # Get best standard model
    best_standard = vi_results[vi_results['robust_type'] == 'standard'].head(1).index
    if len(best_standard) > 0:
        top_models_idx.append(best_standard[0])
    
    # Get best student_t model
    best_student = vi_results[vi_results['robust_type'] == 'student_t'].head(1).index
    if len(best_student) > 0:
        top_models_idx.append(best_student[0])
    
    # Fill remaining slots with top overall
    remaining_slots = top_k - len(top_models_idx)
    for idx in vi_results.head(top_k + 3).index:
        if idx not in top_models_idx and remaining_slots > 0:
            top_models_idx.append(idx)
            remaining_slots -= 1
    
    # Create filtered VI results for MCMC
    vi_results_filtered = vi_results.loc[top_models_idx[:top_k]]
    vi_results_filtered.attrs['full_results'] = vi_results.attrs.get('full_results', [])
    
    print(f"\n🎯 Selected models for MCMC validation:")
    print(vi_results_filtered[['model_name', 'robust_type', 'elbo']])
    
    # Run MCMC
    mcmc_results = run_mcmc_top_models(vi_results_filtered, data, top_k=top_k)
    
    # ========== Phase 3: Comparison and Visualization ==========
    print("\n" + "="*60)
    print("📊 PHASE 3: Model Comparison and Analysis")
    print("="*60)
    
    # Create comparison table
    comparison_df = comprehensive_model_comparison(vi_results_filtered, mcmc_results)
    
    # Save comparison results
    comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
    
    # Display final results
    print("\n🏆 Final Model Ranking:")
    print(comparison_df[['model', 'robust_type', 'vi_elbo', 'mcmc_waic', 
                         'mcmc_divergences', 'combined_score']])
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    plot_vi_screening_results(vi_results, results_dir)
    plot_mcmc_comparison(comparison_df, results_dir)
    
    # Save full results
    print("\n💾 Saving complete results...")
    
    with open(results_dir / 'complete_results.pkl', 'wb') as f:
        pickle.dump({
            'vi_results': vi_results,
            'mcmc_results': mcmc_results,
            'comparison': comparison_df,
            'configs': configs,
            'data_info': {
                'n_samples': data['n_samples'],
                'n_features': data['n_features']
            }
        }, f)
    
    # ========== Summary ==========
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    
    print(f"\n📊 Summary Statistics:")
    print(f"  • Models screened (VI): {len(configs)}")
    print(f"  • Models validated (MCMC): {len(mcmc_results)}")
    print(f"  • Total computation time: {total_time/60:.1f} minutes")
    print(f"  • VI screening time: {vi_results['time_seconds'].sum()/60:.1f} minutes")
    if len(mcmc_results) > 0:
        mcmc_time = sum(r['elapsed_time'] for r in mcmc_results.values())
        print(f"  • MCMC validation time: {mcmc_time/60:.1f} minutes")
        print(f"  • Time saved vs. full MCMC: {(len(configs)-len(mcmc_results))*mcmc_time/len(mcmc_results)/60:.1f} minutes")
    
    print(f"\n🏆 Best Model: {comparison_df.iloc[0]['model']}")
    print(f"  • Robust type: {comparison_df.iloc[0]['robust_type']}")
    print(f"  • VI ELBO: {comparison_df.iloc[0]['vi_elbo']:.2f}")
    print(f"  • MCMC WAIC: {comparison_df.iloc[0]['mcmc_waic']:.2f}")
    
    # Check if epsilon-contamination wins
    if 'epsilon' in comparison_df.iloc[0]['robust_type']:
        print("\n🎉 ε-contamination model demonstrates superior performance!")
    
    print(f"\n📁 Results saved to: {results_dir}")
    
    return {
        'vi_results': vi_results,
        'mcmc_results': mcmc_results,
        'comparison': comparison_df
    }

if __name__ == "__main__":
    results = main()