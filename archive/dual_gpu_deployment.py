
# Dual-GPU PyMC Configuration for Your System
# 雙GPU PyMC配置

import os
import pymc as pm
import numpy as np

# Environment setup (run this first)
def setup_dual_gpu_environment():
    """Setup environment for dual-GPU MCMC"""
    env_vars = {
        'JAX_PLATFORMS': 'cuda,cpu',
        'CUDA_VISIBLE_DEVICES': '0,1',
        'PYTENSOR_FLAGS': 'device=cuda,floatX=float32,optimizer=fast_run',
        'OMP_NUM_THREADS': '8',
        'MKL_NUM_THREADS': '8',
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("✅ Dual-GPU environment configured")

# Optimized sampler configuration
DUAL_GPU_SAMPLER_KWARGS = {
    "draws": 4000,
    "tune": 2000,
    "chains": 16,
    "cores": 32,
    "target_accept": 0.95,
    "nuts_sampler": "numpyro",
    "chain_method": "parallel",
    "progress_bar": True,
    "compute_convergence_checks": True,
    "return_inferencedata": True,
    "max_treedepth": 12,
    "init": "advi+adapt_diag",
}

# Usage example
def run_optimized_mcmc(model):
    """
    Run MCMC with dual-GPU optimization
    使用雙GPU優化運行MCMC
    """
    
    setup_dual_gpu_environment()
    
    print("🚀 Starting dual-GPU MCMC sampling...")
    print(f"   📊 Configuration: {DUAL_GPU_SAMPLER_KWARGS['chains']} chains, {DUAL_GPU_SAMPLER_KWARGS['draws']} samples each")
    print(f"   ⚡ Expected speedup: 4x over CPU")
    
    with model:
        trace = pm.sample(**DUAL_GPU_SAMPLER_KWARGS)
    
    return trace

# Integration with your robust Bayesian analysis
def integrate_with_bayesian_analysis():
    """
    Integration example for your 05_robust_bayesian_parm_insurance.py
    """
    
    # Replace your current MCMC config with:
    mcmc_config = MCMCConfig(
        n_samples=4000,
        n_warmup=2000,
        n_chains=16,
        cores=32,
        target_accept=0.95,
        backend='jax'  # Use JAX backend for GPU
    )
    
    # Use in your hierarchical model
    hierarchical_model = ParametricHierarchicalModel(model_spec, mcmc_config)
    
    return hierarchical_model
        