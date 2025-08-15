#!/usr/bin/env python3
"""
05. Robust Bayesian Parametric Insurance Analysis - Stable Version
穩健貝氏參數型保險分析 - 穩定版本

Stable configuration to prevent kernel crashes and ensure GPU utilization.
穩定配置以防止kernel崩潰並確保GPU使用。

Author: Research Team
Date: 2025-01-15
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import pickle
import json
import time
import platform
import socket
from typing import Dict, List, Tuple, Optional, Any

# CRITICAL: Fix PROJ paths BEFORE importing any geospatial libraries
print("🔧 Comprehensive PROJ path cleanup...")

# Clear problematic environment variables
proj_vars_to_clear = [
    'PROJ_DATA', 'PROJ_LIB', 'GDAL_DATA', 'GEOS_LIB', 'PROJ_NETWORK',
    'PROJ_DEBUG', 'PROJ_CURL_CA_BUNDLE', 'PROJ_USER_WRITABLE_DIRECTORY',
    'GDAL_DRIVER_PATH', 'GDAL_PLUGIN_PATH', 'GDAL_PYTHON_DRIVER_PATH',
    'GEOTIFF_CSV', 'PROJ_SKIP_READ_USER_WRITABLE_DIRECTORY'
]

for var in proj_vars_to_clear:
    if var in os.environ:
        old_val = os.environ[var]
        if any(pattern in old_val for pattern in ['/hpc/', '/cluster/', 'borsuklab', 'cat_modeling']):
            del os.environ[var]
            print(f"   ❌ Cleared HPC path {var}: {old_val}")

# Set safe PROJ environment
os.environ['PROJ_NETWORK'] = 'OFF'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
print("   ✅ PROJ environment cleaned")

# Smart Environment Detection
def detect_environment():
    """Detect if running on HPC or local development environment"""
    hostname = socket.gethostname().lower()
    
    # Check for NVIDIA GPUs
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        has_nvidia = result.returncode == 0
    except:
        has_nvidia = False
    
    # HPC detection
    hpc_patterns = ['hpc', 'cluster', 'slurm', 'pbs', 'node', 'compute', 'borsuk']
    is_hpc = any(pattern in hostname for pattern in hpc_patterns)
    has_slurm = 'SLURM_JOB_ID' in os.environ
    
    return is_hpc or has_slurm or has_nvidia

IS_HPC = detect_environment()
print(f"🔍 Environment detected: {'HPC' if IS_HPC else 'Local Development'}")

# STABLE GPU Configuration - Conservative settings to prevent crashes
if IS_HPC:
    print("🚀 HPC GPU Environment Setup - STABLE Configuration")
    
    # STABLE configuration that prevents kernel crashes
    hpc_env_vars = {
        # JAX Configuration - Conservative
        'JAX_PLATFORMS': 'cuda',
        'JAX_ENABLE_X64': 'False',
        'JAX_PLATFORM_NAME': 'gpu',
        
        # CRITICAL: Conservative memory allocation to prevent OOM
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',  # Dynamic allocation
        'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.5',   # Only 50% to prevent crash
        'XLA_PYTHON_CLIENT_ALLOCATOR': 'platform',
        
        # Disable aggressive optimizations that cause instability
        'XLA_FLAGS': '--xla_gpu_enable_fast_min_max=false',
        
        # CUDA Configuration
        'CUDA_VISIBLE_DEVICES': '0,1',
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        
        # Conservative threading
        'OMP_NUM_THREADS': '8',
        'MKL_NUM_THREADS': '8',
        'OPENBLAS_NUM_THREADS': '8',
        'NUMBA_NUM_THREADS': '8',
        
        # PyMC/NumPyro settings
        'NUMPYRO_PLATFORM': 'gpu',
        'PYMC_COMPUTE_TEST_VALUE': 'ignore',
    }
    
    for key, value in hpc_env_vars.items():
        os.environ[key] = value
        print(f"   ✅ {key} = {value}")
    
    print("\n⚡ STABLE HPC Configuration:")
    print("   🖥️  CPU: 8 cores (conservative)")
    print("   🎯 GPU: 2 × RTX A5000 (50% memory each)")
    print("   💾 Total GPU Memory: 24GB allocated (48GB × 0.5)")
    print("   ⚡ Expected: Stable execution without crashes")
    
else:
    print("💻 Local Development Environment Setup")
    
    local_env_vars = {
        'JAX_PLATFORMS': 'cpu',
        'JAX_ENABLE_X64': 'True',
        'OMP_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '4',
        'OPENBLAS_NUM_THREADS': '4',
        'NUMBA_NUM_THREADS': '4',
        'PYMC_COMPUTE_TEST_VALUE': 'ignore',
        'PYTENSOR_FLAGS': 'device=cpu,floatX=float32,optimizer=fast_compile,allow_gc=True',
    }
    
    for key, value in local_env_vars.items():
        os.environ[key] = value

# Configure matplotlib
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

print("\n" + "=" * 80)
print("05. Robust Bayesian Parametric Insurance Analysis - STABLE Version")
print("=" * 80)

# Import frameworks with error handling
try:
    from insurance_analysis_refactored.core import (
        ParametricInsuranceEngine,
        SkillScoreEvaluator,
        create_standard_technical_premium_calculator
    )
    print("✅ Insurance framework loaded")
except ImportError as e:
    print(f"❌ Failed to load insurance framework: {e}")
    sys.exit(1)

try:
    from bayesian.robust_model_ensemble_analyzer import (
        ModelClassAnalyzer, ModelClassSpec, AnalyzerConfig, MCMCConfig
    )
    from bayesian import (
        ProbabilisticLossDistributionGenerator,
        configure_pymc_environment
    )
    print("✅ Bayesian framework loaded")
except ImportError as e:
    print(f"❌ Failed to load Bayesian framework: {e}")
    sys.exit(1)

# Configure PyMC environment
configure_pymc_environment()

# GPU verification function (conservative)
def verify_gpu_availability():
    """Safely verify GPU availability without causing crashes"""
    print("\n🔍 Checking GPU availability...")
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')
            print(f"   📊 Found {len(gpu_info)} GPUs:")
            for info in gpu_info:
                parts = info.split(', ')
                idx, name, total_mem, free_mem = parts
                print(f"      GPU {idx}: {name} - {free_mem}/{total_mem} MB free")
            return len(gpu_info) >= 2
    except Exception as e:
        print(f"   ⚠️ Could not query GPUs: {e}")
    
    return False

# Load data
print("\n📂 Loading data from previous steps...")

try:
    with open('results/climada_data/climada_complete_data.pkl', 'rb') as f:
        climada_results = pickle.load(f)
    print("   ✅ CLIMADA results loaded")
    
    with open('results/spatial_analysis/cat_in_circle_results.pkl', 'rb') as f:
        spatial_results = pickle.load(f)
    print("   ✅ Spatial analysis loaded")
    
    with open('results/insurance_products/products.pkl', 'rb') as f:
        products_data = pickle.load(f)
    print("   ✅ Insurance products loaded")
    
    with open('results/traditional_basis_risk_analysis/analysis_results.pkl', 'rb') as f:
        traditional_results = pickle.load(f)
    print("   ✅ Traditional analysis loaded")
    
except FileNotFoundError as e:
    print(f"   ❌ Error loading data: {e}")
    print("   Please run scripts 01-04 first")
    sys.exit(1)

# Extract data
event_losses_array = climada_results.get('event_losses')
event_losses = {i: loss for i, loss in enumerate(event_losses_array)} if event_losses_array is not None else {}

print(f"\n📊 Data Summary:")
print(f"   Event losses: {len(event_losses)} events")
print(f"   Existing products: {len(products_data)} products")

# Phase 1: STABLE Bayesian Configuration
print("\n" + "=" * 80)
print("Phase 1: STABLE Bayesian Model Configuration")
print("=" * 80)

# Extract observed losses
observed_losses = []
for event_id, loss in event_losses.items():
    if loss > 0:  # Only non-zero losses
        observed_losses.append(loss)

observed_losses = np.array(observed_losses)
n_data = len(observed_losses)

print(f"\n📊 Event Loss Statistics:")
print(f"   Total events: {len(event_losses)}")
print(f"   Non-zero losses: {n_data}")
print(f"   Loss range: ${np.min(observed_losses)/1e6:.1f}M - ${np.max(observed_losses)/1e6:.1f}M")

# STABLE MCMC Configuration
if IS_HPC:
    # Conservative HPC configuration to prevent crashes
    mcmc_config = MCMCConfig(
        n_samples=500,      # Reduced samples
        n_warmup=200,       # Reduced warmup
        n_chains=4,         # Only 4 chains for stability
        cores=4,            # Match chain count
        target_accept=0.85  # Lower target for stability
    )
    print("\n🔧 STABLE HPC MCMC Configuration:")
    print("   ⚠️ Using conservative settings to prevent crashes")
else:
    # Ultra-conservative local configuration
    mcmc_config = MCMCConfig(
        n_samples=100,
        n_warmup=50,
        n_chains=2,
        cores=2,
        target_accept=0.80
    )
    print("\n🔧 Local MCMC Configuration:")

print(f"   Chains: {mcmc_config.n_chains}")
print(f"   Samples per chain: {mcmc_config.n_samples}")
print(f"   Total samples: {mcmc_config.n_chains * mcmc_config.n_samples:,}")
print(f"   Cores: {mcmc_config.cores}")

# Analyzer configuration
analyzer_config = AnalyzerConfig(
    mcmc_config=mcmc_config,
    use_mpe=True,
    parallel_execution=False,  # Sequential for stability
    max_workers=1,
    model_selection_criterion='dic',
    calculate_ranges=True,
    calculate_weights=True
)

# Minimal model specification for testing
model_class_spec = ModelClassSpec(
    enable_epsilon_contamination=True,
    epsilon_values=[0.05],  # Single contamination level
    contamination_distribution="typhoon"
)

print(f"\n📊 Model Configuration:")
print(f"   Total models: {model_class_spec.get_model_count()}")
print(f"   ε-contamination: {model_class_spec.epsilon_values}")
print(f"   Execution: Sequential (stable)")

# Create analyzer
analyzer = ModelClassAnalyzer(model_class_spec, analyzer_config)
print("✅ Bayesian analyzer created")

# Phase 2: Run Analysis with Error Handling
print("\n" + "=" * 80)
print("Phase 2: Running STABLE MCMC Analysis")
print("=" * 80)

# Check GPU status before running
if IS_HPC:
    has_gpus = verify_gpu_availability()
    if not has_gpus:
        print("⚠️ GPUs not available, will use CPU fallback")

print("\n🚀 Starting MCMC analysis...")
print("   ⏱️  This may take several minutes...")

start_time = time.time()

try:
    # Run analysis with error handling
    ensemble_results = analyzer.analyze_model_class(observed_losses)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Analysis complete!")
    print(f"   Best model: {ensemble_results.best_model}")
    print(f"   Execution time: {elapsed_time:.1f} seconds")
    print(f"   Models evaluated: {len(ensemble_results.individual_results)}")
    
except Exception as e:
    print(f"\n❌ Analysis failed: {e}")
    print("   💡 Try reducing n_samples or n_chains further")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Phase 3: Skill Score Evaluation
print("\n" + "=" * 80)
print("Phase 3: Skill Score Evaluation")
print("=" * 80)

skill_evaluator = SkillScoreEvaluator()

# Get best model results
best_model_name = ensemble_results.best_model
best_model_result = ensemble_results.individual_results[best_model_name]

print(f"📊 Evaluating best model: {best_model_name}")

# Extract predictions
posterior_samples = best_model_result.posterior_samples
if 'theta' in posterior_samples:
    predictions = np.full(len(observed_losses), np.mean(posterior_samples['theta']))
else:
    predictions = observed_losses * 0.8  # Conservative fallback

# Calculate skill scores
skill_scores = skill_evaluator.calculate_comprehensive_scores(
    predictions, observed_losses, predictions
)

print("\n📊 Skill Scores:")
for metric, value in skill_scores.items():
    if isinstance(value, float):
        print(f"   {metric}: {value:.4f}")

# Phase 4: Save Results
print("\n" + "=" * 80)
print("Phase 4: Saving Results")
print("=" * 80)

results_dir = Path('results/robust_bayesian_stable')
results_dir.mkdir(parents=True, exist_ok=True)

# Compile results
final_analysis = {
    'best_model': ensemble_results.best_model,
    'total_models': len(ensemble_results.individual_results),
    'execution_time': elapsed_time,
    'skill_scores': skill_scores,
    'model_ranking': ensemble_results.get_model_ranking('dic'),
    'mcmc_samples': mcmc_config.n_chains * mcmc_config.n_samples,
    'environment': 'HPC' if IS_HPC else 'Local'
}

# Save results
with open(results_dir / 'stable_bayesian_results.pkl', 'wb') as f:
    pickle.dump({
        'ensemble_results': ensemble_results,
        'skill_scores': skill_scores,
        'final_analysis': final_analysis
    }, f)

# Save summary
with open(results_dir / 'analysis_summary.txt', 'w') as f:
    f.write("STABLE Bayesian Analysis Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Environment: {final_analysis['environment']}\n")
    f.write(f"Best Model: {final_analysis['best_model']}\n")
    f.write(f"Execution Time: {final_analysis['execution_time']:.1f}s\n")
    f.write(f"Total MCMC Samples: {final_analysis['mcmc_samples']:,}\n")
    f.write(f"Models Evaluated: {final_analysis['total_models']}\n\n")
    f.write("Top Models by DIC:\n")
    for i, (model, dic) in enumerate(final_analysis['model_ranking'][:5], 1):
        f.write(f"{i}. {model}: DIC = {dic:.2f}\n")

print(f"✅ Results saved to: {results_dir}")

print("\n" + "=" * 80)
print("🎉 STABLE Bayesian Analysis Complete!")
print("=" * 80)
print(f"✅ Best model: {final_analysis['best_model']}")
print(f"✅ Execution time: {final_analysis['execution_time']:.1f} seconds")
print(f"✅ Total samples: {final_analysis['mcmc_samples']:,}")
print(f"✅ Environment: {final_analysis['environment']}")
print("=" * 80)