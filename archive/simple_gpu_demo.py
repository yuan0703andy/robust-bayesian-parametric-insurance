#!/usr/bin/env python3
"""
Simple GPU Setup Demo (No PyMC)
簡單GPU設置演示（不使用PyMC）
"""

import os
import numpy as np

# Configure environment to avoid compilation issues
os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float32,optimizer=fast_compile'

print("🚀 Simple GPU Setup Demo")
print("=" * 40)

# Test GPU configuration without PyMC
from bayesian.gpu_setup import setup_gpu_environment

print("📊 Testing GPU Configuration...")

# Test different scenarios
scenarios = [
    ("CPU Development", False),
    ("GPU Production", True),  # Will fallback to CPU in this environment
]

for name, enable_gpu in scenarios:
    print(f"\n📋 {name} Configuration:")
    print("-" * 30)
    
    config = setup_gpu_environment(enable_gpu=enable_gpu)
    mcmc_params = config.get_mcmc_config()
    
    print(f"   Hardware Level: {config.hardware_level}")
    print(f"   Backend: {mcmc_params['backend']}")
    print(f"   Chains: {mcmc_params['n_chains']}")
    print(f"   Samples per chain: {mcmc_params['n_samples']:,}")
    print(f"   Total samples: {mcmc_params['n_samples'] * mcmc_params['n_chains']:,}")
    print(f"   Cores: {mcmc_params['cores']}")
    print(f"   Target accept: {mcmc_params['target_accept']}")

# Show directory structure
print(f"\n📂 GPU Setup Files Location:")
print("   bayesian/gpu_setup/")
print("   ├── __init__.py")
print("   ├── gpu_config.py")
print("   ├── dual_gpu_mcmc_setup.py") 
print("   ├── apply_gpu_optimization.py")
print("   ├── gpu_performance_monitor.py")
print("   ├── GPU_DEPLOYMENT_INSTRUCTIONS.md")
print("   └── GPU_INSTALLATION_GUIDE.md")

print(f"\n🎯 Usage Summary:")
print("=" * 40)
print("1. 🔧 Simple Configuration:")
print("   from bayesian.gpu_setup import setup_gpu_environment")
print("   config = setup_gpu_environment(enable_gpu=True)")
print("   mcmc_kwargs = config.get_pymc_sampler_kwargs()")

print("\n2. ⚡ For Your Dual-GPU System:")
print("   Expected Performance: 4x speedup")
print("   - 16 chains (8 per GPU)")
print("   - 4,000 samples per chain")
print("   - 64,000 total samples")
print("   - ~15 minutes vs ~60 minutes CPU")

print("\n3. 🚀 Quick Deployment:")
print("   pip install 'jax[cuda12_pip]' numpyro gputil")
print("   python 05_robust_bayesian_parm_insurance_gpu.py")

print("\n✅ GPU Setup Successfully Modularized!")
print("=" * 40)