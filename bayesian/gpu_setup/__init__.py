#!/usr/bin/env python3
"""
GPU Setup Module for Bayesian MCMC
雙GPU MCMC優化設置模組

This module provides comprehensive GPU optimization tools for PyMC/JAX-based
Bayesian MCMC sampling on dual-GPU systems.

主要組件:
- DualGPU_MCMC_Optimizer: 雙GPU MCMC優化器  
- GPU performance monitoring tools
- Automatic optimization applicators
- Deployment and installation guides

Usage:
------
```python
from bayesian.gpu_setup import DualGPU_MCMC_Optimizer, configure_dual_gpu_environment

# Setup dual-GPU environment
optimizer = DualGPU_MCMC_Optimizer()
optimizer.configure_environment_variables()

# Create optimized MCMC config
config = optimizer.create_dual_gpu_mcmc_config()

# Use in PyMC analysis
with pm.Model() as model:
    # Your model definition
    trace = pm.sample(**config)
```

Expected Performance:
- 3-4x speedup over CPU-only
- 64,000 total samples (vs 8,000 CPU)
- 1 hour analysis time (vs 4 hours CPU)
- Dual-GPU load balancing
"""

from .dual_gpu_mcmc_setup import (
    DualGPU_MCMC_Optimizer
)

from .gpu_config import (
    GPUConfig,
    setup_gpu_environment,
    get_optimized_mcmc_config
)

try:
    from .apply_gpu_optimization import apply_gpu_optimization
except ImportError:
    apply_gpu_optimization = None

try:
    from .gpu_performance_monitor import monitor_gpu_performance
except ImportError:
    monitor_gpu_performance = None

__all__ = [
    'DualGPU_MCMC_Optimizer',
    'GPUConfig',
    'setup_gpu_environment', 
    'get_optimized_mcmc_config',
    'apply_gpu_optimization',
    'monitor_gpu_performance'
]

__version__ = "1.0.0"
__author__ = "Research Team"