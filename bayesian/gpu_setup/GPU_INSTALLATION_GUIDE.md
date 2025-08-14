
# Installation Guide for Dual-GPU MCMC
# 雙GPU MCMC安裝指南

## 1. CUDA Setup (if not already installed)
```bash
# Check CUDA version
nvidia-smi

# Install CUDA toolkit (if needed)
# Follow NVIDIA CUDA installation guide for your system
```

## 2. JAX GPU Installation
```bash
# Install JAX with CUDA support
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Or for CUDA 11:
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## 3. PyMC with JAX Backend
```bash
# Install NumPyro (JAX-based MCMC)
pip install numpyro

# Install GPU utilities for monitoring
pip install gputil
```

## 4. Verification
```python
import jax
print("JAX devices:", jax.devices())
# Should show: [cuda(id=0), cuda(id=1), cpu(id=0)]

import numpyro
print("NumPyro version:", numpyro.__version__)
```

## 5. Environment Setup
Add to your ~/.bashrc or run before analysis:
```bash
export JAX_PLATFORMS=cuda,cpu
export CUDA_VISIBLE_DEVICES=0,1
```
        