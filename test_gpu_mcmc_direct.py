#!/usr/bin/env python3
"""
Direct GPU MCMC Test - 直接GPU MCMC測試
測試PyMC + NumPyro是否能正確使用雙GPU

This script tests if PyMC + NumPyro can correctly use dual GPUs
"""

import os
import numpy as np

# 設置環境變數 - 必須在import JAX之前
print("🔧 Setting up GPU environment...")
gpu_env = {
    'JAX_PLATFORMS': 'cuda',
    'JAX_PLATFORM_NAME': 'gpu', 
    'XLA_FLAGS': '--xla_force_host_platform_device_count=2',
    'CUDA_VISIBLE_DEVICES': '0,1',
    'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.7',
    'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
    'NUMPYRO_PLATFORM': 'gpu'
}

for key, value in gpu_env.items():
    os.environ[key] = value
    print(f"   ✅ {key} = {value}")

# 測試JAX GPU檢測
print("\n🔍 Testing JAX GPU detection...")
try:
    import jax
    devices = jax.devices()
    gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
    
    print(f"   📊 JAX devices: {devices}")
    print(f"   🎯 GPU devices: {gpu_devices}")
    
    if len(gpu_devices) >= 2:
        print("   ✅ Dual GPU detected by JAX!")
    else:
        print("   ❌ JAX not detecting dual GPU")
        
except Exception as e:
    print(f"   ❌ JAX test failed: {e}")

# 測試PyMC + NumPyro GPU
print("\n🧪 Testing PyMC + NumPyro GPU sampling...")
try:
    import pymc as pm
    
    # 簡單測試模型
    np.random.seed(42)
    test_data = np.random.normal(5.0, 2.0, 100)
    
    with pm.Model() as test_model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=test_data)
        
        print("🚀 Running GPU MCMC test...")
        print("   Monitor nvidia-smi for GPU usage!")
        
        # 強制GPU參數
        trace = pm.sample(
            draws=500,           # 小樣本測試
            tune=250,           # 快速測試
            chains=4,           # 4條鏈測試
            cores=4,
            nuts_sampler="numpyro",   # 關鍵：NumPyro GPU
            chain_method="parallel",   # 並行
            target_accept=0.85,
            progressbar=True,
            return_inferencedata=True
        )
        
        print("✅ GPU MCMC test completed!")
        print(f"   mu: {trace.posterior.mu.mean().values:.2f} ± {trace.posterior.mu.std().values:.2f}")
        print(f"   sigma: {trace.posterior.sigma.mean().values:.2f} ± {trace.posterior.sigma.std().values:.2f}")
        
except Exception as e:
    print(f"❌ PyMC GPU test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("🔍 GPU Test Results:")
print("   Check nvidia-smi output during sampling")
print("   Expected: GPU usage > 50% during MCMC")
print("   If GPU usage is 0%, NumPyro is not using GPU")
print("="*60)

print("\n💡 If GPUs are not used:")
print("   1. Check JAX-CUDA installation")
print("   2. Restart Python kernel") 
print("   3. Verify CUDA_VISIBLE_DEVICES")
print("   4. Check PyMC + NumPyro compatibility")