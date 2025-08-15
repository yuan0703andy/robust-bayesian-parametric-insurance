#!/usr/bin/env python3
"""
Test Dual GPU MCMC Configuration
測試雙GPU MCMC配置

Quick test to verify both RTX A5000 GPUs are being used for MCMC
快速測試驗證兩個RTX A5000 GPU都被用於MCMC
"""

import os
import numpy as np

# 配置雙GPU環境
print("🔧 Configuring Dual GPU Environment...")
dual_gpu_env = {
    'JAX_PLATFORMS': 'cuda',  # 關鍵：只用CUDA
    'JAX_ENABLE_X64': 'False',
    'JAX_PLATFORM_NAME': 'gpu',
    'XLA_FLAGS': '--xla_force_host_platform_device_count=2',  # 關鍵：強制2個設備
    'CUDA_VISIBLE_DEVICES': '0,1',
    'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.7',  # 保守配置避免OOM
    'XLA_PYTHON_CLIENT_PREALLOCATE': 'true',
    'NUMPYRO_PLATFORM': 'gpu',
    'PYTENSOR_FLAGS': 'device=cuda,floatX=float32,optimizer=fast_run,force_device=True'
}

for key, value in dual_gpu_env.items():
    os.environ[key] = value
    print(f"   ✅ {key} = {value}")

print("\n🧪 Testing JAX dual GPU detection...")

try:
    import jax
    import jax.numpy as jnp
    
    devices = jax.devices()
    print(f"📊 JAX devices: {devices}")
    print(f"🎯 JAX backend: {jax.default_backend()}")
    
    gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
    print(f"🔍 GPU devices found: {len(gpu_devices)}")
    
    if len(gpu_devices) >= 2:
        print("✅ Dual GPU detected!")
        
        # 測試雙GPU並行計算
        print("\n🚀 Testing dual GPU computation...")
        x0 = jax.device_put(jnp.ones((500, 500)), devices[0])
        x1 = jax.device_put(jnp.ones((500, 500)), devices[1])
        
        result0 = jnp.sum(x0 * 2.0)
        result1 = jnp.sum(x1 * 3.0)
        
        print(f"   GPU 0 computation: {result0} on {result0.device()}")
        print(f"   GPU 1 computation: {result1} on {result1.device()}")
        print("✅ Dual GPU computation successful!")
        
    else:
        print("❌ Only single GPU detected - dual GPU not working")

except ImportError:
    print("❌ JAX not available")
except Exception as e:
    print(f"❌ GPU test failed: {e}")

print("\n🧪 Testing PyMC + NumPyro dual GPU MCMC...")

try:
    import pymc as pm
    print("✅ PyMC imported")
    
    # 簡單模型測試雙GPU MCMC
    np.random.seed(42)
    test_data = np.random.normal(5.0, 2.0, 100)
    
    print(f"📊 Test data: mean={np.mean(test_data):.2f}, std={np.std(test_data):.2f}")
    
    with pm.Model() as test_model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=test_data)
        
        # 雙GPU MCMC配置
        mcmc_kwargs = {
            "draws": 500,        # 適中樣本數
            "tune": 250,         # 適中warmup
            "chains": 8,         # 8條鏈 = 每GPU 4條
            "cores": 8,
            "nuts_sampler": "numpyro",  # 關鍵：NumPyro
            "chain_method": "parallel", # 關鍵：並行
            "target_accept": 0.85,
            "return_inferencedata": True,
            "progressbar": True
        }
        
        print(f"🚀 Running MCMC with dual GPU configuration...")
        print(f"   Chains: {mcmc_kwargs['chains']}")
        print(f"   Samples: {mcmc_kwargs['draws']}")
        print(f"   Sampler: {mcmc_kwargs['nuts_sampler']}")
        print(f"   Method: {mcmc_kwargs['chain_method']}")
        
        trace = pm.sample(**mcmc_kwargs)
        
        print("✅ Dual GPU MCMC successful!")
        print(f"📊 Posterior mu: {trace.posterior.mu.mean().values:.2f} ± {trace.posterior.mu.std().values:.2f}")
        print(f"📊 Posterior sigma: {trace.posterior.sigma.mean().values:.2f} ± {trace.posterior.sigma.std().values:.2f}")

except Exception as e:
    print(f"❌ PyMC dual GPU test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🔍 Monitor GPU usage during this test:")
print("   Run: watch -n 1 nvidia-smi")
print("   Look for: Both GPUs showing 70%+ usage")
print("   Expected: ~150W+ power on both GPUs")
print("=" * 60)

print("\n💡 If GPU 1 still shows 0% usage:")
print("   1. Restart Python session")
print("   2. Check JAX installation: pip install 'jax[cuda12_pip]'")
print("   3. Verify environment: echo $XLA_FLAGS")
print("   4. Try: export CUDA_VISIBLE_DEVICES=0,1")