#!/usr/bin/env python3
"""
Test Maximum Load Configuration
測試最大負載配置

Quick test with aggressive settings to push GPU usage to 90%+
使用激進設置的快速測試，將GPU使用率推到90%+
"""

import os
import numpy as np
import time

# 設置最大負載環境
print("🔥 Setting Maximum Load Environment...")
max_env = {
    'JAX_PLATFORMS': 'cuda',
    'JAX_ENABLE_X64': 'False',
    'JAX_PLATFORM_NAME': 'gpu',
    'XLA_FLAGS': '--xla_force_host_platform_device_count=2',
    'CUDA_VISIBLE_DEVICES': '0,1',
    'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.95',  # 95% 記憶體
    'XLA_PYTHON_CLIENT_PREALLOCATE': 'true',
    'NUMPYRO_PLATFORM': 'gpu',
    'NUMPYRO_NUM_CHAINS': '32',
    'OMP_NUM_THREADS': '32',
    'PYTENSOR_FLAGS': 'device=cuda,floatX=float32,optimizer=fast_run,force_device=True'
}

for key, value in max_env.items():
    os.environ[key] = value

print("\n🧪 Testing Maximum Load MCMC...")

try:
    import pymc as pm
    
    # 生成較複雜的測試數據
    np.random.seed(42)
    n_data = 500  # 增加數據點
    true_mu = 10.0
    true_sigma = 3.0
    test_data = np.random.normal(true_mu, true_sigma, n_data)
    
    print(f"📊 Test data: {n_data} points, mu={np.mean(test_data):.2f}")
    
    with pm.Model() as max_load_model:
        # 稍微複雜的模型
        mu = pm.Normal("mu", mu=0, sigma=20)
        sigma = pm.HalfNormal("sigma", sigma=10)
        
        # 添加階層效應增加複雜度
        group_effects = pm.Normal("group_effects", mu=0, sigma=5, shape=10)
        group_idx = np.random.randint(0, 10, n_data)
        
        obs_mu = mu + group_effects[group_idx]
        obs = pm.Normal("obs", mu=obs_mu, sigma=sigma, observed=test_data)
        
        # 最大負載MCMC配置
        max_mcmc_config = {
            "draws": 2000,           # 大樣本
            "tune": 1000,            # 充足warmup
            "chains": 32,            # 32條鏈
            "cores": 32,
            "nuts_sampler": "numpyro",
            "chain_method": "parallel",
            "target_accept": 0.95,   # 高精度
            "return_inferencedata": True,
            "progressbar": True
        }
        
        print("🔥 Running Maximum Load MCMC...")
        print(f"   📊 Configuration: {max_mcmc_config['chains']} chains × {max_mcmc_config['draws']} samples")
        print(f"   🎯 Total samples: {max_mcmc_config['chains'] * max_mcmc_config['draws']:,}")
        print("   ⚡ Monitor nvidia-smi for 90%+ usage on both GPUs!")
        
        start_time = time.time()
        trace = pm.sample(**max_mcmc_config)
        end_time = time.time()
        
        elapsed = end_time - start_time
        total_samples = max_mcmc_config['chains'] * max_mcmc_config['draws']
        
        print("🎉 Maximum Load MCMC Completed!")
        print(f"   ⏱️  Time: {elapsed:.1f} seconds")
        print(f"   🚀 Speed: {total_samples/elapsed:.0f} samples/second")
        print(f"   📊 Results:")
        print(f"      mu: {trace.posterior.mu.mean().values:.2f} ± {trace.posterior.mu.std().values:.2f}")
        print(f"      sigma: {trace.posterior.sigma.mean().values:.2f} ± {trace.posterior.sigma.std().values:.2f}")

except Exception as e:
    print(f"❌ Maximum load test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🔥 Maximum Load Test Results:")
print("=" * 60)
print("💡 Check nvidia-smi output:")
print("   ✅ Success: Both GPUs showing 80%+ usage")
print("   ✅ Success: Both GPUs drawing 180W+ power")
print("   ✅ Success: Memory usage 20GB+ per GPU")
print("   ❌ Need tuning: GPU usage still < 70%")
print("\n🎯 If GPUs not fully utilized:")
print("   • Increase n_chains to 48")
print("   • Increase n_samples to 3000+")
print("   • Check for memory bottlenecks")
print("   • Verify JAX using both devices")