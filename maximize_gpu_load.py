#!/usr/bin/env python3
"""
Maximum GPU Load Configuration
最大GPU負載配置

Aggressive MCMC settings to push dual RTX A5000 to 90%+ utilization
激進的MCMC設置推動雙RTX A5000達到90%+使用率
"""

import os
import numpy as np

def setup_maximum_gpu_environment():
    """設置最大GPU負載環境"""
    
    print("🔥 Setting up MAXIMUM GPU Load Environment...")
    
    # 最激進的雙GPU配置
    max_gpu_env = {
        # JAX最大負載配置
        'JAX_PLATFORMS': 'cuda',
        'JAX_ENABLE_X64': 'False',  # float32 for maximum speed
        'JAX_PLATFORM_NAME': 'gpu',
        'XLA_FLAGS': '--xla_force_host_platform_device_count=2 --xla_gpu_force_compilation_parallelism=2',
        
        # 記憶體激進使用
        'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.95',  # 使用95% GPU記憶體
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'true',
        'XLA_PYTHON_CLIENT_ALLOCATOR': 'platform',
        
        # CUDA最大負載
        'CUDA_VISIBLE_DEVICES': '0,1',
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        
        # NumPyro高負載配置
        'NUMPYRO_PLATFORM': 'gpu',
        'NUMPYRO_NUM_CHAINS': '32',  # 增加到32條鏈
        
        # CPU線程最大化
        'OMP_NUM_THREADS': '32',     # 大幅增加
        'MKL_NUM_THREADS': '32',
        'OPENBLAS_NUM_THREADS': '32',
        'NUMBA_NUM_THREADS': '32',
        
        # PyTensor最大化
        'PYTENSOR_FLAGS': 'device=cuda,floatX=float32,optimizer=fast_run,force_device=True,allow_gc=False',
        'THEANO_FLAGS': 'device=cuda,floatX=float32,force_device=True',
    }
    
    for key, value in max_gpu_env.items():
        os.environ[key] = value
        print(f"   🔥 {key} = {value}")
    
    return max_gpu_env

def get_maximum_mcmc_config():
    """獲取最大MCMC配置"""
    
    # 激進配置 - 推動GPU到極限
    max_mcmc_config = {
        "n_samples": 5000,      # 大幅增加樣本數
        "n_warmup": 2500,       # 增加warmup
        "n_chains": 32,         # 增加到32條鏈 (每GPU 16條)
        "cores": 32,            # 匹配鏈數
        "target_accept": 0.95,  # 高精度增加計算量
        
        # GPU優化參數
        "nuts_sampler": "numpyro",
        "chain_method": "parallel",
        
        # 性能參數
        "progress_bar": True,
        "return_inferencedata": True,
        "compute_convergence_checks": True,
    }
    
    print("🔥 MAXIMUM MCMC Configuration:")
    print("=" * 50)
    for key, value in max_mcmc_config.items():
        print(f"   {key}: {value}")
    
    total_samples = max_mcmc_config["n_chains"] * max_mcmc_config["n_samples"]
    print(f"   📊 Total samples: {total_samples:,}")
    print(f"   🎯 Expected GPU load: 90%+ on both GPUs")
    print(f"   ⚡ Expected power: 200W+ per GPU")
    
    return max_mcmc_config

def test_maximum_load():
    """測試最大負載配置"""
    
    print("\n🧪 Testing Maximum GPU Load...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        devices = jax.devices()
        print(f"🔍 JAX devices: {devices}")
        
        if len(devices) >= 2:
            print("🔥 Testing maximum computational load on both GPUs...")
            
            # 創建大型計算任務
            for i, device in enumerate(devices[:2]):
                print(f"   🚀 Loading GPU {i} with heavy computation...")
                
                # 大型矩陣運算
                x = jax.device_put(jnp.ones((2000, 2000)), device)
                
                # 複雜計算鏈
                @jax.jit
                def heavy_computation(x):
                    for _ in range(10):
                        x = jnp.matmul(x, x.T) / jnp.sum(x)
                        x = jnp.sin(x) + jnp.cos(x)
                    return jnp.sum(x)
                
                result = heavy_computation(x)
                print(f"   ✅ GPU {i} computation: {result:.6f} on {result.device()}")
            
            print("🔥 Heavy GPU load test completed!")
        
    except Exception as e:
        print(f"❌ Maximum load test failed: {e}")

if __name__ == "__main__":
    print("🔥 RTX A5000 Maximum GPU Load Configuration")
    print("=" * 60)
    
    # 1. 設置環境
    env_config = setup_maximum_gpu_environment()
    
    # 2. 獲取MCMC配置
    mcmc_config = get_maximum_mcmc_config()
    
    # 3. 測試最大負載
    test_maximum_load()
    
    print("\n" + "=" * 60)
    print("🔥 MAXIMUM LOAD CONFIGURATION READY!")
    print("=" * 60)
    
    print("📊 Expected Performance:")
    print("   🎯 GPU 0: 90%+ usage, 200W+ power")
    print("   🎯 GPU 1: 90%+ usage, 200W+ power")
    print("   ⚡ Total power: 400W+ (both GPUs)")
    print("   💾 Memory usage: ~45GB (95% of 48GB)")
    
    print("\n⚠️  WARNING - Aggressive Configuration:")
    print("   • High memory usage may cause OOM")
    print("   • High power consumption (400W+)")
    print("   • Monitor temperatures")
    print("   • Reduce settings if system unstable")
    
    print("\n🚀 Usage:")
    print("   1. Apply this configuration to your analysis")
    print("   2. Monitor nvidia-smi for 90%+ usage")
    print("   3. Expect single model < 60 seconds")
    print("   4. Watch for thermal throttling")