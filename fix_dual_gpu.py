#!/usr/bin/env python3
"""
Dual GPU Fix for JAX/NumPyro
雙GPU修復 - JAX/NumPyro多設備並行

Forces JAX to use both RTX A5000 GPUs simultaneously
強制JAX同時使用兩個RTX A5000 GPU
"""

import os

def configure_dual_gpu_environment():
    """配置雙GPU環境變數"""
    
    print("🔧 Configuring Dual RTX A5000 GPU Environment...")
    
    # JAX雙GPU強制配置
    dual_gpu_env = {
        # 關鍵：強制JAX識別多個設備
        'JAX_PLATFORMS': 'cuda',  # 只用CUDA，不要CPU fallback
        'JAX_ENABLE_X64': 'False',
        'JAX_PLATFORM_NAME': 'gpu',
        
        # 關鍵：多設備並行配置
        'XLA_FLAGS': '--xla_force_host_platform_device_count=2',  # 強制2個設備
        'CUDA_VISIBLE_DEVICES': '0,1',  # 確保兩個GPU都可見
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        
        # 記憶體和並行配置
        'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.8',  # 稍微減少避免OOM
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'true',
        'XLA_PYTHON_CLIENT_ALLOCATOR': 'platform',
        
        # NumPyro多設備配置
        'NUMPYRO_PLATFORM': 'gpu',
        'NUMPYRO_NUM_CHAINS': '24',  # 明確指定鏈數
        
        # PyTensor強制CUDA
        'PYTENSOR_FLAGS': 'device=cuda,floatX=float32,optimizer=fast_run,force_device=True',
        'THEANO_FLAGS': 'device=cuda,floatX=float32,force_device=True',
        
        # 多線程配置
        'OMP_NUM_THREADS': '16',
        'MKL_NUM_THREADS': '16',
        'OPENBLAS_NUM_THREADS': '16',
    }
    
    for key, value in dual_gpu_env.items():
        os.environ[key] = value
        print(f"   ✅ {key} = {value}")
    
    return dual_gpu_env

def test_dual_gpu_detection():
    """測試雙GPU檢測"""
    
    print("\n🧪 Testing Dual GPU Detection...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        devices = jax.devices()
        print(f"📊 JAX devices detected: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"   Device {i}: {device}")
        
        # 測試多設備計算
        if len(devices) >= 2:
            print("\n🚀 Testing multi-device computation...")
            
            # 在不同GPU上創建數組
            x0 = jax.device_put(jnp.ones((1000, 1000)), devices[0])
            x1 = jax.device_put(jnp.ones((1000, 1000)), devices[1])
            
            print(f"   Array on GPU 0: {x0.device()}")
            print(f"   Array on GPU 1: {x1.device()}")
            
            # 並行計算
            result0 = jnp.sum(x0)
            result1 = jnp.sum(x1)
            
            print(f"   GPU 0 result: {result0} on {result0.device()}")
            print(f"   GPU 1 result: {result1} on {result1.device()}")
            
            print("✅ Dual GPU computation successful!")
            return True
        else:
            print("❌ Only 1 GPU detected - dual GPU not working")
            return False
            
    except ImportError:
        print("❌ JAX not available")
        return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def create_dual_gpu_mcmc_config():
    """創建雙GPU MCMC配置"""
    
    dual_gpu_mcmc_config = {
        # 鏈分配策略：每個GPU 12條鏈
        "n_samples": 2000,       # 適中樣本數
        "n_warmup": 1000,        # 適中warmup
        "n_chains": 24,          # 24條鏈 = 每GPU 12條
        "cores": 24,             # 匹配鏈數
        "target_accept": 0.88,   # 稍低接受率提高速度
        
        # 關鍵：NumPyro多設備參數
        "nuts_sampler": "numpyro",
        "chain_method": "parallel",
        
        # 明確設備分配
        "num_devices": 2,        # 明確指定2個設備
        "chains_per_device": 12, # 每設備12條鏈
        
        # 性能優化
        "progress_bar": True,
        "return_inferencedata": True,
    }
    
    print("\n📊 Dual GPU MCMC Configuration:")
    print("=" * 40)
    for key, value in dual_gpu_mcmc_config.items():
        print(f"   {key}: {value}")
    
    return dual_gpu_mcmc_config

if __name__ == "__main__":
    print("🚀 Dual RTX A5000 GPU Configuration")
    print("=" * 50)
    
    # 1. 配置環境
    env_config = configure_dual_gpu_environment()
    
    # 2. 測試雙GPU
    gpu_working = test_dual_gpu_detection()
    
    # 3. 創建MCMC配置
    mcmc_config = create_dual_gpu_mcmc_config()
    
    # 4. 總結
    print("\n" + "=" * 50)
    if gpu_working:
        print("🎉 Dual GPU configuration successful!")
        print("💡 Next steps:")
        print("   1. Run your analysis with these settings")
        print("   2. Monitor nvidia-smi for dual GPU usage")
        print("   3. Expect 80%+ usage on BOTH GPUs")
    else:
        print("⚠️  Dual GPU configuration needs adjustment")
        print("💡 Troubleshooting:")
        print("   1. Check CUDA_VISIBLE_DEVICES")
        print("   2. Restart Python session")
        print("   3. Verify JAX-CUDA installation")