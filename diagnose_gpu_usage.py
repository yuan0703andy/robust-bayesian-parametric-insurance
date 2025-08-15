#!/usr/bin/env python3
"""
診斷 GPU 使用狀況
Diagnose GPU Usage Status

檢查 PyMC/JAX 是否真正使用 GPU 加速
Check if PyMC/JAX is actually using GPU acceleration
"""

import os
import sys

# 🔥 關鍵：在導入 PyMC 之前設置環境變數！
print("🔧 Setting GPU environment BEFORE importing PyMC...")
os.environ['PYTENSOR_FLAGS'] = 'device=cuda,floatX=float32,optimizer=fast_run,allow_gc=True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
print("✅ Environment variables set")

import time
import numpy as np

print("=" * 80)
print("🔍 GPU 使用狀況診斷")
print("=" * 80)

# 1. 檢查環境變數
print("\n📋 Step 1: 檢查環境變數")
print("-" * 40)
gpu_env_vars = {
    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
    'JAX_PLATFORM_NAME': os.environ.get('JAX_PLATFORM_NAME', 'not set'),
    'JAX_PLATFORMS': os.environ.get('JAX_PLATFORMS', 'not set'),
    'PYTENSOR_FLAGS': os.environ.get('PYTENSOR_FLAGS', 'not set'),
}
for var, value in gpu_env_vars.items():
    print(f"   {var}: {value}")

# 2. 檢查 JAX GPU 狀態
print("\n📋 Step 2: 檢查 JAX GPU 狀態")
print("-" * 40)
try:
    import jax
    import jax.numpy as jnp
    
    print(f"   JAX 版本: {jax.__version__}")
    print(f"   JAX 後端: {jax.default_backend()}")
    print(f"   JAX 設備: {jax.devices()}")
    
    # 測試簡單計算
    x = jnp.array(np.random.randn(1000, 1000))
    y = jnp.array(np.random.randn(1000, 1000))
    
    start = time.time()
    z = jnp.matmul(x, y)
    z.block_until_ready()  # 等待計算完成
    elapsed = time.time() - start
    
    print(f"   矩陣乘法測試 (1000x1000): {elapsed:.4f} 秒")
    print(f"   計算設備: {z.devices()}")
    
    if 'gpu' in jax.default_backend().lower() or any('cuda' in str(d).lower() for d in jax.devices()):
        print("   ✅ JAX 正在使用 GPU")
    else:
        print("   ❌ JAX 沒有使用 GPU")
        
except Exception as e:
    print(f"   ❌ JAX 測試失敗: {e}")

# 3. 檢查 PyMC/PyTensor GPU 狀態
print("\n📋 Step 3: 檢查 PyMC/PyTensor GPU 狀態")
print("-" * 40)
try:
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt
    
    print(f"   PyMC 版本: {pm.__version__}")
    print(f"   PyTensor 版本: {pytensor.__version__}")
    
    # 檢查 PyTensor 設備
    from pytensor.configdefaults import config
    print(f"   PyTensor device: {config.device}")
    print(f"   PyTensor floatX: {config.floatX}")
    print(f"   PyTensor optimizer: {config.optimizer}")
    
    # 測試簡單模型
    print("\n   測試簡單 PyMC 模型...")
    with pm.Model() as test_model:
        x = pm.Normal('x', mu=0, sigma=1, shape=100)
        y = pm.Normal('y', mu=x, sigma=1, observed=np.random.randn(100))
        
        # 只採樣很少的樣本來測試
        start = time.time()
        trace = pm.sample(
            draws=100,
            tune=50,
            chains=1,
            progressbar=False,
            return_inferencedata=False
        )
        elapsed = time.time() - start
        
    print(f"   PyMC 採樣測試 (100 draws): {elapsed:.2f} 秒")
    
    if config.device == 'cuda':
        print("   ✅ PyTensor 配置為使用 CUDA")
    else:
        print(f"   ⚠️ PyTensor 使用: {config.device}")
        
except Exception as e:
    print(f"   ❌ PyMC/PyTensor 測試失敗: {e}")

# 4. 檢查 GPU 硬體
print("\n📋 Step 4: 檢查 GPU 硬體")
print("-" * 40)

# 嘗試使用 nvidia-ml-py
try:
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"   檢測到 {device_count} 個 NVIDIA GPU")
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        print(f"\n   GPU {i}: {name.decode('utf-8')}")
        print(f"      記憶體: {memory.used/1e9:.1f}/{memory.total/1e9:.1f} GB")
        print(f"      GPU 使用率: {utilization.gpu}%")
        print(f"      記憶體使用率: {utilization.memory}%")
        
    pynvml.nvmlShutdown()
except ImportError:
    print("   ⚠️ pynvml 未安裝，嘗試 GPUtil...")
    
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            print(f"   檢測到 {len(gpus)} 個 GPU")
            for gpu in gpus:
                print(f"\n   GPU {gpu.id}: {gpu.name}")
                print(f"      記憶體: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
                print(f"      GPU 負載: {gpu.load*100:.1f}%")
                print(f"      記憶體使用率: {gpu.memoryUtil*100:.1f}%")
        else:
            print("   ❌ 沒有檢測到 GPU")
    except ImportError:
        print("   ❌ GPUtil 未安裝")
except Exception as e:
    print(f"   ❌ GPU 硬體檢測失敗: {e}")

# 5. 診斷結論
print("\n" + "=" * 80)
print("🎯 診斷結論")
print("=" * 80)

issues = []
recommendations = []

# 檢查 JAX
if 'JAX_PLATFORM_NAME' in os.environ and os.environ['JAX_PLATFORM_NAME'] != 'gpu':
    issues.append("JAX_PLATFORM_NAME 不是設置為 'gpu'")
    recommendations.append("設置 export JAX_PLATFORM_NAME=gpu")

# 檢查 PyTensor
if 'PYTENSOR_FLAGS' in os.environ:
    if 'device=cuda' not in os.environ['PYTENSOR_FLAGS']:
        issues.append("PYTENSOR_FLAGS 沒有包含 device=cuda")
        recommendations.append("確保 PYTENSOR_FLAGS 包含 'device=cuda'")

# 檢查 CUDA
if os.environ.get('CUDA_VISIBLE_DEVICES', 'not set') == 'not set':
    issues.append("CUDA_VISIBLE_DEVICES 未設置")
    recommendations.append("設置 export CUDA_VISIBLE_DEVICES=0,1")

if issues:
    print("❌ 發現的問題:")
    for issue in issues:
        print(f"   • {issue}")
    
    print("\n💡 建議:")
    for rec in recommendations:
        print(f"   • {rec}")
else:
    print("✅ GPU 配置看起來正確")

print("\n📊 性能提示:")
print("   1. PyMC 的 GPU 加速主要對大型模型有效")
print("   2. 小模型可能在 CPU 上更快（GPU 啟動開銷）")
print("   3. 確保使用 NUTS 採樣器（自動選擇）")
print("   4. 考慮使用 JAX 採樣器: pm.sample(nuts_sampler='numpyro')")
print("   5. 對於雙 GPU，使用 chains=2 或 4，cores=2")

print("\n🚀 測試 GPU 加速的最佳方法:")
print("   比較相同模型在 CPU vs GPU 的採樣時間")
print("   export PYTENSOR_FLAGS='device=cpu' vs 'device=cuda'")