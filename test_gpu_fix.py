#!/usr/bin/env python3
"""
測試 GPU 修正
Test GPU Fix

驗證 PyTensor CUDA 後端修正是否成功
"""

import os
import sys

print("🔧 測試 GPU 配置修正...")
print("=" * 50)

# 測試 1: GPU setup 模組
print("\n1️⃣ 測試 GPU setup 模組...")
try:
    from bayesian.gpu_setup import setup_gpu_environment
    print("✅ GPU setup 模組載入成功")
    
    # 測試配置
    gpu_config = setup_gpu_environment(enable_gpu=True)
    print(f"✅ GPU 配置成功: {gpu_config.hardware_level}")
    
except Exception as e:
    print(f"❌ GPU setup 錯誤: {e}")
    import traceback
    traceback.print_exc()

# 測試 2: PyMC + JAX
print("\n2️⃣ 測試 PyMC + JAX...")
try:
    import pymc as pm
    print(f"✅ PyMC 版本: {pm.__version__}")
    
    # 檢查 JAX
    import jax
    print(f"✅ JAX 版本: {jax.__version__}")
    print(f"✅ JAX 設備: {jax.devices()}")
    print(f"✅ JAX 後端: {jax.default_backend()}")
    
    # 簡單測試 NumPyro 採樣器
    print("\n   測試 NumPyro 採樣器...")
    import numpy as np
    
    # 創建簡單模型
    with pm.Model() as test_model:
        x = pm.Normal('x', mu=0, sigma=1)
        y = pm.Normal('y', mu=x, sigma=1, observed=[1.0, 2.0, 1.5])
        
        # 嘗試使用 NumPyro
        trace = pm.sample(
            draws=100,
            tune=50,
            chains=1,
            nuts_sampler='numpyro',
            progressbar=False
        )
    
    print("✅ NumPyro 採樣成功！")
    
except Exception as e:
    print(f"❌ PyMC/JAX 測試失敗: {e}")
    import traceback
    traceback.print_exc()

# 測試 3: 環境變數檢查
print("\n3️⃣ 環境變數檢查...")
env_vars = ['JAX_PLATFORM_NAME', 'JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES', 'PYTENSOR_FLAGS']
for var in env_vars:
    value = os.environ.get(var, 'not set')
    print(f"   {var}: {value}")

print("\n" + "=" * 50)
print("🎉 GPU 修正測試完成！")
print("\n💡 說明:")
print("   • PyTensor CUDA 後端已移除 (正常)")
print("   • 現在使用 JAX + NumPyro 進行 GPU 加速")
print("   • nuts_sampler='numpyro' 是關鍵參數")
print("=" * 50)