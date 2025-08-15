#!/usr/bin/env python3
"""
Quick GPU Acceleration Test
快速GPU加速測試

Test if JAX and PyMC can actually use your RTX A5000 GPUs
測試JAX和PyMC是否能真正使用你的RTX A5000 GPU
"""

import os
import numpy as np

# Set GPU environment before importing JAX/PyMC
print("🔧 Setting GPU environment...")
os.environ.update({
    'JAX_PLATFORMS': 'cuda,cpu',
    'JAX_ENABLE_X64': 'False',
    'JAX_PLATFORM_NAME': 'gpu',
    'CUDA_VISIBLE_DEVICES': '0,1',
    'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.75',
    'PYTENSOR_FLAGS': 'device=cuda,floatX=float32,optimizer=fast_run'
})

print("🧪 Testing JAX GPU detection...")
try:
    import jax
    import jax.numpy as jnp
    
    devices = jax.devices()
    print(f"✅ JAX devices: {devices}")
    print(f"✅ JAX default backend: {jax.default_backend()}")
    
    # Test GPU computation
    if len(devices) > 1:
        print("\n🚀 Testing GPU computation...")
        x = jnp.ones((1000, 1000))
        
        # Place on GPU 0
        gpu0_device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices()[0]
        x_gpu = jax.device_put(x, gpu0_device)
        result = jnp.sum(x_gpu)
        
        print(f"✅ GPU computation test: {result}")
        print(f"✅ Computation device: {result.device()}")
        
except Exception as e:
    print(f"❌ JAX GPU test failed: {e}")

print("\n🧪 Testing PyMC GPU integration...")
try:
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt
    
    print(f"✅ PyTensor devices: {pytensor.config.device}")
    print(f"✅ PyTensor floatX: {pytensor.config.floatX}")
    
    # Simple PyMC model test
    print("\n🚀 Testing PyMC with GPU...")
    with pm.Model() as simple_model:
        theta = pm.Normal("theta", mu=0, sigma=1)
        obs = pm.Normal("obs", mu=theta, sigma=0.1, observed=np.random.normal(0, 0.1, 100))
        
        # Check if NumPyro is available
        try:
            print("   Testing NumPyro sampler...")
            trace = pm.sample(
                draws=100,
                tune=50, 
                chains=2,
                nuts_sampler="numpyro",
                progressbar=False
            )
            print("   ✅ NumPyro GPU sampling successful!")
            
        except Exception as e:
            print(f"   ⚠️ NumPyro failed: {e}")
            print("   Trying default sampler...")
            trace = pm.sample(
                draws=100,
                tune=50,
                chains=2, 
                progressbar=False
            )
            print("   ✅ Default sampling successful")
            
except Exception as e:
    print(f"❌ PyMC test failed: {e}")

print("\n🔍 GPU Status Check...")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                           '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpu_stats = result.stdout.strip().split('\n')
        for i, stats in enumerate(gpu_stats):
            util, mem, power = stats.split(', ')
            print(f"🎯 RTX A5000 #{i}: {util}% GPU, {mem}MB memory, {power}W power")
except Exception as e:
    print(f"⚠️ GPU status check failed: {e}")

print("\n✅ GPU acceleration test complete!")
print("If you see high GPU utilization during PyMC sampling, GPU acceleration is working.")
print("If GPU utilization stays at 0%, there's a configuration issue.")