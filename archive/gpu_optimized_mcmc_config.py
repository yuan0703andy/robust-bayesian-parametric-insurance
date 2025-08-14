#!/usr/bin/env python3
"""
GPU-Optimized MCMC Configuration
GPU優化MCMC配置

針對雙GPU系統的高性能PyMC配置，預期3-4x加速
Optimized for dual-GPU systems with PyMC + JAX backend
"""

import os
import numpy as np
from typing import Optional, Dict, Any
import warnings

def configure_dual_gpu_environment():
    """
    配置雙GPU JAX環境
    Configure dual-GPU JAX environment for maximum performance
    """
    print("🔥 Configuring Dual-GPU JAX Environment...")
    
    # Phase 1: JAX GPU Configuration
    print("\n📱 Phase 1: GPU Detection and JAX Setup")
    
    # Enable JAX GPU support
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # Prioritize CUDA
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Dynamic memory
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'   # Use 80% of GPU memory
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    # JAX threading for dual GPU
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Check available devices
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.device_kind == 'gpu']
        cpu_devices = [d for d in devices if d.device_kind == 'cpu']
        
        print(f"   🎯 Available devices: {len(devices)} total")
        print(f"      📱 GPUs: {len(gpu_devices)} devices")
        print(f"      💻 CPUs: {len(cpu_devices)} devices")
        
        if len(gpu_devices) >= 2:
            print(f"   ✅ Dual-GPU detected: {gpu_devices[0]}, {gpu_devices[1]}")
            return "dual_gpu"
        elif len(gpu_devices) == 1:
            print(f"   ⚠️ Single GPU detected: {gpu_devices[0]}")
            return "single_gpu"
        else:
            print(f"   ❌ No GPU detected, using CPU")
            return "cpu_only"
            
    except ImportError:
        print("   ❌ JAX not available, falling back to PyTensor")
        return "pytensor"
    except Exception as e:
        print(f"   ⚠️ GPU setup warning: {e}")
        return "cpu_fallback"

def configure_pytensor_gpu():
    """
    配置PyTensor GPU後端
    Configure PyTensor for GPU acceleration
    """
    print("\n🔧 Phase 2: PyTensor GPU Configuration")
    
    # PyTensor GPU flags
    gpu_flags = [
        "device=cuda",
        "floatX=float32",
        "optimizer=fast_run",
        "allow_gc=True",
        "scan.allow_gc=True",
        "scan.allow_output_prealloc=True",
        "gpuarray.preallocate=0.8",  # Preallocate 80% GPU memory
        "mode=FAST_RUN"
    ]
    
    os.environ['PYTENSOR_FLAGS'] = ','.join(gpu_flags)
    
    try:
        import pytensor
        print(f"   ✅ PyTensor version: {pytensor.__version__}")
        
        # Test GPU availability
        import pytensor.tensor as pt
        from pytensor.compile import get_default_mode
        
        mode = get_default_mode()
        print(f"   🔧 PyTensor mode: {mode}")
        
        # Check if GPU is available
        if 'cuda' in str(mode).lower() or 'gpu' in str(mode).lower():
            print("   🎯 GPU acceleration enabled")
            return True
        else:
            print("   ⚠️ GPU not detected, using CPU")
            return False
            
    except ImportError:
        print("   ❌ PyTensor not available")
        return False
    except Exception as e:
        print(f"   ⚠️ PyTensor GPU setup failed: {e}")
        return False

def create_optimized_mcmc_config(hardware_level: str = "dual_gpu") -> Dict[str, Any]:
    """
    創建針對硬體優化的MCMC配置
    Create hardware-optimized MCMC configuration
    
    Parameters:
    -----------
    hardware_level : str
        Hardware configuration level
        - "dual_gpu": 雙GPU配置 (最高性能)
        - "single_gpu": 單GPU配置
        - "cpu_only": CPU專用配置
    """
    
    print(f"\n⚡ Phase 3: Creating Optimized MCMC Config for {hardware_level}")
    
    if hardware_level == "dual_gpu":
        config = {
            # MCMC 採樣參數 - 雙GPU優化
            "n_samples": 3000,      # 增加樣本數 (GPU能處理更多)
            "n_warmup": 1500,       # 充分預熱
            "n_chains": 8,          # 更多鏈 (利用雙GPU)
            "cores": 16,            # 更多核心
            "target_accept": 0.95,  # 高接受率
            
            # GPU並行配置
            "chains_per_gpu": 4,    # 每個GPU運行4條鏈
            "parallel_chains": True,
            "gpu_memory_fraction": 0.8,
            
            # JAX優化設定
            "backend": "jax",
            "jax_profile": False,
            "jax_debug": False,
            
            # 進階優化
            "compute_convergence_checks": True,
            "return_inferencedata": True,
            "idata_kwargs": {"log_likelihood": True},
            
            # 預期性能
            "expected_speedup": "3-4x",
            "hardware_utilization": "Dual GPU + 16 cores"
        }
        
    elif hardware_level == "single_gpu":
        config = {
            # MCMC 採樣參數 - 單GPU優化
            "n_samples": 2500,
            "n_warmup": 1000,
            "n_chains": 6,          # 適中的鏈數
            "cores": 12,            # 適中的核心數
            "target_accept": 0.92,
            
            # GPU配置
            "parallel_chains": True,
            "gpu_memory_fraction": 0.7,
            
            # 後端配置
            "backend": "jax",
            "compute_convergence_checks": True,
            "return_inferencedata": True,
            
            # 預期性能
            "expected_speedup": "2-3x",
            "hardware_utilization": "Single GPU + 12 cores"
        }
        
    else:  # CPU only
        config = {
            # MCMC 採樣參數 - CPU優化
            "n_samples": 2000,
            "n_warmup": 1000,
            "n_chains": 4,          # 保守的鏈數
            "cores": 8,             # 適中的核心使用
            "target_accept": 0.90,
            
            # CPU優化
            "parallel_chains": True,
            "compute_convergence_checks": True,
            "return_inferencedata": True,
            
            # 後端配置
            "backend": "pytensor",  # PyTensor for CPU
            
            # 預期性能
            "expected_speedup": "1.5-2x",
            "hardware_utilization": "CPU 8 cores"
        }
    
    print(f"   📊 MCMC Configuration Summary:")
    print(f"      • Samples: {config['n_samples']} per chain")
    print(f"      • Warmup: {config['n_warmup']} samples")
    print(f"      • Chains: {config['n_chains']} parallel")
    print(f"      • Cores: {config['cores']} workers")
    print(f"      • Target Accept: {config['target_accept']}")
    print(f"      • Expected Speedup: {config['expected_speedup']}")
    print(f"      • Hardware: {config['hardware_utilization']}")
    
    return config

def create_pymc_sampler_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    創建PyMC採樣器參數
    Create PyMC sampler kwargs from config
    """
    
    sampler_kwargs = {
        "draws": config["n_samples"],
        "tune": config["n_warmup"],
        "chains": config["n_chains"],
        "cores": config["cores"],
        "target_accept": config["target_accept"],
        "compute_convergence_checks": config.get("compute_convergence_checks", True),
        "return_inferencedata": config.get("return_inferencedata", True),
    }
    
    # JAX特定參數
    if config.get("backend") == "jax":
        sampler_kwargs.update({
            "nuts_sampler": "numpyro",  # Use NumPyro's NUTS
            "chain_method": "parallel" if config.get("parallel_chains") else "sequential",
        })
        
    # idata參數
    if config.get("idata_kwargs"):
        sampler_kwargs["idata_kwargs"] = config["idata_kwargs"]
    
    return sampler_kwargs

def test_gpu_performance():
    """
    測試GPU性能
    Test GPU performance with simple operations
    """
    print("\n🧪 Phase 4: GPU Performance Test")
    
    try:
        import jax
        import jax.numpy as jnp
        import time
        
        # Test matrix operations on GPU
        n = 5000
        print(f"   🔬 Testing {n}x{n} matrix multiplication...")
        
        # Create test data
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (n, n))
        B = jax.random.normal(key, (n, n))
        
        # GPU computation
        start_time = time.time()
        C = jnp.dot(A, B)
        result = C.block_until_ready()  # Wait for computation
        gpu_time = time.time() - start_time
        
        print(f"   ⚡ GPU computation time: {gpu_time:.3f} seconds")
        print(f"   📈 Performance: {(n**3) / (gpu_time * 1e9):.2f} GFLOPS")
        
        # Check for multiple GPUs
        devices = jax.devices("gpu")
        if len(devices) > 1:
            print(f"   🎯 Multiple GPUs available for parallel computation")
            
            # Test multi-GPU
            with jax.default_device(devices[0]):
                A0 = jax.random.normal(key, (n//2, n))
            with jax.default_device(devices[1]):
                A1 = jax.random.normal(key, (n//2, n))
                
            print(f"   ✅ Multi-GPU memory allocation successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GPU performance test failed: {e}")
        return False

def main():
    """
    主要配置流程
    Main configuration workflow
    """
    
    print("🚀 GPU-Optimized MCMC Configuration Setup")
    print("=" * 60)
    
    # Step 1: Detect hardware
    hardware_level = configure_dual_gpu_environment()
    
    # Step 2: Configure backend
    if hardware_level in ["dual_gpu", "single_gpu"]:
        gpu_available = configure_pytensor_gpu()
        if not gpu_available:
            hardware_level = "cpu_fallback"
    
    # Step 3: Create optimized config
    mcmc_config = create_optimized_mcmc_config(hardware_level)
    
    # Step 4: Test performance
    if hardware_level in ["dual_gpu", "single_gpu"]:
        gpu_test_success = test_gpu_performance()
        if not gpu_test_success:
            print("   ⚠️ GPU test failed, consider CPU fallback")
    
    # Step 5: Generate PyMC sampler kwargs
    sampler_kwargs = create_pymc_sampler_kwargs(mcmc_config)
    
    print("\n" + "=" * 60)
    print("🎯 Final Configuration Summary")
    print("=" * 60)
    print(f"Hardware Level: {hardware_level}")
    print(f"Backend: {mcmc_config.get('backend', 'pytensor')}")
    print(f"Expected Speedup: {mcmc_config['expected_speedup']}")
    print("\nPyMC Sample Code:")
    print("```python")
    print("import pymc as pm")
    print("with pm.Model() as model:")
    print("    # Your model definition")
    print("    trace = pm.sample(**sampler_kwargs)")
    print("```")
    print("\nSampler kwargs:")
    for key, value in sampler_kwargs.items():
        print(f"    {key}: {value}")
    
    return mcmc_config, sampler_kwargs

if __name__ == "__main__":
    config, kwargs = main()