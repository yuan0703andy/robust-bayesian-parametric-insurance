#!/usr/bin/env python3
"""
GPU Configuration for Bayesian Analysis
貝氏分析GPU配置

Simplified interface for GPU optimization in main analysis scripts.
為主要分析腳本提供簡化的GPU優化介面。
"""

import os
import warnings
from typing import Dict, Any, Optional
from .dual_gpu_mcmc_setup import DualGPU_MCMC_Optimizer

class GPUConfig:
    """
    簡化的GPU配置類
    Simplified GPU configuration class for easy integration
    """
    
    def __init__(self, enable_gpu: bool = True):
        """
        Initialize GPU configuration
        
        Parameters:
        -----------
        enable_gpu : bool
            Whether to enable GPU acceleration
        """
        self.enable_gpu = enable_gpu
        self.optimizer = None
        self.hardware_level = "cpu_only"
        
        if enable_gpu:
            self._setup_gpu()
    
    def _setup_gpu(self):
        """Setup GPU environment and detect hardware"""
        try:
            self.optimizer = DualGPU_MCMC_Optimizer()
            self.optimizer.configure_environment_variables()
            
            # Detect hardware capability
            import jax
            devices = jax.devices()
            print(f"🔍 JAX devices detected: {devices}")
            
            # 修正 GPU 檢測邏輯 - JAX 在 CUDA 設備上 device_kind 是 GPU 名稱，不是 'gpu'
            gpu_devices = []
            for d in devices:
                device_str = str(d).lower()
                device_kind_str = d.device_kind.lower()
                # 檢測條件：CudaDevice 或包含 GPU 相關關鍵字
                if ('cuda' in device_str or 
                    'gpu' in device_kind_str or 
                    'geforce' in device_kind_str or 
                    'rtx' in device_kind_str or
                    'gtx' in device_kind_str or
                    'tesla' in device_kind_str or
                    'quadro' in device_kind_str):
                    gpu_devices.append(d)
                    print(f"    ✅ GPU device found: {d} (kind: {d.device_kind})")
            print(f"🔍 GPU devices found: {gpu_devices}")
            print(f"🔍 Device kinds: {[d.device_kind for d in devices]}")
            print(f"🔍 JAX backend: {jax.default_backend()}")
            
            if len(gpu_devices) >= 2:
                self.hardware_level = "dual_gpu"
                print("🎯 Dual-GPU configuration detected")
            elif len(gpu_devices) == 1:
                self.hardware_level = "single_gpu" 
                print("🎯 Single-GPU configuration detected")
            else:
                self.hardware_level = "cpu_only"
                print("💻 CPU-only configuration")
                print("💡 Possible causes: JAX CPU-only install, missing CUDA drivers, or GPU not visible")
                
        except ImportError:
            print("⚠️ JAX not available, using CPU-only")
            self.hardware_level = "cpu_only"
        except Exception as e:
            print(f"⚠️ GPU setup failed: {e}, falling back to CPU")
            self.hardware_level = "cpu_only"
    
    def get_mcmc_config(self) -> Dict[str, Any]:
        """
        Get optimized MCMC configuration based on hardware
        根據硬體獲取優化的MCMC配置
        """
        if self.hardware_level == "dual_gpu" and self.optimizer:
            config = self.optimizer.create_dual_gpu_mcmc_config()
            print("🚀 Using dual-GPU optimized MCMC config")
            return {
                "n_samples": config["draws"],
                "n_warmup": config["tune"], 
                "n_chains": config["chains"],
                "cores": config["cores"],
                "target_accept": config["target_accept"],
                "backend": "jax"
            }
            
        elif self.hardware_level == "single_gpu":
            print("⚡ Using single-GPU optimized MCMC config")
            return {
                "n_samples": 2500,
                "n_warmup": 1000,
                "n_chains": 6,
                "cores": 12,
                "target_accept": 0.92,
                "backend": "jax"
            }
            
        else:
            print("💻 Using CPU-optimized MCMC config")
            return {
                "n_samples": 2000,
                "n_warmup": 1000, 
                "n_chains": 4,
                "cores": 8,
                "target_accept": 0.90,
                "backend": "pytensor"
            }
    
    def get_numpyro_mcmc_config(self) -> Dict[str, Any]:
        """
        Get NumPyro MCMC configuration (bypasses PyMC GPU issues)
        獲取NumPyro MCMC配置（繞過PyMC GPU問題）
        """
        config = self.get_mcmc_config()
        
        numpyro_config = {
            "num_samples": config["n_samples"],
            "num_warmup": config["n_warmup"],
            "num_chains": config["n_chains"],
            "chain_method": "parallel" if self.hardware_level != "cpu_only" else "sequential",
            "progress_bar": True,
        }
        
        print(f"🎯 NumPyro MCMC config: {numpyro_config}")
        return numpyro_config
    
    def get_pymc_sampler_kwargs(self) -> Dict[str, Any]:
        """
        Get PyMC sampler kwargs for pm.sample()
        獲取PyMC採樣器參數用於pm.sample()
        
        WARNING: PyMC GPU backend has compatibility issues.
        Consider using get_numpyro_mcmc_config() instead.
        """
        config = self.get_mcmc_config()
        
        # Force CPU backend for PyMC due to GPU compatibility issues
        sampler_kwargs = {
            "draws": config["n_samples"],
            "tune": config["n_warmup"],
            "chains": config["n_chains"], 
            "cores": config["cores"],
            "target_accept": config["target_accept"],
            "compute_convergence_checks": True,
            "return_inferencedata": True,
        }
        
        # Only use CPU backend for PyMC due to GPU issues
        print("⚠️ Using CPU backend for PyMC due to GPU compatibility issues")
        print("💡 Consider using NumPyro directly for GPU acceleration")
        
        return sampler_kwargs
    
    def print_performance_summary(self):
        """Print expected performance summary"""
        config = self.get_mcmc_config()
        total_samples = config["n_samples"] * config["n_chains"]
        
        print(f"\n📊 MCMC Configuration Summary:")
        print(f"   Hardware: {self.hardware_level}")
        print(f"   Backend: {config.get('backend', 'pytensor')}")
        print(f"   Chains: {config['n_chains']}")
        print(f"   Samples per chain: {config['n_samples']:,}")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Target accept: {config['target_accept']}")
        
        if self.hardware_level == "dual_gpu":
            print(f"   Expected speedup: 4x over CPU")
            print(f"   Estimated time: ~15 minutes")
        elif self.hardware_level == "single_gpu":
            print(f"   Expected speedup: 2-3x over CPU")
            print(f"   Estimated time: ~30 minutes")
        else:
            print(f"   CPU baseline performance")
            print(f"   Estimated time: ~60 minutes")

# Convenience functions for direct use
def setup_gpu_environment(enable_gpu: bool = True) -> GPUConfig:
    """
    便利函數：設置GPU環境
    Convenience function to setup GPU environment
    """
    return GPUConfig(enable_gpu=enable_gpu)

def get_optimized_mcmc_config(hardware_level: Optional[str] = None) -> Dict[str, Any]:
    """
    便利函數：獲取優化的MCMC配置
    Convenience function to get optimized MCMC config
    """
    gpu_config = GPUConfig(enable_gpu=True)
    if hardware_level:
        gpu_config.hardware_level = hardware_level
    return gpu_config.get_mcmc_config()