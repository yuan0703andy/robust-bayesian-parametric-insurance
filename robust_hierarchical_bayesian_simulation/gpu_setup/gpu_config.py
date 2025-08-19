#!/usr/bin/env python3
"""
GPU Configuration and Environment Setup
GPU配置與環境設定

為32核CPU + 2GPU環境優化的計算配置
支援PyMC、PyTorch、JAX等多框架GPU加速

Author: Research Team
Date: 2025-01-18
"""

import os
import sys
import psutil
import platform
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings

class GPUFramework(Enum):
    """GPU計算框架"""
    PYMC = "pymc"
    PYTORCH = "pytorch" 
    JAX = "jax"
    TENSORFLOW = "tensorflow"

class ComputeMode(Enum):
    """計算模式"""
    CPU_ONLY = "cpu_only"
    SINGLE_GPU = "single_gpu"
    DUAL_GPU = "dual_gpu"
    HYBRID = "hybrid"  # CPU + GPU混合

@dataclass
class SystemSpecs:
    """系統規格"""
    cpu_cores: int
    physical_cores: int
    memory_gb: float
    gpu_count: int
    gpu_memory_gb: List[float]
    platform: str
    
    def __post_init__(self):
        self.cpu_threads = self.cpu_cores
        self.available_memory = self.memory_gb * 0.8  # 保留20%記憶體

@dataclass
class GPUConfig:
    """GPU配置"""
    framework: GPUFramework
    device_ids: List[int]
    memory_fraction: float = 0.9
    enable_mixed_precision: bool = True
    batch_size_multiplier: float = 1.0
    
class GPUEnvironmentManager:
    """GPU環境管理器"""
    
    def __init__(self):
        self.system_specs = self._detect_system_specs()
        self.available_frameworks = self._check_gpu_frameworks()
        self.configs = {}
        
        print("🔧 GPU環境管理器初始化")
        print(f"   系統: {self.system_specs.platform}")
        print(f"   CPU核心: {self.system_specs.cpu_cores} (物理: {self.system_specs.physical_cores})")
        print(f"   記憶體: {self.system_specs.memory_gb:.1f} GB")
        print(f"   GPU數量: {self.system_specs.gpu_count}")
    
    def _detect_system_specs(self) -> SystemSpecs:
        """檢測系統規格"""
        cpu_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        platform_name = platform.system()
        
        # 檢測GPU
        gpu_count = 0
        gpu_memory = []
        
        # 嘗試檢測NVIDIA GPU
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory.append(info.total / (1024**3))  # GB
                
        except (ImportError, Exception):
            # 嘗試檢測Apple Silicon GPU
            if platform_name == "Darwin":
                # Apple M系列芯片通常有統一記憶體
                gpu_count = 1  # 假設有一個集成GPU
                gpu_memory = [memory_gb * 0.3]  # 估計30%記憶體可用於GPU
        
        return SystemSpecs(
            cpu_cores=cpu_cores,
            physical_cores=physical_cores, 
            memory_gb=memory_gb,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory,
            platform=platform_name
        )
    
    def _check_gpu_frameworks(self) -> Dict[str, bool]:
        """檢查可用的GPU框架"""
        frameworks = {}
        
        # PyMC + PyTensor
        try:
            import pymc as pm
            import pytensor
            frameworks["pymc"] = True
            print("   ✅ PyMC GPU支援可用")
        except ImportError:
            frameworks["pymc"] = False
            print("   ❌ PyMC GPU支援不可用")
        
        # PyTorch
        try:
            import torch
            frameworks["pytorch"] = torch.cuda.is_available() or torch.backends.mps.is_available()
            device = "CUDA" if torch.cuda.is_available() else "MPS" if torch.backends.mps.is_available() else "CPU"
            print(f"   {'✅' if frameworks['pytorch'] else '❌'} PyTorch GPU支援: {device}")
        except ImportError:
            frameworks["pytorch"] = False
            print("   ❌ PyTorch不可用")
        
        # JAX
        try:
            import jax
            frameworks["jax"] = len(jax.devices('gpu')) > 0 or len(jax.devices('tpu')) > 0
            print(f"   {'✅' if frameworks['jax'] else '❌'} JAX GPU支援")
        except ImportError:
            frameworks["jax"] = False
            print("   ❌ JAX不可用")
        
        return frameworks
    
    def create_mcmc_config(self, 
                          n_chains: int = 4,
                          samples_per_chain: int = 1000,
                          warmup_per_chain: int = 500) -> Dict[str, Any]:
        """創建MCMC配置"""
        
        # 根據硬體調整鏈數
        if self.system_specs.gpu_count >= 2:
            # 雙GPU策略
            chains_per_gpu = n_chains // 2
            config = {
                "gpu1_config": {
                    "device": 0,
                    "n_chains": chains_per_gpu,
                    "samples": samples_per_chain,
                    "warmup": warmup_per_chain,
                    "cores": min(8, self.system_specs.physical_cores // 4)
                },
                "gpu2_config": {
                    "device": 1, 
                    "n_chains": n_chains - chains_per_gpu,
                    "samples": samples_per_chain,
                    "warmup": warmup_per_chain,
                    "cores": min(8, self.system_specs.physical_cores // 4)
                },
                "cpu_backup": {
                    "cores": self.system_specs.physical_cores - 16,
                    "use_for": "diagnostics_and_postprocessing"
                }
            }
        else:
            # 單GPU或純CPU策略
            config = {
                "primary_config": {
                    "device": 0 if self.system_specs.gpu_count > 0 else "cpu",
                    "n_chains": n_chains,
                    "samples": samples_per_chain,
                    "warmup": warmup_per_chain,
                    "cores": min(16, self.system_specs.physical_cores)
                },
                "cpu_support": {
                    "cores": max(4, self.system_specs.physical_cores - 16),
                    "use_for": "model_selection_and_analysis"
                }
            }
        
        return config
    
    def setup_gpu_environment(self, enable_gpu: bool = True) -> 'GPUConfig':
        """設定GPU環境"""
        
        if not enable_gpu or self.system_specs.gpu_count == 0:
            return self._setup_cpu_only_config()
        
        # 設定環境變量
        self._set_gpu_environment_variables()
        
        # 創建GPU配置
        if self.available_frameworks.get("pymc", False):
            return self._setup_pymc_gpu_config()
        elif self.available_frameworks.get("pytorch", False):
            return self._setup_pytorch_config()
        else:
            print("⚠️ 沒有可用的GPU框架，回退到CPU模式")
            return self._setup_cpu_only_config()
    
    def _set_gpu_environment_variables(self):
        """設定GPU環境變量"""
        
        # PyTensor GPU配置
        if self.system_specs.platform == "Darwin":  # macOS
            os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64,mode=FAST_COMPILE'
        else:
            os.environ['PYTENSOR_FLAGS'] = 'device=cuda,floatX=float32,mode=FAST_RUN'
        
        # 多線程配置
        os.environ['OMP_NUM_THREADS'] = str(min(8, self.system_specs.physical_cores // 4))
        os.environ['MKL_NUM_THREADS'] = str(min(8, self.system_specs.physical_cores // 4))
        
        # CUDA配置（如果適用）
        if self.system_specs.platform == "Linux":
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(self.system_specs.gpu_count)))
    
    def _setup_pymc_gpu_config(self) -> 'GPUConfig':
        """設定PyMC GPU配置"""
        print("🚀 設定PyMC GPU配置...")
        
        device_ids = list(range(min(2, self.system_specs.gpu_count)))
        
        config = GPUConfig(
            framework=GPUFramework.PYMC,
            device_ids=device_ids,
            memory_fraction=0.8,
            enable_mixed_precision=True
        )
        
        self.configs["pymc"] = config
        return config
    
    def _setup_pytorch_config(self) -> 'GPUConfig':
        """設定PyTorch配置"""
        print("🚀 設定PyTorch配置...")
        
        device_ids = list(range(min(2, self.system_specs.gpu_count)))
        
        config = GPUConfig(
            framework=GPUFramework.PYTORCH,
            device_ids=device_ids,
            memory_fraction=0.9,
            enable_mixed_precision=True,
            batch_size_multiplier=2.0
        )
        
        self.configs["pytorch"] = config
        return config
    
    def _setup_cpu_only_config(self) -> 'GPUConfig':
        """設定純CPU配置"""
        print("💻 設定純CPU配置...")
        
        # 優化CPU環境
        os.environ['OMP_NUM_THREADS'] = str(self.system_specs.physical_cores)
        os.environ['MKL_NUM_THREADS'] = str(self.system_specs.physical_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.system_specs.physical_cores)
        
        config = GPUConfig(
            framework=GPUFramework.PYMC,  # 預設使用PyMC
            device_ids=[],
            memory_fraction=0.8
        )
        
        return config
    
    def create_parallel_execution_plan(self) -> Dict[str, Any]:
        """創建並行執行計劃"""
        
        total_cores = self.system_specs.physical_cores
        
        if total_cores >= 32:
            # 32核心最佳配置
            plan = {
                "model_selection_pool": {
                    "cores": 16,
                    "max_workers": 16,
                    "tasks": "model_fitting_and_hyperparameter_optimization"
                },
                "mcmc_pool": {
                    "cores": 8,
                    "max_workers": 4,  # MCMC需要更多記憶體
                    "tasks": "mcmc_sampling_and_validation",
                    "gpu_support": True
                },
                "analysis_pool": {
                    "cores": 8,
                    "max_workers": 8,
                    "tasks": "posterior_analysis_and_diagnostics"
                }
            }
        else:
            # 少於32核心的配置
            cores_per_pool = max(2, total_cores // 3)
            plan = {
                "model_selection_pool": {
                    "cores": cores_per_pool,
                    "max_workers": cores_per_pool,
                    "tasks": "model_fitting"
                },
                "mcmc_pool": {
                    "cores": cores_per_pool,
                    "max_workers": max(2, cores_per_pool // 2),
                    "tasks": "mcmc_sampling",
                    "gpu_support": self.system_specs.gpu_count > 0
                },
                "analysis_pool": {
                    "cores": total_cores - 2 * cores_per_pool,
                    "max_workers": max(2, total_cores - 2 * cores_per_pool),
                    "tasks": "analysis"
                }
            }
        
        return plan
    
    def print_performance_summary(self):
        """打印效能摘要"""
        print("\n📊 系統效能配置摘要")
        print("=" * 50)
        print(f"CPU核心: {self.system_specs.cpu_cores} (物理: {self.system_specs.physical_cores})")
        print(f"記憶體: {self.system_specs.memory_gb:.1f} GB (可用: {self.system_specs.available_memory:.1f} GB)")
        print(f"GPU: {self.system_specs.gpu_count} 個")
        
        if self.system_specs.gpu_memory_gb:
            for i, mem in enumerate(self.system_specs.gpu_memory_gb):
                print(f"   GPU {i}: {mem:.1f} GB")
        
        print(f"\n可用框架:")
        for framework, available in self.available_frameworks.items():
            print(f"   {framework}: {'✅' if available else '❌'}")
        
        # 預估加速比
        baseline_time = 14.0  # 小時
        if self.system_specs.gpu_count >= 2 and self.system_specs.physical_cores >= 16:
            speedup = 9.0
        elif self.system_specs.gpu_count >= 1 and self.system_specs.physical_cores >= 8:
            speedup = 6.0
        elif self.system_specs.physical_cores >= 16:
            speedup = 4.0
        else:
            speedup = 2.0
        
        estimated_time = baseline_time / speedup
        print(f"\n預估加速:")
        print(f"   基線時間: {baseline_time:.1f} 小時")
        print(f"   預估時間: {estimated_time:.1f} 小時")
        print(f"   加速比: {speedup:.1f}x")

# 便利函數
def setup_gpu_environment(enable_gpu: bool = True) -> Tuple[GPUConfig, Dict[str, Any]]:
    """快速設定GPU環境"""
    manager = GPUEnvironmentManager()
    gpu_config = manager.setup_gpu_environment(enable_gpu)
    execution_plan = manager.create_parallel_execution_plan()
    
    manager.print_performance_summary()
    
    return gpu_config, execution_plan

def get_optimal_mcmc_config(n_models: int = 5, 
                          samples_per_model: int = 1000) -> Dict[str, Any]:
    """獲取最佳MCMC配置"""
    manager = GPUEnvironmentManager()
    
    # 根據模型數量調整
    if manager.system_specs.gpu_count >= 2:
        chains_per_model = 4
        total_chains = n_models * chains_per_model
        
        config = {
            "n_models": n_models,
            "chains_per_model": chains_per_model,
            "samples_per_chain": samples_per_model,
            "parallel_chains": min(8, total_chains),
            "use_gpu": True,
            "gpu_allocation": {
                "gpu_0": {"models": list(range(0, n_models//2))},
                "gpu_1": {"models": list(range(n_models//2, n_models))}
            }
        }
    else:
        config = {
            "n_models": n_models,
            "chains_per_model": 2,
            "samples_per_chain": samples_per_model,
            "parallel_chains": min(4, manager.system_specs.physical_cores//2),
            "use_gpu": manager.system_specs.gpu_count > 0
        }
    
    return config

if __name__ == "__main__":
    # 測試配置
    print("🧪 測試GPU配置...")
    gpu_config, execution_plan = setup_gpu_environment()
    
    print("\n📋 執行計劃:")
    for pool_name, pool_config in execution_plan.items():
        print(f"   {pool_name}: {pool_config['cores']} 核心")
    
    print("\n✅ GPU配置測試完成")