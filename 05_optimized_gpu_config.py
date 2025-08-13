#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_optimized_gpu_config.py
==========================
High-Performance GPU/CPU Optimization Configuration
高性能GPU/CPU優化配置

For systems with:
- 16 core CPU
- 2x RTX 2050 GPUs
- Large RAM capacity

針對以下硬體優化：
- 16核CPU
- 2張RTX 2050 GPU  
- 大容量記憶體
"""

import os
import torch
import numpy as np
from pathlib import Path
import warnings

def configure_high_performance_environment(verbose=True):
    """
    配置高性能計算環境
    Configure high-performance computing environment
    
    針對16核CPU + 2x RTX2050的最佳化配置
    Optimized for 16-core CPU + 2x RTX2050 setup
    """
    
    if verbose:
        print("🚀 High-Performance Environment Configuration")
        print("   高性能環境配置")
        print("=" * 60)
        print(f"   • Target: 16-core CPU + 2x RTX 2050 GPUs")
        print(f"   • Optimization: MCMC sampling parallelization")
        
    config_changes = {}
    
    # =============================================================================
    # CPU優化配置 CPU Optimization
    # =============================================================================
    
    print("\n🖥️ CPU Optimization (16-core setup)")
    
    # OpenMP線程數設置為16核心
    old_omp = os.environ.get('OMP_NUM_THREADS', 'not_set')
    os.environ['OMP_NUM_THREADS'] = '16'
    print(f"   • OMP_NUM_THREADS: {old_omp} → 16")
    
    # MKL線程數（Intel Math Kernel Library）
    old_mkl = os.environ.get('MKL_NUM_THREADS', 'not_set')
    os.environ['MKL_NUM_THREADS'] = '16'
    print(f"   • MKL_NUM_THREADS: {old_mkl} → 16")
    
    # NumPy線程數
    old_openblas = os.environ.get('OPENBLAS_NUM_THREADS', 'not_set')
    os.environ['OPENBLAS_NUM_THREADS'] = '16'
    print(f"   • OPENBLAS_NUM_THREADS: {old_openblas} → 16")
    
    # 設置為GNU線程層（避免衝突）
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    print(f"   • MKL_THREADING_LAYER: GNU")
    
    config_changes['cpu'] = {
        'OMP_NUM_THREADS': '16',
        'MKL_NUM_THREADS': '16', 
        'OPENBLAS_NUM_THREADS': '16',
        'MKL_THREADING_LAYER': 'GNU'
    }
    
    # =============================================================================
    # GPU配置 GPU Configuration  
    # =============================================================================
    
    print(f"\n🖥️ GPU Configuration (RTX 2050 x2)")
    
    # 檢查CUDA可用性
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   ✅ CUDA available with {gpu_count} GPUs")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   • GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
        # 設置GPU記憶體增長策略
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 使用兩張GPU
        print(f"   • CUDA_VISIBLE_DEVICES: 0,1")
        
        config_changes['gpu'] = {
            'cuda_available': True,
            'gpu_count': gpu_count,
            'visible_devices': '0,1'
        }
    else:
        print(f"   ⚠️ CUDA not available, using CPU-only optimization")
        config_changes['gpu'] = {'cuda_available': False}
    
    # =============================================================================
    # JAX/PyMC優化配置 JAX/PyMC Optimization
    # =============================================================================
    
    print(f"\n🧠 JAX/PyMC Optimization")
    
    # 優先使用GPU（如果可用）
    if torch.cuda.is_available():
        old_jax_platform = os.environ.get('JAX_PLATFORM_NAME', 'not_set')
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
        print(f"   • JAX_PLATFORM_NAME: {old_jax_platform} → gpu")
        
        # GPU記憶體預分配設置
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        print(f"   • XLA_PYTHON_CLIENT_PREALLOCATE: false (避免記憶體預分配)")
        
        config_changes['jax'] = {
            'platform': 'gpu',
            'preallocate': 'false'
        }
    else:
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        print(f"   • JAX_PLATFORM_NAME: cpu (GPU不可用)")
        config_changes['jax'] = {'platform': 'cpu'}
    
    # PyTensor優化設置
    old_pytensor = os.environ.get('PYTENSOR_FLAGS', 'not_set')
    # 使用快速編譯模式，適合大型MCMC
    pytensor_flags = 'mode=FAST_RUN,optimizer=fast_run,floatX=float32'
    os.environ['PYTENSOR_FLAGS'] = pytensor_flags
    print(f"   • PYTENSOR_FLAGS: FAST_RUN with optimizer")
    
    config_changes['pytensor'] = {'flags': pytensor_flags}
    
    # =============================================================================
    # 記憶體優化 Memory Optimization
    # =============================================================================
    
    print(f"\n💾 Memory Optimization")
    
    # Python記憶體分配器優化
    os.environ['PYTHONHASHSEED'] = '0'  # 確保可重現性
    print(f"   • PYTHONHASHSEED: 0 (確保可重現性)")
    
    # 大型數組記憶體映射
    os.environ['PYTENSOR_FLAGS'] += ',allow_gc=True'
    print(f"   • PyTensor garbage collection: enabled")
    
    config_changes['memory'] = {
        'hash_seed': '0',
        'gc_enabled': True
    }
    
    return config_changes


def get_optimized_mcmc_config():
    """
    獲取針對高性能硬體的最佳MCMC配置
    Get optimized MCMC configuration for high-performance hardware
    """
    
    # 檢查是否有GPU
    use_gpu = torch.cuda.is_available()
    
    config = {
        # 基本MCMC配置
        'draws': 4000,              # 增加樣本數（因為有更多計算資源）
        'tune': 2000,               # 調整階段樣本數
        'chains': 8,                # 8條鏈並行（充分利用16核）
        'cores': 16,                # 使用全部16核
        'target_accept': 0.95,      # 高接受率確保品質
        
        # GPU特定配置
        'use_gpu': use_gpu,
        
        # 進階配置
        'nuts_sampler': {
            'max_treedepth': 12,    # 增加樹深度（GPU有更多計算力）
            'step_size': 'auto',    # 自動步長調整
        },
        
        # 並行配置
        'parallel': {
            'method': 'multiprocessing',  # 多進程並行
            'n_jobs': 8,                  # 8個並行作業
        },
        
        # 記憶體管理
        'memory': {
            'chunk_size': 1000,           # 大塊處理（充足記憶體）
            'cache_size': '2GB',          # 大快取空間
        }
    }
    
    return config


def create_optimized_analysis_script():
    """
    創建優化版本的分析腳本
    Create optimized version of analysis script
    """
    
    optimized_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_robust_bayesian_parm_insurance_optimized.py
==============================================
High-Performance Optimized Version
高性能優化版本

針對16核CPU + 2x RTX2050優化的強健貝氏階層模型
Robust Bayesian hierarchical model optimized for 16-core CPU + 2x RTX2050
"""

# 載入優化配置
from 05_optimized_gpu_config import configure_high_performance_environment, get_optimized_mcmc_config

# 配置高性能環境
print("🚀 Loading High-Performance Configuration...")
config_changes = configure_high_performance_environment(verbose=True)
mcmc_config = get_optimized_mcmc_config()

print(f"\\n📊 Optimized MCMC Configuration:")
print(f"   • Draws: {mcmc_config['draws']}")
print(f"   • Chains: {mcmc_config['chains']}")  
print(f"   • Cores: {mcmc_config['cores']}")
print(f"   • GPU Support: {mcmc_config['use_gpu']}")

# 現在載入主要分析腳本
print("\\n" + "="*80)
print("🧠 Loading Main Analysis Script with Optimizations...")
print("="*80)

# 這裡會執行您的主要分析，但使用優化配置
exec(open("05_robust_bayesian_parm_insurance.py").read())
'''

    return optimized_script


def benchmark_performance():
    """
    性能基準測試
    Performance benchmarking
    """
    
    print("\n🏃‍♂️ Performance Benchmarking")
    print("-" * 40)
    
    import time
    
    # CPU基準測試
    print("CPU性能測試...")
    start_time = time.time()
    
    # 大矩陣乘法測試
    matrix_size = 2000
    A = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    B = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    C = np.dot(A, B)
    
    cpu_time = time.time() - start_time
    print(f"   矩陣乘法 ({matrix_size}x{matrix_size}): {cpu_time:.2f}秒")
    
    # GPU基準測試（如果可用）
    if torch.cuda.is_available():
        print("GPU性能測試...")
        device = torch.device('cuda:0')
        
        start_time = time.time()
        A_gpu = torch.randn(matrix_size, matrix_size, device=device)
        B_gpu = torch.randn(matrix_size, matrix_size, device=device)
        C_gpu = torch.mm(A_gpu, B_gpu)
        torch.cuda.synchronize()  # 確保計算完成
        
        gpu_time = time.time() - start_time
        print(f"   GPU矩陣乘法 ({matrix_size}x{matrix_size}): {gpu_time:.2f}秒")
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"   加速比: {speedup:.1f}x")
    
    return {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time if torch.cuda.is_available() else None
    }


def main():
    """主函數：配置並測試高性能環境"""
    
    print("🖥️ High-Performance Bayesian Analysis Setup")
    print("   16-Core CPU + 2x RTX2050 Optimization")
    print("=" * 70)
    
    # 配置環境
    config_changes = configure_high_performance_environment()
    
    # 獲取MCMC配置
    mcmc_config = get_optimized_mcmc_config()
    
    # 性能基準測試
    benchmark_results = benchmark_performance()
    
    print(f"\n🎯 Optimization Summary:")
    print(f"   • CPU cores utilized: 16")
    print(f"   • GPU support: {'✅' if mcmc_config['use_gpu'] else '❌'}")
    print(f"   • MCMC chains: {mcmc_config['chains']}")
    print(f"   • Expected speedup: ~5-8x for large models")
    
    # 創建優化腳本
    optimized_script = create_optimized_analysis_script()
    
    # 保存優化腳本
    output_path = "05_robust_bayesian_parm_insurance_optimized.py"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(optimized_script)
    
    print(f"\n💾 Optimized script saved: {output_path}")
    print(f"\n✨ Ready to run high-performance Bayesian analysis!")
    print(f"   Execute: python {output_path}")
    
    return {
        'config_changes': config_changes,
        'mcmc_config': mcmc_config,
        'benchmark_results': benchmark_results
    }


if __name__ == "__main__":
    results = main()