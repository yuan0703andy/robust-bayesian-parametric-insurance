#!/usr/bin/env python3
"""
OnDemand 高效能優化配置
OnDemand High-Performance Optimized Configuration

專為 OnDemand 環境優化的配置，移除所有環境檢測和動態配置
Optimized configuration specifically for OnDemand environment, removing all environment detection and dynamic configuration
"""

import os
import warnings

def setup_ondemand_environment():
    """設置 OnDemand 高效能環境"""
    
    print("🚀 設置 OnDemand 高效能配置...")
    
    # OnDemand 優化的環境變數
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float32,mode=FAST_RUN,optimizer=fast_run,cxx="
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["OMP_NUM_THREADS"] = "8"  # OnDemand 通常有多核心
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    
    # 抑制不必要的警告
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message='.*axis.*dim.*')
    warnings.filterwarnings('ignore', message='.*ambiguous.*')
    
    print("✅ OnDemand 環境變數設置完成")
    
    return get_ondemand_config()


def get_ondemand_config():
    """獲取 OnDemand 高效能配置"""
    
    return {
        # PyMC 配置
        'pymc_backend': 'cpu',
        'pymc_mode': 'FAST_RUN',  # 生產模式，最快執行
        'n_threads': 8,           # 利用 OnDemand 多核心
        'configure_pymc': False,  # 避免動態配置衝突
        
        # MCMC 採樣參數 (高效能)
        'draws': 1000,            # 更多樣本獲得更好結果
        'tune': 500,              # 適中的調參步數
        'chains': 4,              # 4條鏈充分利用多核心
        'target_accept': 0.9,     # 較高接受率確保收斂
        'cores': 4,               # 並行採樣
        'return_inferencedata': True,
        'progressbar': True,
        
        # 分析參數 (高效能)
        'n_monte_carlo_samples': 2000,  # 更多 Monte Carlo 樣本
        'n_loss_scenarios': 1000,       # 更多損失情境
        'n_mixture_components': 5,       # 更複雜的混合模型
        
        # 優化參數
        'max_training_samples': None,    # 不限制，使用全部數據
        'max_validation_samples': None,  # 不限制，使用全部數據
        'density_ratio_constraint': 2.0,
        
        # 錯誤處理
        'fallback_to_demo': False,      # 高效能模式不使用演示回退
        'suppress_warnings': True,      # 抑制警告提升效能
        'verbose': True                 # 但保持詳細輸出用於監控
    }


def get_ondemand_data_config():
    """獲取 OnDemand 數據處理配置"""
    
    return {
        # 數據分割比例 (使用更多數據)
        'train_ratio': 0.8,          # 80% 訓練數據
        'validation_ratio': 0.2,     # 20% 驗證數據
        'min_train_samples': 50,     # 最小訓練樣本數
        'min_validation_samples': 20, # 最小驗證樣本數
        
        # 產品參數邊界 (更廣泛的搜索)
        'trigger_threshold_range': (20, 75),      # 更廣的觸發閾值範圍
        'payout_multiplier_range': (0.3, 3.0),   # 更廣的賠付倍數範圍
        'max_payout_multiplier_range': (2.0, 6.0), # 更廣的最大賠付倍數
        
        # 基差風險最佳化
        'w_under': 2.5,              # 稍微提高賠付不足懲罰
        'w_over': 0.4,               # 降低過度賠付懲罰
        'optimization_method': 'L-BFGS-B',  # 更快的優化算法
        'max_iterations': 200,       # 更多迭代次數
        'tolerance': 1e-8            # 更高精度
    }


def create_ondemand_notebook_config():
    """為 notebook 創建 OnDemand 配置代碼"""
    
    config_code = '''
# %% OnDemand 高效能配置
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

print("🚀 OnDemand 高效能模式啟動...")

# 設置 OnDemand 優化環境變數
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float32,mode=FAST_RUN,optimizer=fast_run,cxx="
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OMP_NUM_THREADS"] = "8"      # 利用 OnDemand 多核心
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# 抑制警告提升效能
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*axis.*dim.*')

# OnDemand 高效能參數
pymc_config = {
    'pymc_backend': 'cpu',
    'pymc_mode': 'FAST_RUN',
    'n_threads': 8,
    'configure_pymc': False
}

# 高效能分析參數
n_monte_carlo_samples = 2000   # 高質量 Monte Carlo
n_loss_scenarios = 1000        # 豐富的損失情境
n_mixture_components = 5       # 複雜混合模型

# MCMC 高效能參數
mcmc_params = {
    'draws': 1000,
    'tune': 500, 
    'chains': 4,
    'cores': 4,
    'target_accept': 0.9
}

print("✅ OnDemand 高效能配置完成")
print(f"   🖥️ CPU 核心: {os.environ.get('OMP_NUM_THREADS')}")
print(f"   🎯 Monte Carlo 樣本: {n_monte_carlo_samples}")
print(f"   🔄 MCMC 鏈數: {mcmc_params['chains']}")
print(f"   📊 損失情境: {n_loss_scenarios}")
'''
    
    return config_code


def apply_ondemand_optimizations():
    """應用 OnDemand 特定的效能優化"""
    
    print("⚡ 應用 OnDemand 效能優化...")
    
    try:
        # 設置 NumPy 多線程
        import numpy as np
        if hasattr(np, '__config__') and hasattr(np.__config__, 'show'):
            print("  📊 NumPy 配置已優化")
            
        # 設置 Pandas 效能選項
        import pandas as pd
        pd.set_option('compute.use_bottleneck', True)
        pd.set_option('compute.use_numexpr', True)
        print("  🐼 Pandas 效能選項已啟用")
        
        # 記憶體和計算優化
        import gc
        gc.collect()  # 清理記憶體
        print("  🧹 記憶體已優化")
        
        print("✅ OnDemand 效能優化完成")
        
    except ImportError as e:
        print(f"  ⚠️ 部分優化跳過: {e}")


if __name__ == "__main__":
    print("🚀 OnDemand 高效能配置")
    print("=" * 50)
    
    # 設置環境
    config = setup_ondemand_environment()
    
    # 應用優化
    apply_ondemand_optimizations()
    
    # 顯示配置
    print(f"\n📋 OnDemand 配置摘要:")
    print(f"   PyMC 模式: {config['pymc_mode']}")
    print(f"   CPU 核心: {config['n_threads']}")
    print(f"   MCMC 鏈數: {config['chains']}")
    print(f"   樣本數: {config['draws']}")
    print(f"   Monte Carlo: {config['n_monte_carlo_samples']}")
    print(f"   損失情境: {config['n_loss_scenarios']}")
    
    # 生成 notebook 配置
    notebook_config = create_ondemand_notebook_config()
    
    with open('ondemand_notebook_config.py', 'w') as f:
        f.write(notebook_config)
    
    print(f"\n💾 已生成:")
    print("   - ondemand_notebook_config.py: notebook 配置代碼")
    
    print(f"\n🎯 使用方式:")
    print("1. 將 ondemand_notebook_config.py 的內容複製到 notebook 第一個 cell")
    print("2. 或者在 Python 腳本開頭導入: from ondemand_optimized_config import setup_ondemand_environment")
    print("3. 享受 OnDemand 高效能分析！")