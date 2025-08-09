
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
