# ============================================================================
# 2. 環境配置部分 - 添加在第16行之後 (import warnings 之後)
# ============================================================================

# %% 環境檢測與 PyMC 配置
print("🌐 檢測運行環境...")

def detect_environment():
    """檢測運行環境類型"""
    import os
    
    if 'SLURM_JOB_ID' in os.environ:
        return 'hpc_slurm'
    elif 'PBS_JOBID' in os.environ:
        return 'hpc_pbs' 
    elif any('OOD' in key for key in os.environ.keys()):
        return 'ondemand'
    elif 'HOSTNAME' in os.environ and 'dcc' in os.environ['HOSTNAME']:
        return 'dcc'  # Duke Compute Cluster
    else:
        return 'local'

# 檢測環境
run_environment = detect_environment()
print(f"   環境類型: {run_environment}")

# 根據環境設置參數
if run_environment in ['hpc_slurm', 'hpc_pbs', 'dcc']:
    # HPC 環境配置
    pymc_config = {
        'pymc_backend': 'cpu',        # HPC 通常用 CPU，除非有 GPU 節點
        'pymc_mode': 'FAST_RUN',      # 生產環境用快速運行
        'n_threads': int(os.environ.get('OMP_NUM_THREADS', 8)),
        'configure_pymc': True
    }
    # HPC 上可以用更多資源
    n_samples = min(1000, 500)  # 根據實際情況調整
    n_monte_carlo_samples = 1000
    n_loss_scenarios = 500
    print(f"   🖥️ HPC 配置: CPU, {pymc_config['n_threads']} threads")
    
elif run_environment == 'ondemand':
    # OnDemand 環境配置
    pymc_config = {
        'pymc_backend': 'cpu',
        'pymc_mode': 'FAST_COMPILE',  # 交互式環境用快速編譯
        'n_threads': 4,
        'configure_pymc': True
    }
    n_samples = 500
    n_monte_carlo_samples = 500
    n_loss_scenarios = 200
    print(f"   🌐 OnDemand 配置: CPU, 4 threads")
    
else:
    # 本地環境配置 (macOS 等)
    pymc_config = {
        'pymc_backend': 'cpu',        # 避免 Metal 問題
        'pymc_mode': 'FAST_COMPILE',
        'n_threads': 1,               # 避免線程衝突
        'configure_pymc': True
    }
    n_samples = 500  # 如果已經定義，保持原值
    n_monte_carlo_samples = 200  # 本地測試用較少樣本
    n_loss_scenarios = 100
    print(f"   💻 本地配置: CPU only, single thread")

print(f"   📊 分析參數: samples={n_samples}, monte_carlo={n_monte_carlo_samples}, scenarios={n_loss_scenarios}")
print(f"   ⚙️ PyMC 配置: {pymc_config}")

# 如果 n_samples 已經在代碼中定義，保持原來的值
try:
    if 'n_samples' in locals():
        print(f"   ℹ️ 保持現有 n_samples 值: {n_samples}")
except:
    pass