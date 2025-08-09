#!/usr/bin/env python3
"""
HPC/OnDemand 環境使用示例
Example usage for HPC/OnDemand environments

此腳本展示如何在 HPC 環境中正確配置和使用 Bayesian 框架
This script demonstrates how to properly configure and use the Bayesian framework in HPC environments
"""

import numpy as np
from bayesian import RobustBayesianAnalyzer
from skill_scores.basis_risk_functions import BasisRiskType

def hpc_example():
    """HPC 環境使用示例"""
    
    print("🖥️ HPC/OnDemand 環境貝葉斯分析示例")
    print("=" * 60)
    
    # 生成示例數據
    np.random.seed(42)
    n_total = 200
    
    # 模擬損失數據
    hazard_indices = np.random.uniform(25, 65, n_total)
    base_losses = np.where(
        hazard_indices > 45,
        np.random.lognormal(np.log(1e8), 0.8),
        np.where(
            hazard_indices > 35,
            np.random.lognormal(np.log(5e7), 0.6),
            np.random.exponential(1e6) * (np.random.random(n_total) < 0.2)
        )
    )
    
    # 數據分割
    n_train = 140
    train_losses = base_losses[:n_train] 
    validation_losses = base_losses[n_train:]
    train_indices = hazard_indices[:n_train]
    
    print(f"📊 數據: 訓練({n_train}) / 驗證({len(validation_losses)})")
    
    # 創建損失情境矩陣
    n_scenarios = 200  # HPC 上可以使用更多情境
    actual_losses_matrix = np.zeros((n_scenarios, len(train_indices)))
    
    for i in range(n_scenarios):
        scenario_multiplier = np.random.lognormal(0, 0.3)
        actual_losses_matrix[i, :] = train_losses * scenario_multiplier
    
    print(f"🎲 生成 {n_scenarios} 個損失情境")
    
    # 初始化分析器
    analyzer = RobustBayesianAnalyzer(
        density_ratio_constraint=2.0,
        n_monte_carlo_samples=1000,  # HPC 上可以用更多樣本
        n_mixture_components=4
    )
    
    # 產品參數邊界
    product_bounds = {
        'trigger_threshold': (30, 60),
        'payout_amount': (5e7, 3e8),
        'max_payout': (1e9, 1e9)
    }
    
    print("\n🚀 執行整合貝葉斯最佳化 (HPC 配置)")
    
    # ============================================================================
    # HPC 環境配置示例
    # ============================================================================
    
    # 情境 1: CPU 密集型 HPC 節點
    print("\n💻 情境 1: CPU 密集型 HPC 節點")
    try:
        results_cpu = analyzer.integrated_bayesian_optimization(
            observations=train_losses,
            validation_data=validation_losses,
            hazard_indices=train_indices,
            actual_losses=actual_losses_matrix,
            product_bounds=product_bounds,
            basis_risk_type=BasisRiskType.WEIGHTED_ASYMMETRIC,
            w_under=2.0, w_over=0.5,
            # CPU 密集型配置
            pymc_backend="cpu",
            pymc_mode="FAST_RUN",      # 生產環境用快速運行
            n_threads=8,               # 使用多核心 (根據 HPC 節點調整)
            configure_pymc=True
        )
        print("✅ CPU 密集型配置成功")
        
    except Exception as e:
        print(f"❌ CPU 配置失敗: {e}")
    
    # 情境 2: GPU 加速 HPC 節點
    print("\n🎮 情境 2: GPU 加速 HPC 節點")
    try:
        results_gpu = analyzer.integrated_bayesian_optimization(
            observations=train_losses[:50],  # 小規模測試
            validation_data=validation_losses[:20],
            hazard_indices=train_indices[:50],
            actual_losses=actual_losses_matrix[:50, :50],
            product_bounds=product_bounds,
            basis_risk_type=BasisRiskType.WEIGHTED_ASYMMETRIC,
            w_under=2.0, w_over=0.5,
            # GPU 加速配置
            pymc_backend="gpu",
            pymc_mode="FAST_RUN",
            n_threads=4,               # GPU 時線程數可以較少
            configure_pymc=True
        )
        print("✅ GPU 加速配置成功")
        
    except Exception as e:
        print(f"❌ GPU 配置失敗: {e}")
        print("   可能原因: GPU 不可用或 JAX GPU 支援未安裝")
    
    # 情境 3: 自動檢測環境
    print("\n🔍 情境 3: 自動檢測環境")
    try:
        results_auto = analyzer.integrated_bayesian_optimization(
            observations=train_losses[:30],  # 最小測試
            validation_data=validation_losses[:10],
            hazard_indices=train_indices[:30],
            actual_losses=actual_losses_matrix[:30, :30],
            product_bounds=product_bounds,
            basis_risk_type=BasisRiskType.WEIGHTED_ASYMMETRIC,
            w_under=2.0, w_over=0.5,
            # 自動檢測配置
            pymc_backend="auto",       # 讓系統自動選擇
            pymc_mode="FAST_COMPILE",  # 快速測試
            n_threads=None,            # 自動設置
            configure_pymc=True
        )
        print("✅ 自動檢測配置成功")
        
    except Exception as e:
        print(f"❌ 自動檢測配置失敗: {e}")
    
    # ============================================================================
    # HPC 使用建議
    # ============================================================================
    print("\n💡 HPC/OnDemand 使用建議")
    print("-" * 40)
    print("📋 配置選擇指南:")
    print()
    print("🖥️ CPU 節點 (推薦):")
    print("   pymc_backend='cpu'")
    print("   pymc_mode='FAST_RUN'")
    print("   n_threads=節點核心數 (如 8, 16, 32)")
    print()
    print("🎮 GPU 節點:")
    print("   pymc_backend='gpu'")
    print("   pymc_mode='FAST_RUN'") 
    print("   n_threads=4-8")
    print("   需要: pip install jax[cuda] 或 jax[tpu]")
    print()
    print("🔍 不確定環境:")
    print("   pymc_backend='auto'")
    print("   pymc_mode='FAST_COMPILE' (測試)")
    print("   n_threads=None (自動)")
    print()
    print("⚡ 性能優化:")
    print("   - 增加 n_monte_carlo_samples (500-2000)")
    print("   - 增加損失情境數 (200-1000)")
    print("   - 使用更大的數據集")
    print("   - 考慮並行化多個分析")
    print()
    print("🔧 故障排除:")
    print("   - 如果 Metal 錯誤 → pymc_backend='cpu'")
    print("   - 如果記憶體不足 → 減少樣本數和情境數")
    print("   - 如果編譯慢 → pymc_mode='FAST_COMPILE'")
    print("   - 如果線程衝突 → n_threads=1")


def create_hpc_batch_script():
    """創建 HPC batch 腳本示例"""
    
    batch_script = """#!/bin/bash
#SBATCH --job-name=bayesian_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=normal

# 載入必要模組
module load python/3.9
module load gcc/9.3.0

# 創建虛擬環境 (如果需要)
# python -m venv bayesian_env
# source bayesian_env/bin/activate

# 安裝依賴 (首次運行時)
# pip install pymc pytensor jax jaxlib numpy pandas scipy

# 設置環境變數
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_MAX_THREADS=16

# 運行分析
echo "開始貝葉斯分析..."
python hpc_example.py

echo "分析完成"
"""
    
    return batch_script


if __name__ == "__main__":
    hpc_example()
    
    # 顯示 batch 腳本
    print("\n" + "=" * 60)
    print("📄 HPC Batch 腳本示例:")
    print("=" * 60)
    print(create_hpc_batch_script())