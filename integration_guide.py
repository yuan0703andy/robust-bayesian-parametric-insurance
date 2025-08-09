#!/usr/bin/env python3
"""
Bayesian 模組與 nc_tc_comprehensive_functional.py 整合指南
Integration Guide for Bayesian Module with nc_tc_comprehensive_functional.py

本文件提供具體的整合步驟和代碼示例
This file provides specific integration steps and code examples
"""

import numpy as np

# ============================================================================
# 第一步：更新 nc_tc_comprehensive_functional.py 的導入部分
# Step 1: Update imports in nc_tc_comprehensive_functional.py
# ============================================================================

def update_bayesian_imports():
    """更新 Bayesian 模組導入"""
    
    updated_import_code = '''
# 進階模組 - 更新版本
try:
    # 新的統一介面
    from bayesian import RobustBayesianAnalyzer
    
    # PyMC 配置模組 (新增)
    from bayesian.pymc_config import configure_pymc_environment, verify_pymc_setup
    
    # 基差風險函數 (新位置)
    from skill_scores.basis_risk_functions import BasisRiskType
    
    modules_available['bayesian'] = True
    print("   ✅ 貝氏分析模組 (v2.0 - 整合版本)")
    print("   ✅ PyMC 配置模組")
    print("   ✅ 基差風險函數模組")
    
    # 驗證 PyMC 環境
    print("   🔧 驗證 PyMC 環境...")
    pymc_setup = verify_pymc_setup()
    if pymc_setup['setup_correct']:
        print("   ✅ PyMC 環境設置正確")
    else:
        print("   ⚠️ PyMC 環境需要調整，但可繼續使用")
        
except ImportError as e:
    modules_available['bayesian'] = False
    print(f"   ⚠️ 貝氏分析模組不可用: {e}")
    '''
    
    return updated_import_code


# ============================================================================
# 第二步：替換舊的 comprehensive_bayesian_analysis 調用
# Step 2: Replace old comprehensive_bayesian_analysis calls
# ============================================================================

def create_new_bayesian_integration():
    """創建新的 Bayesian 整合代碼"""
    
    new_integration_code = '''
# 🧠 新的整合貝葉斯分析 (替換第1242-1350行)
if modules_available['bayesian'] and 'tc_hazard' in main_data and 'exposure' in main_data:
    try:
        print("   🚀 啟動新版穩健貝氏分析器...")
        
        # 初始化新版分析器
        bayesian_analyzer = RobustBayesianAnalyzer(
            density_ratio_constraint=2.0,
            n_monte_carlo_samples=n_samples,
            n_mixture_components=3
        )
        
        # 準備數據
        print("      📊 準備分析數據...")
        
        # 使用醫院或主要曝險數據
        if 'hospital_exposures' in locals() and hospital_exposures is not None:
            exposure_for_bayesian = hospital_exposures
            print(f"      🎯 使用 {len(exposure_for_bayesian.gdf)} 個醫院點進行建模")
        else:
            exposure_for_bayesian = exposure_main
            print(f"      ⚠️ 使用完整LitPop數據 ({len(exposure_for_bayesian.gdf)} 點)")
        
        # 確保損失數據格式正確
        damages_array = np.array(damages, dtype=np.float64)
        n_events = len(damages_array)
        
        # 分割數據用於兩階段分析
        n_train = max(int(0.7 * n_events), min(100, n_events - 20))
        n_validation = n_events - n_train
        
        train_losses = damages_array[:n_train]
        validation_losses = damages_array[n_train:]
        
        print(f"      📋 數據分割: 訓練({n_train}) / 驗證({n_validation})")
        
        # 創建損失情境矩陣
        n_scenarios = min(n_samples, 500)  # 合理的情境數
        actual_losses_matrix = np.zeros((n_scenarios, n_train))
        
        # 基於風險指標生成損失情境
        if 'hospital_wind_series' in locals() and hospital_wind_series:
            # 使用醫院風速數據
            base_winds = np.array(list(hospital_wind_series.values())).mean(axis=0)[:n_train]
        else:
            # 使用模擬風速數據
            base_winds = np.random.uniform(25, 65, n_train)
        
        for i in range(n_scenarios):
            scenario_factor = np.random.lognormal(0, 0.3)  # 情境不確定性
            actual_losses_matrix[i, :] = train_losses * scenario_factor
        
        print(f"      🎲 生成了 {n_scenarios} 個損失情境")
        
        # 定義產品參數邊界 (基於現有產品範圍)
        product_bounds = {
            'trigger_threshold': (25, 70),      # 基於風速範圍
            'payout_amount': (1e7, 1e9),       # 合理的賠付範圍
            'max_payout': (2e9, 2e9)           # 最大賠付限制
        }
        
        print("      🎯 執行新的整合貝葉斯最佳化...")
        
        # 🚀 使用新的整合方法 (方法一 + 方法二)
        bayesian_results = bayesian_analyzer.integrated_bayesian_optimization(
            observations=train_losses,
            validation_data=validation_losses,
            hazard_indices=base_winds,
            actual_losses=actual_losses_matrix,
            product_bounds=product_bounds,
            basis_risk_type=BasisRiskType.WEIGHTED_ASYMMETRIC,
            w_under=2.0,
            w_over=0.5,
            # PyMC 配置 (適合不同環境)
            pymc_backend="cpu",           # 或 "gpu" 在 HPC 上
            pymc_mode="FAST_COMPILE",     # 或 "FAST_RUN" 在生產環境
            n_threads=1,                  # 或更多在 HPC 上
            configure_pymc=True
        )
        
        print("   ✅ 新版貝氏分析完成！")
        
        # 提取結果
        phase1_results = bayesian_results['phase_1_model_comparison']
        phase2_results = bayesian_results['phase_2_decision_optimization']
        
        print(f"      🏆 冠軍模型: {phase1_results['champion_model']['name']}")
        print(f"      🎯 最佳觸發閾值: {phase2_results['optimal_product']['trigger_threshold']:.1f}")
        print(f"      💰 最佳賠付金額: ${phase2_results['optimal_product']['payout_amount']:.1e}")
        print(f"      📉 期望基差風險: ${phase2_results['expected_basis_risk']:.1e}")
        
        # 生成與現有系統兼容的結果格式
        bayesian_optimal_product = {
            'product_id': f"bayesian_optimal",
            'trigger_threshold': phase2_results['optimal_product']['trigger_threshold'],
            'payout_amount': phase2_results['optimal_product']['payout_amount'],
            'max_payout': phase2_results['optimal_product'].get('max_payout', phase2_results['optimal_product']['payout_amount']),
            'method': 'integrated_bayesian_optimization',
            'champion_model': phase1_results['champion_model']['name'],
            'crps_score': phase1_results['champion_model']['crps_score'],
            'expected_basis_risk': phase2_results['expected_basis_risk']
        }
        
        # 創建機率損失分布以兼容現有代碼
        event_loss_distributions = {}
        for event_idx in range(n_train):
            event_loss_distributions[f'event_{event_idx}'] = {
                'mean': train_losses[event_idx],
                'std': train_losses[event_idx] * 0.3,  # 假設30%變異
                'samples': actual_losses_matrix[:, event_idx].tolist(),
                'percentiles': {
                    '5th': np.percentile(actual_losses_matrix[:, event_idx], 5),
                    '95th': np.percentile(actual_losses_matrix[:, event_idx], 95)
                }
            }
        
        loss_distributions = event_loss_distributions
        
        print(f"   ✅ 生成了 {len(loss_distributions)} 個事件的機率性損失分布")
        print(f"      📊 每個分布包含 {n_scenarios} 個樣本")
        
    except Exception as e:
        print(f"   ❌ 新版貝氏分析失敗: {e}")
        print("      繼續使用傳統方法...")
        modules_available['bayesian'] = False
        
else:
    print("   ⚠️ 跳過貝氏分析 (模組不可用或數據未準備)")
    '''
    
    return new_integration_code


# ============================================================================
# 第三步：更新結果整合部分
# Step 3: Update results integration
# ============================================================================

def update_results_integration():
    """更新結果整合代碼"""
    
    updated_results_code = '''
# 🔄 更新結果整合 (替換第1490-1510行)

if modules_available['bayesian'] and 'bayesian_optimal_product' in locals():
    # 使用新的最佳化產品
    bayesian_best = bayesian_optimal_product.copy()
    
    # 計算與現有產品的CRPS比較
    if 'crps_df' in locals() and len(crps_df) > 0:
        # 添加新的最佳化產品到比較中
        print(f"   🎯 比較新最佳化產品與現有產品...")
        
        # 模擬新產品的賠付
        optimal_payouts = []
        for wind in base_winds:
            if wind >= bayesian_best['trigger_threshold']:
                payout = min(bayesian_best['payout_amount'], bayesian_best['max_payout'])
            else:
                payout = 0.0
            optimal_payouts.append(payout)
        
        optimal_payouts = np.array(optimal_payouts)
        
        # 計算新產品的CRPS
        from insurance_analysis_refactored.core.parametric_engine import calculate_crps_score
        
        optimal_crps = calculate_crps_score(train_losses, optimal_payouts, list(loss_distributions.values()))
        
        bayesian_best.update({
            'crps': optimal_crps,
            'payouts': optimal_payouts,
            'correlation': np.corrcoef(train_losses, optimal_payouts)[0, 1] if len(optimal_payouts) > 1 else 0,
            'trigger_rate': np.mean(optimal_payouts > 0)
        })
        
        print(f"      ✅ 新最佳化產品CRPS: ${optimal_crps/1e9:.3f}B")
        print(f"      📊 觸發率: {bayesian_best['trigger_rate']:.1%}")
    
    else:
        # 從原始分析結果提取
        best_crps_idx = 0  # 默認
        if 'crps_df' in locals() and len(crps_df) > 0:
            best_crps_idx = crps_df['crps'].idxmin()
            bayesian_best.update(crps_df.iloc[best_crps_idx].to_dict())

print(f"   ✅ 新版貝氏分析完成")
if 'bayesian_best' in locals():
    print(f"      最佳產品: {bayesian_best.get('product_id', 'bayesian_optimal')}")
    print(f"      期望基差風險: ${bayesian_best.get('expected_basis_risk', 0)/1e9:.3f}B")
    print(f"      冠軍模型: {bayesian_best.get('champion_model', 'unknown')}")
    '''
    
    return updated_results_code


# ============================================================================
# 第四步：HPC 環境配置
# Step 4: HPC Environment Configuration
# ============================================================================

def create_hpc_config_section():
    """創建 HPC 環境配置部分"""
    
    hpc_config_code = '''
# %% HPC/OnDemand 環境配置 (添加在文件開頭，第16行之後)

# 檢測運行環境
def detect_environment():
    """檢測運行環境類型"""
    import os
    
    if 'SLURM_JOB_ID' in os.environ:
        return 'hpc_slurm'
    elif 'PBS_JOBID' in os.environ:
        return 'hpc_pbs'
    elif 'OOD_' in str(os.environ):
        return 'ondemand'
    else:
        return 'local'

# 環境配置
run_environment = detect_environment()
print(f"🌐 檢測到運行環境: {run_environment}")

# 根據環境設置 PyMC 配置
if run_environment in ['hpc_slurm', 'hpc_pbs']:
    # HPC 環境配置
    pymc_config = {
        'backend': 'cpu',        # HPC 通常用 CPU，除非有 GPU 節點
        'mode': 'FAST_RUN',      # 生產環境用快速運行
        'n_threads': int(os.environ.get('OMP_NUM_THREADS', 8)),  # 使用節點核心數
    }
    print(f"   🖥️ HPC 配置: {pymc_config}")
    
elif run_environment == 'ondemand':
    # OnDemand 環境配置
    pymc_config = {
        'backend': 'cpu',
        'mode': 'FAST_COMPILE',  # 交互式環境用快速編譯
        'n_threads': 4,
    }
    print(f"   🌐 OnDemand 配置: {pymc_config}")
    
else:
    # 本地環境配置 (macOS 等)
    pymc_config = {
        'backend': 'cpu',        # 避免 Metal 問題
        'mode': 'FAST_COMPILE',
        'n_threads': 1,
    }
    print(f"   💻 本地配置: {pymc_config}")

# 設置分析參數
if run_environment in ['hpc_slurm', 'hpc_pbs']:
    # HPC 上可以用更多資源
    n_samples = 1000
    n_monte_carlo = 1000
    n_loss_scenarios = 500
else:
    # 本地和 OnDemand 使用較少資源
    n_samples = 500
    n_monte_carlo = 500
    n_loss_scenarios = 200

print(f"   📊 分析參數: samples={n_samples}, monte_carlo={n_monte_carlo}, scenarios={n_loss_scenarios}")
'''
    
    return hpc_config_code


# ============================================================================
# 第五步：完整整合步驟
# Step 5: Complete Integration Steps
# ============================================================================

def create_integration_instructions():
    """創建完整的整合指令"""
    
    instructions = """
🔧 Bayesian 模組整合步驟

1. 📝 更新導入部分 (第72-80行):
   - 替換為 update_bayesian_imports() 的代碼

2. 🔄 替換 Bayesian 分析部分 (第1242-1350行):
   - 使用 create_new_bayesian_integration() 的代碼
   - 這會使用新的 integrated_bayesian_optimization 方法

3. 📊 更新結果整合 (第1490-1510行):
   - 使用 update_results_integration() 的代碼

4. 🌐 添加環境配置 (第16行之後):
   - 使用 create_hpc_config_section() 的代碼

5. ⚙️ 確保依賴模組:
   - skill_scores/basis_risk_functions.py 存在
   - bayesian/pymc_config.py 存在

6. 🧪 測試整合:
   - 先在本地測試 (configure_pymc=True, pymc_backend='cpu')
   - 再到 HPC 測試 (根據環境調整參數)

7. 📈 預期改進:
   - 使用正確的理論框架 (方法一 → 方法二)
   - 動態 PyMC 配置適應不同環境
   - 更好的基差風險最小化
   - 冠軍模型自動選擇和使用

8. 🐛 故障排除:
   - 如果 PyMC 錯誤 → 檢查環境配置
   - 如果記憶體不足 → 減少樣本數
   - 如果慢 → 使用 'FAST_COMPILE' 模式
   """
    
    return instructions


# ============================================================================
# 示例：完整的整合代碼片段
# Example: Complete Integration Code Snippet
# ============================================================================

def generate_complete_integration_example():
    """生成完整的整合示例"""
    
    example_code = '''
# 這是一個完整的整合示例，展示如何在 nc_tc_comprehensive_functional.py 中
# 使用新的 Bayesian 模組

# 在適當的位置 (第1242行左右) 替換現有的 Bayesian 分析代碼:

if modules_available['bayesian'] and 'tc_hazard' in main_data and 'exposure' in main_data:
    try:
        print("   🚀 啟動新版整合貝葉斯分析...")
        
        # 初始化分析器
        bayesian_analyzer = RobustBayesianAnalyzer(
            density_ratio_constraint=2.0,
            n_monte_carlo_samples=n_monte_carlo,
            n_mixture_components=3
        )
        
        # 數據準備
        damages_array = np.array(damages, dtype=np.float64)
        n_events = len(damages_array)
        n_train = int(0.7 * n_events)
        
        # 創建風險指標
        if 'hospital_wind_series' in locals() and hospital_wind_series:
            hazard_indices = np.array(list(hospital_wind_series.values())).mean(axis=0)[:n_train]
        else:
            hazard_indices = np.random.uniform(25, 65, n_train)
        
        # 創建損失情境
        actual_losses_matrix = np.zeros((n_loss_scenarios, n_train))
        for i in range(n_loss_scenarios):
            scenario_factor = np.random.lognormal(0, 0.3)
            actual_losses_matrix[i, :] = damages_array[:n_train] * scenario_factor
        
        # 執行整合最佳化
        bayesian_results = bayesian_analyzer.integrated_bayesian_optimization(
            observations=damages_array[:n_train],
            validation_data=damages_array[n_train:],
            hazard_indices=hazard_indices,
            actual_losses=actual_losses_matrix,
            product_bounds={
                'trigger_threshold': (25, 70),
                'payout_amount': (1e7, 1e9),
                'max_payout': (2e9, 2e9)
            },
            basis_risk_type=BasisRiskType.WEIGHTED_ASYMMETRIC,
            # 使用環境配置
            **pymc_config,
            configure_pymc=True
        )
        
        # 提取最佳產品
        optimal_product = bayesian_results['phase_2_decision_optimization']['optimal_product']
        
        print(f"   ✅ 整合分析完成！最佳觸發閾值: {optimal_product['trigger_threshold']:.1f}")
        
    except Exception as e:
        print(f"   ❌ 整合分析失敗: {e}")
        modules_available['bayesian'] = False
    '''
    
    return example_code


if __name__ == "__main__":
    print("🔧 Bayesian 模組整合指南")
    print("=" * 50)
    
    print("\n📋 整合指令:")
    print(create_integration_instructions())
    
    print("\n💾 儲存所有更新代碼到:")
    print("   - update_imports.py")
    print("   - new_bayesian_integration.py")  
    print("   - update_results.py")
    print("   - hpc_config.py")