#!/usr/bin/env python3
"""
測試 xarray 兼容性修復
Test xarray compatibility fixes
"""

import os
import sys
import numpy as np
import warnings

# 設置環境變數
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float32,force_device=True,mode=FAST_RUN,optimizer=fast_compile,cxx="
os.environ["PYTENSOR_CXX"] = ""
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OMP_NUM_THREADS"] = "1"

# 抑制警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*axis.*dim.*')

def test_bayesian_analyzer():
    """測試修復後的 Bayesian 分析器"""
    
    print("🧪 測試修復後的 Bayesian 分析器...")
    
    try:
        from bayesian import RobustBayesianAnalyzer
        from skill_scores.basis_risk_functions import BasisRiskType
        
        # 創建測試數據
        np.random.seed(42)
        n_train = 20
        n_validation = 10
        n_loss_scenarios = 50
        
        train_losses = np.random.lognormal(15, 1.5, n_train)
        validation_losses = np.random.lognormal(15, 1.5, n_validation)
        hazard_indices = 25 + np.random.uniform(0, 40, n_train)
        
        # 創建損失情境矩陣
        actual_losses_matrix = np.zeros((n_loss_scenarios, n_train))
        for i in range(n_loss_scenarios):
            scenario_factor = np.random.lognormal(0, 0.3)
            actual_losses_matrix[i, :] = train_losses * scenario_factor
        
        # 產品邊界
        product_bounds = {
            'trigger_threshold': (25, 70),
            'payout_amount': (1e7, 1e9),
            'max_payout': (2e9, 2e9)
        }
        
        # 配置
        pymc_config = {
            'pymc_backend': 'cpu',
            'pymc_mode': 'FAST_RUN', 
            'n_threads': 1,
            'configure_pymc': False
        }
        
        print("✅ 測試數據準備完成")
        print(f"   訓練樣本: {len(train_losses)}")
        print(f"   驗證樣本: {len(validation_losses)}")
        print(f"   損失情境: {n_loss_scenarios}")
        
        # 初始化分析器
        print("\n🚀 初始化 Bayesian 分析器...")
        analyzer = RobustBayesianAnalyzer(
            density_ratio_constraint=2.0,
            n_monte_carlo_samples=100,
            n_mixture_components=2
        )
        
        print("✅ 分析器初始化成功")
        
        # 測試整合最佳化
        print("\n🎯 執行整合最佳化 (簡化版)...")
        
        results = analyzer.integrated_bayesian_optimization(
            observations=train_losses,
            validation_data=validation_losses,
            hazard_indices=hazard_indices,
            actual_losses=actual_losses_matrix,
            product_bounds=product_bounds,
            basis_risk_type=BasisRiskType.WEIGHTED_ASYMMETRIC,
            w_under=2.0,
            w_over=0.5,
            **pymc_config
        )
        
        print("\n🎉 測試成功完成！")
        print(f"   結果包含: {list(results.keys())}")
        
        # 顯示結果摘要
        if 'phase_1_model_comparison' in results:
            phase1 = results['phase_1_model_comparison']
            print(f"   冠軍模型: {phase1['champion_model']['name']}")
            
        if 'phase_2_decision_optimization' in results:
            phase2 = results['phase_2_decision_optimization']
            print(f"   最佳觸發閾值: {phase2['optimal_product']['trigger_threshold']:.1f}")
            print(f"   最佳賠付金額: ${phase2['optimal_product']['payout_amount']/1e9:.3f}B")
            
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hierarchical_model():
    """測試修復後的階層模型"""
    
    print("\n🧪 測試階層貝葉斯模型...")
    
    try:
        from bayesian.hierarchical_bayesian_model import HierarchicalBayesianModel
        
        # 創建測試數據
        np.random.seed(42)
        observations = np.random.lognormal(15, 1, 20)
        
        print("✅ 測試數據準備完成")
        print(f"   觀測數: {len(observations)}")
        print(f"   數據範圍: ${np.min(observations)/1e9:.3f}B - ${np.max(observations)/1e9:.3f}B")
        
        # 初始化模型
        print("\n🔧 初始化階層模型...")
        
        class TestConfig:
            n_samples = 100
            n_chains = 1
            n_warmup = 50
        
        model = HierarchicalBayesianModel(TestConfig())
        
        # 測試擬合
        print("\n⚙️ 執行模型擬合...")
        
        result = model.fit_hierarchical_model(observations)
        
        print("✅ 階層模型測試成功！")
        print(f"   後驗樣本數: {len(result['posterior_samples']['alpha'])}")
        print(f"   對數似然: {result['log_likelihood']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 階層模型測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 xarray 兼容性修復驗證")
    print("=" * 50)
    
    # 測試環境
    try:
        import pymc as pm
        import pytensor
        import xarray as xr
        print(f"✅ PyMC: {pm.__version__}")
        print(f"✅ PyTensor: {pytensor.__version__}")
        print(f"✅ xarray: {xr.__version__}")
    except ImportError as e:
        print(f"❌ 環境檢查失敗: {e}")
        exit(1)
    
    print("\n" + "=" * 50)
    
    # 執行測試
    success = True
    
    # 測試1: Bayesian 分析器
    if test_bayesian_analyzer():
        print("✅ Bayesian 分析器測試通過")
    else:
        print("❌ Bayesian 分析器測試失敗")
        success = False
    
    # 測試2: 階層模型
    if test_hierarchical_model():
        print("✅ 階層模型測試通過")
    else:
        print("❌ 階層模型測試失敗")
        success = False
    
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 所有測試通過！xarray 兼容性問題已修復")
        print("\n📋 現在可以安全地使用:")
        print("1. CLIMADA + Bayesian 整合 notebook")
        print("2. nc_tc_comprehensive_functional.py")
        print("3. 所有 Bayesian 分析功能")
    else:
        print("⚠️ 部分測試失敗，可能仍有兼容性問題")
        print("建議檢查套件版本或使用簡化分析模式")