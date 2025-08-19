#!/usr/bin/env python3
"""
Test Script for 32-Core CPU + 2-GPU HPC Framework
測試32核CPU + 2GPU HPC框架的腳本

This script validates the HPC framework integration and performance
本腳本驗證HPC框架整合與效能

Author: Research Team
Date: 2025-01-18
"""

import sys
import time
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_pytorch_mcmc_import():
    """測試PyTorch MCMC模組導入"""
    print("🧪 測試1: PyTorch MCMC模組導入")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pytorch_mcmc", 
            "robust_hierarchical_bayesian_simulation/6_mcmc_validation/pytorch_mcmc.py"
        )
        pytorch_mcmc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pytorch_mcmc_module)
        
        # 檢查是否有必要的函數和類
        assert hasattr(pytorch_mcmc_module, 'run_pytorch_mcmc'), "缺少 run_pytorch_mcmc 函數"
        assert hasattr(pytorch_mcmc_module, 'BayesianHierarchicalMCMC'), "缺少 BayesianHierarchicalMCMC 類"
        assert hasattr(pytorch_mcmc_module, 'MCMCConfig'), "缺少 MCMCConfig 類"
        
        print("   ✅ PyTorch MCMC模組導入成功")
        return True
    except Exception as e:
        print(f"   ❌ PyTorch MCMC模組導入失敗: {e}")
        return False

def test_gpu_config_import():
    """測試GPU配置模組導入"""
    print("\n🧪 測試2: GPU配置模組導入")
    try:
        from robust_hierarchical_bayesian_simulation.gpu_setup.gpu_config import (
            GPUConfig, setup_gpu_environment
        )
        print("   ✅ GPU配置模組導入成功")
        return True
    except Exception as e:
        print(f"   ❌ GPU配置模組導入失敗: {e}")
        return False

def test_hpc_framework_import():
    """測試HPC框架導入"""
    print("\n🧪 測試3: HPC框架導入")
    try:
        from multiprocessing import set_start_method
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        # 導入主框架
        from robust_bayesian_32core_2gpu import RobustBayesianHPC
        print("   ✅ HPC框架導入成功")
        return True
    except Exception as e:
        print(f"   ❌ HPC框架導入失敗: {e}")
        print(f"   詳細錯誤: {traceback.format_exc()}")
        return False

def test_framework_initialization():
    """測試框架初始化"""
    print("\n🧪 測試4: 框架初始化")
    try:
        from multiprocessing import set_start_method
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        from robust_bayesian_32core_2gpu import RobustBayesianHPC
        
        framework = RobustBayesianHPC()
        
        # 檢查組件
        assert hasattr(framework, 'config'), "缺少配置對象"
        assert hasattr(framework, 'gpu_manager'), "缺少GPU管理器"
        assert framework.config.n_cpu_cores == 32, f"CPU核心數不正確: {framework.config.n_cpu_cores}"
        assert framework.config.n_gpu == 2, f"GPU數量不正確: {framework.config.n_gpu}"
        
        print("   ✅ 框架初始化成功")
        print(f"     CPU核心: {framework.config.n_cpu_cores}")
        print(f"     GPU數量: {framework.config.n_gpu}")
        print(f"     數據規模: {framework.config.large_dataset_size:,}")
        print(f"     模型空間: {framework.config.model_space_size:,}")
        
        return framework
    except Exception as e:
        print(f"   ❌ 框架初始化失敗: {e}")
        print(f"   詳細錯誤: {traceback.format_exc()}")
        return None

def test_small_workflow(framework):
    """測試小規模工作流程"""
    print("\n🧪 測試5: 小規模工作流程 (快速驗證)")
    try:
        # 降低數據規模以加快測試
        original_dataset_size = framework.config.large_dataset_size
        original_model_space_size = framework.config.model_space_size
        
        framework.config.large_dataset_size = 1000  # 1K數據點
        framework.config.model_space_size = 50      # 50個模型
        
        print(f"   📊 測試規模: {framework.config.large_dataset_size:,} 觀測, {framework.config.model_space_size} 模型")
        
        start_time = time.time()
        
        # 執行各階段
        framework.stage1_massive_data_processing()
        framework.stage2_comprehensive_model_selection()
        framework.stage3_intensive_hyperparameter_optimization()
        
        # 只測試2個模型的MCMC
        framework.optimized_models = framework.optimized_models[:2]
        framework.stage4_pytorch_mcmc_validation()
        
        framework.stage5_comprehensive_analysis()
        
        total_time = time.time() - start_time
        
        # 恢復原始配置
        framework.config.large_dataset_size = original_dataset_size
        framework.config.model_space_size = original_model_space_size
        
        print(f"   ✅ 小規模工作流程完成")
        print(f"     執行時間: {total_time:.2f} 秒")
        print(f"     數據處理: ✅")
        print(f"     模型選擇: ✅")
        print(f"     超參數優化: ✅")
        print(f"     MCMC驗證: ✅")
        print(f"     分析完成: ✅")
        
        return True
    except Exception as e:
        print(f"   ❌ 小規模工作流程失敗: {e}")
        print(f"   詳細錯誤: {traceback.format_exc()}")
        return False

def test_pytorch_mcmc_functionality():
    """測試PyTorch MCMC功能"""
    print("\n🧪 測試6: PyTorch MCMC功能驗證")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pytorch_mcmc", 
            "robust_hierarchical_bayesian_simulation/6_mcmc_validation/pytorch_mcmc.py"
        )
        pytorch_mcmc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pytorch_mcmc_module)
        
        print("   🚀 執行PyTorch MCMC測試...")
        pytorch_mcmc_module.test_pytorch_mcmc()
        print("   ✅ PyTorch MCMC功能驗證成功")
        return True
    except Exception as e:
        print(f"   ❌ PyTorch MCMC功能驗證失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🚀 32核CPU + 2GPU HPC框架完整測試")
    print("=" * 60)
    
    test_results = []
    
    # 執行所有測試
    test_results.append(("PyTorch MCMC導入", test_pytorch_mcmc_import()))
    test_results.append(("GPU配置導入", test_gpu_config_import()))
    test_results.append(("HPC框架導入", test_hpc_framework_import()))
    
    framework = test_framework_initialization()
    test_results.append(("框架初始化", framework is not None))
    
    if framework:
        test_results.append(("小規模工作流程", test_small_workflow(framework)))
    
    test_results.append(("PyTorch MCMC功能", test_pytorch_mcmc_functionality()))
    
    # 總結測試結果
    print("\n" + "=" * 60)
    print("📋 測試結果總結")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"   {test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\n🎯 總體結果: {passed_tests}/{total_tests} 測試通過")
    
    if passed_tests == total_tests:
        print("🎉 所有測試通過！HPC框架準備就緒")
        print("   ✅ PyTorch MCMC整合完成")
        print("   ✅ 32核CPU + 2GPU配置就緒")
        print("   ✅ 大規模並行處理支援")
        print("   ✅ GPU智能任務分配")
        return True
    else:
        print("⚠️ 部分測試失敗，請檢查配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)