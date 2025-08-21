#!/usr/bin/env python3
"""
Robust Priors Module
穩健先驗模組

重組後的ε-contamination穩健貝氏建模模組
提供統一、整潔的接口用於污染理論分析

模組結構:
- contamination_core.py:    核心理論與基本實現
- epsilon_estimation.py:    ε值估計功能合集  
- contamination_tests.py:   統一測試模組

核心功能:
- ε-contamination理論實現
- 雙重污染模型 (Prior + Likelihood)
- 多種ε值估計方法
- 先驗穩健性分析
- 污染分布生成器

Author: Research Team  
Date: 2025-08-19
Version: 2.0.0 (重組版本)
"""

# 版本信息
__version__ = "2.0.0"
__author__ = "Robust Bayesian Research Team"

# 從核心模組導入主要類別和函數
try:
    from .contamination_core import (
        # 類型定義
        ContaminationDistributionClass,
        RobustnessCriterion, 
        EstimationMethod,
        
        # 配置結構
        EpsilonContaminationSpec,
        ContaminationEstimateResult,
        RobustPosteriorResult,
        
        # 理論函數
        contamination_bound,
        worst_case_risk,
        compute_robustness_radius,
        create_mixed_distribution,
        
        # 核心類別
        ContaminationDistributionGenerator,
        DoubleEpsilonContamination,
        
        # 便利函數
        create_typhoon_contamination_spec,
        demonstrate_dual_process_nature
    )
    
    from .epsilon_estimation import (
        # 主要估計器
        EpsilonEstimator,
        PriorContaminationAnalyzer,
        
        # 便利函數
        quick_contamination_analysis,
        compare_estimation_methods
    )
    
    from .contamination_tests import (
        # 主要測試函數
        run_all_contamination_tests,
        quick_contamination_test,
        
        # 特定測試
        test_contamination_core,
        test_epsilon_estimation,
        test_contamination_integration
    )
    
    IMPORT_SUCCESS = True
    
except ImportError as e:
    # 如果導入失敗，提供fallback
    import warnings
    warnings.warn(f"部分模組導入失敗: {e}")
    IMPORT_SUCCESS = False

# 設置模組級別的便利接口
def create_contamination_analyzer(epsilon_range=(0.01, 0.20), 
                                contamination_type="typhoon_specific"):
    """
    創建便利的污染分析器
    
    Parameters:
    -----------
    epsilon_range : tuple
        ε值範圍
    contamination_type : str
        污染類型
        
    Returns:
    --------
    tuple
        (EpsilonEstimator, PriorContaminationAnalyzer)
    """
    if not IMPORT_SUCCESS:
        raise ImportError("模組導入失敗，無法創建分析器")
    
    # 創建規格
    contamination_class = ContaminationDistributionClass(contamination_type)
    spec = EpsilonContaminationSpec(
        epsilon_range=epsilon_range,
        contamination_class=contamination_class
    )
    
    # 創建分析器
    estimator = EpsilonEstimator(spec)
    prior_analyzer = PriorContaminationAnalyzer(spec)
    
    return estimator, prior_analyzer

def run_basic_contamination_workflow(data, wind_data=None, verbose=True):
    """
    執行基本污染分析工作流程
    
    Parameters:
    -----------
    data : array-like
        觀測數據
    wind_data : array-like, optional
        風速數據
    verbose : bool
        是否顯示詳細信息
        
    Returns:
    --------
    dict
        完整分析結果
    """
    if not IMPORT_SUCCESS:
        raise ImportError("模組導入失敗，無法執行工作流程")
    
    if verbose:
        print("🌀 執行基本污染分析工作流程...")
    
    results = {}
    
    # Step 1: 快速ε估計
    if verbose:
        print("   Step 1: 快速ε估計")
    contamination_result = quick_contamination_analysis(data, wind_data)
    results['epsilon_analysis'] = contamination_result
    
    # Step 2: 雙重過程驗證
    if verbose:
        print("   Step 2: 雙重過程驗證")
    dual_process = demonstrate_dual_process_nature(data, contamination_result.epsilon_consensus)
    results['dual_process'] = dual_process
    
    # Step 3: 雙重污染分析
    if verbose:
        print("   Step 3: 雙重污染分析")
    double_contam = DoubleEpsilonContamination(
        epsilon_prior=contamination_result.epsilon_consensus,
        epsilon_likelihood=min(0.1, contamination_result.epsilon_consensus * 1.5)
    )
    
    import numpy as np
    base_prior = {
        'location': np.median(data),
        'scale': np.std(data)
    }
    
    robust_posterior = double_contam.compute_robust_posterior(data, base_prior, {})
    results['robust_posterior'] = robust_posterior
    
    # 摘要
    if verbose:
        print(f"✅ 工作流程完成:")
        print(f"   估計ε值: {contamination_result.epsilon_consensus:.4f}")
        print(f"   雙重過程驗證: {'✅' if dual_process['dual_process_validated'] else '❌'}")
        print(f"   穩健後驗均值: {robust_posterior['posterior_mean']:.2f}")
    
    return results

# 檢查模組健康狀態
def check_module_health():
    """檢查模組健康狀態"""
    print("🔍 檢查robust_priors模組健康狀態...")
    
    health_status = {
        'import_success': IMPORT_SUCCESS,
        'core_functions_available': False,
        'estimation_functions_available': False, 
        'test_functions_available': False,
        'overall_status': 'unknown'
    }
    
    if IMPORT_SUCCESS:
        try:
            # 檢查核心功能
            spec = create_typhoon_contamination_spec()
            generator = ContaminationDistributionGenerator()
            health_status['core_functions_available'] = True
            
            # 檢查估計功能
            estimator, prior_analyzer = create_contamination_analyzer()
            health_status['estimation_functions_available'] = True
            
            # 檢查測試功能
            test_result = quick_contamination_test()
            health_status['test_functions_available'] = test_result
            
            # 總體狀態
            if all([
                health_status['core_functions_available'],
                health_status['estimation_functions_available'],
                health_status['test_functions_available']
            ]):
                health_status['overall_status'] = 'healthy'
            else:
                health_status['overall_status'] = 'partial'
                
        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['error'] = str(e)
    else:
        health_status['overall_status'] = 'import_failed'
    
    # 顯示結果
    status_emoji = {
        'healthy': '✅',
        'partial': '⚠️', 
        'error': '❌',
        'import_failed': '💥',
        'unknown': '❓'
    }
    
    emoji = status_emoji.get(health_status['overall_status'], '❓')
    print(f"{emoji} 模組狀態: {health_status['overall_status']}")
    
    for key, value in health_status.items():
        if key != 'overall_status':
            status = '✅' if value else '❌'
            print(f"   {key}: {status}")
    
    return health_status

# 模組導出列表
__all__ = [
    # 核心類型和配置
    'ContaminationDistributionClass',
    'RobustnessCriterion',
    'EstimationMethod',
    'EpsilonContaminationSpec',
    'ContaminationEstimateResult',
    'RobustPosteriorResult',
    
    # 主要類別
    'EpsilonEstimator',
    'PriorContaminationAnalyzer', 
    'DoubleEpsilonContamination',
    'ContaminationDistributionGenerator',
    
    # 理論函數
    'contamination_bound',
    'worst_case_risk',
    'compute_robustness_radius',
    'create_mixed_distribution',
    
    # 便利函數
    'create_typhoon_contamination_spec',
    'demonstrate_dual_process_nature',
    'quick_contamination_analysis',
    'compare_estimation_methods',
    
    # 工作流程函數
    'create_contamination_analyzer',
    'run_basic_contamination_workflow',
    
    # 測試函數
    'run_all_contamination_tests',
    'quick_contamination_test',
    
    # 模組管理
    'check_module_health',
    
    # 模組信息
    '__version__',
    'IMPORT_SUCCESS'
]

# 模組初始化時的健康檢查
if __name__ == "__main__":
    print(f"🌀 Robust Priors Module v{__version__}")
    check_module_health()
else:
    # 靜默導入時只做基本檢查
    if IMPORT_SUCCESS:
        print(f"✅ Robust Priors Module v{__version__} loaded successfully")
    else:
        print(f"⚠️ Robust Priors Module v{__version__} loaded with warnings")