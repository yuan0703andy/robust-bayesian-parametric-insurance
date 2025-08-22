#!/usr/bin/env python3
"""
Contamination Tests Module
污染測試模組

提供污染理論和雙重contamination的測試功能
確保robust_priors模組的工作流程函數正常運行

Author: Research Team
Date: 2025-08-22
"""

import numpy as np
from typing import Dict, Any, Optional
import warnings

def quick_contamination_test() -> bool:
    """
    快速污染測試
    
    Returns:
    --------
    bool : 測試是否通過
    """
    try:
        # 簡單的功能測試
        from .contamination_core import DoubleEpsilonContamination, EpsilonContaminationSpec
        
        # 測試基本功能
        spec = EpsilonContaminationSpec()
        double_contam = DoubleEpsilonContamination()
        
        # 生成測試數據
        test_data = np.random.lognormal(15, 1, 50)
        base_prior = {'location': np.mean(test_data), 'scale': np.std(test_data)}
        
        # 測試雙重污染功能
        robust_posterior = double_contam.compute_robust_posterior(
            data=test_data,
            base_prior_params=base_prior,
            likelihood_params={}
        )
        
        return robust_posterior is not None and 'posterior_mean' in robust_posterior
        
    except Exception as e:
        warnings.warn(f"Quick contamination test failed: {e}")
        return False

def test_contamination_core() -> Dict[str, bool]:
    """測試contamination_core功能"""
    results = {
        'double_contamination': False,
        'contamination_spec': False,
        'mixed_distribution': False
    }
    
    try:
        from .contamination_core import (
            DoubleEpsilonContamination,
            EpsilonContaminationSpec,
            create_mixed_distribution
        )
        
        # 測試EpsilonContaminationSpec
        spec = EpsilonContaminationSpec()
        results['contamination_spec'] = spec.epsilon_range == (0.01, 0.20)
        
        # 測試DoubleEpsilonContamination
        double_contam = DoubleEpsilonContamination(epsilon_prior=0.1, epsilon_likelihood=0.05)
        test_data = np.random.lognormal(15, 1, 30)
        base_prior = {'location': np.mean(test_data), 'scale': np.std(test_data)}
        
        robust_posterior = double_contam.compute_robust_posterior(
            data=test_data, base_prior_params=base_prior, likelihood_params={}
        )
        results['double_contamination'] = 'posterior_mean' in robust_posterior
        
        # 測試mixed distribution (簡化版)
        from scipy import stats
        base_dist = stats.norm(0, 1)
        contam_dist = stats.norm(3, 2)
        mixed = create_mixed_distribution(base_dist, contam_dist, 0.1)
        samples = mixed.rvs(100)
        results['mixed_distribution'] = len(samples) == 100
        
    except Exception as e:
        warnings.warn(f"Contamination core test failed: {e}")
    
    return results

def test_epsilon_estimation() -> Dict[str, bool]:
    """測試epsilon_estimation功能"""
    results = {
        'epsilon_estimator': False,
        'prior_contamination_analyzer': False
    }
    
    try:
        from .epsilon_estimation import EpsilonEstimator, PriorContaminationAnalyzer
        from .contamination_core import EpsilonContaminationSpec
        
        spec = EpsilonContaminationSpec()
        
        # 測試EpsilonEstimator  
        estimator = EpsilonEstimator(spec)
        results['epsilon_estimator'] = estimator is not None
        
        # 測試PriorContaminationAnalyzer
        prior_analyzer = PriorContaminationAnalyzer(spec)
        results['prior_contamination_analyzer'] = prior_analyzer is not None
        
    except Exception as e:
        warnings.warn(f"Epsilon estimation test failed: {e}")
        
    return results

def test_contamination_integration() -> bool:
    """測試contamination integration"""
    try:
        # 測試完整工作流程的核心組件
        from .contamination_core import DoubleEpsilonContamination
        from .epsilon_estimation import EpsilonEstimator
        
        # 簡單集成測試
        double_contam = DoubleEpsilonContamination(epsilon_prior=0.08, epsilon_likelihood=0.12)
        test_data = np.random.lognormal(15, 1, 40)
        
        contaminated_data = double_contam.create_contaminated_likelihood(test_data)
        
        return len(contaminated_data) > 0
        
    except Exception as e:
        warnings.warn(f"Contamination integration test failed: {e}")
        return False

def run_all_contamination_tests() -> Dict[str, Any]:
    """執行所有污染測試"""
    
    print("🧪 執行污染理論測試套件...")
    
    results = {
        'quick_test': quick_contamination_test(),
        'core_tests': test_contamination_core(), 
        'estimation_tests': test_epsilon_estimation(),
        'integration_test': test_contamination_integration()
    }
    
    # 計算總體狀態
    core_passed = sum(results['core_tests'].values())
    estimation_passed = sum(results['estimation_tests'].values())
    
    results['summary'] = {
        'quick_test_passed': results['quick_test'],
        'core_tests_passed': f"{core_passed}/{len(results['core_tests'])}",
        'estimation_tests_passed': f"{estimation_passed}/{len(results['estimation_tests'])}",  
        'integration_test_passed': results['integration_test'],
        'overall_status': 'healthy' if all([
            results['quick_test'],
            core_passed >= 2,  # 至少2個核心測試通過
            results['integration_test']
        ]) else 'partial'
    }
    
    print(f"📊 測試結果摘要:")
    print(f"   快速測試: {'✅' if results['quick_test'] else '❌'}")
    print(f"   核心功能: {core_passed}/{len(results['core_tests'])} 通過")
    print(f"   估計功能: {estimation_passed}/{len(results['estimation_tests'])} 通過") 
    print(f"   集成測試: {'✅' if results['integration_test'] else '❌'}")
    print(f"   總體狀態: {results['summary']['overall_status']}")
    
    return results

# 模組導出
__all__ = [
    'quick_contamination_test',
    'test_contamination_core', 
    'test_epsilon_estimation',
    'test_contamination_integration',
    'run_all_contamination_tests'
]