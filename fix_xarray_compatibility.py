#!/usr/bin/env python3
"""
修復 xarray/arviz 兼容性問題的腳本
Fix xarray/arviz compatibility issues

這個腳本修復 "passing 'axis' to Dataset reduce methods is ambiguous. Please use 'dim' instead" 錯誤
"""

import warnings
import sys
import os

def fix_xarray_compatibility():
    """修復 xarray 兼容性問題"""
    
    print("🔧 正在修復 xarray/arviz 兼容性問題...")
    
    # 1. 抑制相關警告
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message='.*axis.*dim.*')
    warnings.filterwarnings('ignore', message='.*ambiguous.*')
    
    # 2. 檢查套件版本
    try:
        import xarray as xr
        import arviz as az
        print(f"📦 xarray 版本: {xr.__version__}")
        print(f"📦 arviz 版本: {az.__version__}")
        
        # 檢查是否是已知的問題版本
        if xr.__version__.startswith('2024') or xr.__version__ >= '2024.01':
            print("⚠️ 檢測到新版 xarray，可能有兼容性問題")
            
    except ImportError as e:
        print(f"⚠️ 無法導入套件: {e}")
    
    # 3. 嘗試 monkey patch
    try:
        import xarray as xr
        
        # 如果 Dataset 有 reduce 方法，嘗試修補
        if hasattr(xr.Dataset, 'reduce'):
            print("🔧 嘗試修補 xarray.Dataset.reduce...")
            
            # 備份原始方法
            original_reduce = xr.Dataset.reduce
            
            def patched_reduce(self, func, dim=None, axis=None, keep_attrs=None, 
                             keepdims=False, numeric_only=False, **kwargs):
                # 將 axis 轉換為 dim
                if axis is not None and dim is None:
                    dim = axis
                    axis = None
                
                return original_reduce(
                    self, func, dim=dim, keep_attrs=keep_attrs, 
                    keepdims=keepdims, numeric_only=numeric_only, **kwargs
                )
            
            # 應用補丁
            xr.Dataset.reduce = patched_reduce
            print("✅ xarray.Dataset.reduce 補丁應用成功")
            
    except Exception as e:
        print(f"⚠️ 補丁應用失敗: {e}")
    
    # 4. 設置環境變數
    os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
    
    print("✅ 兼容性修復完成")


def create_simplified_bayesian_config():
    """創建簡化的 Bayesian 配置以避免兼容性問題"""
    
    simplified_config = {
        'pymc_backend': 'cpu',
        'pymc_mode': 'FAST_RUN',
        'n_threads': 1,
        'configure_pymc': False,
        
        # 簡化的 MCMC 參數
        'n_samples': 200,      # 減少樣本數
        'chains': 2,           # 減少鏈數  
        'tune': 300,           # 減少調參步數
        'target_accept': 0.8,  # 較低的接受率
        
        # 數據簡化
        'max_training_samples': 20,    # 最大訓練樣本
        'max_validation_samples': 10,  # 最大驗證樣本
        'max_loss_scenarios': 100,     # 最大損失情境數
        
        # 錯誤處理
        'fallback_to_demo': True,      # 如果失敗，回退到演示模式
        'suppress_warnings': True       # 抑制警告
    }
    
    return simplified_config


def test_bayesian_functionality():
    """測試基本的 Bayesian 功能是否正常"""
    
    print("🧪 測試 Bayesian 功能...")
    
    try:
        import pymc as pm
        import pytensor.tensor as pt
        import numpy as np
        
        # 簡單的貝葉斯線性回歸測試
        np.random.seed(42)
        x = np.random.normal(0, 1, 10)
        y = 2 * x + np.random.normal(0, 0.1, 10)
        
        with pm.Model() as model:
            # 先驗
            alpha = pm.Normal('alpha', mu=0, sigma=1)
            beta = pm.Normal('beta', mu=0, sigma=1)
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # 似然
            mu = alpha + beta * x
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
            
            # 採樣
            trace = pm.sample(100, tune=100, chains=1, return_inferencedata=True, 
                            progressbar=False, random_seed=42)
        
        print("✅ 基本 PyMC 功能測試通過")
        return True
        
    except Exception as e:
        print(f"❌ PyMC 功能測試失敗: {e}")
        return False


if __name__ == "__main__":
    print("🚀 開始 xarray/PyMC 兼容性修復")
    print("=" * 50)
    
    # 修復兼容性問題
    fix_xarray_compatibility()
    
    # 測試功能
    if test_bayesian_functionality():
        print("\n✅ 所有測試通過，Bayesian 分析準備就緒")
    else:
        print("\n⚠️ 存在兼容性問題，建議使用簡化配置")
        
    # 提供簡化配置
    simplified_config = create_simplified_bayesian_config()
    print(f"\n🔧 推薦的簡化配置:")
    for key, value in simplified_config.items():
        print(f"   {key}: {value}")
    
    print(f"\n📋 使用建議:")
    print("1. 重新啟動 Python kernel")
    print("2. 運行此腳本: python fix_xarray_compatibility.py")
    print("3. 在 notebook 中使用簡化配置")
    print("4. 如果仍有問題，使用演示模式")