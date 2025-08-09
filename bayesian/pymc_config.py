#!/usr/bin/env python3
"""
PyMC 配置模組
PyMC Configuration Module

統一設置 PyMC/JAX/NumPyro 的環境變數，避免 macOS Metal 後端問題
Unified setup for PyMC/JAX/NumPyro environment variables to avoid macOS Metal backend issues
"""

import os
import warnings
from typing import Optional


def configure_pymc_environment(
    backend: str = "cpu", 
    mode: str = "FAST_COMPILE",
    n_threads: Optional[int] = None,
    verbose: bool = True
):
    """
    配置 PyMC 環境 - 適用於本地和 HPC/OnDemand 環境
    Configure PyMC environment - suitable for local and HPC/OnDemand environments
    
    Parameters:
    -----------
    backend : str
        JAX 後端選擇 ("cpu", "gpu", "auto")
    mode : str
        PyTensor 編譯模式 ("FAST_COMPILE", "FAST_RUN", "DEBUG_MODE")
    n_threads : int, optional
        OpenMP 線程數，None 為自動設置
    verbose : bool
        是否顯示配置信息
        
    Returns:
    --------
    dict: 配置前後的環境變數比較
    """
    
    if verbose:
        print(f"🔧 配置 PyMC 環境 (後端: {backend}, 模式: {mode})...")
    
    # 記錄配置前的狀態
    old_config = {
        'JAX_PLATFORM_NAME': os.environ.get('JAX_PLATFORM_NAME'),
        'PYTENSOR_FLAGS': os.environ.get('PYTENSOR_FLAGS'),
        'MKL_THREADING_LAYER': os.environ.get('MKL_THREADING_LAYER'),
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS')
    }
    
    # 設置 JAX 後端
    if backend == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    elif backend == "gpu":
        # 在 HPC 上可能需要 GPU
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
    elif backend == "auto":
        # 讓 JAX 自動選擇
        if "JAX_PLATFORM_NAME" in os.environ:
            del os.environ["JAX_PLATFORM_NAME"]
    
    # 設置 PyTensor 編譯模式
    optimizer = "None" if mode == "FAST_COMPILE" else "fast_run"
    os.environ["PYTENSOR_FLAGS"] = f"cxx=,mode={mode},optimizer={optimizer}"
    
    # 設置線程相關環境變數
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    
    # OpenMP 線程數設置
    if n_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
    elif "OMP_NUM_THREADS" not in os.environ:
        # 默認設置為 1（避免 HPC 上的線程衝突）
        os.environ["OMP_NUM_THREADS"] = "1"
    
    # 記錄配置後的狀態
    new_config = {
        'JAX_PLATFORM_NAME': os.environ.get('JAX_PLATFORM_NAME'),
        'PYTENSOR_FLAGS': os.environ.get('PYTENSOR_FLAGS'),
        'MKL_THREADING_LAYER': os.environ.get('MKL_THREADING_LAYER'),
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS')
    }
    
    if verbose:
        print("配置結果:")
        for key, value in new_config.items():
            old_val = old_config.get(key, 'None')
            print(f"   {key}: {old_val} → {value}")
    
    return {'old_config': old_config, 'new_config': new_config}


def verify_pymc_setup():
    """
    驗證 PyMC 設置是否正確
    Verify PyMC setup is correct
    
    Returns:
    --------
    dict: 設置驗證結果
    """
    
    results = {
        'pymc_available': False,
        'jax_available': False,
        'jax_using_cpu': False,
        'pymc_version': None,
        'jax_version': None,
        'jax_devices': [],
        'setup_correct': False
    }
    
    # 檢查 PyMC
    try:
        import pymc as pm
        results['pymc_available'] = True
        results['pymc_version'] = pm.__version__
        print(f"✅ PyMC 版本: {pm.__version__}")
    except ImportError as e:
        print(f"❌ PyMC 導入失敗: {e}")
        return results
    
    # 檢查 JAX
    try:
        import jax
        results['jax_available'] = True
        results['jax_version'] = jax.__version__
        results['jax_devices'] = [str(device) for device in jax.devices()]
        
        print(f"✅ JAX 版本: {jax.__version__}")
        print(f"✅ JAX 設備: {jax.devices()}")
        
        # 確認 JAX 使用 CPU
        if any('cpu' in str(device).lower() for device in jax.devices()):
            results['jax_using_cpu'] = True
            print("✅ JAX 正確使用 CPU 後端")
        else:
            print("⚠️ 警告：JAX 可能未使用 CPU 後端")
            
    except ImportError:
        print("ℹ️ JAX 未安裝，PyMC 將使用默認後端")
    
    # 測試簡單的 PyMC 模型
    try:
        import numpy as np
        
        print("🧪 測試 PyMC 模型建立...")
        
        with pm.Model() as simple_model:
            # 簡單的線性模型
            alpha = pm.Normal('alpha', mu=0, sigma=1)
            beta = pm.Normal('beta', mu=0, sigma=1)
            
            # 模擬數據
            x = np.linspace(0, 1, 10)
            y_true = 2 * x + 1
            y_obs = pm.Normal('y_obs', 
                            mu=alpha + beta * x, 
                            sigma=0.1, 
                            observed=y_true + np.random.normal(0, 0.1, len(x)))
        
        print("✅ PyMC 模型建立成功")
        
        # 嘗試採樣（小規模測試）
        with simple_model:
            trace = pm.sample(
                draws=100,  # 小規模測試
                chains=1,
                progressbar=False,
                random_seed=42
            )
        
        print("✅ PyMC 採樣測試成功")
        results['setup_correct'] = True
        
    except Exception as e:
        print(f"❌ PyMC 測試失敗: {e}")
        print("可能的解決方案:")
        print("1. 重新啟動 Python kernel/session")
        print("2. 檢查套件安裝: pip install pymc pytensor jax jaxlib")
        print("3. 如果使用 conda: conda install -c conda-forge pymc")
        
    return results


def create_pymc_test_script():
    """創建一個獨立的 PyMC 測試腳本"""
    
    test_script = '''#!/usr/bin/env python3
"""
PyMC 環境測試腳本
獨立測試腳本，用於驗證 PyMC 環境設置
"""

# 導入配置模組
from bayesian.pymc_config import configure_pymc_environment, verify_pymc_setup

def main():
    print("🚀 PyMC 環境測試")
    print("=" * 50)
    
    # 配置環境
    configure_pymc_environment(verbose=True)
    
    print("\\n" + "=" * 50)
    
    # 驗證設置
    results = verify_pymc_setup()
    
    print("\\n📋 測試總結:")
    print("-" * 30)
    print(f"PyMC 可用: {'✅' if results['pymc_available'] else '❌'}")
    print(f"JAX 可用: {'✅' if results['jax_available'] else '❌'}")
    print(f"JAX 使用 CPU: {'✅' if results['jax_using_cpu'] else '❌'}")
    print(f"設置正確: {'✅' if results['setup_correct'] else '❌'}")
    
    if results['setup_correct']:
        print("\\n🎉 PyMC 環境設置成功！")
    else:
        print("\\n❌ PyMC 環境設置需要調整")
        
    return results

if __name__ == "__main__":
    main()
'''
    
    return test_script


# Note: 不自動配置環境 - 讓用戶在需要時手動調用
# 這樣更適合 HPC/OnDemand 環境，避免全域影響