#!/usr/bin/env python3
"""
PyMC 5.25.1 兼容性測試
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pymc_imports():
    """測試 PyMC 導入"""
    try:
        import pymc as pm
        import pytensor.tensor as pt
        import arviz as az
        print(f"✅ PyMC 版本: {pm.__version__}")
        print(f"✅ pytensor 可用")
        print(f"✅ ArviZ 版本: {az.__version__}")
        return True
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return False

def test_basic_operations():
    """測試基本 pytensor 操作"""
    try:
        import pytensor.tensor as pt
        import numpy as np
        
        # 測試基本函數
        x = pt.scalar('x')
        y = pt.log(pt.exp(x))
        z = pt.switch(x > 0, x, 0)
        w = pt.maximum(x, 0)
        
        print("✅ pytensor 基本操作測試通過")
        return True
    except Exception as e:
        print(f"❌ pytensor 操作失敗: {e}")
        return False

def test_bayesian_module():
    """測試貝氏模組導入"""
    try:
        from bayesian.parametric_bayesian_hierarchy import (
            ParametricHierarchicalModel, ModelSpec, VulnerabilityData
        )
        print("✅ 貝氏模組導入成功")
        return True
    except ImportError as e:
        print(f"❌ 貝氏模組導入失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🧪 PyMC 5.25.1 兼容性測試")
    print("=" * 40)
    
    tests = [
        ("PyMC 導入測試", test_pymc_imports),
        ("pytensor 操作測試", test_basic_operations),
        ("貝氏模組測試", test_bayesian_module),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        result = test_func()
        results.append(result)
    
    print(f"\n📊 測試結果: {sum(results)}/{len(results)} 通過")
    
    if all(results):
        print("🎉 所有測試通過！PyMC 5.25.1 兼容性良好")
    else:
        print("⚠️  部分測試失敗，需要進一步修復")

if __name__ == "__main__":
    main()
