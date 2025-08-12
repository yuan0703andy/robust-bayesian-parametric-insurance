#!/usr/bin/env python3
"""
PyMC 5.25.1 兼容性修復腳本
Fix PyMC Compatibility for pytensor.tensor

將舊的 pm.math 語法更新為 pytensor.tensor
"""

import os
import re
from pathlib import Path

def fix_pymc_math_imports():
    """修復 PyMC math 導入和用法"""
    
    bayesian_dir = Path("bayesian")
    target_file = bayesian_dir / "parametric_bayesian_hierarchy.py"
    
    if not target_file.exists():
        print(f"❌ 文件不存在: {target_file}")
        return
    
    print(f"🔧 修復 PyMC 兼容性: {target_file}")
    
    # 讀取原始內容
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 備份原始文件
    backup_file = target_file.with_suffix('.py.backup')
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"📁 已備份到: {backup_file}")
    
    # 修復導入
    print("🔄 更新導入語句...")
    if "import pytensor.tensor as pt" not in content:
        # 在 PyMC 導入後添加 pytensor 導入
        content = re.sub(
            r'(import pymc as pm\n)',
            r'\1    import pytensor.tensor as pt\n',
            content
        )
    
    # 修復 pm.math 用法的映射表
    replacements = {
        # 基本數學函數
        'pm.math.clip': 'pt.clip',
        'pm.math.maximum': 'pt.maximum',
        'pm.math.minimum': 'pt.minimum',
        'pm.math.log': 'pt.log',
        'pm.math.exp': 'pt.exp',
        'pm.math.power': 'pt.power',
        'pm.math.pow': 'pt.power',  # pow 也映射到 power
        'pm.math.switch': 'pt.switch',
        'pm.math.sqrt': 'pt.sqrt',
        'pm.math.sin': 'pt.sin',
        'pm.math.cos': 'pt.cos',
        'pm.math.tan': 'pt.tan',
        'pm.math.abs': 'pt.abs_',
        
        # 特殊函數
        'pm.math.set_subtensor': 'pt.set_subtensor',
        'pm.math.op.vectorize': 'pt.as_op',  # 這個需要更複雜的處理
    }
    
    print("🔄 替換 pm.math 用法...")
    replacement_count = 0
    
    for old_pattern, new_pattern in replacements.items():
        old_count = content.count(old_pattern)
        if old_count > 0:
            content = content.replace(old_pattern, new_pattern)
            replacement_count += old_count
            print(f"   ✅ {old_pattern} → {new_pattern} ({old_count} 次)")
    
    # 特殊處理：@pm.math.op.vectorize 裝飾器
    vectorize_pattern = r'@pm\.math\.op\.vectorize'
    if re.search(vectorize_pattern, content):
        print("⚠️  發現 @pm.math.op.vectorize 裝飾器，需要手動處理")
        print("   建議替換為更簡單的實現或使用 pt.as_op")
    
    # 檢查是否還有遺漏的 pm.math
    remaining_pm_math = re.findall(r'pm\.math\.\w+', content)
    if remaining_pm_math:
        print("⚠️  還有未處理的 pm.math 用法:")
        for match in set(remaining_pm_math):
            print(f"   - {match}")
    
    # 寫入修復後的內容
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 修復完成！共替換 {replacement_count} 處")
    print(f"📁 原文件已備份為: {backup_file}")
    
    return True

def create_compatibility_test():
    """創建兼容性測試腳本"""
    
    test_content = '''#!/usr/bin/env python3
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
        print(f"\\n🔍 {test_name}...")
        result = test_func()
        results.append(result)
    
    print(f"\\n📊 測試結果: {sum(results)}/{len(results)} 通過")
    
    if all(results):
        print("🎉 所有測試通過！PyMC 5.25.1 兼容性良好")
    else:
        print("⚠️  部分測試失敗，需要進一步修復")

if __name__ == "__main__":
    main()
'''
    
    test_file = Path("test_pymc_compatibility.py")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"📝 已創建兼容性測試腳本: {test_file}")

def main():
    """主函數"""
    print("🔧 PyMC 5.25.1 兼容性修復工具")
    print("=" * 50)
    
    # 修復兼容性
    success = fix_pymc_math_imports()
    
    if success:
        # 創建測試腳本
        create_compatibility_test()
        
        print(f"\\n✅ 修復完成！")
        print(f"📋 下一步:")
        print(f"   1. 運行測試: python test_pymc_compatibility.py")
        print(f"   2. 檢查模型可視化是否正常工作")
        print(f"   3. 如有問題，檢查備份文件並手動調整")
        
        print(f"\\n⚠️  注意事項:")
        print(f"   - @pm.math.op.vectorize 需要手動處理")
        print(f"   - 複雜的數學操作可能需要進一步調整")
        print(f"   - 建議測試所有功能確保正常工作")

if __name__ == "__main__":
    main()