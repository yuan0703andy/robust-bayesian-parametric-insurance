#!/usr/bin/env python3
"""
PROJ Path Fixer - 修復PROJ路徑問題
"""

import os
import sys

def fix_proj_paths():
    """修復所有PROJ相關的環境變數路徑問題"""
    
    print("🔧 PROJ Path Fixer - 修復PROJ路徑問題")
    print("=" * 50)
    
    # 所有可能的PROJ相關環境變數
    proj_vars = [
        'PROJ_DATA', 'PROJ_LIB', 'GDAL_DATA', 'GEOS_LIB', 'PROJ_NETWORK',
        'PROJ_DEBUG', 'PROJ_CURL_CA_BUNDLE', 'PROJ_USER_WRITABLE_DIRECTORY',
        'GDAL_DRIVER_PATH', 'GDAL_PLUGIN_PATH', 'GDAL_PYTHON_DRIVER_PATH',
        'GEOTIFF_CSV', 'PROJ_SKIP_READ_USER_WRITABLE_DIRECTORY'
    ]
    
    cleared_vars = []
    
    for var in proj_vars:
        if var in os.environ:
            old_val = os.environ[var]
            # 檢查是否包含HPC路徑
            if '/hpc/' in old_val or '/cluster/' in old_val or 'borsuklab' in old_val:
                del os.environ[var]
                cleared_vars.append((var, old_val))
                print(f"   ❌ Cleared {var}: {old_val}")
    
    if cleared_vars:
        print(f"\n✅ Cleared {len(cleared_vars)} problematic PROJ variables")
        print("   This should fix the PROJ database path error")
    else:
        print("   ℹ️ No problematic PROJ variables found")
    
    # 設置安全的環境變數
    safe_vars = {
        'PROJ_NETWORK': 'OFF',  # 禁用網路存取
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',  # 禁用目錄掃描
    }
    
    print(f"\n🔧 Setting safe PROJ variables:")
    for key, value in safe_vars.items():
        os.environ[key] = value
        print(f"   ✅ {key} = {value}")
    
    print("\n🎯 PROJ修復完成 - 現在可以安全導入CLIMADA")
    return True

if __name__ == "__main__":
    fix_proj_paths()