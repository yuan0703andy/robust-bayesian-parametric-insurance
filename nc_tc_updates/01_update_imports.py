# ============================================================================
# 1. 更新導入部分 - 替換第72-80行
# ============================================================================

# 進階模組 - 更新為新版本
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
    print("   💡 建議: 確保 bayesian/ 和 skill_scores/ 目錄在路徑中")