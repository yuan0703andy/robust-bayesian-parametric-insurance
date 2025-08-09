# ============================================================================
# 4. 更新結果整合代碼 - 替換第1490-1510行
# ============================================================================

# %% 🔄 結果整合 - 新版 Bayesian 分析
print("📊 整合分析結果...")

# 處理新版 Bayesian 結果
if modules_available['bayesian'] and 'bayesian_optimal_product' in locals() and bayesian_optimal_product is not None:
    print("   🎯 整合新版 Bayesian 最佳化結果...")
    
    # 使用新的整合最佳化產品
    bayesian_best = bayesian_optimal_product.copy()
    
    # 如果有現有的 CRPS 分析結果，進行比較
    if 'crps_df' in locals() and len(crps_df) > 0:
        print("      📈 比較新最佳化產品與現有產品...")
        
        # 計算新產品的 CRPS (如果有分布數據)
        if 'loss_distributions' in locals() and len(loss_distributions) > 0:
            try:
                # 使用現有的 CRPS 計算函數
                new_product_crps = calculate_crps_score(
                    damages[:len(bayesian_best['payouts'])], 
                    bayesian_best['payouts'], 
                    list(loss_distributions.values())
                )
                bayesian_best['crps'] = new_product_crps
                print(f"         新產品 CRPS: ${new_product_crps/1e9:.3f}B")
                
                # 比較與最佳現有產品
                current_best_crps = crps_df['crps'].min()
                improvement = (current_best_crps - new_product_crps) / current_best_crps * 100
                if improvement > 0:
                    print(f"         🎊 CRPS 改進: {improvement:.1f}%")
                else:
                    print(f"         ℹ️ CRPS 變化: {improvement:.1f}%")
                    
            except Exception as e:
                print(f"         ⚠️ CRPS 計算失敗: {e}")
                bayesian_best['crps'] = bayesian_best.get('champion_crps', 0)
        else:
            bayesian_best['crps'] = bayesian_best.get('champion_crps', 0)
    
    else:
        # 沒有現有結果可比較，直接使用新結果
        print("      ℹ️ 無現有結果可比較，使用新最佳化產品")
        bayesian_best['crps'] = bayesian_best.get('champion_crps', 0)
    
    # 確保所有必需的字段都存在
    bayesian_best.setdefault('product_id', 'bayesian_integrated_optimal')
    bayesian_best.setdefault('correlation', bayesian_best.get('correlation', 0))
    bayesian_best.setdefault('trigger_rate', bayesian_best.get('trigger_rate', 0))
    
    print(f"   ✅ 新版 Bayesian 結果整合完成")
    print(f"      產品: {bayesian_best['product_id']}")
    print(f"      理論框架: {bayesian_best.get('theoretical_framework', 'integrated')}")
    
elif 'crps_df' in locals() and len(crps_df) > 0:
    # 回退到現有的 CRPS 分析結果
    print("   🔄 使用現有 CRPS 分析結果...")
    best_crps_idx = crps_df['crps'].idxmin()
    bayesian_best = crps_df.iloc[best_crps_idx].to_dict()
    print(f"      回退產品: {bayesian_best.get('product_id', 'unknown')}")

else:
    # 沒有任何 Bayesian 結果
    print("   ❌ 無可用的 Bayesian 分析結果")
    bayesian_best = {
        'product_id': 'no_bayesian_result',
        'crps': float('inf'),
        'correlation': 0,
        'trigger_rate': 0,
        'method': 'fallback'
    }

print(f"   📋 Bayesian 最終結果:")
if bayesian_best and bayesian_best.get('product_id') != 'no_bayesian_result':
    print(f"      最佳產品: {bayesian_best.get('product_id', 'unknown')}")
    crps_value = bayesian_best.get('crps', 0)
    if crps_value < float('inf'):
        print(f"      CRPS: ${crps_value/1e9:.3f}B")
    print(f"      相關性: {bayesian_best.get('correlation', 0):.3f}")
    print(f"      觸發率: {bayesian_best.get('trigger_rate', 0):.1%}")
    
    # 如果有基差風險信息
    if 'expected_basis_risk' in bayesian_best:
        print(f"      期望基差風險: ${bayesian_best['expected_basis_risk']/1e9:.3f}B")
    if 'champion_model' in bayesian_best:
        print(f"      基礎模型: {bayesian_best['champion_model']}")
else:
    print(f"      狀態: 無有效結果")

# 整合到現有的結果結構中
bayesian_results = {
    'method': 'integrated_bayesian_optimization' if bayesian_best.get('method') != 'fallback' else 'fallback',
    'results_df': crps_df if 'crps_df' in locals() else pd.DataFrame(),
    'best_product': bayesian_best,
    'summary': {
        'best_crps': bayesian_best.get('crps', float('inf')),
        'best_correlation': bayesian_best.get('correlation', 0),
        'mean_trigger_rate': bayesian_best.get('trigger_rate', 0),
        'theoretical_framework': bayesian_best.get('theoretical_framework', 'unknown'),
        'analysis_method': bayesian_best.get('method', 'unknown')
    }
}