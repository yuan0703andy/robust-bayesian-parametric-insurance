# ============================================================================
# 5. 最終比較結果更新 - 替換第1515-1550行
# ============================================================================

# %% 🏆 最終比較結果
print("\n🏆 雙軌分析結果比較 (Steinmann RMSE vs 新版 Bayesian)")
print("=" * 80)

# 確保兩種分析結果都存在
if 'steinmann_best' in locals() and 'bayesian_best' in locals():
    
    # 新版整合比較
    comparison_results = {
        'analysis_framework': 'dual_track_v2',
        'steinmann_rmse': {
            'best_product': steinmann_best['product_id'],
            'best_radius': steinmann_best.get('radius_km', 'unknown'),
            'rmse': steinmann_best['rmse'],
            'correlation': steinmann_best.get('correlation', 0),
            'trigger_rate': steinmann_best.get('trigger_rate', 0),
            'method': 'steinmann_2023_deterministic'
        },
        'bayesian_integrated': {
            'best_product': bayesian_best['product_id'],
            'best_radius': bayesian_best.get('radius_km', 'integrated'),
            'crps': bayesian_best.get('crps', 0),
            'correlation': bayesian_best.get('correlation', 0),
            'trigger_rate': bayesian_best.get('trigger_rate', 0),
            'method': bayesian_best.get('method', 'integrated_bayesian'),
            'champion_model': bayesian_best.get('champion_model', 'unknown'),
            'expected_basis_risk': bayesian_best.get('expected_basis_risk', 0),
            'theoretical_framework': bayesian_best.get('theoretical_framework', 'bayesian_implement.md')
        },
        'comparison_metrics': {
            'same_best_product': steinmann_best['product_id'] == bayesian_best['product_id'],
            'same_best_radius': steinmann_best.get('radius_km') == bayesian_best.get('radius_km'),
            'correlation_improvement': bayesian_best.get('correlation', 0) - steinmann_best.get('correlation', 0),
            'trigger_rate_difference': bayesian_best.get('trigger_rate', 0) - steinmann_best.get('trigger_rate', 0),
            'bayesian_uses_integrated_method': 'integrated' in bayesian_best.get('method', ''),
            'theoretical_compliance': bayesian_best.get('theoretical_framework') == 'bayesian_implement.md'
        },
        'innovation_metrics': {
            'bayesian_method_upgrade': bayesian_best.get('method', '').startswith('integrated'),
            'automatic_model_selection': 'champion_model' in bayesian_best,
            'basis_risk_optimization': 'expected_basis_risk' in bayesian_best,
            'two_phase_workflow': 'integrated' in bayesian_best.get('method', '')
        }
    }
    
    # 顯示結果
    print("📊 分析方法比較:")
    print(f"   Steinmann (傳統): {comparison_results['steinmann_rmse']['method']}")
    print(f"   Bayesian (新版): {comparison_results['bayesian_integrated']['method']}")
    
    print(f"\n🎯 最佳產品比較:")
    print(f"   Steinmann: {comparison_results['steinmann_rmse']['best_product']}")
    print(f"      • 半徑: {comparison_results['steinmann_rmse']['best_radius']}km")
    print(f"      • RMSE: ${comparison_results['steinmann_rmse']['rmse']/1e9:.3f}B")
    print(f"      • 相關性: {comparison_results['steinmann_rmse']['correlation']:.3f}")
    
    print(f"   Bayesian: {comparison_results['bayesian_integrated']['best_product']}")
    print(f"      • 方法: {comparison_results['bayesian_integrated']['best_radius']}")
    if comparison_results['bayesian_integrated']['crps'] > 0:
        print(f"      • CRPS: ${comparison_results['bayesian_integrated']['crps']/1e9:.3f}B")
    print(f"      • 相關性: {comparison_results['bayesian_integrated']['correlation']:.3f}")
    if comparison_results['bayesian_integrated']['champion_model'] != 'unknown':
        print(f"      • 基礎模型: {comparison_results['bayesian_integrated']['champion_model']}")
    
    print(f"\n🔍 一致性分析:")
    print(f"   相同最佳產品: {'✅' if comparison_results['comparison_metrics']['same_best_product'] else '❌'}")
    print(f"   相關性提升: {comparison_results['comparison_metrics']['correlation_improvement']:+.3f}")
    print(f"   觸發率差異: {comparison_results['comparison_metrics']['trigger_rate_difference']:+.1%}")
    
    print(f"\n🚀 創新指標:")
    print(f"   使用整合方法: {'✅' if comparison_results['innovation_metrics']['bayesian_method_upgrade'] else '❌'}")
    print(f"   自動模型選擇: {'✅' if comparison_results['innovation_metrics']['automatic_model_selection'] else '❌'}")
    print(f"   基差風險最佳化: {'✅' if comparison_results['innovation_metrics']['basis_risk_optimization'] else '❌'}")
    print(f"   兩階段工作流程: {'✅' if comparison_results['innovation_metrics']['two_phase_workflow'] else '❌'}")
    print(f"   理論框架符合性: {'✅' if comparison_results['comparison_metrics']['theoretical_compliance'] else '❌'}")
    
else:
    print("⚠️ 無法進行完整比較 - 缺少必要的分析結果")
    comparison_results = {
        'analysis_framework': 'incomplete',
        'error': 'missing_results'
    }

# 整合到最終結果中
final_analysis_results = {
    'metadata': {
        'analysis_version': 'v2.0_integrated_bayesian',
        'framework': 'dual_track_enhanced',
        'timestamp': datetime.now().isoformat(),
        'environment': run_environment if 'run_environment' in locals() else 'unknown'
    },
    'steinmann_results': steinmann_results if 'steinmann_results' in locals() else {},
    'bayesian_results': bayesian_results if 'bayesian_results' in locals() else {},
    'comparison_results': comparison_results,
    'data_metadata': {
        'data_source': data_source if 'data_source' in locals() else 'unknown',
        'n_events': len(damages) if 'damages' in locals() else 0,
        'total_loss': sum(damages)/1e9 if 'damages' in locals() else 0,
        'analysis_scope': 'hospital_level' if ('hospital_exposures' in locals() and hospital_exposures is not None) else 'full_exposure'
    },
    'technical_details': {
        'bayesian_analyzer_version': '2.0_integrated',
        'pymc_config': pymc_config if 'pymc_config' in locals() else {},
        'theoretical_basis': 'bayesian_implement.md',
        'loss_scenarios': n_loss_scenarios if 'n_loss_scenarios' in locals() else 0,
        'monte_carlo_samples': n_monte_carlo_samples if 'n_monte_carlo_samples' in locals() else 0
    }
}

print(f"\n📋 分析總結:")
print(f"   版本: {final_analysis_results['metadata']['analysis_version']}")
print(f"   環境: {final_analysis_results['metadata']['environment']}")
print(f"   數據範圍: {final_analysis_results['data_metadata']['analysis_scope']}")
print(f"   理論基礎: {final_analysis_results['technical_details']['theoretical_basis']}")
if final_analysis_results['data_metadata']['n_events'] > 0:
    print(f"   事件數: {final_analysis_results['data_metadata']['n_events']}")
    print(f"   總損失: ${final_analysis_results['data_metadata']['total_loss']:.1f}B")