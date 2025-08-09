# ============================================================================
# 3. 新的 Bayesian 整合代碼 - 替換第1242-1350行
# ============================================================================

# %% 🧠 新版整合貝葉斯分析
if modules_available['bayesian'] and 'tc_hazard' in main_data and 'exposure' in main_data:
    try:
        print("   🚀 啟動新版整合貝葉斯分析器...")
        print("   📖 理論基礎: bayesian_implement.md - 方法一 → 方法二連貫流程")
        
        # 初始化新版分析器
        bayesian_analyzer = RobustBayesianAnalyzer(
            density_ratio_constraint=2.0,
            n_monte_carlo_samples=n_monte_carlo_samples,
            n_mixture_components=3
        )
        
        # 準備分析數據
        print("      📊 準備兩階段分析數據...")
        
        # 選擇曝險數據
        if 'hospital_exposures' in locals() and hospital_exposures is not None:
            exposure_for_bayesian = hospital_exposures
            print(f"      🎯 使用 {len(exposure_for_bayesian.gdf)} 個醫院點進行建模")
        else:
            exposure_for_bayesian = exposure_main
            print(f"      ⚠️ 使用完整LitPop數據 ({len(exposure_for_bayesian.gdf)} 點)")
        
        # 確保損失數據格式正確
        damages_array = np.array(damages, dtype=np.float64)
        n_events = len(damages_array)
        
        # 智能數據分割 (方法一需要訓練/驗證分割)
        if n_events >= 50:
            n_train = max(int(0.7 * n_events), 30)  # 至少30個訓練樣本
        else:
            n_train = max(int(0.8 * n_events), 10)  # 小數據集用更多訓練樣本
            
        n_validation = n_events - n_train
        
        if n_validation < 5:  # 確保至少有5個驗證樣本
            n_train = max(n_events - 5, 10)
            n_validation = n_events - n_train
        
        train_losses = damages_array[:n_train]
        validation_losses = damages_array[n_train:]
        
        print(f"      📋 智能數據分割: 訓練({n_train}) / 驗證({n_validation})")
        
        # 創建風險指標 (方法二需要)
        print("      🌪️ 建立風險指標...")
        if 'hospital_wind_series' in locals() and hospital_wind_series:
            # 使用醫院風速數據
            hazard_indices = np.array(list(hospital_wind_series.values())).mean(axis=0)[:n_train]
            print(f"         使用醫院風速數據 ({len(hazard_indices)} 個事件)")
        else:
            # 基於損失大小推估風險指標
            # 高損失 -> 高風險指標，低損失 -> 低風險指標
            normalized_losses = (train_losses - np.min(train_losses)) / (np.max(train_losses) - np.min(train_losses) + 1e-10)
            hazard_indices = 25 + normalized_losses * 40  # 對應風速 25-65
            print(f"         基於損失推估風險指標 ({len(hazard_indices)} 個事件)")
        
        # 創建損失情境矩陣 (方法二的期望損失計算需要)
        print(f"      🎲 生成 {n_loss_scenarios} 個損失情境...")
        actual_losses_matrix = np.zeros((n_loss_scenarios, n_train))
        
        for i in range(n_loss_scenarios):
            # 基於不確定性生成情境
            hazard_uncertainty = np.random.normal(1.0, 0.15, n_train)  # 15% 風險不確定性
            exposure_uncertainty = np.random.lognormal(0, 0.20)        # 20% 曝險不確定性
            vulnerability_uncertainty = np.random.normal(1.0, 0.10)   # 10% 脆弱性不確定性
            
            scenario_losses = (train_losses * 
                             hazard_uncertainty * 
                             exposure_uncertainty * 
                             vulnerability_uncertainty)
            
            actual_losses_matrix[i, :] = np.maximum(scenario_losses, 0)  # 確保非負
        
        print(f"         平均情境損失: ${np.mean(actual_losses_matrix)/1e9:.2f}B")
        print(f"         損失變異範圍: ${np.std(actual_losses_matrix)/1e9:.2f}B")
        
        # 定義產品參數最佳化邊界
        # 基於現有數據範圍設置合理邊界
        min_wind, max_wind = np.min(hazard_indices), np.max(hazard_indices)
        mean_loss = np.mean(train_losses)
        max_loss = np.max(train_losses)
        
        product_bounds = {
            'trigger_threshold': (max(min_wind - 5, 20), min(max_wind + 5, 70)),
            'payout_amount': (mean_loss * 0.5, max_loss * 2.0),
            'max_payout': (max_loss * 3.0, max_loss * 5.0)
        }
        
        print(f"      ⚙️ 產品參數邊界:")
        print(f"         觸發閾值: {product_bounds['trigger_threshold'][0]:.1f} - {product_bounds['trigger_threshold'][1]:.1f}")
        print(f"         賠付金額: ${product_bounds['payout_amount'][0]/1e9:.2f}B - ${product_bounds['payout_amount'][1]/1e9:.2f}B")
        
        print("      🚀 執行整合貝葉斯最佳化 (方法一 + 方法二)...")
        
        # 🎯 使用新的整合方法
        bayesian_results = bayesian_analyzer.integrated_bayesian_optimization(
            observations=train_losses,
            validation_data=validation_losses,
            hazard_indices=hazard_indices,
            actual_losses=actual_losses_matrix,
            product_bounds=product_bounds,
            basis_risk_type=BasisRiskType.WEIGHTED_ASYMMETRIC,
            w_under=2.0,  # 賠不夠的懲罰權重較高
            w_over=0.5,   # 賠多了的懲罰權重較低
            **pymc_config  # 使用環境配置
        )
        
        print("   ✅ 新版整合貝葉斯分析完成！")
        
        # 提取結果
        phase1_results = bayesian_results['phase_1_model_comparison']
        phase2_results = bayesian_results['phase_2_decision_optimization']
        integration_validation = bayesian_results['integration_validation']
        
        print(f"      🏆 方法一結果:")
        print(f"         冠軍模型: {phase1_results['champion_model']['name']}")
        print(f"         CRPS 分數: {phase1_results['champion_model']['crps_score']:.3e}")
        print(f"         候選模型數: {len(phase1_results['candidate_models'])}")
        
        print(f"      🎯 方法二結果:")
        print(f"         最佳觸發閾值: {phase2_results['optimal_product']['trigger_threshold']:.1f}")
        print(f"         最佳賠付金額: ${phase2_results['optimal_product']['payout_amount']/1e9:.2f}B")
        print(f"         期望基差風險: ${phase2_results['expected_basis_risk']/1e9:.3f}B")
        
        print(f"      ✅ 理論驗證: {integration_validation['theoretical_compliance']}")
        
        # 創建與現有系統兼容的結果格式
        bayesian_optimal_product = {
            'product_id': 'bayesian_integrated_optimal',
            'trigger_threshold': phase2_results['optimal_product']['trigger_threshold'],
            'payout_amount': phase2_results['optimal_product']['payout_amount'],
            'max_payout': phase2_results['optimal_product'].get('max_payout', phase2_results['optimal_product']['payout_amount']),
            'method': 'integrated_bayesian_optimization_v2',
            'champion_model': phase1_results['champion_model']['name'],
            'champion_crps': phase1_results['champion_model']['crps_score'],
            'expected_basis_risk': phase2_results['expected_basis_risk'],
            'optimization_method': phase2_results['methodology'],
            'theoretical_framework': 'bayesian_implement.md'
        }
        
        # 模擬計算新產品在原始數據上的表現
        print("      📊 計算最佳產品在全部數據上的表現...")
        optimal_payouts = []
        for i, loss in enumerate(damages_array):
            if i < len(hazard_indices):
                wind = hazard_indices[i]
            else:
                # 對驗證集推估風險指標
                val_normalized = (loss - np.min(damages_array)) / (np.max(damages_array) - np.min(damages_array) + 1e-10)
                wind = 25 + val_normalized * 40
            
            if wind >= bayesian_optimal_product['trigger_threshold']:
                payout = min(bayesian_optimal_product['payout_amount'], bayesian_optimal_product['max_payout'])
            else:
                payout = 0.0
            optimal_payouts.append(payout)
        
        optimal_payouts = np.array(optimal_payouts)
        
        # 計算統計指標
        correlation = np.corrcoef(damages_array, optimal_payouts)[0, 1] if len(optimal_payouts) > 1 else 0
        trigger_rate = np.mean(optimal_payouts > 0)
        
        bayesian_optimal_product.update({
            'payouts': optimal_payouts,
            'correlation': correlation,
            'trigger_rate': trigger_rate,
            'radius_km': 'integrated',  # 不使用固定半徑，而是整合分析
        })
        
        print(f"         相關性: {correlation:.3f}")
        print(f"         觸發率: {trigger_rate:.1%}")
        print(f"         總賠付: ${np.sum(optimal_payouts)/1e9:.2f}B")
        
        # 為兼容性創建機率損失分布
        print("      🔄 建立兼容性損失分布...")
        event_loss_distributions = {}
        for event_idx in range(min(n_train, len(damages_array))):
            if event_idx < actual_losses_matrix.shape[1]:
                event_samples = actual_losses_matrix[:, event_idx]
            else:
                # 為超出訓練範圍的事件生成分布
                base_loss = damages_array[event_idx]
                event_samples = base_loss * np.random.lognormal(0, 0.3, n_loss_scenarios)
            
            event_loss_distributions[f'event_{event_idx}'] = {
                'mean': float(np.mean(event_samples)),
                'std': float(np.std(event_samples)),
                'samples': event_samples.tolist()[:min(len(event_samples), 100)],  # 限制樣本數
                'percentiles': {
                    '5th': float(np.percentile(event_samples, 5)),
                    '95th': float(np.percentile(event_samples, 95)),
                    '50th': float(np.percentile(event_samples, 50))
                }
            }
        
        loss_distributions = event_loss_distributions
        
        print(f"   ✅ 整合分析全部完成！")
        print(f"      📊 生成了 {len(loss_distributions)} 個事件的機率性損失分布")
        print(f"      🎯 每個分布包含最多 {n_loss_scenarios} 個樣本")
        print(f"      🏆 推薦產品: {bayesian_optimal_product['product_id']}")
        
    except Exception as e:
        print(f"   ❌ 新版整合貝葉斯分析失敗: {e}")
        print(f"      錯誤詳情: {str(e)}")
        print("      🔄 回退到原始方法...")
        
        # 導入具體的錯誤信息
        import traceback
        print(f"      📋 詳細錯誤追蹤:")
        for line in traceback.format_exc().split('\n')[:5]:  # 只顯示前5行
            if line.strip():
                print(f"         {line}")
        
        modules_available['bayesian'] = False
        bayesian_optimal_product = None
        loss_distributions = {}

else:
    print("   ⚠️ 跳過新版貝氏分析")
    if not modules_available['bayesian']:
        print("      原因: Bayesian 模組不可用")
    elif 'tc_hazard' not in main_data:
        print("      原因: tc_hazard 數據未準備")
    elif 'exposure' not in main_data:
        print("      原因: exposure 數據未準備")
    
    bayesian_optimal_product = None
    loss_distributions = {}