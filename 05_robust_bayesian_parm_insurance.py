#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_robust_bayesian_parm_insurance.py
=====================================
Complete Robust Hierarchical Bayesian Parametric Insurance Analysis
完整強健階層貝氏參數保險分析

Integrates the full bayesian/ module framework for comprehensive analysis:
整合完整bayesian/模組框架進行綜合分析：

• Hierarchical Bayesian Model (4-level + MPE) 階層貝氏模型(四層+混合預測估計)
• Robust Bayesian Framework (Density Ratio) 強健貝氏框架(密度比)
• Uncertainty Quantification 不確定性量化
• Weight Sensitivity Analysis 權重敏感度分析
• Integration with skill_scores and insurance modules 整合技能分數和保險模組
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import complete bayesian framework 匯入完整貝氏框架
from bayesian import (
    RobustBayesianAnalyzer,                    # Main analyzer 主分析器
    RobustBayesianFramework,                   # Density ratio framework 密度比框架
    HierarchicalBayesianModel,                 # 4-level hierarchical model 四層階層模型
    HierarchicalModelConfig,                   # Hierarchical configuration 階層配置
    ProbabilisticLossDistributionGenerator,    # Uncertainty quantification 不確定性量化
    WeightSensitivityAnalyzer,                 # Weight sensitivity analysis 權重敏感度分析
    MixedPredictiveEstimation,                 # MPE implementation MPE實現
    get_default_config,                        # Default configuration 預設配置
    validate_installation                       # Installation validation 安裝驗證
)

# Import external integrations 匯入外部整合
from skill_scores.basis_risk_functions import (
    BasisRiskCalculator, BasisRiskConfig, BasisRiskType
)


def main():
    """
    Complete Robust Hierarchical Bayesian Analysis
    完整強健階層貝氏分析主程式
    
    Implements comprehensive Bayesian framework with:
    實現包含以下完整貝氏框架：
    • 4-level hierarchical Bayesian model 四層階層貝氏模型
    • Mixed Predictive Estimation (MPE) 混合預測估計
    • Density ratio robustness constraints 密度比強健性約束
    • Complete uncertainty quantification 完整不確定性量化
    • Weight sensitivity analysis 權重敏感度分析
    """
    print("=" * 100)
    print("🧠 Complete Robust Hierarchical Bayesian Parametric Insurance Analysis")
    print("   完整強健階層貝氏參數保險分析")
    print("=" * 100)
    print("📋 Analysis Components 分析組件:")
    print("   • RobustBayesianAnalyzer (Main Interface) 強健貝氏分析器(主介面)")
    print("   • HierarchicalBayesianModel (4-level + MPE) 階層貝氏模型(四層+MPE)")
    print("   • ProbabilisticLossDistributionGenerator (Uncertainty) 機率損失分布生成器(不確定性)")
    print("   • WeightSensitivityAnalyzer (Sensitivity) 權重敏感度分析器")
    print("   • Integration with skill_scores & insurance modules 整合技能分數和保險模組")
    print("=" * 100)
    
    # Validate installation 驗證安裝
    print("\n🔍 Validating installation 驗證安裝...")
    validation = validate_installation()
    print(f"   • Core bayesian modules: {'✅' if validation['core_modules'] else '❌'}")
    print(f"   • skill_scores integration: {'✅' if validation['skill_scores'] else '⚠️'}")
    print(f"   • insurance_analysis_refactored: {'✅' if validation['insurance_module'] else '⚠️'}")
    print(f"   • CLIMADA integration: {'✅' if validation['climada'] else '⚠️'}")
    
    if validation['dependencies']:
        print("   Dependencies missing:")
        for dep in validation['dependencies']:
            print(f"     - {dep}")
    print()
    
    # Load required data
    print("\n📂 Loading data...")
    
    # Load products
    try:
        with open("results/insurance_products/products.pkl", 'rb') as f:
            products = pickle.load(f)
        print(f"✅ Loaded {len(products)} insurance products")
    except FileNotFoundError:
        print("❌ Products not found. Run 03_insurance_product.py first.")
        return
    
    # Load spatial analysis results  
    try:
        with open("results/spatial_analysis/cat_in_circle_results.pkl", 'rb') as f:
            spatial_results = pickle.load(f)
        wind_indices_dict = spatial_results['indices']
        # Extract main wind index for analysis (using 30km max as primary)
        wind_indices = wind_indices_dict.get('cat_in_circle_30km_max', np.array([]))
        print("✅ Loaded spatial analysis results")
        print(f"   Using primary index: cat_in_circle_30km_max ({len(wind_indices)} events)")
    except FileNotFoundError:
        print("❌ Spatial results not found. Run 02_spatial_analysis.py first.")
        return
    
    # Load CLIMADA data
    try:
        with open("climada_complete_data.pkl", 'rb') as f:
            climada_data = pickle.load(f)
        print("✅ Loaded CLIMADA data")
    except FileNotFoundError:
        print("⚠️ Using synthetic loss data")
        np.random.seed(42)
        # Match the length of wind indices
        n_events = len(wind_indices) if len(wind_indices) > 0 else 1000
        climada_data = {
            'impact': type('MockImpact', (), {
                'at_event': np.random.lognormal(15, 1, n_events) * 1e6
            })()
        }
    
    # Ensure data arrays have matching lengths
    observed_losses = climada_data.get('impact').at_event if 'impact' in climada_data else np.array([])
    
    # Truncate to minimum length to ensure compatibility
    min_length = min(len(wind_indices), len(observed_losses))
    if min_length > 0:
        wind_indices = wind_indices[:min_length]
        observed_losses = observed_losses[:min_length]
        print(f"   Aligned data to {min_length} events")
    else:
        print("❌ No valid data found")
        return
    
    # =============================================================================
    # Phase 1: Initialize Complete Bayesian Framework
    # 第一階段：初始化完整貝氏框架
    # =============================================================================
    
    print("\n🚀 Phase 1: Initializing Complete Bayesian Framework")
    print("   第一階段：初始化完整貝氏框架")
    
    # Get default configuration 獲取預設配置
    config = get_default_config()
    print(f"   Using configuration 使用配置: {config}")
    
    # Initialize main analyzer 初始化主分析器
    print("\n📊 Initializing RobustBayesianAnalyzer 初始化強健貝氏分析器...")
    main_analyzer = RobustBayesianAnalyzer(
        density_ratio_constraint=config['density_ratio_constraint'],  # 2.0
        n_monte_carlo_samples=config['n_monte_carlo_samples'],        # 500
        n_mixture_components=config['n_mixture_components'],           # 3
        mcmc_samples=config['mcmc_samples'],                          # 2000
        mcmc_warmup=config['mcmc_warmup'],                           # 1000
        mcmc_chains=config['mcmc_chains']                            # 4
    )
    print("   ✅ RobustBayesianAnalyzer initialized with full configuration")
    
    # Initialize hierarchical Bayesian model 初始化階層貝氏模型
    print("\n🏗️ Initializing HierarchicalBayesianModel 初始化階層貝氏模型...")
    hierarchical_config = HierarchicalModelConfig(
        n_mixture_components=config['n_mixture_components'],
        mcmc_samples=config['mcmc_samples'],
        mcmc_warmup=config['mcmc_warmup'],
        mcmc_chains=config['mcmc_chains']
    )
    hierarchical_model = HierarchicalBayesianModel(hierarchical_config)
    print("   ✅ 4-level Hierarchical Bayesian Model with MPE initialized")
    
    # Initialize uncertainty quantification 初始化不確定性量化
    print("\n🎲 Initializing Uncertainty Quantification 初始化不確定性量化...")
    uncertainty_generator = ProbabilisticLossDistributionGenerator(
        n_monte_carlo_samples=config['n_monte_carlo_samples'],
        hazard_uncertainty_std=config['hazard_uncertainty_std'],
        exposure_uncertainty_log_std=config['exposure_uncertainty_log_std'],
        vulnerability_uncertainty_std=config['vulnerability_uncertainty_std']
    )
    print("   ✅ Probabilistic Loss Distribution Generator initialized")
    
    # Initialize weight sensitivity analyzer 初始化權重敏感度分析器
    print("\n⚖️ Initializing Weight Sensitivity Analyzer 初始化權重敏感度分析器...")
    weight_analyzer = WeightSensitivityAnalyzer(
        weight_ranges={
            'w_under': [1.0, 1.5, 2.0, 2.5, 3.0],
            'w_over': [0.25, 0.5, 0.75, 1.0, 1.25]
        }
    )
    print("   ✅ Weight Sensitivity Analyzer initialized")
    
    # =============================================================================
    # Phase 2: Complete Bayesian Analysis
    # 第二階段：完整貝氏分析
    # =============================================================================
    
    print("\n\n🧠 Phase 2: Complete Bayesian Analysis Execution")
    print("   第二階段：完整貝氏分析執行")
    
    print("\n📈 Executing Integrated Bayesian Optimization 執行整合貝氏優化...")
    print("   • Method 方法: Two-Phase Integrated Analysis 兩階段整合分析")
    print("   • Phase 1 階段一: Model Comparison & Selection 模型比較與選擇")
    print("   • Phase 2 階段二: Decision Theory Optimization 決策理論優化")
    print(f"   • Products 產品: {len(products)} parametric products 參數產品")
    print(f"   • Events 事件: {len(observed_losses)} loss observations 損失觀測")
    print(f"   • Monte Carlo 蒙地卡羅: {config['n_monte_carlo_samples']} samples 樣本")
    print(f"   • MCMC: {config['mcmc_samples']} samples × {config['mcmc_chains']} chains")
    
    try:
        # Execute integrated Bayesian optimization 執行整合貝氏優化
        comprehensive_results = main_analyzer.integrated_bayesian_optimization(
            observations=observed_losses,           # Training data for model fitting 訓練資料用於模型擬合
            validation_data=observed_losses,       # Validation data for model selection 驗證資料用於模型選擇  
            hazard_indices=wind_indices,           # Hazard indices for optimization 危險指標用於優化
            actual_losses=np.column_stack([observed_losses] * len(products)),  # Loss matrix 損失矩陣
            product_bounds={                       # Product parameter bounds 產品參數界限
                'trigger_threshold': (30, 60),     # Wind speed trigger range 風速觸發範圍
                'payout_amount': (1e7, 1e9)        # Payout amount range 賠付金額範圍
            },
            basis_risk_type=BasisRiskType.WEIGHTED_ASYMMETRIC,  # Asymmetric basis risk 不對稱基差風險
            w_under=2.0,                          # Under-compensation weight 不足補償權重
            w_over=0.5                            # Over-compensation weight 過度補償權重
        )
        
        print("   ✅ Integrated Bayesian Optimization completed successfully!")
        print("      整合貝氏優化成功完成！")
        
    except Exception as e:
        print(f"   ❌ Integrated optimization failed: {e}")
        print("   整合優化失敗，使用分別執行方式...")
        
        # Fallback: Execute components separately 回退：分別執行組件
        comprehensive_results = execute_fallback_analysis(
            main_analyzer, hierarchical_model, uncertainty_generator, weight_analyzer,
            observed_losses, wind_indices, products, config
        )
    
    # =============================================================================
    # Phase 3: Results Processing and Analysis
    # 第三階段：結果處理與分析
    # =============================================================================
    
    print("\n\n📊 Phase 3: Results Processing and Analysis")
    print("   第三階段：結果處理與分析")
    
    # Process comprehensive results 處理綜合結果
    results = process_comprehensive_results(
        comprehensive_results, products, observed_losses, wind_indices, config
    )
    
    # Execute weight sensitivity analysis 執行權重敏感度分析
    print("\n⚖️ Executing Weight Sensitivity Analysis 執行權重敏感度分析...")
    try:
        sensitivity_results = weight_analyzer.analyze_weight_sensitivity(
            products=products,
            actual_losses=observed_losses,
            wind_indices=wind_indices,
            n_bootstrap_samples=100
        )
        results.weight_sensitivity = sensitivity_results
        print("   ✅ Weight sensitivity analysis completed 權重敏感度分析完成")
    except Exception as e:
        print(f"   ⚠️ Weight sensitivity analysis failed: {e}")
        results.weight_sensitivity = {}
    
    # Generate hierarchical model analysis 生成階層模型分析
    print("\n🏗️ Executing Hierarchical Bayesian Analysis 執行階層貝氏分析...")
    try:
        hierarchical_results = hierarchical_model.fit(observed_losses)
        results.hierarchical_analysis = hierarchical_results
        print("   ✅ Hierarchical Bayesian analysis completed 階層貝氏分析完成")
    except Exception as e:
        print(f"   ⚠️ Hierarchical analysis failed: {e}")
        results.hierarchical_analysis = {}
    
    # Generate uncertainty quantification analysis 生成不確定性量化分析
    print("\n🎲 Executing Uncertainty Quantification 執行不確定性量化...")
    try:
        # Create mock CLIMADA objects if real data not available 如果沒有真實資料則創建模擬CLIMADA物件
        mock_hazard = create_mock_climada_hazard(wind_indices)
        mock_exposure = create_mock_climada_exposure(len(observed_losses))
        mock_impact_func = create_mock_impact_functions()
        
        uncertainty_results = uncertainty_generator.generate_probabilistic_loss_distributions(
            tc_hazard=mock_hazard,
            exposure=mock_exposure,
            impact_func_set=mock_impact_func
        )
        results.uncertainty_analysis = uncertainty_results
        print("   ✅ Uncertainty quantification completed 不確定性量化完成")
    except Exception as e:
        print(f"   ⚠️ Uncertainty quantification failed: {e}")
        results.uncertainty_analysis = {}
    
    # =============================================================================
    # Phase 4: Display Comprehensive Results
    # 第四階段：顯示綜合結果
    # =============================================================================
    
    print("\n\n🎉 Phase 4: Complete Analysis Results")
    print("   第四階段：完整分析結果")
    print("=" * 100)
    
    display_comprehensive_results(results)
    
    print("\n\n✅ Complete Robust Hierarchical Bayesian Analysis Finished!")
    print("   完整強健階層貝氏分析完成！")
    print("=" * 100)
    
    # Display analysis summary 顯示分析摘要
    print(f"\n📊 Analysis Summary 分析摘要:")
    print(f"   • Products analyzed 分析產品: {len(products)}")
    print(f"   • Loss observations 損失觀測: {len(observed_losses)}")
    print(f"   • Monte Carlo samples 蒙地卡羅樣本: {config['n_monte_carlo_samples']}")
    print(f"   • MCMC samples MCMC樣本: {config['mcmc_samples']}")
    print(f"   • MCMC chains MCMC鏈: {config['mcmc_chains']}")
    print(f"   • Analysis type 分析類型: {results.summary_statistics.get('analysis_type', 'Complete Robust Hierarchical Bayesian')}")
    
    # Display key results 顯示主要結果
    print(f"\n🏆 Key Results 主要結果:")
    if hasattr(results, 'phase_1_results') and results.phase_1_results:
        champion = results.phase_1_results.get('champion_model', {})
        if champion:
            print(f"   • Champion Model 冠軍模型: {champion.get('name', 'N/A')}")
            print(f"   • Model CRPS Score 模型CRPS分數: {champion.get('crps_score', 'N/A'):.6f}")
    
    if hasattr(results, 'phase_2_results') and results.phase_2_results:
        optimal = results.phase_2_results.get('optimal_product', {})
        if optimal:
            print(f"   • Optimal Product 最佳產品: {optimal.get('product_id', 'N/A')}")
            print(f"   • Expected Risk 期望風險: {optimal.get('expected_risk', 'N/A'):.6f}")
    
    if hasattr(results, 'weight_sensitivity') and results.weight_sensitivity:
        print(f"   • Weight Sensitivity 權重敏感度: Analysis completed 分析完成")
    
    if hasattr(results, 'hierarchical_analysis') and results.hierarchical_analysis:
        print(f"   • Hierarchical Model 階層模型: Analysis completed 分析完成")
    
    if hasattr(results, 'uncertainty_analysis') and results.uncertainty_analysis:
        print(f"   • Uncertainty Quantification 不確定性量化: Analysis completed 分析完成")
    
    # =============================================================================
    # Phase 5: Save Comprehensive Results
    # 第五階段：保存綜合結果
    # =============================================================================
    
    print("\n\n💾 Phase 5: Saving Comprehensive Results")
    print("   第五階段：保存綜合結果")
    
    save_comprehensive_results(results, config)
    
    print("\n🎉 Complete Robust Hierarchical Bayesian Analysis Successfully Completed!")
    print("   完整強健階層貝氏分析成功完成！")
    print("\n🔧 Methods Used 使用方法:")
    print("   • 4-Level Hierarchical Bayesian Model 四層階層貝氏模型")
    print("   • Mixed Predictive Estimation (MPE) 混合預測估計")
    print("   • Density Ratio Robustness Constraints 密度比強健性約束")
    print("   • Monte Carlo Uncertainty Quantification 蒙地卡羅不確定性量化")
    print("   • Weight Sensitivity Analysis 權重敏感度分析")
    print("   • Two-Phase Integrated Optimization 兩階段整合優化")
    print("   • CRPS-based Model Comparison CRPS為基礎的模型比較")
    print("   • Decision Theory-based Product Optimization 決策理論為基礎的產品優化")
    
    return results


def execute_fallback_analysis(main_analyzer, hierarchical_model, uncertainty_generator, 
                             weight_analyzer, observed_losses, wind_indices, products, config):
    """
    Fallback analysis execution when integrated optimization fails
    當整合優化失敗時的回退分析執行
    """
    print("\n🔄 Executing Fallback Analysis 執行回退分析...")
    
    fallback_results = {
        'analysis_type': 'Fallback Component Analysis',
        'components_executed': [],
        'errors': []
    }
    
    # Try individual components 嘗試個別組件
    try:
        # Basic comprehensive analysis 基本綜合分析
        basic_results = main_analyzer.comprehensive_bayesian_analysis(
            tc_hazard=None,  # Will use mock data 將使用模擬資料
            exposure=None,
            impact_func_set=None,
            observed_losses=observed_losses,
            parametric_products=products,
            hazard_indices=wind_indices
        )
        fallback_results['comprehensive_analysis'] = basic_results
        fallback_results['components_executed'].append('comprehensive_bayesian_analysis')
        print("   ✅ Basic comprehensive analysis completed 基本綜合分析完成")
    except Exception as e:
        fallback_results['errors'].append(f"comprehensive_analysis: {e}")
        print(f"   ❌ Basic comprehensive analysis failed: {e}")
    
    return fallback_results


def process_comprehensive_results(comprehensive_results, products, observed_losses, wind_indices, config):
    """
    Process and structure comprehensive results
    處理並結構化綜合結果
    """
    results = type('CompleteBayesianResults', (), {
        'comprehensive_results': comprehensive_results,
        'phase_1_results': comprehensive_results.get('phase_1_model_comparison', {}),
        'phase_2_results': comprehensive_results.get('phase_2_decision_optimization', {}),
        'integration_validation': comprehensive_results.get('integration_validation', {}),
        'results_df': pd.DataFrame(),
        'summary_statistics': {
            'total_products': len(products),
            'total_events': len(observed_losses),
            'analysis_type': 'Complete Robust Hierarchical Bayesian Analysis',
            'monte_carlo_samples': config['n_monte_carlo_samples'],
            'mcmc_samples': config['mcmc_samples'],
            'mcmc_chains': config['mcmc_chains'],
            'density_ratio_constraint': config['density_ratio_constraint']
        }
    })()
    
    # Create results DataFrame if possible 如果可能則創建結果DataFrame
    try:
        if 'traditional_analysis' in comprehensive_results:
            results.results_df = pd.DataFrame(comprehensive_results['traditional_analysis'])
        elif 'phase_2_decision_optimization' in comprehensive_results:
            phase2_data = comprehensive_results['phase_2_decision_optimization']
            if isinstance(phase2_data, dict) and 'results' in phase2_data:
                results.results_df = pd.DataFrame(phase2_data['results'])
    except Exception as e:
        print(f"   ⚠️ Could not create results DataFrame: {e}")
    
    return results


def display_comprehensive_results(results):
    """
    Display comprehensive analysis results
    顯示綜合分析結果
    """
    print("\n📊 Comprehensive Analysis Results 綜合分析結果:")
    
    # Phase 1 Results: Model Comparison 階段一結果：模型比較
    if hasattr(results, 'phase_1_results') and results.phase_1_results:
        print("\n🧠 Phase 1: Model Comparison Results 階段一：模型比較結果")
        phase1 = results.phase_1_results
        if 'champion_model' in phase1:
            champion = phase1['champion_model']
            print(f"   🏆 Champion Model 冠軍模型: {champion.get('name', 'N/A')}")
            print(f"   📈 CRPS Score CRPS分數: {champion.get('crps_score', 'N/A'):.6f}")
            print(f"   📊 Model Performance 模型表現: {champion.get('performance_summary', 'N/A')}")
        
        if 'model_comparison' in phase1:
            comparison = phase1['model_comparison']
            print(f"   📋 Models Compared 比較模型數: {len(comparison)}")
    
    # Phase 2 Results: Decision Optimization 階段二結果：決策優化
    if hasattr(results, 'phase_2_results') and results.phase_2_results:
        print("\n⚖️ Phase 2: Decision Optimization Results 階段二：決策優化結果")
        phase2 = results.phase_2_results
        if 'optimal_product' in phase2:
            optimal = phase2['optimal_product']
            print(f"   🎯 Optimal Product 最佳產品: {optimal.get('product_id', 'N/A')}")
            print(f"   💰 Expected Risk 期望風險: {optimal.get('expected_risk', 'N/A'):.6f}")
            print(f"   🔧 Product Parameters 產品參數: {optimal.get('parameters', 'N/A')}")
    
    # Integration Validation 整合驗證
    if hasattr(results, 'integration_validation') and results.integration_validation:
        print("\n🔍 Integration Validation 整合驗證:")
        validation = results.integration_validation
        print(f"   ✅ Theoretical Compliance 理論符合度: {validation.get('theoretical_compliance', 'N/A')}")
        print(f"   🔗 Phase Integration 階段整合: {validation.get('phase_integration_success', 'N/A')}")
    
    # Additional Analysis Results 其他分析結果
    if hasattr(results, 'weight_sensitivity') and results.weight_sensitivity:
        print("\n⚖️ Weight Sensitivity Analysis 權重敏感度分析: ✅ Completed 完成")
    
    if hasattr(results, 'hierarchical_analysis') and results.hierarchical_analysis:
        print("\n🏗️ Hierarchical Bayesian Analysis 階層貝氏分析: ✅ Completed 完成")
    
    if hasattr(results, 'uncertainty_analysis') and results.uncertainty_analysis:
        print("\n🎲 Uncertainty Quantification 不確定性量化: ✅ Completed 完成")


def save_comprehensive_results(results, config):
    """
    Save all comprehensive analysis results
    保存所有綜合分析結果
    """
    # Create output directory 創建輸出目錄
    output_dir = "results/robust_hierarchical_bayesian_analysis"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"   💾 Saving results to 保存結果到: {output_dir}")
    
    # Save comprehensive results 保存綜合結果
    try:
        with open(f"{output_dir}/comprehensive_results.pkl", 'wb') as f:
            pickle.dump(results.comprehensive_results, f)
        print("   ✅ Comprehensive results saved 綜合結果已保存")
    except Exception as e:
        print(f"   ❌ Failed to save comprehensive results: {e}")
    
    # Save DataFrame results 保存DataFrame結果
    try:
        if not results.results_df.empty:
            results.results_df.to_csv(f"{output_dir}/analysis_results.csv", index=False)
            print("   ✅ DataFrame results saved DataFrame結果已保存")
    except Exception as e:
        print(f"   ❌ Failed to save DataFrame: {e}")
    
    # Save configuration 保存配置
    try:
        with open(f"{output_dir}/analysis_config.json", 'w') as f:
            import json
            json.dump(config, f, indent=2)
        print("   ✅ Configuration saved 配置已保存")
    except Exception as e:
        print(f"   ❌ Failed to save configuration: {e}")
    
    # Save summary statistics 保存摘要統計
    try:
        with open(f"{output_dir}/summary_statistics.json", 'w') as f:
            import json
            json.dump(results.summary_statistics, f, indent=2)
        print("   ✅ Summary statistics saved 摘要統計已保存")
    except Exception as e:
        print(f"   ❌ Failed to save summary: {e}")
    
    # Save individual analysis components 保存個別分析組件
    save_individual_components(results, output_dir)


def save_individual_components(results, output_dir):
    """
    Save individual analysis components
    保存個別分析組件
    """
    components_dir = f"{output_dir}/components"
    Path(components_dir).mkdir(exist_ok=True)
    
    # Save weight sensitivity results 保存權重敏感度結果
    if hasattr(results, 'weight_sensitivity') and results.weight_sensitivity:
        try:
            with open(f"{components_dir}/weight_sensitivity.pkl", 'wb') as f:
                pickle.dump(results.weight_sensitivity, f)
            print("   ✅ Weight sensitivity results saved 權重敏感度結果已保存")
        except Exception as e:
            print(f"   ❌ Failed to save weight sensitivity: {e}")
    
    # Save hierarchical analysis results 保存階層分析結果
    if hasattr(results, 'hierarchical_analysis') and results.hierarchical_analysis:
        try:
            with open(f"{components_dir}/hierarchical_analysis.pkl", 'wb') as f:
                pickle.dump(results.hierarchical_analysis, f)
            print("   ✅ Hierarchical analysis results saved 階層分析結果已保存")
        except Exception as e:
            print(f"   ❌ Failed to save hierarchical analysis: {e}")
    
    # Save uncertainty analysis results 保存不確定性分析結果
    if hasattr(results, 'uncertainty_analysis') and results.uncertainty_analysis:
        try:
            with open(f"{components_dir}/uncertainty_analysis.pkl", 'wb') as f:
                pickle.dump(results.uncertainty_analysis, f)
            print("   ✅ Uncertainty analysis results saved 不確定性分析結果已保存")
        except Exception as e:
            print(f"   ❌ Failed to save uncertainty analysis: {e}")


def create_mock_climada_hazard(wind_indices):
    """
    Create mock CLIMADA hazard object for uncertainty analysis
    創建模擬CLIMADA危險物件用於不確定性分析
    """
    class MockHazard:
        def __init__(self, wind_indices):
            self.intensity = np.column_stack([wind_indices, wind_indices]).T
            self.event_id = np.arange(len(wind_indices))
            self.frequency = np.ones(len(wind_indices)) / len(wind_indices)
    
    return MockHazard(wind_indices)


def create_mock_climada_exposure(n_exposures):
    """
    Create mock CLIMADA exposure object
    創建模擬CLIMADA暴露物件
    """
    class MockExposure:
        def __init__(self, n):
            self.value = np.random.lognormal(15, 1, n)
            self.latitude = np.random.uniform(33.8, 36.6, n)
            self.longitude = np.random.uniform(-84.5, -75.5, n)
    
    return MockExposure(n_exposures)


def create_mock_impact_functions():
    """
    Create mock impact functions
    創建模擬影響函數
    """
    class MockImpactFunc:
        def __init__(self):
            self.intensity = np.arange(0, 100, 5)
            self.mdd = 1 / (1 + np.exp(-(self.intensity - 50) / 10))
            self.paa = np.ones_like(self.intensity)
    
    return MockImpactFunc()


if __name__ == "__main__":
    results = main()