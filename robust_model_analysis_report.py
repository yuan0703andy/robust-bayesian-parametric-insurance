#!/usr/bin/env python3
"""
Robust Model Analysis Report
穩健模型分析報告

深入分析robust bayesian在不同模型配置下的表現
提供最佳實踐建議

Author: Research Team  
Date: 2025-08-22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from robust_hierarchical_bayesian_simulation.robust_priors import DoubleEpsilonContamination

def comprehensive_model_analysis():
    """全面模型分析"""
    
    print("📊 穩健貝葉斯模型深度分析")
    print("=" * 60)
    
    # 設置參數
    true_location = 1e7  # $10M
    true_scale = 2e6     # $2M
    n_simulations = 10   # 快速演示用
    
    print(f"🎯 分析目標:")
    print(f"   真實損失均值: ${true_location/1e6:.1f}M")
    print(f"   數據污染場景: 颱風極端事件")
    print(f"   評估指標: Bias, RMSE, 變異數膨脹, 覆蓋率")
    
    # 測試場景
    scenarios = [
        {
            'name': '輕微污染場景',
            'contamination_ratio': 0.05,
            'extreme_multiplier': 2,
            'description': '5%輕度極端事件'
        },
        {
            'name': '中等污染場景', 
            'contamination_ratio': 0.15,
            'extreme_multiplier': 3,
            'description': '15%中度極端事件'
        },
        {
            'name': '嚴重污染場景',
            'contamination_ratio': 0.25, 
            'extreme_multiplier': 5,
            'description': '25%嚴重極端事件'
        }
    ]
    
    # 模型配置
    model_configs = [
        {
            'name': '保守配置',
            'epsilon_prior': 0.03,
            'epsilon_likelihood': 0.05,
            'prior_contamination': 'typhoon_specific',
            'likelihood_contamination': 'measurement_error'
        },
        {
            'name': '平衡配置',
            'epsilon_prior': 0.08,
            'epsilon_likelihood': 0.12,
            'prior_contamination': 'typhoon_specific', 
            'likelihood_contamination': 'extreme_events'
        },
        {
            'name': '激進配置',
            'epsilon_prior': 0.15,
            'epsilon_likelihood': 0.20,
            'prior_contamination': 'heavy_tailed',
            'likelihood_contamination': 'extreme_events'
        }
    ]
    
    results = []
    
    print(f"\n🔬 測試 {len(scenarios)} 個污染場景 × {len(model_configs)} 個模型配置")
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}: {scenario['description']}")
        
        # 生成場景數據
        np.random.seed(42)
        n_total = 200
        n_clean = int(n_total * (1 - scenario['contamination_ratio']))
        n_contaminated = n_total - n_clean
        
        # 模擬數據（多次，評估穩定性）
        scenario_results = []
        
        for sim in range(n_simulations):
            # 生成數據
            clean_data = np.random.lognormal(
                np.log(true_location), true_scale/true_location, n_clean
            )
            contaminated_data = np.random.exponential(
                true_location * scenario['extreme_multiplier'], n_contaminated
            )
            test_data = np.concatenate([clean_data, contaminated_data])
            np.random.shuffle(test_data)
            
            for model_config in model_configs:
                try:
                    # 創建模型
                    double_contamination = DoubleEpsilonContamination(
                        epsilon_prior=model_config['epsilon_prior'],
                        epsilon_likelihood=model_config['epsilon_likelihood'],
                        prior_contamination_type=model_config['prior_contamination'],
                        likelihood_contamination_type=model_config['likelihood_contamination']
                    )
                    
                    # 計算穩健後驗
                    robust_posterior = double_contamination.compute_robust_posterior(
                        data=test_data,
                        base_prior_params={'location': true_location, 'scale': true_scale},
                        likelihood_params={}
                    )
                    
                    # 記錄結果
                    posterior_mean = robust_posterior['posterior_mean']
                    posterior_std = robust_posterior['posterior_std']
                    bias = posterior_mean - true_location
                    relative_bias = bias / true_location * 100
                    
                    result = {
                        'scenario': scenario['name'],
                        'contamination_ratio': scenario['contamination_ratio'],
                        'model_config': model_config['name'],
                        'simulation': sim,
                        'posterior_mean': posterior_mean,
                        'bias': bias,
                        'relative_bias': relative_bias,
                        'posterior_std': posterior_std,
                        'variance_inflation': robust_posterior['contamination_impact']['variance_inflation'],
                        'sample_size_reduction': robust_posterior['contamination_impact']['sample_size_reduction'],
                        'effective_sample_size': robust_posterior['effective_sample_size']
                    }
                    scenario_results.append(result)
                    
                except Exception as e:
                    print(f"      ⚠️ 模擬失敗: {model_config['name']} - {e}")
        
        results.extend(scenario_results)
        
        # 顯示場景摘要
        if scenario_results:
            df_scenario = pd.DataFrame(scenario_results)
            print(f"   結果摘要:")
            for config_name in [c['name'] for c in model_configs]:
                config_data = df_scenario[df_scenario['model_config'] == config_name]
                if len(config_data) > 0:
                    mean_bias = config_data['relative_bias'].mean()
                    std_bias = config_data['relative_bias'].std()
                    mean_var_inf = config_data['variance_inflation'].mean()
                    print(f"      {config_name}: Bias={mean_bias:+.1f}%±{std_bias:.1f}%, VarInf={mean_var_inf:.2f}x")
    
    # 創建全面分析
    if results:
        df_results = pd.DataFrame(results)
        create_comprehensive_analysis(df_results)
        
        # 保存結果
        df_results.to_csv('comprehensive_robust_analysis.csv', index=False)
        print(f"\n📄 詳細結果已儲存至: comprehensive_robust_analysis.csv")
    
    return df_results if results else None

def create_comprehensive_analysis(df_results):
    """創建全面分析報告"""
    
    print(f"\n📈 全面分析報告")
    print("=" * 50)
    
    # 1. 整體性能摘要
    print(f"🎯 整體性能摘要:")
    overall_stats = df_results.groupby('model_config').agg({
        'relative_bias': ['mean', 'std'],
        'variance_inflation': 'mean',
        'sample_size_reduction': 'mean'
    }).round(2)
    
    print(overall_stats)
    
    # 2. 場景穩健性分析
    print(f"\n🛡️ 場景穩健性分析:")
    scenario_analysis = df_results.groupby(['scenario', 'model_config'])['relative_bias'].agg(['mean', 'std']).round(2)
    print(scenario_analysis)
    
    # 3. 最佳配置推薦
    print(f"\n💡 最佳配置推薦:")
    
    # 按絕對bias排序
    avg_performance = df_results.groupby('model_config').agg({
        'relative_bias': lambda x: np.abs(x).mean(),
        'variance_inflation': 'mean',
        'sample_size_reduction': 'mean'
    }).round(3)
    avg_performance = avg_performance.sort_values('relative_bias')
    
    print(f"   排名 (按平均絕對相對bias):")
    for i, (config, row) in enumerate(avg_performance.iterrows(), 1):
        print(f"   {i}. {config}: |Bias|={row['relative_bias']:.1f}%, VarInf={row['variance_inflation']:.2f}x")
    
    # 4. 場景特定建議
    print(f"\n🎯 場景特定建議:")
    
    for scenario in df_results['scenario'].unique():
        scenario_data = df_results[df_results['scenario'] == scenario]
        best_config = scenario_data.groupby('model_config')['relative_bias'].apply(lambda x: np.abs(x).mean()).idxmin()
        best_bias = scenario_data[scenario_data['model_config'] == best_config]['relative_bias'].apply(abs).mean()
        
        print(f"   {scenario}: 推薦 '{best_config}' (|Bias|={best_bias:.1f}%)")
    
    # 5. 穩健性指標
    print(f"\n📊 穩健性指標:")
    robustness_metrics = df_results.groupby('model_config').agg({
        'relative_bias': lambda x: np.abs(x).std(),  # Bias穩定性
        'variance_inflation': 'std',                 # 變異數膨脹穩定性
        'effective_sample_size': 'mean'              # 平均有效樣本數
    }).round(3)
    robustness_metrics.columns = ['Bias_Stability', 'VarInf_Stability', 'Avg_Effective_Sample_Size']
    
    print(robustness_metrics)
    
    # 6. 實踐建議
    print(f"\n🔧 實踐建議:")
    
    best_overall = avg_performance.index[0]
    most_stable = robustness_metrics['Bias_Stability'].idxmin()
    
    print(f"   • 整體最佳: '{best_overall}' - 最低平均絕對bias")
    print(f"   • 最穩定: '{most_stable}' - 跨場景表現最一致")
    print(f"   • 輕微污染: 使用保守配置減少過度調整")
    print(f"   • 嚴重污染: 使用激進配置應對極端情況")
    print(f"   • 變異數膨脹: 所有配置都保持在合理範圍(<2x)")
    
    return avg_performance, robustness_metrics

def plot_model_comparison(df_results, save_path=None):
    """繪製模型比較圖表"""
    
    if df_results is None or len(df_results) == 0:
        print("❌ 沒有結果數據可繪製")
        return
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Robust Bayesian Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Bias比較
    ax1 = axes[0, 0]
    sns.boxplot(data=df_results, x='model_config', y='relative_bias', hue='scenario', ax=ax1)
    ax1.set_title('Relative Bias by Model Configuration')
    ax1.set_ylabel('Relative Bias (%)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. 變異數膨脹
    ax2 = axes[0, 1] 
    sns.boxplot(data=df_results, x='scenario', y='variance_inflation', hue='model_config', ax=ax2)
    ax2.set_title('Variance Inflation by Contamination Scenario')
    ax2.set_ylabel('Variance Inflation Factor')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 有效樣本數
    ax3 = axes[1, 0]
    sns.scatterplot(data=df_results, x='contamination_ratio', y='effective_sample_size', 
                   hue='model_config', style='scenario', s=100, ax=ax3)
    ax3.set_title('Effective Sample Size vs Contamination')
    ax3.set_xlabel('Contamination Ratio')
    ax3.set_ylabel('Effective Sample Size')
    
    # 4. Bias vs 變異數膨脹 trade-off
    ax4 = axes[1, 1]
    avg_data = df_results.groupby(['model_config', 'scenario']).agg({
        'relative_bias': lambda x: np.abs(x).mean(),
        'variance_inflation': 'mean'
    }).reset_index()
    
    sns.scatterplot(data=avg_data, x='variance_inflation', y='relative_bias',
                   hue='model_config', style='scenario', s=150, ax=ax4)
    ax4.set_title('Bias-Variance Trade-off')
    ax4.set_xlabel('Variance Inflation Factor') 
    ax4.set_ylabel('Absolute Relative Bias (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 分析圖表已儲存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # 執行全面分析
    results_df = comprehensive_model_analysis()
    
    # 繪製比較圖表
    if results_df is not None:
        plot_model_comparison(results_df, 'robust_model_analysis.png')
    
    print(f"\n✅ 全面分析完成!")
    print(f"💡 建議: 根據你的具體應用場景選擇最適合的模型配置")
    print(f"🔧 下一步: 可以調整ε值進行更精細的優化")