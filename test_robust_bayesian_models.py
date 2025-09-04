#!/usr/bin/env python3
"""
Robust Bayesian Model Testing Framework
穩健貝葉斯模型測試框架

系統性測試不同prior和likelihood組合在robust bayesian框架下的效果
評估框架在不同模型假設下的穩健性

Author: Research Team
Date: 2025-08-22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
import time
from pathlib import Path

# 導入robust bayesian組件
from robust_hierarchical_bayesian_simulation.robust_priors import (
    DoubleEpsilonContamination,
    EpsilonContaminationSpec,
    ContaminationDistributionClass,
    create_typhoon_contamination_spec
)

@dataclass
class ModelTestConfig:
    """模型測試配置"""
    prior_family: str = "normal"  # normal, t, gev, laplace, gamma
    likelihood_contamination: str = "extreme_events"  # outliers, extreme_events, measurement_error
    prior_contamination: str = "typhoon_specific"  # heavy_tailed, extreme_value, misspecified
    epsilon_prior: float = 0.05
    epsilon_likelihood: float = 0.08
    sample_size: int = 500
    n_simulations: int = 100

@dataclass 
class ModelTestResult:
    """模型測試結果"""
    config: ModelTestConfig
    posterior_means: np.ndarray
    posterior_stds: np.ndarray
    effective_sample_sizes: np.ndarray
    variance_inflations: np.ndarray
    sample_size_reductions: np.ndarray
    crps_scores: np.ndarray
    coverage_probabilities: np.ndarray
    bias: float
    rmse: float
    mean_variance_inflation: float
    mean_sample_reduction: float

class RobustBayesianModelTester:
    """穩健貝葉斯模型測試器"""
    
    def __init__(self, true_parameters: Dict[str, float] = None):
        self.true_parameters = true_parameters or {
            'location': 1e7,  # $10M 
            'scale': 2e6      # $2M
        }
        self.test_results: Dict[str, ModelTestResult] = {}
        
        print("🧪 穩健貝葉斯模型測試器初始化")
        print(f"   真實參數: μ=${self.true_parameters['location']/1e6:.1f}M, σ=${self.true_parameters['scale']/1e6:.1f}M")
    
    def generate_test_data(self, config: ModelTestConfig, 
                          contamination_ratio: float = None) -> Tuple[np.ndarray, Dict]:
        """
        生成測試數據（含污染）
        
        Returns:
        --------
        Tuple[np.ndarray, Dict] : (數據, 生成參數)
        """
        if contamination_ratio is None:
            contamination_ratio = config.epsilon_likelihood
            
        n_clean = int(config.sample_size * (1 - contamination_ratio))
        n_contaminated = config.sample_size - n_clean
        
        # 生成乾淨數據
        clean_data = np.random.lognormal(
            mean=np.log(self.true_parameters['location']),
            sigma=self.true_parameters['scale'] / self.true_parameters['location'],
            size=n_clean
        )
        
        # 生成污染數據（根據contamination type）
        if config.likelihood_contamination == "outliers":
            # 極值離群值
            contaminated_data = np.random.lognormal(
                mean=np.log(self.true_parameters['location'] * 5),
                sigma=2.0,
                size=n_contaminated
            )
        elif config.likelihood_contamination == "extreme_events":
            # 極端事件（颱風）
            contaminated_data = np.random.exponential(
                scale=self.true_parameters['location'] * 3,
                size=n_contaminated
            )
        elif config.likelihood_contamination == "measurement_error":
            # 測量誤差
            contaminated_data = clean_data[:n_contaminated] + np.random.normal(
                0, self.true_parameters['scale'] * 2, n_contaminated
            )
        else:
            contaminated_data = clean_data[:n_contaminated]
        
        # 混合數據
        mixed_data = np.concatenate([clean_data, contaminated_data])
        np.random.shuffle(mixed_data)
        
        generation_info = {
            'n_clean': n_clean,
            'n_contaminated': n_contaminated,
            'contamination_ratio': contamination_ratio,
            'clean_mean': np.mean(clean_data),
            'contaminated_mean': np.mean(contaminated_data) if len(contaminated_data) > 0 else 0,
            'mixed_mean': np.mean(mixed_data)
        }
        
        return mixed_data, generation_info
    
    def create_prior_parameters(self, config: ModelTestConfig) -> Dict[str, Any]:
        """創建先驗參數（基於family類型）"""
        
        if config.prior_family == "normal":
            return {
                'location': self.true_parameters['location'],
                'scale': self.true_parameters['scale'],
                'family': 'normal'
            }
        elif config.prior_family == "t":
            return {
                'location': self.true_parameters['location'],
                'scale': self.true_parameters['scale'],
                'df': 5,  # degrees of freedom
                'family': 't'
            }
        elif config.prior_family == "gev":
            return {
                'location': self.true_parameters['location'],
                'scale': self.true_parameters['scale'],
                'shape': 0.1,  # GEV shape parameter
                'family': 'gev'
            }
        elif config.prior_family == "laplace":
            return {
                'location': self.true_parameters['location'],
                'scale': self.true_parameters['scale'] / np.sqrt(2),  # Laplace scale adjustment
                'family': 'laplace'
            }
        elif config.prior_family == "gamma":
            # Convert to Gamma parameters
            mean = self.true_parameters['location']
            var = self.true_parameters['scale'] ** 2
            alpha = mean ** 2 / var
            beta = mean / var
            return {
                'alpha': alpha,
                'beta': beta,
                'family': 'gamma'
            }
        else:
            raise ValueError(f"未支援的prior family: {config.prior_family}")
    
    def compute_crps(self, posterior_samples: np.ndarray, true_value: float) -> float:
        """計算CRPS (Continuous Ranked Probability Score)"""
        posterior_samples = np.sort(posterior_samples)
        n = len(posterior_samples)
        
        # CRPS公式實現
        crps = 0.0
        for i, sample in enumerate(posterior_samples):
            p_i = (i + 1) / n
            indicator = 1.0 if true_value <= sample else 0.0
            crps += (p_i - indicator) ** 2
        
        return crps / n
    
    def compute_coverage_probability(self, posterior_samples: np.ndarray, 
                                   true_value: float, confidence: float = 0.95) -> float:
        """計算覆蓋概率"""
        alpha = 1 - confidence
        lower = np.percentile(posterior_samples, 100 * alpha / 2)
        upper = np.percentile(posterior_samples, 100 * (1 - alpha / 2))
        return 1.0 if lower <= true_value <= upper else 0.0
    
    def test_single_configuration(self, config: ModelTestConfig) -> ModelTestResult:
        """測試單一配置"""
        print(f"\n🔬 測試配置: {config.prior_family} prior + {config.likelihood_contamination} likelihood")
        print(f"   污染程度: ε₁={config.epsilon_prior:.3f}, ε₂={config.epsilon_likelihood:.3f}")
        
        # 儲存結果
        posterior_means = []
        posterior_stds = []
        effective_sample_sizes = []
        variance_inflations = []
        sample_size_reductions = []
        crps_scores = []
        coverage_probabilities = []
        
        start_time = time.time()
        
        for sim_i in range(config.n_simulations):
            if sim_i % 20 == 0:
                print(f"   模擬 {sim_i+1}/{config.n_simulations}...")
            
            # 生成測試數據
            test_data, gen_info = self.generate_test_data(config)
            
            # 創建先驗參數
            base_prior_params = self.create_prior_parameters(config)
            
            # 創建雙重污染模型
            double_contamination = DoubleEpsilonContamination(
                epsilon_prior=config.epsilon_prior,
                epsilon_likelihood=config.epsilon_likelihood,
                prior_contamination_type=config.prior_contamination,
                likelihood_contamination_type=config.likelihood_contamination
            )
            
            try:
                # 計算穩健後驗
                robust_posterior = double_contamination.compute_robust_posterior(
                    data=test_data,
                    base_prior_params=base_prior_params,
                    likelihood_params={}
                )
                
                # 儲存結果
                posterior_means.append(robust_posterior['posterior_mean'])
                posterior_stds.append(robust_posterior['posterior_std'])
                effective_sample_sizes.append(robust_posterior['effective_sample_size'])
                variance_inflations.append(robust_posterior['contamination_impact']['variance_inflation'])
                sample_size_reductions.append(robust_posterior['contamination_impact']['sample_size_reduction'])
                
                # 生成後驗樣本用於評估
                posterior_samples = np.random.normal(
                    robust_posterior['posterior_mean'],
                    robust_posterior['posterior_std'],
                    1000
                )
                
                # 計算CRPS和覆蓋概率
                crps = self.compute_crps(posterior_samples, self.true_parameters['location'])
                coverage = self.compute_coverage_probability(posterior_samples, self.true_parameters['location'])
                
                crps_scores.append(crps)
                coverage_probabilities.append(coverage)
                
            except Exception as e:
                warnings.warn(f"模擬 {sim_i} 失敗: {e}")
                # 使用NaN填充
                posterior_means.append(np.nan)
                posterior_stds.append(np.nan)
                effective_sample_sizes.append(np.nan)
                variance_inflations.append(np.nan)
                sample_size_reductions.append(np.nan)
                crps_scores.append(np.nan)
                coverage_probabilities.append(np.nan)
        
        elapsed_time = time.time() - start_time
        print(f"   完成! 耗時: {elapsed_time:.1f}秒")
        
        # 轉換為numpy數組並計算摘要統計
        posterior_means = np.array(posterior_means)
        posterior_stds = np.array(posterior_stds)
        effective_sample_sizes = np.array(effective_sample_sizes)
        variance_inflations = np.array(variance_inflations)
        sample_size_reductions = np.array(sample_size_reductions)
        crps_scores = np.array(crps_scores)
        coverage_probabilities = np.array(coverage_probabilities)
        
        # 去除NaN值計算統計
        valid_mask = ~np.isnan(posterior_means)
        if np.sum(valid_mask) == 0:
            raise ValueError("所有模擬都失敗了")
        
        bias = np.nanmean(posterior_means) - self.true_parameters['location']
        rmse = np.sqrt(np.nanmean((posterior_means - self.true_parameters['location']) ** 2))
        mean_variance_inflation = np.nanmean(variance_inflations)
        mean_sample_reduction = np.nanmean(sample_size_reductions)
        
        return ModelTestResult(
            config=config,
            posterior_means=posterior_means,
            posterior_stds=posterior_stds,
            effective_sample_sizes=effective_sample_sizes,
            variance_inflations=variance_inflations,
            sample_size_reductions=sample_size_reductions,
            crps_scores=crps_scores,
            coverage_probabilities=coverage_probabilities,
            bias=bias,
            rmse=rmse,
            mean_variance_inflation=mean_variance_inflation,
            mean_sample_reduction=mean_sample_reduction
        )
    
    def run_comprehensive_test(self, 
                             prior_families: List[str] = None,
                             likelihood_contaminations: List[str] = None,
                             epsilon_values: List[Tuple[float, float]] = None) -> Dict[str, ModelTestResult]:
        """執行全面測試"""
        
        if prior_families is None:
            prior_families = ["normal", "t", "gev", "laplace"]
            
        if likelihood_contaminations is None:
            likelihood_contaminations = ["outliers", "extreme_events", "measurement_error"]
            
        if epsilon_values is None:
            epsilon_values = [(0.05, 0.08), (0.10, 0.12), (0.15, 0.18)]
        
        print("🚀 開始全面穩健貝葉斯模型測試")
        print(f"   Prior families: {prior_families}")
        print(f"   Likelihood contaminations: {likelihood_contaminations}")
        print(f"   Epsilon值: {epsilon_values}")
        
        results = {}
        total_tests = len(prior_families) * len(likelihood_contaminations) * len(epsilon_values)
        test_count = 0
        
        for prior_family in prior_families:
            for likelihood_cont in likelihood_contaminations:
                for eps_prior, eps_likelihood in epsilon_values:
                    test_count += 1
                    config_name = f"{prior_family}_{likelihood_cont}_eps{eps_prior:.2f}_{eps_likelihood:.2f}"
                    
                    print(f"\n📊 測試 {test_count}/{total_tests}: {config_name}")
                    
                    config = ModelTestConfig(
                        prior_family=prior_family,
                        likelihood_contamination=likelihood_cont,
                        epsilon_prior=eps_prior,
                        epsilon_likelihood=eps_likelihood,
                        n_simulations=50  # 減少模擬次數以加快測試
                    )
                    
                    try:
                        result = self.test_single_configuration(config)
                        results[config_name] = result
                        
                        print(f"   結果: Bias=${result.bias/1e6:.2f}M, RMSE=${result.rmse/1e6:.2f}M")
                        print(f"         變異數膨脹={result.mean_variance_inflation:.2f}x")
                        print(f"         平均覆蓋率={np.nanmean(result.coverage_probabilities):.3f}")
                        
                    except Exception as e:
                        print(f"   ❌ 測試失敗: {e}")
                        warnings.warn(f"配置 {config_name} 測試失敗: {e}")
        
        self.test_results = results
        print(f"\n✅ 全面測試完成! 成功測試了 {len(results)}/{total_tests} 個配置")
        return results
    
    def create_comparison_report(self, save_path: str = None) -> pd.DataFrame:
        """創建比較報告"""
        if not self.test_results:
            raise ValueError("沒有測試結果，請先運行測試")
        
        report_data = []
        
        for config_name, result in self.test_results.items():
            row = {
                'Configuration': config_name,
                'Prior_Family': result.config.prior_family,
                'Likelihood_Contamination': result.config.likelihood_contamination, 
                'Epsilon_Prior': result.config.epsilon_prior,
                'Epsilon_Likelihood': result.config.epsilon_likelihood,
                'Bias_Million': result.bias / 1e6,
                'RMSE_Million': result.rmse / 1e6,
                'Mean_Variance_Inflation': result.mean_variance_inflation,
                'Mean_Sample_Reduction': result.mean_sample_reduction,
                'Mean_CRPS': np.nanmean(result.crps_scores),
                'Mean_Coverage': np.nanmean(result.coverage_probabilities),
                'Std_Coverage': np.nanstd(result.coverage_probabilities)
            }
            report_data.append(row)
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('RMSE_Million')
        
        if save_path:
            report_df.to_csv(save_path, index=False)
            print(f"📄 比較報告已儲存至: {save_path}")
        
        return report_df
    
    def plot_comparison_results(self, save_dir: str = None):
        """繪製比較結果"""
        if not self.test_results:
            raise ValueError("沒有測試結果，請先運行測試")
        
        # 創建比較報告
        report_df = self.create_comparison_report()
        
        # 設置繪圖風格
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Robust Bayesian Model Comparison Results', fontsize=16, fontweight='bold')
        
        # 1. Bias comparison
        ax1 = axes[0, 0]
        sns.boxplot(data=report_df, x='Prior_Family', y='Bias_Million', ax=ax1)
        ax1.set_title('Bias by Prior Family')
        ax1.set_ylabel('Bias ($M)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. RMSE comparison
        ax2 = axes[0, 1]
        sns.boxplot(data=report_df, x='Likelihood_Contamination', y='RMSE_Million', ax=ax2)
        ax2.set_title('RMSE by Likelihood Contamination')
        ax2.set_ylabel('RMSE ($M)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Variance inflation
        ax3 = axes[0, 2]
        sns.scatterplot(data=report_df, x='Epsilon_Prior', y='Mean_Variance_Inflation', 
                       hue='Prior_Family', ax=ax3)
        ax3.set_title('Variance Inflation vs Prior Epsilon')
        ax3.set_ylabel('Variance Inflation')
        
        # 4. Coverage probability
        ax4 = axes[1, 0]
        sns.boxplot(data=report_df, x='Prior_Family', y='Mean_Coverage', ax=ax4)
        ax4.set_title('Coverage Probability by Prior Family')
        ax4.set_ylabel('Coverage Probability')
        ax4.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target 95%')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. CRPS scores
        ax5 = axes[1, 1]
        sns.scatterplot(data=report_df, x='Epsilon_Likelihood', y='Mean_CRPS',
                       hue='Likelihood_Contamination', ax=ax5)
        ax5.set_title('CRPS vs Likelihood Epsilon')
        ax5.set_ylabel('Mean CRPS')
        
        # 6. Sample size reduction
        ax6 = axes[1, 2]
        sns.heatmap(report_df.pivot_table(values='Mean_Sample_Reduction', 
                                        index='Prior_Family', 
                                        columns='Likelihood_Contamination'),
                   annot=True, fmt='.3f', ax=ax6, cmap='YlOrRd')
        ax6.set_title('Sample Size Reduction Heatmap')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'robust_bayesian_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 結果圖表已儲存至: {save_path}")
        
        plt.show()

def main():
    """主測試函數"""
    print("🧪 穩健貝葉斯模型效果測試")
    print("=" * 50)
    
    # 創建測試器
    tester = RobustBayesianModelTester()
    
    # 執行全面測試
    results = tester.run_comprehensive_test(
        prior_families=["normal", "t", "gev"],
        likelihood_contaminations=["outliers", "extreme_events"],
        epsilon_values=[(0.05, 0.08), (0.10, 0.15)]
    )
    
    # 創建報告
    report_df = tester.create_comparison_report('robust_bayesian_comparison_report.csv')
    print("\n📊 前5名最佳配置 (按RMSE排序):")
    print(report_df.head()[['Configuration', 'Bias_Million', 'RMSE_Million', 
                          'Mean_Variance_Inflation', 'Mean_Coverage']].round(3))
    
    # 繪製結果
    tester.plot_comparison_results(save_dir='.')
    
    print("\n✅ 測試完成!")

if __name__ == "__main__":
    main()