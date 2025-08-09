#!/usr/bin/env python3
"""
Enhancement Script for Bayesian Reports
增強貝氏報告統計指標

This script demonstrates how to enhance the bayesian/reports/ module to include
all statistical metrics mentioned in main_analysis.py and robust_bayesian_crps_analysis.py
"""

import numpy as np
from typing import Dict, Any
from bayesian.reports.bayesian_report_generator import BayesianReportGenerator, ReportConfig


def create_enhanced_analysis_metadata(wind_uncertainty_cv: float = 0.15,
                                    exposure_uncertainty_log_std: float = 0.20,
                                    vulnerability_param_uncertainty: float = 0.10,
                                    n_monte_carlo: int = 500,
                                    best_distribution: str = "Normal",
                                    mcmc_convergence: str = "converged",
                                    n_mixture_components: int = 3,
                                    density_ratio_constraint: float = 2.0) -> Dict[str, Any]:
    """
    Create enhanced analysis metadata with all required statistical parameters
    創造包含所有必要統計參數的增強分析元數據
    """
    
    return {
        # Uncertainty quantification parameters
        'wind_uncertainty_cv': wind_uncertainty_cv,
        'exposure_uncertainty_log_std': exposure_uncertainty_log_std, 
        'vulnerability_param_uncertainty': vulnerability_param_uncertainty,
        
        # Monte Carlo parameters
        'n_monte_carlo': n_monte_carlo,
        
        # Model selection results
        'best_distribution': best_distribution,
        'mcmc_convergence': mcmc_convergence,
        
        # Mixed predictive estimation
        'n_mixture_components': n_mixture_components,
        
        # Robustness parameters
        'density_ratio_constraint': density_ratio_constraint,
        
        # Additional metadata
        'analysis_timestamp': '2025-08-05T04:06:25',
        'analysis_version': '2.1.0',
        'framework': 'Robust Bayesian CRPS',
        
        # Skill score parameters
        'skill_score_metrics': ['CRPS', 'CRPSS', 'EDI', 'TSS', 'RMSE', 'MAE'],
        'cross_validation_folds': 5,
        'confidence_levels': [0.5, 0.8, 0.9, 0.95],
        
        # Bayesian specific parameters
        'prior_scenarios': ['informative_normal', 'weak_normal', 'uniform', 'gamma_shape1', 'gamma_shape2'],
        'mcmc_chains': 4,
        'mcmc_warmup': 1000,
        'mcmc_samples': 2000,
        
        # Robustness thresholds
        'rhat_threshold': 1.1,
        'ess_threshold': 400,
        'sensitivity_threshold': 0.3
    }


def generate_enhanced_statistical_summary(analysis_metadata: Dict[str, Any]) -> str:
    """
    Generate an enhanced statistical summary with all key metrics
    生成包含所有關鍵指標的增強統計摘要
    """
    
    summary = []
    summary.append("=" * 80)
    summary.append("                增強統計參數摘要")
    summary.append("=" * 80)
    summary.append("")
    
    # Uncertainty Quantification Section
    summary.append("🎲 不確定性量化參數")
    summary.append("-" * 40)
    summary.append(f"風速不確定性 CV: {analysis_metadata['wind_uncertainty_cv']:.1%}")
    summary.append(f"曝險不確定性 log-σ: {analysis_metadata['exposure_uncertainty_log_std']:.2f}")
    summary.append(f"脆弱性參數不確定性: {analysis_metadata['vulnerability_param_uncertainty']:.1%}")
    summary.append(f"Monte Carlo樣本數: {analysis_metadata['n_monte_carlo']:,}")
    summary.append("")
    
    # Model Selection Section
    summary.append("📊 模型選擇結果")
    summary.append("-" * 40)
    summary.append(f"最佳分布模型: {analysis_metadata['best_distribution']}")
    summary.append(f"MCMC收斂狀態: {analysis_metadata['mcmc_convergence']}")
    summary.append(f"混合組件數: {analysis_metadata['n_mixture_components']}")
    summary.append("")
    
    # Robustness Parameters Section
    summary.append("🛡️ 穩健性參數")
    summary.append("-" * 40)
    summary.append(f"密度比約束 (γ): {analysis_metadata['density_ratio_constraint']:.1f}")
    summary.append(f"R-hat 閾值: {analysis_metadata['rhat_threshold']:.2f}")
    summary.append(f"ESS 閾值: {analysis_metadata['ess_threshold']}")
    summary.append(f"敏感度閾值: {analysis_metadata['sensitivity_threshold']:.1%}")
    summary.append("")
    
    # MCMC Parameters Section
    summary.append("⛓️ MCMC 參數")
    summary.append("-" * 40)
    summary.append(f"MCMC 鏈數: {analysis_metadata['mcmc_chains']}")
    summary.append(f"暖身期樣本: {analysis_metadata['mcmc_warmup']:,}")
    summary.append(f"採樣數: {analysis_metadata['mcmc_samples']:,}")
    summary.append("")
    
    # Skill Score Metrics Section
    summary.append("📈 技能評分指標")
    summary.append("-" * 40)
    skill_metrics = analysis_metadata['skill_score_metrics']
    summary.append(f"評分指標: {', '.join(skill_metrics)}")
    summary.append(f"交叉驗證摺數: {analysis_metadata['cross_validation_folds']}")
    confidence_levels = [f"{level:.0%}" for level in analysis_metadata['confidence_levels']]
    summary.append(f"信賴水準: {', '.join(confidence_levels)}")
    summary.append("")
    
    # Prior Scenarios Section
    summary.append("🔄 先驗場景")
    summary.append("-" * 40)
    prior_scenarios = analysis_metadata['prior_scenarios']
    for i, scenario in enumerate(prior_scenarios, 1):
        summary.append(f"{i}. {scenario}")
    summary.append("")
    
    # Framework Information Section
    summary.append("ℹ️ 分析框架資訊")
    summary.append("-" * 40)
    summary.append(f"框架: {analysis_metadata['framework']}")
    summary.append(f"版本: {analysis_metadata['analysis_version']}")
    summary.append(f"分析時間: {analysis_metadata['analysis_timestamp']}")
    summary.append("")
    
    return "\n".join(summary)


class EnhancedBayesianReportGenerator(BayesianReportGenerator):
    """
    Enhanced Bayesian Report Generator with additional statistical metrics
    增強的貝氏報告生成器，包含額外的統計指標
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _generate_enhanced_executive_summary(self, 
                                           mcmc_diagnostics, 
                                           model_comparison, 
                                           posterior_analysis, 
                                           robustness_analysis,
                                           analysis_metadata: Dict[str, Any]) -> str:
        """Generate enhanced executive summary with all statistical parameters"""
        
        summary = []
        summary.append('<div class="section">')
        summary.append('<h2>🎯 增強執行摘要</h2>')
        
        # Statistical Parameters Section
        summary.append('<div class="subsection">')
        summary.append('<h3>核心統計參數</h3>')
        
        # Create metrics display
        metrics = [
            ("風速不確定性 CV", f"{analysis_metadata.get('wind_uncertainty_cv', 0):.1%}"),
            ("曝險不確定性 log-σ", f"{analysis_metadata.get('exposure_uncertainty_log_std', 0):.2f}"),
            ("脆弱性參數不確定性", f"{analysis_metadata.get('vulnerability_param_uncertainty', 0):.1%}"),
            ("Monte Carlo樣本數", f"{analysis_metadata.get('n_monte_carlo', 0):,}"),
            ("最佳分布模型", analysis_metadata.get('best_distribution', 'Unknown')),
            ("MCMC收斂狀態", analysis_metadata.get('mcmc_convergence', 'Unknown')),
            ("混合組件數", str(analysis_metadata.get('n_mixture_components', 0))),
            ("密度比約束 (γ)", f"{analysis_metadata.get('density_ratio_constraint', 0):.1f}")
        ]
        
        for name, value in metrics:
            summary.append(f'<div class="metric"><strong>{value}</strong><br>{name}</div>')
        
        summary.append('</div>')
        
        # Model Performance Section
        summary.append('<div class="subsection">')
        summary.append('<h3>模型性能指標</h3>')
        
        if robustness_analysis:
            robustness_score = robustness_analysis.robustness_score
            robustness_class = "success" if robustness_score >= 80 else "warning" if robustness_score >= 60 else "error"
            summary.append(f'<div class="{robustness_class}">穩健性評分: {robustness_score:.1f}/100</div>')
        
        if mcmc_diagnostics and mcmc_diagnostics.summary_statistics:
            convergence_rate = mcmc_diagnostics.summary_statistics.get('convergence_rate', 0) * 100
            mean_rhat = mcmc_diagnostics.summary_statistics.get('mean_rhat', 0)
            min_ess = mcmc_diagnostics.summary_statistics.get('min_ess', 0)
            
            summary.append(f'<div class="success">MCMC 收斂率: {convergence_rate:.1f}%</div>')
            summary.append(f'<div class="metric">平均 R̂: {mean_rhat:.3f}</div>')
            summary.append(f'<div class="metric">最小 ESS: {min_ess:.0f}</div>')
        
        summary.append('</div>')
        summary.append('</div>')
        
        return '\n'.join(summary)
    
    def generate_enhanced_comprehensive_report(self, 
                                             posterior_samples,
                                             model_results,
                                             observed_data,
                                             analysis_metadata: Dict[str, Any],
                                             **kwargs) -> str:
        """Generate comprehensive report with enhanced statistical metrics"""
        
        print("📊 生成增強型綜合貝氏分析報告...")
        
        # Run standard analysis first
        report_path = self.generate_comprehensive_report(
            posterior_samples, model_results, observed_data, **kwargs
        )
        
        # Add enhanced statistical summary
        enhanced_summary = generate_enhanced_statistical_summary(analysis_metadata)
        
        # Save enhanced summary as separate file
        enhanced_summary_path = report_path.replace('.html', '_enhanced_summary.txt')
        with open(enhanced_summary_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_summary)
        
        print(f"✅ 增強統計摘要已生成: {enhanced_summary_path}")
        
        return report_path, enhanced_summary_path


def demo_enhanced_reporting():
    """Demonstrate enhanced reporting capabilities"""
    
    print("🚀 增強貝氏報告功能演示")
    print("=" * 60)
    
    # Create enhanced analysis metadata
    enhanced_metadata = create_enhanced_analysis_metadata(
        wind_uncertainty_cv=0.15,
        exposure_uncertainty_log_std=0.20,
        vulnerability_param_uncertainty=0.10,
        n_monte_carlo=500,
        best_distribution="Normal",
        mcmc_convergence="converged",
        n_mixture_components=3,
        density_ratio_constraint=2.0
    )
    
    print("✓ 增強分析元數據已創建")
    print(f"  包含 {len(enhanced_metadata)} 個統計參數")
    
    # Generate enhanced statistical summary
    enhanced_summary = generate_enhanced_statistical_summary(enhanced_metadata)
    print("\n✓ 增強統計摘要:")
    print(enhanced_summary)
    
    # Demonstrate all required metrics are covered
    required_metrics = [
        '風速不確定性 CV',
        '曝險不確定性 log-σ', 
        '脆弱性參數不確定性',
        'Monte Carlo樣本數',
        '最佳分布模型',
        'MCMC收斂狀態',
        '穩健性評分',
        '密度比約束',
        '混合組件數'
    ]
    
    print("\n📊 必要統計指標覆蓋檢查:")
    for metric in required_metrics:
        if metric in enhanced_summary:
            print(f"  ✓ {metric}")
        else:
            print(f"  ❌ {metric}")
    
    print("\n💡 建議改進:")
    print("1. 將 analysis_metadata 參數添加到所有報告生成方法")
    print("2. 在 HTML 模板中添加統計參數專用部分")
    print("3. 創建統計參數驗證機制")
    print("4. 支持統計參數的動態更新和配置")
    
    return enhanced_metadata, enhanced_summary


if __name__ == "__main__":
    demo_enhanced_reporting()