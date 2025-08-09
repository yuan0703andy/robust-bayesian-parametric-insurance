"""
Bayesian Reports Module
貝氏報告模組

提供全面的貝氏統計分析報告生成功能。
包含 MCMC 診斷、模型比較、後驗分析、穩健性評估等完整報告。

Modules:
- bayesian_diagnostics: MCMC 收斂診斷和模型檢驗
- robustness_reporter: 穩健性分析詳細報告
- model_comparison_reporter: 全面的模型比較和選擇報告
- posterior_analysis_reporter: 後驗分布深度分析
- bayesian_report_generator: 主報告生成器，整合所有報告模組

Quick Start:
```python
from bayesian.reports import BayesianReportGenerator, ReportConfig

# 創建報告生成器
config = ReportConfig(title="我的貝氏分析報告", include_plots=True)
generator = BayesianReportGenerator(config=config)

# 生成綜合報告
report_path = generator.generate_comprehensive_report(
    posterior_samples=posterior_samples,
    model_results=model_results,
    observed_data=observed_data
)
```
"""

# Core report components
from .bayesian_diagnostics import (
    BayesianDiagnostics,
    MCMCDiagnostics,
    ConvergenceStatus,
    DiagnosticResult
)

from .robustness_reporter import (
    RobustnessReporter,
    RobustnessReport,
    RobustnessLevel,
    DensityRatioAnalysis,
    SensitivityAnalysis
)

from .model_comparison_reporter import (
    ModelComparisonReporter,
    ModelComparisonReport,
    ModelSupport,
    ModelComparison,
    ModelAveraging
)

from .posterior_analysis_reporter import (
    PosteriorAnalysisReporter,
    PosteriorAnalysisReport,
    DistributionType,
    LearningEffectiveness,
    PriorPosteriorComparison,
    ParameterCorrelation,
    PosteriorPredictiveCheck
)

from .bayesian_report_generator import (
    BayesianReportGenerator,
    ReportConfig,
    ReportFormat,
    ReportSection,
    ComprehensiveAnalysisResult
)

# Public API
__all__ = [
    # === Main Report Generator (Recommended Entry Point) ===
    'BayesianReportGenerator',           # Primary report generator
    'ReportConfig',                      # Configuration class
    'ReportFormat',                      # Output format options
    'ReportSection',                     # Report section types
    'ComprehensiveAnalysisResult',       # Combined analysis results
    
    # === Individual Report Components ===
    # MCMC Diagnostics
    'BayesianDiagnostics',               # MCMC convergence diagnostics
    'MCMCDiagnostics',                   # Diagnostic results
    'ConvergenceStatus',                 # Convergence status enum
    'DiagnosticResult',                  # Individual parameter diagnostics
    
    # Robustness Analysis
    'RobustnessReporter',                # Robustness analysis reporter
    'RobustnessReport',                  # Robustness analysis results
    'RobustnessLevel',                   # Robustness level enum
    'DensityRatioAnalysis',              # Density ratio constraint analysis
    'SensitivityAnalysis',               # Prior sensitivity analysis
    
    # Model Comparison
    'ModelComparisonReporter',           # Model comparison reporter
    'ModelComparisonReport',             # Model comparison results
    'ModelSupport',                      # Bayes factor support levels
    'ModelComparison',                   # Individual model comparison
    'ModelAveraging',                    # Model averaging results
    
    # Posterior Analysis
    'PosteriorAnalysisReporter',         # Posterior distribution analyzer
    'PosteriorAnalysisReport',           # Posterior analysis results
    'DistributionType',                  # Distribution classification
    'LearningEffectiveness',             # Learning effect levels
    'PriorPosteriorComparison',          # Prior vs posterior comparison
    'ParameterCorrelation',              # Parameter correlation analysis
    'PosteriorPredictiveCheck',          # Posterior predictive checks
]

__version__ = "1.0.0"
__author__ = "Bayesian Analysis Reporting Team"

# Module information
def get_reports_info():
    """Get information about the reports module"""
    return {
        'version': __version__,
        'components': [
            'bayesian_diagnostics.py - MCMC 收斂診斷和模型檢驗',
            'robustness_reporter.py - 穩健性分析詳細報告',
            'model_comparison_reporter.py - 全面的模型比較和選擇報告', 
            'posterior_analysis_reporter.py - 後驗分布深度分析',
            'bayesian_report_generator.py - 主報告生成器，整合所有報告模組'
        ],
        'features': [
            'MCMC 收斂診斷 (R-hat, ESS, MCSE)',
            '密度比約束分析',
            '先驗敏感度分析',
            'Bayes Factor 模型比較',
            '後驗預測檢驗',
            '視覺化圖表生成',
            'HTML/Text/JSON 報告輸出'
        ]
    }

def get_usage_guide():
    """Get usage guide for the reports module"""
    return """
    🚀 貝氏報告模組使用指南
    
    === 基本使用 (推薦) ===
    ```python
    from bayesian.reports import BayesianReportGenerator, ReportConfig
    
    # 設定報告配置
    config = ReportConfig(
        title="熱帶氣旋風險貝氏分析報告",
        author="風險分析師",
        include_plots=True,
        output_format=ReportFormat.HTML
    )
    
    # 創建報告生成器
    generator = BayesianReportGenerator(
        output_directory="reports",
        report_config=config
    )
    
    # 生成綜合報告
    report_path = generator.generate_comprehensive_report(
        posterior_samples=posterior_samples,      # 後驗樣本字典
        model_results=model_results,              # ModelComparisonResult 列表
        observed_data=observed_data,              # 觀測資料
        hierarchical_result=hierarchical_result, # 階層模型結果 (可選)
        prior_specifications=prior_specs,         # 先驗規格 (可選)
        chains=mcmc_chains                        # MCMC 鏈 (可選)
    )
    
    print(f"報告已生成: {report_path}")
    ```
    
    === 個別組件使用 ===
    ```python
    from bayesian.reports import (
        BayesianDiagnostics,
        RobustnessReporter, 
        ModelComparisonReporter,
        PosteriorAnalysisReporter
    )
    
    # 1. MCMC 診斷
    diagnostics = BayesianDiagnostics()
    mcmc_result = diagnostics.diagnose_mcmc_convergence(posterior_samples)
    
    # 2. 穩健性分析
    robustness = RobustnessReporter()
    robustness_result = robustness.analyze_robustness(
        posterior_samples, observed_data, model_results
    )
    
    # 3. 模型比較
    comparison = ModelComparisonReporter()
    comparison_result = comparison.compare_models(
        model_results, observed_data, model_posterior_samples
    )
    
    # 4. 後驗分析
    posterior = PosteriorAnalysisReporter()
    posterior_result = posterior.analyze_posterior_distributions(
        posterior_samples, prior_specifications, observed_data
    )
    ```
    
    === 報告配置選項 ===
    ```python
    from bayesian.reports import ReportConfig, ReportFormat, ReportSection
    
    config = ReportConfig(
        title="自定義報告標題",
        author="分析師姓名",
        include_plots=True,                    # 是否包含圖表
        include_details=True,                  # 是否包含詳細資訊
        output_format=ReportFormat.HTML,       # HTML/TEXT/JSON
        sections=[                             # 選擇報告章節
            ReportSection.EXECUTIVE_SUMMARY,
            ReportSection.MCMC_DIAGNOSTICS,
            ReportSection.MODEL_COMPARISON,
            ReportSection.POSTERIOR_ANALYSIS,
            ReportSection.ROBUSTNESS_ANALYSIS,
            ReportSection.RECOMMENDATIONS
        ]
    )
    ```
    
    === 關鍵特點 ===
    • 🔍 全面的 MCMC 收斂診斷
    • 🛡️ 穩健性和敏感度分析  
    • 📊 多模型比較和選擇
    • 📈 後驗分布深度分析
    • 🎨 自動圖表生成
    • 📄 多格式報告輸出
    • 🔧 模組化設計，可單獨使用
    """

# Default configuration for quick start
DEFAULT_REPORT_CONFIG = ReportConfig(
    title="貝氏分析報告",
    author="Bayesian Analysis System",
    include_plots=True,
    include_details=True,
    output_format=ReportFormat.HTML
)

def create_quick_report(posterior_samples: dict, 
                       model_results: list,
                       observed_data, 
                       output_dir: str = "reports",
                       title: str = "快速貝氏分析報告") -> str:
    """
    快速生成貝氏分析報告的便利函數
    
    Parameters:
    -----------
    posterior_samples : dict
        後驗樣本字典
    model_results : list
        模型比較結果列表
    observed_data : array-like
        觀測資料
    output_dir : str
        輸出目錄
    title : str
        報告標題
        
    Returns:
    --------
    str
        報告檔案路徑
    """
    
    config = ReportConfig(title=title, include_plots=True)
    generator = BayesianReportGenerator(output_directory=output_dir, report_config=config)
    
    return generator.generate_comprehensive_report(
        posterior_samples=posterior_samples,
        model_results=model_results,
        observed_data=observed_data
    )

# Add quick report function to exports
__all__.append('create_quick_report')
__all__.append('DEFAULT_REPORT_CONFIG')
__all__.append('get_reports_info')
__all__.append('get_usage_guide')