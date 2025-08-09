"""
Bayesian Report Generator Module
貝氏報告生成器模組

整合所有報告模組，提供統一的報告生成介面。
生成綜合的貝氏分析報告，包括 HTML 和 PDF 輸出。

Key Features:
- 統一報告生成介面
- 整合所有分析模組
- HTML 報告輸出
- 圖表整合和管理
- 報告模板系統
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import warnings
from datetime import datetime
import json
import sys
import os

# Import parent bayesian modules - use relative imports
try:
    from ..robust_bayesian_analysis import ModelComparisonResult, RobustBayesianFramework
    from ..hierarchical_bayesian_model import HierarchicalModelResult
    from ..robust_bayesian_analyzer import RobustBayesianAnalyzer
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from robust_bayesian_analysis import ModelComparisonResult, RobustBayesianFramework
    from hierarchical_bayesian_model import HierarchicalModelResult
    from robust_bayesian_analyzer import RobustBayesianAnalyzer

# Import report modules
from .bayesian_diagnostics import BayesianDiagnostics, MCMCDiagnostics
from .robustness_reporter import RobustnessReporter, RobustnessReport
from .model_comparison_reporter import ModelComparisonReporter, ModelComparisonReport
from .posterior_analysis_reporter import PosteriorAnalysisReporter, PosteriorAnalysisReport

class ReportFormat(Enum):
    """報告格式"""
    HTML = "html"
    TEXT = "text"
    JSON = "json"

class ReportSection(Enum):
    """報告章節"""
    EXECUTIVE_SUMMARY = "executive_summary"
    MCMC_DIAGNOSTICS = "mcmc_diagnostics"
    MODEL_COMPARISON = "model_comparison"
    POSTERIOR_ANALYSIS = "posterior_analysis"
    ROBUSTNESS_ANALYSIS = "robustness_analysis"
    RECOMMENDATIONS = "recommendations"
    APPENDIX = "appendix"

@dataclass
class ReportConfig:
    """報告配置"""
    title: str = "貝氏分析報告"
    author: str = "Bayesian Analysis System"
    include_plots: bool = True
    include_details: bool = True
    output_format: ReportFormat = ReportFormat.HTML
    sections: List[ReportSection] = None
    custom_css: Optional[str] = None
    
    def __post_init__(self):
        if self.sections is None:
            self.sections = list(ReportSection)

@dataclass
class ComprehensiveAnalysisResult:
    """綜合分析結果"""
    mcmc_diagnostics: Optional[MCMCDiagnostics] = None
    model_comparison: Optional[ModelComparisonReport] = None
    posterior_analysis: Optional[PosteriorAnalysisReport] = None
    robustness_analysis: Optional[RobustnessReport] = None
    analysis_metadata: Dict[str, Any] = None

class BayesianReportGenerator:
    """
    貝氏報告生成器
    
    整合所有分析模組，生成統一的綜合報告
    """
    
    def __init__(self, 
                 output_directory: str = "reports",
                 report_config: Optional[ReportConfig] = None):
        """
        初始化報告生成器
        
        Parameters:
        -----------
        output_directory : str
            輸出目錄
        report_config : ReportConfig, optional
            報告配置
        """
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        self.config = report_config or ReportConfig()
        
        # 初始化各報告模組
        self.diagnostics_reporter = BayesianDiagnostics()
        self.robustness_reporter = RobustnessReporter()
        self.model_comparison_reporter = ModelComparisonReporter()
        self.posterior_analysis_reporter = PosteriorAnalysisReporter()
        
        # 存儲生成的圖表路徑
        self.generated_plots = {}
        
        # 設置樣式
        plt.style.use('default')
        sns.set_style("whitegrid")
    
    def generate_comprehensive_report(self, 
                                    posterior_samples: Dict[str, np.ndarray],
                                    model_results: List[ModelComparisonResult],
                                    observed_data: np.ndarray,
                                    hierarchical_result: Optional[HierarchicalModelResult] = None,
                                    prior_specifications: Optional[Dict[str, Dict]] = None,
                                    chains: Optional[List[np.ndarray]] = None,
                                    save_path: Optional[str] = None) -> str:
        """
        生成綜合貝氏分析報告
        
        Parameters:
        -----------
        posterior_samples : Dict[str, np.ndarray]
            後驗樣本
        model_results : List[ModelComparisonResult]
            模型比較結果
        observed_data : np.ndarray
            觀測資料
        hierarchical_result : HierarchicalModelResult, optional
            階層模型結果
        prior_specifications : Dict[str, Dict], optional
            先驗規格
        chains : List[np.ndarray], optional
            MCMC 鏈
        save_path : str, optional
            保存路徑
            
        Returns:
        --------
        str
            報告文件路徑
        """
        
        print("📊 開始生成綜合貝氏分析報告...")
        
        # 執行所有分析
        analysis_results = self._perform_comprehensive_analysis(
            posterior_samples, model_results, observed_data,
            hierarchical_result, prior_specifications, chains
        )
        
        # 生成圖表
        if self.config.include_plots:
            print("  🎨 生成分析圖表...")
            self._generate_all_plots(analysis_results, posterior_samples, model_results)
        
        # 生成報告
        print("  📝 生成報告文件...")
        report_content = self._create_report_content(analysis_results)
        
        # 保存報告
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"bayesian_analysis_report_{timestamp}.{self.config.output_format.value}"
        else:
            save_path = Path(save_path)
        
        self._save_report(report_content, save_path)
        
        print(f"✅ 報告已生成: {save_path}")
        return str(save_path)
    
    def _perform_comprehensive_analysis(self, 
                                      posterior_samples: Dict[str, np.ndarray],
                                      model_results: List[ModelComparisonResult],
                                      observed_data: np.ndarray,
                                      hierarchical_result: Optional[HierarchicalModelResult],
                                      prior_specifications: Optional[Dict[str, Dict]],
                                      chains: Optional[List[np.ndarray]]) -> ComprehensiveAnalysisResult:
        """執行綜合分析"""
        
        print("  🔍 執行 MCMC 診斷...")
        mcmc_diagnostics = self.diagnostics_reporter.diagnose_mcmc_convergence(
            posterior_samples, chains
        )
        
        print("  📊 執行模型比較...")
        # 為模型比較準備後驗樣本字典
        model_posterior_samples = {}
        for result in model_results:
            model_posterior_samples[result.model_name] = posterior_samples
        
        model_comparison = self.model_comparison_reporter.compare_models(
            model_results, observed_data, model_posterior_samples
        )
        
        print("  📈 執行後驗分析...")
        posterior_analysis = self.posterior_analysis_reporter.analyze_posterior_distributions(
            posterior_samples, prior_specifications, observed_data, hierarchical_result
        )
        
        print("  🛡️ 執行穩健性分析...")
        robustness_analysis = self.robustness_reporter.analyze_robustness(
            posterior_samples, observed_data, model_results
        )
        
        # 創建元資料
        analysis_metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'n_parameters': len([s for s in posterior_samples.values() if s.ndim == 1]),
            'n_models': len(model_results),
            'n_observations': len(observed_data),
            'has_hierarchical_result': hierarchical_result is not None,
            'has_prior_specs': prior_specifications is not None,
            'has_chains': chains is not None
        }
        
        return ComprehensiveAnalysisResult(
            mcmc_diagnostics=mcmc_diagnostics,
            model_comparison=model_comparison,
            posterior_analysis=posterior_analysis,
            robustness_analysis=robustness_analysis,
            analysis_metadata=analysis_metadata
        )
    
    def _generate_all_plots(self, 
                          analysis_results: ComprehensiveAnalysisResult,
                          posterior_samples: Dict[str, np.ndarray],
                          model_results: List[ModelComparisonResult]):
        """生成所有分析圖表"""
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. MCMC 診斷圖
        if analysis_results.mcmc_diagnostics:
            try:
                fig = self.diagnostics_reporter.plot_convergence_diagnostics(
                    analysis_results.mcmc_diagnostics, posterior_samples
                )
                if fig:
                    plot_path = plots_dir / "mcmc_diagnostics.png"
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    self.generated_plots['mcmc_diagnostics'] = plot_path
                    plt.close(fig)
            except Exception as e:
                print(f"Warning: MCMC 診斷圖生成失敗: {e}")
        
        # 2. 模型比較圖
        if analysis_results.model_comparison:
            try:
                fig = self.model_comparison_reporter.plot_model_comparison(
                    analysis_results.model_comparison
                )
                if fig:
                    plot_path = plots_dir / "model_comparison.png"
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    self.generated_plots['model_comparison'] = plot_path
                    plt.close(fig)
            except Exception as e:
                print(f"Warning: 模型比較圖生成失敗: {e}")
        
        # 3. 後驗分析圖
        if analysis_results.posterior_analysis:
            try:
                fig = self.posterior_analysis_reporter.plot_posterior_analysis(
                    analysis_results.posterior_analysis, posterior_samples
                )
                if fig:
                    plot_path = plots_dir / "posterior_analysis.png"
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    self.generated_plots['posterior_analysis'] = plot_path
                    plt.close(fig)
            except Exception as e:
                print(f"Warning: 後驗分析圖生成失敗: {e}")
        
        # 4. 穩健性分析圖
        if analysis_results.robustness_analysis:
            try:
                fig = self.robustness_reporter.plot_robustness_analysis(
                    analysis_results.robustness_analysis
                )
                if fig:
                    plot_path = plots_dir / "robustness_analysis.png"
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    self.generated_plots['robustness_analysis'] = plot_path
                    plt.close(fig)
            except Exception as e:
                print(f"Warning: 穩健性分析圖生成失敗: {e}")
    
    def _create_report_content(self, 
                             analysis_results: ComprehensiveAnalysisResult) -> str:
        """創建報告內容"""
        
        if self.config.output_format == ReportFormat.HTML:
            return self._create_html_report(analysis_results)
        elif self.config.output_format == ReportFormat.JSON:
            return self._create_json_report(analysis_results)
        else:
            return self._create_text_report(analysis_results)
    
    def _create_html_report(self, analysis_results: ComprehensiveAnalysisResult) -> str:
        """創建 HTML 報告"""
        
        html_parts = []
        
        # HTML 頭部
        html_parts.append(self._get_html_header())
        
        # 報告標題和摘要
        html_parts.append(self._create_html_title_section())
        
        # 執行摘要
        if ReportSection.EXECUTIVE_SUMMARY in self.config.sections:
            html_parts.append(self._create_html_executive_summary(analysis_results))
        
        # MCMC 診斷
        if ReportSection.MCMC_DIAGNOSTICS in self.config.sections and analysis_results.mcmc_diagnostics:
            html_parts.append(self._create_html_mcmc_section(analysis_results.mcmc_diagnostics))
        
        # 模型比較
        if ReportSection.MODEL_COMPARISON in self.config.sections and analysis_results.model_comparison:
            html_parts.append(self._create_html_model_comparison_section(analysis_results.model_comparison))
        
        # 後驗分析
        if ReportSection.POSTERIOR_ANALYSIS in self.config.sections and analysis_results.posterior_analysis:
            html_parts.append(self._create_html_posterior_section(analysis_results.posterior_analysis))
        
        # 穩健性分析
        if ReportSection.ROBUSTNESS_ANALYSIS in self.config.sections and analysis_results.robustness_analysis:
            html_parts.append(self._create_html_robustness_section(analysis_results.robustness_analysis))
        
        # 建議
        if ReportSection.RECOMMENDATIONS in self.config.sections:
            html_parts.append(self._create_html_recommendations_section(analysis_results))
        
        # HTML 尾部
        html_parts.append(self._get_html_footer())
        
        return "\\n".join(html_parts)
    
    def _get_html_header(self) -> str:
        """HTML 頭部"""
        
        default_css = """
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; line-height: 1.6; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
            .section { margin: 30px 0; padding: 20px; border-left: 4px solid #667eea; background-color: #f8f9fa; }
            .subsection { margin: 20px 0; padding: 15px; background-color: white; border-radius: 5px; }
            .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #e9ecef; border-radius: 5px; min-width: 120px; text-align: center; }
            .plot { text-align: center; margin: 20px 0; }
            .plot img { max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .success { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; margin: 10px 0; }
            table { width: 100%; border-collapse: collapse; margin: 15px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; font-weight: bold; }
            .recommendation { background-color: #e3f2fd; border-left: 4px solid #2196f3; padding: 15px; margin: 10px 0; }
        </style>
        """
        
        css = self.config.custom_css or default_css
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{self.config.title}</title>
            {css}
        </head>
        <body>
        """
    
    def _get_html_footer(self) -> str:
        """HTML 尾部"""
        
        timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        
        return f"""
        <div class="section">
            <h2>📋 報告資訊</h2>
            <p><strong>生成時間:</strong> {timestamp}</p>
            <p><strong>作者:</strong> {self.config.author}</p>
            <p><strong>系統:</strong> Bayesian Analysis Reporting System v2.0</p>
        </div>
        </body>
        </html>
        """
    
    def _create_html_title_section(self) -> str:
        """創建標題區段"""
        
        return f"""
        <div class="header">
            <h1>📊 {self.config.title}</h1>
            <p>完整的貝氏統計分析報告，包含模型診斷、比較、穩健性評估及建議</p>
        </div>
        """
    
    def _create_html_executive_summary(self, analysis_results: ComprehensiveAnalysisResult) -> str:
        """創建執行摘要"""
        
        html = """
        <div class="section">
            <h2>🎯 執行摘要</h2>
        """
        
        metadata = analysis_results.analysis_metadata
        
        # 基本統計
        html += f"""
        <div class="subsection">
            <h3>分析概覽</h3>
            <div class="metric">
                <strong>{metadata.get('n_parameters', 0)}</strong><br>
                分析參數
            </div>
            <div class="metric">
                <strong>{metadata.get('n_models', 0)}</strong><br>
                比較模型
            </div>
            <div class="metric">
                <strong>{metadata.get('n_observations', 0)}</strong><br>
                觀測資料點
            </div>
        </div>
        """
        
        # 主要發現
        key_findings = self._extract_key_findings(analysis_results)
        
        html += """
        <div class="subsection">
            <h3>主要發現</h3>
        """
        
        for finding in key_findings:
            finding_class = "success" if "✅" in finding else "warning" if "⚠️" in finding else "error" if "❌" in finding else ""
            html += f'<div class="{finding_class}">{finding}</div>'
        
        html += """
        </div>
        </div>
        """
        
        return html
    
    def _extract_key_findings(self, analysis_results: ComprehensiveAnalysisResult) -> List[str]:
        """提取主要發現"""
        
        findings = []
        
        # MCMC 收斂
        if analysis_results.mcmc_diagnostics:
            status = analysis_results.mcmc_diagnostics.overall_convergence
            if status.value in ['excellent', 'good']:
                findings.append("✅ MCMC 收斂狀況良好")
            elif status.value == 'acceptable':
                findings.append("⚠️ MCMC 收斂可接受")
            else:
                findings.append("❌ MCMC 收斂問題需要注意")
        
        # 模型選擇
        if analysis_results.model_comparison and analysis_results.model_comparison.best_model:
            best_model = analysis_results.model_comparison.best_model
            findings.append(f"🏆 推薦模型: {best_model.model_name}")
            
            if best_model.model_weight > 0.7:
                findings.append("✅ 模型選擇確信度高")
            else:
                findings.append("⚠️ 建議考慮模型平均")
        
        # 穩健性
        if analysis_results.robustness_analysis:
            robustness = analysis_results.robustness_analysis.overall_robustness
            score = analysis_results.robustness_analysis.robustness_score
            
            if score > 75:
                findings.append(f"✅ 模型穩健性良好 (評分: {score:.0f}/100)")
            elif score > 50:
                findings.append(f"⚠️ 模型穩健性中等 (評分: {score:.0f}/100)")
            else:
                findings.append(f"❌ 模型穩健性需要改善 (評分: {score:.0f}/100)")
        
        return findings
    
    def _create_html_mcmc_section(self, mcmc_diagnostics: MCMCDiagnostics) -> str:
        """創建 MCMC 診斷區段"""
        
        html = """
        <div class="section">
            <h2>🔍 MCMC 收斂診斷</h2>
        """
        
        # 診斷圖表
        if 'mcmc_diagnostics' in self.generated_plots:
            html += f"""
            <div class="plot">
                <img src="{self.generated_plots['mcmc_diagnostics']}" alt="MCMC 診斷圖">
            </div>
            """
        
        # 摘要統計
        summary = mcmc_diagnostics.summary_statistics
        html += f"""
        <div class="subsection">
            <h3>收斂摘要</h3>
            <div class="metric">
                <strong>{summary.get('convergence_rate', 0):.1%}</strong><br>
                收斂率
            </div>
            <div class="metric">
                <strong>{summary.get('mean_rhat', 0):.3f}</strong><br>
                平均 R̂
            </div>
            <div class="metric">
                <strong>{summary.get('min_ess', 0):.0f}</strong><br>
                最小 ESS
            </div>
        </div>
        """
        
        # 建議
        html += """
        <div class="subsection">
            <h3>診斷建議</h3>
        """
        
        for recommendation in mcmc_diagnostics.recommendations:
            html += f"<div class='recommendation'>{recommendation}</div>"
        
        html += """
        </div>
        </div>
        """
        
        return html
    
    def _create_html_model_comparison_section(self, model_comparison: ModelComparisonReport) -> str:
        """創建模型比較區段"""
        
        html = """
        <div class="section">
            <h2>📊 模型比較分析</h2>
        """
        
        # 比較圖表
        if 'model_comparison' in self.generated_plots:
            html += f"""
            <div class="plot">
                <img src="{self.generated_plots['model_comparison']}" alt="模型比較圖">
            </div>
            """
        
        # 比較表格
        if not model_comparison.comparison_summary.empty:
            html += """
            <div class="subsection">
                <h3>模型比較表</h3>
                <table>
                    <tr>
                        <th>排名</th>
                        <th>模型</th>
                        <th>AIC</th>
                        <th>BIC</th>
                        <th>權重</th>
                        <th>支持度</th>
                    </tr>
            """
            
            for _, row in model_comparison.comparison_summary.iterrows():
                rank_icon = "🥇" if row['Rank'] == 1 else "🥈" if row['Rank'] == 2 else "🥉" if row['Rank'] == 3 else ""
                html += f"""
                <tr>
                    <td>{rank_icon} {row['Rank']}</td>
                    <td>{row['Model']}</td>
                    <td>{row['AIC']:.2f}</td>
                    <td>{row['BIC']:.2f}</td>
                    <td>{row['Weight']:.3f}</td>
                    <td>{row['Support']}</td>
                </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _create_html_posterior_section(self, posterior_analysis: PosteriorAnalysisReport) -> str:
        """創建後驗分析區段"""
        
        html = """
        <div class="section">
            <h2>📈 後驗分布分析</h2>
        """
        
        # 後驗分析圖表
        if 'posterior_analysis' in self.generated_plots:
            html += f"""
            <div class="plot">
                <img src="{self.generated_plots['posterior_analysis']}" alt="後驗分析圖">
            </div>
            """
        
        # 學習效果統計
        summary = posterior_analysis.summary_statistics
        if 'learning_effects' in summary:
            effects = summary['learning_effects']
            html += f"""
            <div class="subsection">
                <h3>學習效果統計</h3>
                <div class="metric">
                    <strong>{effects.get('dramatic', 0)}</strong><br>
                    顯著學習
                </div>
                <div class="metric">
                    <strong>{effects.get('substantial', 0)}</strong><br>
                    實質學習
                </div>
                <div class="metric">
                    <strong>{effects.get('moderate', 0)}</strong><br>
                    中等學習
                </div>
                <div class="metric">
                    <strong>{effects.get('minimal', 0)}</strong><br>
                    微弱學習
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _create_html_robustness_section(self, robustness_analysis: RobustnessReport) -> str:
        """創建穩健性分析區段"""
        
        html = """
        <div class="section">
            <h2>🛡️ 穩健性分析</h2>
        """
        
        # 穩健性圖表
        if 'robustness_analysis' in self.generated_plots:
            html += f"""
            <div class="plot">
                <img src="{self.generated_plots['robustness_analysis']}" alt="穩健性分析圖">
            </div>
            """
        
        # 穩健性評分
        score = robustness_analysis.robustness_score
        score_class = "success" if score > 75 else "warning" if score > 50 else "error"
        
        html += f"""
        <div class="subsection">
            <h3>穩健性評分</h3>
            <div class="{score_class}">
                <h2 style="margin: 0; text-align: center;">{score:.1f} / 100</h2>
                <p style="margin: 5px 0; text-align: center;">整體穩健性評分</p>
            </div>
        </div>
        """
        
        html += "</div>"
        return html
    
    def _create_html_recommendations_section(self, analysis_results: ComprehensiveAnalysisResult) -> str:
        """創建建議區段"""
        
        html = """
        <div class="section">
            <h2>💡 綜合建議</h2>
        """
        
        # 收集所有建議
        all_recommendations = []
        
        if analysis_results.mcmc_diagnostics:
            all_recommendations.extend(analysis_results.mcmc_diagnostics.recommendations)
        
        if analysis_results.model_comparison:
            all_recommendations.extend(analysis_results.model_comparison.recommendations)
        
        if analysis_results.posterior_analysis:
            all_recommendations.extend(analysis_results.posterior_analysis.recommendations)
        
        if analysis_results.robustness_analysis:
            all_recommendations.extend(analysis_results.robustness_analysis.recommendations)
        
        # 分類建議
        urgent_recommendations = [rec for rec in all_recommendations if "❌" in rec or "🚨" in rec]
        important_recommendations = [rec for rec in all_recommendations if "⚠️" in rec]
        general_recommendations = [rec for rec in all_recommendations if "✅" in rec or "💡" in rec]
        
        if urgent_recommendations:
            html += """
            <div class="subsection">
                <h3>🚨 緊急建議</h3>
            """
            for rec in urgent_recommendations:
                html += f'<div class="error">{rec}</div>'
            html += "</div>"
        
        if important_recommendations:
            html += """
            <div class="subsection">
                <h3>⚠️ 重要建議</h3>
            """
            for rec in important_recommendations:
                html += f'<div class="warning">{rec}</div>'
            html += "</div>"
        
        if general_recommendations:
            html += """
            <div class="subsection">
                <h3>💡 一般建議</h3>
            """
            for rec in general_recommendations:
                html += f'<div class="recommendation">{rec}</div>'
            html += "</div>"
        
        html += "</div>"
        return html
    
    def _create_text_report(self, analysis_results: ComprehensiveAnalysisResult) -> str:
        """創建文字報告"""
        
        report_parts = []
        
        # 標題
        report_parts.append("=" * 80)
        report_parts.append(f"                {self.config.title}")
        report_parts.append("=" * 80)
        report_parts.append("")
        
        # 各區段
        if analysis_results.mcmc_diagnostics:
            report_parts.append(self.diagnostics_reporter.generate_diagnostic_report(
                analysis_results.mcmc_diagnostics, self.config.include_details
            ))
            report_parts.append("")
        
        if analysis_results.model_comparison:
            report_parts.append(self.model_comparison_reporter.generate_comparison_report(
                analysis_results.model_comparison, self.config.include_details
            ))
            report_parts.append("")
        
        if analysis_results.posterior_analysis:
            report_parts.append(self.posterior_analysis_reporter.generate_posterior_report(
                analysis_results.posterior_analysis, self.config.include_details
            ))
            report_parts.append("")
        
        if analysis_results.robustness_analysis:
            report_parts.append(self.robustness_reporter.generate_robustness_report(
                analysis_results.robustness_analysis, self.config.include_details
            ))
        
        return "\\n".join(report_parts)
    
    def _create_json_report(self, analysis_results: ComprehensiveAnalysisResult) -> str:
        """創建 JSON 報告"""
        
        # 將分析結果轉換為可序列化的字典
        report_data = {
            'metadata': analysis_results.analysis_metadata,
            'config': {
                'title': self.config.title,
                'author': self.config.author,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # 添加各組件結果 (簡化版)
        if analysis_results.mcmc_diagnostics:
            report_data['mcmc_diagnostics'] = {
                'overall_convergence': analysis_results.mcmc_diagnostics.overall_convergence.value,
                'summary_statistics': analysis_results.mcmc_diagnostics.summary_statistics,
                'n_recommendations': len(analysis_results.mcmc_diagnostics.recommendations)
            }
        
        if analysis_results.model_comparison:
            report_data['model_comparison'] = {
                'best_model': analysis_results.model_comparison.best_model.model_name if analysis_results.model_comparison.best_model else None,
                'n_models': len(analysis_results.model_comparison.model_comparisons),
                'n_recommendations': len(analysis_results.model_comparison.recommendations)
            }
        
        if analysis_results.robustness_analysis:
            report_data['robustness_analysis'] = {
                'overall_robustness': analysis_results.robustness_analysis.overall_robustness.value,
                'robustness_score': analysis_results.robustness_analysis.robustness_score,
                'n_recommendations': len(analysis_results.robustness_analysis.recommendations)
            }
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def _save_report(self, content: str, save_path: Path):
        """保存報告"""
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def generate_quick_summary(self, 
                             analysis_results: ComprehensiveAnalysisResult) -> str:
        """生成快速摘要"""
        
        summary = []
        summary.append("📊 貝氏分析快速摘要")
        summary.append("=" * 40)
        
        # MCMC 狀態
        if analysis_results.mcmc_diagnostics:
            status = analysis_results.mcmc_diagnostics.overall_convergence.value
            icon = "✅" if status in ['excellent', 'good'] else "⚠️" if status == 'acceptable' else "❌"
            summary.append(f"{icon} MCMC 收斂: {status}")
        
        # 最佳模型
        if analysis_results.model_comparison and analysis_results.model_comparison.best_model:
            best_model = analysis_results.model_comparison.best_model
            summary.append(f"🏆 推薦模型: {best_model.model_name}")
            summary.append(f"📊 模型權重: {best_model.model_weight:.3f}")
        
        # 穩健性評分
        if analysis_results.robustness_analysis:
            score = analysis_results.robustness_analysis.robustness_score
            icon = "✅" if score > 75 else "⚠️" if score > 50 else "❌"
            summary.append(f"{icon} 穩健性評分: {score:.1f}/100")
        
        return "\\n".join(summary)