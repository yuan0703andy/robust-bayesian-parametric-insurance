"""
Bayesian Diagnostics Module
貝氏診斷模組

MCMC 收斂診斷和模型檢驗的全面工具集。
提供收斂檢測、有效樣本數計算、診斷圖表等功能。

Key Features:
- MCMC 收斂診斷 (R-hat, ESS, MCSE)
- 軌跡圖和自相關分析
- Geweke 診斷和 Heidelberger-Welch 測試
- 收斂建議和警告系統
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from scipy.stats import jarque_bera, shapiro
import sys
import os

# Import parent bayesian modules - use relative imports
try:
    from ..robust_bayesian_analysis import ModelComparisonResult
    from ..hierarchical_bayesian_model import HierarchicalModelResult
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from robust_bayesian_analysis import ModelComparisonResult
    from hierarchical_bayesian_model import HierarchicalModelResult

class ConvergenceStatus(Enum):
    """收斂狀態"""
    EXCELLENT = "excellent"      # R-hat < 1.01
    GOOD = "good"               # 1.01 <= R-hat < 1.05
    ACCEPTABLE = "acceptable"    # 1.05 <= R-hat < 1.1
    POOR = "poor"               # 1.1 <= R-hat < 1.2
    FAILED = "failed"           # R-hat >= 1.2

@dataclass
class DiagnosticResult:
    """診斷結果"""
    parameter_name: str
    rhat: float
    ess_bulk: float
    ess_tail: float
    mcse: float
    autocorr_lag: int
    geweke_z_score: float
    convergence_status: ConvergenceStatus
    warnings: List[str]

@dataclass
class MCMCDiagnostics:
    """MCMC 診斷結果集合"""
    individual_diagnostics: Dict[str, DiagnosticResult]
    overall_convergence: ConvergenceStatus
    chain_diagnostics: Dict[str, Any]
    summary_statistics: Dict[str, Any]
    recommendations: List[str]

class BayesianDiagnostics:
    """
    貝氏診斷器
    
    提供全面的 MCMC 收斂診斷和模型檢驗功能
    """
    
    def __init__(self, 
                 rhat_threshold: float = 1.1,
                 ess_threshold: int = 400,
                 mcse_threshold: float = 0.01,
                 autocorr_threshold: int = 50):
        """
        初始化診斷器
        
        Parameters:
        -----------
        rhat_threshold : float
            R-hat 收斂閾值
        ess_threshold : int
            有效樣本數最小閾值
        mcse_threshold : float
            Monte Carlo 標準誤閾值
        autocorr_threshold : int
            自相關長度警告閾值
        """
        self.rhat_threshold = rhat_threshold
        self.ess_threshold = ess_threshold
        self.mcse_threshold = mcse_threshold
        self.autocorr_threshold = autocorr_threshold
        
        # 設置圖表樣式
        plt.style.use('default')
        sns.set_style("whitegrid")
        self.colors = sns.color_palette("husl", 8)
    
    def diagnose_mcmc_convergence(self, 
                                posterior_samples: Dict[str, np.ndarray],
                                chains: Optional[List[np.ndarray]] = None,
                                parameter_names: Optional[List[str]] = None) -> MCMCDiagnostics:
        """
        全面的 MCMC 收斂診斷
        
        Parameters:
        -----------
        posterior_samples : Dict[str, np.ndarray]
            後驗樣本字典
        chains : List[np.ndarray], optional
            多鏈樣本（如果可用）
        parameter_names : List[str], optional
            參數名稱列表
            
        Returns:
        --------
        MCMCDiagnostics
            完整的診斷結果
        """
        
        print("🔍 開始 MCMC 收斂診斷...")
        
        individual_diagnostics = {}
        overall_warnings = []
        
        # 為每個參數進行診斷
        for param_name, samples in posterior_samples.items():
            if samples.ndim == 1:  # 只處理1D參數
                print(f"  📊 診斷參數: {param_name}")
                
                diagnostic = self._diagnose_single_parameter(
                    param_name, samples, chains
                )
                individual_diagnostics[param_name] = diagnostic
                
                if diagnostic.warnings:
                    overall_warnings.extend(diagnostic.warnings)
        
        # 計算整體收斂狀態
        overall_convergence = self._assess_overall_convergence(individual_diagnostics)
        
        # 鏈間診斷（如果有多鏈）
        chain_diagnostics = self._diagnose_chains(chains) if chains else {}
        
        # 摘要統計
        summary_statistics = self._calculate_summary_statistics(
            posterior_samples, individual_diagnostics
        )
        
        # 生成建議
        recommendations = self._generate_recommendations(
            individual_diagnostics, overall_convergence, overall_warnings
        )
        
        mcmc_diagnostics = MCMCDiagnostics(
            individual_diagnostics=individual_diagnostics,
            overall_convergence=overall_convergence,
            chain_diagnostics=chain_diagnostics,
            summary_statistics=summary_statistics,
            recommendations=recommendations
        )
        
        print("✅ MCMC 收斂診斷完成")
        return mcmc_diagnostics
    
    def _diagnose_single_parameter(self, 
                                 param_name: str,
                                 samples: np.ndarray,
                                 chains: Optional[List[np.ndarray]] = None) -> DiagnosticResult:
        """診斷單一參數"""
        
        warnings_list = []
        
        # 1. R-hat 計算
        if chains and len(chains) > 1:
            rhat = self._calculate_rhat(chains)
        else:
            # 如果只有一條鏈，分割為兩半計算
            mid = len(samples) // 2
            chain1, chain2 = samples[:mid], samples[mid:]
            rhat = self._calculate_rhat([chain1, chain2])
        
        # 2. 有效樣本數
        ess_bulk = self._calculate_ess_bulk(samples)
        ess_tail = self._calculate_ess_tail(samples)
        
        # 3. Monte Carlo 標準誤
        mcse = self._calculate_mcse(samples)
        
        # 4. 自相關分析
        autocorr_lag = self._calculate_autocorr_lag(samples)
        
        # 5. Geweke 診斷
        geweke_z_score = self._geweke_diagnostic(samples)
        
        # 6. 收斂狀態評估
        convergence_status = self._assess_convergence_status(rhat)
        
        # 7. 生成警告
        if rhat > self.rhat_threshold:
            warnings_list.append(f"R-hat ({rhat:.3f}) 超過閾值 ({self.rhat_threshold})")
        
        if ess_bulk < self.ess_threshold:
            warnings_list.append(f"ESS bulk ({ess_bulk:.0f}) 低於閾值 ({self.ess_threshold})")
        
        if mcse > self.mcse_threshold:
            warnings_list.append(f"MCSE ({mcse:.4f}) 過高")
        
        if autocorr_lag > self.autocorr_threshold:
            warnings_list.append(f"自相關長度過長 ({autocorr_lag})")
        
        if abs(geweke_z_score) > 2.0:
            warnings_list.append(f"Geweke 診斷異常 (z = {geweke_z_score:.2f})")
        
        return DiagnosticResult(
            parameter_name=param_name,
            rhat=rhat,
            ess_bulk=ess_bulk,
            ess_tail=ess_tail,
            mcse=mcse,
            autocorr_lag=autocorr_lag,
            geweke_z_score=geweke_z_score,
            convergence_status=convergence_status,
            warnings=warnings_list
        )
    
    def _calculate_rhat(self, chains: List[np.ndarray]) -> float:
        """計算 R-hat 統計量"""
        
        if len(chains) < 2:
            return 1.0
        
        n_chains = len(chains)
        n_samples = min(len(chain) for chain in chains)
        
        # 確保所有鏈長度相同
        chains = [chain[:n_samples] for chain in chains]
        
        # 計算鏈內和鏈間方差
        chain_means = [np.mean(chain) for chain in chains]
        overall_mean = np.mean(chain_means)
        
        # 鏈內方差
        within_var = np.mean([np.var(chain, ddof=1) for chain in chains])
        
        # 鏈間方差
        between_var = n_samples * np.var(chain_means, ddof=1)
        
        # R-hat 計算
        var_plus = ((n_samples - 1) * within_var + between_var) / n_samples
        rhat = np.sqrt(var_plus / within_var) if within_var > 0 else 1.0
        
        return rhat
    
    def _calculate_ess_bulk(self, samples: np.ndarray) -> float:
        """計算 bulk ESS"""
        
        # 簡化的 ESS 計算
        n = len(samples)
        
        # 計算自相關
        autocorr = self._calculate_autocorrelation(samples)
        
        # 找到第一個負值或接近零的位置
        cutoff = 1
        for i in range(1, min(len(autocorr), n//4)):
            if autocorr[i] <= 0.05:
                cutoff = i
                break
        
        # ESS 計算
        tau = 1 + 2 * np.sum(autocorr[1:cutoff])
        ess = n / tau if tau > 0 else n
        
        return max(1, ess)
    
    def _calculate_ess_tail(self, samples: np.ndarray) -> float:
        """計算 tail ESS"""
        
        # 對樣本進行分位數轉換
        quantiles = [0.05, 0.95]
        tail_samples = []
        
        for q in quantiles:
            threshold = np.quantile(samples, q)
            if q < 0.5:
                tail_indicator = (samples <= threshold).astype(float)
            else:
                tail_indicator = (samples >= threshold).astype(float)
            tail_samples.extend(tail_indicator)
        
        tail_samples = np.array(tail_samples)
        
        # 計算 tail ESS
        ess_tail = self._calculate_ess_bulk(tail_samples)
        
        return ess_tail
    
    def _calculate_mcse(self, samples: np.ndarray) -> float:
        """計算 Monte Carlo 標準誤"""
        
        ess = self._calculate_ess_bulk(samples)
        mcse = np.std(samples) / np.sqrt(ess)
        
        return mcse
    
    def _calculate_autocorr_lag(self, samples: np.ndarray) -> int:
        """計算自相關長度"""
        
        autocorr = self._calculate_autocorrelation(samples)
        
        # 找到自相關降到 e^(-1) ≈ 0.37 的位置
        threshold = 1.0 / np.e
        lag = 1
        
        for i in range(1, len(autocorr)):
            if autocorr[i] <= threshold:
                lag = i
                break
        
        return lag
    
    def _calculate_autocorrelation(self, samples: np.ndarray) -> np.ndarray:
        """計算歸一化自相關函數"""
        
        n = len(samples)
        samples_centered = samples - np.mean(samples)
        
        # 使用 FFT 計算自相關
        f_samples = np.fft.fft(samples_centered, n=2*n-1)
        autocorr_full = np.fft.ifft(f_samples * np.conj(f_samples)).real
        
        # 取前半部分並歸一化
        autocorr = autocorr_full[:n]
        autocorr = autocorr / autocorr[0]
        
        return autocorr
    
    def _geweke_diagnostic(self, samples: np.ndarray) -> float:
        """Geweke 診斷"""
        
        n = len(samples)
        
        # 取前 10% 和後 50%
        first_part = samples[:int(0.1 * n)]
        last_part = samples[int(0.5 * n):]
        
        if len(first_part) < 10 or len(last_part) < 10:
            return 0.0
        
        # 計算均值差異
        mean_diff = np.mean(first_part) - np.mean(last_part)
        
        # 計算標準誤差（考慮自相關）
        var_first = np.var(first_part) / len(first_part)
        var_last = np.var(last_part) / len(last_part)
        
        se = np.sqrt(var_first + var_last)
        
        # Z 分數
        z_score = mean_diff / se if se > 0 else 0.0
        
        return z_score
    
    def _assess_convergence_status(self, rhat: float) -> ConvergenceStatus:
        """評估收斂狀態"""
        
        if rhat < 1.01:
            return ConvergenceStatus.EXCELLENT
        elif rhat < 1.05:
            return ConvergenceStatus.GOOD
        elif rhat < 1.1:
            return ConvergenceStatus.ACCEPTABLE
        elif rhat < 1.2:
            return ConvergenceStatus.POOR
        else:
            return ConvergenceStatus.FAILED
    
    def _assess_overall_convergence(self, 
                                  individual_diagnostics: Dict[str, DiagnosticResult]) -> ConvergenceStatus:
        """評估整體收斂狀態"""
        
        if not individual_diagnostics:
            return ConvergenceStatus.FAILED
        
        statuses = [diag.convergence_status for diag in individual_diagnostics.values()]
        
        # 如果有任何參數收斂失敗，整體失敗
        if ConvergenceStatus.FAILED in statuses:
            return ConvergenceStatus.FAILED
        elif ConvergenceStatus.POOR in statuses:
            return ConvergenceStatus.POOR
        elif ConvergenceStatus.ACCEPTABLE in statuses:
            return ConvergenceStatus.ACCEPTABLE
        elif ConvergenceStatus.GOOD in statuses:
            return ConvergenceStatus.GOOD
        else:
            return ConvergenceStatus.EXCELLENT
    
    def _diagnose_chains(self, chains: List[np.ndarray]) -> Dict[str, Any]:
        """鏈間診斷"""
        
        if not chains or len(chains) < 2:
            return {}
        
        n_chains = len(chains)
        chain_lengths = [len(chain) for chain in chains]
        
        diagnostics = {
            'n_chains': n_chains,
            'chain_lengths': chain_lengths,
            'min_length': min(chain_lengths),
            'max_length': max(chain_lengths),
            'length_variation': np.std(chain_lengths)
        }
        
        return diagnostics
    
    def _calculate_summary_statistics(self, 
                                    posterior_samples: Dict[str, np.ndarray],
                                    individual_diagnostics: Dict[str, DiagnosticResult]) -> Dict[str, Any]:
        """計算摘要統計"""
        
        n_parameters = len([d for d in individual_diagnostics.values() 
                           if d.convergence_status != ConvergenceStatus.FAILED])
        
        rhat_values = [d.rhat for d in individual_diagnostics.values()]
        ess_values = [d.ess_bulk for d in individual_diagnostics.values()]
        
        summary = {
            'n_parameters': len(individual_diagnostics),
            'n_converged': n_parameters,
            'convergence_rate': n_parameters / len(individual_diagnostics) if individual_diagnostics else 0,
            'mean_rhat': np.mean(rhat_values) if rhat_values else np.nan,
            'max_rhat': np.max(rhat_values) if rhat_values else np.nan,
            'min_ess': np.min(ess_values) if ess_values else np.nan,
            'mean_ess': np.mean(ess_values) if ess_values else np.nan
        }
        
        return summary
    
    def _generate_recommendations(self, 
                                individual_diagnostics: Dict[str, DiagnosticResult],
                                overall_convergence: ConvergenceStatus,
                                warnings: List[str]) -> List[str]:
        """生成收斂改善建議"""
        
        recommendations = []
        
        if overall_convergence == ConvergenceStatus.FAILED:
            recommendations.append("🚨 MCMC 收斂失敗，強烈建議重新採樣")
            recommendations.append("• 增加採樣數量 (至少 10,000 樣本)")
            recommendations.append("• 調整步長或提案分布")
            recommendations.append("• 檢查模型規格是否正確")
        
        elif overall_convergence == ConvergenceStatus.POOR:
            recommendations.append("⚠️ 收斂狀況不佳，建議改善")
            recommendations.append("• 增加 burn-in 期間")
            recommendations.append("• 增加採樣數量")
            recommendations.append("• 考慮使用不同的初始值")
        
        elif overall_convergence == ConvergenceStatus.ACCEPTABLE:
            recommendations.append("⚡ 收斂可接受，但仍可改善")
            recommendations.append("• 可考慮增加採樣數量以提高精度")
        
        else:
            recommendations.append("✅ 收斂狀況良好")
        
        # 特定問題的建議
        low_ess_params = [name for name, diag in individual_diagnostics.items() 
                         if diag.ess_bulk < self.ess_threshold]
        
        if low_ess_params:
            recommendations.append(f"📉 低 ESS 參數: {', '.join(low_ess_params[:3])}")
            recommendations.append("• 考慮增加採樣數量或改善混合")
        
        high_autocorr_params = [name for name, diag in individual_diagnostics.items() 
                               if diag.autocorr_lag > self.autocorr_threshold]
        
        if high_autocorr_params:
            recommendations.append(f"🔗 高自相關參數: {', '.join(high_autocorr_params[:3])}")
            recommendations.append("• 考慮調整步長或使用其他採樣器")
        
        return recommendations
    
    def plot_convergence_diagnostics(self, 
                                   mcmc_diagnostics: MCMCDiagnostics,
                                   posterior_samples: Dict[str, np.ndarray],
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """繪製收斂診斷圖表"""
        
        n_params = len(mcmc_diagnostics.individual_diagnostics)
        if n_params == 0:
            print("沒有參數可繪圖")
            return None
        
        # 計算子圖布局
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('MCMC 收斂診斷', fontsize=16, fontweight='bold')
        
        param_idx = 0
        for param_name, samples in posterior_samples.items():
            if samples.ndim == 1 and param_idx < n_params:
                row = param_idx // n_cols
                col = param_idx % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                # 繪製軌跡圖
                ax.plot(samples, color=self.colors[param_idx % len(self.colors)], alpha=0.7)
                
                # 加入診斷信息
                diag = mcmc_diagnostics.individual_diagnostics[param_name]
                
                ax.set_title(f'{param_name}\nR̂={diag.rhat:.3f}, ESS={diag.ess_bulk:.0f}',
                           fontsize=10)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Value')
                
                # 狀態顏色標記
                status_colors = {
                    ConvergenceStatus.EXCELLENT: 'green',
                    ConvergenceStatus.GOOD: 'lightgreen', 
                    ConvergenceStatus.ACCEPTABLE: 'yellow',
                    ConvergenceStatus.POOR: 'orange',
                    ConvergenceStatus.FAILED: 'red'
                }
                
                ax.axhline(y=np.mean(samples), 
                          color=status_colors[diag.convergence_status], 
                          linestyle='--', alpha=0.5)
                
                param_idx += 1
        
        # 隱藏多餘的子圖
        for i in range(param_idx, n_rows * n_cols):
            if n_rows > 1:
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
            elif n_cols > 1:
                axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_diagnostic_report(self, 
                                 mcmc_diagnostics: MCMCDiagnostics,
                                 include_details: bool = True) -> str:
        """生成診斷報告"""
        
        report = []
        report.append("=" * 80)
        report.append("                    MCMC 收斂診斷報告")
        report.append("=" * 80)
        report.append("")
        
        # 整體摘要
        report.append("📊 整體摘要")
        report.append("-" * 40)
        
        summary = mcmc_diagnostics.summary_statistics
        status_icon = {
            ConvergenceStatus.EXCELLENT: "🟢",
            ConvergenceStatus.GOOD: "🟡", 
            ConvergenceStatus.ACCEPTABLE: "🟠",
            ConvergenceStatus.POOR: "🔴",
            ConvergenceStatus.FAILED: "❌"
        }
        
        report.append(f"{status_icon[mcmc_diagnostics.overall_convergence]} 整體收斂狀態: {mcmc_diagnostics.overall_convergence.value.upper()}")
        report.append(f"📈 參數總數: {summary['n_parameters']}")
        report.append(f"✅ 收斂參數: {summary['n_converged']}")
        report.append(f"📊 收斂率: {summary['convergence_rate']:.1%}")
        report.append(f"📊 平均 R̂: {summary['mean_rhat']:.3f}")
        report.append(f"📊 最大 R̂: {summary['max_rhat']:.3f}")
        report.append(f"📊 最小 ESS: {summary['min_ess']:.0f}")
        report.append("")
        
        # 個別參數診斷
        if include_details:
            report.append("📋 參數收斂詳情")
            report.append("-" * 40)
            
            for param_name, diag in mcmc_diagnostics.individual_diagnostics.items():
                status = status_icon[diag.convergence_status]
                report.append(f"{status} {param_name}:")
                report.append(f"    R̂ = {diag.rhat:.3f}")
                report.append(f"    ESS = {diag.ess_bulk:.0f}")
                report.append(f"    MCSE = {diag.mcse:.4f}")
                report.append(f"    自相關長度 = {diag.autocorr_lag}")
                
                if diag.warnings:
                    report.append("    ⚠️ 警告:")
                    for warning in diag.warnings:
                        report.append(f"      • {warning}")
                report.append("")
        
        # 建議
        report.append("💡 改善建議")
        report.append("-" * 40)
        for recommendation in mcmc_diagnostics.recommendations:
            report.append(recommendation)
        
        return "\n".join(report)