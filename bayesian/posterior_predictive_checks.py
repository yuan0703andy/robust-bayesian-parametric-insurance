#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
posterior_predictive_checks.py
===============================
Modular Posterior Predictive Checks for Robust Bayesian Models
模組化後驗預測檢查：用於強健貝氏模型

This module provides reusable PPC functionality that can be imported and used
across different analysis scripts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import warnings
warnings.filterwarnings('ignore')


class PPCValidator:
    """
    Posterior Predictive Check Validator
    後驗預測檢查驗證器
    
    Provides modular PPC functionality for model validation
    """
    
    def __init__(self, model_name: str = "BayesianModel"):
        """
        Initialize PPC validator
        
        Parameters:
        -----------
        model_name : str
            Name for this model (used in plots and reports)
        """
        self.model_name = model_name
        self.ppc_results = {}
        
    def compute_test_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute test statistics for observed data
        計算觀測數據的檢驗統計量
        
        Parameters:
        -----------
        data : np.ndarray
            Observed data
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of test statistics
        """
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
    
    def generate_replications(self, 
                            fitted_params: Dict[str, Any],
                            distribution: str,
                            n_replications: int = 100,
                            n_observations: Optional[int] = None) -> np.ndarray:
        """
        Generate posterior predictive replications
        生成後驗預測重複樣本
        
        Parameters:
        -----------
        fitted_params : Dict[str, Any]
            Fitted model parameters
        distribution : str
            Distribution type ('normal', 'studentt', 'lognormal')
        n_replications : int
            Number of replications to generate
        n_observations : int, optional
            Number of observations per replication
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_replications, n_observations)
        """
        
        if distribution == 'normal':
            mu = fitted_params['mu']
            sigma = fitted_params['sigma']
            replications = np.random.normal(mu, sigma, (n_replications, n_observations))
            
        elif distribution == 'studentt':
            nu = fitted_params['nu']
            mu = fitted_params['mu'] 
            sigma = fitted_params['sigma']
            replications = np.random.standard_t(nu, (n_replications, n_observations)) * sigma + mu
            
        elif distribution == 'lognormal':
            mu = fitted_params['mu']
            sigma = fitted_params['sigma']
            replications = np.random.lognormal(mu, sigma, (n_replications, n_observations))
            
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
            
        return replications
    
    def compute_ppc_pvalues(self, 
                           observed_data: np.ndarray,
                           replications: np.ndarray) -> Dict[str, float]:
        """
        Compute PPC p-values for multiple test statistics
        計算多個檢驗統計量的PPC p值
        
        Parameters:
        -----------
        observed_data : np.ndarray
            Observed data
        replications : np.ndarray
            Posterior predictive replications
            
        Returns:
        --------
        Dict[str, float]
            P-values for each test statistic
        """
        
        # Compute observed statistics
        obs_stats = self.compute_test_statistics(observed_data)
        
        # Compute replicated statistics
        rep_stats = {}
        for stat_name in obs_stats.keys():
            if stat_name == 'mean':
                rep_values = np.mean(replications, axis=1)
            elif stat_name == 'std':
                rep_values = np.std(replications, axis=1)
            elif stat_name == 'min':
                rep_values = np.min(replications, axis=1)
            elif stat_name == 'max':
                rep_values = np.max(replications, axis=1)
            elif stat_name == 'median':
                rep_values = np.median(replications, axis=1)
            elif stat_name == 'q25':
                rep_values = np.percentile(replications, 25, axis=1)
            elif stat_name == 'q75':
                rep_values = np.percentile(replications, 75, axis=1)
            elif stat_name == 'skewness':
                rep_values = [stats.skew(rep) for rep in replications]
            elif stat_name == 'kurtosis':
                rep_values = [stats.kurtosis(rep) for rep in replications]
            
            # Compute p-value
            p_value = np.mean(rep_values >= obs_stats[stat_name])
            rep_stats[stat_name] = {
                'observed': obs_stats[stat_name],
                'replicated_mean': np.mean(rep_values),
                'replicated_std': np.std(rep_values),
                'p_value': p_value
            }
        
        return rep_stats
    
    def assess_fit_quality(self, ppc_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess overall model fit quality from PPC results
        從PPC結果評估整體模型擬合品質
        
        Parameters:
        -----------
        ppc_results : Dict[str, Any]
            Results from compute_ppc_pvalues
            
        Returns:
        --------
        Dict[str, Any]
            Fit quality assessment
        """
        
        p_values = [result['p_value'] for result in ppc_results.values()]
        
        # Count p-values in reasonable range (0.05 to 0.95)
        good_pvalues = sum(0.05 <= p <= 0.95 for p in p_values)
        total_stats = len(p_values)
        fit_quality = good_pvalues / total_stats
        
        # Identify problematic statistics
        problematic_stats = []
        for stat_name, result in ppc_results.items():
            p_val = result['p_value']
            if p_val < 0.05:
                problematic_stats.append(f"{stat_name} (too low: {p_val:.3f})")
            elif p_val > 0.95:
                problematic_stats.append(f"{stat_name} (too high: {p_val:.3f})")
        
        return {
            'fit_quality_score': fit_quality,
            'good_statistics': good_pvalues,
            'total_statistics': total_stats,
            'fit_quality_percent': fit_quality * 100,
            'problematic_statistics': problematic_stats,
            'overall_assessment': self._get_fit_assessment(fit_quality)
        }
    
    def _get_fit_assessment(self, fit_quality: float) -> str:
        """Get text assessment of fit quality"""
        if fit_quality >= 0.8:
            return "Excellent fit"
        elif fit_quality >= 0.6:
            return "Good fit"
        elif fit_quality >= 0.4:
            return "Adequate fit"
        elif fit_quality >= 0.2:
            return "Poor fit"
        else:
            return "Very poor fit"
    
    def run_full_ppc(self,
                     observed_data: np.ndarray,
                     fitted_params: Dict[str, Any],
                     distribution: str,
                     n_replications: int = 100) -> Dict[str, Any]:
        """
        Run complete PPC analysis
        執行完整的PPC分析
        
        Parameters:
        -----------
        observed_data : np.ndarray
            Observed data
        fitted_params : Dict[str, Any]
            Fitted model parameters
        distribution : str
            Distribution type
        n_replications : int
            Number of replications
            
        Returns:
        --------
        Dict[str, Any]
            Complete PPC results
        """
        
        # Generate replications
        replications = self.generate_replications(
            fitted_params, distribution, n_replications, len(observed_data)
        )
        
        # Compute p-values
        ppc_stats = self.compute_ppc_pvalues(observed_data, replications)
        
        # Assess fit quality
        fit_assessment = self.assess_fit_quality(ppc_stats)
        
        # Store results
        results = {
            'model_name': self.model_name,
            'distribution': distribution,
            'n_replications': n_replications,
            'n_observations': len(observed_data),
            'ppc_statistics': ppc_stats,
            'fit_assessment': fit_assessment,
            'replications': replications,
            'observed_data': observed_data
        }
        
        # Cache for plotting
        self.ppc_results[f"{self.model_name}_{distribution}"] = results
        
        return results
    
    def plot_ppc_diagnostics(self, 
                           ppc_results: Dict[str, Any],
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create PPC diagnostic plots
        創建PPC診斷圖
        
        Parameters:
        -----------
        ppc_results : Dict[str, Any]
            Results from run_full_ppc
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'PPC Diagnostics: {ppc_results["model_name"]} ({ppc_results["distribution"]})', 
                     fontsize=16)
        
        observed = ppc_results['observed_data']
        replications = ppc_results['replications']
        
        # Plot 1: Histogram overlay
        axes[0, 0].hist(observed, bins=30, alpha=0.7, density=True, 
                       color='black', label='Observed')
        for i in range(min(20, replications.shape[0])):
            axes[0, 0].hist(replications[i], bins=30, alpha=0.1, 
                           density=True, color='blue')
        axes[0, 0].set_title('Data vs Replications')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        
        # Plot 2: P-value plot
        stats_names = list(ppc_results['ppc_statistics'].keys())
        p_values = [ppc_results['ppc_statistics'][stat]['p_value'] for stat in stats_names]
        
        colors = ['red' if p < 0.05 or p > 0.95 else 'green' for p in p_values]
        axes[0, 1].bar(range(len(stats_names)), p_values, color=colors)
        axes[0, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=0.95, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('PPC P-values')
        axes[0, 1].set_ylabel('P-value')
        axes[0, 1].set_xticks(range(len(stats_names)))
        axes[0, 1].set_xticklabels(stats_names, rotation=45)
        
        # Plot 3: Q-Q plot
        theoretical_quantiles = np.linspace(0.01, 0.99, len(observed))
        empirical_quantiles = np.quantile(observed, theoretical_quantiles)
        
        # Generate theoretical quantiles from fitted distribution
        if ppc_results['distribution'] == 'normal':
            mu = np.mean(replications)
            sigma = np.std(replications)
            theoretical_values = stats.norm.ppf(theoretical_quantiles, mu, sigma)
        elif ppc_results['distribution'] == 'studentt':
            # Use mean parameters from replications
            theoretical_values = np.quantile(replications.flatten(), theoretical_quantiles)
        else:
            theoretical_values = np.quantile(replications.flatten(), theoretical_quantiles)
        
        axes[1, 0].scatter(theoretical_values, empirical_quantiles, alpha=0.6)
        min_val = min(theoretical_values.min(), empirical_quantiles.min())
        max_val = max(theoretical_values.max(), empirical_quantiles.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].set_xlabel('Theoretical Quantiles')
        axes[1, 0].set_ylabel('Sample Quantiles')
        
        # Plot 4: Fit quality summary
        fit_assessment = ppc_results['fit_assessment']
        axes[1, 1].text(0.1, 0.8, f"Fit Quality: {fit_assessment['fit_quality_percent']:.1f}%", 
                        fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f"Assessment: {fit_assessment['overall_assessment']}", 
                        fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Good statistics: {fit_assessment['good_statistics']}/{fit_assessment['total_statistics']}", 
                        fontsize=12, transform=axes[1, 1].transAxes)
        
        if fit_assessment['problematic_statistics']:
            axes[1, 1].text(0.1, 0.4, "Problematic statistics:", 
                           fontsize=10, transform=axes[1, 1].transAxes)
            for i, stat in enumerate(fit_assessment['problematic_statistics'][:3]):
                axes[1, 1].text(0.1, 0.35 - i*0.05, f"• {stat}", 
                               fontsize=9, transform=axes[1, 1].transAxes)
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Fit Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class PPCComparator:
    """
    Compare multiple models using PPC
    使用PPC比較多個模型
    """
    
    def __init__(self):
        self.models = {}
        
    def add_model(self, model_name: str, ppc_results: Dict[str, Any]):
        """Add model PPC results for comparison"""
        self.models[model_name] = ppc_results
        
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all added models
        比較所有添加的模型
        
        Returns:
        --------
        pd.DataFrame
            Comparison results
        """
        
        comparison_data = []
        
        for model_name, results in self.models.items():
            fit_assessment = results['fit_assessment']
            
            row = {
                'Model': model_name,
                'Distribution': results['distribution'],
                'Fit_Quality_%': fit_assessment['fit_quality_percent'],
                'Good_Stats': f"{fit_assessment['good_statistics']}/{fit_assessment['total_statistics']}",
                'Assessment': fit_assessment['overall_assessment'],
                'N_Problematic': len(fit_assessment['problematic_statistics'])
            }
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Fit_Quality_%', ascending=False)
        
        return comparison_df
    
    def plot_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot model comparison
        繪製模型比較圖
        """
        
        comparison_df = self.compare_models()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Comparison via PPC', fontsize=16)
        
        # Plot 1: Fit quality comparison
        models = comparison_df['Model']
        fit_qualities = comparison_df['Fit_Quality_%']
        
        colors = ['green' if q >= 70 else 'orange' if q >= 50 else 'red' for q in fit_qualities]
        bars = ax1.bar(models, fit_qualities, color=colors)
        ax1.set_title('PPC Fit Quality Comparison')
        ax1.set_ylabel('Fit Quality (%)')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, quality in zip(bars, fit_qualities):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{quality:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Detailed comparison table
        ax2.axis('tight')
        ax2.axis('off')
        table = ax2.table(cellText=comparison_df.values,
                         colLabels=comparison_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


# Convenience functions for easy usage
def quick_ppc(observed_data: np.ndarray,
              fitted_params: Dict[str, Any], 
              distribution: str,
              model_name: str = "Model") -> Dict[str, Any]:
    """
    Quick PPC analysis function
    快速PPC分析函數
    
    Parameters:
    -----------
    observed_data : np.ndarray
        Observed data
    fitted_params : Dict[str, Any]
        Fitted parameters
    distribution : str
        Distribution type
    model_name : str
        Model name
        
    Returns:
    --------
    Dict[str, Any]
        PPC results
    """
    
    validator = PPCValidator(model_name)
    return validator.run_full_ppc(observed_data, fitted_params, distribution)


def compare_distributions(observed_data: np.ndarray,
                         models_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple distributions on the same data
    在相同數據上比較多個分佈
    
    Parameters:
    -----------
    observed_data : np.ndarray
        Observed data
    models_dict : Dict[str, Dict]
        Dictionary with model_name -> {'params': {...}, 'distribution': '...'}
        
    Returns:
    --------
    pd.DataFrame
        Comparison results
    """
    
    comparator = PPCComparator()
    
    for model_name, model_info in models_dict.items():
        ppc_results = quick_ppc(
            observed_data, 
            model_info['params'], 
            model_info['distribution'],
            model_name
        )
        comparator.add_model(model_name, ppc_results)
    
    return comparator.compare_models()