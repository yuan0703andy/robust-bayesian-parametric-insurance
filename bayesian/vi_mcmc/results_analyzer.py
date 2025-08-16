"""
Results Analyzer Module
結果分析模組
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from skill_scores.crps_score import calculate_crps
from skill_scores.rmse_score import calculate_rmse
from skill_scores.mae_score import calculate_mae
from insurance_analysis_refactored.core.skill_evaluator import SkillScoreEvaluator


class ResultsAnalyzer:
    """Analyze and compare VI and MCMC results"""
    
    def __init__(self):
        """Initialize Results Analyzer"""
        self.skill_evaluator = SkillScoreEvaluator()
        self.vi_results = None
        self.mcmc_results = None
        
    def analyze_vi_results(self, vi_df: pd.DataFrame) -> Dict:
        """
        Analyze VI screening results
        
        Args:
            vi_df: DataFrame with VI results
            
        Returns:
            Dictionary with analysis
        """
        self.vi_results = vi_df
        
        analysis = {
            'total_models': len(vi_df),
            'converged': vi_df['converged'].sum(),
            'convergence_rate': vi_df['converged'].mean(),
            'best_model': vi_df.iloc[0]['name'] if len(vi_df) > 0 else None,
            'best_elbo': vi_df.iloc[0]['elbo'] if len(vi_df) > 0 else None,
            'mean_time': vi_df['time'].mean(),
            'total_time': vi_df['time'].sum()
        }
        
        # Group by model type
        type_summary = vi_df.groupby('type').agg({
            'elbo': ['mean', 'max', 'std'],
            'converged': 'mean',
            'time': 'mean'
        }).round(2)
        
        analysis['type_summary'] = type_summary
        
        print("\n📊 VI Screening Analysis:")
        print(f"   Total models: {analysis['total_models']}")
        print(f"   Converged: {analysis['converged']} ({analysis['convergence_rate']:.1%})")
        print(f"   Best model: {analysis['best_model']} (ELBO: {analysis['best_elbo']:.2f})")
        print(f"   Total VI time: {analysis['total_time']/60:.1f} minutes")
        
        return analysis
    
    def analyze_mcmc_results(self, mcmc_results: Dict) -> pd.DataFrame:
        """
        Analyze MCMC validation results
        
        Args:
            mcmc_results: Dictionary of MCMC results
            
        Returns:
            DataFrame with comparison
        """
        self.mcmc_results = mcmc_results
        
        comparison_data = []
        
        for name, result in mcmc_results.items():
            comparison_data.append({
                'model': name,
                'type': result['config'].model_type,
                'waic': result.get('waic'),
                'loo': result.get('loo'),
                'divergences': result['divergences'],
                'mean_ess': result['mean_ess'],
                'mean_rhat': result['mean_rhat'],
                'time': result['time']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by WAIC
        if 'waic' in df.columns and df['waic'].notna().any():
            df = df.sort_values('waic', ascending=False)
        
        print("\n🔥 MCMC Validation Analysis:")
        print(df.to_string())
        
        return df
    
    def compare_methods(self, vi_df: pd.DataFrame, mcmc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare VI and MCMC results
        
        Args:
            vi_df: VI results DataFrame
            mcmc_df: MCMC results DataFrame
            
        Returns:
            Combined comparison DataFrame
        """
        # Merge results
        comparison = pd.merge(
            vi_df[['name', 'type', 'elbo', 'converged', 'time']],
            mcmc_df[['model', 'waic', 'loo', 'divergences', 'mean_rhat', 'time']],
            left_on='name', right_on='model', how='inner'
        )
        
        # Rename columns
        comparison = comparison.rename(columns={
            'time_x': 'vi_time',
            'time_y': 'mcmc_time'
        })
        
        # Calculate combined score
        if 'waic' in comparison.columns:
            comparison['combined_score'] = (
                comparison['elbo'] + comparison['waic'].fillna(0)
            )
        
        # Sort by combined score
        if 'combined_score' in comparison.columns:
            comparison = comparison.sort_values('combined_score', ascending=False)
        
        print("\n🏆 Combined VI+MCMC Comparison:")
        print(comparison[['name', 'type', 'elbo', 'waic', 'divergences', 'combined_score']].to_string())
        
        return comparison
    
    def calculate_skill_scores(self, mcmc_results: Dict, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Calculate skill scores for MCMC results
        
        Args:
            mcmc_results: Dictionary of MCMC results
            X: Feature matrix
            y: Observed values
            
        Returns:
            DataFrame with skill scores
        """
        from .model_factory import ModelFactory
        
        scores_data = []
        
        for name, result in mcmc_results.items():
            trace = result['trace']
            config = result['config']
            
            # Generate posterior predictive
            model = ModelFactory.create_model(config, X, y)
            
            with model:
                pp = pm.sample_posterior_predictive(trace, progressbar=False)
            
            # Get predictions
            y_pred = pp.posterior_predictive['y_obs'].mean(dim=['chain', 'draw']).values
            
            # Calculate scores
            rmse = calculate_rmse(y, y_pred)
            mae = calculate_mae(y, y_pred)
            
            # CRPS (using ensemble)
            pp_ensemble = pp.posterior_predictive['y_obs'].values
            n_samples = pp_ensemble.shape[-1]
            pp_ensemble = pp_ensemble.reshape(-1, n_samples).T
            
            crps_scores = []
            for i, obs in enumerate(y):
                ensemble = pp_ensemble[i, :]
                crps = calculate_crps([obs], forecasts_ensemble=ensemble.reshape(1, -1))
                crps_scores.append(crps)
            
            mean_crps = np.mean(crps_scores)
            
            scores_data.append({
                'model': name,
                'type': config.model_type,
                'rmse': rmse,
                'mae': mae,
                'crps': mean_crps
            })
        
        df = pd.DataFrame(scores_data)
        df = df.sort_values('crps')  # Lower is better
        
        print("\n📈 Skill Scores Analysis:")
        print(df.to_string())
        
        return df
    
    def plot_comparison(self, comparison_df: pd.DataFrame, save_path: Optional[Path] = None):
        """
        Create comparison plots
        
        Args:
            comparison_df: Comparison DataFrame
            save_path: Path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # ELBO comparison
        ax = axes[0, 0]
        ax.bar(range(len(comparison_df)), comparison_df['elbo'])
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels(comparison_df['name'], rotation=45, ha='right')
        ax.set_ylabel('ELBO')
        ax.set_title('VI ELBO Scores')
        
        # WAIC comparison
        ax = axes[0, 1]
        if 'waic' in comparison_df.columns:
            ax.bar(range(len(comparison_df)), comparison_df['waic'].fillna(0))
            ax.set_xticks(range(len(comparison_df)))
            ax.set_xticklabels(comparison_df['name'], rotation=45, ha='right')
            ax.set_ylabel('WAIC')
            ax.set_title('MCMC WAIC Scores')
        
        # Time comparison
        ax = axes[1, 0]
        x = np.arange(len(comparison_df))
        width = 0.35
        ax.bar(x - width/2, comparison_df['vi_time'], width, label='VI')
        ax.bar(x + width/2, comparison_df['mcmc_time'], width, label='MCMC')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['name'], rotation=45, ha='right')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Computation Time')
        ax.legend()
        
        # Divergences
        ax = axes[1, 1]
        ax.bar(range(len(comparison_df)), comparison_df['divergences'])
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels(comparison_df['name'], rotation=45, ha='right')
        ax.set_ylabel('Divergences')
        ax.set_title('MCMC Divergences')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / 'comparison_plots.png', dpi=150)
            print(f"\n📊 Plots saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: Path) -> str:
        """
        Generate comprehensive report
        
        Args:
            save_path: Path to save report
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("VI+MCMC Framework Analysis Report")
        report.append("=" * 60)
        
        if self.vi_results is not None:
            report.append("\n## VI Screening Results")
            report.append(f"Total models screened: {len(self.vi_results)}")
            report.append(f"Best model: {self.vi_results.iloc[0]['name']}")
            report.append(f"Best ELBO: {self.vi_results.iloc[0]['elbo']:.2f}")
            
        if self.mcmc_results is not None:
            report.append("\n## MCMC Validation Results")
            report.append(f"Models validated: {len(self.mcmc_results)}")
            
            # Find best by WAIC
            best_waic = max(self.mcmc_results.values(), 
                          key=lambda x: x.get('waic', -np.inf))
            if best_waic and best_waic.get('waic'):
                report.append(f"Best model (WAIC): {best_waic['config'].name}")
                report.append(f"WAIC: {best_waic['waic']:.2f}")
        
        report_text = "\n".join(report)
        
        # Save report
        with open(save_path / 'analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print("\n📄 Report saved to", save_path / 'analysis_report.txt')
        
        return report_text