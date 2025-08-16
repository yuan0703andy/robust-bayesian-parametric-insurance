#!/usr/bin/env python3
"""
05. Comprehensive Model Comparison - Research Paper Grade
完整模型比較 - 研究論文級

Complete Bayesian model comparison framework for demonstrating ε-contamination superiority.
完整貝氏模型比較框架，用於證明 ε-contamination 的優越性。

Author: Research Team
Date: 2025-01-16
"""

import os
import sys
import argparse
import pickle
import time
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# HPC environment setup
if 'SLURM_CPUS_PER_TASK' in os.environ:
    os.environ['OMP_NUM_THREADS'] = os.environ['SLURM_CPUS_PER_TASK']
    os.environ['MKL_NUM_THREADS'] = os.environ['SLURM_CPUS_PER_TASK']
    HPC_CORES = int(os.environ['SLURM_CPUS_PER_TASK'])
    print(f"🖥️ SLURM HPC: Using {HPC_CORES} cores")
else:
    import multiprocessing
    HPC_CORES = multiprocessing.cpu_count()

# Force CPU-only execution
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64,mode=FAST_COMPILE,linker=py,allow_gc=True'
os.environ['PYMC_PROGRESS'] = 'True'
os.environ['PYTHONUNBUFFERED'] = '1'

print("=" * 80)
print("📊 Comprehensive Model Comparison Framework")
print("🎯 Research Paper Grade Analysis")
print("=" * 80)

# Import PyMC and ArviZ
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    HAS_PYMC = True
except ImportError:
    print("❌ PyMC not available")
    HAS_PYMC = False
    sys.exit(1)

# Import Bayesian framework
from bayesian import (
    EpsilonContaminationMCMC,
    MCMCConfig as EpsilonMCMCConfig,
    quick_contamination_analysis,
    get_cpu_optimized_mcmc_config,
    configure_pymc_environment
)

# ============================================================================
# Model Specifications
# ============================================================================

class PriorType(Enum):
    """Prior distribution types"""
    WEAK_INFORMATIVE = "weak_informative"
    STRONG_INFORMATIVE = "strong_informative"
    NON_INFORMATIVE = "non_informative"
    HORSESHOE = "horseshoe"
    STUDENT_T = "student_t"
    LAPLACE = "laplace"

class LikelihoodType(Enum):
    """Likelihood function types"""
    NORMAL = "normal"
    STUDENT_T = "student_t"
    LAPLACE = "laplace"
    CAUCHY = "cauchy"
    HUBER = "huber"
    ASYMMETRIC_LAPLACE = "asymmetric_laplace"

class ModelStructure(Enum):
    """Model structure types"""
    FULL_HIERARCHY = "full_hierarchy"
    PARTIAL_HIERARCHY = "partial_hierarchy"
    POOLED = "pooled"
    EPSILON_CONTAMINATION = "epsilon_contamination"

@dataclass
class ModelSpecification:
    """Complete model specification"""
    model_id: str
    prior_type: PriorType
    likelihood_type: LikelihoodType
    structure: ModelStructure
    epsilon: Optional[float] = None
    description: str = ""

@dataclass
class ConvergenceDiagnostics:
    """Detailed convergence diagnostics"""
    rhat: Dict[str, float]
    ess: Dict[str, float]
    bfmi: float
    divergences: int
    max_treedepth_reached: int
    converged: bool
    diagnostics_summary: str

@dataclass
class ModelResults:
    """Complete model results"""
    model_spec: ModelSpecification
    trace: Any  # PyMC trace
    convergence: ConvergenceDiagnostics
    dic: float
    waic: float
    loo: float
    execution_time: float
    posterior_predictive: Optional[np.ndarray] = None

# ============================================================================
# Model Builders
# ============================================================================

class BayesianModelBuilder:
    """Build various Bayesian models for comparison"""
    
    def __init__(self, mcmc_config: dict):
        self.mcmc_config = mcmc_config
        configure_pymc_environment()
    
    def build_model(self, data: np.ndarray, spec: ModelSpecification) -> pm.Model:
        """Build a model based on specification"""
        
        if spec.structure == ModelStructure.EPSILON_CONTAMINATION:
            return self._build_epsilon_contamination(data, spec)
        elif spec.structure == ModelStructure.FULL_HIERARCHY:
            return self._build_full_hierarchy(data, spec)
        elif spec.structure == ModelStructure.PARTIAL_HIERARCHY:
            return self._build_partial_hierarchy(data, spec)
        elif spec.structure == ModelStructure.POOLED:
            return self._build_pooled(data, spec)
        else:
            raise ValueError(f"Unknown structure: {spec.structure}")
    
    def _build_epsilon_contamination(self, data: np.ndarray, spec: ModelSpecification) -> pm.Model:
        """Build ε-contamination model"""
        with pm.Model() as model:
            # Standardize data
            data_std = (data - np.mean(data)) / np.std(data)
            
            # Hyperpriors
            alpha = pm.Normal("alpha", mu=0, sigma=0.5)
            log_beta = pm.Normal("log_beta", mu=-1, sigma=0.3)
            beta = pm.Deterministic("beta", pt.exp(log_beta))
            
            # Hierarchical structure
            phi_raw = pm.Normal("phi_raw", mu=0, sigma=1)
            phi = pm.Deterministic("phi", alpha + beta * phi_raw)
            
            log_tau = pm.Normal("log_tau", mu=-1, sigma=0.3)
            tau = pm.Deterministic("tau", pt.exp(log_tau))
            
            theta_raw = pm.Normal("theta_raw", mu=0, sigma=1)
            theta = pm.Deterministic("theta", phi + tau * theta_raw)
            
            # ε-contamination likelihood
            log_sigma = pm.Normal("log_sigma", mu=-1, sigma=0.3)
            sigma_obs = pm.Deterministic("sigma_obs", pt.exp(log_sigma))
            
            # Student-t approximation of ε-contamination
            epsilon = spec.epsilon or 0.05
            nu = 1.0 / epsilon - 1.0
            
            # Observations
            obs = pm.StudentT("obs", nu=nu, mu=theta, sigma=sigma_obs, observed=data_std)
            
        return model
    
    def _build_full_hierarchy(self, data: np.ndarray, spec: ModelSpecification) -> pm.Model:
        """Build full hierarchical model"""
        with pm.Model() as model:
            # Standardize data
            data_std = (data - np.mean(data)) / np.std(data)
            
            # Choose prior based on spec
            if spec.prior_type == PriorType.WEAK_INFORMATIVE:
                mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
                sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
            elif spec.prior_type == PriorType.STRONG_INFORMATIVE:
                mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=0.1)
                sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.1)
            elif spec.prior_type == PriorType.NON_INFORMATIVE:
                mu_alpha = pm.Flat("mu_alpha")
                sigma_alpha = pm.HalfFlat("sigma_alpha")
            elif spec.prior_type == PriorType.HORSESHOE:
                # Horseshoe prior for sparsity
                tau_0 = pm.HalfCauchy("tau_0", beta=1)
                lambda_i = pm.HalfCauchy("lambda_i", beta=1, shape=len(data))
                mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=tau_0 * lambda_i)
                sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
            else:
                mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
                sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
            
            # Hierarchical parameters
            alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha)
            
            # Choose likelihood based on spec
            if spec.likelihood_type == LikelihoodType.NORMAL:
                sigma = pm.HalfNormal("sigma", sigma=1)
                obs = pm.Normal("obs", mu=alpha, sigma=sigma, observed=data_std)
            elif spec.likelihood_type == LikelihoodType.STUDENT_T:
                nu = pm.Exponential("nu", 1/30)
                sigma = pm.HalfNormal("sigma", sigma=1)
                obs = pm.StudentT("obs", nu=nu, mu=alpha, sigma=sigma, observed=data_std)
            elif spec.likelihood_type == LikelihoodType.LAPLACE:
                b = pm.HalfNormal("b", sigma=1)
                obs = pm.Laplace("obs", mu=alpha, b=b, observed=data_std)
            elif spec.likelihood_type == LikelihoodType.CAUCHY:
                sigma = pm.HalfNormal("sigma", sigma=1)
                obs = pm.Cauchy("obs", alpha=alpha, beta=sigma, observed=data_std)
            else:
                sigma = pm.HalfNormal("sigma", sigma=1)
                obs = pm.Normal("obs", mu=alpha, sigma=sigma, observed=data_std)
            
        return model
    
    def _build_partial_hierarchy(self, data: np.ndarray, spec: ModelSpecification) -> pm.Model:
        """Build partial hierarchical model (2-level)"""
        with pm.Model() as model:
            # Standardize data
            data_std = (data - np.mean(data)) / np.std(data)
            
            # Simpler hierarchy
            mu = pm.Normal("mu", mu=0, sigma=1)
            
            # Likelihood
            if spec.likelihood_type == LikelihoodType.STUDENT_T:
                nu = pm.Exponential("nu", 1/30)
                sigma = pm.HalfNormal("sigma", sigma=1)
                obs = pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=data_std)
            else:
                sigma = pm.HalfNormal("sigma", sigma=1)
                obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data_std)
            
        return model
    
    def _build_pooled(self, data: np.ndarray, spec: ModelSpecification) -> pm.Model:
        """Build pooled model (no hierarchy)"""
        with pm.Model() as model:
            # Standardize data
            data_std = (data - np.mean(data)) / np.std(data)
            
            # Single parameter
            mu = pm.Normal("mu", mu=0, sigma=1)
            sigma = pm.HalfNormal("sigma", sigma=1)
            
            # Likelihood
            if spec.likelihood_type == LikelihoodType.NORMAL:
                obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data_std)
            elif spec.likelihood_type == LikelihoodType.STUDENT_T:
                nu = pm.Exponential("nu", 1/30)
                obs = pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=data_std)
            else:
                obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data_std)
            
        return model
    
    def run_mcmc(self, model: pm.Model, spec: ModelSpecification) -> Tuple[Any, float]:
        """Run MCMC sampling with convergence monitoring"""
        print(f"\n🔧 Running MCMC for {spec.model_id}...")
        
        start_time = time.time()
        
        with model:
            # Run sampling
            trace = pm.sample(
                draws=self.mcmc_config['n_samples'],
                tune=self.mcmc_config['n_warmup'],
                chains=self.mcmc_config['n_chains'],
                cores=self.mcmc_config['cores'],
                target_accept=self.mcmc_config['target_accept'],
                return_inferencedata=True,
                progressbar=True
            )
        
        execution_time = time.time() - start_time
        
        # Real-time convergence monitoring
        self.monitor_convergence(trace, spec.model_id)
        
        return trace, execution_time
    
    def monitor_convergence(self, trace: Any, model_id: str):
        """Real-time convergence monitoring with detailed diagnostics"""
        print(f"\n📊 Convergence Diagnostics for {model_id}:")
        print("=" * 60)
        
        # Get parameter names
        var_names = list(trace.posterior.data_vars.keys())
        
        # R-hat diagnostics
        print("   R-hat (target < 1.01):")
        rhat_issues = []
        for param in var_names[:10]:  # Show first 10 parameters
            try:
                rhat = float(az.rhat(trace, var_names=[param]).to_array().max())
                status = "✅" if rhat < 1.01 else "⚠️" if rhat < 1.05 else "❌"
                print(f"      {param:15s}: {rhat:.4f} {status}")
                if rhat >= 1.01:
                    rhat_issues.append(f"{param}={rhat:.3f}")
            except:
                pass
        
        if rhat_issues:
            print(f"   ⚠️ R-hat issues: {', '.join(rhat_issues)}")
        
        # ESS diagnostics
        print("\n   ESS (Effective Sample Size, target > 400):")
        ess_issues = []
        for param in var_names[:10]:
            try:
                ess_bulk = float(az.ess(trace, var_names=[param], method="bulk").to_array().min())
                ess_tail = float(az.ess(trace, var_names=[param], method="tail").to_array().min())
                ess_min = min(ess_bulk, ess_tail)
                status = "✅" if ess_min > 400 else "⚠️" if ess_min > 200 else "❌"
                print(f"      {param:15s}: bulk={ess_bulk:6.0f}, tail={ess_tail:6.0f} {status}")
                if ess_min < 400:
                    ess_issues.append(f"{param}={ess_min:.0f}")
            except:
                pass
        
        if ess_issues:
            print(f"   ⚠️ ESS issues: {', '.join(ess_issues)}")
        
        # Divergences
        try:
            divergences = trace.sample_stats.diverging.sum().values
            print(f"\n   Divergences: {divergences}")
            if divergences > 0:
                print(f"   ⚠️ {divergences} divergent transitions detected!")
        except:
            print("   Divergences: N/A")
        
        # BFMI (Bayesian Fraction of Missing Information)
        try:
            bfmi = az.bfmi(trace)
            print(f"\n   BFMI: {bfmi:.3f} (target > 0.3)")
            if bfmi < 0.3:
                print("   ⚠️ Low BFMI indicates poor exploration")
        except:
            print("   BFMI: N/A")
        
        # Max tree depth
        try:
            max_depth_reached = (trace.sample_stats.tree_depth == 
                               trace.sample_stats.max_tree_depth).sum().values
            if max_depth_reached > 0:
                print(f"   ⚠️ Max tree depth reached {max_depth_reached} times")
        except:
            pass
        
        # Overall convergence assessment
        print("\n   📈 Overall Assessment:")
        all_converged = len(rhat_issues) == 0 and len(ess_issues) == 0 and divergences == 0
        if all_converged:
            print("   ✅ Model converged successfully!")
        else:
            print("   ⚠️ Convergence issues detected - results may be unreliable")
        
        print("=" * 60)
    
    def extract_diagnostics(self, trace: Any) -> ConvergenceDiagnostics:
        """Extract comprehensive convergence diagnostics"""
        var_names = list(trace.posterior.data_vars.keys())
        
        # R-hat
        rhat_dict = {}
        for param in var_names:
            try:
                rhat_dict[param] = float(az.rhat(trace, var_names=[param]).to_array().max())
            except:
                rhat_dict[param] = np.nan
        
        # ESS
        ess_dict = {}
        for param in var_names:
            try:
                ess_bulk = float(az.ess(trace, var_names=[param], method="bulk").to_array().min())
                ess_tail = float(az.ess(trace, var_names=[param], method="tail").to_array().min())
                ess_dict[param] = min(ess_bulk, ess_tail)
            except:
                ess_dict[param] = np.nan
        
        # Other diagnostics
        try:
            bfmi = float(az.bfmi(trace))
        except:
            bfmi = np.nan
        
        try:
            divergences = int(trace.sample_stats.diverging.sum().values)
        except:
            divergences = 0
        
        try:
            max_treedepth = int((trace.sample_stats.tree_depth == 
                               trace.sample_stats.max_tree_depth).sum().values)
        except:
            max_treedepth = 0
        
        # Check convergence
        max_rhat = max(rhat_dict.values())
        min_ess = min(ess_dict.values())
        converged = (max_rhat < 1.01 and min_ess > 400 and 
                    divergences == 0 and bfmi > 0.3)
        
        # Summary
        summary = f"R-hat={max_rhat:.3f}, ESS={min_ess:.0f}, Div={divergences}"
        if converged:
            summary = "✅ " + summary
        else:
            summary = "⚠️ " + summary
        
        return ConvergenceDiagnostics(
            rhat=rhat_dict,
            ess=ess_dict,
            bfmi=bfmi,
            divergences=divergences,
            max_treedepth_reached=max_treedepth,
            converged=converged,
            diagnostics_summary=summary
        )

# ============================================================================
# Model Comparison Framework
# ============================================================================

class ComprehensiveModelComparison:
    """Complete model comparison framework"""
    
    def __init__(self, mcmc_config: dict):
        self.mcmc_config = mcmc_config
        self.builder = BayesianModelBuilder(mcmc_config)
        self.results = []
    
    def create_model_specifications(self) -> List[ModelSpecification]:
        """Create all model specifications for comparison"""
        specifications = []
        
        # 1. ε-contamination models (our champion)
        for epsilon in [0.01, 0.05, 0.10]:
            specifications.append(ModelSpecification(
                model_id=f"epsilon_{epsilon:.2f}",
                prior_type=PriorType.WEAK_INFORMATIVE,
                likelihood_type=LikelihoodType.STUDENT_T,
                structure=ModelStructure.EPSILON_CONTAMINATION,
                epsilon=epsilon,
                description=f"ε-contamination with ε={epsilon}"
            ))
        
        # 2. Traditional hierarchical models
        for prior in [PriorType.WEAK_INFORMATIVE, PriorType.STRONG_INFORMATIVE]:
            for likelihood in [LikelihoodType.NORMAL, LikelihoodType.STUDENT_T]:
                specifications.append(ModelSpecification(
                    model_id=f"hierarchy_{prior.value}_{likelihood.value}",
                    prior_type=prior,
                    likelihood_type=likelihood,
                    structure=ModelStructure.FULL_HIERARCHY,
                    description=f"Full hierarchy with {prior.value} prior and {likelihood.value} likelihood"
                ))
        
        # 3. Robust alternatives
        specifications.extend([
            ModelSpecification(
                model_id="horseshoe_normal",
                prior_type=PriorType.HORSESHOE,
                likelihood_type=LikelihoodType.NORMAL,
                structure=ModelStructure.FULL_HIERARCHY,
                description="Horseshoe prior for sparsity"
            ),
            ModelSpecification(
                model_id="laplace_robust",
                prior_type=PriorType.WEAK_INFORMATIVE,
                likelihood_type=LikelihoodType.LAPLACE,
                structure=ModelStructure.FULL_HIERARCHY,
                description="Laplace likelihood for robustness"
            ),
            ModelSpecification(
                model_id="cauchy_heavy",
                prior_type=PriorType.WEAK_INFORMATIVE,
                likelihood_type=LikelihoodType.CAUCHY,
                structure=ModelStructure.FULL_HIERARCHY,
                description="Cauchy likelihood for heavy tails"
            )
        ])
        
        # 4. Baseline models
        specifications.extend([
            ModelSpecification(
                model_id="pooled_normal",
                prior_type=PriorType.WEAK_INFORMATIVE,
                likelihood_type=LikelihoodType.NORMAL,
                structure=ModelStructure.POOLED,
                description="Simple pooled model (baseline)"
            ),
            ModelSpecification(
                model_id="partial_studentt",
                prior_type=PriorType.WEAK_INFORMATIVE,
                likelihood_type=LikelihoodType.STUDENT_T,
                structure=ModelStructure.PARTIAL_HIERARCHY,
                description="Partial hierarchy with Student-t"
            )
        ])
        
        return specifications
    
    def run_comparison(self, data: np.ndarray, 
                       specifications: Optional[List[ModelSpecification]] = None) -> pd.DataFrame:
        """Run complete model comparison"""
        
        if specifications is None:
            specifications = self.create_model_specifications()
        
        print(f"\n🚀 Running {len(specifications)} models for comparison...")
        print(f"   Estimated time: {len(specifications) * 5}-{len(specifications) * 15} minutes")
        
        for i, spec in enumerate(specifications, 1):
            print(f"\n{'='*80}")
            print(f"Model {i}/{len(specifications)}: {spec.model_id}")
            print(f"Description: {spec.description}")
            print(f"{'='*80}")
            
            try:
                # Build model
                model = self.builder.build_model(data, spec)
                
                # Run MCMC
                trace, exec_time = self.builder.run_mcmc(model, spec)
                
                # Extract diagnostics
                convergence = self.builder.extract_diagnostics(trace)
                
                # Calculate model selection criteria
                with model:
                    dic = float(az.dic(trace).dic)
                    waic = float(az.waic(trace).waic)
                    loo = float(az.loo(trace).loo)
                
                # Store results
                result = ModelResults(
                    model_spec=spec,
                    trace=trace,
                    convergence=convergence,
                    dic=dic,
                    waic=waic,
                    loo=loo,
                    execution_time=exec_time
                )
                
                self.results.append(result)
                
                print(f"\n✅ Model {spec.model_id} completed:")
                print(f"   DIC={dic:.2f}, WAIC={waic:.2f}, LOO={loo:.2f}")
                print(f"   Convergence: {convergence.diagnostics_summary}")
                print(f"   Time: {exec_time:.1f} seconds")
                
            except Exception as e:
                print(f"\n❌ Model {spec.model_id} failed: {e}")
                continue
        
        # Create comparison table
        comparison_df = self.create_comparison_table()
        
        return comparison_df
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comprehensive comparison table"""
        
        rows = []
        for result in self.results:
            rows.append({
                'Model ID': result.model_spec.model_id,
                'Structure': result.model_spec.structure.value,
                'Prior': result.model_spec.prior_type.value,
                'Likelihood': result.model_spec.likelihood_type.value,
                'Epsilon': result.model_spec.epsilon,
                'DIC': result.dic,
                'WAIC': result.waic,
                'LOO': result.loo,
                'Converged': result.convergence.converged,
                'R-hat Max': max(result.convergence.rhat.values()),
                'ESS Min': min(result.convergence.ess.values()),
                'Divergences': result.convergence.divergences,
                'BFMI': result.convergence.bfmi,
                'Time (s)': result.execution_time
            })
        
        df = pd.DataFrame(rows)
        
        # Sort by WAIC (lower is better)
        df = df.sort_values('WAIC')
        
        # Add ranking
        df['Rank'] = range(1, len(df) + 1)
        
        return df
    
    def print_summary(self, df: pd.DataFrame):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE MODEL COMPARISON RESULTS")
        print("="*80)
        
        print("\n🏆 Top 5 Models by WAIC:")
        print(df[['Rank', 'Model ID', 'WAIC', 'DIC', 'LOO', 'Converged']].head())
        
        print("\n📈 Convergence Summary:")
        converged = df['Converged'].sum()
        total = len(df)
        print(f"   Converged: {converged}/{total} ({converged/total*100:.1f}%)")
        
        print("\n🎯 Best Model Analysis:")
        best = df.iloc[0]
        print(f"   Model: {best['Model ID']}")
        print(f"   Structure: {best['Structure']}")
        print(f"   WAIC: {best['WAIC']:.2f}")
        print(f"   DIC: {best['DIC']:.2f}")
        print(f"   Converged: {'✅' if best['Converged'] else '❌'}")
        
        # Check if ε-contamination is best
        epsilon_models = df[df['Structure'] == 'epsilon_contamination']
        if not epsilon_models.empty:
            best_epsilon = epsilon_models.iloc[0]
            if best_epsilon['Model ID'] == best['Model ID']:
                print("\n🎉 ε-Contamination model is the winner!")
            else:
                print(f"\n📊 Best ε-contamination model:")
                print(f"   Rank: {best_epsilon['Rank']}")
                print(f"   WAIC: {best_epsilon['WAIC']:.2f}")
        
        return df

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Comprehensive Model Comparison')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with fewer models')
    parser.add_argument('--max-models', type=int, default=None,
                       help='Maximum number of models to compare')
    parser.add_argument('--unleashed', action='store_true',
                       help='Use maximum cores/chains')
    args = parser.parse_args()
    
    # Load data
    print("\n📂 Loading data...")
    try:
        with open('results/climada_data/climada_complete_data.pkl', 'rb') as f:
            climada_data = pickle.load(f)
        event_losses = climada_data.get('event_losses')
        data = np.array([loss for loss in event_losses if loss > 0]) / 1e6
        print(f"   ✅ Loaded {len(data)} events")
    except:
        # Synthetic data
        np.random.seed(42)
        data = np.concatenate([
            np.random.lognormal(14, 1.2, 80),
            np.random.lognormal(16.5, 1.8, 20)
        ])
        print(f"   🔧 Using synthetic data: {len(data)} events")
    
    # Setup MCMC configuration
    if args.unleashed:
        mcmc_config = get_cpu_optimized_mcmc_config(
            n_cores=HPC_CORES,
            max_cores=HPC_CORES,
            max_chains=min(32, HPC_CORES),
            quick_test=args.quick_test
        )
    else:
        mcmc_config = get_cpu_optimized_mcmc_config(
            quick_test=args.quick_test,
            balanced_mode=True
        )
    
    print(f"\n📊 MCMC Configuration:")
    print(f"   Chains: {mcmc_config['n_chains']}")
    print(f"   Samples: {mcmc_config['n_samples']}")
    print(f"   Cores: {mcmc_config['cores']}")
    
    # Create comparison framework
    comparison = ComprehensiveModelComparison(mcmc_config)
    
    # Get model specifications
    all_specs = comparison.create_model_specifications()
    
    if args.quick_test:
        # Quick test: only key models
        specs = [
            s for s in all_specs 
            if 'epsilon' in s.model_id or 'pooled' in s.model_id or 'hierarchy_weak' in s.model_id
        ][:5]
    elif args.max_models:
        specs = all_specs[:args.max_models]
    else:
        specs = all_specs
    
    print(f"\n🎯 Comparing {len(specs)} models...")
    
    # Run comparison
    start_time = time.time()
    results_df = comparison.run_comparison(data, specs)
    total_time = time.time() - start_time
    
    # Print summary
    comparison.print_summary(results_df)
    
    # Save results
    results_dir = Path('results/comprehensive_model_comparison')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(results_dir / 'comparison_results.csv', index=False)
    
    with open(results_dir / 'full_results.pkl', 'wb') as f:
        pickle.dump({
            'comparison_table': results_df,
            'model_results': comparison.results,
            'execution_time': total_time
        }, f)
    
    print(f"\n💾 Results saved to: {results_dir}")
    print(f"\n⏱️ Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("\n🎉 Comprehensive model comparison complete!")

if __name__ == "__main__":
    main()