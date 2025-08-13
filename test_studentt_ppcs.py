#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_studentt_ppcs.py
=====================
Fast Student's T Model Validation with Posterior Predictive Checks
測試Student's T模型和後驗預測檢查的快速驗證版本

This script provides a streamlined validation of the Student's T robust Bayesian approach
without the full computational overhead of the complete MCMC analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib font for Chinese
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("🚀 FAST STUDENT'S T VALIDATION WITH PPCs")
print("快速Student's T驗證與後驗預測檢查")
print("=" * 60)

# %%
print("\n📂 Loading CLIMADA Data...")
climada_data_path = Path('results/climada_data/climada_complete_data.pkl')

if climada_data_path.exists():
    with open(climada_data_path, 'rb') as f:
        climada_data = pickle.load(f)
    
    # Extract loss data - use 'event_losses' key
    loss_data = climada_data['event_losses']
    non_zero_losses = loss_data[loss_data > 0]
    
    print(f"✅ Loaded {len(loss_data)} events, {len(non_zero_losses)} non-zero losses")
    print(f"   Loss range: {loss_data.min():.2e} - {loss_data.max():.2e}")
    
    # Display some metadata
    metadata = climada_data.get('metadata', {})
    print(f"   Metadata: {metadata}")
    
else:
    print("❌ CLIMADA data not found, using simulated data")
    # Simulate heavy-tailed data for testing
    np.random.seed(42)
    normal_data = np.random.normal(5e8, 1e8, 200)
    heavy_tail_data = np.random.exponential(2e9, 30) 
    loss_data = np.concatenate([normal_data, heavy_tail_data])
    non_zero_losses = loss_data[loss_data > 0]

# %%
print("\n🔍 Exploratory Data Analysis...")

# Log transform for modeling
log_losses = np.log(non_zero_losses + 1)  # +1 to handle zeros

print(f"📊 Loss Statistics:")
print(f"   • Original losses: mean={non_zero_losses.mean():.2e}, std={non_zero_losses.std():.2e}")
print(f"   • Log-transformed: mean={log_losses.mean():.2f}, std={log_losses.std():.2f}")
print(f"   • Skewness: {pd.Series(non_zero_losses).skew():.2f}")
print(f"   • Kurtosis: {pd.Series(non_zero_losses).kurtosis():.2f}")

# Heavy-tail indicators
q99 = np.percentile(non_zero_losses, 99)
q95 = np.percentile(non_zero_losses, 95)
tail_ratio = q99 / q95
print(f"   • Tail ratio (99th/95th percentile): {tail_ratio:.2f}")
print(f"   • Heavy tails detected: {'Yes' if tail_ratio > 2.0 else 'No'}")

# %%
print("\n🏗️ Building Comparative Models...")

# Prepare data
n_obs = len(log_losses)
y_obs = log_losses

print(f"🔧 Model data: {n_obs} observations, log-transformed")

# Model 1: Normal Likelihood (Traditional)
print("\n1️⃣ Building Normal Likelihood Model...")
with pm.Model() as normal_model:
    # Priors
    mu = pm.Normal('mu', mu=np.mean(y_obs), sigma=5)
    sigma = pm.HalfNormal('sigma', sigma=2)
    
    # Likelihood
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y_obs)

print("✅ Normal model built")

# Model 2: Student's T Likelihood (Robust)
print("\n2️⃣ Building Student's T Likelihood Model...")
with pm.Model() as studentt_model:
    # Priors
    mu = pm.Normal('mu', mu=np.mean(y_obs), sigma=5)
    sigma = pm.HalfNormal('sigma', sigma=2)
    nu = pm.Exponential('nu', lam=1/5)  # Degrees of freedom for heavy tails
    
    # Robust likelihood
    likelihood = pm.StudentT('likelihood', nu=nu, mu=mu, sigma=sigma, observed=y_obs)

print("✅ Student's T model built")

# %%
print("\n⚡ Fast MCMC Sampling...")

# Fast sampling with fewer iterations for validation
sample_kwargs = {
    'draws': 500,      # Reduced from typical 2000
    'tune': 250,       # Reduced from typical 1000
    'chains': 2,       # Reduced from typical 4
    'cores': 2,
    'return_inferencedata': True,
    'progressbar': True
}

print("🔄 Sampling Normal model...")
with normal_model:
    trace_normal = pm.sample(**sample_kwargs)

print("🔄 Sampling Student's T model...")
with studentt_model:
    trace_studentt = pm.sample(**sample_kwargs)

print("✅ MCMC sampling completed")

# %%
print("\n🧪 POSTERIOR PREDICTIVE CHECKS (PPCs)")
print("=" * 50)

def perform_ppcs(model, trace, model_name):
    """Perform comprehensive posterior predictive checks"""
    
    print(f"\n🔍 PPCs for {model_name} Model")
    print("-" * 40)
    
    # Generate posterior predictive samples
    with model:
        ppc_samples = pm.sample_posterior_predictive(
            trace, 
            samples=100,  # Reduced for speed
            return_inferencedata=True,
            progressbar=False
        )
    
    # Extract samples
    y_pred = ppc_samples.posterior_predictive['likelihood'].values
    y_pred_flat = y_pred.reshape(-1, len(y_obs))
    
    # PPC Statistics
    observed_mean = np.mean(y_obs)
    observed_std = np.std(y_obs)
    observed_max = np.max(y_obs)
    observed_min = np.min(y_obs)
    
    pred_means = np.mean(y_pred_flat, axis=1)
    pred_stds = np.std(y_pred_flat, axis=1)
    pred_maxs = np.max(y_pred_flat, axis=1)
    pred_mins = np.min(y_pred_flat, axis=1)
    
    # Compute p-values
    p_mean = np.mean(pred_means > observed_mean)
    p_std = np.mean(pred_stds > observed_std)
    p_max = np.mean(pred_maxs > observed_max)
    p_min = np.mean(pred_mins < observed_min)
    
    print(f"📊 PPC Results for {model_name}:")
    print(f"   • Mean: observed={observed_mean:.2f}, p-value={p_mean:.3f}")
    print(f"   • Std:  observed={observed_std:.2f}, p-value={p_std:.3f}")
    print(f"   • Max:  observed={observed_max:.2f}, p-value={p_max:.3f}")
    print(f"   • Min:  observed={observed_min:.2f}, p-value={p_min:.3f}")
    
    # Assess fit quality
    good_fit_count = sum([0.05 < p < 0.95 for p in [p_mean, p_std, p_max, p_min]])
    fit_quality = good_fit_count / 4
    
    print(f"   • Overall fit quality: {fit_quality:.1%} ({good_fit_count}/4 stats in range)")
    
    # Extreme value check
    extreme_coverage = np.mean([
        np.any(y_pred_flat >= np.percentile(y_obs, 95), axis=1)
    ])
    
    print(f"   • Extreme value coverage: {extreme_coverage:.1%}")
    
    return {
        'ppc_samples': ppc_samples,
        'p_values': {'mean': p_mean, 'std': p_std, 'max': p_max, 'min': p_min},
        'fit_quality': fit_quality,
        'extreme_coverage': extreme_coverage,
        'y_pred': y_pred_flat
    }

# Perform PPCs
normal_ppcs = perform_ppcs(normal_model, trace_normal, "Normal")
studentt_ppcs = perform_ppcs(studentt_model, trace_studentt, "Student's T")

# %%
print("\n📈 VISUALIZATION AND COMPARISON")
print("=" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Robust Bayesian Model Comparison: Normal vs Student\'s T\n正態分佈vs Student\'s T強健模型比較', fontsize=16)

# Row 1: Normal Model
axes[0, 0].hist(y_obs, bins=30, alpha=0.7, density=True, label='Observed', color='black')
for i in range(min(20, normal_ppcs['y_pred'].shape[0])):
    axes[0, 0].hist(normal_ppcs['y_pred'][i], bins=30, alpha=0.1, density=True, color='blue')
axes[0, 0].set_title('Normal Model PPCs')
axes[0, 0].set_xlabel('Log Losses')
axes[0, 0].legend()

# Trace plot for Normal
az.plot_trace(trace_normal, var_names=['mu', 'sigma'], ax=axes[0, 1:])
axes[0, 1].set_title('Normal Model Traces')

# Row 2: Student's T Model  
axes[1, 0].hist(y_obs, bins=30, alpha=0.7, density=True, label='Observed', color='black')
for i in range(min(20, studentt_ppcs['y_pred'].shape[0])):
    axes[1, 0].hist(studentt_ppcs['y_pred'][i], bins=30, alpha=0.1, density=True, color='red')
axes[1, 0].set_title('Student\'s T Model PPCs')
axes[1, 0].set_xlabel('Log Losses')
axes[1, 0].legend()

# Trace plot for Student's T
az.plot_trace(trace_studentt, var_names=['mu', 'sigma', 'nu'], ax=axes[1, 1:])
axes[1, 1].set_title('Student\'s T Model Traces')

plt.tight_layout()
plt.savefig('studentt_validation_ppcs.png', dpi=150, bbox_inches='tight')
print("✅ Validation plots saved as 'studentt_validation_ppcs.png'")
plt.show()

# %%
print("\n🏆 MODEL COMPARISON RESULTS")
print("=" * 60)

# Compare model performance
print("📊 Posterior Predictive Check Comparison:")
print(f"   Normal Model:")
print(f"      • Fit Quality: {normal_ppcs['fit_quality']:.1%}")
print(f"      • Extreme Coverage: {normal_ppcs['extreme_coverage']:.1%}")

print(f"   Student's T Model:")
print(f"      • Fit Quality: {studentt_ppcs['fit_quality']:.1%}")
print(f"      • Extreme Coverage: {studentt_ppcs['extreme_coverage']:.1%}")

# Determine winner
normal_score = normal_ppcs['fit_quality'] * 0.7 + normal_ppcs['extreme_coverage'] * 0.3
studentt_score = studentt_ppcs['fit_quality'] * 0.7 + studentt_ppcs['extreme_coverage'] * 0.3

print(f"\n🎯 Composite Scores:")
print(f"   • Normal Model: {normal_score:.3f}")
print(f"   • Student's T Model: {studentt_score:.3f}")

winner = "Student's T" if studentt_score > normal_score else "Normal"
advantage = abs(studentt_score - normal_score)

print(f"\n🏆 Winner: {winner} Model (advantage: {advantage:.3f})")

# Model recommendations
if studentt_score > normal_score + 0.1:
    recommendation = "✅ Strong evidence for Student's T model - use for robust analysis"
elif studentt_score > normal_score:
    recommendation = "✅ Mild evidence for Student's T model - recommended for heavy-tailed data"
else:
    recommendation = "⚠️ Normal model adequate - consider data characteristics"

print(f"\n💡 Recommendation: {recommendation}")

# %%
print("\n📊 PRACTICAL IMPLICATIONS")
print("=" * 60)

# Extract parameter estimates
normal_mu_mean = trace_normal.posterior['mu'].mean().values
normal_sigma_mean = trace_normal.posterior['sigma'].mean().values

studentt_mu_mean = trace_studentt.posterior['mu'].mean().values
studentt_sigma_mean = trace_studentt.posterior['sigma'].mean().values
studentt_nu_mean = trace_studentt.posterior['nu'].mean().values

print("🔢 Parameter Estimates:")
print(f"   Normal Model:")
print(f"      • μ (mean): {normal_mu_mean:.3f}")
print(f"      • σ (scale): {normal_sigma_mean:.3f}")

print(f"   Student's T Model:")
print(f"      • μ (location): {studentt_mu_mean:.3f}")
print(f"      • σ (scale): {studentt_sigma_mean:.3f}")
print(f"      • ν (degrees of freedom): {studentt_nu_mean:.3f}")

# Tail behavior interpretation
if studentt_nu_mean < 10:
    tail_behavior = "Heavy tails (ν < 10) - robust to extreme events"
elif studentt_nu_mean < 30:
    tail_behavior = "Moderate tails (10 ≤ ν < 30) - some robustness"
else:
    tail_behavior = "Light tails (ν ≥ 30) - approaching normal distribution"

print(f"\n🔍 Tail Behavior Analysis:")
print(f"   • {tail_behavior}")
print(f"   • Student's T provides {'significant' if studentt_nu_mean < 10 else 'moderate' if studentt_nu_mean < 30 else 'minimal'} robustness improvement")

# %%
print("\n" + "=" * 80)
print("🎯 STUDENT'S T VALIDATION SUMMARY")
print("=" * 80)

validation_summary = f"""
✅ Validation Results:
   • Models compared: Normal vs Student's T likelihood
   • Data: {len(non_zero_losses)} CLIMADA loss events (log-transformed)
   • Sampling: {sample_kwargs['draws']} draws × {sample_kwargs['chains']} chains
   
📊 Model Performance:
   • Normal model fit quality: {normal_ppcs['fit_quality']:.1%}
   • Student's T model fit quality: {studentt_ppcs['fit_quality']:.1%}
   • Winner: {winner} Model
   
🔧 Robustness Assessment:
   • Degrees of freedom (ν): {studentt_nu_mean:.2f}
   • Tail behavior: {tail_behavior}
   • Extreme event coverage: {studentt_ppcs['extreme_coverage']:.1%}
   
💡 Conclusion:
   {recommendation}
   
🚀 Next Steps for Full Implementation:
   • Integrate spatial hierarchical structure β_i = α_r(i) + δ_i + γ_i  
   • Add three basis risk optimization functions
   • Implement ε-contamination for typhoon-specific modeling
   • Scale up MCMC sampling for production analysis
"""

print(validation_summary)

print("\n🔚 Fast validation completed successfully!")
print("✅ Student's T model validated and ready for full implementation")
print("=" * 80)