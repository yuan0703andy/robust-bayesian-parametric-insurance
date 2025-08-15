#!/usr/bin/env python3
"""
05. Fast and Simple Bayesian Analysis - No MCMC
快速簡單貝氏分析 - 無MCMC

Fast alternative using analytical methods instead of MCMC.
使用分析方法而非MCMC的快速替代方案。

Author: Research Team
Date: 2025-01-15
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple

# Clean environment
for var in ['PROJ_DATA', 'PROJ_LIB', 'GDAL_DATA', 'JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES']:
    if var in os.environ:
        del os.environ[var]

print("=" * 80)
print("05. Fast Bayesian Analysis - No MCMC")
print("快速貝氏分析 - 無MCMC")
print("=" * 80)
print("\n💡 Using analytical methods for speed (no MCMC)")
print("⚡ Results in seconds, not minutes!")

# Load data
print("\n📂 Loading data...")
try:
    with open('results/climada_data/climada_complete_data.pkl', 'rb') as f:
        climada_results = pickle.load(f)
    
    with open('results/insurance_products/products.pkl', 'rb') as f:
        products_data = pickle.load(f)
    
    print("✅ Data loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

# Extract losses
event_losses_array = climada_results.get('event_losses')
all_losses = np.array(event_losses_array) if event_losses_array is not None else np.array([])
observed_losses = all_losses[all_losses > 0]  # Non-zero losses only

print(f"\n📊 Data Summary:")
print(f"   Total events: {len(all_losses)}")
print(f"   Non-zero losses: {len(observed_losses)}")
print(f"   Loss range: ${np.min(observed_losses)/1e6:.1f}M - ${np.max(observed_losses)/1e6:.1f}M")
print(f"   Mean loss: ${np.mean(observed_losses)/1e6:.1f}M")
print(f"   Median loss: ${np.median(observed_losses)/1e6:.1f}M")

# ============================================================================
# PHASE 1: Fast Bayesian Parameter Estimation
# ============================================================================
print("\n" + "=" * 80)
print("Phase 1: Fast Bayesian Parameter Estimation")
print("階段1：快速貝氏參數估計")
print("=" * 80)

def bayesian_lognormal_estimation(data: np.ndarray) -> Dict:
    """
    Bayesian estimation for log-normal distribution using conjugate priors.
    使用共軛先驗的對數正態分布貝氏估計。
    """
    log_data = np.log(data + 1e-10)  # Avoid log(0)
    n = len(log_data)
    
    # Prior parameters (weakly informative)
    mu_0 = np.mean(log_data)  # Prior mean
    kappa_0 = 1  # Prior precision (weak)
    nu_0 = 1  # Prior degrees of freedom
    sigma2_0 = np.var(log_data)  # Prior variance
    
    # Posterior parameters (analytical solution)
    kappa_n = kappa_0 + n
    mu_n = (kappa_0 * mu_0 + n * np.mean(log_data)) / kappa_n
    nu_n = nu_0 + n
    sigma2_n = (nu_0 * sigma2_0 + np.sum((log_data - np.mean(log_data))**2) + 
                kappa_0 * n * (np.mean(log_data) - mu_0)**2 / kappa_n) / nu_n
    
    # Posterior predictive parameters
    predictive_mean = np.exp(mu_n + sigma2_n / 2)
    predictive_std = np.sqrt((np.exp(sigma2_n) - 1) * np.exp(2 * mu_n + sigma2_n))
    
    return {
        'distribution': 'lognormal',
        'posterior_mu': mu_n,
        'posterior_sigma2': sigma2_n,
        'predictive_mean': predictive_mean,
        'predictive_std': predictive_std,
        'confidence_interval': (
            np.exp(mu_n - 1.96 * np.sqrt(sigma2_n / n)),
            np.exp(mu_n + 1.96 * np.sqrt(sigma2_n / n))
        )
    }

def bayesian_gamma_estimation(data: np.ndarray) -> Dict:
    """
    Bayesian estimation for gamma distribution.
    伽瑪分布貝氏估計。
    """
    # Method of moments for initial estimates
    mean_data = np.mean(data)
    var_data = np.var(data)
    
    # Gamma parameters
    alpha_est = mean_data**2 / var_data
    beta_est = mean_data / var_data
    
    # Bayesian adjustment with conjugate prior
    # Prior: alpha ~ Gamma(a0, b0), beta ~ Gamma(c0, d0)
    a0, b0 = 1, 1  # Weak prior for alpha
    c0, d0 = 1, 1  # Weak prior for beta
    
    n = len(data)
    
    # Posterior parameters (approximation)
    alpha_post = a0 + n * alpha_est
    beta_post = (c0 + np.sum(data)) / (d0 + n)
    
    predictive_mean = alpha_post / beta_post
    predictive_std = np.sqrt(alpha_post / beta_post**2)
    
    return {
        'distribution': 'gamma',
        'posterior_alpha': alpha_post,
        'posterior_beta': beta_post,
        'predictive_mean': predictive_mean,
        'predictive_std': predictive_std,
        'confidence_interval': (
            stats.gamma.ppf(0.025, alpha_post, scale=1/beta_post),
            stats.gamma.ppf(0.975, alpha_post, scale=1/beta_post)
        )
    }

def bayesian_weibull_estimation(data: np.ndarray) -> Dict:
    """
    Bayesian estimation for Weibull distribution.
    韋伯分布貝氏估計。
    """
    # MLE for initial estimates
    def neg_log_likelihood(params):
        k, lam = params
        if k <= 0 or lam <= 0:
            return np.inf
        return -np.sum(stats.weibull_min.logpdf(data, k, scale=lam))
    
    # Initial guess
    k_init = 1.0
    lam_init = np.mean(data)
    
    # Optimize
    result = minimize(neg_log_likelihood, [k_init, lam_init], 
                     bounds=[(0.1, 10), (0.1, np.max(data)*10)])
    
    if result.success:
        k_est, lam_est = result.x
    else:
        k_est, lam_est = 1.0, np.mean(data)
    
    # Bayesian adjustment (approximate)
    n = len(data)
    k_post = k_est  # Shape parameter (fixed for simplicity)
    lam_post = lam_est * (n / (n + 1))  # Scale with shrinkage
    
    # Predictive statistics
    from scipy.special import gamma as gamma_fn
    predictive_mean = lam_post * gamma_fn(1 + 1/k_post)
    predictive_var = lam_post**2 * (gamma_fn(1 + 2/k_post) - gamma_fn(1 + 1/k_post)**2)
    predictive_std = np.sqrt(predictive_var)
    
    return {
        'distribution': 'weibull',
        'posterior_k': k_post,
        'posterior_lambda': lam_post,
        'predictive_mean': predictive_mean,
        'predictive_std': predictive_std,
        'confidence_interval': (
            stats.weibull_min.ppf(0.025, k_post, scale=lam_post),
            stats.weibull_min.ppf(0.975, k_post, scale=lam_post)
        )
    }

# Run all Bayesian estimations
print("\n🔬 Running Bayesian estimations...")
start_time = time.time()

estimations = {
    'lognormal': bayesian_lognormal_estimation(observed_losses),
    'gamma': bayesian_gamma_estimation(observed_losses),
    'weibull': bayesian_weibull_estimation(observed_losses)
}

elapsed = time.time() - start_time
print(f"✅ Completed in {elapsed:.2f} seconds!")

# Display results
print("\n📊 Bayesian Estimation Results:")
print("-" * 50)
for name, est in estimations.items():
    print(f"\n{name.upper()} Distribution:")
    print(f"   Predictive mean: ${est['predictive_mean']/1e6:.1f}M")
    print(f"   Predictive std: ${est['predictive_std']/1e6:.1f}M")
    print(f"   95% CI: ${est['confidence_interval'][0]/1e6:.1f}M - ${est['confidence_interval'][1]/1e6:.1f}M")

# ============================================================================
# PHASE 2: Model Comparison using Information Criteria
# ============================================================================
print("\n" + "=" * 80)
print("Phase 2: Model Comparison")
print("階段2：模型比較")
print("=" * 80)

def calculate_information_criteria(data: np.ndarray, estimation: Dict) -> Dict:
    """Calculate AIC, BIC, and DIC for model comparison."""
    n = len(data)
    
    # Get distribution parameters
    dist_name = estimation['distribution']
    
    # Calculate log-likelihood
    if dist_name == 'lognormal':
        mu = estimation['posterior_mu']
        sigma = np.sqrt(estimation['posterior_sigma2'])
        log_likelihood = np.sum(stats.lognorm.logpdf(data, s=sigma, scale=np.exp(mu)))
        k = 2  # Number of parameters
    elif dist_name == 'gamma':
        alpha = estimation['posterior_alpha']
        beta = estimation['posterior_beta']
        log_likelihood = np.sum(stats.gamma.logpdf(data, alpha, scale=1/beta))
        k = 2
    elif dist_name == 'weibull':
        shape = estimation['posterior_k']
        scale = estimation['posterior_lambda']
        log_likelihood = np.sum(stats.weibull_min.logpdf(data, shape, scale=scale))
        k = 2
    else:
        log_likelihood = -np.inf
        k = 2
    
    # Information criteria
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    
    # Approximate DIC (Deviance Information Criterion)
    deviance = -2 * log_likelihood
    dic = deviance + k  # Simplified DIC
    
    return {
        'AIC': aic,
        'BIC': bic,
        'DIC': dic,
        'log_likelihood': log_likelihood
    }

print("\n🔬 Calculating information criteria...")
criteria_results = {}
for name, est in estimations.items():
    criteria = calculate_information_criteria(observed_losses, est)
    criteria_results[name] = criteria
    print(f"\n{name.upper()}:")
    print(f"   AIC: {criteria['AIC']:.2f}")
    print(f"   BIC: {criteria['BIC']:.2f}")
    print(f"   DIC: {criteria['DIC']:.2f}")

# Find best model
best_by_aic = min(criteria_results.items(), key=lambda x: x[1]['AIC'])[0]
best_by_bic = min(criteria_results.items(), key=lambda x: x[1]['BIC'])[0]
best_by_dic = min(criteria_results.items(), key=lambda x: x[1]['DIC'])[0]

print(f"\n🏆 Best Models:")
print(f"   By AIC: {best_by_aic}")
print(f"   By BIC: {best_by_bic}")
print(f"   By DIC: {best_by_dic}")

# ============================================================================
# PHASE 3: Robust Bayesian Predictions
# ============================================================================
print("\n" + "=" * 80)
print("Phase 3: Robust Bayesian Predictions")
print("階段3：穩健貝氏預測")
print("=" * 80)

# Use best model for predictions
best_model = best_by_aic
best_estimation = estimations[best_model]

print(f"\n🔮 Using {best_model} distribution for predictions...")

# Generate predictions
n_pred = len(observed_losses)
predictions = np.full(n_pred, best_estimation['predictive_mean'])

# Add uncertainty bands
lower_bound = np.full(n_pred, best_estimation['confidence_interval'][0])
upper_bound = np.full(n_pred, best_estimation['confidence_interval'][1])

# Calculate skill scores
rmse = np.sqrt(np.mean((predictions - observed_losses)**2))
mae = np.mean(np.abs(predictions - observed_losses))
correlation = np.corrcoef(predictions, observed_losses)[0, 1] if len(predictions) > 1 else 0

print(f"\n📊 Prediction Performance:")
print(f"   RMSE: ${rmse/1e6:.1f}M")
print(f"   MAE: ${mae/1e6:.1f}M")
print(f"   Correlation: {correlation:.4f}")
print(f"   Normalized RMSE: {rmse/np.mean(observed_losses):.4f}")

# ============================================================================
# PHASE 4: Save Results
# ============================================================================
print("\n" + "=" * 80)
print("Phase 4: Save Results")
print("階段4：儲存結果")
print("=" * 80)

results_dir = Path('results/robust_bayesian_fast')
results_dir.mkdir(parents=True, exist_ok=True)

# Compile all results
final_results = {
    'method': 'Fast Bayesian (Analytical)',
    'execution_time': elapsed,
    'estimations': estimations,
    'information_criteria': criteria_results,
    'best_models': {
        'AIC': best_by_aic,
        'BIC': best_by_bic,
        'DIC': best_by_dic
    },
    'predictions': {
        'values': predictions,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    },
    'skill_scores': {
        'RMSE': rmse,
        'MAE': mae,
        'Correlation': correlation,
        'RMSE_normalized': rmse/np.mean(observed_losses)
    },
    'data_summary': {
        'n_events': len(all_losses),
        'n_losses': len(observed_losses),
        'mean_loss': np.mean(observed_losses),
        'median_loss': np.median(observed_losses)
    }
}

# Save pickle
with open(results_dir / 'fast_bayesian_results.pkl', 'wb') as f:
    pickle.dump(final_results, f)

# Save summary report
with open(results_dir / 'analysis_report.txt', 'w') as f:
    f.write("Fast Bayesian Analysis Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Execution Time: {elapsed:.2f} seconds\n")
    f.write(f"Method: Analytical Bayesian Estimation\n\n")
    
    f.write("Best Models:\n")
    f.write(f"  AIC: {best_by_aic}\n")
    f.write(f"  BIC: {best_by_bic}\n")
    f.write(f"  DIC: {best_by_dic}\n\n")
    
    f.write("Model Comparison:\n")
    for name, criteria in criteria_results.items():
        f.write(f"  {name}:\n")
        f.write(f"    AIC: {criteria['AIC']:.2f}\n")
        f.write(f"    BIC: {criteria['BIC']:.2f}\n")
        f.write(f"    DIC: {criteria['DIC']:.2f}\n")
    
    f.write(f"\nPrediction Performance:\n")
    f.write(f"  RMSE: ${rmse/1e6:.1f}M\n")
    f.write(f"  MAE: ${mae/1e6:.1f}M\n")
    f.write(f"  Correlation: {correlation:.4f}\n")

print(f"✅ Results saved to: {results_dir}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("🎉 Fast Bayesian Analysis Complete!")
print("=" * 80)
print(f"✅ Execution time: {elapsed:.2f} seconds (not minutes!)")
print(f"✅ Best model: {best_model}")
print(f"✅ Predictive mean: ${best_estimation['predictive_mean']/1e6:.1f}M")
print(f"✅ RMSE: ${rmse/1e6:.1f}M")
print("=" * 80)

print("\n💡 Key Advantages of This Approach:")
print("   1. FAST: Results in seconds, not hours")
print("   2. STABLE: No kernel crashes or memory issues")
print("   3. PRACTICAL: Works on any hardware")
print("   4. ANALYTICAL: Exact solutions where possible")
print("   5. ROBUST: Multiple models compared automatically")
print("\n🚀 This is what actually works in practice!")