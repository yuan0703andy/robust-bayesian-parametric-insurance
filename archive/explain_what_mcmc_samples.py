#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explain_what_mcmc_samples.py
============================
解釋pm.sample()到底在取樣什麼
Explain what pm.sample() is actually sampling

回答您的問題：
"我們不能簡單地將 CRPS 當作一個損失函數丟進 pm.sample() 
我還好奇這個部分 那我們現在 pm.sample() 的是什麼"
"""

import numpy as np
import matplotlib.pyplot as plt

def explain_mcmc_sampling():
    """
    解釋MCMC在採樣什麼
    """
    
    print("🎲 What is pm.sample() Actually Sampling?")
    print("=" * 80)
    
    print("\n📊 YOUR CURRENT MODEL STRUCTURE")
    print("-" * 60)
    model_structure = """
    空間階層貝氏模型 (Spatial Hierarchical Bayesian Model):
    
    β_i = α_r(i) + δ_i + γ_i
    
    Where:
    • β_i: Vulnerability parameter for location i (脆弱度參數)
    • α_r(i): Regional effect for region r containing location i (區域效應)
    • δ_i: Spatial correlation effect (空間相關效應)
    • γ_i: Local random effect (局部隨機效應)
    """
    print(model_structure)
    
    print("\n🎯 WHAT pm.sample() IS SAMPLING")
    print("-" * 60)
    sampling_explanation = """
    pm.sample() is sampling the POSTERIOR DISTRIBUTIONS of:
    
    1. PARAMETERS (參數的後驗分布):
       • α (regional effects): shape = (n_regions,)
       • δ (spatial effects): shape = (n_locations,)
       • γ (local effects): shape = (n_locations,)
       • σ (observation error): scalar
       • ρ (spatial correlation): scalar
       • τ (hierarchical variance): scalar
    
    2. DERIVED QUANTITIES (衍生量):
       • β = α[region_idx] + δ + γ (total vulnerability)
       • Expected losses = exposure × β × f(hazard)
    
    3. POSTERIOR PREDICTIVE (後驗預測):
       • Future loss predictions given new hazard scenarios
    """
    print(sampling_explanation)
    
    print("\n⚠️ WHAT pm.sample() IS NOT SAMPLING")
    print("-" * 60)
    not_sampling = """
    pm.sample() is NOT:
    
    ❌ Using CRPS as a loss function
    ❌ Using basis risk as a loss function
    ❌ Optimizing insurance products
    ❌ Minimizing any decision-theoretic loss
    
    These are SEPARATE processes that happen AFTER sampling!
    """
    print(not_sampling)

def show_likelihood_vs_loss():
    """
    展示Likelihood vs Loss Function的差異
    """
    
    print("\n🔍 LIKELIHOOD vs LOSS FUNCTION")
    print("=" * 80)
    
    print("\n1️⃣ LIKELIHOOD (Used in pm.sample())")
    print("-" * 40)
    likelihood_code = """
    # This is what pm.sample() uses:
    
    with pm.Model() as model:
        # Priors
        α ~ Normal(0, 1)  # Regional effects
        δ ~ CAR(W, τ)     # Spatial effects
        γ ~ Normal(0, σ)  # Local effects
        
        # Vulnerability model
        β = α[region_idx] + δ + γ
        
        # LIKELIHOOD FUNCTION (這是pm.sample()使用的):
        expected_losses = exposure * β * emanuel_usa_function(hazard)
        observed_losses ~ Normal(mu=expected_losses, sigma=σ_obs)
        #                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                 This Normal distribution is the likelihood!
        
        # MCMC sampling
        trace = pm.sample(4000, chains=8)  # Samples α, δ, γ, σ, etc.
    """
    print(likelihood_code)
    
    print("\n2️⃣ LOSS FUNCTION (Used AFTER pm.sample())")
    print("-" * 40)
    loss_function_code = """
    # This happens AFTER getting the posterior:
    
    # Step 1: Get posterior samples from MCMC
    posterior_samples = trace.posterior
    
    # Step 2: Define basis risk loss functions
    basis_risk_functions = {
        'absolute': lambda actual, payout: |actual - payout|,
        'asymmetric': lambda actual, payout: max(0, actual - payout),
        'weighted': lambda actual, payout: w_under*max(0, actual-payout) + 
                                          w_over*max(0, payout-actual)
    }
    
    # Step 3: Optimize products using loss functions
    for risk_type, loss_func in basis_risk_functions.items():
        optimal_product = minimize(
            lambda params: E[loss_func(posterior_losses, payout(params))]
        )
    
    # Note: This optimization uses the posterior FROM pm.sample(),
    # but the loss function is NOT part of pm.sample() itself!
    """
    print(loss_function_code)

def show_concrete_example():
    """
    展示具體例子
    """
    
    print("\n📌 CONCRETE EXAMPLE")
    print("=" * 80)
    
    print("\n🎲 What Your Current pm.sample() Actually Samples:")
    print("-" * 50)
    
    # 模擬採樣結果
    np.random.seed(42)
    n_samples = 4000
    n_chains = 8
    n_regions = 5
    
    # 模擬後驗樣本
    alpha_samples = np.random.normal(0, 0.3, (n_chains, n_samples, n_regions))
    delta_samples = np.random.normal(0, 0.2, (n_chains, n_samples, 100))
    gamma_samples = np.random.normal(0, 0.1, (n_chains, n_samples, 100))
    sigma_samples = np.random.gamma(2, 0.5, (n_chains, n_samples))
    
    print(f"""
    Sampled Parameters (後驗樣本):
    • α (regional effects):  shape={alpha_samples.shape}, mean={alpha_samples.mean():.3f}
    • δ (spatial effects):   shape={delta_samples.shape}, mean={delta_samples.mean():.3f}
    • γ (local effects):     shape={gamma_samples.shape}, mean={gamma_samples.mean():.3f}
    • σ (observation error): shape={sigma_samples.shape}, mean={sigma_samples.mean():.3f}
    
    These samples represent UNCERTAINTY in the model parameters!
    They are NOT optimizing any insurance product.
    """)
    
    print("\n🎯 What SHOULD Happen Next (But Isn't):")
    print("-" * 50)
    next_steps = """
    1. Use posterior samples to generate loss distributions
    2. Define parametric insurance products (trigger, payout)
    3. For each basis risk type:
       - Calculate expected basis risk using posterior
       - Optimize product parameters to minimize risk
    4. Compare optimal products from different risk definitions
    5. Select best product based on criteria
    """
    print(next_steps)

def show_mathematical_difference():
    """
    展示數學上的差異
    """
    
    print("\n📐 MATHEMATICAL DIFFERENCE")
    print("=" * 80)
    
    print("\n1️⃣ MCMC Objective (What pm.sample() maximizes):")
    print("-" * 50)
    mcmc_math = r"""
    Maximize: log p(θ|D) = log p(D|θ) + log p(θ) - log p(D)
                           ^^^^^^^^^^^
                           Likelihood
    
    Where:
    • θ = {α, δ, γ, σ, τ, ρ} are model parameters
    • D = observed_losses are the data
    • p(D|θ) = ∏ Normal(loss_i | f(θ, hazard_i, exposure_i), σ)
    
    This finds the most likely parameter values given the data!
    """
    print(mcmc_math)
    
    print("\n2️⃣ Product Optimization Objective (What SHOULD happen after):")
    print("-" * 50)
    optimization_math = r"""
    Minimize: E_θ|D[BasisRisk(L(θ), P(ξ))]
    
    Where:
    • θ|D ~ Posterior from MCMC (已經從pm.sample()獲得)
    • L(θ) = Loss given parameters θ
    • P(ξ) = Payout given product parameters ξ
    • BasisRisk = One of three definitions (absolute/asymmetric/weighted)
    
    This finds the best insurance product given parameter uncertainty!
    """
    print(optimization_math)
    
    print("\n⚠️ KEY INSIGHT:")
    print("-" * 50)
    key_insight = """
    CRPS is NOT a likelihood function for MCMC!
    
    • CRPS measures prediction skill (用於評估預測技巧)
    • Likelihood measures data fit (用於擬合數據)
    
    You cannot "just throw CRPS into pm.sample()" because:
    1. CRPS needs ensemble predictions vs observations
    2. MCMC needs a proper probability distribution
    3. They serve completely different purposes!
    
    Correct workflow:
    MCMC (get posteriors) → Generate predictions → Calculate CRPS → Compare models
    """
    print(key_insight)

def show_visualization():
    """
    視覺化展示差異
    """
    
    print("\n📊 VISUAL REPRESENTATION")
    print("=" * 80)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. MCMC Sampling
    ax = axes[0, 0]
    np.random.seed(42)
    samples = np.random.normal(0, 1, 1000)
    ax.hist(samples, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='True value')
    ax.set_title('MCMC Sampling: Parameter Posterior\n(What pm.sample() gives you)')
    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # 2. Loss Distribution
    ax = axes[0, 1]
    losses = np.random.lognormal(15, 1, 1000)
    ax.hist(losses/1e6, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.set_title('Posterior Predictive Losses\n(Generated from MCMC posteriors)')
    ax.set_xlabel('Loss (M$)')
    ax.set_ylabel('Frequency')
    
    # 3. Basis Risk Functions
    ax = axes[1, 0]
    actual = np.linspace(0, 100, 100)
    payout = 50
    absolute_risk = np.abs(actual - payout)
    asymmetric_risk = np.maximum(0, actual - payout)
    weighted_risk = 2*np.maximum(0, actual - payout) + 0.5*np.maximum(0, payout - actual)
    
    ax.plot(actual, absolute_risk, label='Absolute', linewidth=2)
    ax.plot(actual, asymmetric_risk, label='Asymmetric', linewidth=2)
    ax.plot(actual, weighted_risk, label='Weighted', linewidth=2)
    ax.axvline(payout, color='red', linestyle='--', alpha=0.5, label='Payout')
    ax.set_title('Basis Risk Functions\n(NOT used in pm.sample())')
    ax.set_xlabel('Actual Loss')
    ax.set_ylabel('Basis Risk')
    ax.legend()
    
    # 4. Optimization Surface
    ax = axes[1, 1]
    triggers = np.linspace(20, 80, 50)
    payouts = np.linspace(10, 90, 50)
    X, Y = np.meshgrid(triggers, payouts)
    Z = np.sqrt((X - 50)**2 + (Y - 45)**2)  # Simplified basis risk surface
    
    contour = ax.contour(X, Y, Z, levels=10)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.plot(50, 45, 'r*', markersize=15, label='Optimal product')
    ax.set_title('Product Optimization\n(Uses posteriors to minimize basis risk)')
    ax.set_xlabel('Trigger Threshold')
    ax.set_ylabel('Payout Amount')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('mcmc_vs_optimization.png', dpi=150, bbox_inches='tight')
    print("\n✅ Visualization saved as 'mcmc_vs_optimization.png'")
    
    plt.show()

def main():
    """主函數"""
    
    print("🎓 EXPLANATION: What pm.sample() Actually Samples")
    print("=" * 80)
    print()
    
    # 解釋MCMC採樣
    explain_mcmc_sampling()
    
    # 展示Likelihood vs Loss
    show_likelihood_vs_loss()
    
    # 具體例子
    show_concrete_example()
    
    # 數學差異
    show_mathematical_difference()
    
    # 視覺化
    show_visualization()
    
    print("\n" + "=" * 80)
    print("💡 SUMMARY ANSWER TO YOUR QUESTION:")
    print("-" * 40)
    summary = """
    Q: "那我們現在 pm.sample() 的是什麼？"
    
    A: pm.sample() is sampling the POSTERIOR DISTRIBUTIONS of your spatial
       hierarchical model parameters (α, δ, γ, σ, τ, ρ).
    
    It uses a Normal likelihood: observed_losses ~ Normal(μ=predictions, σ)
    
    It does NOT use CRPS or basis risk as loss functions!
    
    The correct workflow is:
    1. pm.sample() → Get parameter posteriors (現在在做的)
    2. Use posteriors → Optimize products with basis risk (缺少的部分)
    
    You need to ADD Step 2 to complete your analysis!
    """
    print(summary)

if __name__ == "__main__":
    main()