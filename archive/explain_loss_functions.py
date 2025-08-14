#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explain_loss_functions.py
=========================
解釋Loss Function的兩個不同層次
Explain two different levels of loss functions

說明：
1. MCMC模型擬合的likelihood（不是basis risk）
2. 產品優化的basis risk loss function（三種定義）
"""

def explain_two_levels():
    """
    解釋兩個層次的Loss Function
    """
    
    print("🎯 Understanding Two Levels of Loss Functions")
    print("=" * 70)
    
    print("\n📊 Level 1: MCMC Model Fitting (Likelihood Function)")
    print("-" * 60)
    print("""
    這是用於擬合空間階層貝氏模型的概似函數：
    
    Model: β_i = α_r(i) + δ_i + γ_i
    
    Likelihood: p(observed_losses | β, hazard, exposure)
    
    例如：
    observed_losses ~ Normal(μ = E × β × f(H), σ)
    
    其中：
    • E = exposure values (暴險值)
    • β = vulnerability parameters (脆弱度參數) 
    • f(H) = Emanuel USA function of hazard intensity
    • σ = observation error
    
    ⚠️ 這裡沒有basis risk！這只是擬合觀測數據。
    """)
    
    print("\n🎯 Level 2: Product Optimization (Basis Risk Loss Functions)")
    print("-" * 60)
    print("""
    在獲得後驗分布之後，使用basis risk作為loss function來優化產品：
    
    Given:
    • Posterior samples from MCMC (後驗樣本)
    • Insurance product parameters (保險產品參數)
    
    Optimize: min E[Basis_Risk(actual_losses, payouts)]
    
    Three Definitions of Basis Risk:
    
    1. ABSOLUTE:
       L = |actual_loss - payout|
       
    2. ASYMMETRIC:
       L = max(0, actual_loss - payout)
       
    3. WEIGHTED ASYMMETRIC:
       L = w_under × max(0, actual_loss - payout) + 
           w_over × max(0, payout - actual_loss)
    
    ⚠️ 這些是決策理論的loss function，不是MCMC的likelihood！
    """)

def show_complete_workflow():
    """
    展示完整的工作流程
    """
    
    print("\n🔄 Complete Workflow in Your Framework")
    print("=" * 70)
    
    workflow = """
    📂 Input Data
    ├── CLIMADA objects (tc_hazard, exposure, impact_func_set)
    ├── Observed losses
    └── Wind indices
    
    ⬇️
    
    🧠 Step 1: Spatial Hierarchical Bayesian Model (MCMC)
    ├── Model: β_i = α_r(i) + δ_i + γ_i
    ├── Likelihood: Normal(observed_losses | predictions)
    ├── MCMC sampling: 4000 samples × 8 chains
    └── Output: Posterior distributions of parameters
    
    ⬇️
    
    🎯 Step 2: Bayesian Decision Theory (Product Optimization)
    ├── Input: Posterior samples from Step 1
    ├── Loss Function: THREE basis risk definitions
    │   ├── BasisRiskType.ABSOLUTE
    │   ├── BasisRiskType.ASYMMETRIC
    │   └── BasisRiskType.WEIGHTED_ASYMMETRIC
    ├── Optimization: Find product that minimizes E[Basis_Risk]
    └── Output: Optimal insurance product for each basis risk type
    
    ⬇️
    
    📊 Step 3: Comparison and Selection
    ├── Compare products from three basis risk definitions
    ├── Evaluate performance metrics
    └── Select best product based on criteria
    """
    
    print(workflow)

def show_implementation_in_05():
    """
    顯示在05腳本中應該如何實現
    """
    
    print("\n💡 What's Missing in 05_robust_bayesian_parm_insurance.py")
    print("=" * 70)
    
    print("""
    Current Implementation:
    ✅ Step 1: MCMC sampling (spatial hierarchical model)
    ✅ Step 1: Posterior distributions obtained
    ❌ Step 2: Product optimization with basis risk
    ❌ Step 2: Comparison of three basis risk types
    
    What Should Be Added:
    """)
    
    code_snippet = '''
    # After MCMC sampling completes...
    
    # Initialize BayesianDecisionOptimizer with different basis risk types
    from bayesian import BayesianDecisionOptimizer, OptimizerConfig
    from skill_scores.basis_risk_functions import BasisRiskType
    
    # Define three optimizer configurations
    configs = {
        'absolute': OptimizerConfig(basis_risk_type=BasisRiskType.ABSOLUTE),
        'asymmetric': OptimizerConfig(basis_risk_type=BasisRiskType.ASYMMETRIC),
        'weighted': OptimizerConfig(
            basis_risk_type=BasisRiskType.WEIGHTED_ASYMMETRIC,
            w_under=2.0,
            w_over=0.5
        )
    }
    
    # Optimize products for each basis risk type
    optimal_products = {}
    for name, config in configs.items():
        optimizer = BayesianDecisionOptimizer(config)
        result = optimizer.optimize_expected_risk(
            posterior_samples=hierarchical_results.posterior_samples,
            hazard_indices=wind_indices,
            actual_losses=observed_losses,
            product_space=product_space
        )
        optimal_products[name] = result
        print(f"{name}: Risk={result.expected_risk:.2e}")
    
    # Compare the three optimal products
    comparison_results = compare_basis_risk_products(optimal_products)
    '''
    
    print(code_snippet)

def main():
    """主函數"""
    
    print("🎓 EXPLANATION: Loss Functions in Bayesian Framework")
    print("=" * 70)
    print()
    
    # 解釋兩個層次
    explain_two_levels()
    
    # 展示完整流程
    show_complete_workflow()
    
    # 顯示實現建議
    show_implementation_in_05()
    
    print("\n" + "=" * 70)
    print("📌 KEY INSIGHT:")
    print("""
    Basis Risk is NOT part of the MCMC model likelihood!
    
    • MCMC: Fits model to observed data (uses likelihood)
    • Decision Theory: Optimizes products (uses basis risk as loss)
    
    These are TWO SEPARATE stages that work together:
    1. First: Get posterior distributions via MCMC
    2. Then: Use posteriors to optimize products with different basis risks
    
    You can test all three basis risk types using THE SAME posterior samples!
    No need to run MCMC three times.
    """)

if __name__ == "__main__":
    main()