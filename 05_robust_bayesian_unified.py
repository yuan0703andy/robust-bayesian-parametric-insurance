#!/usr/bin/env python3
"""
05. Unified Robust Bayesian Parametric Insurance Analysis
統一穩健貝氏參數型保險分析 - 完整框架整合版

This unified version combines:
- Full GPU acceleration support (auto-detect dual/single/CPU)
- Complete insurance_analysis_refactored framework integration
- Modular Bayesian uncertainty quantification
- Command-line flexibility

NO DUPLICATE IMPLEMENTATIONS - All functionality from existing frameworks

Author: Research Team
Date: 2025-01-14
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

# %%
# ============================================================================
# PART 1: Environment Configuration
# ============================================================================
print("=" * 80)
print("05. Unified Robust Bayesian Parametric Insurance Analysis")
print("統一穩健貝氏參數型保險分析")
print("=" * 80)

def parse_arguments():
    """Parse command-line arguments for flexible deployment"""
    parser = argparse.ArgumentParser(
        description='Unified Robust Bayesian Analysis with GPU Support'
    )
    parser.add_argument('--enable-gpu', action='store_true',
                       help='Enable GPU acceleration if available')
    parser.add_argument('--force-dual-gpu', action='store_true',
                       help='Force dual-GPU mode (requires 2 GPUs)')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Force CPU-only execution')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-detect and use best configuration (default)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run with minimal samples for testing')
    parser.add_argument('--output-dir', type=str, 
                       default='results/unified_bayesian_analysis',
                       help='Output directory for results')
    return parser.parse_args()

# Parse arguments
args = parse_arguments()

# Configure environment based on arguments
if not args.cpu_only:
    print("\n🔍 Detecting hardware capabilities...")
    try:
        from bayesian.gpu_setup import GPUConfig, setup_gpu_environment
        
        # Setup GPU environment
        gpu_config = setup_gpu_environment(enable_gpu=(args.enable_gpu or args.auto))
        gpu_config.print_performance_summary()
        
        # Get optimized MCMC configuration
        mcmc_kwargs = gpu_config.get_pymc_sampler_kwargs()
        hardware_level = gpu_config.hardware_level
        
        if args.force_dual_gpu and hardware_level != "dual_gpu":
            print("⚠️ Dual-GPU requested but not available, using best available")
            
    except ImportError:
        print("⚠️ GPU setup module not available, using CPU")
        hardware_level = "cpu_only"
        mcmc_kwargs = None
else:
    print("💻 CPU-only mode requested")
    hardware_level = "cpu_only"
    mcmc_kwargs = None

# Fallback CPU configuration
if mcmc_kwargs is None:
    mcmc_kwargs = {
        "draws": 500 if args.quick_test else 2000,
        "tune": 250 if args.quick_test else 1000,
        "chains": 2 if args.quick_test else 4,
        "cores": 4,
        "target_accept": 0.90,
        "return_inferencedata": True
    }
    print(f"💻 Using CPU configuration: {mcmc_kwargs['chains']} chains × {mcmc_kwargs['draws']} samples")

# %%
# ============================================================================
# PART 2: Import Framework Components (NO DUPLICATES)
# ============================================================================
print("\n📦 Loading unified framework components...")

# Configuration
from config.settings import (
    NC_BOUNDS, YEAR_RANGE, RESOLUTION,
    IMPACT_FUNC_PARAMS, EXPOSURE_PARAMS
)

# Insurance Analysis Framework - COMPLETE FRAMEWORK
from insurance_analysis_refactored.core import (
    # Core engines
    ParametricInsuranceEngine,
    SkillScoreEvaluator,
    InsuranceProductManager,
    
    # Premium and market analysis
    TechnicalPremiumCalculator,
    MarketAcceptabilityAnalyzer,
    MultiObjectiveOptimizer,
    
    # Input adapters for flexibility
    InputAdapter,
    CLIMADAInputAdapter,
    BayesianInputAdapter,
    
    # Product generation
    ParametricProduct,
    PayoutFunctionType,
    ParametricIndexType
)

# Specialized imports (not in __init__.py)
from insurance_analysis_refactored.core.enhanced_spatial_analysis import EnhancedCatInCircleAnalyzer
from insurance_analysis_refactored.core.saffir_simpson_products import (
    generate_steinmann_2023_products,
    validate_steinmann_compatibility
)

# Bayesian Uncertainty Framework
from bayesian import (
    # Hierarchical models
    ParametricHierarchicalModel,
    ModelSpec,
    MCMCConfig,
    LikelihoodFamily,
    PriorScenario,
    
    # Robust methods
    MixedPredictiveEstimation,
    ModelClassAnalyzer,
    RobustCredibleIntervalCalculator,
    BayesianDecisionOptimizer,
    
    # Contamination models
    EpsilonContaminationClass,
    create_typhoon_contamination_spec,
    
    # Environment setup
    configure_pymc_environment
)

print("✅ All framework components loaded - NO DUPLICATES")

# %%
# ============================================================================
# PART 3: Data Loading
# ============================================================================
print("\n📂 Loading analysis data...")

data_files = {
    'climada': 'climada_complete_data.pkl',
    'spatial': 'results/spatial_analysis/cat_in_circle_results.pkl',
    'products': 'results/insurance_products/products.pkl',
    'traditional': 'results/traditional_basis_risk_analysis/analysis_results.pkl'
}

loaded_data = {}
for key, filepath in data_files.items():
    try:
        file_path = Path(filepath) if not filepath.startswith('results/') else Path(filepath)
        if file_path.exists():
            with open(file_path, 'rb') as f:
                loaded_data[key] = pickle.load(f)
            print(f"   ✅ {key.capitalize()} data loaded")
        else:
            print(f"   ⚠️ {key.capitalize()} data not found: {file_path}")
    except Exception as e:
        print(f"   ❌ Error loading {key}: {e}")

# %%
# ============================================================================
# PART 4: Unified Analysis Pipeline
# ============================================================================
print("\n🚀 Starting unified analysis pipeline...")

@dataclass
class UnifiedAnalysisResults:
    """Container for all analysis results"""
    hardware_config: str
    mcmc_samples: int
    
    # Framework results
    skill_scores: Dict[str, Any]
    technical_premiums: Dict[str, Any]
    market_acceptability: Dict[str, Any]
    
    # Bayesian results
    hierarchical_results: Optional[Any] = None
    contamination_results: Optional[Any] = None
    robust_intervals: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    execution_time: float = 0.0
    convergence_diagnostics: Optional[Dict[str, Any]] = None

def run_unified_analysis():
    """
    Main analysis function using complete framework
    NO DUPLICATE IMPLEMENTATIONS - all from existing modules
    """
    
    start_time = time.time()
    results = UnifiedAnalysisResults(
        hardware_config=hardware_level,
        mcmc_samples=mcmc_kwargs['draws'] * mcmc_kwargs['chains']
    )
    
    # ========================================================================
    # STEP 1: Parametric Product Analysis (using framework)
    # ========================================================================
    print("\n📊 Step 1: Parametric Product Analysis")
    
    # Use existing product engine
    engine = ParametricInsuranceEngine()
    
    # Generate or load products
    if 'products' in loaded_data:
        products = loaded_data['products']
        print(f"   Using {len(products)} loaded products")
    else:
        print("   Generating Steinmann 2023 products...")
        products = generate_steinmann_2023_products()
        print(f"   Generated {len(products)} products")
    
    # ========================================================================
    # STEP 2: Skill Score Evaluation (using framework)
    # ========================================================================
    print("\n📈 Step 2: Skill Score Evaluation")
    
    evaluator = SkillScoreEvaluator()
    
    # Prepare data (use loaded or generate sample)
    if 'climada' in loaded_data:
        # Extract losses and indices from CLIMADA data
        climada_data = loaded_data['climada']
        observed_losses = climada_data.get('annual_impacts', np.random.gamma(2, 1e8, 100))
        parametric_indices = climada_data.get('wind_speeds', np.random.uniform(20, 60, 100))
    else:
        # Sample data for demonstration
        print("   ⚠️ Using sample data for demonstration")
        observed_losses = np.random.gamma(2, 1e8, 100)
        parametric_indices = np.random.uniform(20, 60, 100)
    
    # Evaluate all products
    skill_results = {}
    for product in products[:10] if args.quick_test else products:
        payouts = engine.calculate_payouts(product, parametric_indices)
        scores = evaluator.evaluate_all_skills(payouts, observed_losses)
        skill_results[product.product_id] = scores
    
    results.skill_scores = skill_results
    print(f"   ✅ Evaluated {len(skill_results)} products")
    
    # ========================================================================
    # STEP 3: Bayesian Uncertainty Quantification (using framework)
    # ========================================================================
    print("\n🎲 Step 3: Bayesian Uncertainty Quantification")
    
    # Configure PyMC environment
    configure_pymc_environment(backend="gpu" if "gpu" in hardware_level else "cpu")
    
    # Create hierarchical model with optimized MCMC config
    model_spec = ModelSpec(
        likelihood_family=LikelihoodFamily.LOGNORMAL,
        prior_scenario=PriorScenario.WEAK_INFORMATIVE,
        include_spatial_effects=True
    )
    
    # Use framework's MCMC configuration
    bayesian_mcmc_config = MCMCConfig(
        n_samples=mcmc_kwargs['draws'],
        n_warmup=mcmc_kwargs['tune'],
        n_chains=mcmc_kwargs['chains'],
        cores=mcmc_kwargs['cores'],
        target_accept=mcmc_kwargs['target_accept']
    )
    
    try:
        # Fit hierarchical model
        hierarchical_model = ParametricHierarchicalModel(
            model_spec=model_spec,
            mcmc_config=bayesian_mcmc_config
        )
        
        if not args.quick_test:
            hierarchical_results = hierarchical_model.fit(observed_losses)
            results.hierarchical_results = hierarchical_results
            print(f"   ✅ Hierarchical model fitted with {bayesian_mcmc_config.n_chains} chains")
        else:
            print("   ⏭️ Skipping full Bayesian analysis in quick test mode")
            
    except Exception as e:
        print(f"   ⚠️ Bayesian analysis error: {e}")
    
    # ========================================================================
    # STEP 4: ε-Contamination Analysis (using framework)
    # ========================================================================
    print("\n🌀 Step 4: ε-Contamination Robust Analysis")
    
    try:
        # Create typhoon-specific contamination model
        contamination_spec = create_typhoon_contamination_spec(
            epsilon_range=(0.01, 0.15)  # 1-15% typhoon events
        )
        
        contamination_model = EpsilonContaminationClass(contamination_spec)
        
        # Estimate contamination level
        contamination_results = contamination_model.estimate_contamination_level(
            data=observed_losses,
            wind_data=parametric_indices
        )
        
        results.contamination_results = contamination_results
        print(f"   ✅ Contamination level: ε = {contamination_results.epsilon_consensus:.3f}")
        
    except Exception as e:
        print(f"   ⚠️ Contamination analysis error: {e}")
    
    # ========================================================================
    # STEP 5: Technical Premium Calculation (using framework)
    # ========================================================================
    print("\n💰 Step 5: Technical Premium Optimization")
    
    premium_calculator = TechnicalPremiumCalculator()
    
    # Calculate premiums for top products
    premium_results = {}
    top_products = sorted(
        skill_results.items(), 
        key=lambda x: x[1].get('rmse', float('inf'))
    )[:5]  # Top 5 products by RMSE
    
    for product_id, _ in top_products:
        product = next((p for p in products if p.product_id == product_id), None)
        if product:
            premium = premium_calculator.calculate_technical_premium(
                product=product,
                historical_losses=observed_losses,
                confidence_level=0.95
            )
            premium_results[product_id] = premium
    
    results.technical_premiums = premium_results
    print(f"   ✅ Calculated premiums for {len(premium_results)} products")
    
    # ========================================================================
    # STEP 6: Market Acceptability Analysis (using framework)
    # ========================================================================
    print("\n📊 Step 6: Market Acceptability Analysis")
    
    market_analyzer = MarketAcceptabilityAnalyzer()
    
    market_results = {}
    for product_id, premium in premium_results.items():
        product = next((p for p in products if p.product_id == product_id), None)
        if product:
            acceptability = market_analyzer.analyze_acceptability(
                product=product,
                premium=premium,
                market_benchmark=np.mean(observed_losses) * 0.1  # 10% of average loss
            )
            market_results[product_id] = acceptability
    
    results.market_acceptability = market_results
    print(f"   ✅ Market analysis for {len(market_results)} products")
    
    # ========================================================================
    # STEP 7: Multi-Objective Optimization (using framework)
    # ========================================================================
    if not args.quick_test:
        print("\n🎯 Step 7: Multi-Objective Pareto Optimization")
        
        optimizer = MultiObjectiveOptimizer()
        
        # Prepare objectives
        objectives = []
        for product_id in premium_results.keys():
            obj = {
                'product_id': product_id,
                'rmse': skill_results[product_id].get('rmse', float('inf')),
                'premium': premium_results[product_id],
                'acceptability': market_results[product_id].get('score', 0)
            }
            objectives.append(obj)
        
        # Find Pareto frontier
        pareto_frontier = optimizer.find_pareto_frontier(
            objectives,
            minimize=['rmse', 'premium'],
            maximize=['acceptability']
        )
        
        print(f"   ✅ Pareto frontier: {len(pareto_frontier)} optimal products")
    
    # ========================================================================
    # Final metrics
    # ========================================================================
    results.execution_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✅ UNIFIED ANALYSIS COMPLETE")
    print(f"   Hardware: {results.hardware_config}")
    print(f"   Total MCMC samples: {results.mcmc_samples:,}")
    print(f"   Execution time: {results.execution_time:.1f} seconds")
    print(f"   Products analyzed: {len(skill_results)}")
    print("=" * 80)
    
    return results

# %%
# ============================================================================
# PART 5: Save Results
# ============================================================================
def save_results(results: UnifiedAnalysisResults):
    """Save unified analysis results"""
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    with open(output_dir / 'unified_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary report
    summary = {
        'hardware': results.hardware_config,
        'mcmc_samples': results.mcmc_samples,
        'execution_time': results.execution_time,
        'products_analyzed': len(results.skill_scores),
        'contamination_level': getattr(
            results.contamination_results, 
            'epsilon_consensus', 
            None
        ) if results.contamination_results else None
    }
    
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        import json
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {output_dir}")

# %%
# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    try:
        # Run unified analysis
        analysis_results = run_unified_analysis()
        
        # Save results
        save_results(analysis_results)
        
        print("\n🎉 Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)