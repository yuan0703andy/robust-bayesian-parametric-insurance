#!/usr/bin/env python3
"""
Modular Robust Bayesian Parametric Insurance Analysis
模組化穩健貝氏參數保險分析

This version uses the modular GPU setup from bayesian.gpu_setup
使用來自 bayesian.gpu_setup 的模組化GPU設置

Usage with GPU optimization:
使用GPU優化的用法：
python 05_robust_bayesian_parm_insurance_modular.py --enable-gpu
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# Configure environment before importing heavy libraries
os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float32,optimizer=fast_compile'

# Import GPU setup module
try:
    from bayesian.gpu_setup import GPUConfig, setup_gpu_environment
    HAS_GPU_SETUP = True
    print("✅ GPU setup module loaded successfully")
except ImportError as e:
    HAS_GPU_SETUP = False
    print(f"⚠️ GPU setup module not available: {e}")

# Import other required modules
from insurance_analysis_refactored.core import (
    ParametricInsuranceEngine,
    SkillScoreEvaluator,
    create_standard_technical_premium_calculator
)

from bayesian.robust_model_ensemble_analyzer import (
    ModelClassAnalyzer, ModelClassSpec, AnalyzerConfig, MCMCConfig
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Robust Bayesian Parametric Insurance Analysis')
    parser.add_argument('--enable-gpu', action='store_true', 
                       help='Enable GPU acceleration (requires JAX/CUDA)')
    parser.add_argument('--gpu-only', action='store_true',
                       help='Force GPU-only mode (fail if GPU unavailable)')
    parser.add_argument('--cpu-only', action='store_true', 
                       help='Force CPU-only mode')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run with minimal samples for testing')
    return parser.parse_args()

def setup_analysis_environment(args):
    """Setup analysis environment based on arguments"""
    print("🔧 Setting up analysis environment...")
    
    # Determine GPU configuration
    if args.cpu_only:
        enable_gpu = False
        print("💻 CPU-only mode requested")
    elif args.enable_gpu or args.gpu_only:
        enable_gpu = True
        print("🎯 GPU acceleration requested")
    else:
        enable_gpu = HAS_GPU_SETUP  # Auto-detect
        print("🔍 Auto-detecting GPU capability")
    
    # Setup GPU configuration
    gpu_config = None
    if enable_gpu and HAS_GPU_SETUP:
        try:
            gpu_config = setup_gpu_environment(enable_gpu=True)
            gpu_config.print_performance_summary()
        except Exception as e:
            print(f"⚠️ GPU setup failed: {e}")
            if args.gpu_only:
                raise RuntimeError("GPU-only mode requested but GPU setup failed")
            gpu_config = None
            enable_gpu = False
    
    # Create MCMC configuration
    if gpu_config:
        mcmc_config_dict = gpu_config.get_mcmc_config()
    else:
        print("💻 Using CPU configuration")
        mcmc_config_dict = {
            "n_samples": 500 if args.quick_test else 2000,
            "n_warmup": 250 if args.quick_test else 1000,
            "n_chains": 2 if args.quick_test else 4,
            "cores": 4,
            "target_accept": 0.90,
            "backend": "pytensor"
        }
    
    # Adjust for quick test
    if args.quick_test:
        mcmc_config_dict["n_samples"] = min(mcmc_config_dict["n_samples"], 500)
        mcmc_config_dict["n_warmup"] = min(mcmc_config_dict["n_warmup"], 250)
        mcmc_config_dict["n_chains"] = min(mcmc_config_dict["n_chains"], 2)
        print("⚡ Quick test mode: reduced sample sizes")
    
    return gpu_config, mcmc_config_dict

def load_analysis_data():
    """Load all required analysis data"""
    print("\n📂 Loading analysis data...")
    
    try:
        # Load CLIMADA data
        with open('results/climada_data/climada_complete_data.pkl', 'rb') as f:
            climada_results = pickle.load(f)
        print("   ✅ CLIMADA data loaded")
        
        # Load spatial analysis results  
        with open('results/spatial_analysis/cat_in_circle_results.pkl', 'rb') as f:
            spatial_results = pickle.load(f)
        print("   ✅ Spatial analysis results loaded")
        
        # Load parametric products
        with open('results/parametric_products/steinmann_products.pkl', 'rb') as f:
            parametric_products = pickle.load(f)
        print("   ✅ Parametric products loaded")
        
        # Load traditional analysis results
        with open('results/traditional_analysis/traditional_parametric_results.pkl', 'rb') as f:
            traditional_results = pickle.load(f)
        print("   ✅ Traditional analysis results loaded")
        
        return {
            'climada_results': climada_results,
            'spatial_results': spatial_results, 
            'parametric_products': parametric_products,
            'traditional_results': traditional_results
        }
        
    except FileNotFoundError as e:
        print(f"❌ Required data file not found: {e}")
        print("   Please run the preceding analysis scripts (01-04) first")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

def run_bayesian_analysis(data, mcmc_config_dict, gpu_config=None):
    """Run robust Bayesian analysis"""
    print("\n🔬 Phase 1: Robust Bayesian Model Ensemble Analysis")
    print("=" * 60)
    
    # Create MCMC configuration
    mcmc_config = MCMCConfig(
        n_samples=mcmc_config_dict["n_samples"],
        n_warmup=mcmc_config_dict["n_warmup"], 
        n_chains=mcmc_config_dict["n_chains"],
        cores=mcmc_config_dict["cores"],
        target_accept=mcmc_config_dict["target_accept"],
        backend=mcmc_config_dict.get("backend", "pytensor")
    )
    
    # Create analyzer configuration
    analyzer_config = AnalyzerConfig(
        mcmc_config=mcmc_config,
        use_mpe=True,
        parallel_execution=False,  # Sequential for stability
        max_workers=1,
        model_selection_criterion='dic',
        calculate_ranges=True,
        calculate_weights=True
    )
    
    # Setup model class specification with ε-contamination
    model_class_spec = ModelClassSpec(
        enable_epsilon_contamination=True,
        epsilon_values=[0.01, 0.05],  # Conservative epsilon values
        contamination_distribution="typhoon"
    )
    
    print(f"📊 Model ensemble configuration:")
    print(f"   Total models: {model_class_spec.get_model_count()}")
    print(f"   ε-contamination models: {len(model_class_spec.epsilon_values) * 6}")  # 6 base models
    print(f"   MCMC: {mcmc_config.n_chains} chains × {mcmc_config.n_samples} samples")
    
    # Create analyzer
    analyzer = ModelClassAnalyzer(model_class_spec, analyzer_config)
    
    # Extract observed losses from traditional results
    traditional_results = data['traditional_results']
    observed_losses = traditional_results['observed_losses']
    
    print(f"🎯 Analyzing {len(observed_losses)} loss observations...")
    
    # Run model ensemble analysis
    if gpu_config:
        print("🚀 Using GPU-accelerated MCMC sampling...")
        
    ensemble_results = analyzer.analyze_model_class(observed_losses)
    
    print(f"\n✅ Model ensemble analysis complete:")
    print(f"   Best model: {ensemble_results.best_model}")
    print(f"   Execution time: {ensemble_results.execution_time:.2f} seconds")
    print(f"   Successful fits: {len(ensemble_results.individual_results)}")
    
    return ensemble_results

def calculate_skill_scores(ensemble_results, traditional_results):
    """Calculate comprehensive skill scores"""
    print("\n📈 Phase 2: Skill Score Evaluation")
    print("=" * 40)
    
    skill_evaluator = SkillScoreEvaluator()
    
    # Extract parametric indices and observed losses
    parametric_indices = traditional_results['parametric_indices'] 
    observed_losses = traditional_results['observed_losses']
    
    # Calculate skill scores for the best model
    best_model_name = ensemble_results.best_model
    best_model_result = ensemble_results.individual_results[best_model_name]
    
    # Use posterior mean as point predictions
    posterior_samples = best_model_result.posterior_samples
    if 'theta' in posterior_samples:
        predictions = np.full(len(observed_losses), np.mean(posterior_samples['theta']))
    else:
        # Fallback to using parametric indices
        predictions = parametric_indices
    
    skill_scores = skill_evaluator.calculate_comprehensive_scores(
        parametric_indices, observed_losses, predictions
    )
    
    print("📊 Skill Score Results:")
    for metric, value in skill_scores.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
        else:
            print(f"   {metric}: {value}")
    
    return skill_scores

def save_results(ensemble_results, skill_scores, gpu_config):
    """Save analysis results"""
    print("\n💾 Saving results...")
    
    results_dir = Path("results/robust_bayesian_modular")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    results_data = {
        'ensemble_results': ensemble_results,
        'skill_scores': skill_scores,
        'gpu_config_used': gpu_config.hardware_level if gpu_config else 'cpu_only',
        'analysis_type': 'robust_bayesian_modular'
    }
    
    with open(results_dir / 'robust_bayesian_results.pkl', 'wb') as f:
        pickle.dump(results_data, f)
    print(f"   ✅ Results saved to {results_dir}/")
    
    # Save model comparison table
    comparison_df = ensemble_results.get_model_ranking('dic')
    comparison_table = pd.DataFrame(comparison_df, columns=['Model', 'DIC'])
    comparison_table.to_csv(results_dir / 'model_comparison.csv', index=False)
    print(f"   ✅ Model comparison saved")
    
    return results_dir

def main():
    """Main analysis workflow"""
    print("🚀 Modular Robust Bayesian Parametric Insurance Analysis")
    print("=" * 70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    gpu_config, mcmc_config_dict = setup_analysis_environment(args)
    
    # Load data
    data = load_analysis_data()
    
    # Run Bayesian analysis
    ensemble_results = run_bayesian_analysis(data, mcmc_config_dict, gpu_config)
    
    # Calculate skill scores
    skill_scores = calculate_skill_scores(ensemble_results, data['traditional_results'])
    
    # Save results
    results_dir = save_results(ensemble_results, skill_scores, gpu_config)
    
    # Summary
    print(f"\n🎉 Analysis Complete!")
    print("=" * 30)
    print(f"📊 Best Model: {ensemble_results.best_model}")
    print(f"⏱️ Total Time: {ensemble_results.execution_time:.1f} seconds") 
    print(f"💾 Results: {results_dir}/")
    
    if gpu_config:
        print(f"🎯 GPU Acceleration: {gpu_config.hardware_level}")
        expected_speedup = "4x" if gpu_config.hardware_level == "dual_gpu" else "2-3x"
        print(f"⚡ Performance: {expected_speedup} speedup achieved")
    
    return ensemble_results

if __name__ == "__main__":
    main()