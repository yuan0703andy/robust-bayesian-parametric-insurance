#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_traditional_parm_insurance.py
=================================
Traditional Parametric Insurance Analysis using existing framework

Simply configures and runs the existing analysis components for 
traditional RMSE-based basis risk evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from pathlib import Path

# Import existing framework components
from insurance_analysis_refactored.core.analysis_framework import (
    UnifiedAnalysisFramework, EvaluationMode
)
from insurance_analysis_refactored.core.skill_evaluator import SkillScoreEvaluator
from insurance_analysis_refactored.core.input_adapters import CLIMADAInputAdapter


def main():
    """
    Main program: Traditional basis risk analysis using existing framework
    """
    print("=" * 80)
    print("Traditional Parametric Insurance Analysis")
    print("Using existing insurance_analysis_refactored framework")
    print("RMSE-based deterministic evaluation")
    print("=" * 80)
    
    # Load required data
    print("\n📂 Loading data...")
    
    # Load products
    try:
        with open("results/insurance_products/products.pkl", 'rb') as f:
            products = pickle.load(f)
        print(f"✅ Loaded {len(products)} insurance products")
    except FileNotFoundError:
        print("❌ Products not found. Run 03_insurance_product.py first.")
        return
    
    # Load spatial analysis results  
    try:
        with open("results/spatial_analysis/cat_in_circle_results.pkl", 'rb') as f:
            spatial_results = pickle.load(f)
        wind_indices = spatial_results['indices']
        print("✅ Loaded spatial analysis results")
    except FileNotFoundError:
        print("❌ Spatial results not found. Run 02_spatial_analysis.py first.")
        return
    
    # Load CLIMADA data
    try:
        with open("climada_complete_data.pkl", 'rb') as f:
            climada_data = pickle.load(f)
        print("✅ Loaded CLIMADA data")
    except FileNotFoundError:
        print("⚠️ Using synthetic loss data")
        np.random.seed(42)
        climada_data = {
            'impact': type('MockImpact', (), {
                'at_event': np.random.lognormal(15, 1, 1000) * 1e6
            })()
        }
    
    # Initialize framework components
    print("\n🔧 Initializing framework components...")
    
    # Create input adapter for CLIMADA data
    climada_adapter = CLIMADAInputAdapter(
        hazard_data=wind_indices,
        exposure_data=None,  # Not needed for this analysis
        impact_data=climada_data.get('impact')
    )
    
    # Initialize unified framework
    framework = UnifiedAnalysisFramework()
    
    # Configure for traditional analysis (RMSE-based)
    traditional_config = {
        'evaluation_metrics': ['rmse', 'mae', 'correlation', 'r_squared'],
        'use_probabilistic_methods': False,  # Traditional deterministic approach
        'monte_carlo_samples': 1,  # No Monte Carlo for traditional
        'basis_risk_calculation': 'deterministic',
        'skill_score_methods': ['rmse', 'mae', 'correlation']
    }
    
    print("📊 Running traditional analysis...")
    print("   • Method: RMSE-based deterministic")
    print("   • Metrics: RMSE, MAE, Correlation, R²") 
    print("   • Approach: Point estimates only")
    
    # Configure framework for traditional analysis
    framework.config.evaluation_mode = EvaluationMode.TRADITIONAL
    framework.config.skill_scores = ['rmse', 'mae', 'correlation', 'r_squared']
    
    # Run analysis using existing framework
    results = framework.run_comprehensive_analysis(
        parametric_indices=wind_indices,
        observed_losses=climada_data.get('impact').at_event if 'impact' in climada_data else np.array([]),
        analysis_name="traditional_analysis"
    )
    
    # Extract and display results
    print("\n✅ Traditional analysis complete!")
    print(f"📊 Analyzed {len(products)} products")
    
    if 'skill_evaluation' in results:
        skill_results = results['skill_evaluation']
        print("\n🏆 Top 5 Products (by RMSE):")
        print("-" * 40)
        
        # Sort products by RMSE (lower is better)
        if 'product_rankings' in skill_results:
            rankings = skill_results['product_rankings']
            rmse_ranking = rankings.get('rmse', [])[:5]
            
            for i, (product_id, rmse_score) in enumerate(rmse_ranking, 1):
                print(f"{i}. {product_id}")
                print(f"   RMSE: ${rmse_score:,.0f}")
                
                # Find product info
                product_info = next((p for p in products if p.get('product_id') == product_id), {})
                if product_info:
                    print(f"   Structure: {product_info.get('structure_type', 'Unknown')}")
                print()
    
    # Save results
    output_dir = "results/traditional_analysis"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f"{output_dir}/traditional_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"💾 Results saved to: {output_dir}/traditional_results.pkl")
    
    # Generate visualizations using existing framework
    if hasattr(framework, 'visualize_results'):
        print("📈 Generating visualizations...")
        framework.visualize_results(results, output_dir=output_dir)
    
    print("\n✅ Traditional analysis complete!")
    print("   Framework components used:")
    print("   • UnifiedAnalysisFramework")
    print("   • SkillScoreEvaluator") 
    print("   • CLIMADAInputAdapter")
    
    return results


if __name__ == "__main__":
    results = main()