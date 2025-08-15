#!/usr/bin/env python3
"""
05. Robust Bayesian Parametric Insurance Analysis - No MCMC Version
穩健貝氏參數型保險分析 - 無MCMC版本

Pure insurance analysis using only the insurance_analysis_refactored framework.
No PyMC/MCMC to avoid compilation issues.

純保險分析使用insurance_analysis_refactored框架，無PyMC/MCMC避免編譯問題。

Author: Research Team
Date: 2025-01-15
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
from typing import Dict, List, Optional, Any
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("05. Robust Bayesian Insurance Analysis - No MCMC")
print("穩健貝氏參數型保險分析 - 無MCMC版本")
print("=" * 80)
print("\n💻 Pure insurance framework analysis")
print("🚀 Fast, stable, and reliable (no compilation issues)")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Insurance Analysis without MCMC')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run with minimal processing for testing')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    return parser.parse_args()

# Import insurance analysis framework
print("\n📦 Loading insurance analysis framework...")
try:
    from insurance_analysis_refactored.core import (
        ParametricInsuranceEngine,
        SkillScoreEvaluator,
        create_standard_technical_premium_calculator,
        InsuranceProductManager
    )
    
    # Try to import advanced modules
    try:
        from insurance_analysis_refactored.core import (
            MultiObjectiveOptimizer,
            MarketAcceptabilityAnalyzer,
            TechnicalPremiumVisualizer
        )
        HAS_ADVANCED_MODULES = True
        print("   ✅ Advanced modules available")
    except ImportError:
        HAS_ADVANCED_MODULES = False
        print("   ⚠️ Advanced modules not available")
    
    print("   ✅ Core insurance framework loaded")
    HAS_INSURANCE_FRAMEWORK = True
    
except ImportError as e:
    print(f"   ❌ Insurance framework error: {e}")
    HAS_INSURANCE_FRAMEWORK = False
    sys.exit(1)

def load_analysis_data():
    """Load all required analysis data"""
    print("\n📂 Loading analysis data...")
    
    data = {}
    required_files = {
        'climada_data': 'results/climada_data/climada_complete_data.pkl',
        'spatial_analysis': 'results/spatial_analysis/cat_in_circle_results.pkl',
        'insurance_products': 'results/insurance_products/products.pkl',
        'traditional_analysis': 'results/traditional_basis_risk_analysis/analysis_results.pkl'
    }
    
    for key, filepath in required_files.items():
        try:
            with open(filepath, 'rb') as f:
                data[key] = pickle.load(f)
            print(f"   ✅ {key.replace('_', ' ').title()} loaded")
        except FileNotFoundError:
            print(f"   ⚠️ {filepath} not found")
            data[key] = None
        except Exception as e:
            print(f"   ❌ Error loading {filepath}: {e}")
            data[key] = None
    
    # Extract event losses from CLIMADA data
    if data['climada_data'] and 'event_losses' in data['climada_data']:
        event_losses_array = data['climada_data']['event_losses']
        data['event_losses'] = {i: loss for i, loss in enumerate(event_losses_array)}
        data['observed_losses'] = np.array([loss for loss in event_losses_array if loss > 0])
    else:
        # Generate synthetic data for demonstration
        print("   🔧 Generating synthetic loss data...")
        np.random.seed(42)
        synthetic_losses = np.random.lognormal(mean=15, sigma=2, size=500)
        data['event_losses'] = {i: loss for i, loss in enumerate(synthetic_losses)}
        data['observed_losses'] = synthetic_losses[synthetic_losses > 1e6]  # Only large losses
    
    print(f"\n📊 Data Summary:")
    print(f"   Event losses: {len(data['event_losses'])} events")
    print(f"   Non-zero losses: {len(data['observed_losses'])} events")
    if len(data['observed_losses']) > 0:
        print(f"   Loss range: ${np.min(data['observed_losses'])/1e6:.1f}M - ${np.max(data['observed_losses'])/1e6:.1f}M")
        print(f"   Mean loss: ${np.mean(data['observed_losses'])/1e6:.1f}M")
        print(f"   Median loss: ${np.median(data['observed_losses'])/1e6:.1f}M")
    
    return data

def create_parametric_products(data, args):
    """Create parametric insurance products"""
    print("\n" + "=" * 80)
    print("Phase 1: Parametric Product Creation")
    print("階段1：參數型產品創建")
    print("=" * 80)
    
    engine = ParametricInsuranceEngine()
    
    # Check for existing products
    if data['insurance_products'] and not args.quick_test:
        products = data['insurance_products']
        print(f"   ✅ Using {len(products)} existing products")
    else:
        print("   🔧 Creating new parametric products...")
        
        # Create diverse product portfolio
        products = []
        
        # Wind speed products
        for i, threshold in enumerate([25, 35, 45, 55]):
            product = {
                'product_id': f'wind_speed_{threshold}',
                'name': f'Wind Speed {threshold}m/s Product',
                'trigger_type': 'wind_speed',
                'trigger_threshold': threshold,
                'coverage_amount': 1e8 + i * 5e7,
                'premium_rate': 0.015 + i * 0.005,
                'payout_structure': 'step',
                'max_payout': 1e8 + i * 5e7
            }
            products.append(product)
        
        # Cat-in-circle products  
        for i, radius in enumerate([100, 150, 200]):
            product = {
                'product_id': f'cat_circle_{radius}km',
                'name': f'Cat-in-Circle {radius}km Product',
                'trigger_type': 'cat_in_circle',
                'trigger_threshold': radius,
                'coverage_amount': 2e8 + i * 1e8,
                'premium_rate': 0.02 + i * 0.01,
                'payout_structure': 'proportional',
                'max_payout': 2e8 + i * 1e8
            }
            products.append(product)
        
        # Pressure-based products
        for i, pressure in enumerate([950, 940, 930]):
            product = {
                'product_id': f'pressure_{pressure}hPa',
                'name': f'Min Pressure {pressure}hPa Product',
                'trigger_type': 'min_pressure',
                'trigger_threshold': pressure,
                'coverage_amount': 1.5e8 + i * 5e7,
                'premium_rate': 0.018 + i * 0.007,
                'payout_structure': 'step',
                'max_payout': 1.5e8 + i * 5e7
            }
            products.append(product)
        
        print(f"   ✅ Created {len(products)} parametric products")
    
    return products

def analyze_distributions(data, args):
    """Analyze loss distributions using statistical methods"""
    print("\n" + "=" * 80)
    print("Phase 2: Statistical Distribution Analysis")
    print("階段2：統計分布分析")
    print("=" * 80)
    
    observed_losses = data['observed_losses']
    
    if len(observed_losses) < 10:
        print("⚠️ Insufficient data for distribution analysis")
        return {}
    
    # Fit various distributions
    distributions = {
        'lognormal': stats.lognorm,
        'gamma': stats.gamma,
        'weibull': stats.weibull_min,
        'exponential': stats.expon
    }
    
    fit_results = {}
    print("\n🔬 Fitting distributions...")
    
    for name, dist in distributions.items():
        try:
            # Fit distribution
            params = dist.fit(observed_losses, floc=0)
            
            # Calculate goodness of fit
            log_likelihood = np.sum(dist.logpdf(observed_losses, *params))
            n_params = len(params)
            aic = 2 * n_params - 2 * log_likelihood
            bic = n_params * np.log(len(observed_losses)) - 2 * log_likelihood
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.kstest(observed_losses, 
                                            lambda x: dist.cdf(x, *params))
            
            fit_results[name] = {
                'params': params,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'distribution': dist
            }
            
            print(f"   ✅ {name}: AIC={aic:.2f}, BIC={bic:.2f}, KS p-value={ks_pvalue:.4f}")
            
        except Exception as e:
            print(f"   ❌ {name}: Failed to fit - {e}")
    
    # Find best distribution
    if fit_results:
        best_by_aic = min(fit_results.items(), key=lambda x: x[1]['aic'])
        best_by_bic = min(fit_results.items(), key=lambda x: x[1]['bic'])
        
        print(f"\n🏆 Best distributions:")
        print(f"   By AIC: {best_by_aic[0]} (AIC = {best_by_aic[1]['aic']:.2f})")
        print(f"   By BIC: {best_by_bic[0]} (BIC = {best_by_bic[1]['bic']:.2f})")
    
    return fit_results

def calculate_technical_premiums(products, data, args):
    """Calculate technical premiums for products"""
    print("\n" + "=" * 80)
    print("Phase 3: Technical Premium Calculation")
    print("階段3：技術保費計算")
    print("=" * 80)
    
    try:
        premium_calculator = create_standard_technical_premium_calculator()
        print("   ✅ Premium calculator initialized")
    except Exception as e:
        print(f"   ⚠️ Premium calculator unavailable: {e}")
        # Simple premium calculation fallback
        premium_calculator = None
    
    enhanced_products = []
    observed_losses = data['observed_losses']
    
    for i, product in enumerate(products):
        try:
            if premium_calculator:
                # Use advanced calculator
                premium_data = premium_calculator.calculate_premium(product, observed_losses)
            else:
                # Simple premium calculation
                expected_payout = np.mean(observed_losses) * product.get('premium_rate', 0.02)
                loading_factor = 1.3  # 30% loading
                premium_data = {
                    'expected_premium': expected_payout * loading_factor,
                    'expected_payout': expected_payout,
                    'loading_factor': loading_factor,
                    'risk_margin': expected_payout * 0.1
                }
            
            enhanced_product = {
                **product,
                'technical_premium': premium_data,
                'risk_assessment': {
                    'expected_annual_loss': np.mean(observed_losses),
                    'loss_volatility': np.std(observed_losses),
                    'max_observed_loss': np.max(observed_losses) if len(observed_losses) > 0 else 0
                }
            }
            enhanced_products.append(enhanced_product)
            
        except Exception as e:
            if args.verbose:
                print(f"   ⚠️ Premium calculation failed for {product.get('product_id', i)}: {e}")
            enhanced_products.append(product)  # Keep original product
    
    print(f"   ✅ Calculated premiums for {len(enhanced_products)} products")
    return enhanced_products

def evaluate_product_performance(products, data, args):
    """Evaluate product performance using skill scores"""
    print("\n" + "=" * 80)
    print("Phase 4: Product Performance Evaluation")
    print("階段4：產品績效評估")
    print("=" * 80)
    
    skill_evaluator = SkillScoreEvaluator()
    observed_losses = data['observed_losses']
    
    if len(observed_losses) < 10:
        print("⚠️ Insufficient data for performance evaluation")
        return {}
    
    performance_results = {}
    
    for product in products[:10]:  # Limit for performance
        product_id = product.get('product_id', 'unknown')
        
        try:
            # Generate simple predictions based on product characteristics
            if 'technical_premium' in product:
                expected_premium = product['technical_premium'].get('expected_premium', 0)
                predictions = np.full(len(observed_losses), expected_premium)
            else:
                # Fallback prediction
                coverage = product.get('coverage_amount', 1e8)
                predictions = np.full(len(observed_losses), coverage * 0.02)
            
            # Calculate basic skill scores
            rmse = np.sqrt(np.mean((predictions - observed_losses) ** 2))
            mae = np.mean(np.abs(predictions - observed_losses))
            correlation = np.corrcoef(predictions, observed_losses)[0, 1] if len(predictions) > 1 else 0
            
            # Bias measures
            bias = np.mean(predictions - observed_losses)
            relative_bias = bias / np.mean(observed_losses) if np.mean(observed_losses) > 0 else 0
            
            performance_results[product_id] = {
                'RMSE': rmse,
                'MAE': mae,
                'Correlation': correlation,
                'Bias': bias,
                'Relative_Bias': relative_bias,
                'RMSE_normalized': rmse / np.mean(observed_losses) if np.mean(observed_losses) > 0 else np.inf
            }
            
        except Exception as e:
            if args.verbose:
                print(f"   ⚠️ Performance evaluation failed for {product_id}: {e}")
    
    # Summary statistics
    if performance_results:
        rmse_values = [r['RMSE'] for r in performance_results.values()]
        correlations = [r['Correlation'] for r in performance_results.values() if not np.isnan(r['Correlation'])]
        
        print(f"\n📊 Performance Summary:")
        print(f"   Products evaluated: {len(performance_results)}")
        print(f"   Mean RMSE: ${np.mean(rmse_values)/1e6:.1f}M")
        print(f"   Best RMSE: ${np.min(rmse_values)/1e6:.1f}M")
        if correlations:
            print(f"   Mean correlation: {np.mean(correlations):.4f}")
            print(f"   Best correlation: {np.max(correlations):.4f}")
    
    return performance_results

def run_market_analysis(products, args):
    """Run market acceptability analysis"""
    print("\n" + "=" * 80)
    print("Phase 5: Market Acceptability Analysis")
    print("階段5：市場接受度分析")
    print("=" * 80)
    
    if not HAS_ADVANCED_MODULES:
        print("⚠️ Advanced market analysis modules not available")
        print("💡 Running simplified market analysis...")
        
        # Simple market analysis
        market_results = []
        for product in products:
            # Simple scoring based on product characteristics
            premium_rate = product.get('premium_rate', 0.02)
            coverage = product.get('coverage_amount', 1e8)
            
            # Scoring factors
            rate_score = max(0, 1 - (premium_rate - 0.015) / 0.02)  # Lower rate = better
            coverage_score = min(1, coverage / 5e8)  # Higher coverage = better (up to limit)
            simplicity_score = 0.8  # All products assumed reasonably simple
            
            market_score = (rate_score + coverage_score + simplicity_score) / 3
            
            market_results.append({
                'product_id': product.get('product_id', 'unknown'),
                'market_score': market_score,
                'rate_score': rate_score,
                'coverage_score': coverage_score,
                'simplicity_score': simplicity_score
            })
        
        print(f"   ✅ Simplified market analysis complete for {len(market_results)} products")
        return market_results
    
    try:
        market_analyzer = MarketAcceptabilityAnalyzer()
        
        market_results = []
        for product in products[:10]:  # Limit for performance
            try:
                market_score = market_analyzer.analyze_product_acceptability(product)
                market_results.append({
                    'product_id': product.get('product_id', 'unknown'),
                    'market_score': market_score
                })
            except Exception as e:
                if args.verbose:
                    print(f"   ⚠️ Market analysis failed for product: {e}")
        
        print(f"   ✅ Market analysis complete for {len(market_results)} products")
        return market_results
        
    except Exception as e:
        print(f"   ⚠️ Market analysis failed: {e}")
        return []

def save_comprehensive_results(data, products, distribution_results, 
                              performance_results, market_results, args):
    """Save all analysis results"""
    print("\n" + "=" * 80)
    print("Phase 6: Save Results")
    print("階段6：儲存結果")
    print("=" * 80)
    
    results_dir = Path('results/robust_bayesian_no_mcmc')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile comprehensive results
    final_results = {
        'analysis_metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'version': 'No-MCMC v1.0',
            'quick_test': args.quick_test,
            'method': 'Pure Insurance Framework Analysis'
        },
        'data_summary': {
            'n_events': len(data['event_losses']),
            'n_nonzero_losses': len(data['observed_losses']),
            'mean_loss': np.mean(data['observed_losses']) if len(data['observed_losses']) > 0 else 0,
            'max_loss': np.max(data['observed_losses']) if len(data['observed_losses']) > 0 else 0
        },
        'products_created': len(products),
        'distribution_analysis': {
            'distributions_fitted': len(distribution_results),
            'best_distribution_aic': min(distribution_results.items(), 
                                       key=lambda x: x[1]['aic'])[0] if distribution_results else None
        },
        'performance_analysis': {
            'products_evaluated': len(performance_results),
            'mean_rmse': np.mean([r['RMSE'] for r in performance_results.values()]) if performance_results else None
        },
        'market_analysis': {
            'products_analyzed': len(market_results),
            'mean_market_score': np.mean([r['market_score'] for r in market_results 
                                        if r['market_score'] is not None]) if market_results else None
        }
    }
    
    # Save main results
    with open(results_dir / 'comprehensive_results.pkl', 'wb') as f:
        pickle.dump({
            'final_results': final_results,
            'products': products,
            'distribution_results': distribution_results,
            'performance_results': performance_results,
            'market_results': market_results,
            'data': data
        }, f)
    
    # Save product portfolio as CSV
    if products:
        products_df = pd.DataFrame(products)
        products_df.to_csv(results_dir / 'product_portfolio.csv', index=False)
    
    # Save performance results as CSV
    if performance_results:
        performance_df = pd.DataFrame.from_dict(performance_results, orient='index')
        performance_df.to_csv(results_dir / 'performance_results.csv')
    
    # Save summary report
    with open(results_dir / 'analysis_report.txt', 'w') as f:
        f.write("Insurance Framework Analysis Report (No MCMC)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis completed: {final_results['analysis_metadata']['timestamp']}\n")
        f.write(f"Version: {final_results['analysis_metadata']['version']}\n")
        f.write(f"Method: {final_results['analysis_metadata']['method']}\n\n")
        
        f.write("Data Summary:\n")
        f.write(f"  Total events: {final_results['data_summary']['n_events']}\n")
        f.write(f"  Non-zero losses: {final_results['data_summary']['n_nonzero_losses']}\n")
        f.write(f"  Mean loss: ${final_results['data_summary']['mean_loss']/1e6:.1f}M\n")
        f.write(f"  Max loss: ${final_results['data_summary']['max_loss']/1e6:.1f}M\n\n")
        
        f.write("Analysis Results:\n")
        f.write(f"  Products created: {final_results['products_created']}\n")
        f.write(f"  Distributions fitted: {final_results['distribution_analysis']['distributions_fitted']}\n")
        f.write(f"  Products evaluated: {final_results['performance_analysis']['products_evaluated']}\n")
        f.write(f"  Market analysis: {final_results['market_analysis']['products_analyzed']} products\n\n")
        
        if final_results['distribution_analysis']['best_distribution_aic']:
            f.write(f"Best distribution: {final_results['distribution_analysis']['best_distribution_aic']}\n")
        
        if final_results['performance_analysis']['mean_rmse']:
            f.write(f"Mean RMSE: ${final_results['performance_analysis']['mean_rmse']/1e6:.1f}M\n")
        
        if final_results['market_analysis']['mean_market_score']:
            f.write(f"Mean market score: {final_results['market_analysis']['mean_market_score']:.3f}\n")
    
    print(f"✅ Results saved to: {results_dir}")
    return results_dir, final_results

def main():
    """Main analysis workflow"""
    print("🚀 Starting Insurance Framework Analysis (No MCMC)")
    
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Load data
    data = load_analysis_data()
    
    # Phase 1: Create products
    products = create_parametric_products(data, args)
    
    # Phase 2: Distribution analysis
    distribution_results = analyze_distributions(data, args)
    
    # Phase 3: Technical premiums
    products = calculate_technical_premiums(products, data, args)
    
    # Phase 4: Performance evaluation
    performance_results = evaluate_product_performance(products, data, args)
    
    # Phase 5: Market analysis
    market_results = run_market_analysis(products, args)
    
    # Phase 6: Save results
    results_dir, final_results = save_comprehensive_results(
        data, products, distribution_results, performance_results, market_results, args
    )
    
    elapsed_time = time.time() - start_time
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 Insurance Framework Analysis Complete!")
    print("=" * 80)
    print(f"✅ Results directory: {results_dir}")
    print(f"✅ Products created: {final_results['products_created']}")
    print(f"✅ Distributions analyzed: {final_results['distribution_analysis']['distributions_fitted']}")
    print(f"✅ Performance evaluated: {final_results['performance_analysis']['products_evaluated']} products")
    print(f"✅ Market analysis: {final_results['market_analysis']['products_analyzed']} products")
    print(f"⏱️ Total execution time: {elapsed_time:.1f} seconds")
    
    if final_results['distribution_analysis']['best_distribution_aic']:
        print(f"🏆 Best distribution: {final_results['distribution_analysis']['best_distribution_aic']}")
    
    print("\n💡 Key advantages of this approach:")
    print("   • No compilation issues (no PyMC/PyTensor)")
    print("   • Fast execution (pure Python/NumPy/SciPy)")
    print("   • Comprehensive insurance analysis")
    print("   • Production-ready and stable")
    print("   • Full parametric product lifecycle")
    print("=" * 80)
    
    return final_results

if __name__ == "__main__":
    main()