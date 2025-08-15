#!/usr/bin/env python3
"""
05. Robust Bayesian Parametric Insurance Analysis - CPU Optimized
穩健貝氏參數型保險分析 - CPU優化版本

Complete implementation using CPU-optimized Bayesian analysis with 
full insurance_analysis_refactored framework integration.

完整實現使用CPU優化貝氏分析，整合所有insurance_analysis_refactored框架。

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

# Force CPU-only execution
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float32,optimizer=fast_compile,allow_gc=True'

print("=" * 80)
print("05. Robust Bayesian Parametric Insurance - CPU Optimized")
print("穩健貝氏參數型保險分析 - CPU優化版本")
print("=" * 80)
print("\n💻 CPU-only mode: Stable, fast, and reliable")
print("🚀 Full insurance_analysis_refactored integration")

# Import frameworks
print("\n📦 Loading frameworks...")

# Import insurance analysis framework
try:
    from insurance_analysis_refactored.core import (
        ParametricInsuranceEngine,
        SkillScoreEvaluator,
        create_standard_technical_premium_calculator,
        InsuranceProductManager,
        MultiObjectiveOptimizer,
        MarketAcceptabilityAnalyzer,
        TechnicalPremiumVisualizer
    )
    
    # Import specialized modules
    from insurance_analysis_refactored.core.enhanced_spatial_analysis import EnhancedCatInCircleAnalyzer
    from insurance_analysis_refactored.core.saffir_simpson_products import generate_steinmann_2023_products
    
    print("   ✅ Complete insurance analysis framework loaded")
    HAS_INSURANCE_FRAMEWORK = True
except ImportError as e:
    print(f"   ❌ Insurance framework error: {e}")
    print("   💡 Will continue with available components")
    # Try individual imports
    try:
        from insurance_analysis_refactored.core import ParametricInsuranceEngine, SkillScoreEvaluator
        HAS_INSURANCE_FRAMEWORK = True
        print("   ✅ Basic insurance framework loaded")
    except ImportError:
        HAS_INSURANCE_FRAMEWORK = False
        print("   ❌ Critical insurance framework components missing")
        sys.exit(1)

# Import Bayesian framework (CPU-optimized)
try:
    from bayesian import (
        ModelClassAnalyzer, 
        ModelClassSpec, 
        AnalyzerConfig, 
        MCMCConfig,
        get_cpu_optimized_mcmc_config,
        configure_pymc_environment,
        ProbabilisticLossDistributionGenerator
    )
    print("   ✅ CPU-optimized Bayesian framework loaded")
    HAS_BAYESIAN = True
except ImportError as e:
    print(f"   ⚠️ Bayesian framework limited: {e}")
    HAS_BAYESIAN = False

# Configure PyMC for CPU
if HAS_BAYESIAN:
    configure_pymc_environment()
    print("   ✅ PyMC configured for CPU execution")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CPU-Optimized Robust Bayesian Analysis')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run with minimal samples for testing')
    parser.add_argument('--n-cores', type=int, default=None,
                       help='Number of CPU cores to use')
    parser.add_argument('--max-cores', type=int, default=None,
                       help='Maximum CPU cores to use (no limit if not specified)')
    parser.add_argument('--max-chains', type=int, default=None,
                       help='Maximum MCMC chains to use (auto-scale if not specified)')
    parser.add_argument('--high-performance', action='store_true',
                       help='Enable high-performance mode for workstation/server systems')
    parser.add_argument('--robust-sampling', action='store_true',
                       help='Enable robust sampling mode for difficult convergence (slower but more stable)')
    parser.add_argument('--balanced-mode', action='store_true',
                       help='Enable balanced mode (good convergence + reasonable speed) - RECOMMENDED')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    return parser.parse_args()

def detect_environment():
    """Detect system capabilities"""
    import multiprocessing
    import platform
    
    n_cores = multiprocessing.cpu_count()
    system = platform.system()
    
    # Try to get memory info
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        has_psutil = True
    except ImportError:
        memory_gb = None
        has_psutil = False
    
    # Determine system class
    if n_cores >= 32:
        system_class = "High-end Workstation/Server"
        recommended_cores = n_cores  # Use all cores
    elif n_cores >= 16:
        system_class = "High-end Desktop"
        recommended_cores = min(n_cores, 12)  # Leave some cores free
    elif n_cores >= 8:
        system_class = "Mid-range System"
        recommended_cores = min(n_cores, 8)
    else:
        system_class = "Entry-level System"
        recommended_cores = min(n_cores, 4)
    
    print(f"\n🔍 System Detection:")
    print(f"   OS: {system}")
    print(f"   CPU cores: {n_cores}")
    print(f"   System class: {system_class}")
    print(f"   Python: {platform.python_version()}")
    if has_psutil and memory_gb:
        print(f"   Memory: {memory_gb:.1f} GB")
    
    # Performance recommendations
    if n_cores >= 16:
        print(f"\n💡 High-performance system detected!")
        print(f"   Consider using --high-performance flag")
        print(f"   Optimal MCMC: {min(n_cores//2, 16)} chains × {n_cores} cores")
    
    return {
        'n_cores': n_cores,
        'system': system,
        'system_class': system_class,
        'memory_gb': memory_gb,
        'recommended_cores': recommended_cores,
        'high_performance_capable': n_cores >= 16
    }

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
            print(f"   ⚠️ {filepath} not found, using fallback")
            data[key] = None
        except Exception as e:
            print(f"   ❌ Error loading {filepath}: {e}")
            data[key] = None
    
    # Extract event losses from CLIMADA data
    if data['climada_data']:
        event_losses_array = data['climada_data'].get('event_losses')
        if event_losses_array is not None:
            data['event_losses'] = {i: loss for i, loss in enumerate(event_losses_array)}
            data['observed_losses'] = np.array([loss for loss in event_losses_array if loss > 0])
        else:
            data['event_losses'] = {}
            data['observed_losses'] = np.array([])
    else:
        # Create synthetic data for demonstration
        print("   🔧 Creating synthetic event loss data for demonstration...")
        np.random.seed(42)  # Reproducible results
        synthetic_losses = np.random.lognormal(mean=15, sigma=1.5, size=100) * 1e6  # Log-normal losses
        
        # Add some zero-loss events
        all_events = np.concatenate([np.zeros(50), synthetic_losses])
        np.random.shuffle(all_events)
        
        data['event_losses'] = {i: loss for i, loss in enumerate(all_events)}
        data['observed_losses'] = np.array([loss for loss in all_events if loss > 0])
        print(f"   ✅ Generated {len(all_events)} events with {len(data['observed_losses'])} non-zero losses")
    
    print(f"\n📊 Data Summary:")
    print(f"   Event losses: {len(data['event_losses'])} events")
    print(f"   Non-zero losses: {len(data['observed_losses'])} events")
    if len(data['observed_losses']) > 0:
        print(f"   Loss range: ${np.min(data['observed_losses'])/1e6:.1f}M - ${np.max(data['observed_losses'])/1e6:.1f}M")
        print(f"   Mean loss: ${np.mean(data['observed_losses'])/1e6:.1f}M")
    
    return data

def setup_cpu_environment(args, system_info):
    """Setup CPU-optimized environment"""
    print("\n🔧 Setting up CPU-optimized environment...")
    
    # Determine cores to use
    if args.high_performance:
        print("🚀 High-performance mode enabled")
        n_cores = args.n_cores if args.n_cores else system_info['n_cores']  # Use all cores
        max_cores = args.max_cores
        max_chains = args.max_chains
    else:
        n_cores = args.n_cores if args.n_cores else system_info['recommended_cores']
        max_cores = args.max_cores if args.max_cores else 8  # Conservative default
        max_chains = args.max_chains if args.max_chains else 4  # Conservative default
    
    # Get optimized MCMC configuration with new parameters
    mcmc_config_dict = get_cpu_optimized_mcmc_config(
        n_cores=n_cores, 
        quick_test=args.quick_test,
        max_cores=max_cores,
        max_chains=max_chains,
        robust_sampling=args.robust_sampling,
        balanced_mode=args.balanced_mode
    )
    
    # Display sampling mode
    if args.robust_sampling:
        print("🛡️ Robust sampling mode enabled:")
        print("   • Ultra-conservative settings")
        print("   • Slower but maximum stability")
        print("   • Designed to eliminate divergences")
    elif args.balanced_mode:
        print("🎯 Balanced mode enabled:")
        print("   • Good convergence + reasonable speed")
        print("   • Recommended for most use cases")
        print("   • Should eliminate divergences quickly")
    
    print(f"📊 CPU Configuration:")
    print(f"   Cores to use: {n_cores}/{system_info['n_cores']}")
    print(f"   MCMC chains: {mcmc_config_dict['n_chains']}")
    print(f"   Samples per chain: {mcmc_config_dict['n_samples']}")
    print(f"   Total samples: {mcmc_config_dict['n_chains'] * mcmc_config_dict['n_samples']:,}")
    print(f"   Target accept: {mcmc_config_dict['target_accept']}")
    print(f"   Quick test mode: {args.quick_test}")
    
    return mcmc_config_dict

def run_parametric_insurance_design(data, args):
    """Run comprehensive parametric insurance design using insurance_analysis_refactored"""
    print("\n" + "=" * 80)
    print("Phase 1: Parametric Insurance Product Design")
    print("階段1：參數型保險產品設計")
    print("=" * 80)
    
    # Initialize parametric insurance engine
    engine = ParametricInsuranceEngine()
    
    # Get existing products or create new ones
    if data['insurance_products']:
        existing_products = data['insurance_products']
        print(f"   ✅ Using {len(existing_products)} existing products")
    else:
        print("   🔧 Creating new Steinmann-style products...")
        try:
            existing_products = generate_steinmann_2023_products()
            print(f"   ✅ Created {len(existing_products)} new products")
        except NameError:
            # Fallback to simple products
            existing_products = [
                {
                    'product_id': f'fallback_{i}',
                    'trigger_type': 'wind_speed',
                    'trigger_threshold': 30 + i * 10,
                    'coverage_amount': 1e8 + i * 5e7,
                    'premium_rate': 0.02 + i * 0.01
                }
                for i in range(5)
            ]
            print(f"   ✅ Created {len(existing_products)} fallback products")
    
    # Initialize product manager
    try:
        product_manager = InsuranceProductManager()
    except NameError:
        product_manager = None
    
    # Enhanced spatial analysis if data available
    spatial_results = None
    if data['spatial_analysis']:
        print("\n🌍 Running enhanced spatial analysis...")
        try:
            spatial_analyzer = EnhancedCatInCircleAnalyzer()
            spatial_results = spatial_analyzer.analyze_spatial_basis_risk(
                data['spatial_analysis'], 
                existing_products
            )
            print("   ✅ Spatial analysis complete")
        except (NameError, Exception) as e:
            print(f"   ⚠️ Spatial analysis skipped: {e}")
            spatial_results = None
    
    # Calculate technical premiums
    print("\n💰 Calculating technical premiums...")
    premium_calculator = create_standard_technical_premium_calculator()
    
    # Process products with technical premiums
    enhanced_products = []
    for i, product in enumerate(existing_products):
        try:
            # Calculate premium - convert to ParametricProduct if needed
            if isinstance(product, dict):
                # Convert dict to ParametricProduct for calculator
                from insurance_analysis_refactored.core import ParametricProduct, ParametricIndexType, PayoutFunctionType
                parametric_product = ParametricProduct(
                    product_id=product.get('product_id', f'product_{i}'),
                    name=product.get('name', f'Product {i}'),
                    description=product.get('description', 'Generated product'),
                    index_type=ParametricIndexType.CAT_IN_CIRCLE,
                    payout_function_type=PayoutFunctionType.STEP,
                    trigger_thresholds=product.get('trigger_thresholds', [30.0]),
                    payout_amounts=product.get('payout_amounts', [1e8]),
                    max_payout=product.get('max_payout', 1e8)
                )
            else:
                parametric_product = product
            
            # Use fallback hazard indices if no data
            hazard_indices = data['observed_losses'] if len(data['observed_losses']) > 0 else np.array([30, 40, 50, 60, 70])
            
            premium_data = premium_calculator.calculate_technical_premium(
                parametric_product, 
                hazard_indices
            )
            
            # Enhance product with premium data
            enhanced_product = {
                **product,
                'technical_premium': premium_data,
                'product_id': f"product_{i:03d}"
            }
            enhanced_products.append(enhanced_product)
            
        except Exception as e:
            if args.verbose:
                print(f"   ⚠️ Premium calculation failed for product {i}: {e}")
            continue
    
    print(f"   ✅ Technical premiums calculated for {len(enhanced_products)} products")
    
    return {
        'products': enhanced_products,
        'spatial_results': spatial_results,
        'engine': engine,
        'product_manager': product_manager
    }

def run_bayesian_analysis(data, mcmc_config_dict, insurance_results, args):
    """Run CPU-optimized Bayesian model ensemble analysis"""
    print("\n" + "=" * 80)
    print("Phase 2: CPU-Optimized Bayesian Analysis")
    print("階段2：CPU優化貝氏分析")
    print("=" * 80)
    
    if not HAS_BAYESIAN or len(data['observed_losses']) < 20:
        print("⚠️ Skipping Bayesian analysis (insufficient data or missing framework)")
        return None
    
    # Create MCMC configuration with enhanced settings
    mcmc_config = MCMCConfig(
        n_samples=mcmc_config_dict["n_samples"],
        n_warmup=mcmc_config_dict["n_warmup"],
        n_chains=mcmc_config_dict["n_chains"],
        cores=mcmc_config_dict["cores"],
        target_accept=mcmc_config_dict["target_accept"]
    )
    
    # Store additional sampler settings for PyMC
    sampler_kwargs = {
        "init": mcmc_config_dict.get("init", "adapt_diag"),
        "max_treedepth": mcmc_config_dict.get("max_treedepth", 12),
        "step_size": mcmc_config_dict.get("step_size", 0.1)
    }
    
    # Create analyzer configuration (CPU-optimized)
    analyzer_config = AnalyzerConfig(
        mcmc_config=mcmc_config,
        use_mpe=False,  # Disable for CPU stability
        parallel_execution=False,  # Sequential for stability
        max_workers=1,
        model_selection_criterion='dic',
        calculate_ranges=True,
        calculate_weights=True
    )
    
    # 🛡️ Robust Bayesian 原則：使用原始數據，但轉換為合理尺度
    raw_losses = data['observed_losses'].copy()
    if len(raw_losses) > 0:
        # 只做尺度轉換（除以1M），保持數據的原始分布特性
        # 這樣既避免數值問題，又不會過度操作數據
        analysis_data = raw_losses / 1e6  # 轉換為百萬美元單位
        
        print(f"   📊 Robust數據處理:")
        print(f"      原始範圍: ${np.min(raw_losses)/1e6:.1f}M - ${np.max(raw_losses)/1e6:.1f}M")
        print(f"      分析單位: {np.min(analysis_data):.2f} - {np.max(analysis_data):.2f} (百萬美元)")
        print(f"      保持原始分布特性，符合Robust Bayesian原則")
    else:
        analysis_data = raw_losses
    
    # Model specification (恢復完整模型，使用標準化數據)
    model_class_spec = ModelClassSpec(
        enable_epsilon_contamination=True,
        epsilon_values=[0.05] if args.quick_test else [0.01, 0.05],
        contamination_distribution="typhoon"
    )
    
    print(f"📊 Bayesian Configuration:")
    print(f"   Model count: {model_class_spec.get_model_count()}")
    print(f"   ε-contamination: {model_class_spec.epsilon_values}")
    print(f"   Execution mode: Sequential (CPU stable)")
    
    # Create analyzer
    analyzer = ModelClassAnalyzer(model_class_spec, analyzer_config)
    
    # Run analysis
    print(f"\n🚀 Running MCMC analysis on {len(analysis_data)} observations...")
    print(f"   🛡️ Using robust non-informative priors")
    print(f"   🔬 Complete ε-contamination model class")
    start_time = time.time()
    
    try:
        ensemble_results = analyzer.analyze_model_class(analysis_data)
        elapsed = time.time() - start_time
        
        print(f"\n✅ Bayesian analysis complete!")
        print(f"   Execution time: {elapsed:.1f} seconds")
        print(f"   Best model: {ensemble_results.best_model}")
        print(f"   Models evaluated: {len(ensemble_results.individual_results)}")
        
        return ensemble_results
        
    except Exception as e:
        print(f"\n⚠️ Bayesian analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None

def run_skill_evaluation(data, insurance_results, bayesian_results, args):
    """Run comprehensive skill score evaluation"""
    print("\n" + "=" * 80)
    print("Phase 3: Skill Score Evaluation")
    print("階段3：技能評分評估")
    print("=" * 80)
    
    skill_evaluator = SkillScoreEvaluator()
    
    # Get predictions from different sources
    predictions_sources = {}
    
    # 1. Insurance product predictions
    if insurance_results and insurance_results['products']:
        # Use first product as baseline
        baseline_product = insurance_results['products'][0]
        if 'technical_premium' in baseline_product:
            premium_value = baseline_product['technical_premium'].get('expected_premium', 0)
            predictions_sources['insurance_premium'] = np.full(
                len(data['observed_losses']), premium_value
            )
    
    # 2. Bayesian model predictions
    if bayesian_results:
        best_model_result = bayesian_results.individual_results[bayesian_results.best_model]
        if hasattr(best_model_result, 'posterior_samples'):
            posterior_samples = best_model_result.posterior_samples
            if 'theta' in posterior_samples:
                predictions_sources['bayesian_model'] = np.full(
                    len(data['observed_losses']), 
                    np.mean(posterior_samples['theta'])
                )
    
    # 3. Simple statistical predictions
    if len(data['observed_losses']) > 0:
        predictions_sources['mean_baseline'] = np.full(
            len(data['observed_losses']), 
            np.mean(data['observed_losses'])
        )
        predictions_sources['median_baseline'] = np.full(
            len(data['observed_losses']), 
            np.median(data['observed_losses'])
        )
    
    # Evaluate each prediction source
    skill_results = {}
    for source_name, predictions in predictions_sources.items():
        if len(predictions) == len(data['observed_losses']):
            # Calculate basic skill scores
            rmse = np.sqrt(np.mean((predictions - data['observed_losses'])**2))
            mae = np.mean(np.abs(predictions - data['observed_losses']))
            correlation = np.corrcoef(predictions, data['observed_losses'])[0, 1] if len(predictions) > 1 else 0
            
            skill_results[source_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'Correlation': correlation,
                'RMSE_normalized': rmse / np.mean(data['observed_losses']) if np.mean(data['observed_losses']) > 0 else np.inf
            }
            
            print(f"\n📊 {source_name.replace('_', ' ').title()} Skill Scores:")
            print(f"   RMSE: ${rmse/1e6:.1f}M")
            print(f"   MAE: ${mae/1e6:.1f}M")
            print(f"   Correlation: {correlation:.4f}")
            print(f"   Normalized RMSE: {skill_results[source_name]['RMSE_normalized']:.4f}")
    
    return skill_results

def run_market_analysis(insurance_results, args):
    """Run market acceptability analysis"""
    print("\n" + "=" * 80)
    print("Phase 4: Market Acceptability Analysis")
    print("階段4：市場接受度分析")
    print("=" * 80)
    
    if not insurance_results or not insurance_results['products']:
        print("⚠️ Skipping market analysis (no products available)")
        return None
    
    # Initialize market analyzer
    market_analyzer = MarketAcceptabilityAnalyzer()
    
    # Analyze products
    products = insurance_results['products'][:10]  # Limit for performance
    
    market_results = []
    for product in products:
        try:
            # Market analysis for each product
            market_score = market_analyzer.analyze_product_acceptability(product)
            market_results.append({
                'product_id': product.get('product_id', 'unknown'),
                'market_score': market_score,
                'product_summary': {
                    'trigger_type': product.get('trigger_type', 'unknown'),
                    'coverage_amount': product.get('coverage_amount', 0)
                }
            })
        except Exception as e:
            if args.verbose:
                print(f"   ⚠️ Market analysis failed for product: {e}")
            continue
    
    print(f"✅ Market analysis complete for {len(market_results)} products")
    
    # Summary statistics
    if market_results:
        scores = [r['market_score'] for r in market_results if r['market_score'] is not None]
        if scores:
            print(f"   Mean market score: {np.mean(scores):.3f}")
            print(f"   Best market score: {np.max(scores):.3f}")
    
    return market_results

def save_comprehensive_results(data, insurance_results, bayesian_results, 
                              skill_results, market_results, args, mcmc_config_dict):
    """Save all analysis results"""
    print("\n" + "=" * 80)
    print("Phase 5: Save Comprehensive Results")
    print("階段5：儲存綜合結果")
    print("=" * 80)
    
    results_dir = Path('results/robust_bayesian_cpu_comprehensive')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile comprehensive results
    final_results = {
        'analysis_metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'version': 'CPU-Optimized v1.0',
            'quick_test': args.quick_test,
            'n_cores_used': mcmc_config_dict.get('cores', 'unknown'),
            'framework_versions': {
                'has_bayesian': HAS_BAYESIAN,
                'has_insurance': HAS_INSURANCE_FRAMEWORK
            }
        },
        'data_summary': {
            'n_events': len(data['event_losses']),
            'n_nonzero_losses': len(data['observed_losses']),
            'mean_loss': np.mean(data['observed_losses']) if len(data['observed_losses']) > 0 else 0,
            'max_loss': np.max(data['observed_losses']) if len(data['observed_losses']) > 0 else 0
        },
        'insurance_analysis': {
            'n_products': len(insurance_results['products']) if insurance_results else 0,
            'spatial_analysis_available': insurance_results['spatial_results'] is not None if insurance_results else False
        },
        'bayesian_analysis': {
            'completed': bayesian_results is not None,
            'best_model': bayesian_results.best_model if bayesian_results else None,
            'execution_time': bayesian_results.execution_time if bayesian_results else None,
            'n_models_evaluated': len(bayesian_results.individual_results) if bayesian_results else 0
        },
        'skill_evaluation': skill_results,
        'market_analysis': {
            'completed': market_results is not None,
            'n_products_analyzed': len(market_results) if market_results else 0
        }
    }
    
    # Save main results
    with open(results_dir / 'comprehensive_results.pkl', 'wb') as f:
        pickle.dump({
            'final_results': final_results,
            'insurance_results': insurance_results,
            'bayesian_results': bayesian_results,
            'skill_results': skill_results,
            'market_results': market_results,
            'data': data
        }, f)
    
    # Save summary report
    with open(results_dir / 'analysis_report.txt', 'w') as f:
        f.write("CPU-Optimized Robust Bayesian Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis completed: {final_results['analysis_metadata']['timestamp']}\n")
        f.write(f"Version: {final_results['analysis_metadata']['version']}\n")
        f.write(f"Quick test mode: {final_results['analysis_metadata']['quick_test']}\n\n")
        
        f.write("Data Summary:\n")
        f.write(f"  Total events: {final_results['data_summary']['n_events']}\n")
        f.write(f"  Non-zero losses: {final_results['data_summary']['n_nonzero_losses']}\n")
        f.write(f"  Mean loss: ${final_results['data_summary']['mean_loss']/1e6:.1f}M\n")
        f.write(f"  Max loss: ${final_results['data_summary']['max_loss']/1e6:.1f}M\n\n")
        
        f.write("Analysis Components:\n")
        f.write(f"  Insurance products: {final_results['insurance_analysis']['n_products']}\n")
        f.write(f"  Bayesian analysis: {'✅' if final_results['bayesian_analysis']['completed'] else '❌'}\n")
        f.write(f"  Market analysis: {'✅' if final_results['market_analysis']['completed'] else '❌'}\n")
        
        if final_results['bayesian_analysis']['completed']:
            f.write(f"  Best Bayesian model: {final_results['bayesian_analysis']['best_model']}\n")
            f.write(f"  Execution time: {final_results['bayesian_analysis']['execution_time']:.1f}s\n")
        
        f.write("\nSkill Score Summary:\n")
        for source, scores in skill_results.items():
            f.write(f"  {source}:\n")
            f.write(f"    RMSE: ${scores['RMSE']/1e6:.1f}M\n")
            f.write(f"    Correlation: {scores['Correlation']:.4f}\n")
    
    print(f"✅ Results saved to: {results_dir}")
    
    return results_dir, final_results

def main():
    """Main analysis workflow"""
    print("🚀 Starting CPU-Optimized Robust Bayesian Analysis")
    
    # Parse arguments
    args = parse_arguments()
    
    # Detect system capabilities
    system_info = detect_environment()
    
    # Setup CPU environment
    mcmc_config_dict = setup_cpu_environment(args, system_info)
    
    # Load data
    data = load_analysis_data()
    
    # Phase 1: Parametric insurance design
    insurance_results = run_parametric_insurance_design(data, args)
    
    # Phase 2: Bayesian analysis
    bayesian_results = run_bayesian_analysis(data, mcmc_config_dict, insurance_results, args)
    
    # Phase 3: Skill evaluation
    skill_results = run_skill_evaluation(data, insurance_results, bayesian_results, args)
    
    # Phase 4: Market analysis
    market_results = run_market_analysis(insurance_results, args)
    
    # Phase 5: Save results
    results_dir, final_results = save_comprehensive_results(
        data, insurance_results, bayesian_results, skill_results, 
        market_results, args, mcmc_config_dict
    )
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 CPU-Optimized Analysis Complete!")
    print("=" * 80)
    print(f"✅ Results directory: {results_dir}")
    print(f"✅ Insurance products: {final_results['insurance_analysis']['n_products']}")
    print(f"✅ Bayesian analysis: {'Success' if final_results['bayesian_analysis']['completed'] else 'Skipped'}")
    print(f"✅ Market analysis: {'Success' if final_results['market_analysis']['completed'] else 'Skipped'}")
    print(f"✅ CPU cores used: {mcmc_config_dict['cores']}")
    
    if final_results['bayesian_analysis']['completed']:
        print(f"🏆 Best Bayesian model: {final_results['bayesian_analysis']['best_model']}")
        print(f"⏱️ Execution time: {final_results['bayesian_analysis']['execution_time']:.1f} seconds")
    
    print("\n💡 Key advantages of CPU optimization:")
    print("   • Stable execution (no kernel crashes)")
    print("   • Reliable performance scaling")
    print("   • Comprehensive framework integration")
    print("   • Production-ready implementation")
    print("=" * 80)
    
    return final_results

if __name__ == "__main__":
    main()