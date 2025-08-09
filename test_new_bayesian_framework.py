"""
Test script for the new Bayesian framework
測試新的貝葉斯框架

This script demonstrates the integrated Method 1 (Model Comparison) and 
Method 2 (Bayesian Decision Theory) frameworks.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_sample_data(n_observations=100, random_seed=42):
    """Generate sample loss and hazard data for testing"""
    
    np.random.seed(random_seed)
    
    # Generate hazard indices (e.g., wind speeds)
    hazard_indices = np.random.uniform(25, 65, n_observations)
    
    # Generate losses based on hazard indices with some noise
    base_losses = np.zeros_like(hazard_indices)
    
    # Threshold-based loss generation
    for i, hazard in enumerate(hazard_indices):
        if hazard < 35:
            # Minor events - mostly zero losses
            base_losses[i] = np.random.exponential(1e6) if np.random.random() < 0.2 else 0
        elif hazard < 45:
            # Moderate events
            base_losses[i] = np.random.lognormal(np.log(5e7), 0.5)
        elif hazard < 55:
            # Severe events
            base_losses[i] = np.random.lognormal(np.log(1.5e8), 0.6)
        else:
            # Extreme events
            base_losses[i] = np.random.lognormal(np.log(4e8), 0.8)
    
    # Add some additional noise
    noise_factor = np.random.lognormal(0, 0.1, n_observations)
    observed_losses = base_losses * noise_factor
    
    return observed_losses, hazard_indices

def test_model_comparison_framework():
    """Test Method 1: Model Comparison Framework"""
    
    print("🔬 測試方法一：模型比較框架")
    print("=" * 60)
    
    try:
        from bayesian import BayesianModelComparison
        
        # Generate sample data
        observed_losses, hazard_indices = generate_sample_data(50)  # Smaller dataset for testing
        
        # Split into train/validation
        n_train = 40
        train_losses = observed_losses[:n_train]
        val_losses = observed_losses[n_train:]
        train_indices = hazard_indices[:n_train]
        val_indices = hazard_indices[n_train:]
        
        print(f"📊 數據: 訓練({n_train}) / 驗證({len(val_losses)})")
        
        # Initialize model comparison
        model_comparison = BayesianModelComparison(
            n_samples=200,  # Small for testing
            n_chains=2,
            random_seed=42
        )
        
        # Prepare model arguments
        model_kwargs = {
            'wind_speed': train_indices,
            'rainfall': None,
            'storm_surge': None
        }
        
        # Run model comparison
        results = model_comparison.fit_all_models(
            train_data=train_losses,
            validation_data=val_losses,
            **model_kwargs
        )
        
        # Get best model
        best_model = model_comparison.get_best_model()
        
        if best_model:
            print(f"\n✅ 方法一測試成功")
            print(f"   最佳模型: {best_model.model_name}")
            print(f"   CRPS 分數: {best_model.crps_score:.2e}")
        else:
            print("\n❌ 方法一測試失敗：未找到有效模型")
            
        return best_model, train_losses, train_indices
        
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        return None, None, None

def test_decision_theory_framework(best_model, train_losses, train_indices):
    """Test Method 2: Bayesian Decision Theory Framework"""
    
    print("\n🎯 測試方法二：貝葉斯決策理論框架")
    print("=" * 60)
    
    if best_model is None:
        print("⚠️ 跳過方法二測試（方法一未成功）")
        return
    
    try:
        from bayesian import BayesianDecisionTheory, BasisRiskLossFunction, BasisRiskType, ProductParameters
        
        # Create loss function
        loss_function = BasisRiskLossFunction(
            risk_type=BasisRiskType.WEIGHTED_ASYMMETRIC,
            w_under=2.0,
            w_over=0.5
        )
        
        # Initialize decision theory framework
        decision_theory = BayesianDecisionTheory(
            loss_function=loss_function,
            random_seed=42
        )
        
        # Extract posterior samples (simplified)
        # In real implementation, this would extract from the best model's trace
        posterior_samples = np.random.normal(np.log(1e8), 0.5, 100)
        print(f"📊 使用 {len(posterior_samples)} 個後驗樣本")
        
        # Simulate actual losses
        actual_losses_matrix = decision_theory.simulate_actual_losses(
            posterior_samples=posterior_samples,
            hazard_indices=train_indices
        )
        
        print(f"📊 生成真實損失矩陣: {actual_losses_matrix.shape}")
        
        # Define optimization bounds
        product_bounds = {
            'trigger_threshold': (35, 55),
            'payout_amount': (5e7, 3e8),
            'max_payout': (5e8, 5e8)
        }
        
        # Optimize product
        optimization_result = decision_theory.optimize_single_product(
            posterior_samples=posterior_samples,
            hazard_indices=train_indices,
            actual_losses=actual_losses_matrix,
            product_bounds=product_bounds
        )
        
        print(f"\n✅ 方法二測試成功")
        print(f"   最優觸發閾值: {optimization_result.optimal_product.trigger_threshold:.2f}")
        print(f"   最優賠付金額: ${optimization_result.optimal_product.payout_amount:.2e}")
        print(f"   期望基差風險: ${optimization_result.expected_loss:.2e}")
        
        return optimization_result
        
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        return None
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        return None

def test_integrated_framework():
    """Test the integrated framework through RobustBayesianAnalyzer"""
    
    print("\n🧠 測試整合框架：RobustBayesianAnalyzer")
    print("=" * 60)
    
    try:
        from bayesian import RobustBayesianAnalyzer
        
        # Generate sample data
        observed_losses, hazard_indices = generate_sample_data(80)
        
        print(f"📊 數據: {len(observed_losses)} 個觀測值")
        
        # Initialize analyzer
        analyzer = RobustBayesianAnalyzer(
            density_ratio_constraint=2.0,
            n_monte_carlo_samples=100,  # Small for testing
            n_mixture_components=2
        )
        
        # Create sample parametric products
        parametric_products = [
            {
                'product_id': 'test_product_1',
                'wind_threshold': 40,
                'payout_rate': 0.5,
                'max_payout': 2e8,
                'type': 'single_threshold'
            },
            {
                'product_id': 'test_product_2', 
                'wind_threshold': 45,
                'payout_rate': 0.7,
                'max_payout': 3e8,
                'type': 'single_threshold'
            }
        ]
        
        # Run comprehensive analysis
        results = analyzer.comprehensive_bayesian_analysis(
            tc_hazard=None,  # Mock data
            exposure_main=None,  # Mock data
            impact_func_set=None,  # Mock data
            observed_losses=observed_losses,
            parametric_products=parametric_products,
            hazard_indices=hazard_indices
        )
        
        print(f"\n✅ 整合框架測試成功")
        print(f"   框架版本: {results['meta_analysis']['framework_version']}")
        print(f"   使用方法: {results['meta_analysis']['methods_used']}")
        
        if results['meta_analysis']['best_model_name']:
            print(f"   最佳模型: {results['meta_analysis']['best_model_name']}")
            
        if results['meta_analysis']['optimal_product']:
            optimal = results['meta_analysis']['optimal_product']
            print(f"   最優產品觸發閾值: {optimal['trigger_threshold']:.2f}")
            print(f"   最優產品賠付金額: ${optimal['payout_amount']:.2e}")
        
        return results
        
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        return None
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        return None

def main():
    """Main test function"""
    
    print("🧪 新貝葉斯框架測試")
    print("=" * 80)
    print("實現 bayesian_implement.md 中的方法一和方法二")
    print()
    
    # Test Method 1
    best_model, train_losses, train_indices = test_model_comparison_framework()
    
    # Test Method 2
    optimization_result = test_decision_theory_framework(best_model, train_losses, train_indices)
    
    # Test Integrated Framework
    integrated_results = test_integrated_framework()
    
    # Summary
    print("\n📋 測試總結")
    print("=" * 40)
    
    if best_model:
        print("✅ 方法一（模型比較）: 成功")
    else:
        print("❌ 方法一（模型比較）: 失敗")
    
    if optimization_result:
        print("✅ 方法二（決策理論）: 成功")
    else:
        print("❌ 方法二（決策理論）: 失敗")
    
    if integrated_results:
        print("✅ 整合框架: 成功")
    else:
        print("❌ 整合框架: 失敗")
    
    print("\n🎉 測試完成")

if __name__ == "__main__":
    main()