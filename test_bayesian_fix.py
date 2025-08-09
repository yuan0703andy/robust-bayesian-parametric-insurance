#!/usr/bin/env python3
"""
Test script for fixed Bayesian implementation
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append('/hpc/home/yh421/CAT_INSURANCE')

def test_hierarchical_bayesian_model():
    """Test the fixed hierarchical Bayesian model"""
    
    print("🧪 Testing Fixed Hierarchical Bayesian Model")
    print("=" * 50)
    
    try:
        # Import the fixed module
        from bayesian.hierarchical_bayesian_model import HierarchicalBayesianModel, HierarchicalModelConfig
        
        print("✅ Successfully imported hierarchical Bayesian model")
        
        # Generate test data
        np.random.seed(42)
        test_observations = np.random.normal(loc=100, scale=20, size=50)  # Simulate some loss data
        
        print(f"📊 Generated {len(test_observations)} test observations")
        print(f"   Mean: {np.mean(test_observations):.2f}")
        print(f"   Std:  {np.std(test_observations):.2f}")
        
        # Create model configuration
        config = HierarchicalModelConfig(
            n_chains=2,  # Reduce for faster testing
            n_samples=200,  # Reduce for faster testing
            n_warmup=100,   # Reduce for faster testing
            n_mixture_components=2  # Reduce for faster testing
        )
        
        print("⚙️ Created model configuration")
        
        # Initialize model
        model = HierarchicalBayesianModel(config)
        print("🏗️ Initialized hierarchical Bayesian model")
        
        # Fit model
        print("\n🔄 Fitting model...")
        result = model.fit(test_observations)
        
        print(f"✅ Model fitting completed!")
        print(f"   Log-likelihood: {result.log_likelihood:.2f}")
        print(f"   DIC: {result.dic:.2f}")
        print(f"   WAIC: {result.waic:.2f}")
        print(f"   Posterior samples keys: {list(result.posterior_samples.keys())}")
        print(f"   MPE components: {len(result.mpe_components)} variables")
        
        # Test model summary
        summary = model.get_model_summary()
        print(f"\n📋 Model Summary:")
        if not summary.empty:
            print(summary.to_string())
        else:
            print("   No summary available")
        
        # Test predictions
        print("\n🔮 Testing predictions...")
        predictions = model.predict(n_predictions=100)
        print(f"   Generated predictions for {len(predictions)} variables")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_robust_bayesian_analyzer():
    """Test the robust Bayesian analyzer"""
    
    print("\n🧠 Testing Robust Bayesian Analyzer")
    print("=" * 50)
    
    try:
        from bayesian.robust_bayesian_analyzer import RobustBayesianAnalyzer
        
        print("✅ Successfully imported robust Bayesian analyzer")
        
        # Initialize analyzer with reduced parameters for testing
        analyzer = RobustBayesianAnalyzer(
            n_monte_carlo_samples=100,  # Reduce for faster testing
            n_mixture_components=2
        )
        
        print("🔧 Initialized robust Bayesian analyzer")
        
        # Generate test data
        np.random.seed(42)
        test_losses = np.random.exponential(scale=50, size=30)  # Exponential loss data
        
        print(f"📊 Generated {len(test_losses)} test loss observations")
        
        # Test individual components
        print("\n🔍 Testing robust analysis component...")
        robust_results = analyzer._perform_robust_analysis(test_losses)
        print(f"   ✅ Robust analysis completed")
        
        print("\n🏗️ Testing hierarchical model component...")
        hierarchical_results = analyzer._perform_hierarchical_analysis(test_losses)
        print(f"   ✅ Hierarchical analysis completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Analyzer test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Bayesian Model Tests")
    print("=" * 60)
    
    # Test 1: Hierarchical Bayesian Model
    test1_success = test_hierarchical_bayesian_model()
    
    # Test 2: Robust Bayesian Analyzer  
    test2_success = test_robust_bayesian_analyzer()
    
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print(f"   Hierarchical Model: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"   Robust Analyzer:    {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! PyMC compatibility issues resolved.")
    else:
        print("\n⚠️ Some tests failed. Check error messages above.")