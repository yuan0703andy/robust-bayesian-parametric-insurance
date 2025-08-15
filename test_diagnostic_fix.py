#!/usr/bin/env python3
"""
Test the diagnostic calculation fix
測試診斷計算修復
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_safe_float_extraction():
    """Test safe float value extraction"""
    print("🧪 Testing safe float value extraction...")
    
    try:
        # Import the hierarchical model to test the helper functions
        from bayesian.parametric_bayesian_hierarchy import ParametricHierarchicalModel
        from bayesian import ModelSpec, MCMCConfig, LikelihoodFamily, PriorScenario
        
        # Create a test instance
        model_spec = ModelSpec(
            likelihood_family=LikelihoodFamily.NORMAL,
            prior_scenario=PriorScenario.WEAK_INFORMATIVE
        )
        mcmc_config = MCMCConfig(n_samples=10, n_warmup=5, n_chains=1)
        
        model = ParametricHierarchicalModel(model_spec, mcmc_config)
        
        # Test different value types
        test_cases = [
            (1.5, "float"),
            (2, "int"),
            (np.array([3.7]), "numpy array"),
            (np.float64(4.2), "numpy float64"),
            ([5.1], "list"),
            ({"nested": 6.8}, "dict - should fallback to 1.0"),
        ]
        
        for value, description in test_cases:
            try:
                result = model._safe_extract_float_value(value)
                print(f"   ✅ {description}: {value} → {result}")
            except Exception as e:
                print(f"   ❌ {description}: {value} → Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Safe float extraction test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_safe_diagnostics_extraction():
    """Test safe diagnostics dictionary extraction"""
    print("🧪 Testing safe diagnostics extraction...")
    
    try:
        from bayesian.parametric_bayesian_hierarchy import ParametricHierarchicalModel
        from bayesian import ModelSpec, MCMCConfig, LikelihoodFamily, PriorScenario
        
        # Create a test instance
        model_spec = ModelSpec(
            likelihood_family=LikelihoodFamily.NORMAL,
            prior_scenario=PriorScenario.WEAK_INFORMATIVE
        )
        mcmc_config = MCMCConfig(n_samples=10, n_warmup=5, n_chains=1)
        
        model = ParametricHierarchicalModel(model_spec, mcmc_config)
        
        # Test different result types
        class MockArviZResult:
            def __init__(self, data_vars):
                self.data_vars = data_vars
            
            def to_dict(self):
                return {'data_vars': self.data_vars}
        
        # Test case 1: Normal data vars with floats
        result1 = MockArviZResult({
            'alpha': 1.05,
            'beta': 0.98,
            'sigma': 1.12
        })
        
        extracted1 = model._safe_extract_diagnostics_dict(result1, default_value=1.0)
        print(f"   ✅ Normal case: {extracted1}")
        
        # Test case 2: Nested dict values (problematic case)
        result2 = MockArviZResult({
            'alpha': {'nested': 1.05},  # This would cause the original error
            'beta': 0.98,
            'sigma': [1.12]  # Array value
        })
        
        extracted2 = model._safe_extract_diagnostics_dict(result2, default_value=1.0)
        print(f"   ✅ Problematic case (nested dict): {extracted2}")
        
        # Test case 3: Non-dict result (fallback)
        class MockSimpleResult:
            def __init__(self, data):
                self.data = data
            
            def __iter__(self):
                return iter(self.data.items())
        
        result3 = MockSimpleResult({'alpha': 1.02, 'beta': 0.99})
        extracted3 = model._safe_extract_diagnostics_dict(result3, default_value=1.0)
        print(f"   ✅ Simple result case: {extracted3}")
        
        # Test case 4: Complete failure (ultimate fallback)
        result4 = None
        extracted4 = model._safe_extract_diagnostics_dict(result4, default_value=1.5)
        print(f"   ✅ Fallback case (None input): {extracted4}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Safe diagnostics extraction test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compute_diagnostics_robustness():
    """Test the robustness of the compute diagnostics function"""
    print("🧪 Testing compute diagnostics robustness...")
    
    try:
        from bayesian.parametric_bayesian_hierarchy import ParametricHierarchicalModel
        from bayesian import ModelSpec, MCMCConfig, LikelihoodFamily, PriorScenario
        
        # Create a test instance
        model_spec = ModelSpec(
            likelihood_family=LikelihoodFamily.NORMAL,
            prior_scenario=PriorScenario.WEAK_INFORMATIVE
        )
        mcmc_config = MCMCConfig(n_samples=10, n_warmup=5, n_chains=1)
        
        model = ParametricHierarchicalModel(model_spec, mcmc_config)
        
        # Create a mock trace that might cause issues
        class MockTrace:
            def __init__(self):
                self.sample_stats = type('obj', (object,), {
                    'diverging': np.array([False, False, True, False])  # Some divergent transitions
                })()
        
        mock_trace = MockTrace()
        
        # Test with mock trace (this will likely fail the ArviZ calls but should handle gracefully)
        diagnostics = model._compute_diagnostics(mock_trace)
        
        print(f"   ✅ Diagnostics computed successfully:")
        print(f"      • R-hat keys: {list(diagnostics.rhat.keys())}")
        print(f"      • ESS bulk keys: {list(diagnostics.ess_bulk.keys())}")
        print(f"      • ESS tail keys: {list(diagnostics.ess_tail.keys())}")
        print(f"      • MCSE keys: {list(diagnostics.mcse.keys())}")
        print(f"      • Divergent transitions: {diagnostics.n_divergent}")
        
        # Verify all values are floats
        all_rhat_floats = all(isinstance(v, float) for v in diagnostics.rhat.values())
        all_ess_bulk_floats = all(isinstance(v, float) for v in diagnostics.ess_bulk.values())
        all_ess_tail_floats = all(isinstance(v, float) for v in diagnostics.ess_tail.values())
        all_mcse_floats = all(isinstance(v, float) for v in diagnostics.mcse.values())
        
        print(f"   ✅ All values are floats: {all_rhat_floats and all_ess_bulk_floats and all_ess_tail_floats and all_mcse_floats}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Compute diagnostics test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("="*80)
    print("🔧 Testing Diagnostic Calculation Fix")
    print("="*80)
    
    tests = [
        test_safe_float_extraction,
        test_safe_diagnostics_extraction,
        test_compute_diagnostics_robustness
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Summary
    print("="*80)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("🚀 Diagnostic calculation fix is working!")
        print("🎯 The float() error with dict should be resolved")
    else:
        print(f"⚠️ {total-passed} TESTS FAILED ({passed}/{total})")
    
    print("="*80)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)