#!/usr/bin/env python3
"""
Test ε-contamination Parameter Fix
測試 ε-污染參數修正
"""

import numpy as np
import sys
import os

# Configure environment
os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float32,optimizer=fast_compile'

print("🧪 Testing ε-contamination Parameter Fix")
print("=" * 50)

try:
    # Test EpsilonContaminationSpec creation
    print("\n📦 Testing EpsilonContaminationSpec...")
    from bayesian.epsilon_contamination import EpsilonContaminationSpec, create_typhoon_contamination_spec
    
    # Test 1: Direct creation with correct parameters
    spec1 = EpsilonContaminationSpec(
        epsilon_range=(0.01, 0.05),
        nominal_prior_family="normal",
        contamination_prior_family="gev"
    )
    print(f"   ✅ Direct creation successful: epsilon_range = {spec1.epsilon_range}")
    
    # Test 2: Factory function
    spec2 = create_typhoon_contamination_spec((0.02, 0.08))
    print(f"   ✅ Factory creation successful: epsilon_range = {spec2.epsilon_range}")
    
except Exception as e:
    print(f"   ❌ EpsilonContaminationSpec test failed: {e}")
    
try:
    # Test model ensemble analyzer
    print("\n🔬 Testing ModelClassAnalyzer...")
    from bayesian.robust_model_ensemble_analyzer import ModelClassSpec
    
    # Test with epsilon contamination enabled
    model_spec = ModelClassSpec(
        enable_epsilon_contamination=True,
        epsilon_values=[0.01, 0.05],
        contamination_distribution="typhoon"
    )
    
    print(f"   ✅ ModelClassSpec creation successful")
    print(f"       - ε-contamination enabled: {model_spec.enable_epsilon_contamination}")
    print(f"       - ε values: {model_spec.epsilon_values}")
    print(f"       - Total models: {model_spec.get_model_count()}")
    
except Exception as e:
    print(f"   ❌ ModelClassSpec test failed: {e}")
    
try:
    # Test model spec generation
    print("\n⚙️ Testing model spec generation...")
    from bayesian.robust_model_ensemble_analyzer import ModelClassAnalyzer, AnalyzerConfig, MCMCConfig
    
    # Create minimal configuration
    mcmc_config = MCMCConfig(
        n_samples=100,  # Minimal for testing
        n_warmup=50,
        n_chains=1,
        target_accept=0.8
    )
    
    analyzer_config = AnalyzerConfig(
        mcmc_config=mcmc_config,
        use_mpe=False,  # Disable for testing
        parallel_execution=False,  # Single thread for testing
        max_workers=1
    )
    
    analyzer = ModelClassAnalyzer(model_spec, analyzer_config)
    
    # Generate model specifications
    specs = analyzer.model_class_spec.generate_all_specs()
    print(f"   ✅ Generated {len(specs)} model specifications")
    
    # Check for contamination models
    contamination_models = [spec for spec in specs if 'epsilon' in spec.model_name]
    print(f"   ✅ Found {len(contamination_models)} contamination models")
    
    if contamination_models:
        sample_model = contamination_models[0]
        print(f"       Sample model: {sample_model.model_name}")
    
except Exception as e:
    print(f"   ❌ Model specification generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("🎯 Test Summary:")
print("   The ε-contamination parameter fixes should resolve:")
print("   ❌ EpsilonContaminationSpec.__init__() got unexpected keyword 'epsilon'")
print("   ✅ Now using correct 'epsilon_range' parameter")
print("   ✅ Simplified contamination application to avoid complex dependencies")
print("=" * 50)