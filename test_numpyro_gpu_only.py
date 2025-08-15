#!/usr/bin/env python3
"""
NumPyro-only GPU MCMC test - avoiding PyMC compatibility issues
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import time

def setup_gpu_environment():
    """Setup JAX for GPU usage"""
    print("🔧 Setting up GPU environment...")
    
    # JAX GPU configuration
    os.environ['JAX_PLATFORMS'] = 'cuda'
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['NUMPYRO_PLATFORM'] = 'gpu'
    
    # Print confirmation
    for key in ['JAX_PLATFORMS', 'JAX_PLATFORM_NAME', 'XLA_FLAGS', 
                'CUDA_VISIBLE_DEVICES', 'XLA_PYTHON_CLIENT_MEM_FRACTION', 
                'XLA_PYTHON_CLIENT_PREALLOCATE', 'NUMPYRO_PLATFORM']:
        print(f"   ✅ {key} = {os.environ.get(key)}")

def check_jax_gpu():
    """Check JAX GPU detection"""
    print("\n🔍 Testing JAX GPU detection...")
    
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.device_kind == 'gpu']
    
    print(f"   📊 JAX devices: {devices}")
    print(f"   🎯 GPU devices: {gpu_devices}")
    
    if len(gpu_devices) >= 2:
        print("   ✅ Dual GPU detected by JAX!")
        return True
    elif len(gpu_devices) == 1:
        print("   ⚠️  Single GPU detected by JAX")
        return True
    else:
        print("   ❌ No GPU devices detected by JAX")
        return False

def bayesian_regression_model(X, y=None):
    """Simple Bayesian regression model for testing"""
    # Priors
    alpha = numpyro.sample("alpha", dist.Normal(0, 1))
    beta = numpyro.sample("beta", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    
    # Mean prediction
    mu = alpha + beta * X
    
    # Likelihood
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

def test_numpyro_gpu_sampling():
    """Test NumPyro GPU sampling performance"""
    print("\n🧪 Testing NumPyro GPU sampling...")
    
    try:
        # Generate synthetic data
        np.random.seed(42)
        n_data = 1000
        X = np.random.normal(0, 1, n_data)
        y = 2.0 + 3.0 * X + np.random.normal(0, 0.5, n_data)
        
        # Convert to JAX arrays
        X_jax = jnp.array(X)
        y_jax = jnp.array(y)
        
        # Setup MCMC
        rng_key = random.PRNGKey(42)
        kernel = NUTS(bayesian_regression_model)
        mcmc = MCMC(
            kernel, 
            num_warmup=500, 
            num_samples=1000,
            num_chains=2,  # Use multiple chains for GPU
            chain_method='parallel'  # Parallel chains on GPU
        )
        
        print("   🚀 Starting MCMC sampling...")
        start_time = time.time()
        
        # Run MCMC
        mcmc.run(rng_key, X_jax, y_jax)
        
        sampling_time = time.time() - start_time
        print(f"   ✅ NumPyro GPU sampling completed in {sampling_time:.2f} seconds")
        
        # Get samples
        samples = mcmc.get_samples()
        print(f"   📊 Sample shapes: {[(k, v.shape) for k, v in samples.items()]}")
        
        # Print summary statistics
        for param, values in samples.items():
            mean_val = jnp.mean(values)
            std_val = jnp.std(values)
            print(f"   📈 {param}: mean={mean_val:.3f}, std={std_val:.3f}")
        
        return True, sampling_time
        
    except Exception as e:
        print(f"   ❌ NumPyro GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def run_comprehensive_gpu_test():
    """Run comprehensive GPU testing"""
    print("=" * 60)
    print("🚀 NumPyro-Only GPU MCMC Test")
    print("=" * 60)
    
    # Setup GPU environment
    setup_gpu_environment()
    
    # Check JAX GPU detection
    gpu_available = check_jax_gpu()
    
    if not gpu_available:
        print("\n❌ No GPU detected. Exiting test.")
        return
    
    # Test NumPyro GPU sampling
    success, sampling_time = test_numpyro_gpu_sampling()
    
    print("\n" + "=" * 60)
    print("🔍 GPU Test Results:")
    if success:
        print(f"   ✅ NumPyro GPU sampling: {sampling_time:.2f}s")
        print("   💡 Check nvidia-smi during sampling for GPU usage")
    else:
        print("   ❌ NumPyro GPU sampling failed")
    
    print("=" * 60)
    
    print("\n💡 Monitoring tips:")
    print("   1. Run 'nvidia-smi' in another terminal during sampling")
    print("   2. Expected: GPU usage > 50% during MCMC")
    print("   3. Memory usage should increase during sampling")
    print("   4. Multiple GPU utilization if dual GPU setup works")

if __name__ == "__main__":
    run_comprehensive_gpu_test()