
# GPU Performance Monitoring
# GPU性能監控

import time
import psutil
import GPUtil  # pip install gputil
from contextlib import contextmanager

@contextmanager
def monitor_gpu_performance():
    """Monitor GPU and CPU usage during MCMC sampling"""
    
    print("📊 Starting performance monitoring...")
    
    start_time = time.time()
    start_cpu = psutil.cpu_percent()
    
    try:
        # Monitor GPU usage
        gpus = GPUtil.getGPUs()
        if len(gpus) >= 2:
            print(f"   🎯 GPU 0: {gpus[0].memoryUtil*100:.1f}% memory, {gpus[0].load*100:.1f}% load")
            print(f"   🎯 GPU 1: {gpus[1].memoryUtil*100:.1f}% memory, {gpus[1].load*100:.1f}% load")
        
        yield
        
    finally:
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        
        duration = end_time - start_time
        
        print(f"\n📈 Performance Summary:")
        print(f"   ⏱️ Total time: {duration/60:.1f} minutes")
        print(f"   💻 CPU usage: {end_cpu:.1f}%")
        
        gpus = GPUtil.getGPUs()
        if len(gpus) >= 2:
            print(f"   🎯 Final GPU 0: {gpus[0].memoryUtil*100:.1f}% memory")
            print(f"   🎯 Final GPU 1: {gpus[1].memoryUtil*100:.1f}% memory")

# Usage example:
# with monitor_gpu_performance():
#     trace = pm.sample(**DUAL_GPU_SAMPLER_KWARGS)
        