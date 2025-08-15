#!/usr/bin/env python3
"""
HPC GPU 診斷腳本
HPC GPU Diagnostic Script

運行此腳本在 HPC 上診斷 GPU 檢測問題
Run this script on HPC to diagnose GPU detection issues
"""

import os
import sys
import subprocess

def check_nvidia_tools():
    """檢查 NVIDIA 工具可用性"""
    print("🔧 檢查 NVIDIA 工具...")
    
    # 檢查 nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi 可用")
            print("GPU 信息:")
            print(result.stdout[:500])  # 前500字符
        else:
            print(f"❌ nvidia-smi 執行失敗: {result.stderr}")
    except FileNotFoundError:
        print("❌ nvidia-smi 未找到")
    except Exception as e:
        print(f"❌ nvidia-smi 錯誤: {e}")
    
    # 檢查 nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ nvcc 可用")
            print(result.stdout.strip())
        else:
            print("❌ nvcc 不可用")
    except FileNotFoundError:
        print("❌ nvcc 未找到")
    except Exception as e:
        print(f"❌ nvcc 錯誤: {e}")

def check_environment_variables():
    """檢查 CUDA 相關環境變數"""
    print("\n🌍 檢查環境變數...")
    
    cuda_vars = [
        'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_HOME', 'CUDA_PATH',
        'LD_LIBRARY_PATH', 'PATH', 'JAX_PLATFORM_NAME', 'PYTENSOR_FLAGS'
    ]
    
    for var in cuda_vars:
        value = os.environ.get(var, '未設置')
        print(f"   {var}: {value}")

def check_python_packages():
    """檢查 Python 套件"""
    print("\n📦 檢查 Python 套件...")
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        relevant_packages = []
        for line in lines:
            if any(pkg in line.lower() for pkg in ['jax', 'cuda', 'cupy', 'torch', 'tensorflow']):
                relevant_packages.append(line)
        
        if relevant_packages:
            print("相關套件:")
            for pkg in relevant_packages:
                print(f"   {pkg}")
        else:
            print("❌ 沒有找到相關 GPU 套件")
            
    except Exception as e:
        print(f"❌ 套件檢查錯誤: {e}")

def check_jax_installation():
    """詳細檢查 JAX 安裝"""
    print("\n🔍 詳細檢查 JAX...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        print(f"✅ JAX 版本: {jax.__version__}")
        print(f"✅ JAX 後端: {jax.default_backend()}")
        
        # 檢查設備
        devices = jax.devices()
        print(f"📱 JAX 設備: {devices}")
        
        for i, device in enumerate(devices):
            print(f"   設備 {i}:")
            print(f"     ID: {device.id}")
            print(f"     類型: {device.device_kind}")
            print(f"     平台: {device.platform}")
        
        # 嘗試設置 GPU 平台
        print("\n🎯 嘗試強制 GPU 模式...")
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
        
        # 重新導入
        import importlib
        importlib.reload(jax)
        
        new_devices = jax.devices()
        print(f"強制 GPU 後設備: {new_devices}")
        
        # 測試計算
        try:
            x = jnp.array([1, 2, 3, 4])
            y = jnp.sum(x)
            print(f"✅ JAX 計算測試: {y}")
            print(f"計算設備: {y.device()}")
        except Exception as e:
            print(f"❌ JAX 計算測試失敗: {e}")
        
        # 恢復設置
        if 'JAX_PLATFORM_NAME' in os.environ:
            del os.environ['JAX_PLATFORM_NAME']
        
    except ImportError as e:
        print(f"❌ JAX 導入失敗: {e}")
    except Exception as e:
        print(f"❌ JAX 檢查錯誤: {e}")

def check_pytorch_cuda():
    """檢查 PyTorch CUDA 支援"""
    print("\n🔥 檢查 PyTorch CUDA...")
    
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU 數量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   記憶體: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        else:
            print("❌ PyTorch 無法檢測到 CUDA")
            
    except ImportError:
        print("❌ PyTorch 未安裝")
    except Exception as e:
        print(f"❌ PyTorch 檢查錯誤: {e}")

def main():
    """主診斷函數"""
    print("=" * 80)
    print("🚀 HPC GPU 診斷腳本")
    print("=" * 80)
    
    check_nvidia_tools()
    check_environment_variables() 
    check_python_packages()
    check_jax_installation()
    check_pytorch_cuda()
    
    print("\n" + "=" * 80)
    print("🎯 診斷建議:")
    print("=" * 80)
    print("如果 nvidia-smi 可用但 JAX 檢測不到 GPU:")
    print("1. 檢查 JAX 安裝: pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    print("2. 檢查 CUDA 版本兼容性")
    print("3. 檢查 LD_LIBRARY_PATH 包含 CUDA 庫路徑")
    print("4. 在 HPC 上可能需要載入 CUDA 模組: module load cuda")
    print("\n如果是 SLURM 系統，確保:")
    print("1. 請求了 GPU 資源: srun --gres=gpu:2")
    print("2. CUDA_VISIBLE_DEVICES 已正確設置")

if __name__ == "__main__":
    main()