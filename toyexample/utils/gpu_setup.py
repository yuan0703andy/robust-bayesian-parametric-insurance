from __future__ import annotations
import torch

# GPU配置檢測
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"🚀 檢測到 {gpu_count} 個GPU:")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # 設置默認設備
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    
    # 為雙GPU訓練設置
    if gpu_count >= 2:
        print("✅ 啟用雙GPU並行訓練模式")
        USE_MULTI_GPU = True
        GPU_DEVICES = [0, 1]  # 使用GPU 0和1
    else:
        print("⚠️ 僅檢測到單GPU，使用單GPU模式")
        USE_MULTI_GPU = False
        GPU_DEVICES = [0]
else:
    print("⚠️ 未檢測到GPU，使用CPU模式")
    device = torch.device("cpu")
    USE_MULTI_GPU = False
    GPU_DEVICES = []
    
print("🚀 環境初始化完成")