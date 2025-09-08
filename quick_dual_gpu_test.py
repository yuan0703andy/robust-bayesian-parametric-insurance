#!/usr/bin/env python3
"""
簡單直接的雙GPU測試
修復所有變量定義問題
"""

# 基礎導入
import numpy as np
import torch
import time

# 設置隨機種子
np.random.seed(42)
torch.manual_seed(42)

# GPU檢測
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"🚀 檢測到 {gpu_count} 個GPU")
    USE_MULTI_GPU = gpu_count >= 2
    GPU_DEVICES = [0, 1] if gpu_count >= 2 else [0]
    
    for i in range(min(2, gpu_count)):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
else:
    print("❌ 無GPU可用")
    USE_MULTI_GPU = False
    GPU_DEVICES = []

# 從主文件加載必要的類
print("\n📁 載入主要類別定義...")
try:
    exec(open('toy_example_complete.py').read())
    print("✅ 類別定義載入成功")
except Exception as e:
    print(f"❌ 載入失敗: {e}")
    exit(1)

# 創建數據
print("\n📊 生成測試數據...")
generator = ToyDataGenerator(n_hospitals=20, n_events=50, n_regions=4)
climada_data = generator.generate_climada_data()
spatial_data = generator.generate_spatial_data(climada_data.hospital_coords)

# 準備訓練數據
n_events = climada_data.n_events
n_train = int(0.7 * n_events)
event_indices = np.random.permutation(n_events)
train_indices = event_indices[:n_train]

train_hazards = torch.tensor(climada_data.hazard_intensities[:, train_indices], dtype=torch.float32)
train_losses = torch.tensor(climada_data.observed_losses[:, train_indices], dtype=torch.float32) 
exposure_tensor = torch.tensor(climada_data.exposure_values, dtype=torch.float32)

print(f"✅ 數據準備完成: {generator.n_hospitals}家醫院, {train_hazards.shape[1]}個事件")

# 配置模型
product_config = {
    'name': 'Multi-Level階梯保險',
    'trigger_threshold': 30.0,
    'max_payout': 10e6,
    'steepness': 0.1
}

print("\n🧠 創建雙GPU模型...")
model = UnifiedEndToEndVIModel(
    n_hospitals=generator.n_hospitals,
    n_regions=generator.n_regions,
    n_events=train_hazards.shape[1],
    distance_matrix=spatial_data.distance_matrix,
    product_config=product_config,
    n_hbm_params=7
)

# 創建訓練器
trainer = EndToEndTrainer(model, learning_rate=0.01, enable_multi_gpu=USE_MULTI_GPU)

print("✅ 模型和訓練器創建成功")
print(f"   多GPU模式: {'啟用' if USE_MULTI_GPU else '停用'}")

# 快速訓練測試
print("\n🔥 執行訓練測試...")

try:
    # 執行3個epoch的快速測試
    for epoch in range(3):
        start_time = time.time()
        
        loss_dict = trainer.train_epoch(
            train_hazards,
            exposure_tensor,
            train_losses.mean(dim=1),  # 使用平均損失
            n_samples=5  # 少量樣本加快測試
        )
        
        epoch_time = time.time() - start_time
        print(f"  Epoch {epoch+1}: Loss={loss_dict['total_loss']:.4f}, Time={epoch_time:.3f}s")
    
    # 獲取性能統計
    stats = trainer.get_performance_stats()
    print(f"\n📊 性能統計:")
    print(f"   多GPU啟用: {stats['multi_gpu_enabled']}")
    print(f"   平均Epoch時間: {stats['avg_epoch_time']:.3f}秒")
    print(f"   使用設備: {stats['device']}")
    
    if 'gpu_memory_mb' in stats:
        print(f"   GPU記憶體: {stats['gpu_memory_mb']['current']}MB")
    
    print("✅ 雙GPU並行變分推斷測試成功！")

except Exception as e:
    print(f"❌ 訓練測試失敗: {e}")
    import traceback
    traceback.print_exc()

# 並行CRPS測試
print("\n🔧 測試並行CRPS計算...")

try:
    crps_computer = ParallelCRPSComputer(use_multi_gpu=USE_MULTI_GPU)
    
    # 生成測試數據
    test_losses = torch.rand(100) * 1e8
    test_params = {
        'mu_payout_log': torch.randn(100) + 16,
        'sigma_payout_log': torch.ones(100) * 0.5
    }
    
    start_time = time.time()
    crps_scores = crps_computer.compute_crps_parallel(test_losses, test_params, n_pred_samples=50)
    compute_time = time.time() - start_time
    
    print(f"✅ CRPS計算成功:")
    print(f"   計算時間: {compute_time:.3f}秒")
    print(f"   平均CRPS: {crps_scores.mean():.2f}")
    print(f"   使用多GPU: {crps_computer.use_multi_gpu}")
    
except Exception as e:
    print(f"❌ CRPS測試失敗: {e}")

print("\n🎉 雙GPU系統測試完成！")
print("現在所有必要的變量都已正確定義，可以進行完整的分析。")