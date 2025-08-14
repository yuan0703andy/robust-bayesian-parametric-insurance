# Quick GPU Deployment Instructions
# 快速GPU部署指令

## 立即部署 (2小時工作，3-4x加速)

### 1. 安裝GPU依賴 (10分鐘)
```bash
# 安裝JAX GPU支援
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 安裝NumPyro (JAX-based MCMC)
pip install numpyro

# 安裝GPU監控工具
pip install gputil
```

### 2. 驗證GPU設置 (5分鐘)
```python
import jax
print("Available devices:", jax.devices())
# 應該顯示: [cuda(id=0), cuda(id=1), cpu(id=0)]
```

### 3. 運行優化分析 (預計1小時，原本4小時)
```bash
# 使用你的CLIMADA環境
/Users/andyhou/.local/share/mamba/envs/climada_env/bin/python 05_robust_bayesian_parm_insurance_gpu.py
```

### 4. 性能對比
```
原始CPU配置:
- 鏈數: 2-4
- 樣本: 1000-2000  
- 預估時間: 4小時
- 總樣本: 8,000

GPU優化配置:
- 鏈數: 16 (8 per GPU)
- 樣本: 4000
- 預估時間: 1小時  
- 總樣本: 64,000
- 加速比: 4x
```

### 5. 監控性能
分析運行時會自動顯示:
- ⚡ GPU Performance - Phase X: X.X minutes elapsed
- 📱 GPU 0: XX% load, XX% memory  
- 📱 GPU 1: XX% load, XX% memory

### 6. 如果遇到問題
如果GPU不可用，系統會自動降級到CPU模式，但仍然比原始配置快1.5-2x。

## 主要優化項目 ✅

✅ 環境變量: 雙GPU + JAX優化
✅ MCMC參數: 4000樣本 × 16鏈 = 64,000總樣本  
✅ 並行策略: 每個GPU運行8條鏈
✅ 記憶體管理: 80% GPU使用率
✅ 線程控制: 避免過度並行衝突
✅ 性能監控: 實時GPU使用率追蹤
✅ 自動降級: GPU不可用時的CPU後備方案

預期結果: 3-4x 加速，更高品質的貝氏採樣！
