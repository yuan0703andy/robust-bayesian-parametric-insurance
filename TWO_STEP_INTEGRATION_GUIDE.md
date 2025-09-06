# 兩步驟架構集成指南
# Two-Step Architecture Integration Guide

## 🎯 目標

將你的 `05_complete_integrated_framework_v4_correct.py` 從複雜的多層比較架構簡化為清晰的兩步驟架構：

- **Step 1**: 階層貝葉斯損失預測器訓練 (CRPS-VI)
- **Step 2**: 產品評估與排名

## 🔧 集成方案

### 方案 A: 直接替換 (推薦)

將 05 腳本的階段4-8 替換為新的兩步驟架構：

```python
# 在 05 腳本的階段3完成後，添加以下代碼：

# %%
# =============================================================================
# 導入兩步驟架構
# =============================================================================

# 方法1: 導入集成補丁
from 05_integrated_two_step_patch import execute_two_step_architecture

# 執行兩步驟架構
two_step_results = execute_two_step_architecture(
    spatial_data=spatial_data,
    train_data=train_data, 
    val_data=val_data,
    test_data=test_data,
    final_epsilon=final_epsilon,
    USE_GPU=USE_GPU
)

# 添加結果到 integrated_results
integrated_results['two_step_architecture'] = two_step_results
```

### 方案 B: 漸進式集成

如果你想保留原有結構，可以將兩步驟架構作為新的階段添加：

```python
# %%
# =============================================================================
# 階段4B: 兩步驟架構 (新方法)
# =============================================================================

print("\n階段4B: 兩步驟架構分析")

# 執行兩步驟架構
from 05_integrated_two_step_patch import execute_two_step_architecture

two_step_results = execute_two_step_architecture(
    spatial_data=spatial_data,
    train_data=train_data,
    val_data=val_data, 
    test_data=test_data,
    final_epsilon=final_epsilon,
    USE_GPU=USE_GPU
)

# 保留原有的階段4-8作為比較基準
```

## 📋 具體步驟

### Step 1: 找到替換點

在你的 05 腳本中找到這一行：
```python
# 階段4: 基差風險導向變分推斷
```

### Step 2: 替換階段4-8

將從階段4到階段8的所有複雜代碼替換為：

```python
# %%
# =============================================================================
# 階段4-8: 兩步驟架構 (簡化版)
# =============================================================================

print("\n階段4-8: 兩步驟架構分析")
print("   Step 1: 階層貝葉斯損失預測器 (CRPS-VI)")
print("   Step 2: 350產品評估與排名")

# 導入集成補丁
import sys
import os
sys.path.append(os.path.dirname(__file__))
from 05_integrated_two_step_patch import execute_two_step_architecture

# 執行兩步驟架構  
two_step_results = execute_two_step_architecture(
    spatial_data=spatial_data,
    train_data=train_data,
    val_data=val_data,
    test_data=test_data, 
    final_epsilon=final_epsilon,
    USE_GPU=USE_GPU
)

# 更新 integrated_results 結構
integrated_results.update({
    'two_step_architecture': two_step_results,
    'methodology': 'Two-Step: Hierarchical Bayesian Loss Predictor + Product Evaluator',
    'primary_results': two_step_results['summary']
})

print(f"\n✅ 兩步驟架構完成!")
print(f"   最佳基差風險: {two_step_results['summary']['best_basis_risk']/1e6:.2f}M")
print(f"   總計算時間: {two_step_results['summary']['total_time']:.1f}秒")
```

### Step 3: 更新結果保存

在腳本最後的結果保存部分，確保包含兩步驟結果：

```python
# 在 integrated_results 創建部分添加：
'two_step_architecture_summary': {
    'step_1_crps': two_step_results['step_1_results']['trained_params']['final_crps'],
    'step_2_best_product': two_step_results['step_2_results']['best_product'],
    'total_time': two_step_results['summary']['total_time'],
    'champion_basis_risk': two_step_results['summary']['best_basis_risk']
}
```

## 🎯 預期結果

執行後你將看到：

```
階段4-8: 兩步驟架構分析
   Step 1: 階層貝葉斯損失預測器 (CRPS-VI)
   Step 2: 350產品評估與排名

📈 Step 1: 階層貝葉斯損失預測器訓練
   目標: 使用 CRPS-VI 訓練最準確的損失預測器

✅ 適應性損失預測器初始化
   計算設備: GPU
   醫院數: 100, 區域數: 3
   階層參數數: 107

🔥 開始CRPS-VI訓練 (1500次迭代)...
   迭代 200: CRPS=0.8542, lr=0.009500
   迭代 400: CRPS=0.7834, lr=0.009025
   ...
✅ 損失預測器訓練完成: CRPS=0.6234

🏆 Step 2: 產品評估與排名
   使用訓練好的損失預測器評估350個產品

✅ 生成 150 個評估產品
📊 評估 150 個產品在 25 個事件上...
   進度: 20/150
   ...

🏆 產品排名 (前15名):
================================================================================
排名 ID     基差風險(M)   半徑   類型     閾值           
================================================================================
1    42     12.34        30     dual     [33, 50]       
2    67     13.78        50     single   [43]           
...

✅ Step 2完成 (45.2秒)

📊 兩步驟架構完整結果:
============================================================
🧠 Step 1 (損失預測器):
   訓練時間: 67.8秒
   最終CRPS: 0.6234
   參數數量: 107

🏆 Step 2 (產品評估):
   評估時間: 45.2秒
   評估產品數: 150
   冠軍基差風險: 12.34M

⏱️ 總時間: 113.0秒
✅ 兩步驟架構分析完成!
```

## ✅ 優勢

1. **邏輯清晰**: 兩個步驟各司其職，不混合概念
2. **計算高效**: 避免複雜的多層比較，專注核心邏輯  
3. **結果可解釋**: 每步的目標和結果都很明確
4. **易於維護**: 代碼結構簡單，容易調試和改進
5. **學術合規**: 符合你最初設想的兩步驟架構

## 🚀 立即執行

```bash
# 1. 備份原腳本
cp 05_complete_integrated_framework_v4_correct.py 05_complete_integrated_framework_v4_correct_backup.py

# 2. 按照上述指南修改腳本

# 3. 執行修改後的腳本
python 05_complete_integrated_framework_v4_correct.py
```

這樣你就能得到清晰、高效的兩步驟架構，完全實現了你最初的設想！