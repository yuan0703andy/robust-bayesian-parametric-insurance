# Sigmoid代理優化使用指南
Sigmoid Proxy Optimization Usage Guide

## 概述

我們成功實現了**兩階段代理優化(Surrogate Optimization)**功能，解決了階梯函數不可微的根本問題，實現了端到端的CRPS-VI優化。

## 核心原理

### 問題背景
- 原始Steinmann階梯函數：`Payout = 25%, 50%, 75%, 100%` 在閾值處跳躍
- 階梯函數**不可微分** → 無法提供梯度 → VI優化器無法學習
- 解決方案：**Sigmoid代理函數** 平滑近似階梯行為

### 兩階段策略

#### 🎯 階段一：訓練（Sigmoid代理）
```python
# 使用平滑的Sigmoid函數作為階梯的"替身"
Payout(x) = max_payout * Σ[(rᵢ - rᵢ₋₁) * sigmoid(k*(x - tᵢ))]

# 提供連續梯度信號給VI優化器
Loss = CRPS(y, F_smooth(G(θ))) → ∇θ ≠ 0 ✅
```

#### 📊 階段二：評估（真實階梯）
```python  
# 切換回原始階梯函數進行真實評估
for threshold in thresholds:
    if wind_speed >= threshold:
        payout = max_payout * ratio

# 確保結果的實用性和準確性
CRPS(y, F_step(G(θ*))) → 真實產品表現
```

## 使用方法

### 1. 基本初始化

```python
from robust_hierarchical_bayesian_simulation.model_selection.basis_risk_vi import BasisRiskAwareVI

# 創建VI引擎 - 啟用Sigmoid代理
vi_engine = BasisRiskAwareVI(
    n_features=1,
    use_sigmoid_proxy=True,        # 🔑 啟用代理優化
    sigmoid_steepness=0.1,         # Sigmoid陡峭度k
    training_mode=True,            # 初始：訓練模式  
    objective='crps_basis_risk',   # CRPS優化目標
    n_params=350                   # 350個Steinmann產品
)
```

### 2. 訓練階段（使用Sigmoid）

```python
# 🎯 訓練階段：平滑可微分優化
vi_engine.set_optimization_mode(training_mode=True)

# 執行VI訓練 - Sigmoid提供梯度信號
results = vi_engine.run_comprehensive_screening(
    X_train, y_train, 
    X_val, y_val
)

# 獲得訓練好的θ*參數
best_theta = results['best_theta']
```

### 3. 評估階段（切換到階梯）

```python
# 📊 評估階段：真實階梯函數評估
vi_engine.set_optimization_mode(training_mode=False)

# 用θ*評估350個真實Steinmann產品
final_results = []
for product_id in range(350):
    proxy = vi_engine.get_steinmann_payout_proxy(product_id)
    
    # 使用真實階梯函數計算賠付
    real_payouts = proxy.calculate_payout_step(loss_samples)
    crps = calculate_crps(y_test, real_payouts)
    
    final_results.append({
        'product_id': product_id,
        'crps': crps
    })

# 選出真正最佳的產品
best_product = min(final_results, key=lambda x: x['crps'])
```

### 4. HBM兩步法集成

```python
# HBM兩步法專用配置
vi_engine = BasisRiskAwareVI(
    n_features=1,
    objective='hbm_two_step',      # 🔑 HBM兩步法模式
    hierarchical_model=your_hbm,   # 您的HBM模型實例
    use_sigmoid_proxy=True,        # 啟用代理優化
    sigmoid_steepness=0.1,
    training_mode=True
)

# Step 1: 優化G(θ)使用Sigmoid代理
step1_results = vi_engine.run_hbm_two_step_optimization(
    X_train, y_train, prior_likelihood_configs,
    X_val, y_val
)

# Step 2: 評估350產品使用真實階梯（自動切換）
best_config = step1_results['best_config'] 
step2_results = step1_results['step2_results']
```

## 關鍵特性

### ✅ 確保參數傳遞正確性
```python
# θ參數變化 → 基差風險變化的正確傳遞鏈
θ → HBM.predict_distribution(θ) → loss_samples → 
    SigmoidProxy.calculate_payout_tensor(loss_samples) → 
    CRPS(y, payouts) → basis_risk
```

### ✅ GPU加速支持
```python
# 雙GPU並行計算350個產品
# GPU0: 產品0-174, GPU1: 產品175-349
payouts_tensor = proxy.calculate_payout_tensor(loss_tensor)  # 可微分
```

### ✅ 無縫模式切換
```python
# 運行時動態切換
vi_engine.set_optimization_mode(training_mode=True)   # Sigmoid模式
vi_engine.set_optimization_mode(training_mode=False)  # 階梯模式
```

## 測試驗證

```bash
# 運行完整測試套件
python test_sigmoid_proxy_optimization.py

# 預期輸出：
# ✅ SigmoidPayoutProxy: 兩階段代理函數工作正常
# ✅ BasisRiskAwareVI: 集成Sigmoid代理功能成功  
# ✅ 兩階段優化: 訓練+評估工作流程完整
# ✅ 參數傳遞: θ變化能夠影響基差風險計算
```

## 配置參數

### Sigmoid陡峭度 `k`
- **k=0.01**: 非常平滑，遠離階梯
- **k=0.1**: 平衡選擇（推薦）  
- **k=1.0**: 較陡峭，接近階梯
- **k=10**: 極陡峭，幾乎等同階梯

### 應用建議
- **風速輸入**: k=0.1 (風速單位m/s)
- **損失輸入**: k=1e-6 (損失單位USD)
- **調節原則**: k越大越接近真實階梯，但梯度越小

## 理論保證

1. **收斂性**: Sigmoid代理在k→∞時收斂到真實階梯函數
2. **可微性**: 訓練階段保持全程可微分，梯度信號連續
3. **一致性**: 評估階段使用完全相同的Steinmann產品定義
4. **最優性**: 兩階段優化確保在可微分約束下達到最優解

這個實現為您的CRPS-VI框架提供了完整的端到端優化能力，同時保持了與真實世界階梯式保險產品的完美兼容性。