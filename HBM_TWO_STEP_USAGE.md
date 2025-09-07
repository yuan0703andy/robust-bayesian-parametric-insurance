# HBM兩步法使用說明

## 概述

我已經基於您現有的 `basis_risk_vi.py` 創建了一個接口，可以接入您在 `05_complete_integrated_framework_v4_correct.py` 階段3中設定的 HBM (Hierarchical Bayesian Model)。

## 核心修改

### 1. BasisRiskAwareVI 類增強

```python
# 新增參數支援HBM
vi_engine = BasisRiskAwareVI(
    n_features=1,
    objective='hbm_two_step',  # 🔑 新模式
    hierarchical_model=your_hbm_instance,  # 🔑 您的HBM實例
    use_gpu=True,
    n_params=350  # 或其他維度
)
```

### 2. 三種優化模式

- `'traditional_elbo'`: 傳統ELBO (第二層比較)  
- `'crps_basis_risk'`: CRPS-based ELBO創新 (第三層比較)
- `'hbm_two_step'`: **HBM兩步法** - 新增模式

## 使用方法

### Step 1: 準備您的HBM模型

確保您的 HBM 模型有以下接口：

```python
class YourHBMModel:
    def update_configuration(self, prior_scenario, likelihood_family):
        """更新先驗和似然配置"""
        pass
    
    def predict_distribution(self, theta, X, n_samples=100):
        """
        預測損失分布G(θ)
        
        Returns:
            損失樣本 [n_events, n_samples]
        """
        pass
```

### Step 2: 創建階段3配置

```python
from robust_hierarchical_bayesian_simulation.hierarchical_modeling.prior_specifications import (
    PriorScenario, LikelihoodFamily
)

# 與您的階段3完全相同的配置
prior_likelihood_configs = [
    {
        'name': '基線-非信息先驗+正態似然',
        'prior': PriorScenario.NON_INFORMATIVE, 
        'likelihood': LikelihoodFamily.NORMAL, 
        'epsilon': 0.0
    },
    {
        'name': '中污染-悲觀先驗+學生t',
        'prior': PriorScenario.PESSIMISTIC, 
        'likelihood': LikelihoodFamily.STUDENT_T, 
        'epsilon': 0.05
    },
    # ... 更多配置
]
```

### Step 3: 執行兩步法優化

```python
# 載入真實數據
X, y = load_climada_data()  # 您的數據載入函數
X_train, X_val, y_train, y_val = train_test_split(...)

# 初始化VI引擎
vi_engine = BasisRiskAwareVI(
    n_features=X.shape[1],
    objective='hbm_two_step',
    hierarchical_model=your_hbm_model,  # 您的HBM實例
    use_gpu=True
)

# 執行兩步法優化  
results = vi_engine.run_hbm_two_step_optimization(
    X_train, y_train, 
    prior_likelihood_configs,
    X_val, y_val
)
```

## 結果結構

```python
results = {
    'step1_results': [
        {
            'config_name': '基線-非信息先驗+正態似然',
            'final_basis_risk': 15.2e6,  # CRPS(y, G(θ))
            'prior': PriorScenario.NON_INFORMATIVE,
            'likelihood': LikelihoodFamily.NORMAL,
            'epsilon': 0.0,
            'best_theta': array([...]),  # 最佳θ參數
            'elbo': -1234.5,
            # ... 其他VI結果
        },
        # ... 其他配置結果
    ],
    'step2_results': {
        'best_product': {
            'product_id': 127,
            'crps': 12.8e6,  # CRPS(y, F_k(θ*))
            'thresholds': [35, 50, 70, 999],
            'ratios': [0.25, 0.5, 1.0, 0.0],
            'max_payout': 25e6
        },
        'best_products': [...],  # Top 10產品
        'all_products': [...]    # 全部350個產品
    },
    'best_config': {...},  # Step1最佳配置
    'method': 'hbm_two_step'
}
```

## 兩步法核心邏輯

### Step 1: 優化 G(θ) 的 CRPS
```
對每個Prior/Likelihood組合:
  1. 更新HBM配置: model.update_configuration(prior, likelihood)
  2. 優化ELBO: ℒHBM(φ) = -E[CRPS(y, G(θ))] - KL(q||p)  
  3. 獲得最佳θ*

選擇CRPS最小的配置作為最佳θ*
```

### Step 2: 評估 350 個 Steinmann 產品
```
使用最佳θ*:
  1. 生成損失預測: G(θ*) → 損失分布樣本
  2. 對每個產品k計算: F_k(θ*) → 賠付分布樣本  
  3. 計算CRPS(y, F_k(θ*))
  4. 選擇CRPS最小的產品
```

## 執行示例

```bash
# 運行示例 (使用模擬數據)
python hbm_two_step_example.py

# 您的實際使用 (接入真實HBM)
# 修改 create_mock_hbm() 為您的真實HBM載入
# 修改 load_real_data() 為您的climada_complete_data.pkl載入
```

## 優勢

1. **完全兼容**: 無縫接入您現有的階段3 Prior/Likelihood 配置
2. **GPU加速**: 支持雙GPU並行350產品評估  
3. **真實VI**: 完整的變分推斷，不是簡化的網格搜索
4. **模組化**: 可獨立使用Step1或Step2
5. **驗證集支持**: 避免過度擬合

## 與原框架的區別

| 原框架 (crps_basis_risk) | 新框架 (hbm_two_step) |
|--------------------------|----------------------|
| 直接優化 CRPS(y, F(θ)) | Step1: 優化 CRPS(y, G(θ)) |
| 350維產品權重同時優化 | Step2: 事後評估350產品 |
| 端到端聯合優化 | 兩步分離優化 |
| 單一目標函數 | 兩階段目標函數 |

這個接口讓您可以靈活地使用現有的HBM模型，同時保持與階段3配置的完全兼容性。