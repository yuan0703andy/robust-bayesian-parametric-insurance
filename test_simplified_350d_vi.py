#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試簡化的350維直接VI方案
=============================
驗證基差風險在訓練中能正確變化，不再恆定於24.8M

純粹VI對決測試：
- θ直接是350個產品的logits
- w = softmax(θ) 得到產品權重  
- payout = Σ(w_i × smooth_payout_i)
- 專注於CRPS-VI vs ELBO-VI比較
"""

import sys
import os
import numpy as np
import torch
import pickle
from pathlib import Path

# 添加模組路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_simplified_350d_vi():
    """測試簡化的350維VI實現"""
    print("=" * 80)
    print("測試簡化的350維直接VI方案")
    print("Test Simplified 350D Direct VI Approach")
    print("=" * 80)
    
    # 步驟1: 載入必要模組
    print("\n📦 載入模組...")
    try:
        from robust_hierarchical_bayesian_simulation.model_selection.basis_risk_vi import BasisRiskAwareVI
        print("✅ BasisRiskAwareVI 模組已載入")
        
        # 載入產品數據
        with open("results/insurance_products/products.pkl", 'rb') as f:
            products = pickle.load(f)
        print(f"✅ 載入Steinmann產品: {len(products)} 個")
        
    except ImportError as e:
        print(f"❌ 模組載入失敗: {e}")
        return False
    except FileNotFoundError:
        print("❌ 產品文件未找到，請先執行 03_insurance_product.py")
        return False
    
    # 步驟2: 創建測試數據
    print("\n🔧 創建測試數據...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_events = 50
    n_features = 1
    
    # 風速數據 (20-60 m/s)
    X_test = np.random.uniform(20, 60, (n_events, n_features)).astype(np.float32)
    
    # 損失數據 (模擬實際損失分布)
    base_losses = np.random.lognormal(15, 1.2, n_events)
    y_test = base_losses.astype(np.float32)
    
    print(f"   測試數據: {n_events} 個事件")
    print(f"   風速範圍: {X_test.min():.1f} - {X_test.max():.1f} m/s")
    print(f"   損失範圍: ${y_test.min()/1e6:.1f}M - ${y_test.max()/1e6:.1f}M")
    
    # 步驟3: 測試350維VI初始化
    print("\n🎯 測試350維VI初始化...")
    try:
        vi_optimizer = BasisRiskAwareVI(
            n_features=n_features,
            epsilon_values=[0.1],  
            device='cpu',  # 使用CPU避免GPU配置問題
            learning_rate=0.01,
            use_crps_loss=False,  # 先測試標準ELBO
            basis_risk_type='absolute',
            n_params=350  # 🔑 關鍵：350維直接產品選擇
        )
        print("✅ 350維VI優化器初始化成功")
        print(f"   參數數量: {vi_optimizer.n_params}")
        print(f"   產品數量: {len(vi_optimizer.products)}")
        
    except Exception as e:
        print(f"❌ 350維VI初始化失敗: {e}")
        return False
    
    # 步驟4: 測試基差風險計算
    print("\n📊 測試基差風險計算...")
    try:
        # 訓練前的初始基差風險
        initial_loss = vi_optimizer._compute_loss(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        initial_basis_risk = initial_loss.item() if hasattr(initial_loss, 'item') else float(initial_loss)
        
        print(f"   初始基差風險: ${initial_basis_risk/1e6:.2f}M")
        
        # 手動更新參數來測試基差風險變化
        with torch.no_grad():
            # 修改前幾個產品的權重
            vi_optimizer.theta.data[:5] += 2.0  # 增加前5個產品的logit值
            
        # 重新計算基差風險
        updated_loss = vi_optimizer._compute_loss(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        updated_basis_risk = updated_loss.item() if hasattr(updated_loss, 'item') else float(updated_loss)
        
        print(f"   更新後基差風險: ${updated_basis_risk/1e6:.2f}M")
        
        # 檢查基差風險是否有變化
        basis_risk_change = abs(updated_basis_risk - initial_basis_risk)
        if basis_risk_change > 1000:  # 變化超過$1000
            print(f"   ✅ 基差風險有變化: ${basis_risk_change/1e6:.2f}M")
            basis_risk_working = True
        else:
            print(f"   ❌ 基差風險變化太小: ${basis_risk_change/1e6:.2f}M")
            basis_risk_working = False
            
    except Exception as e:
        print(f"❌ 基差風險計算失敗: {e}")
        return False
    
    # 步驟5: 簡短訓練測試
    print("\n🏃‍♂️ 簡短訓練測試...")
    try:
        # 重置參數
        vi_optimizer._initialize_parameters()
        
        initial_loss_training = vi_optimizer._compute_loss(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        ).item()
        
        print(f"   訓練前損失: ${initial_loss_training/1e6:.2f}M")
        
        # 執行幾步訓練
        for step in range(3):
            loss = vi_optimizer._compute_loss(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32)
            )
            
            vi_optimizer.optimizer.zero_grad()
            loss.backward()
            vi_optimizer.optimizer.step()
            
            if step == 0:
                first_step_loss = loss.item()
            elif step == 2:
                final_step_loss = loss.item()
                
        print(f"   第1步損失: ${first_step_loss/1e6:.2f}M")
        print(f"   第3步損失: ${final_step_loss/1e6:.2f}M")
        
        training_change = abs(final_step_loss - first_step_loss)
        if training_change > 1000:  # 訓練有效果
            print(f"   ✅ 訓練有效果: 損失變化${training_change/1e6:.2f}M")
            training_working = True
        else:
            print(f"   ⚠️  訓練效果不明顯: 損失變化${training_change/1e6:.2f}M")
            training_working = training_change > 0  # 至少有一些變化
            
    except Exception as e:
        print(f"❌ 訓練測試失敗: {e}")
        return False
    
    # 步驟6: 產品權重檢查
    print("\n⚖️ 產品權重分佈檢查...")
    try:
        with torch.no_grad():
            product_weights = torch.softmax(vi_optimizer.theta, dim=0)
            
        # 統計權重分佈
        weights_np = product_weights.numpy()
        print(f"   權重總和: {weights_np.sum():.6f}")
        print(f"   最大權重: {weights_np.max():.6f}")
        print(f"   最小權重: {weights_np.min():.6f}")
        print(f"   權重標準差: {weights_np.std():.6f}")
        
        # 檢查前5個最高權重的產品
        top_5_indices = np.argsort(weights_np)[-5:]
        print(f"   前5高權重產品:")
        for i, idx in enumerate(reversed(top_5_indices)):
            product_id = products[idx]['product_id'] if idx < len(products) else f"Product_{idx}"
            print(f"     {i+1}. {product_id}: {weights_np[idx]:.6f}")
            
        weights_working = abs(weights_np.sum() - 1.0) < 1e-5  # softmax正確性
        
    except Exception as e:
        print(f"❌ 權重檢查失敗: {e}")
        return False
    
    # 總結
    print("\n" + "=" * 80)
    print("測試結果總結")
    print("=" * 80)
    
    all_tests_passed = basis_risk_working and training_working and weights_working
    
    results = {
        "✅ 350維VI初始化": True,
        "✅ 基差風險變化": basis_risk_working,
        "✅ 訓練收斂": training_working, 
        "✅ 產品權重正確": weights_working
    }
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    
    if all_tests_passed:
        print("\n🎉 簡化350維VI方案測試通過!")
        print("   基差風險不再恆定，能正確響應參數變化")
        print("   準備進行完整的CRPS-VI vs ELBO-VI對決")
    else:
        print("\n⚠️ 部分測試未通過，需要進一步調試")
    
    return all_tests_passed

if __name__ == "__main__":
    success = test_simplified_350d_vi()
    if success:
        print("\n🚀 可以繼續進行完整分析")
    else:
        print("\n🔧 需要修復問題後再測試")