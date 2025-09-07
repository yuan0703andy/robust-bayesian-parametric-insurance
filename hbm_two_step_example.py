#!/usr/bin/env python3
"""
HBM兩步法示例 - 接入階段3的Prior/Likelihood配置

展示如何使用增強的BasisRiskAwareVI執行兩步法優化：
Step 1: 測試不同Prior/Likelihood組合，優化G(θ)的CRPS
Step 2: 用最佳θ*評估350個Steinmann產品的F_k(θ*)

Author: Research Team
Date: 2025-01-17
"""

import numpy as np
import sys
import os

# 添加模組路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robust_hierarchical_bayesian_simulation.model_selection.basis_risk_vi import BasisRiskAwareVI
from robust_hierarchical_bayesian_simulation.hierarchical_modeling.prior_specifications import (
    PriorScenario, LikelihoodFamily
)

def create_stage3_configs(test_mode='selected'):
    """
    創建階段3的Prior/Likelihood測試配置
    
    Args:
        test_mode: 'selected' 或 'systematic'
        
    Returns:
        prior_likelihood_configs列表
    """
    if test_mode == 'systematic':
        # 系統性矩陣測試 - 所有組合
        print("🔬 系統性矩陣測試模式")
        
        priors_to_test = [PriorScenario.NON_INFORMATIVE, PriorScenario.WEAK_INFORMATIVE, 
                         PriorScenario.PESSIMISTIC, PriorScenario.CONSERVATIVE]
        likelihoods_to_test = [LikelihoodFamily.NORMAL, LikelihoodFamily.LOGNORMAL, 
                              LikelihoodFamily.STUDENT_T, LikelihoodFamily.GAMMA]
        epsilons_to_test = [0.0, 0.05, 0.15]  # 3種污染水平
        
        configs = []
        for prior in priors_to_test:
            for likelihood in likelihoods_to_test:
                for epsilon in epsilons_to_test:
                    config_name = f"{prior.value}+{likelihood.value}+ε{epsilon:.2f}"
                    configs.append({
                        'name': config_name,
                        'prior': prior,
                        'likelihood': likelihood,
                        'epsilon': epsilon
                    })
        
        print(f"生成{len(configs)}種系統性組合")
        return configs
        
    else:
        # 精選組合模式 - 6種代表性配置
        print("🎯 精選組合模式")
        
        configs = [
            # 基線組合 (無污染)
            {
                'name': '基線-非信息先驗+正態似然', 
                'prior': PriorScenario.NON_INFORMATIVE, 
                'likelihood': LikelihoodFamily.NORMAL, 
                'epsilon': 0.0
            },
            {
                'name': '基線-弱信息先驗+對數正態', 
                'prior': PriorScenario.WEAK_INFORMATIVE, 
                'likelihood': LikelihoodFamily.LOGNORMAL, 
                'epsilon': 0.0
            },
            
            # 中等污染組合
            {
                'name': '中污染-悲觀先驗+學生t', 
                'prior': PriorScenario.PESSIMISTIC, 
                'likelihood': LikelihoodFamily.STUDENT_T, 
                'epsilon': 0.05
            },
            {
                'name': '中污染-保守先驗+伽瑪分佈', 
                'prior': PriorScenario.CONSERVATIVE, 
                'likelihood': LikelihoodFamily.GAMMA, 
                'epsilon': 0.05
            },
            
            # 高污染極端組合  
            {
                'name': '高污染-悲觀先驗+學生t', 
                'prior': PriorScenario.PESSIMISTIC, 
                'likelihood': LikelihoodFamily.STUDENT_T, 
                'epsilon': 0.15
            },
            {
                'name': '高污染-保守先驗+對數正態', 
                'prior': PriorScenario.CONSERVATIVE, 
                'likelihood': LikelihoodFamily.LOGNORMAL, 
                'epsilon': 0.15
            }
        ]
        
        print(f"生成{len(configs)}種精選組合")
        return configs

def load_real_data():
    """
    載入真實的CLIMADA數據 (示例)
    實際使用時應該載入您的climada_complete_data.pkl
    """
    print("📊 載入真實數據...")
    
    # 這裡使用模擬數據作為示例
    # 實際應該從您的數據文件載入
    np.random.seed(42)
    n_events = 100
    n_features = 1
    
    # 模擬風速數據 (m/s)
    X = np.random.uniform(30, 90, (n_events, n_features))
    
    # 模擬損失數據 (美元) - 基於風速的非線性關係
    base_losses = (X.flatten() - 30) ** 2.5 * 1e6
    noise = np.random.lognormal(0, 0.5, n_events) 
    y = base_losses * noise
    
    print(f"   數據規模: {n_events}個事件, {n_features}個特徵")
    print(f"   風速範圍: {X.min():.1f} - {X.max():.1f} m/s")
    print(f"   損失範圍: ${y.min()/1e6:.1f}M - ${y.max()/1e6:.1f}M")
    
    return X, y

def create_mock_hbm():
    """
    創建模擬的HBM模型實例
    實際使用時應該載入您的hierarchical_modeling模組
    """
    class MockHBM:
        def __init__(self):
            self.current_prior = None
            self.current_likelihood = None
            
        def update_configuration(self, prior_scenario, likelihood_family):
            """更新先驗和似然配置"""
            self.current_prior = prior_scenario
            self.current_likelihood = likelihood_family
            print(f"    🔧 HBM配置更新: {prior_scenario.value} + {likelihood_family.value}")
            
        def predict_distribution(self, theta, X, n_samples=100):
            """
            預測損失分布G(θ)
            
            Args:
                theta: 模型參數 [n_params]
                X: 輸入特徵 [n_events, n_features] 
                n_samples: 每個事件的樣本數
                
            Returns:
                損失樣本 [n_events, n_samples]
            """
            n_events = X.shape[0]
            
            # 基於當前配置調整預測行為
            if self.current_likelihood == LikelihoodFamily.LOGNORMAL:
                # 對數正態分佈
                base_pred = X.flatten() ** 2 * 1e4  
                samples = []
                for i in range(n_events):
                    log_mean = np.log(base_pred[i])
                    log_std = abs(theta[0]) if len(theta) > 0 else 0.5
                    event_samples = np.random.lognormal(log_mean, log_std, n_samples)
                    samples.append(event_samples)
                return np.array(samples)
                
            elif self.current_likelihood == LikelihoodFamily.STUDENT_T:
                # 學生t分佈 (重尾)
                base_pred = X.flatten() ** 1.8 * 2e4
                samples = []
                for i in range(n_events):
                    scale = abs(theta[1]) if len(theta) > 1 else base_pred[i] * 0.3
                    # 使用正態近似學生t
                    event_samples = np.random.normal(base_pred[i], scale, n_samples)
                    event_samples = np.maximum(event_samples, 0)  # 確保非負
                    samples.append(event_samples)
                return np.array(samples)
                
            else:
                # 默認正態分佈
                base_pred = X.flatten() ** 2.2 * 1.5e4
                samples = []
                for i in range(n_events):
                    mean = base_pred[i]
                    std = abs(theta[-1]) if len(theta) > 0 else mean * 0.4
                    event_samples = np.random.normal(mean, std, n_samples)
                    event_samples = np.maximum(event_samples, 0)  # 確保非負
                    samples.append(event_samples)
                return np.array(samples)
    
    return MockHBM()

def run_hbm_two_step_example():
    """
    執行完整的HBM兩步法示例
    """
    print("🚀 HBM兩步法優化示例")
    print("=" * 60)
    
    # 1. 載入數據
    X, y = load_real_data()
    
    # 創建訓練/驗證分割
    n_total = len(y)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)
    
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_val, y_val = X[indices[n_train:]], y[indices[n_train:]]
    
    print(f"   訓練集: {len(y_train)}個事件")
    print(f"   驗證集: {len(y_val)}個事件")
    
    # 2. 創建模擬HBM
    hbm_model = create_mock_hbm()
    
    # 3. 初始化BasisRiskAwareVI (兩步法模式)
    vi_engine = BasisRiskAwareVI(
        n_features=X.shape[1],
        objective='hbm_two_step',  # 關鍵：兩步法模式
        hierarchical_model=hbm_model,  # 提供HBM實例
        use_gpu=False,  # 示例用CPU模式
        n_params=5  # HBM參數維度
    )
    
    # 4. 創建階段3配置
    test_configs = create_stage3_configs(test_mode='selected')  # 'selected' 或 'systematic'
    
    # 5. 執行兩步法優化
    print(f"\n🎯 開始兩步法優化...")
    results = vi_engine.run_hbm_two_step_optimization(
        X_train, y_train, test_configs, 
        X_val, y_val
    )
    
    # 6. 分析結果
    print(f"\n📊 結果分析")
    print("=" * 40)
    
    # Step 1結果
    print(f"\n=== Step 1: G(θ)最優化結果 ===")
    best_config = results['best_config']
    print(f"🏆 最佳配置: {best_config['config_name']}")
    print(f"   CRPS(y, G(θ*)): {best_config['final_basis_risk']/1e6:.1f}M")
    print(f"   先驗: {best_config['prior'].value}")
    print(f"   似然: {best_config['likelihood'].value}")
    print(f"   ε污染: {best_config['epsilon']:.3f}")
    
    print(f"\n📋 所有Step 1配置排名:")
    for i, config in enumerate(results['step1_results'][:3]):
        print(f"   {i+1}. {config['config_name']}: {config['final_basis_risk']/1e6:.1f}M")
    
    # Step 2結果
    print(f"\n=== Step 2: 350個產品評估結果 ===")
    step2 = results['step2_results']
    best_product = step2['best_product']
    print(f"🏆 最佳產品: ID={best_product['product_id']}")
    print(f"   CRPS(y, F_k(θ*)): {best_product['crps']/1e6:.1f}M")
    print(f"   閾值: {best_product['thresholds']}")
    print(f"   賠付比例: {best_product['ratios']}")
    print(f"   最大賠付: ${best_product['max_payout']/1e6:.1f}M")
    
    print(f"\n📋 最佳產品Top 5:")
    for i, product in enumerate(step2['best_products'][:5]):
        print(f"   {i+1}. ID={product['product_id']}: CRPS={product['crps']/1e6:.1f}M")
    
    # 7. 保存結果
    print(f"\n💾 結果已保存到內存 (可擴展保存到文件)")
    
    return results

if __name__ == "__main__":
    # 執行示例
    try:
        results = run_hbm_two_step_example()
        print(f"\n✅ HBM兩步法優化完成!")
        
    except Exception as e:
        print(f"\n❌ 執行出錯: {str(e)}")
        import traceback
        traceback.print_exc()