#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
what_are_we_fitting.py
======================
解釋「擬合觀測數據」到底在擬合什麼
Explain what exactly we are fitting when we say "fitting observed data"

回答您的問題：「擬合觀測數據 是在擬合什麼」
"""

import numpy as np
import matplotlib.pyplot as plt

def explain_what_we_are_fitting():
    """
    解釋我們在擬合什麼
    """
    
    print("🎯 What Are We Actually Fitting?")
    print("=" * 80)
    
    print("\n📊 THE PHYSICAL PROCESS WE'RE MODELING")
    print("-" * 60)
    physical_process = """
    我們在擬合的是「脆弱度函數的空間變異性」：
    
    🌪️ 物理過程：
    當熱帶氣旋襲擊北卡羅來納州時：
    • 不同地點經歷不同的風速 (Hazard)
    • 不同地點有不同的暴險值 (Exposure) 
    • 不同地點有不同的脆弱度 (Vulnerability)
    
    🎯 我們想要理解的核心問題：
    "為什麼相同風速在不同地點造成不同程度的損失？"
    
    答案：脆弱度的空間變異性 β_i = α_r(i) + δ_i + γ_i
    """
    print(physical_process)

def show_concrete_fitting_target():
    """
    展示具體的擬合目標
    """
    
    print("\n🔍 CONCRETE FITTING TARGET")
    print("-" * 60)
    
    concrete_target = """
    具體來說，我們在擬合：
    
    📍 輸入數據 (Input Data):
    • Observed losses: [1.2e6, 5.3e6, 0.8e6, 12.1e6, ...]  (CLIMADA計算的損失)
    • Hazard intensities: [45, 38, 52, 67, ...] m/s         (各地點最大風速)
    • Exposure values: [2.1e8, 1.5e8, 3.2e8, ...]           (各地點資產價值)
    • Spatial locations: [(35.1, -80.2), (35.8, -79.1), ...] (經緯度)
    
    🎯 我們想要估計的參數 (Parameters to Estimate):
    • α_r: Regional vulnerability levels (區域脆弱度基準)
    • δ_i: Spatial correlation effects (空間相關效應)  
    • γ_i: Local random effects (局部隨機效應)
    • σ: Observation noise (觀測誤差)
    • ρ: Spatial correlation range (空間相關範圍)
    
    🔬 物理假設 (Physical Assumption):
    Loss_i = Exposure_i × β_i × f(Hazard_i) + ε_i
    
    其中：
    • β_i = α_r(i) + δ_i + γ_i  (total vulnerability at location i)
    • f(Hazard_i) = Emanuel USA function  (標準化風損函數)
    • ε_i ~ Normal(0, σ)  (觀測誤差)
    """
    print(concrete_target)

def show_what_parameters_mean():
    """
    解釋參數的物理意義
    """
    
    print("\n🧬 PHYSICAL MEANING OF PARAMETERS")
    print("-" * 60)
    
    parameter_meanings = """
    每個參數都有明確的物理意義：
    
    🏛️ α_r (Regional Effects):
    • 代表不同區域的基準脆弱度
    • 例如：沿海 vs 內陸, 都市 vs 鄉村
    • 反映建築標準、經濟發展水平的差異
    
    🌐 δ_i (Spatial Correlation Effects):  
    • 代表由於地理位置造成的相關性
    • 相鄰地點往往有相似的脆弱度
    • 反映地形、土壤、微氣候的影響
    
    🎲 γ_i (Local Random Effects):
    • 代表每個地點獨特的局部因素
    • 無法由區域或空間效應解釋的變異
    • 反映建築品質、維護狀況等局部因素
    
    📏 σ (Observation Error):
    • 代表模型無法完美預測的不確定性
    • 包含測量誤差、模型簡化誤差
    • 反映現實世界的復雜性
    
    📡 ρ (Spatial Correlation Range):
    • 代表空間相關性的影響範圍
    • 多遠的地點還會互相影響
    • 反映地理和氣象過程的尺度
    """
    print(parameter_meanings)

def show_fitting_process():
    """
    展示擬合過程
    """
    
    print("\n⚙️ THE FITTING PROCESS")
    print("-" * 60)
    
    fitting_process = """
    擬合過程就是在問：
    
    🔍 給定觀測到的損失數據，最可能的參數組合是什麼？
    
    Step by Step:
    
    1️⃣ 設定先驗分布 (Prior Beliefs):
    α_r ~ Normal(0, 1)           # 區域效應先驗
    δ_i ~ CAR(W, τ_δ)           # 空間相關先驗  
    γ_i ~ Normal(0, τ_γ)        # 局部效應先驗
    σ ~ HalfNormal(1)           # 誤差項先驗
    
    2️⃣ 定義likelihood (Data Generation Process):
    expected_loss_i = exposure_i × β_i × emanuel_usa(hazard_i)
    observed_loss_i ~ Normal(expected_loss_i, σ)
    
    3️⃣ 貝氏推論 (Bayesian Inference):
    posterior ∝ likelihood × prior
    
    找到參數值使得：
    "在這些參數下，觀測到這些損失數據的機率最大"
    
    4️⃣ MCMC取樣 (MCMC Sampling):
    從後驗分布取樣，得到參數的不確定性分布
    """
    print(fitting_process)

def demonstrate_with_example():
    """
    用具體例子演示
    """
    
    print("\n📌 CONCRETE EXAMPLE")
    print("-" * 60)
    
    # 模擬數據
    np.random.seed(42)
    n_locations = 10
    
    # 真實參數值 (未知，要估計的)
    true_alpha_coastal = 0.3
    true_alpha_inland = 0.1
    true_sigma = 0.2
    
    # 模擬觀測數據
    locations = ['Coastal'] * 5 + ['Inland'] * 5
    exposures = np.random.lognormal(17, 0.5, n_locations)  # 資產價值
    hazards = np.random.uniform(30, 70, n_locations)       # 風速
    
    # 真實脆弱度 (未知)
    true_betas = np.array([true_alpha_coastal if loc == 'Coastal' else true_alpha_inland 
                          for loc in locations])
    
    # 觀測損失 (已知數據)
    emanuel_factors = hazards / 50.0  # 簡化的Emanuel function
    expected_losses = exposures * true_betas * emanuel_factors
    observed_losses = expected_losses + np.random.normal(0, true_sigma * expected_losses)
    
    print("模擬的觀測數據 (這是我們已知的):")
    print("-" * 40)
    for i in range(n_locations):
        print(f"地點{i+1:2d} ({locations[i]:7s}): "
              f"損失={observed_losses[i]/1e6:5.1f}M$, "
              f"風速={hazards[i]:4.1f}m/s, "
              f"暴險={exposures[i]/1e8:4.1f}億$")
    
    print(f"\n我們想要估計的未知參數 (真實值):")
    print(f"• α_coastal = {true_alpha_coastal}")
    print(f"• α_inland  = {true_alpha_inland}")
    print(f"• σ         = {true_sigma}")
    
    print(f"\n擬合過程會嘗試找到最佳的參數估計值，使得：")
    print(f"p(observed_losses | α, σ, exposures, hazards) 最大")

def show_what_we_learn():
    """
    展示我們從擬合中學到什麼
    """
    
    print("\n🎓 WHAT WE LEARN FROM FITTING")
    print("-" * 60)
    
    learning_outcomes = """
    從擬合結果我們可以學到：
    
    🗺️ 空間脆弱度地圖 (Spatial Vulnerability Map):
    • 哪些區域特別脆弱？
    • 脆弱度的空間分布模式？
    • 相鄰地點的相似程度？
    
    📊 不確定性量化 (Uncertainty Quantification):
    • 我們對脆弱度估計有多確定？
    • 哪些地點的估計比較可靠？
    • 預測的置信區間有多寬？
    
    🔮 預測能力 (Predictive Capability):
    • 對於新的颶風事件，各地預期損失？
    • 氣候變遷下的風險如何變化？
    • 極端事件的影響程度？
    
    💡 科學洞察 (Scientific Insights):
    • 區域效應 vs 局部效應的相對重要性？
    • 空間相關性的典型範圍？
    • Emanuel函數在當地的適用性？
    
    🎯 保險應用 (Insurance Applications):
    • 基於脆弱度的風險分級
    • 空間相關性對組合風險的影響
    • 參數不確定性對保險定價的影響
    """
    print(learning_outcomes)

def show_NOT_fitting():
    """
    強調我們不是在擬合什麼
    """
    
    print("\n❌ WHAT WE ARE NOT FITTING")
    print("-" * 60)
    
    not_fitting = """
    重要：我們不是在擬合保險產品！
    
    ❌ 我們不是在優化：
    • 保險觸發閾值 (trigger thresholds)
    • 保險賠付金額 (payout amounts)  
    • 基差風險 (basis risk)
    • 保險費率 (premium rates)
    • 產品結構 (product structure)
    
    ❌ 我們不是在最小化：
    • CRPS (那是評估指標)
    • Basis risk (那是決策目標)
    • Insurance losses (那是商業考量)
    
    ✅ 我們只是在理解：
    • 自然災害的脆弱度過程
    • 空間變異性的統計模式
    • 觀測數據背後的物理機制
    
    這是純粹的科學建模，不是商業優化！
    保險產品設計是後續的應用，不是模型擬合的目標。
    """
    print(not_fitting)

def create_visualization():
    """
    視覺化展示擬合目標
    """
    
    print("\n📊 VISUALIZATION: What We Are Fitting")
    print("-" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Observed data scatter
    ax = axes[0, 0]
    np.random.seed(42)
    hazards = np.random.uniform(30, 70, 50)
    coastal = np.random.choice([True, False], 50, p=[0.3, 0.7])
    true_betas = np.where(coastal, 0.3, 0.1)
    losses = true_betas * hazards + np.random.normal(0, 5, 50)
    
    ax.scatter(hazards[coastal], losses[coastal], c='blue', label='Coastal', alpha=0.7)
    ax.scatter(hazards[~coastal], losses[~coastal], c='red', label='Inland', alpha=0.7)
    ax.set_xlabel('Hazard Intensity (m/s)')
    ax.set_ylabel('Normalized Loss')
    ax.set_title('Observed Data\n(What we have)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Vulnerability surface
    ax = axes[0, 1]
    x = np.linspace(0, 100, 20)
    y = np.linspace(0, 50, 20)
    X, Y = np.meshgrid(x, y)
    
    # 模擬脆弱度表面
    coastal_region = (X < 30) | (Y > 30)
    Z = np.where(coastal_region, 0.3, 0.1) + 0.05 * np.sin(X/10) * np.cos(Y/10)
    
    contour = ax.contourf(X, Y, Z, levels=15, cmap='RdYlBu_r')
    ax.set_xlabel('Longitude (km)')
    ax.set_ylabel('Latitude (km)')
    ax.set_title('Vulnerability Surface β(x,y)\n(What we want to estimate)')
    plt.colorbar(contour, ax=ax, label='Vulnerability')
    
    # 3. Parameter posterior
    ax = axes[1, 0]
    posterior_samples = np.random.normal(0.2, 0.05, 1000)
    ax.hist(posterior_samples, bins=30, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(0.2, color='red', linestyle='--', label='True value')
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Posterior Density')
    ax.set_title('Parameter Posterior\n(Uncertainty in estimates)')
    ax.legend()
    
    # 4. Spatial correlation
    ax = axes[1, 1]
    distances = np.linspace(0, 100, 100)
    correlation = np.exp(-distances / 20)  # Exponential decay
    ax.plot(distances, correlation, 'b-', linewidth=2, label='Spatial correlation')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Correlation')
    ax.set_title('Spatial Correlation ρ(d)\n(How locations influence each other)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('what_we_are_fitting.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved as 'what_we_are_fitting.png'")
    plt.show()

def main():
    """主函數"""
    
    print("🎯 EXPLANATION: What Are We Fitting in MCMC?")
    print("=" * 80)
    print()
    
    # 解釋我們在擬合什麼
    explain_what_we_are_fitting()
    
    # 具體的擬合目標
    show_concrete_fitting_target()
    
    # 參數的物理意義
    show_what_parameters_mean()
    
    # 擬合過程
    show_fitting_process()
    
    # 具體例子
    demonstrate_with_example()
    
    # 學習成果
    show_what_we_learn()
    
    # 強調不是在擬合什麼
    show_NOT_fitting()
    
    # 視覺化
    create_visualization()
    
    print("\n" + "=" * 80)
    print("💡 SUMMARY: What 'Fitting Observed Data' Means")
    print("-" * 40)
    summary = """
    「擬合觀測數據」means:
    
    🎯 我們在擬合：
    • 脆弱度函數的空間變異性 β_i = α_r(i) + δ_i + γ_i
    • 不同地點對相同颶風強度的不同反應
    • 空間相關性和區域差異的統計模式
    
    📊 具體來說：
    • 輸入：CLIMADA計算的歷史損失 + 風速 + 暴險值 + 位置
    • 輸出：每個地點的脆弱度參數 + 不確定性估計
    • 目標：理解「為什麼不同地方損失不同」的科學機制
    
    ⚠️ 我們不是在擬合：
    • 保險產品參數
    • 基差風險
    • 商業決策
    
    這是純粹的科學建模！保險產品設計是後續應用。
    """
    print(summary)

if __name__ == "__main__":
    main()