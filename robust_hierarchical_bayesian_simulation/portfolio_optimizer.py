#!/usr/bin/env python3
"""
Portfolio Optimizer for Multi-Scale Modeling
投資組合優化器 - 多尺度建模

實現多尺度架構的核心：
- 精細尺度建模: 醫院級Cat-in-Circle觸發
- 集合尺度優化: 投資組合級基差風險最小化

優化目標:
a* = argmin_a E[|L_total - Payout_total|]
其中:
- L_total = Σ(i=1 to n) L_i  (投資組合總損失)  
- Payout_total = Σ(i=1 to n) Payout(a_i, H_i)  (投資組合總賠付)

用法:
from robust_hierarchical_bayesian_simulation.portfolio_optimizer import PortfolioOptimizer
optimizer = PortfolioOptimizer(spatial_data, products)
optimal_allocation = optimizer.optimize_portfolio_allocation()
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from .spatial_data_processor import SpatialData


@dataclass 
class ProductAllocation:
    """產品分配結果"""
    hospital_products: Dict[int, str]    # 每家醫院的產品分配
    portfolio_basis_risk: float          # 投資組合基差風險
    individual_basis_risks: np.ndarray   # 每家醫院的基差風險
    total_expected_payout: float         # 總期望賠付
    total_expected_loss: float           # 總期望損失
    coverage_ratio: float                # 覆蓋率
    allocation_weights: np.ndarray       # 分配權重 (如果使用)


class PortfolioOptimizer:
    """投資組合優化器"""
    
    def __init__(self, 
                 spatial_data: SpatialData,
                 insurance_products: List[Dict],
                 loss_function: str = "weighted_asymmetric"):
        """
        初始化優化器
        
        Parameters:
        -----------
        spatial_data : SpatialData
            空間數據 (包含Cat-in-Circle觸發數據)
        insurance_products : List[Dict]
            保險產品列表 (來自03_insurance_product.py)
        loss_function : str
            損失函數類型 ("absolute", "asymmetric", "weighted_asymmetric")
        """
        self.spatial_data = spatial_data
        self.products = insurance_products
        self.loss_function = loss_function
        
        # 從產品中提取觸發參數
        self.product_params = self._extract_product_parameters()
        
        print(f"🎯 投資組合優化器初始化")
        print(f"   醫院數: {spatial_data.n_hospitals}")
        print(f"   產品數: {len(insurance_products)}")
        print(f"   損失函數: {loss_function}")
    
    def optimize_portfolio_allocation(self, 
                                    method: str = "differential_evolution",
                                    n_monte_carlo: int = 1000) -> ProductAllocation:
        """
        優化投資組合產品分配
        
        Parameters:
        -----------
        method : str
            優化方法 ("differential_evolution", "discrete_search", "genetic")
        n_monte_carlo : int
            蒙特卡羅模擬次數
            
        Returns:
        --------
        ProductAllocation : 最優分配結果
        """
        print(f"\n🔍 開始投資組合優化...")
        print(f"   優化方法: {method}")
        print(f"   蒙特卡羅次數: {n_monte_carlo}")
        
        if method == "differential_evolution":
            return self._optimize_continuous(n_monte_carlo)
        elif method == "discrete_search":
            return self._optimize_discrete()
        else:
            raise ValueError(f"未支持的優化方法: {method}")
    
    def evaluate_allocation(self,
                          hospital_product_indices: np.ndarray,
                          return_details: bool = False) -> float:
        """
        評估給定分配的投資組合基差風險
        
        Parameters:
        -----------
        hospital_product_indices : np.ndarray, shape (n_hospitals,)
            每家醫院的產品索引
        return_details : bool
            是否返回詳細信息
            
        Returns:
        --------
        float : 投資組合基差風險
        """
        n_hospitals = self.spatial_data.n_hospitals
        n_events = self.spatial_data.hazard_intensities.shape[1]
        
        # 計算每個事件的投資組合基差風險
        event_basis_risks = []
        
        for event_idx in range(n_events):
            # 投資組合總實際損失
            total_actual_loss = self.spatial_data.observed_losses[:, event_idx].sum()
            
            # 投資組合總賠付
            total_payout = 0.0
            for hospital_idx in range(n_hospitals):
                product_idx = int(hospital_product_indices[hospital_idx])
                product = self.products[product_idx]
                
                # 該醫院在該事件中的Cat-in-Circle觸發
                hospital_hazard = self.spatial_data.hazard_intensities[hospital_idx, event_idx]
                
                # 計算賠付
                payout = self._calculate_payout(product, hospital_hazard)
                total_payout += payout
            
            # 計算基差風險
            basis_risk = self._calculate_basis_risk(total_actual_loss, total_payout)
            event_basis_risks.append(basis_risk)
        
        # 投資組合基差風險 = 所有事件基差風險的期望
        portfolio_basis_risk = np.mean(event_basis_risks)
        
        if return_details:
            return {
                "portfolio_basis_risk": portfolio_basis_risk,
                "event_basis_risks": np.array(event_basis_risks),
                "mean_total_loss": np.mean([self.spatial_data.observed_losses[:, i].sum() 
                                          for i in range(n_events)]),
                "mean_total_payout": np.mean([self._calculate_total_payout(hospital_product_indices, i) 
                                            for i in range(n_events)])
            }
        
        return portfolio_basis_risk
    
    def analyze_spatial_correlation_impact(self,
                                         allocation: np.ndarray) -> Dict:
        """
        分析空間相關性對投資組合風險的影響
        
        Parameters:
        -----------
        allocation : np.ndarray
            產品分配
            
        Returns:
        --------
        Dict : 空間相關性分析結果
        """
        n_hospitals = self.spatial_data.n_hospitals
        n_events = self.spatial_data.hazard_intensities.shape[1]
        
        # 計算醫院級基差風險矩陣
        hospital_basis_risks = np.zeros((n_hospitals, n_events))
        
        for event_idx in range(n_events):
            for hospital_idx in range(n_hospitals):
                actual_loss = self.spatial_data.observed_losses[hospital_idx, event_idx]
                
                product_idx = int(allocation[hospital_idx])
                product = self.products[product_idx]
                hazard = self.spatial_data.hazard_intensities[hospital_idx, event_idx]
                
                payout = self._calculate_payout(product, hazard)
                basis_risk = self._calculate_basis_risk(actual_loss, payout)
                
                hospital_basis_risks[hospital_idx, event_idx] = basis_risk
        
        # 計算空間相關性
        avg_hospital_risks = hospital_basis_risks.mean(axis=1)
        risk_correlation_matrix = np.corrcoef(hospital_basis_risks)
        
        # 距離 vs 風險相關性
        distance_risk_correlations = []
        for i in range(n_hospitals):
            for j in range(i+1, n_hospitals):
                distance = self.spatial_data.distance_matrix[i, j]
                risk_corr = risk_correlation_matrix[i, j]
                distance_risk_correlations.append((distance, risk_corr))
        
        return {
            "hospital_basis_risks": hospital_basis_risks,
            "risk_correlation_matrix": risk_correlation_matrix,
            "distance_risk_correlations": distance_risk_correlations,
            "spatial_clustering_effect": np.mean([corr for dist, corr in distance_risk_correlations if dist < 50])
        }
    
    def _optimize_continuous(self, n_monte_carlo: int) -> ProductAllocation:
        """連續優化 (使用差分進化)"""
        n_hospitals = self.spatial_data.n_hospitals
        n_products = len(self.products)
        
        def objective(x):
            # x是連續變量，轉換為離散產品索引
            product_indices = np.clip(np.round(x * (n_products - 1)), 0, n_products - 1)
            return self.evaluate_allocation(product_indices)
        
        # 優化
        bounds = [(0, 1) for _ in range(n_hospitals)]
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=100,
            popsize=15
        )
        
        # 轉換最優解
        optimal_indices = np.clip(np.round(result.x * (n_products - 1)), 0, n_products - 1)
        
        return self._create_allocation_result(optimal_indices, result.fun)
    
    def _optimize_discrete(self) -> ProductAllocation:
        """離散搜索優化"""
        n_hospitals = self.spatial_data.n_hospitals
        n_products = len(self.products)
        
        if n_hospitals <= 5:
            # 小規模：窮舉搜索
            return self._exhaustive_search()
        else:
            # 大規模：隨機搜索 + 局部改進
            return self._random_search_with_local_improvement(n_iterations=1000)
    
    def _exhaustive_search(self) -> ProductAllocation:
        """窮舉搜索 (僅適用於小規模問題)"""
        from itertools import product as itertools_product
        
        n_hospitals = self.spatial_data.n_hospitals
        n_products = len(self.products)
        
        best_allocation = None
        best_risk = float('inf')
        
        print(f"   窮舉搜索: {n_products}^{n_hospitals} = {n_products**n_hospitals} 種組合")
        
        count = 0
        for allocation in itertools_product(range(n_products), repeat=n_hospitals):
            if count % 1000 == 0:
                print(f"     進度: {count}/{n_products**n_hospitals}")
            
            risk = self.evaluate_allocation(np.array(allocation))
            
            if risk < best_risk:
                best_risk = risk
                best_allocation = allocation
            
            count += 1
        
        return self._create_allocation_result(np.array(best_allocation), best_risk)
    
    def _random_search_with_local_improvement(self, n_iterations: int) -> ProductAllocation:
        """隨機搜索 + 局部改進"""
        n_hospitals = self.spatial_data.n_hospitals
        n_products = len(self.products)
        
        best_allocation = np.random.randint(0, n_products, n_hospitals)
        best_risk = self.evaluate_allocation(best_allocation)
        
        for i in range(n_iterations):
            if i % 100 == 0:
                print(f"     隨機搜索進度: {i}/{n_iterations}, 當前最佳: {best_risk:.6f}")
            
            # 隨機搜索
            if np.random.random() < 0.7:
                # 70%概率：隨機分配
                candidate = np.random.randint(0, n_products, n_hospitals)
            else:
                # 30%概率：從當前最佳進行局部改進
                candidate = best_allocation.copy()
                # 隨機改變1-2個醫院的產品
                n_changes = np.random.randint(1, min(3, n_hospitals + 1))
                change_indices = np.random.choice(n_hospitals, n_changes, replace=False)
                for idx in change_indices:
                    candidate[idx] = np.random.randint(0, n_products)
            
            risk = self.evaluate_allocation(candidate)
            
            if risk < best_risk:
                best_risk = risk
                best_allocation = candidate.copy()
        
        return self._create_allocation_result(best_allocation, best_risk)
    
    def _calculate_payout(self, product: Dict, hazard_intensity: float) -> float:
        """計算單個產品的賠付"""
        thresholds = product.get('trigger_thresholds', [])
        ratios = product.get('payout_ratios', [])
        max_payout = product.get('max_payout', 0)
        
        if not thresholds or not ratios:
            return 0.0
        
        # 階梯式賠付
        for i, threshold in enumerate(thresholds):
            if hazard_intensity >= threshold:
                if i < len(ratios):
                    return max_payout * ratios[i]
        
        return 0.0
    
    def _calculate_total_payout(self, allocation: np.ndarray, event_idx: int) -> float:
        """計算總賠付"""
        total = 0.0
        for hospital_idx in range(self.spatial_data.n_hospitals):
            product_idx = int(allocation[hospital_idx])
            product = self.products[product_idx]
            hazard = self.spatial_data.hazard_intensities[hospital_idx, event_idx]
            total += self._calculate_payout(product, hazard)
        
        return total
    
    def _calculate_basis_risk(self, actual_loss: float, payout: float) -> float:
        """計算基差風險"""
        diff = actual_loss - payout
        
        if self.loss_function == "absolute":
            return abs(diff)
        elif self.loss_function == "asymmetric":
            return max(0, diff)  # 只懲罰賠付不足
        elif self.loss_function == "weighted_asymmetric":
            if diff > 0:  # 賠付不足
                return 2.0 * diff
            else:  # 賠付過多
                return 0.5 * abs(diff)
        else:
            return abs(diff)
    
    def _create_allocation_result(self, allocation: np.ndarray, portfolio_risk: float) -> ProductAllocation:
        """創建分配結果對象"""
        n_events = self.spatial_data.hazard_intensities.shape[1]
        
        # 計算詳細統計
        hospital_products = {}
        individual_risks = np.zeros(self.spatial_data.n_hospitals)
        
        total_expected_payout = 0.0
        total_expected_loss = 0.0
        
        for hospital_idx in range(self.spatial_data.n_hospitals):
            product_idx = int(allocation[hospital_idx])
            product = self.products[product_idx]
            hospital_products[hospital_idx] = product.get('product_id', f'product_{product_idx}')
            
            # 計算該醫院的平均基差風險
            hospital_risks = []
            hospital_payouts = []
            hospital_losses = []
            
            for event_idx in range(n_events):
                actual_loss = self.spatial_data.observed_losses[hospital_idx, event_idx]
                hazard = self.spatial_data.hazard_intensities[hospital_idx, event_idx]
                payout = self._calculate_payout(product, hazard)
                
                risk = self._calculate_basis_risk(actual_loss, payout)
                hospital_risks.append(risk)
                hospital_payouts.append(payout)
                hospital_losses.append(actual_loss)
            
            individual_risks[hospital_idx] = np.mean(hospital_risks)
            total_expected_payout += np.mean(hospital_payouts)
            total_expected_loss += np.mean(hospital_losses)
        
        coverage_ratio = total_expected_payout / total_expected_loss if total_expected_loss > 0 else 0
        
        return ProductAllocation(
            hospital_products=hospital_products,
            portfolio_basis_risk=portfolio_risk,
            individual_basis_risks=individual_risks,
            total_expected_payout=total_expected_payout,
            total_expected_loss=total_expected_loss,
            coverage_ratio=coverage_ratio,
            allocation_weights=allocation / np.sum(allocation) if np.sum(allocation) > 0 else None
        )
    
    def _extract_product_parameters(self) -> List[Dict]:
        """提取產品參數"""
        params = []
        for product in self.products:
            params.append({
                "thresholds": product.get('trigger_thresholds', []),
                "ratios": product.get('payout_ratios', []),
                "max_payout": product.get('max_payout', 0),
                "radius": product.get('radius_km', 50)
            })
        return params


# 使用範例
if __name__ == "__main__":
    from .spatial_data_processor import SpatialDataProcessor
    
    # 創建測試數據
    np.random.seed(42)
    hospital_coords = np.random.uniform([35.0, -84.0], [36.5, -75.5], (3, 2))
    
    processor = SpatialDataProcessor()
    spatial_data = processor.process_hospital_spatial_data(hospital_coords)
    
    # 模擬Cat-in-Circle和產品數據
    n_hospitals, n_events = 3, 10
    hazard_intensities = np.random.uniform(20, 60, (n_hospitals, n_events))
    exposure_values = np.random.uniform(1e7, 5e7, n_hospitals)
    observed_losses = np.random.lognormal(15, 1, (n_hospitals, n_events))
    
    spatial_data = processor.add_cat_in_circle_data(
        hazard_intensities, exposure_values, observed_losses
    )
    
    # 模擬保險產品
    products = [
        {
            "product_id": f"product_{i}",
            "trigger_thresholds": [30 + i*5, 45 + i*5],
            "payout_ratios": [0.5, 1.0],
            "max_payout": 1e8,
            "radius_km": 50
        }
        for i in range(5)
    ]
    
    # 優化投資組合
    optimizer = PortfolioOptimizer(spatial_data, products)
    optimal_allocation = optimizer.optimize_portfolio_allocation(method="discrete_search")
    
    print(f"\n最優分配結果:")
    print(f"投資組合基差風險: {optimal_allocation.portfolio_basis_risk:.6f}")
    print(f"覆蓋率: {optimal_allocation.coverage_ratio:.3f}")
    print(f"醫院產品分配: {optimal_allocation.hospital_products}")