"""
Bayesian Decision Theory Framework (方法二)
貝葉斯決策理論框架

Implements the decision-theoretic approach from bayesian_implement.md:
1. Define loss function L(θ, a) for basis risk quantification
2. Calculate expected loss over posterior distribution
3. Optimize product parameters to minimize expected basis risk

This module implements various basis risk loss functions:
- Absolute basis risk: |Actual_Loss(θ) - Payout(a)|
- Asymmetric basis risk: max(0, Actual_Loss(θ) - Payout(a))
- Weighted asymmetric basis risk with different penalties
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy.optimize import minimize, differential_evolution
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BasisRiskType(Enum):
    """基差風險類型"""
    ABSOLUTE = "absolute"
    ASYMMETRIC_UNDER = "asymmetric_under"
    WEIGHTED_ASYMMETRIC = "weighted_asymmetric"

@dataclass
class ProductParameters:
    """保險產品參數"""
    product_id: str
    trigger_threshold: float  # 觸發閾值 (如風速 m/s)
    payout_amount: float     # 賠付金額 (USD)
    max_payout: float        # 最大賠付 (USD)
    product_type: str = "single_threshold"
    additional_params: Dict[str, Any] = None

@dataclass 
class BasisRiskLossFunction:
    """基差風險損失函數"""
    risk_type: BasisRiskType
    w_under: float = 1.0     # 賠不夠的懲罰權重
    w_over: float = 0.3      # 賠多了的懲罰權重
    
    def calculate_loss(self, actual_loss: float, payout: float) -> float:
        """計算基差風險損失"""
        
        if self.risk_type == BasisRiskType.ABSOLUTE:
            # 絕對基差風險
            return abs(actual_loss - payout)
            
        elif self.risk_type == BasisRiskType.ASYMMETRIC_UNDER:
            # 不對稱基差風險 (只懲罰賠不夠)
            return max(0, actual_loss - payout)
            
        elif self.risk_type == BasisRiskType.WEIGHTED_ASYMMETRIC:
            # 加權不對稱基差風險
            under_coverage = max(0, actual_loss - payout)
            over_coverage = max(0, payout - actual_loss)
            return self.w_under * under_coverage + self.w_over * over_coverage
            
        else:
            raise ValueError(f"Unsupported risk type: {self.risk_type}")

@dataclass
class DecisionTheoryResult:
    """決策理論優化結果"""
    optimal_product: ProductParameters
    expected_loss: float
    loss_breakdown: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]

class BayesianDecisionTheory:
    """
    貝葉斯決策理論優化框架
    
    實現方法二：使用後驗分佈優化產品參數以最小化期望基差風險
    """
    
    def __init__(self,
                 loss_function: BasisRiskLossFunction,
                 random_seed: int = 42):
        """
        初始化決策理論框架
        
        Parameters:
        -----------
        loss_function : BasisRiskLossFunction
            基差風險損失函數
        random_seed : int
            隨機種子
        """
        self.loss_function = loss_function
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 存儲結果
        self.optimization_results = []
        
    def simulate_actual_losses(self,
                              posterior_samples: np.ndarray,
                              hazard_indices: np.ndarray,
                              loss_model: Optional[Callable] = None) -> np.ndarray:
        """
        根據後驗樣本模擬真實經濟損失
        
        Parameters:
        -----------
        posterior_samples : np.ndarray
            MCMC 後驗樣本 (θ_i)
        hazard_indices : np.ndarray
            災害指標 (如風速、降雨等)
        loss_model : Callable, optional
            損失模型函數
            
        Returns:
        --------
        np.ndarray
            模擬的真實損失分佈
        """
        
        if loss_model is None:
            # 使用預設的損失模型
            loss_model = self._default_loss_model
        
        n_samples = len(posterior_samples)
        n_events = len(hazard_indices)
        
        # 為每個後驗樣本和每個事件生成損失
        actual_losses = np.zeros((n_samples, n_events))
        
        for i, theta_i in enumerate(posterior_samples):
            for j, hazard_idx in enumerate(hazard_indices):
                actual_losses[i, j] = loss_model(theta_i, hazard_idx)
        
        return actual_losses
    
    def _default_loss_model(self, theta: float, hazard_index: float) -> float:
        """
        預設損失模型
        
        基於災害指標和模型參數計算經濟損失
        """
        
        # 簡化的災害-損失關係
        if hazard_index < 30:  # 輕微災害
            base_loss = 0
        elif hazard_index < 40:  # 中等災害
            base_loss = 1e7 * (hazard_index - 30) / 10
        elif hazard_index < 50:  # 嚴重災害
            base_loss = 1e7 + 5e7 * (hazard_index - 40) / 10
        else:  # 極端災害
            base_loss = 6e7 + 2e8 * min((hazard_index - 50) / 20, 1.0)
        
        # 加入模型不確定性
        uncertainty_factor = np.exp(np.random.normal(0, 0.2))  # 20% 不確定性
        
        return base_loss * theta * uncertainty_factor
    
    def calculate_product_payout(self,
                                product: ProductParameters,
                                hazard_index: float) -> float:
        """
        計算保險產品賠付
        
        Parameters:
        -----------
        product : ProductParameters
            產品參數
        hazard_index : float
            災害指標值
            
        Returns:
        --------
        float
            賠付金額
        """
        
        if product.product_type == "single_threshold":
            if hazard_index >= product.trigger_threshold:
                return min(product.payout_amount, product.max_payout)
            else:
                return 0.0
                
        elif product.product_type == "double_threshold":
            # 假設產品有兩個閾值
            low_threshold = product.additional_params.get('low_threshold', product.trigger_threshold * 0.8)
            high_threshold = product.additional_params.get('high_threshold', product.trigger_threshold * 1.2)
            low_payout = product.additional_params.get('low_payout', product.payout_amount * 0.5)
            
            if hazard_index >= high_threshold:
                return min(product.payout_amount, product.max_payout)
            elif hazard_index >= low_threshold:
                return min(low_payout, product.max_payout)
            else:
                return 0.0
        
        else:
            raise ValueError(f"Unsupported product type: {product.product_type}")
    
    def calculate_expected_loss(self,
                               product: ProductParameters,
                               posterior_samples: np.ndarray,
                               hazard_indices: np.ndarray,
                               actual_losses: np.ndarray) -> float:
        """
        計算產品的期望損失
        
        Expected_Loss(product) = (1/N) * Σ L(θ_i, product)
        
        Parameters:
        -----------
        product : ProductParameters
            產品參數
        posterior_samples : np.ndarray
            後驗樣本
        hazard_indices : np.ndarray
            災害指標
        actual_losses : np.ndarray
            真實損失矩陣 (n_samples × n_events)
            
        Returns:
        --------
        float
            期望損失
        """
        
        n_samples, n_events = actual_losses.shape
        total_loss = 0.0
        
        # 遍歷每個後驗樣本
        for i in range(n_samples):
            sample_loss = 0.0
            
            # 遍歷每個事件
            for j in range(n_events):
                # 計算真實損失
                true_loss = actual_losses[i, j]
                
                # 計算保險賠付
                payout = self.calculate_product_payout(product, hazard_indices[j])
                
                # 計算基差風險損失
                basis_risk_loss = self.loss_function.calculate_loss(true_loss, payout)
                
                sample_loss += basis_risk_loss
            
            total_loss += sample_loss / n_events  # 平均每個事件的損失
        
        # 計算期望損失
        expected_loss = total_loss / n_samples
        
        return expected_loss
    
    def optimize_single_product(self,
                               posterior_samples: np.ndarray,
                               hazard_indices: np.ndarray,
                               actual_losses: np.ndarray,
                               product_bounds: Dict[str, Tuple[float, float]],
                               initial_guess: Optional[ProductParameters] = None) -> DecisionTheoryResult:
        """
        優化單一產品參數以最小化期望基差風險
        
        Parameters:
        -----------
        posterior_samples : np.ndarray
            後驗樣本
        hazard_indices : np.ndarray
            災害指標
        actual_losses : np.ndarray
            真實損失矩陣
        product_bounds : Dict[str, Tuple[float, float]]
            參數邊界，如 {'trigger_threshold': (30, 60), 'payout_amount': (1e7, 1e9)}
        initial_guess : ProductParameters, optional
            初始猜測
            
        Returns:
        --------
        DecisionTheoryResult
            優化結果
        """
        
        print("🎯 開始方法二：貝葉斯決策理論優化")
        print("=" * 80)
        
        # 準備優化函數
        optimization_history = []
        
        def objective_function(x: np.ndarray) -> float:
            """目標函數：期望基差風險損失"""
            
            trigger_threshold = x[0]
            payout_amount = x[1]
            
            # 創建臨時產品
            temp_product = ProductParameters(
                product_id="temp_optimization",
                trigger_threshold=trigger_threshold,
                payout_amount=payout_amount,
                max_payout=product_bounds.get('max_payout', (1e9, 1e9))[1],
                product_type="single_threshold"
            )
            
            # 計算期望損失
            expected_loss = self.calculate_expected_loss(
                temp_product, posterior_samples, hazard_indices, actual_losses
            )
            
            # 記錄優化歷程
            optimization_history.append({
                'trigger_threshold': trigger_threshold,
                'payout_amount': payout_amount,
                'expected_loss': expected_loss
            })
            
            return expected_loss
        
        # 設置邊界
        bounds = [
            product_bounds.get('trigger_threshold', (30, 60)),
            product_bounds.get('payout_amount', (1e7, 1e9))
        ]
        
        # 初始猜測
        if initial_guess is not None:
            x0 = [initial_guess.trigger_threshold, initial_guess.payout_amount]
        else:
            x0 = [
                (bounds[0][0] + bounds[0][1]) / 2,
                (bounds[1][0] + bounds[1][1]) / 2
            ]
        
        print(f"初始參數: trigger={x0[0]:.1f}, payout={x0[1]:.2e}")
        
        # 執行優化
        print("🔧 執行差分進化優化...")
        
        result = differential_evolution(
            objective_function,
            bounds=bounds,
            seed=self.random_seed,
            maxiter=100,
            popsize=15
        )
        
        if result.success:
            print("✅ 優化成功完成")
            optimal_trigger = result.x[0]
            optimal_payout = result.x[1]
            optimal_loss = result.fun
            
            print(f"最優參數:")
            print(f"  觸發閾值: {optimal_trigger:.2f}")
            print(f"  賠付金額: ${optimal_payout:.2e}")
            print(f"  期望基差風險: ${optimal_loss:.2e}")
        else:
            print("❌ 優化失敗")
            optimal_trigger = x0[0]
            optimal_payout = x0[1]
            optimal_loss = float('inf')
        
        # 創建最優產品
        optimal_product = ProductParameters(
            product_id=f"optimized_{self.loss_function.risk_type.value}",
            trigger_threshold=optimal_trigger,
            payout_amount=optimal_payout,
            max_payout=product_bounds.get('max_payout', (1e9, 1e9))[1],
            product_type="single_threshold"
        )
        
        # 計算損失分解
        loss_breakdown = self._analyze_loss_breakdown(
            optimal_product, posterior_samples, hazard_indices, actual_losses
        )
        
        # 收集收斂信息
        convergence_info = {
            'success': result.success,
            'nfev': result.nfev,
            'nit': result.nit if hasattr(result, 'nit') else 0,
            'message': result.message if hasattr(result, 'message') else ""
        }
        
        decision_result = DecisionTheoryResult(
            optimal_product=optimal_product,
            expected_loss=optimal_loss,
            loss_breakdown=loss_breakdown,
            optimization_history=optimization_history,
            convergence_info=convergence_info
        )
        
        self.optimization_results.append(decision_result)
        
        return decision_result
    
    def _analyze_loss_breakdown(self,
                               product: ProductParameters,
                               posterior_samples: np.ndarray,
                               hazard_indices: np.ndarray,
                               actual_losses: np.ndarray) -> Dict[str, float]:
        """分析損失組成"""
        
        n_samples, n_events = actual_losses.shape
        
        total_under_coverage = 0.0
        total_over_coverage = 0.0
        total_absolute_diff = 0.0
        trigger_events = 0
        
        for i in range(n_samples):
            for j in range(n_events):
                true_loss = actual_losses[i, j]
                payout = self.calculate_product_payout(product, hazard_indices[j])
                
                if payout > 0:
                    trigger_events += 1
                
                under_coverage = max(0, true_loss - payout)
                over_coverage = max(0, payout - true_loss)
                
                total_under_coverage += under_coverage
                total_over_coverage += over_coverage
                total_absolute_diff += abs(true_loss - payout)
        
        n_total = n_samples * n_events
        
        return {
            'average_under_coverage': total_under_coverage / n_total,
            'average_over_coverage': total_over_coverage / n_total,
            'average_absolute_diff': total_absolute_diff / n_total,
            'trigger_rate': trigger_events / n_total,
            'n_evaluations': n_total
        }
    
    def compare_multiple_products(self,
                                 products: List[ProductParameters],
                                 posterior_samples: np.ndarray,
                                 hazard_indices: np.ndarray,
                                 actual_losses: np.ndarray) -> pd.DataFrame:
        """
        比較多個產品的期望基差風險
        
        Parameters:
        -----------
        products : List[ProductParameters]
            候選產品列表
        posterior_samples : np.ndarray
            後驗樣本
        hazard_indices : np.ndarray
            災害指標
        actual_losses : np.ndarray
            真實損失矩陣
            
        Returns:
        --------
        pd.DataFrame
            產品比較結果
        """
        
        print("📊 比較候選產品的期望基差風險...")
        
        comparison_results = []
        
        for product in products:
            expected_loss = self.calculate_expected_loss(
                product, posterior_samples, hazard_indices, actual_losses
            )
            
            loss_breakdown = self._analyze_loss_breakdown(
                product, posterior_samples, hazard_indices, actual_losses
            )
            
            comparison_results.append({
                'product_id': product.product_id,
                'trigger_threshold': product.trigger_threshold,
                'payout_amount': product.payout_amount,
                'expected_basis_risk': expected_loss,
                'under_coverage': loss_breakdown['average_under_coverage'],
                'over_coverage': loss_breakdown['average_over_coverage'],
                'trigger_rate': loss_breakdown['trigger_rate']
            })
        
        df_results = pd.DataFrame(comparison_results)
        df_results = df_results.sort_values('expected_basis_risk')
        
        print("\n產品比較結果:")
        print(df_results.to_string(index=False))
        
        return df_results
    
    def create_loss_function(self,
                            risk_type: str = "weighted_asymmetric",
                            w_under: float = 2.0,
                            w_over: float = 0.5) -> BasisRiskLossFunction:
        """
        創建基差風險損失函數
        
        Parameters:
        -----------
        risk_type : str
            風險類型 ("absolute", "asymmetric_under", "weighted_asymmetric")
        w_under : float
            賠不夠的懲罰權重
        w_over : float
            賠多了的懲罰權重
            
        Returns:
        --------
        BasisRiskLossFunction
            基差風險損失函數
        """
        
        risk_type_enum = BasisRiskType(risk_type)
        
        return BasisRiskLossFunction(
            risk_type=risk_type_enum,
            w_under=w_under,
            w_over=w_over
        )