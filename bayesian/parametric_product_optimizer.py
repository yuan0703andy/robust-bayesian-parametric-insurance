#!/usr/bin/env python3
"""
Bayesian Decision Theory Module
貝氏決策理論模組

實現您理論框架中的決策理論核心：
- 後驗期望風險最小化: a* = argmin_a R(a|Data)
- Γ-極小化極大決策: a*_minimax = argmin_a (sup_π R(a|π,Data))
- 完整的決策理論框架

核心功能:
- 後驗期望風險計算
- 產品參數優化
- Γ-minimax穩健決策
- 決策不確定性量化

使用範例:
```python
from bayesian.parametric_product_optimizer import (
    BayesianDecisionOptimizer, ProductSpace, DecisionResult
)

# 定義產品空間
product_space = ProductSpace(
    trigger_bounds=(30, 60),
    payout_bounds=(1e7, 1e9)
)

# 初始化優化器
optimizer = BayesianDecisionOptimizer()

# 執行貝氏最優決策
result = optimizer.optimize_expected_risk(
    posterior_samples_dict,
    hazard_indices, 
    actual_losses,
    product_space
)

print(f"最優產品: {result.optimal_product}")
print(f"期望風險: {result.expected_risk:.2e}")

# 執行Γ-minimax決策
minimax_result = optimizer.gamma_minimax_optimization(
    posterior_samples_dict,
    hazard_indices,
    actual_losses, 
    product_space
)
```

Author: Research Team
Date: 2025-01-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import minimize, differential_evolution
import warnings
import time

# 導入基差風險計算
try:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from skill_scores.basis_risk_functions import (
        BasisRiskCalculator, BasisRiskType, BasisRiskLossFunction
    )
    HAS_BASIS_RISK = True
except ImportError:
    HAS_BASIS_RISK = False
    warnings.warn("basis_risk_functions not available")

@dataclass
class ProductParameters:
    """產品參數"""
    trigger_threshold: float
    payout_amount: float
    max_payout: Optional[float] = None
    product_type: str = "single_threshold"
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.max_payout is None:
            self.max_payout = self.payout_amount
    
    def calculate_payout(self, hazard_value: float) -> float:
        """計算給定災害值的賠付金額"""
        if hazard_value >= self.trigger_threshold:
            return min(self.payout_amount, self.max_payout)
        return 0.0

@dataclass
class ProductSpace:
    """產品參數空間"""
    trigger_bounds: Tuple[float, float]
    payout_bounds: Tuple[float, float]
    max_payout_bounds: Optional[Tuple[float, float]] = None
    grid_resolution: Tuple[int, int] = (20, 20)
    
    def __post_init__(self):
        if self.max_payout_bounds is None:
            self.max_payout_bounds = self.payout_bounds
    
    def sample_random_product(self) -> ProductParameters:
        """隨機採樣一個產品"""
        trigger = np.random.uniform(*self.trigger_bounds)
        payout = np.random.uniform(*self.payout_bounds)
        max_payout = np.random.uniform(*self.max_payout_bounds)
        
        return ProductParameters(
            trigger_threshold=trigger,
            payout_amount=payout,
            max_payout=max_payout
        )
    
    def generate_grid_products(self) -> List[ProductParameters]:
        """生成網格化的產品組合"""
        trigger_grid = np.linspace(*self.trigger_bounds, self.grid_resolution[0])
        payout_grid = np.linspace(*self.payout_bounds, self.grid_resolution[1])
        
        products = []
        for trigger in trigger_grid:
            for payout in payout_grid:
                products.append(ProductParameters(
                    trigger_threshold=trigger,
                    payout_amount=payout
                ))
        
        return products

@dataclass
class DecisionResult:
    """決策結果"""
    optimal_product: ProductParameters
    expected_risk: float
    optimization_method: str
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    risk_breakdown: Dict[str, float] = field(default_factory=dict)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0

@dataclass
class GammaMinimaxResult:
    """Γ-minimax決策結果"""
    minimax_product: ProductParameters
    worst_case_risk: float
    risk_across_models: Dict[str, float] = field(default_factory=dict)
    robustness_metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

class OptimizationMethod(Enum):
    """優化方法"""
    GRID_SEARCH = "grid_search"
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"

@dataclass
class OptimizerConfig:
    """優化器配置"""
    method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    basis_risk_type: BasisRiskType = BasisRiskType.WEIGHTED_ASYMMETRIC
    w_under: float = 2.0  # 賠不夠的懲罰權重
    w_over: float = 0.5   # 賠多了的懲罰權重
    max_iterations: int = 1000
    tolerance: float = 1e-6
    n_random_starts: int = 10
    use_parallel: bool = False

class BayesianDecisionOptimizer:
    """
    貝氏決策理論優化器
    
    實現您理論框架中的決策理論核心：
    1. 後驗期望風險: R(a|Data) = E_p(L|Data)[L(L, a)]
    2. 貝氏最優決策: a* = argmin_a R(a|Data)
    3. Γ-極小化極大: a*_minimax = argmin_a (sup_π R(a|π,Data))
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        初始化貝氏決策優化器
        
        Parameters:
        -----------
        config : OptimizerConfig, optional
            優化器配置
        """
        self.config = config or OptimizerConfig()
        
        # 初始化基差風險計算器
        if HAS_BASIS_RISK:
            self.basis_risk_calculator = BasisRiskCalculator()
            self.loss_function = BasisRiskLossFunction(
                risk_type=self.config.basis_risk_type,
                w_under=self.config.w_under,
                w_over=self.config.w_over
            )
        else:
            self.basis_risk_calculator = None
            self.loss_function = None
            warnings.warn("基差風險計算不可用，使用簡化損失函數")
        
        # 結果存儲
        self.optimization_history: List[DecisionResult] = []
        self.minimax_history: List[GammaMinimaxResult] = []
        
        print("🎯 貝氏決策理論優化器初始化完成")
        print(f"   優化方法: {self.config.method.value}")
        print(f"   基差風險類型: {self.config.basis_risk_type.value}")
        print(f"   懲罰權重: w_under={self.config.w_under}, w_over={self.config.w_over}")
    
    def compute_posterior_expected_risk(self,
                                      product: ProductParameters,
                                      posterior_samples_dict: Dict[str, np.ndarray],
                                      hazard_indices: np.ndarray,
                                      actual_losses: np.ndarray) -> float:
        """
        計算後驗期望風險
        
        實現您理論中的核心公式：
        R(a|Data) = E_p(L_pred|Data)[L(L_pred, a)]
        
        Parameters:
        -----------
        product : ProductParameters
            產品參數
        posterior_samples_dict : Dict[str, np.ndarray]
            後驗樣本字典
        hazard_indices : np.ndarray
            災害指標
        actual_losses : np.ndarray
            實際損失
            
        Returns:
        --------
        float
            後驗期望風險
        """
        total_risk = 0.0
        n_models = len(posterior_samples_dict)
        
        if n_models == 0:
            return float('inf')
        
        # 對每個模型計算期望風險，然後平均
        for model_name, samples in posterior_samples_dict.items():
            model_risk = self._compute_single_model_risk(
                product, samples, hazard_indices, actual_losses
            )
            total_risk += model_risk
        
        return total_risk / n_models
    
    def _compute_single_model_risk(self,
                                 product: ProductParameters,
                                 posterior_samples: Union[np.ndarray, Dict[str, np.ndarray]],
                                 hazard_indices: np.ndarray,
                                 actual_losses: np.ndarray) -> float:
        """計算單一模型的期望風險"""
        if isinstance(posterior_samples, dict):
            # 使用字典中的參數樣本
            n_samples = len(next(iter(posterior_samples.values())))
            sample_indices = np.random.choice(n_samples, size=min(n_samples, 100), replace=False)
        else:
            # 如果是數組，直接使用
            n_samples = len(posterior_samples)
            sample_indices = np.random.choice(n_samples, size=min(n_samples, 100), replace=False)
        
        total_sample_risk = 0.0
        
        # 對選中的樣本計算平均風險
        for sample_idx in sample_indices:
            sample_risk = 0.0
            n_events = min(len(hazard_indices), len(actual_losses))
            
            # 對每個事件計算基差風險
            for event_idx in range(n_events):
                hazard_value = hazard_indices[event_idx]
                actual_loss = actual_losses[event_idx]
                
                # 計算產品賠付
                payout = product.calculate_payout(hazard_value)
                
                # 計算基差風險
                if self.loss_function:
                    basis_risk = self.loss_function.calculate_loss(actual_loss, payout)
                else:
                    # 簡化的基差風險計算
                    under_coverage = max(0, actual_loss - payout)
                    over_coverage = max(0, payout - actual_loss)
                    basis_risk = (self.config.w_under * under_coverage + 
                                self.config.w_over * over_coverage)
                
                sample_risk += basis_risk
            
            # 平均每個事件的風險
            total_sample_risk += sample_risk / n_events if n_events > 0 else 0
        
        # 平均所有樣本的風險
        return total_sample_risk / len(sample_indices) if len(sample_indices) > 0 else float('inf')
    
    def optimize_expected_risk(self,
                             posterior_samples_dict: Dict[str, np.ndarray],
                             hazard_indices: np.ndarray,
                             actual_losses: np.ndarray,
                             product_space: ProductSpace) -> DecisionResult:
        """
        優化後驗期望風險
        
        實現貝氏最優決策：a* = argmin_a R(a|Data)
        
        Parameters:
        -----------
        posterior_samples_dict : Dict[str, np.ndarray]
            後驗樣本字典
        hazard_indices : np.ndarray
            災害指標
        actual_losses : np.ndarray
            實際損失
        product_space : ProductSpace
            產品參數空間
            
        Returns:
        --------
        DecisionResult
            決策結果
        """
        print(f"🎯 執行貝氏最優決策優化...")
        print(f"   模型數量: {len(posterior_samples_dict)}")
        print(f"   事件數量: {len(hazard_indices)}")
        print(f"   優化方法: {self.config.method.value}")
        
        start_time = time.time()
        
        # 定義目標函數
        def objective_function(params):
            trigger_threshold, payout_amount = params
            
            product = ProductParameters(
                trigger_threshold=trigger_threshold,
                payout_amount=payout_amount
            )
            
            return self.compute_posterior_expected_risk(
                product, posterior_samples_dict, hazard_indices, actual_losses
            )
        
        # 根據配置選擇優化方法
        if self.config.method == OptimizationMethod.GRID_SEARCH:
            result = self._optimize_grid_search(objective_function, product_space)
        elif self.config.method == OptimizationMethod.GRADIENT_BASED:
            result = self._optimize_gradient_based(objective_function, product_space)
        elif self.config.method == OptimizationMethod.EVOLUTIONARY:
            result = self._optimize_evolutionary(objective_function, product_space)
        else:
            raise ValueError(f"不支援的優化方法: {self.config.method}")
        
        execution_time = time.time() - start_time
        
        # 創建決策結果
        decision_result = DecisionResult(
            optimal_product=result["product"],
            expected_risk=result["risk"],
            optimization_method=self.config.method.value,
            convergence_info=result["convergence"],
            execution_time=execution_time
        )
        
        self.optimization_history.append(decision_result)
        
        print(f"✅ 貝氏最優決策完成")
        print(f"   最優觸發閾值: {result['product'].trigger_threshold:.2f}")
        print(f"   最優賠付金額: {result['product'].payout_amount:.2e}")
        print(f"   最小期望風險: {result['risk']:.2e}")
        print(f"   執行時間: {execution_time:.2f} 秒")
        
        return decision_result
    
    def _optimize_grid_search(self, objective_func: Callable, product_space: ProductSpace) -> Dict[str, Any]:
        """網格搜索優化"""
        print("   🔍 執行網格搜索...")
        
        trigger_grid = np.linspace(*product_space.trigger_bounds, product_space.grid_resolution[0])
        payout_grid = np.linspace(*product_space.payout_bounds, product_space.grid_resolution[1])
        
        best_risk = float('inf')
        best_params = None
        history = []
        
        total_evaluations = len(trigger_grid) * len(payout_grid)
        evaluation_count = 0
        
        for trigger in trigger_grid:
            for payout in payout_grid:
                try:
                    risk = objective_func([trigger, payout])
                    
                    history.append({
                        "trigger": trigger,
                        "payout": payout,
                        "risk": risk
                    })
                    
                    if risk < best_risk:
                        best_risk = risk
                        best_params = (trigger, payout)
                    
                    evaluation_count += 1
                    if evaluation_count % 50 == 0:
                        print(f"      進度: {evaluation_count}/{total_evaluations}")
                        
                except Exception as e:
                    continue
        
        if best_params is None:
            raise ValueError("網格搜索未找到有效解")
        
        best_product = ProductParameters(
            trigger_threshold=best_params[0],
            payout_amount=best_params[1]
        )
        
        return {
            "product": best_product,
            "risk": best_risk,
            "convergence": {
                "method": "grid_search",
                "evaluations": evaluation_count,
                "history": history
            }
        }
    
    def _optimize_gradient_based(self, objective_func: Callable, product_space: ProductSpace) -> Dict[str, Any]:
        """基於梯度的優化"""
        print("   📈 執行基於梯度的優化...")
        
        # 多個隨機起始點
        best_risk = float('inf')
        best_result = None
        
        for start_idx in range(self.config.n_random_starts):
            # 隨機初始點
            initial_trigger = np.random.uniform(*product_space.trigger_bounds)
            initial_payout = np.random.uniform(*product_space.payout_bounds)
            initial_guess = [initial_trigger, initial_payout]
            
            # 邊界約束
            bounds = [product_space.trigger_bounds, product_space.payout_bounds]
            
            try:
                opt_result = minimize(
                    objective_func,
                    initial_guess,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'ftol': self.config.tolerance,
                        'maxiter': self.config.max_iterations
                    }
                )
                
                if opt_result.success and opt_result.fun < best_risk:
                    best_risk = opt_result.fun
                    best_result = opt_result
                    
            except Exception as e:
                print(f"      起始點 {start_idx+1} 優化失敗: {e}")
                continue
        
        if best_result is None:
            raise ValueError("梯度優化未找到有效解")
        
        best_product = ProductParameters(
            trigger_threshold=best_result.x[0],
            payout_amount=best_result.x[1]
        )
        
        return {
            "product": best_product,
            "risk": best_result.fun,
            "convergence": {
                "method": "gradient_based",
                "success": best_result.success,
                "n_iterations": best_result.nit,
                "n_function_evaluations": best_result.nfev
            }
        }
    
    def _optimize_evolutionary(self, objective_func: Callable, product_space: ProductSpace) -> Dict[str, Any]:
        """進化算法優化"""
        print("   🧬 執行進化算法優化...")
        
        bounds = [product_space.trigger_bounds, product_space.payout_bounds]
        
        result = differential_evolution(
            objective_func,
            bounds,
            maxiter=self.config.max_iterations // 10,  # 調整迭代次數
            popsize=15,
            seed=42,
            atol=self.config.tolerance,
            polish=True
        )
        
        if not result.success:
            raise ValueError(f"進化算法優化失敗: {result.message}")
        
        best_product = ProductParameters(
            trigger_threshold=result.x[0],
            payout_amount=result.x[1]
        )
        
        return {
            "product": best_product,
            "risk": result.fun,
            "convergence": {
                "method": "evolutionary",
                "success": result.success,
                "n_iterations": result.nit,
                "n_function_evaluations": result.nfev
            }
        }
    
    def gamma_minimax_optimization(self,
                                 posterior_samples_dict: Dict[str, np.ndarray],
                                 hazard_indices: np.ndarray,
                                 actual_losses: np.ndarray,
                                 product_space: ProductSpace) -> GammaMinimaxResult:
        """
        Γ-極小化極大優化
        
        實現您理論中的穩健決策：
        a*_minimax = argmin_a (sup_π R(a|π, Data))
        
        Parameters:
        -----------
        posterior_samples_dict : Dict[str, np.ndarray]
            每個模型的後驗樣本
        hazard_indices : np.ndarray
            災害指標
        actual_losses : np.ndarray
            實際損失
        product_space : ProductSpace
            產品參數空間
            
        Returns:
        --------
        GammaMinimaxResult
            Γ-minimax決策結果
        """
        print(f"🛡️ 執行 Γ-極小化極大優化...")
        print(f"   模型數量: {len(posterior_samples_dict)}")
        
        start_time = time.time()
        
        def compute_worst_case_risk(params):
            """計算給定產品參數的最壞情況風險"""
            trigger_threshold, payout_amount = params
            
            product = ProductParameters(
                trigger_threshold=trigger_threshold,
                payout_amount=payout_amount
            )
            
            # 計算每個模型的風險
            model_risks = {}
            for model_name, samples in posterior_samples_dict.items():
                risk = self._compute_single_model_risk(
                    product, samples, hazard_indices, actual_losses
                )
                model_risks[model_name] = risk
            
            # 返回最大風險（worst case）
            worst_case_risk = max(model_risks.values()) if model_risks else float('inf')
            return worst_case_risk, model_risks
        
        # 定義 minimax 目標函數
        def minimax_objective(params):
            worst_risk, _ = compute_worst_case_risk(params)
            return worst_risk
        
        # 執行優化
        if self.config.method == OptimizationMethod.GRID_SEARCH:
            result = self._minimax_grid_search(compute_worst_case_risk, product_space)
        else:
            # 使用梯度方法
            bounds = [product_space.trigger_bounds, product_space.payout_bounds]
            
            best_minimax_risk = float('inf')
            best_result = None
            
            for start_idx in range(self.config.n_random_starts):
                initial_trigger = np.random.uniform(*product_space.trigger_bounds)
                initial_payout = np.random.uniform(*product_space.payout_bounds)
                initial_guess = [initial_trigger, initial_payout]
                
                try:
                    opt_result = minimize(
                        minimax_objective,
                        initial_guess,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'ftol': self.config.tolerance,
                            'maxiter': self.config.max_iterations
                        }
                    )
                    
                    if opt_result.success and opt_result.fun < best_minimax_risk:
                        best_minimax_risk = opt_result.fun
                        best_result = opt_result
                        
                except Exception as e:
                    continue
            
            if best_result is None:
                raise ValueError("Γ-minimax 優化未找到有效解")
            
            # 獲取詳細結果
            worst_risk, model_risks = compute_worst_case_risk(best_result.x)
            
            result = {
                "product": ProductParameters(
                    trigger_threshold=best_result.x[0],
                    payout_amount=best_result.x[1]
                ),
                "worst_case_risk": worst_risk,
                "model_risks": model_risks
            }
        
        execution_time = time.time() - start_time
        
        # 計算穩健性指標
        risk_values = list(result["model_risks"].values())
        robustness_metrics = {
            "risk_range": (np.min(risk_values), np.max(risk_values)),
            "risk_std": np.std(risk_values),
            "risk_coefficient_variation": np.std(risk_values) / np.mean(risk_values) if np.mean(risk_values) > 0 else np.inf,
            "worst_to_best_ratio": np.max(risk_values) / np.min(risk_values) if np.min(risk_values) > 0 else np.inf
        }
        
        minimax_result = GammaMinimaxResult(
            minimax_product=result["product"],
            worst_case_risk=result["worst_case_risk"],
            risk_across_models=result["model_risks"],
            robustness_metrics=robustness_metrics,
            execution_time=execution_time
        )
        
        self.minimax_history.append(minimax_result)
        
        print(f"✅ Γ-極小化極大優化完成")
        print(f"   最優觸發閾值: {result['product'].trigger_threshold:.2f}")
        print(f"   最優賠付金額: {result['product'].payout_amount:.2e}")
        print(f"   最壞情況風險: {result['worst_case_risk']:.2e}")
        print(f"   風險變異係數: {robustness_metrics['risk_coefficient_variation']:.3f}")
        print(f"   執行時間: {execution_time:.2f} 秒")
        
        return minimax_result
    
    def _minimax_grid_search(self, risk_func: Callable, product_space: ProductSpace) -> Dict[str, Any]:
        """Minimax網格搜索"""
        trigger_grid = np.linspace(*product_space.trigger_bounds, product_space.grid_resolution[0])
        payout_grid = np.linspace(*product_space.payout_bounds, product_space.grid_resolution[1])
        
        best_minimax_risk = float('inf')
        best_params = None
        best_model_risks = {}
        
        for trigger in trigger_grid:
            for payout in payout_grid:
                try:
                    worst_risk, model_risks = risk_func([trigger, payout])
                    
                    if worst_risk < best_minimax_risk:
                        best_minimax_risk = worst_risk
                        best_params = (trigger, payout)
                        best_model_risks = model_risks
                        
                except Exception as e:
                    continue
        
        if best_params is None:
            raise ValueError("Minimax網格搜索未找到有效解")
        
        return {
            "product": ProductParameters(
                trigger_threshold=best_params[0],
                payout_amount=best_params[1]
            ),
            "worst_case_risk": best_minimax_risk,
            "model_risks": best_model_risks
        }
    
    def compare_decision_strategies(self,
                                  posterior_samples_dict: Dict[str, np.ndarray],
                                  hazard_indices: np.ndarray,
                                  actual_losses: np.ndarray,
                                  product_space: ProductSpace) -> Dict[str, Any]:
        """
        比較不同決策策略
        
        Parameters:
        -----------
        posterior_samples_dict : Dict[str, np.ndarray]
            後驗樣本字典
        hazard_indices : np.ndarray
            災害指標
        actual_losses : np.ndarray
            實際損失
        product_space : ProductSpace
            產品參數空間
            
        Returns:
        --------
        Dict[str, Any]
            決策策略比較結果
        """
        print(f"📊 比較決策策略...")
        
        # 執行貝氏最優決策
        print("\n1. 執行貝氏最優決策...")
        bayes_optimal = self.optimize_expected_risk(
            posterior_samples_dict, hazard_indices, actual_losses, product_space
        )
        
        # 執行Γ-minimax決策
        print("\n2. 執行Γ-極小化極大決策...")
        gamma_minimax = self.gamma_minimax_optimization(
            posterior_samples_dict, hazard_indices, actual_losses, product_space
        )
        
        # 計算每種策略在所有模型下的風險
        bayes_risks = {}
        minimax_risks = {}
        
        for model_name, samples in posterior_samples_dict.items():
            # 貝氏最優在該模型下的風險
            bayes_risk = self._compute_single_model_risk(
                bayes_optimal.optimal_product, samples, hazard_indices, actual_losses
            )
            bayes_risks[model_name] = bayes_risk
            
            # Minimax在該模型下的風險
            minimax_risk = self._compute_single_model_risk(
                gamma_minimax.minimax_product, samples, hazard_indices, actual_losses
            )
            minimax_risks[model_name] = minimax_risk
        
        # 比較分析
        bayes_risks_values = list(bayes_risks.values())
        minimax_risks_values = list(minimax_risks.values())
        
        comparison = {
            "bayes_optimal": {
                "product": bayes_optimal.optimal_product,
                "expected_risk": bayes_optimal.expected_risk,
                "risks_by_model": bayes_risks,
                "worst_case_risk": np.max(bayes_risks_values),
                "risk_std": np.std(bayes_risks_values),
                "execution_time": bayes_optimal.execution_time
            },
            "gamma_minimax": {
                "product": gamma_minimax.minimax_product,
                "worst_case_risk": gamma_minimax.worst_case_risk,
                "risks_by_model": minimax_risks,
                "average_risk": np.mean(minimax_risks_values),
                "risk_std": np.std(minimax_risks_values),
                "execution_time": gamma_minimax.execution_time
            },
            "comparison_metrics": {
                "worst_case_improvement": (
                    np.max(bayes_risks_values) - gamma_minimax.worst_case_risk
                ) / np.max(bayes_risks_values) if np.max(bayes_risks_values) > 0 else 0,
                "average_risk_trade_off": (
                    np.mean(minimax_risks_values) - bayes_optimal.expected_risk
                ) / bayes_optimal.expected_risk if bayes_optimal.expected_risk > 0 else 0,
                "robustness_gain": (
                    np.std(bayes_risks_values) - np.std(minimax_risks_values)
                ) / np.std(bayes_risks_values) if np.std(bayes_risks_values) > 0 else 0
            }
        }
        
        print(f"\n📋 決策策略比較結果:")
        print(f"   貝氏最優期望風險: {bayes_optimal.expected_risk:.2e}")
        print(f"   貝氏最優最壞風險: {np.max(bayes_risks_values):.2e}")
        print(f"   Minimax最壞風險: {gamma_minimax.worst_case_risk:.2e}")
        print(f"   最壞情況改善: {comparison['comparison_metrics']['worst_case_improvement']:.1%}")
        
        return comparison
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """獲取優化歷史摘要"""
        if not self.optimization_history:
            return pd.DataFrame()
        
        summary_data = []
        
        for i, result in enumerate(self.optimization_history):
            summary_data.append({
                "優化序號": i + 1,
                "方法": result.optimization_method,
                "觸發閾值": result.optimal_product.trigger_threshold,
                "賠付金額": result.optimal_product.payout_amount,
                "期望風險": result.expected_risk,
                "執行時間": result.execution_time
            })
        
        return pd.DataFrame(summary_data)

# 便利函數
def quick_bayesian_optimization(posterior_samples_dict: Dict[str, np.ndarray],
                               hazard_indices: np.ndarray,
                               actual_losses: np.ndarray,
                               trigger_bounds: Tuple[float, float] = (30, 60),
                               payout_bounds: Tuple[float, float] = (1e7, 1e9)) -> DecisionResult:
    """
    便利函數：快速貝氏優化
    
    Parameters:
    -----------
    posterior_samples_dict : Dict[str, np.ndarray]
        後驗樣本字典
    hazard_indices : np.ndarray
        災害指標
    actual_losses : np.ndarray
        實際損失
    trigger_bounds : Tuple[float, float]
        觸發閾值範圍
    payout_bounds : Tuple[float, float]
        賠付金額範圍
        
    Returns:
    --------
    DecisionResult
        決策結果
    """
    product_space = ProductSpace(
        trigger_bounds=trigger_bounds,
        payout_bounds=payout_bounds,
        grid_resolution=(10, 10)
    )
    
    optimizer = BayesianDecisionOptimizer()
    return optimizer.optimize_expected_risk(
        posterior_samples_dict, hazard_indices, actual_losses, product_space
    )

def test_bayesian_decision_theory():
    """測試貝氏決策理論功能"""
    print("🧪 測試貝氏決策理論模組...")
    
    # 生成測試數據
    np.random.seed(42)
    n_events = 100
    
    # 災害指標（風速）
    hazard_indices = np.random.gamma(2, 20, n_events)  # 風速分布
    
    # 實際損失（基於Emanuel關係的模擬）
    actual_losses = np.zeros(n_events)
    for i, wind in enumerate(hazard_indices):
        if wind > 33:  # 颶風風速
            base_loss = ((wind / 33) ** 3.5) * 1e8
            actual_losses[i] = base_loss * np.random.lognormal(0, 0.5)
        else:
            actual_losses[i] = np.random.exponential(1e6) if np.random.random() < 0.1 else 0
    
    # 生成多個模型的後驗樣本
    posterior_samples = {
        "normal_weak": np.random.normal(0, 1, 500),
        "normal_pessimistic": np.random.normal(0, 0.5, 500),
        "student_t": np.random.standard_t(4, 500)
    }
    
    print(f"\n測試數據摘要:")
    print(f"   事件數量: {n_events}")
    print(f"   災害指標範圍: {hazard_indices.min():.1f} - {hazard_indices.max():.1f}")
    print(f"   損失範圍: {actual_losses.min():.2e} - {actual_losses.max():.2e}")
    print(f"   非零損失比例: {np.mean(actual_losses > 0):.1%}")
    
    # 定義產品空間
    product_space = ProductSpace(
        trigger_bounds=(30, 60),
        payout_bounds=(1e7, 5e8),
        grid_resolution=(8, 8)
    )
    
    # 測試貝氏決策優化
    print(f"\n🎯 測試貝氏決策優化...")
    optimizer = BayesianDecisionOptimizer()
    
    bayes_result = optimizer.optimize_expected_risk(
        posterior_samples, hazard_indices, actual_losses, product_space
    )
    
    print(f"最優產品: 觸發={bayes_result.optimal_product.trigger_threshold:.1f}, "
          f"賠付={bayes_result.optimal_product.payout_amount:.2e}")
    
    # 測試Γ-minimax優化
    print(f"\n🛡️ 測試Γ-極小化極大優化...")
    minimax_result = optimizer.gamma_minimax_optimization(
        posterior_samples, hazard_indices, actual_losses, product_space
    )
    
    print(f"Minimax產品: 觸發={minimax_result.minimax_product.trigger_threshold:.1f}, "
          f"賠付={minimax_result.minimax_product.payout_amount:.2e}")
    
    # 比較決策策略
    print(f"\n📊 比較決策策略...")
    comparison = optimizer.compare_decision_strategies(
        posterior_samples, hazard_indices, actual_losses, product_space
    )
    
    print(f"最壞情況改善: {comparison['comparison_metrics']['worst_case_improvement']:.1%}")
    print(f"平均風險權衡: {comparison['comparison_metrics']['average_risk_trade_off']:.1%}")
    
    # 優化摘要
    print(f"\n📋 優化摘要:")
    summary_table = optimizer.get_optimization_summary()
    if not summary_table.empty:
        print(summary_table[['方法', '觸發閾值', '期望風險', '執行時間']])
    
    print(f"\n✅ 貝氏決策理論測試完成")
    return bayes_result, minimax_result, comparison

if __name__ == "__main__":
    test_bayesian_decision_theory()