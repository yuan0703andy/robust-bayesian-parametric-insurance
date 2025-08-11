#!/usr/bin/env python3
"""
Multi-Objective Optimizer Module
多目標優化器模組

專門負責多目標優化和Pareto前緣分析，包含:
- 多目標產品評估
- Pareto效率前緣計算
- 決策偏好分析
- 優化結果排序

Author: Research Team
Date: 2025-01-10
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from scipy.optimize import differential_evolution

# 導入相關模組
from .parametric_engine import ParametricProduct
from .technical_premium_calculator import TechnicalPremiumCalculator, TechnicalPremiumResult
from .market_acceptability_analyzer import MarketAcceptabilityAnalyzer, MarketAcceptabilityResult
from skill_scores.basis_risk_functions import BasisRiskCalculator, BasisRiskType


class OptimizationObjective(Enum):
    """優化目標"""
    MINIMIZE_TECHNICAL_PREMIUM = "minimize_technical_premium"
    MINIMIZE_BASIS_RISK = "minimize_basis_risk"
    MAXIMIZE_MARKET_ACCEPTABILITY = "maximize_market_acceptability"
    MINIMIZE_LOSS_RATIO = "minimize_loss_ratio"
    MAXIMIZE_TRIGGER_RATE = "maximize_trigger_rate"
    MINIMIZE_COMPLEXITY = "minimize_complexity"


class DecisionPreferenceType(Enum):
    """決策偏好類型"""
    RISK_AVERSE = "risk_averse"              # 風險厭惡型
    COST_SENSITIVE = "cost_sensitive"        # 成本敏感型
    MARKET_ORIENTED = "market_oriented"      # 市場導向型
    BALANCED = "balanced"                    # 平衡型


@dataclass
class OptimizationConfig:
    """優化配置"""
    objectives: List[OptimizationObjective] = field(default_factory=lambda: [
        OptimizationObjective.MINIMIZE_TECHNICAL_PREMIUM,
        OptimizationObjective.MINIMIZE_BASIS_RISK,
        OptimizationObjective.MAXIMIZE_MARKET_ACCEPTABILITY
    ])
    objective_weights: Dict[OptimizationObjective, float] = field(default_factory=dict)
    n_candidates: int = 100                  # 候選解數量
    enable_pareto_analysis: bool = True      # 啟用Pareto分析
    enable_preference_ranking: bool = True   # 啟用偏好排序
    random_seed: int = 42                   # 隨機種子


@dataclass
class ProductEvaluation:
    """產品評估結果"""
    product_id: str
    product: ParametricProduct
    technical_premium_result: TechnicalPremiumResult
    market_acceptability_result: MarketAcceptabilityResult
    basis_risk: float
    trigger_rate: float
    # 目標函數值
    objective_values: Dict[OptimizationObjective, float] = field(default_factory=dict)
    # 標準化評分
    normalized_scores: Dict[OptimizationObjective, float] = field(default_factory=dict)


@dataclass
class ParetoSolution:
    """Pareto解"""
    solution_id: str
    product_evaluation: ProductEvaluation
    pareto_rank: int                        # Pareto排序
    crowding_distance: float               # 擁擠距離
    preference_scores: Dict[DecisionPreferenceType, float] = field(default_factory=dict)


@dataclass 
class MultiObjectiveResult:
    """多目標優化結果"""
    all_evaluations: List[ProductEvaluation]
    pareto_front_indices: List[int]
    pareto_solutions: List[ParetoSolution]
    preference_rankings: Dict[DecisionPreferenceType, List[ParetoSolution]]
    optimization_summary: Dict[str, Any]


class ObjectiveFunction(ABC):
    """目標函數抽象基類"""
    
    @abstractmethod
    def evaluate(self, evaluation: ProductEvaluation) -> float:
        """計算目標函數值"""
        pass
    
    @abstractmethod
    def is_minimization(self) -> bool:
        """是否為最小化問題"""
        pass


class TechnicalPremiumObjective(ObjectiveFunction):
    """技術保費目標函數"""
    
    def evaluate(self, evaluation: ProductEvaluation) -> float:
        return evaluation.technical_premium_result.technical_premium
    
    def is_minimization(self) -> bool:
        return True


class BasisRiskObjective(ObjectiveFunction):
    """基差風險目標函數"""
    
    def evaluate(self, evaluation: ProductEvaluation) -> float:
        return evaluation.basis_risk
    
    def is_minimization(self) -> bool:
        return True


class MarketAcceptabilityObjective(ObjectiveFunction):
    """市場接受度目標函數"""
    
    def evaluate(self, evaluation: ProductEvaluation) -> float:
        return evaluation.market_acceptability_result.overall_acceptability
    
    def is_minimization(self) -> bool:
        return False


class ProductEvaluator:
    """產品評估器"""
    
    def __init__(self,
                 premium_calculator: TechnicalPremiumCalculator,
                 market_analyzer: MarketAcceptabilityAnalyzer,
                 basis_risk_calculator: Optional[BasisRiskCalculator] = None):
        """
        初始化產品評估器
        
        Parameters:
        -----------
        premium_calculator : TechnicalPremiumCalculator
            技術保費計算器
        market_analyzer : MarketAcceptabilityAnalyzer
            市場接受度分析器
        basis_risk_calculator : BasisRiskCalculator, optional
            基差風險計算器
        """
        self.premium_calculator = premium_calculator
        self.market_analyzer = market_analyzer
        self.basis_risk_calculator = basis_risk_calculator or BasisRiskCalculator()
        
        # 目標函數映射
        self.objective_functions = {
            OptimizationObjective.MINIMIZE_TECHNICAL_PREMIUM: TechnicalPremiumObjective(),
            OptimizationObjective.MINIMIZE_BASIS_RISK: BasisRiskObjective(),
            OptimizationObjective.MAXIMIZE_MARKET_ACCEPTABILITY: MarketAcceptabilityObjective()
        }
    
    def evaluate_product(self,
                        product: ParametricProduct,
                        actual_losses: np.ndarray,
                        hazard_indices: np.ndarray) -> ProductEvaluation:
        """
        全面評估產品性能
        
        Parameters:
        -----------
        product : ParametricProduct
            保險產品
        actual_losses : np.ndarray
            實際損失數據
        hazard_indices : np.ndarray
            災害指標數據
            
        Returns:
        --------
        ProductEvaluation
            產品評估結果
        """
        # 1. 技術保費計算
        premium_result = self.premium_calculator.calculate_technical_premium(
            product, hazard_indices
        )
        
        # 2. 基差風險計算
        payouts = self._calculate_payouts(product, hazard_indices)
        basis_risks = []
        
        for loss, payout in zip(actual_losses, payouts):
            risk = self.basis_risk_calculator.calculate_weighted_asymmetric_basis_risk(
                loss, payout, w_under=2.0, w_over=0.5
            )
            basis_risks.append(risk)
        
        mean_basis_risk = np.mean(basis_risks)
        trigger_rate = np.mean(payouts > 0)
        
        # 3. 市場接受度分析
        market_result = self.market_analyzer.analyze_market_acceptability(
            product, premium_result.technical_premium, 
            premium_result.expected_payout, trigger_rate
        )
        
        # 4. 計算目標函數值
        evaluation = ProductEvaluation(
            product_id=product.product_id,
            product=product,
            technical_premium_result=premium_result,
            market_acceptability_result=market_result,
            basis_risk=mean_basis_risk,
            trigger_rate=trigger_rate
        )
        
        # 計算目標函數值
        for objective, func in self.objective_functions.items():
            evaluation.objective_values[objective] = func.evaluate(evaluation)
        
        return evaluation
    
    def _calculate_payouts(self, product: ParametricProduct, hazard_indices: np.ndarray) -> np.ndarray:
        """計算產品賠付"""
        payouts = np.zeros(len(hazard_indices))
        
        for i, index_value in enumerate(hazard_indices):
            # 階梯式賠付邏輯 - 支持多閾值
            for j, threshold in enumerate(product.trigger_thresholds):
                if index_value >= threshold:
                    payouts[i] = product.payout_amounts[j]
        
        # 應用賠付限額
        payouts = np.minimum(payouts, product.max_payout)
        
        return payouts


class ParetoFrontAnalyzer:
    """Pareto前緣分析器"""
    
    def __init__(self, objective_functions: Dict[OptimizationObjective, ObjectiveFunction]):
        """
        初始化Pareto前緣分析器
        
        Parameters:
        -----------
        objective_functions : Dict[OptimizationObjective, ObjectiveFunction]
            目標函數字典
        """
        self.objective_functions = objective_functions
    
    def find_pareto_front(self,
                         evaluations: List[ProductEvaluation],
                         objectives: List[OptimizationObjective]) -> List[int]:
        """
        尋找Pareto效率前緣
        
        Parameters:
        -----------
        evaluations : List[ProductEvaluation]
            產品評估結果
        objectives : List[OptimizationObjective]
            優化目標
            
        Returns:
        --------
        List[int]
            Pareto前緣解的索引
        """
        n_solutions = len(evaluations)
        pareto_front = []
        
        for i in range(n_solutions):
            is_dominated = False
            
            for j in range(n_solutions):
                if i == j:
                    continue
                
                # 檢查解i是否被解j支配
                if self._dominates(evaluations[j], evaluations[i], objectives):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(i)
        
        return pareto_front
    
    def _dominates(self,
                  solution_a: ProductEvaluation,
                  solution_b: ProductEvaluation,
                  objectives: List[OptimizationObjective]) -> bool:
        """
        檢查解A是否支配解B
        
        Parameters:
        -----------
        solution_a : ProductEvaluation
            解A
        solution_b : ProductEvaluation
            解B
        objectives : List[OptimizationObjective]
            優化目標
            
        Returns:
        --------
        bool
            是否支配
        """
        better_in_any = False
        
        for objective in objectives:
            value_a = solution_a.objective_values[objective]
            value_b = solution_b.objective_values[objective]
            
            is_minimization = self.objective_functions[objective].is_minimization()
            
            if is_minimization:
                if value_a > value_b:  # A在這個目標上更差
                    return False
                elif value_a < value_b:  # A在這個目標上更好
                    better_in_any = True
            else:  # 最大化問題
                if value_a < value_b:  # A在這個目標上更差
                    return False
                elif value_a > value_b:  # A在這個目標上更好
                    better_in_any = True
        
        return better_in_any
    
    def calculate_crowding_distance(self,
                                  pareto_solutions: List[ProductEvaluation],
                                  objectives: List[OptimizationObjective]) -> List[float]:
        """
        計算擁擠距離
        
        Parameters:
        -----------
        pareto_solutions : List[ProductEvaluation]
            Pareto解集
        objectives : List[OptimizationObjective]
            優化目標
            
        Returns:
        --------
        List[float]
            擁擠距離
        """
        n_solutions = len(pareto_solutions)
        if n_solutions <= 2:
            return [float('inf')] * n_solutions
        
        distances = [0.0] * n_solutions
        
        for objective in objectives:
            # 按該目標排序
            values = [sol.objective_values[objective] for sol in pareto_solutions]
            sorted_indices = sorted(range(n_solutions), key=lambda i: values[i])
            
            # 邊界解設為無窮大
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # 計算範圍
            obj_range = values[sorted_indices[-1]] - values[sorted_indices[0]]
            if obj_range == 0:
                continue
            
            # 計算中間解的擁擠距離
            for i in range(1, n_solutions - 1):
                idx = sorted_indices[i]
                idx_next = sorted_indices[i + 1]
                idx_prev = sorted_indices[i - 1]
                
                distances[idx] += (values[idx_next] - values[idx_prev]) / obj_range
        
        return distances


class PreferenceBasedRanker:
    """基於偏好的排序器"""
    
    def __init__(self):
        """初始化偏好排序器"""
        pass
    
    def rank_by_preference(self,
                          pareto_solutions: List[ParetoSolution],
                          preference_type: DecisionPreferenceType) -> List[ParetoSolution]:
        """
        根據偏好類型排序Pareto解
        
        Parameters:
        -----------
        pareto_solutions : List[ParetoSolution]
            Pareto解集
        preference_type : DecisionPreferenceType
            偏好類型
            
        Returns:
        --------
        List[ParetoSolution]
            排序後的解集
        """
        if preference_type == DecisionPreferenceType.RISK_AVERSE:
            return sorted(pareto_solutions, 
                         key=lambda x: x.product_evaluation.basis_risk)
        
        elif preference_type == DecisionPreferenceType.COST_SENSITIVE:
            return sorted(pareto_solutions, 
                         key=lambda x: x.product_evaluation.technical_premium_result.technical_premium)
        
        elif preference_type == DecisionPreferenceType.MARKET_ORIENTED:
            return sorted(pareto_solutions, 
                         key=lambda x: x.product_evaluation.market_acceptability_result.overall_acceptability, 
                         reverse=True)
        
        elif preference_type == DecisionPreferenceType.BALANCED:
            # 計算平衡評分
            for solution in pareto_solutions:
                solution.preference_scores[preference_type] = self._calculate_balanced_score(solution, pareto_solutions)
            
            return sorted(pareto_solutions, 
                         key=lambda x: x.preference_scores[preference_type], 
                         reverse=True)
        
        return pareto_solutions
    
    def _calculate_balanced_score(self, 
                                 solution: ParetoSolution, 
                                 all_solutions: List[ParetoSolution]) -> float:
        """計算平衡評分"""
        # 提取所有解的目標值
        premiums = [s.product_evaluation.technical_premium_result.technical_premium for s in all_solutions]
        risks = [s.product_evaluation.basis_risk for s in all_solutions]
        acceptabilities = [s.product_evaluation.market_acceptability_result.overall_acceptability for s in all_solutions]
        
        # 標準化 (0-1)
        def normalize(value, values, reverse=False):
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return 1.0
            norm = (value - min_val) / (max_val - min_val)
            return 1 - norm if not reverse else norm
        
        # 計算標準化評分
        norm_premium = normalize(solution.product_evaluation.technical_premium_result.technical_premium, premiums)
        norm_risk = normalize(solution.product_evaluation.basis_risk, risks)
        norm_accept = normalize(solution.product_evaluation.market_acceptability_result.overall_acceptability, acceptabilities, reverse=True)
        
        # 平衡評分 (等權重)
        return (norm_premium + norm_risk + norm_accept) / 3


class MultiObjectiveOptimizer:
    """多目標優化器"""
    
    def __init__(self,
                 premium_calculator: TechnicalPremiumCalculator,
                 market_analyzer: MarketAcceptabilityAnalyzer,
                 basis_risk_calculator: Optional[BasisRiskCalculator] = None):
        """
        初始化多目標優化器
        
        Parameters:
        -----------
        premium_calculator : TechnicalPremiumCalculator
            技術保費計算器
        market_analyzer : MarketAcceptabilityAnalyzer
            市場接受度分析器
        basis_risk_calculator : BasisRiskCalculator, optional
            基差風險計算器
        """
        self.product_evaluator = ProductEvaluator(premium_calculator, market_analyzer, basis_risk_calculator)
        self.pareto_analyzer = ParetoFrontAnalyzer(self.product_evaluator.objective_functions)
        self.preference_ranker = PreferenceBasedRanker()
    
    def optimize(self,
                 candidate_products: List[ParametricProduct],
                 actual_losses: np.ndarray,
                 hazard_indices: np.ndarray,
                 config: OptimizationConfig) -> MultiObjectiveResult:
        """
        執行多目標優化
        
        Parameters:
        -----------
        candidate_products : List[ParametricProduct]
            候選產品列表
        actual_losses : np.ndarray
            實際損失數據
        hazard_indices : np.ndarray
            災害指標數據
        config : OptimizationConfig
            優化配置
            
        Returns:
        --------
        MultiObjectiveResult
            優化結果
        """
        print(f"🎯 執行多目標優化 ({len(candidate_products)} 個候選產品)...")
        
        # 1. 評估所有候選產品
        all_evaluations = []
        for i, product in enumerate(candidate_products):
            if (i + 1) % 20 == 0:
                print(f"  評估進度: {i+1}/{len(candidate_products)}")
            
            evaluation = self.product_evaluator.evaluate_product(
                product, actual_losses, hazard_indices
            )
            all_evaluations.append(evaluation)
        
        # 2. Pareto前緣分析
        pareto_front_indices = []
        pareto_solutions = []
        
        if config.enable_pareto_analysis:
            pareto_front_indices = self.pareto_analyzer.find_pareto_front(
                all_evaluations, config.objectives
            )
            
            # 創建Pareto解
            pareto_evaluations = [all_evaluations[i] for i in pareto_front_indices]
            crowding_distances = self.pareto_analyzer.calculate_crowding_distance(
                pareto_evaluations, config.objectives
            )
            
            for i, (eval_idx, eval_result) in enumerate(zip(pareto_front_indices, pareto_evaluations)):
                solution = ParetoSolution(
                    solution_id=f"Pareto_{i+1}",
                    product_evaluation=eval_result,
                    pareto_rank=1,  # 所有Pareto解的rank都是1
                    crowding_distance=crowding_distances[i]
                )
                pareto_solutions.append(solution)
        
        # 3. 偏好排序
        preference_rankings = {}
        if config.enable_preference_ranking and pareto_solutions:
            for preference_type in DecisionPreferenceType:
                ranked_solutions = self.preference_ranker.rank_by_preference(
                    pareto_solutions.copy(), preference_type
                )
                preference_rankings[preference_type] = ranked_solutions
        
        # 4. 生成優化摘要
        optimization_summary = {
            'total_candidates': len(candidate_products),
            'pareto_efficient_solutions': len(pareto_front_indices),
            'pareto_efficiency_rate': len(pareto_front_indices) / len(candidate_products) if candidate_products else 0,
            'objectives': [obj.value for obj in config.objectives],
            'best_solutions_by_preference': {}
        }
        
        # 添加最佳解
        for pref_type, ranked_solutions in preference_rankings.items():
            if ranked_solutions:
                optimization_summary['best_solutions_by_preference'][pref_type.value] = {
                    'product_id': ranked_solutions[0].product_evaluation.product_id,
                    'technical_premium': ranked_solutions[0].product_evaluation.technical_premium_result.technical_premium,
                    'basis_risk': ranked_solutions[0].product_evaluation.basis_risk,
                    'market_acceptability': ranked_solutions[0].product_evaluation.market_acceptability_result.overall_acceptability
                }
        
        print(f"✅ 多目標優化完成，找到 {len(pareto_front_indices)} 個Pareto效率解")
        
        return MultiObjectiveResult(
            all_evaluations=all_evaluations,
            pareto_front_indices=pareto_front_indices,
            pareto_solutions=pareto_solutions,
            preference_rankings=preference_rankings,
            optimization_summary=optimization_summary
        )


def create_standard_multi_objective_optimizer(
    premium_calculator: TechnicalPremiumCalculator,
    market_analyzer: MarketAcceptabilityAnalyzer) -> MultiObjectiveOptimizer:
    """
    創建標準多目標優化器
    
    Parameters:
    -----------
    premium_calculator : TechnicalPremiumCalculator
        技術保費計算器
    market_analyzer : MarketAcceptabilityAnalyzer
        市場接受度分析器
        
    Returns:
    --------
    MultiObjectiveOptimizer
        標準多目標優化器
    """
    return MultiObjectiveOptimizer(premium_calculator, market_analyzer)