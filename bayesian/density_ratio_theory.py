"""
Density Ratio Framework
密度比框架

Implements the core density ratio class framework from robust Bayesian theory:
Γ = {P : dP/dP₀ ≤ γ(x)}

This module provides the mathematical foundation for robust Bayesian analysis,
focusing specifically on density ratio constraints and model selection.

Key Components:
1. Density Ratio Class Implementation
2. Multiple Prior Scenario Testing
3. Multiple Likelihood Function Comparison
4. Automatic Model Selection via AIC/BIC
5. Robustness Evaluation

Note: This is the theoretical foundation - use RobustBayesianAnalyzer for practical analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.optimize import minimize
import logging

class ModelSelectionCriterion(Enum):
    """模型選擇標準"""
    AIC = "aic"
    BIC = "bic"
    WAIC = "waic"
    LOO_CV = "loo_cv"

@dataclass
class ModelConfiguration:
    """模型配置"""
    name: str
    prior_params: Dict[str, Any]
    likelihood_family: str
    density_ratio_constraint: float = 2.0

@dataclass
class ModelComparisonResult:
    """模型比較結果"""
    model_name: str
    log_likelihood: float
    aic: float
    bic: float
    waic: Optional[float] = None
    loo_cv: Optional[float] = None
    posterior_samples: Optional[np.ndarray] = None
    density_ratio_violations: int = 0
    
    @property
    def selection_score(self) -> float:
        """返回主要選擇分數 (AIC)"""
        return self.aic

class DensityRatioClass:
    """
    密度比類別實現
    
    Implementation of Γ = {P : dP/dP₀ ≤ γ(x)} where:
    - P₀ is the reference prior distribution
    - γ(x) is the density ratio constraint function
    - Γ is the class of admissible priors
    """
    
    def __init__(self, 
                 gamma_constraint: float = 2.0,
                 reference_prior: Optional[Callable] = None,
                 constraint_function: Optional[Callable] = None):
        """
        初始化密度比類別
        
        Parameters:
        -----------
        gamma_constraint : float
            密度比上界 γ
        reference_prior : Callable, optional
            參考先驗分布 P₀
        constraint_function : Callable, optional
            約束函數 γ(x)
        """
        self.gamma_constraint = gamma_constraint
        self.reference_prior = reference_prior or self._default_reference_prior
        self.constraint_function = constraint_function or self._default_constraint_function
        
        # 記錄違反約束的次數
        self.constraint_violations = 0
        
    def _default_reference_prior(self, x: np.ndarray) -> np.ndarray:
        """預設參考先驗 (標準正態)"""
        return stats.norm.pdf(x, loc=0, scale=1)
        
    def _default_constraint_function(self, x: np.ndarray) -> np.ndarray:
        """預設約束函數 (常數)"""
        return np.full_like(x, self.gamma_constraint)
    
    def evaluate_density_ratio(self, 
                             candidate_prior: Callable,
                             evaluation_points: np.ndarray) -> np.ndarray:
        """
        評估密度比 dP/dP₀
        
        Parameters:
        -----------
        candidate_prior : Callable
            候選先驗分布 P
        evaluation_points : np.ndarray
            評估點
            
        Returns:
        --------
        np.ndarray
            密度比值
        """
        p_values = candidate_prior(evaluation_points)
        p0_values = self.reference_prior(evaluation_points)
        
        # 避免除以零
        p0_values = np.maximum(p0_values, 1e-10)
        density_ratios = p_values / p0_values
        
        return density_ratios
    
    def check_constraint_satisfaction(self,
                                    candidate_prior: Callable,
                                    evaluation_points: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        檢查密度比約束是否滿足
        
        Returns:
        --------
        Tuple[bool, np.ndarray]
            (是否滿足約束, 違反點的索引)
        """
        density_ratios = self.evaluate_density_ratio(candidate_prior, evaluation_points)
        constraint_values = self.constraint_function(evaluation_points)
        
        violations = density_ratios > constraint_values
        violation_indices = np.where(violations)[0]
        
        self.constraint_violations = len(violation_indices)
        
        is_satisfied = len(violation_indices) == 0
        
        return is_satisfied, violation_indices
    
    def generate_constrained_prior_ensemble(self,
                                          base_parameters: Dict[str, Any],
                                          n_priors: int = 10,
                                          perturbation_scale: float = 0.1) -> List[Callable]:
        """
        生成滿足密度比約束的先驗集合
        
        Parameters:
        -----------
        base_parameters : Dict[str, Any]
            基礎參數
        n_priors : int
            生成的先驗數量
        perturbation_scale : float
            擾動規模
            
        Returns:
        --------
        List[Callable]
            滿足約束的先驗分布列表
        """
        valid_priors = []
        attempts = 0
        max_attempts = n_priors * 10
        
        evaluation_grid = np.linspace(-5, 5, 100)
        
        while len(valid_priors) < n_priors and attempts < max_attempts:
            attempts += 1
            
            # 擾動參數
            perturbed_params = {}
            for key, value in base_parameters.items():
                if isinstance(value, (int, float)):
                    noise = np.random.normal(0, perturbation_scale * abs(value))
                    perturbed_params[key] = value + noise
                else:
                    perturbed_params[key] = value
            
            # 創建候選先驗
            def candidate_prior(x, params=perturbed_params):
                return stats.norm.pdf(x, 
                                    loc=params.get('loc', 0),
                                    scale=params.get('scale', 1))
            
            # 檢查約束
            is_valid, _ = self.check_constraint_satisfaction(candidate_prior, evaluation_grid)
            
            if is_valid:
                valid_priors.append(candidate_prior)
        
        if len(valid_priors) < n_priors:
            warnings.warn(f"只生成了 {len(valid_priors)} 個滿足約束的先驗，目標是 {n_priors} 個")
        
        return valid_priors

class RobustBayesianFramework:
    """
    穩健貝氏框架主類別
    
    Implements comprehensive robust Bayesian analysis with:
    1. Multiple prior scenario testing
    2. Density ratio constraint checking
    3. Model comparison and selection
    4. Robustness evaluation
    """
    
    def __init__(self,
                 density_ratio_constraint: float = 2.0,
                 model_selection_criterion: ModelSelectionCriterion = ModelSelectionCriterion.AIC):
        """
        初始化穩健貝氏框架
        
        Parameters:
        -----------
        density_ratio_constraint : float
            密度比約束上界
        model_selection_criterion : ModelSelectionCriterion
            模型選擇標準
        """
        self.density_ratio_class = DensityRatioClass(gamma_constraint=density_ratio_constraint)
        self.selection_criterion = model_selection_criterion
        
        # 模型配置庫
        self.model_configurations = self._initialize_model_configurations()
        
        # 分析結果
        self.comparison_results: List[ModelComparisonResult] = []
        self.best_model: Optional[ModelComparisonResult] = None
        
    def _initialize_model_configurations(self) -> List[ModelConfiguration]:
        """初始化模型配置庫"""
        configurations = [
            ModelConfiguration(
                name="normal_informative",
                prior_params={"loc": 0, "scale": 0.5},
                likelihood_family="normal",
                density_ratio_constraint=self.density_ratio_class.gamma_constraint
            ),
            ModelConfiguration(
                name="normal_weakly_informative", 
                prior_params={"loc": 0, "scale": 1.0},
                likelihood_family="normal",
                density_ratio_constraint=self.density_ratio_class.gamma_constraint
            ),
            ModelConfiguration(
                name="normal_vague",
                prior_params={"loc": 0, "scale": 2.0},
                likelihood_family="normal",
                density_ratio_constraint=self.density_ratio_class.gamma_constraint
            ),
            ModelConfiguration(
                name="student_t_robust",
                prior_params={"df": 3, "loc": 0, "scale": 1.0},
                likelihood_family="t",
                density_ratio_constraint=self.density_ratio_class.gamma_constraint
            ),
            ModelConfiguration(
                name="laplace_sparse",
                prior_params={"loc": 0, "scale": 1.0},
                likelihood_family="laplace",
                density_ratio_constraint=self.density_ratio_class.gamma_constraint
            ),
            ModelConfiguration(
                name="gamma_positive_constraint",
                prior_params={"a": 2, "scale": 1},
                likelihood_family="gamma",
                density_ratio_constraint=self.density_ratio_class.gamma_constraint
            )
        ]
        
        return configurations
    
    def add_custom_model_configuration(self, config: ModelConfiguration):
        """添加自定義模型配置"""
        self.model_configurations.append(config)
    
    def fit_single_model(self, 
                        data: np.ndarray,
                        config: ModelConfiguration) -> ModelComparisonResult:
        """
        擬合單一模型並評估
        
        Parameters:
        -----------
        data : np.ndarray
            觀測資料
        config : ModelConfiguration
            模型配置
            
        Returns:
        --------
        ModelComparisonResult
            模型比較結果
        """
        try:
            # 計算對數似然
            log_likelihood = self._calculate_log_likelihood(data, config)
            
            # 計算資訊標準
            n_params = len(config.prior_params)
            n_data = len(data)
            
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + n_params * np.log(n_data)
            
            # 檢查密度比約束違反
            evaluation_points = np.linspace(np.min(data), np.max(data), 100)
            prior_func = self._create_prior_function(config)
            _, violation_indices = self.density_ratio_class.check_constraint_satisfaction(
                prior_func, evaluation_points
            )
            
            result = ModelComparisonResult(
                model_name=config.name,
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic,
                density_ratio_violations=len(violation_indices)
            )
            
            return result
            
        except Exception as e:
            warnings.warn(f"模型 {config.name} 擬合失敗: {e}")
            return ModelComparisonResult(
                model_name=config.name,
                log_likelihood=-np.inf,
                aic=np.inf,
                bic=np.inf,
                density_ratio_violations=np.inf
            )
    
    def _calculate_log_likelihood(self, 
                                data: np.ndarray, 
                                config: ModelConfiguration) -> float:
        """計算對數似然"""
        if config.likelihood_family == "normal":
            # 最大似然估計
            mu_hat = np.mean(data)
            sigma_hat = np.std(data, ddof=1)
            return np.sum(stats.norm.logpdf(data, loc=mu_hat, scale=sigma_hat))
            
        elif config.likelihood_family == "t":
            # Student-t 分布
            df = config.prior_params.get("df", 3)
            params = stats.t.fit(data, fdf=df)
            return np.sum(stats.t.logpdf(data, *params))
            
        elif config.likelihood_family == "laplace":
            # Laplace 分布
            params = stats.laplace.fit(data)
            return np.sum(stats.laplace.logpdf(data, *params))
            
        elif config.likelihood_family == "gamma":
            # Gamma 分布 (僅限正值資料)
            if np.any(data <= 0):
                return -np.inf
            params = stats.gamma.fit(data)
            return np.sum(stats.gamma.logpdf(data, *params))
            
        else:
            raise ValueError(f"未支援的似然家族: {config.likelihood_family}")
    
    def _create_prior_function(self, config: ModelConfiguration) -> Callable:
        """根據配置創建先驗函數"""
        if config.likelihood_family == "normal":
            loc = config.prior_params.get("loc", 0)
            scale = config.prior_params.get("scale", 1)
            return lambda x: stats.norm.pdf(x, loc=loc, scale=scale)
            
        elif config.likelihood_family == "t":
            df = config.prior_params.get("df", 3)
            loc = config.prior_params.get("loc", 0)
            scale = config.prior_params.get("scale", 1)
            return lambda x: stats.t.pdf(x, df=df, loc=loc, scale=scale)
            
        elif config.likelihood_family == "laplace":
            loc = config.prior_params.get("loc", 0)
            scale = config.prior_params.get("scale", 1)
            return lambda x: stats.laplace.pdf(x, loc=loc, scale=scale)
            
        elif config.likelihood_family == "gamma":
            a = config.prior_params.get("a", 2)
            scale = config.prior_params.get("scale", 1)
            return lambda x: stats.gamma.pdf(x, a=a, scale=scale)
            
        else:
            raise ValueError(f"未支援的似然家族: {config.likelihood_family}")
    
    def compare_all_models(self, data: np.ndarray) -> List[ModelComparisonResult]:
        """
        比較所有模型配置
        
        Parameters:
        -----------
        data : np.ndarray
            觀測資料
            
        Returns:
        --------
        List[ModelComparisonResult]
            所有模型的比較結果
        """
        print(f"🔄 開始穩健貝氏模型比較，共 {len(self.model_configurations)} 個模型...")
        
        self.comparison_results = []
        
        for config in self.model_configurations:
            print(f"  📊 擬合模型: {config.name}")
            result = self.fit_single_model(data, config)
            self.comparison_results.append(result)
        
        # 根據選擇標準排序
        if self.selection_criterion == ModelSelectionCriterion.AIC:
            self.comparison_results.sort(key=lambda x: x.aic)
        elif self.selection_criterion == ModelSelectionCriterion.BIC:
            self.comparison_results.sort(key=lambda x: x.bic)
        
        self.best_model = self.comparison_results[0] if self.comparison_results else None
        
        print(f"✅ 模型比較完成，最佳模型: {self.best_model.model_name if self.best_model else 'None'}")
        
        return self.comparison_results
    
    def evaluate_robustness(self, data: np.ndarray) -> Dict[str, Any]:
        """
        評估模型的穩健性
        
        Returns:
        --------
        Dict[str, Any]
            穩健性評估結果
        """
        if not self.comparison_results:
            self.compare_all_models(data)
        
        # 計算模型權重 (基於 AIC 權重)
        aic_values = np.array([r.aic for r in self.comparison_results])
        aic_min = np.min(aic_values)
        delta_aic = aic_values - aic_min
        weights = np.exp(-0.5 * delta_aic)
        weights = weights / np.sum(weights)
        
        # 評估密度比約束違反程度
        violation_counts = [r.density_ratio_violations for r in self.comparison_results]
        total_violations = sum(violation_counts)
        
        # 計算模型不確定性
        top_models = [r for r in self.comparison_results if r.aic - aic_min < 2]
        model_uncertainty = len(top_models) / len(self.comparison_results)
        
        robustness_results = {
            "best_model": self.best_model.model_name if self.best_model else None,
            "model_weights": {
                r.model_name: w for r, w in zip(self.comparison_results, weights)
            },
            "total_density_ratio_violations": total_violations,
            "model_uncertainty_ratio": model_uncertainty,
            "top_models": [r.model_name for r in top_models],
            "worst_aic": np.max(aic_values) if len(aic_values) > 0 else np.inf,
            "best_aic": aic_min if len(aic_values) > 0 else np.inf,
            "aic_range": np.max(aic_values) - aic_min if len(aic_values) > 0 else 0
        }
        
        return robustness_results
    
    def get_model_comparison_summary(self) -> pd.DataFrame:
        """獲取模型比較摘要表"""
        if not self.comparison_results:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.comparison_results:
            summary_data.append({
                "Model": result.model_name,
                "Log_Likelihood": result.log_likelihood,
                "AIC": result.aic,
                "BIC": result.bic,
                "Density_Ratio_Violations": result.density_ratio_violations
            })
        
        return pd.DataFrame(summary_data)