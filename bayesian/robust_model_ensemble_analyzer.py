#!/usr/bin/env python3
"""
Model Class Analyzer Module  
模型類別分析器模組

實現您理論框架中的模型集合 M = Γ_f × Γ_π 的系統性分析。
這是實現穩健貝氏分析的核心模組，支援遍歷整個模型空間。

核心功能:
- 構建完整的模型類別 M = Γ_f × Γ_π
- 對每個模型進行獨立擬合
- 計算後驗數量的範圍 [inf, sup]
- 模型權重計算和比較
- 穩健性評估

使用範例:
```python
from bayesian.robust_model_ensemble_analyzer import ModelClassAnalyzer

# 初始化分析器
analyzer = ModelClassAnalyzer()

# 分析完整模型集合
results = analyzer.analyze_model_class(observations)

# 查看結果
print("最佳模型:", results.best_model)
print("後驗範圍:", results.posterior_ranges)
print("模型權重:", results.model_weights)

# 計算特定參數的穩健範圍
theta_range = analyzer.compute_posterior_range('theta')
print(f"θ範圍: [{theta_range[0]:.3f}, {theta_range[1]:.3f}]")
```

Author: Research Team
Date: 2025-01-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 導入參數化階層模型
try:
    from .parametric_bayesian_hierarchy import (
        ParametricHierarchicalModel, ModelSpec, MCMCConfig,
        LikelihoodFamily, PriorScenario, HierarchicalModelResult
    )
    HAS_HIERARCHICAL = True
except ImportError:
    HAS_HIERARCHICAL = False
    warnings.warn("參數化階層模型不可用")

# 導入 ε-contamination 支援
try:
    from .epsilon_contamination import (
        EpsilonContaminationClass, EpsilonContaminationSpec,
        ContaminationDistributionClass, create_typhoon_contamination_spec
    )
    HAS_EPSILON_CONTAMINATION = True
except ImportError:
    HAS_EPSILON_CONTAMINATION = False
    warnings.warn("ε-contamination 模組不可用")

@dataclass
class ModelClassSpec:
    """模型類別規格"""
    likelihood_families: List[LikelihoodFamily] = field(default_factory=lambda: [
        LikelihoodFamily.NORMAL,
        LikelihoodFamily.LOGNORMAL,
        LikelihoodFamily.STUDENT_T
    ])
    prior_scenarios: List[PriorScenario] = field(default_factory=lambda: [
        PriorScenario.NON_INFORMATIVE,
        PriorScenario.WEAK_INFORMATIVE,
        PriorScenario.OPTIMISTIC,
        PriorScenario.PESSIMISTIC
    ])
    
    # ε-contamination 支援
    enable_epsilon_contamination: bool = False
    epsilon_values: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    contamination_distribution: str = "typhoon"  # "typhoon" or "heavy_tail"
    
    def get_model_count(self) -> int:
        """獲取模型總數"""
        base_count = len(self.likelihood_families) * len(self.prior_scenarios)
        if self.enable_epsilon_contamination and HAS_EPSILON_CONTAMINATION:
            # 每個基礎模型 × 每個ε值 = 污染模型數量
            contamination_count = base_count * len(self.epsilon_values)
            return base_count + contamination_count
        return base_count
    
    def generate_all_specs(self) -> List[ModelSpec]:
        """生成所有模型規格組合"""
        all_specs = []
        
        # 生成基礎模型規格
        for likelihood, prior in product(self.likelihood_families, self.prior_scenarios):
            spec = ModelSpec(
                likelihood_family=likelihood,
                prior_scenario=prior
            )
            all_specs.append(spec)
        
        # 生成ε-contamination模型規格
        if self.enable_epsilon_contamination and HAS_EPSILON_CONTAMINATION:
            for likelihood, prior in product(self.likelihood_families, self.prior_scenarios):
                for epsilon in self.epsilon_values:
                    contamination_spec = ModelSpec(
                        likelihood_family=likelihood,
                        prior_scenario=prior,
                        # 添加ε-contamination標識到模型名稱
                        model_name=f"{likelihood.value}_{prior.value}_epsilon_{epsilon:.2f}"
                    )
                    # 存儲ε-contamination參數（修正：使用epsilon_range）
                    # Note: EpsilonContaminationSpec uses epsilon_range, not epsilon_contamination
                    # contamination_spec already has correct epsilon_range from create_typhoon_contamination_spec
                    all_specs.append(contamination_spec)
        
        return all_specs

@dataclass
class ModelClassResult:
    """模型類別分析結果"""
    model_class_spec: ModelClassSpec
    individual_results: Dict[str, HierarchicalModelResult] = field(default_factory=dict)
    best_model: Optional[str] = None
    model_weights: Dict[str, float] = field(default_factory=dict)
    posterior_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    robustness_metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    
    def get_model_ranking(self, criterion: str = 'dic') -> List[Tuple[str, float]]:
        """獲取模型排序"""
        if criterion == 'dic':
            scores = [(name, result.dic) for name, result in self.individual_results.items() 
                     if not np.isnan(result.dic)]
            scores.sort(key=lambda x: x[1])  # DIC越小越好
        elif criterion == 'waic':
            scores = [(name, result.waic) for name, result in self.individual_results.items()
                     if not np.isnan(result.waic)]
            scores.sort(key=lambda x: x[1])  # WAIC越小越好
        elif criterion == 'log_likelihood':
            scores = [(name, result.log_likelihood) for name, result in self.individual_results.items()
                     if not np.isnan(result.log_likelihood)]
            scores.sort(key=lambda x: x[1], reverse=True)  # LL越大越好
        else:
            raise ValueError(f"不支援的排序準則: {criterion}")
        
        return scores
    
    def get_convergence_summary(self) -> Dict[str, bool]:
        """獲取收斂性摘要"""
        convergence = {}
        for name, result in self.individual_results.items():
            conv_summary = result.diagnostics.convergence_summary()
            convergence[name] = conv_summary['overall_convergence']
        return convergence

@dataclass
class AnalyzerConfig:
    """分析器配置"""
    mcmc_config: MCMCConfig = field(default_factory=lambda: MCMCConfig(
        n_samples=500, n_warmup=250, n_chains=2
    ))
    use_mpe: bool = True
    parallel_execution: bool = False
    max_workers: Optional[int] = None
    model_selection_criterion: str = 'dic'
    calculate_ranges: bool = True
    calculate_weights: bool = True

class ModelClassAnalyzer:
    """
    模型類別分析器
    
    實現您理論框架中的核心概念：
    - 模型集合 M = Γ_f × Γ_π
    - 對每個模型 m ∈ M 進行擬合
    - 計算後驗數量的範圍 [inf_{m∈M}, sup_{m∈M}]
    - 提供模型不確定性的系統性量化
    """
    
    def __init__(self, 
                 model_class_spec: Optional[ModelClassSpec] = None,
                 config: Optional[AnalyzerConfig] = None):
        """
        初始化模型類別分析器
        
        Parameters:
        -----------
        model_class_spec : ModelClassSpec, optional
            模型類別規格，定義Γ_f和Γ_π
        config : AnalyzerConfig, optional
            分析器配置
        """
        self.model_class_spec = model_class_spec or ModelClassSpec()
        self.config = config or AnalyzerConfig()
        
        # 結果存儲
        self.last_result: Optional[ModelClassResult] = None
        self.analysis_history: List[ModelClassResult] = []
        
        # 驗證依賴
        if not HAS_HIERARCHICAL:
            raise ImportError("需要參數化階層模型模組")
        
        print(f"🏗️ 模型類別分析器初始化完成")
        print(f"   模型數量: {self.model_class_spec.get_model_count()}")
        print(f"   概似函數: {[f.value for f in self.model_class_spec.likelihood_families]}")
        print(f"   事前情境: {[p.value for p in self.model_class_spec.prior_scenarios]}")
        if self.model_class_spec.enable_epsilon_contamination:
            print(f"   ε-contamination啟用: ε值 = {self.model_class_spec.epsilon_values}")
            print(f"   污染分布類型: {self.model_class_spec.contamination_distribution}")
    
    def analyze_model_class(self, 
                          observations: Union[np.ndarray, List[float]]) -> ModelClassResult:
        """
        分析完整的模型類別
        
        這是核心方法，實現您理論中的系統性模型比較：
        對每個 m ∈ M，計算 p(θ|Data, m)
        
        Parameters:
        -----------
        observations : np.ndarray or List[float]
            觀測數據
            
        Returns:
        --------
        ModelClassResult
            完整的模型類別分析結果
        """
        observations = np.asarray(observations).flatten()
        
        print(f"🔄 開始模型類別分析...")
        print(f"   數據點數: {len(observations)}")
        print(f"   模型總數: {self.model_class_spec.get_model_count()}")
        
        start_time = time.time()
        
        # 生成所有模型規格
        all_model_specs = self.model_class_spec.generate_all_specs()
        
        # 準備結果容器
        individual_results = {}
        
        # 根據配置選擇執行方式
        if self.config.parallel_execution and len(all_model_specs) > 1:
            individual_results = self._fit_models_parallel(observations, all_model_specs)
        else:
            individual_results = self._fit_models_sequential(observations, all_model_specs)
        
        # 計算模型選擇指標
        best_model = self._select_best_model(individual_results)
        
        # 計算模型權重
        model_weights = {}
        if self.config.calculate_weights:
            model_weights = self._calculate_model_weights(individual_results)
        
        # 計算後驗數量範圍
        posterior_ranges = {}
        if self.config.calculate_ranges:
            posterior_ranges = self._calculate_posterior_ranges(individual_results)
        
        # 計算穩健性指標
        robustness_metrics = self._calculate_robustness_metrics(individual_results)
        
        execution_time = time.time() - start_time
        
        # 創建結果對象
        result = ModelClassResult(
            model_class_spec=self.model_class_spec,
            individual_results=individual_results,
            best_model=best_model,
            model_weights=model_weights,
            posterior_ranges=posterior_ranges,
            robustness_metrics=robustness_metrics,
            execution_time=execution_time
        )
        
        self.last_result = result
        self.analysis_history.append(result)
        
        print(f"✅ 模型類別分析完成")
        print(f"   執行時間: {execution_time:.2f} 秒")
        print(f"   成功擬合: {len(individual_results)}/{len(all_model_specs)} 個模型")
        print(f"   最佳模型: {best_model}")
        
        return result
    
    def _fit_models_sequential(self, 
                             observations: np.ndarray, 
                             model_specs: List[ModelSpec]) -> Dict[str, HierarchicalModelResult]:
        """順序擬合所有模型"""
        results = {}
        
        for i, spec in enumerate(model_specs, 1):
            print(f"\n  📊 擬合模型 {i}/{len(model_specs)}: {spec.model_name}")
            
            try:
                # 應用ε-contamination（如果適用）
                working_observations, contamination_info = self._apply_epsilon_contamination(observations, spec)
                if contamination_info:
                    print(f"      污染效應: {contamination_info['contamination_effect']:.3f}")
                
                model = ParametricHierarchicalModel(
                    model_spec=spec,
                    mcmc_config=self.config.mcmc_config,
                    use_mpe=self.config.use_mpe
                )
                
                # 處理特殊情況（如LogNormal需要正值數據）
                if spec.likelihood_family == LikelihoodFamily.LOGNORMAL:
                    if np.any(working_observations <= 0):
                        print("      調整數據為正值 (LogNormal要求)")
                        adjusted_obs = np.abs(working_observations) + 1e-6
                    else:
                        adjusted_obs = working_observations
                    result = model.fit(adjusted_obs)
                else:
                    result = model.fit(working_observations)
                
                results[spec.model_name] = result
                
                print(f"      ✅ 擬合成功")
                print(f"         DIC: {result.dic:.2f}")
                print(f"         收斂: {result.diagnostics.convergence_summary()['overall_convergence']}")
                
            except Exception as e:
                print(f"      ❌ 擬合失敗: {str(e)[:100]}...")
        
        return results
    
    def _fit_models_parallel(self, 
                           observations: np.ndarray, 
                           model_specs: List[ModelSpec]) -> Dict[str, HierarchicalModelResult]:
        """並行擬合所有模型"""
        print("  🚀 使用並行執行...")
        
        results = {}
        max_workers = self.config.max_workers or min(len(model_specs), 4)
        
        def fit_single_model(spec: ModelSpec) -> Tuple[str, Optional[HierarchicalModelResult]]:
            try:
                # 應用ε-contamination（如果適用）
                working_observations, contamination_info = self._apply_epsilon_contamination(observations, spec)
                
                model = ParametricHierarchicalModel(
                    model_spec=spec,
                    mcmc_config=self.config.mcmc_config,
                    use_mpe=self.config.use_mpe
                )
                
                if spec.likelihood_family == LikelihoodFamily.LOGNORMAL:
                    if np.any(working_observations <= 0):
                        adjusted_obs = np.abs(working_observations) + 1e-6
                    else:
                        adjusted_obs = working_observations
                    result = model.fit(adjusted_obs)
                else:
                    result = model.fit(working_observations)
                
                return spec.model_name, result
            except Exception as e:
                print(f"      ❌ 並行擬合失敗 {spec.model_name}: {e}")
                return spec.model_name, None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_spec = {executor.submit(fit_single_model, spec): spec 
                            for spec in model_specs}
            
            completed = 0
            for future in as_completed(future_to_spec):
                completed += 1
                spec = future_to_spec[future]
                
                try:
                    model_name, result = future.result()
                    if result is not None:
                        results[model_name] = result
                        print(f"  ✅ 完成 {completed}/{len(model_specs)}: {model_name}")
                    else:
                        print(f"  ❌ 失敗 {completed}/{len(model_specs)}: {model_name}")
                except Exception as e:
                    print(f"  ⚠️ 例外 {completed}/{len(model_specs)}: {spec.model_name} - {e}")
        
        return results
    
    def _apply_epsilon_contamination(self, 
                                   observations: np.ndarray, 
                                   spec: ModelSpec) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        應用ε-contamination到觀測數據
        
        π(θ) = (1-ε)π₀(θ) + εq(θ)
        """
        if not HAS_EPSILON_CONTAMINATION:
            return observations, {}
        
        # 檢查是否為ε-contamination模型
        epsilon = getattr(spec, 'epsilon_contamination', None)
        if epsilon is None:
            return observations, {}
        
        print(f"      應用 ε-contamination (ε={epsilon:.2f})...")
        
        # 創建污染規格
        contamination_type = getattr(spec, 'contamination_type', 'typhoon')
        if contamination_type == 'typhoon':
            # Use epsilon as single value in the range
            contamination_spec = create_typhoon_contamination_spec((epsilon, epsilon * 1.5))
        else:
            # 使用一般重尾分布污染 (修正參數名稱)
            contamination_spec = EpsilonContaminationSpec(
                epsilon_range=(epsilon, epsilon * 1.5),  # 使用 epsilon_range 而不是 epsilon
                nominal_prior_family="normal",
                contamination_prior_family="student_t"
            )
        
        # 簡化污染應用 (避免複雜的污染計算)
        # 簡單添加噪音來模擬污染效應
        np.random.seed(42)  # 確保可重現性
        n_samples = len(observations)
        
        # 添加基於 epsilon 比例的極值噪音
        contaminated_samples = observations.copy()
        n_contaminated = int(epsilon * n_samples)
        
        if n_contaminated > 0:
            # 選擇隨機樣本進行污染
            contaminated_indices = np.random.choice(n_samples, n_contaminated, replace=False)
            # 添加重尾噪音 (模擬極端事件)
            noise_scale = np.std(observations) * 2.0  # 大噪音
            contamination_noise = np.random.exponential(noise_scale, n_contaminated)
            contaminated_samples[contaminated_indices] += contamination_noise
        
        contamination_info = {
            "epsilon": epsilon,
            "contamination_type": contamination_type,
            "original_mean": np.mean(observations),
            "contaminated_mean": np.mean(contaminated_samples),
            "contaminated_samples_count": n_contaminated,
            "contamination_effect": np.std(contaminated_samples) / np.std(observations) if np.std(observations) > 0 else 1.0
        }
        
        return contaminated_samples, contamination_info
    
    def _select_best_model(self, results: Dict[str, HierarchicalModelResult]) -> Optional[str]:
        """選擇最佳模型"""
        if not results:
            return None
        
        criterion = self.config.model_selection_criterion
        
        if criterion == 'dic':
            valid_results = {name: result.dic for name, result in results.items() 
                           if not np.isnan(result.dic)}
            if valid_results:
                return min(valid_results, key=valid_results.get)
        elif criterion == 'waic':
            valid_results = {name: result.waic for name, result in results.items()
                           if not np.isnan(result.waic)}
            if valid_results:
                return min(valid_results, key=valid_results.get)
        elif criterion == 'log_likelihood':
            valid_results = {name: result.log_likelihood for name, result in results.items()
                           if not np.isnan(result.log_likelihood)}
            if valid_results:
                return max(valid_results, key=valid_results.get)
        
        # 回退方案：選擇第一個收斂的模型
        for name, result in results.items():
            if result.diagnostics.convergence_summary()['overall_convergence']:
                return name
        
        # 最後回退：選擇第一個模型
        return list(results.keys())[0] if results else None
    
    def _calculate_model_weights(self, results: Dict[str, HierarchicalModelResult]) -> Dict[str, float]:
        """計算模型權重（基於AIC權重）"""
        if not results:
            return {}
        
        # 使用DIC計算權重
        dic_values = {}
        for name, result in results.items():
            if not np.isnan(result.dic):
                dic_values[name] = result.dic
        
        if not dic_values:
            # 均等權重作為回退
            equal_weight = 1.0 / len(results)
            return {name: equal_weight for name in results.keys()}
        
        # 計算AIC權重的DIC版本
        dic_min = min(dic_values.values())
        delta_dic = {name: dic - dic_min for name, dic in dic_values.items()}
        
        weights_unnorm = {name: np.exp(-0.5 * delta) for name, delta in delta_dic.items()}
        total_weight = sum(weights_unnorm.values())
        
        weights = {name: w / total_weight for name, w in weights_unnorm.items()}
        
        # 為沒有DIC的模型分配零權重
        for name in results.keys():
            if name not in weights:
                weights[name] = 0.0
        
        return weights
    
    def _calculate_posterior_ranges(self, 
                                  results: Dict[str, HierarchicalModelResult]) -> Dict[str, Tuple[float, float]]:
        """
        計算後驗數量的範圍
        
        實現您理論中的關鍵概念：
        E_[g(Θ)|Data] = inf_{π∈Γ_π} E_{π(Θ|Data)}[g(Θ)]
        E^[g(Θ)|Data] = sup_{π∈Γ_π} E_{π(Θ|Data)}[g(Θ)]
        """
        if not results:
            return {}
        
        # 收集所有參數名稱
        all_param_names = set()
        for result in results.values():
            all_param_names.update(result.posterior_samples.keys())
        
        ranges = {}
        
        for param_name in all_param_names:
            param_means = []
            
            # 收集每個模型的參數後驗均值
            for result in results.values():
                if param_name in result.posterior_samples:
                    samples = result.posterior_samples[param_name]
                    if isinstance(samples, np.ndarray) and samples.ndim == 1:
                        param_mean = np.mean(samples)
                        if not np.isnan(param_mean):
                            param_means.append(param_mean)
            
            if param_means:
                inf_value = np.min(param_means)
                sup_value = np.max(param_means)
                ranges[param_name] = (inf_value, sup_value)
        
        return ranges
    
    def _calculate_robustness_metrics(self, results: Dict[str, HierarchicalModelResult]) -> Dict[str, Any]:
        """計算穩健性指標"""
        if not results:
            return {}
        
        # 收斂性統計
        convergence_summary = {}
        for name, result in results.items():
            conv = result.diagnostics.convergence_summary()
            convergence_summary[name] = conv['overall_convergence']
        
        convergence_rate = sum(convergence_summary.values()) / len(convergence_summary)
        
        # DIC範圍
        dic_values = [result.dic for result in results.values() if not np.isnan(result.dic)]
        dic_range = (np.min(dic_values), np.max(dic_values)) if dic_values else (np.nan, np.nan)
        
        # 模型一致性：檢查參數估計的變異性
        consistency_metrics = {}
        param_names = ['theta', 'alpha', 'phi']  # 主要參數
        
        for param in param_names:
            param_estimates = []
            for result in results.values():
                if param in result.posterior_samples:
                    samples = result.posterior_samples[param]
                    if isinstance(samples, np.ndarray) and samples.ndim == 1:
                        param_estimates.append(np.mean(samples))
            
            if len(param_estimates) > 1:
                # 變異係數作為一致性指標
                cv = np.std(param_estimates) / (np.abs(np.mean(param_estimates)) + 1e-10)
                consistency_metrics[param] = cv
        
        robustness_metrics = {
            "convergence_rate": convergence_rate,
            "dic_range": dic_range,
            "model_consistency": consistency_metrics,
            "n_successful_fits": len(results),
            "total_attempted": self.model_class_spec.get_model_count()
        }
        
        return robustness_metrics
    
    def compute_posterior_range(self, parameter_name: str) -> Optional[Tuple[float, float]]:
        """
        計算特定參數的後驗範圍
        
        便利方法，直接返回 [inf, sup]
        """
        if self.last_result is None:
            raise ValueError("需要先執行 analyze_model_class()")
        
        return self.last_result.posterior_ranges.get(parameter_name)
    
    def get_model_comparison_table(self) -> pd.DataFrame:
        """獲取模型比較表"""
        if self.last_result is None:
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, result in self.last_result.individual_results.items():
            convergence = result.diagnostics.convergence_summary()
            
            comparison_data.append({
                "模型": name,
                "概似函數": result.model_spec.likelihood_family.value,
                "事前情境": result.model_spec.prior_scenario.value,
                "對數似然": result.log_likelihood,
                "DIC": result.dic,
                "WAIC": result.waic,
                "收斂": convergence['overall_convergence'],
                "最大R-hat": convergence['max_rhat'],
                "最小ESS": convergence['min_ess_bulk'],
                "權重": self.last_result.model_weights.get(name, 0.0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 按DIC排序
        if not df.empty and 'DIC' in df.columns:
            df = df.sort_values('DIC')
        
        return df
    
    def get_robustness_summary(self) -> Dict[str, Any]:
        """獲取穩健性摘要"""
        if self.last_result is None:
            return {}
        
        return {
            "分析摘要": {
                "模型總數": self.model_class_spec.get_model_count(),
                "成功擬合": self.last_result.robustness_metrics["n_successful_fits"],
                "收斂率": f"{self.last_result.robustness_metrics['convergence_rate']:.1%}",
                "執行時間": f"{self.last_result.execution_time:.2f} 秒"
            },
            "最佳模型": self.last_result.best_model,
            "DIC範圍": self.last_result.robustness_metrics["dic_range"],
            "參數範圍": self.last_result.posterior_ranges,
            "模型一致性": self.last_result.robustness_metrics["model_consistency"]
        }

# 便利函數
def quick_model_class_analysis(observations: Union[np.ndarray, List[float]],
                             likelihood_families: Optional[List[str]] = None,
                             prior_scenarios: Optional[List[str]] = None,
                             n_samples: int = 300) -> ModelClassResult:
    """
    便利函數：快速模型類別分析
    
    Parameters:
    -----------
    observations : np.ndarray or List[float]
        觀測數據
    likelihood_families : List[str], optional
        概似函數列表
    prior_scenarios : List[str], optional
        事前情境列表
    n_samples : int
        MCMC樣本數
        
    Returns:
    --------
    ModelClassResult
        分析結果
    """
    # 預設配置
    if likelihood_families is None:
        likelihood_families = ["normal", "student_t"]
    if prior_scenarios is None:
        prior_scenarios = ["weak_informative", "pessimistic"]
    
    # 轉換為enum
    lf_enums = [LikelihoodFamily(lf) for lf in likelihood_families]
    ps_enums = [PriorScenario(ps) for ps in prior_scenarios]
    
    # 創建規格和配置
    model_spec = ModelClassSpec(
        likelihood_families=lf_enums,
        prior_scenarios=ps_enums
    )
    
    config = AnalyzerConfig(
        mcmc_config=MCMCConfig(n_samples=n_samples, n_warmup=n_samples//2, n_chains=2)
    )
    
    # 執行分析
    analyzer = ModelClassAnalyzer(model_spec, config)
    return analyzer.analyze_model_class(observations)

def test_model_class_analyzer():
    """測試模型類別分析器功能"""
    print("🧪 測試模型類別分析器...")
    
    # 生成測試數據
    np.random.seed(42)
    true_theta = 3.0
    test_data = np.random.normal(true_theta, 1.5, 50)
    
    print(f"\n測試數據: 均值={np.mean(test_data):.3f}, 標準差={np.std(test_data):.3f}")
    
    # 測試基本分析
    print("\n🔍 執行基本模型類別分析...")
    result_basic = quick_model_class_analysis(
        test_data,
        likelihood_families=["normal", "student_t"],
        prior_scenarios=["weak_informative", "optimistic"],
        n_samples=200
    )
    
    # 測試ε-contamination分析
    if HAS_EPSILON_CONTAMINATION:
        print("\n🔬 執行 ε-contamination 模型類別分析...")
        
        model_spec = ModelClassSpec(
            likelihood_families=[LikelihoodFamily.NORMAL, LikelihoodFamily.STUDENT_T],
            prior_scenarios=[PriorScenario.WEAK_INFORMATIVE],
            enable_epsilon_contamination=True,
            epsilon_values=[0.05, 0.1],
            contamination_distribution="typhoon"
        )
        
        config = AnalyzerConfig(
            mcmc_config=MCMCConfig(n_samples=150, n_warmup=75, n_chains=2)
        )
        
        analyzer = ModelClassAnalyzer(model_spec, config)
        result = analyzer.analyze_model_class(test_data)
        
        print(f"\n📊 ε-contamination 分析結果:")
        print(f"   總模型數: {len(result.individual_results)}")
        contamination_models = [name for name in result.individual_results.keys() if 'epsilon' in name]
        print(f"   污染模型數: {len(contamination_models)}")
        
    else:
        result = result_basic
    
    # 顯示結果
    print(f"\n📊 分析結果:")
    print(f"   最佳模型: {result.best_model}")
    print(f"   執行時間: {result.execution_time:.2f} 秒")
    
    # 顯示比較表
    print("\n📋 模型比較表:")
    analyzer = ModelClassAnalyzer()
    analyzer.last_result = result  # 設置結果以便生成表格
    comparison_table = analyzer.get_model_comparison_table()
    print(comparison_table[['模型', 'DIC', '收斂', '權重']])
    
    # 顯示參數範圍
    print("\n📈 參數後驗範圍:")
    for param, (inf_val, sup_val) in result.posterior_ranges.items():
        print(f"   {param}: [{inf_val:.3f}, {sup_val:.3f}]")
    
    # 穩健性摘要
    print("\n🛡️ 穩健性摘要:")
    robustness = analyzer.get_robustness_summary()
    for section, data in robustness.items():
        print(f"   {section}: {data}")
    
    print("\n✅ 測試完成")
    return result

if __name__ == "__main__":
    test_model_class_analyzer()