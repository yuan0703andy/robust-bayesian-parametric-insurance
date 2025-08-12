#!/usr/bin/env python3
"""
Robust Credible Intervals Module
穩健可信區間模組

實現您理論框架中的穩健可信區間計算：
對所有模型 π ∈ Γ_π 都滿足信賴水準的區間 C，使得
inf_{π∈Γ_π} P_{π(θ|Data)}(θ ∈ C) ≥ 1-α

核心功能:
- 多約束優化求解穩健區間
- 跨模型的同時覆蓋率計算
- 區間寬度與模型不確定性分析
- 貝氏與頻率派穩健區間比較

使用範例:
```python
from bayesian.minimax_credible_intervals import RobustCredibleIntervalCalculator

# 初始化計算器
calculator = RobustCredibleIntervalCalculator()

# 計算穩健可信區間
robust_interval = calculator.compute_robust_interval(
    posterior_samples_dict,  # Dict[model_name, samples]
    parameter_name='theta',
    alpha=0.05
)

print(f"穩健95%可信區間: [{robust_interval[0]:.3f}, {robust_interval[1]:.3f}]")

# 與標準區間比較
comparison = calculator.compare_interval_types(
    posterior_samples_dict, 'theta', alpha=0.05
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
from scipy.optimize import minimize, minimize_scalar
from scipy import stats
import warnings

@dataclass
class IntervalResult:
    """區間計算結果"""
    parameter_name: str
    alpha: float
    interval: Tuple[float, float]
    coverage_rates: Dict[str, float] = field(default_factory=dict)
    interval_width: float = 0.0
    method: str = "unknown"
    optimization_success: bool = True
    optimization_details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.interval:
            self.interval_width = self.interval[1] - self.interval[0]

@dataclass 
class IntervalComparison:
    """區間比較結果"""
    parameter_name: str
    alpha: float
    standard_interval: Tuple[float, float]
    robust_interval: Tuple[float, float]
    width_ratio: float  # robust_width / standard_width
    coverage_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)

class IntervalOptimizationMethod(Enum):
    """區間優化方法"""
    GRID_SEARCH = "grid_search"
    CONSTRAINED_OPTIMIZATION = "constrained_optimization"
    QUANTILE_BASED = "quantile_based"
    BAYESIAN_BOOTSTRAP = "bayesian_bootstrap"

@dataclass
class CalculatorConfig:
    """計算器配置"""
    optimization_method: IntervalOptimizationMethod = IntervalOptimizationMethod.CONSTRAINED_OPTIMIZATION
    grid_resolution: int = 1000
    optimization_tolerance: float = 1e-6
    max_iterations: int = 1000
    use_parallel: bool = False
    min_coverage_tolerance: float = 1e-3

class RobustCredibleIntervalCalculator:
    """
    穩健可信區間計算器
    
    實現您理論框架中最具挑戰性的部分：
    尋找對所有後驗分布都滿足信賴水準的區間
    
    數學問題：
    Given: {π₁(θ|Data), π₂(θ|Data), ..., πₖ(θ|Data)}
    Find: C such that min_i P_πᵢ(θ ∈ C) ≥ 1-α
    Minimize: |C| (區間長度)
    """
    
    def __init__(self, config: Optional[CalculatorConfig] = None):
        """
        初始化穩健可信區間計算器
        
        Parameters:
        -----------
        config : CalculatorConfig, optional
            計算器配置
        """
        self.config = config or CalculatorConfig()
        
        # 結果緩存
        self.calculation_cache: Dict[str, IntervalResult] = {}
        self.comparison_cache: Dict[str, IntervalComparison] = {}
        
        print("🛡️ 穩健可信區間計算器初始化完成")
        print(f"   優化方法: {self.config.optimization_method.value}")
    
    def compute_robust_interval(self,
                              posterior_samples_dict: Dict[str, np.ndarray],
                              parameter_name: str,
                              alpha: float = 0.05,
                              method: Optional[IntervalOptimizationMethod] = None) -> Tuple[float, float]:
        """
        計算穩健可信區間
        
        這是核心方法，實現多約束優化：
        minimize |C| subject to P_πᵢ(θ ∈ C) ≥ 1-α for all i
        
        Parameters:
        -----------
        posterior_samples_dict : Dict[str, np.ndarray]
            每個模型的後驗樣本，格式: {model_name: samples}
        parameter_name : str
            參數名稱
        alpha : float
            顯著水平 (預設 0.05 for 95% 區間)
        method : IntervalOptimizationMethod, optional
            優化方法
            
        Returns:
        --------
        Tuple[float, float]
            穩健可信區間 (lower_bound, upper_bound)
        """
        print(f"🔍 計算參數 '{parameter_name}' 的穩健 {100*(1-alpha):.1f}% 可信區間...")
        print(f"   模型數量: {len(posterior_samples_dict)}")
        
        # 提取所有模型的樣本
        all_samples = {}
        for model_name, samples in posterior_samples_dict.items():
            if isinstance(samples, dict) and parameter_name in samples:
                param_samples = np.asarray(samples[parameter_name]).flatten()
            elif isinstance(samples, np.ndarray):
                param_samples = samples.flatten()
            else:
                continue
            
            # 過濾有效樣本
            valid_samples = param_samples[~np.isnan(param_samples)]
            if len(valid_samples) > 0:
                all_samples[model_name] = valid_samples
                print(f"   {model_name}: {len(valid_samples)} 個有效樣本")
        
        if not all_samples:
            raise ValueError(f"沒有找到參數 '{parameter_name}' 的有效樣本")
        
        # 選擇優化方法
        opt_method = method or self.config.optimization_method
        
        if opt_method == IntervalOptimizationMethod.CONSTRAINED_OPTIMIZATION:
            result = self._compute_robust_interval_optimization(all_samples, alpha)
        elif opt_method == IntervalOptimizationMethod.GRID_SEARCH:
            result = self._compute_robust_interval_grid_search(all_samples, alpha)
        elif opt_method == IntervalOptimizationMethod.QUANTILE_BASED:
            result = self._compute_robust_interval_quantile(all_samples, alpha)
        elif opt_method == IntervalOptimizationMethod.BAYESIAN_BOOTSTRAP:
            result = self._compute_robust_interval_bootstrap(all_samples, alpha)
        else:
            raise ValueError(f"不支援的優化方法: {opt_method}")
        
        # 創建結果對象
        interval_result = IntervalResult(
            parameter_name=parameter_name,
            alpha=alpha,
            interval=result["interval"],
            coverage_rates=result["coverage_rates"],
            method=opt_method.value,
            optimization_success=result["success"],
            optimization_details=result["details"]
        )
        
        # 緩存結果
        cache_key = f"{parameter_name}_{alpha}_{len(all_samples)}"
        self.calculation_cache[cache_key] = interval_result
        
        print(f"✅ 穩健區間計算完成: [{result['interval'][0]:.4f}, {result['interval'][1]:.4f}]")
        print(f"   區間寬度: {interval_result.interval_width:.4f}")
        print(f"   最小覆蓋率: {min(result['coverage_rates'].values()):.1%}")
        
        return result["interval"]
    
    def _compute_robust_interval_optimization(self,
                                            samples_dict: Dict[str, np.ndarray],
                                            alpha: float) -> Dict[str, Any]:
        """使用約束優化計算穩健區間"""
        print("   🎯 使用約束優化方法...")
        
        # 合併所有樣本以確定搜索範圍
        all_values = np.concatenate(list(samples_dict.values()))
        data_min = np.min(all_values)
        data_max = np.max(all_values)
        data_range = data_max - data_min
        
        # 擴展搜索範圍
        search_min = data_min - 0.1 * data_range
        search_max = data_max + 0.1 * data_range
        
        def coverage_rate(samples: np.ndarray, lower: float, upper: float) -> float:
            """計算覆蓋率"""
            return np.mean((samples >= lower) & (samples <= upper))
        
        def objective(params):
            """目標函數：最小化區間寬度"""
            lower, upper = params[0], params[1]
            if lower >= upper:
                return 1e10  # 懲罰無效區間
            return upper - lower
        
        def constraint_func(params):
            """約束函數：所有模型的覆蓋率 >= 1-α"""
            lower, upper = params[0], params[1]
            if lower >= upper:
                return -1e10
            
            min_coverage = float('inf')
            for samples in samples_dict.values():
                coverage = coverage_rate(samples, lower, upper)
                min_coverage = min(min_coverage, coverage)
            
            return min_coverage - (1 - alpha) + self.config.min_coverage_tolerance
        
        # 約束定義
        constraint = {
            'type': 'ineq',
            'fun': constraint_func
        }
        
        # 邊界約束
        bounds = [(search_min, search_max), (search_min, search_max)]
        
        # 初始猜測：使用所有樣本的標準分位數
        combined_samples = np.concatenate(list(samples_dict.values()))
        initial_lower = np.percentile(combined_samples, 100 * alpha / 2)
        initial_upper = np.percentile(combined_samples, 100 * (1 - alpha / 2))
        initial_guess = [initial_lower, initial_upper]
        
        # 執行優化
        try:
            opt_result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint,
                options={
                    'ftol': self.config.optimization_tolerance,
                    'maxiter': self.config.max_iterations
                }
            )
            
            if opt_result.success:
                robust_lower, robust_upper = opt_result.x
                
                # 確保順序正確
                if robust_lower > robust_upper:
                    robust_lower, robust_upper = robust_upper, robust_lower
                
                # 計算每個模型的實際覆蓋率
                coverage_rates = {}
                for model_name, samples in samples_dict.items():
                    coverage_rates[model_name] = coverage_rate(samples, robust_lower, robust_upper)
                
                return {
                    "interval": (robust_lower, robust_upper),
                    "coverage_rates": coverage_rates,
                    "success": True,
                    "details": {
                        "optimization_result": opt_result,
                        "method": "constrained_optimization"
                    }
                }
            else:
                print("   ⚠️ 約束優化失敗，回退到分位數方法")
                return self._compute_robust_interval_quantile(samples_dict, alpha)
                
        except Exception as e:
            print(f"   ⚠️ 優化過程出錯: {e}")
            return self._compute_robust_interval_quantile(samples_dict, alpha)
    
    def _compute_robust_interval_grid_search(self,
                                           samples_dict: Dict[str, np.ndarray],
                                           alpha: float) -> Dict[str, Any]:
        """使用網格搜索計算穩健區間"""
        print("   🔍 使用網格搜索方法...")
        
        # 確定搜索網格
        all_values = np.concatenate(list(samples_dict.values()))
        data_min = np.min(all_values)
        data_max = np.max(all_values)
        
        # 創建網格
        grid_points = np.linspace(data_min, data_max, self.config.grid_resolution)
        
        best_interval = None
        best_width = float('inf')
        best_coverage = {}
        
        # 遍歷所有可能的區間
        for i in range(len(grid_points)):
            for j in range(i + 1, len(grid_points)):
                lower = grid_points[i]
                upper = grid_points[j]
                
                # 計算每個模型的覆蓋率
                min_coverage = float('inf')
                coverage_rates = {}
                
                for model_name, samples in samples_dict.items():
                    coverage = np.mean((samples >= lower) & (samples <= upper))
                    coverage_rates[model_name] = coverage
                    min_coverage = min(min_coverage, coverage)
                
                # 檢查是否滿足約束
                if min_coverage >= (1 - alpha - self.config.min_coverage_tolerance):
                    width = upper - lower
                    if width < best_width:
                        best_width = width
                        best_interval = (lower, upper)
                        best_coverage = coverage_rates.copy()
        
        if best_interval is None:
            print("   ⚠️ 網格搜索未找到滿足約束的區間，使用保守估計")
            return self._compute_robust_interval_quantile(samples_dict, alpha)
        
        return {
            "interval": best_interval,
            "coverage_rates": best_coverage,
            "success": True,
            "details": {
                "method": "grid_search",
                "grid_resolution": self.config.grid_resolution,
                "best_width": best_width
            }
        }
    
    def _compute_robust_interval_quantile(self,
                                        samples_dict: Dict[str, np.ndarray],
                                        alpha: float) -> Dict[str, Any]:
        """使用分位數方法計算穩健區間（保守估計）"""
        print("   📊 使用分位數方法（保守估計）...")
        
        # 對每個模型計算標準可信區間
        individual_intervals = {}
        for model_name, samples in samples_dict.items():
            lower = np.percentile(samples, 100 * alpha / 2)
            upper = np.percentile(samples, 100 * (1 - alpha / 2))
            individual_intervals[model_name] = (lower, upper)
        
        # 保守的穩健區間：取所有區間的並集
        all_lowers = [interval[0] for interval in individual_intervals.values()]
        all_uppers = [interval[1] for interval in individual_intervals.values()]
        
        robust_lower = np.min(all_lowers)
        robust_upper = np.max(all_uppers)
        
        # 計算覆蓋率
        coverage_rates = {}
        for model_name, samples in samples_dict.items():
            coverage = np.mean((samples >= robust_lower) & (samples <= robust_upper))
            coverage_rates[model_name] = coverage
        
        return {
            "interval": (robust_lower, robust_upper),
            "coverage_rates": coverage_rates,
            "success": True,
            "details": {
                "method": "quantile_based",
                "individual_intervals": individual_intervals,
                "conservatism_note": "保守估計，可能過寬"
            }
        }
    
    def _compute_robust_interval_bootstrap(self,
                                         samples_dict: Dict[str, np.ndarray],
                                         alpha: float) -> Dict[str, Any]:
        """使用貝氏拔靴法計算穩健區間"""
        print("   🥾 使用貝氏拔靴法...")
        
        n_bootstrap = 1000
        bootstrap_intervals = []
        
        # 對每個模型進行拔靴法抽樣
        for model_name, samples in samples_dict.items():
            model_intervals = []
            
            for _ in range(n_bootstrap):
                # 拔靴法重抽樣
                boot_samples = np.random.choice(samples, size=len(samples), replace=True)
                lower = np.percentile(boot_samples, 100 * alpha / 2)
                upper = np.percentile(boot_samples, 100 * (1 - alpha / 2))
                model_intervals.append((lower, upper))
            
            bootstrap_intervals.append(model_intervals)
        
        # 對於每次拔靴法，計算穩健區間
        robust_intervals = []
        
        for boot_idx in range(n_bootstrap):
            # 取該次拔靴法所有模型的區間聯集
            boot_lowers = []
            boot_uppers = []
            
            for model_intervals in bootstrap_intervals:
                lower, upper = model_intervals[boot_idx]
                boot_lowers.append(lower)
                boot_uppers.append(upper)
            
            robust_lower = np.min(boot_lowers)
            robust_upper = np.max(boot_uppers)
            robust_intervals.append((robust_lower, robust_upper))
        
        # 計算拔靴法區間的中位數
        robust_lowers = [interval[0] for interval in robust_intervals]
        robust_uppers = [interval[1] for interval in robust_intervals]
        
        final_lower = np.percentile(robust_lowers, 100 * alpha / 2)
        final_upper = np.percentile(robust_uppers, 100 * (1 - alpha / 2))
        
        # 計算覆蓋率
        coverage_rates = {}
        for model_name, samples in samples_dict.items():
            coverage = np.mean((samples >= final_lower) & (samples <= final_upper))
            coverage_rates[model_name] = coverage
        
        return {
            "interval": (final_lower, final_upper),
            "coverage_rates": coverage_rates,
            "success": True,
            "details": {
                "method": "bayesian_bootstrap",
                "n_bootstrap": n_bootstrap,
                "bootstrap_intervals": robust_intervals
            }
        }
    
    def compute_standard_interval(self,
                                samples: np.ndarray,
                                alpha: float = 0.05) -> Tuple[float, float]:
        """計算標準可信區間（單一模型）"""
        lower = np.percentile(samples, 100 * alpha / 2)
        upper = np.percentile(samples, 100 * (1 - alpha / 2))
        return lower, upper
    
    def compare_interval_types(self,
                             posterior_samples_dict: Dict[str, np.ndarray],
                             parameter_name: str,
                             alpha: float = 0.05) -> IntervalComparison:
        """
        比較標準區間與穩健區間
        
        Parameters:
        -----------
        posterior_samples_dict : Dict[str, np.ndarray]
            每個模型的後驗樣本
        parameter_name : str
            參數名稱
        alpha : float
            顯著水平
            
        Returns:
        --------
        IntervalComparison
            區間比較結果
        """
        print(f"📊 比較標準與穩健可信區間 (參數: {parameter_name})...")
        
        # 計算穩健區間
        robust_interval = self.compute_robust_interval(
            posterior_samples_dict, parameter_name, alpha
        )
        
        # 計算標準區間（使用合併樣本）
        all_samples = []
        coverage_by_model = {"standard": {}, "robust": {}}
        
        for model_name, samples in posterior_samples_dict.items():
            if isinstance(samples, dict) and parameter_name in samples:
                param_samples = np.asarray(samples[parameter_name]).flatten()
            else:
                param_samples = np.asarray(samples).flatten()
            
            valid_samples = param_samples[~np.isnan(param_samples)]
            if len(valid_samples) > 0:
                all_samples.extend(valid_samples)
                
                # 計算每個模型對兩種區間的覆蓋率
                coverage_by_model["robust"][model_name] = np.mean(
                    (valid_samples >= robust_interval[0]) & 
                    (valid_samples <= robust_interval[1])
                )
        
        standard_interval = self.compute_standard_interval(np.array(all_samples), alpha)
        
        # 計算標準區間的覆蓋率
        for model_name, samples in posterior_samples_dict.items():
            if isinstance(samples, dict) and parameter_name in samples:
                param_samples = np.asarray(samples[parameter_name]).flatten()
            else:
                param_samples = np.asarray(samples).flatten()
            
            valid_samples = param_samples[~np.isnan(param_samples)]
            if len(valid_samples) > 0:
                coverage_by_model["standard"][model_name] = np.mean(
                    (valid_samples >= standard_interval[0]) & 
                    (valid_samples <= standard_interval[1])
                )
        
        # 計算寬度比
        standard_width = standard_interval[1] - standard_interval[0]
        robust_width = robust_interval[1] - robust_interval[0]
        width_ratio = robust_width / standard_width if standard_width > 0 else np.inf
        
        comparison = IntervalComparison(
            parameter_name=parameter_name,
            alpha=alpha,
            standard_interval=standard_interval,
            robust_interval=robust_interval,
            width_ratio=width_ratio,
            coverage_comparison=coverage_by_model
        )
        
        # 緩存比較結果
        cache_key = f"{parameter_name}_{alpha}_comparison"
        self.comparison_cache[cache_key] = comparison
        
        print(f"   標準區間: [{standard_interval[0]:.4f}, {standard_interval[1]:.4f}] (寬度: {standard_width:.4f})")
        print(f"   穩健區間: [{robust_interval[0]:.4f}, {robust_interval[1]:.4f}] (寬度: {robust_width:.4f})")
        print(f"   寬度比率: {width_ratio:.2f}")
        
        return comparison
    
    def analyze_interval_robustness(self,
                                  posterior_samples_dict: Dict[str, np.ndarray],
                                  parameter_name: str,
                                  alpha_levels: List[float] = None) -> Dict[str, Any]:
        """
        分析區間的穩健性特性
        
        Parameters:
        -----------
        posterior_samples_dict : Dict[str, np.ndarray]
            每個模型的後驗樣本
        parameter_name : str
            參數名稱
        alpha_levels : List[float], optional
            不同的顯著水平
            
        Returns:
        --------
        Dict[str, Any]
            穩健性分析結果
        """
        if alpha_levels is None:
            alpha_levels = [0.01, 0.05, 0.1, 0.2]
        
        print(f"🔍 分析參數 '{parameter_name}' 的區間穩健性...")
        
        robustness_results = {
            "parameter_name": parameter_name,
            "alpha_levels": alpha_levels,
            "interval_analysis": {},
            "width_analysis": {},
            "coverage_analysis": {}
        }
        
        for alpha in alpha_levels:
            print(f"   分析 α={alpha} (置信度={100*(1-alpha):.0f}%)...")
            
            comparison = self.compare_interval_types(
                posterior_samples_dict, parameter_name, alpha
            )
            
            robustness_results["interval_analysis"][alpha] = {
                "standard_interval": comparison.standard_interval,
                "robust_interval": comparison.robust_interval,
                "width_ratio": comparison.width_ratio
            }
            
            # 分析寬度變化
            standard_width = comparison.standard_interval[1] - comparison.standard_interval[0]
            robust_width = comparison.robust_interval[1] - comparison.robust_interval[0]
            
            robustness_results["width_analysis"][alpha] = {
                "standard_width": standard_width,
                "robust_width": robust_width,
                "width_difference": robust_width - standard_width,
                "width_ratio": comparison.width_ratio
            }
            
            # 分析覆蓋率
            standard_coverages = list(comparison.coverage_comparison["standard"].values())
            robust_coverages = list(comparison.coverage_comparison["robust"].values())
            
            robustness_results["coverage_analysis"][alpha] = {
                "standard_min_coverage": np.min(standard_coverages),
                "standard_coverage_std": np.std(standard_coverages),
                "robust_min_coverage": np.min(robust_coverages),
                "robust_coverage_std": np.std(robust_coverages),
                "coverage_improvement": np.min(robust_coverages) - np.min(standard_coverages)
            }
        
        # 總結分析
        width_ratios = [robustness_results["width_analysis"][α]["width_ratio"] 
                       for α in alpha_levels]
        
        robustness_results["summary"] = {
            "mean_width_ratio": np.mean(width_ratios),
            "width_ratio_std": np.std(width_ratios),
            "max_width_ratio": np.max(width_ratios),
            "stability_assessment": "穩定" if np.std(width_ratios) < 0.5 else "不穩定"
        }
        
        print(f"✅ 穩健性分析完成")
        print(f"   平均寬度比率: {robustness_results['summary']['mean_width_ratio']:.2f}")
        print(f"   穩定性評估: {robustness_results['summary']['stability_assessment']}")
        
        return robustness_results
    
    def get_calculation_summary(self) -> pd.DataFrame:
        """獲取計算摘要表"""
        if not self.calculation_cache:
            return pd.DataFrame()
        
        summary_data = []
        
        for cache_key, result in self.calculation_cache.items():
            min_coverage = min(result.coverage_rates.values()) if result.coverage_rates else np.nan
            
            summary_data.append({
                "參數": result.parameter_name,
                "α": result.alpha,
                "置信度": f"{100*(1-result.alpha):.1f}%",
                "下界": result.interval[0],
                "上界": result.interval[1],
                "寬度": result.interval_width,
                "最小覆蓋率": f"{min_coverage:.1%}" if not np.isnan(min_coverage) else "N/A",
                "方法": result.method,
                "優化成功": result.optimization_success
            })
        
        return pd.DataFrame(summary_data)

# 便利函數
def compute_robust_credible_interval(posterior_samples_dict: Dict[str, np.ndarray],
                                   parameter_name: str,
                                   alpha: float = 0.05) -> Tuple[float, float]:
    """
    便利函數：快速計算穩健可信區間
    
    Parameters:
    -----------
    posterior_samples_dict : Dict[str, np.ndarray]
        每個模型的後驗樣本
    parameter_name : str
        參數名稱
    alpha : float
        顯著水平
        
    Returns:
    --------
    Tuple[float, float]
        穩健可信區間
    """
    calculator = RobustCredibleIntervalCalculator()
    return calculator.compute_robust_interval(posterior_samples_dict, parameter_name, alpha)

def compare_credible_intervals(posterior_samples_dict: Dict[str, np.ndarray],
                             parameter_name: str,
                             alpha: float = 0.05) -> IntervalComparison:
    """
    便利函數：比較標準與穩健區間
    
    Parameters:
    -----------
    posterior_samples_dict : Dict[str, np.ndarray]
        每個模型的後驗樣本
    parameter_name : str
        參數名稱
    alpha : float
        顯著水平
        
    Returns:
    --------
    IntervalComparison
        比較結果
    """
    calculator = RobustCredibleIntervalCalculator()
    return calculator.compare_interval_types(posterior_samples_dict, parameter_name, alpha)

def test_robust_credible_intervals():
    """測試穩健可信區間功能"""
    print("🧪 測試穩健可信區間計算...")
    
    # 生成測試數據（3個不同的模型）
    np.random.seed(42)
    
    # 模型1：正態分布
    model1_samples = np.random.normal(5.0, 1.0, 500)
    
    # 模型2：稍微偏移的正態分布
    model2_samples = np.random.normal(5.5, 1.2, 500)
    
    # 模型3：更寬的正態分布
    model3_samples = np.random.normal(4.8, 1.5, 500)
    
    posterior_samples = {
        "normal_weak": model1_samples,
        "normal_optimistic": model2_samples, 
        "student_t_conservative": model3_samples
    }
    
    print(f"\n測試數據摘要:")
    for name, samples in posterior_samples.items():
        print(f"   {name}: 均值={np.mean(samples):.3f}, 標準差={np.std(samples):.3f}")
    
    # 測試穩健區間計算
    print(f"\n🔍 計算穩健可信區間...")
    calculator = RobustCredibleIntervalCalculator()
    
    robust_interval = calculator.compute_robust_interval(
        posterior_samples, "theta", alpha=0.05
    )
    
    print(f"穩健95%區間: [{robust_interval[0]:.4f}, {robust_interval[1]:.4f}]")
    
    # 比較不同類型的區間
    print(f"\n📊 比較區間類型...")
    comparison = calculator.compare_interval_types(
        posterior_samples, "theta", alpha=0.05
    )
    
    print(f"標準區間: [{comparison.standard_interval[0]:.4f}, {comparison.standard_interval[1]:.4f}]")
    print(f"穩健區間: [{comparison.robust_interval[0]:.4f}, {comparison.robust_interval[1]:.4f}]") 
    print(f"寬度比率: {comparison.width_ratio:.2f}")
    
    # 穩健性分析
    print(f"\n🔍 穩健性分析...")
    robustness = calculator.analyze_interval_robustness(
        posterior_samples, "theta", alpha_levels=[0.05, 0.1, 0.2]
    )
    
    print(f"平均寬度比率: {robustness['summary']['mean_width_ratio']:.2f}")
    print(f"穩定性評估: {robustness['summary']['stability_assessment']}")
    
    # 顯示摘要表
    print(f"\n📋 計算摘要:")
    summary_table = calculator.get_calculation_summary()
    if not summary_table.empty:
        print(summary_table[['參數', '置信度', '寬度', '最小覆蓋率', '方法']])
    
    print(f"\n✅ 穩健可信區間測試完成")
    return robust_interval, comparison, robustness

if __name__ == "__main__":
    test_robust_credible_intervals()