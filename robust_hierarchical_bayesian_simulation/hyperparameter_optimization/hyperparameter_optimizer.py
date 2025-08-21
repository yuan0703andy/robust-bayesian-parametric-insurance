#!/usr/bin/env python3
"""
Hyperparameter Optimizer for CRPS Framework
CRPS框架超參數優化器

專門負責超參數調優，特別是λ權重的智能搜索

Author: Research Team
Date: 2025-01-17
Version: 2.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import warnings
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import logging

logger = logging.getLogger(__name__)

@dataclass
class HyperparameterSearchSpace:
    """超參數搜索空間定義"""
    lambda_crps_range: Tuple[float, float] = (0.01, 100.0)
    lambda_under_range: Tuple[float, float] = (0.1, 10.0)
    lambda_over_range: Tuple[float, float] = (0.1, 10.0)
    epsilon_range: Tuple[float, float] = (0.01, 0.3)
    spatial_correlation_range: Tuple[float, float] = (10, 500)  # km
    
    def sample_random_point(self) -> Dict[str, float]:
        """隨機採樣一個點"""
        return {
            'lambda_crps': np.random.uniform(*self.lambda_crps_range),
            'lambda_under': np.random.uniform(*self.lambda_under_range),
            'lambda_over': np.random.uniform(*self.lambda_over_range),
            'epsilon': np.random.uniform(*self.epsilon_range),
            'spatial_corr': np.random.uniform(*self.spatial_correlation_range)
        }
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """獲取優化器的邊界"""
        return [
            self.lambda_crps_range,
            self.lambda_under_range,
            self.lambda_over_range,
            self.epsilon_range,
            self.spatial_correlation_range
        ]

class AdaptiveHyperparameterOptimizer:
    """
    自適應超參數優化器
    
    使用多種策略智能搜索最佳超參數：
    1. 網格搜索（基礎）
    2. 貝葉斯優化（高效）
    3. 進化算法（全局）
    4. 自適應策略（智能）
    """
    
    def __init__(self,
                 objective_function: Callable,
                 search_space: Optional[HyperparameterSearchSpace] = None,
                 strategy: str = 'adaptive'):
        """
        初始化優化器
        
        Parameters:
        -----------
        objective_function : Callable
            目標函數，輸入超參數，返回分數（越高越好）
        search_space : HyperparameterSearchSpace
            搜索空間
        strategy : str
            優化策略：'grid', 'bayesian', 'evolutionary', 'adaptive'
        """
        self.objective_function = objective_function
        self.search_space = search_space or HyperparameterSearchSpace()
        self.strategy = strategy
        
        # 優化歷史
        self.history = []
        self.best_params = None
        self.best_score = -np.inf
        
        # 貝葉斯優化的GP模型
        if strategy in ['bayesian', 'adaptive']:
            self.gp_model = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10
            )
            self.X_observed = []
            self.y_observed = []
    
    def optimize(self, 
                n_iterations: int = 50,
                n_initial_points: int = 10) -> Dict[str, Any]:
        """
        執行超參數優化
        
        Parameters:
        -----------
        n_iterations : int
            總迭代次數
        n_initial_points : int
            初始隨機探索點數
            
        Returns:
        --------
        Dict[str, Any]
            最佳超參數和優化結果
        """
        print(f"🎯 開始{self.strategy}超參數優化...")
        
        if self.strategy == 'grid':
            return self._grid_search(n_iterations)
        elif self.strategy == 'bayesian':
            return self._bayesian_optimization(n_iterations, n_initial_points)
        elif self.strategy == 'evolutionary':
            return self._evolutionary_optimization(n_iterations)
        elif self.strategy == 'adaptive':
            return self._adaptive_optimization(n_iterations, n_initial_points)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _grid_search(self, n_points: int) -> Dict[str, Any]:
        """網格搜索"""
        print("📊 執行網格搜索...")
        
        # 生成網格點
        grid_points = []
        n_per_dim = int(np.power(n_points, 1/5))  # 5維空間
        
        lambda_crps_grid = np.logspace(
            np.log10(self.search_space.lambda_crps_range[0]),
            np.log10(self.search_space.lambda_crps_range[1]),
            n_per_dim
        )
        
        for lambda_crps in lambda_crps_grid:
            for lambda_under in np.linspace(*self.search_space.lambda_under_range, 3):
                for lambda_over in np.linspace(*self.search_space.lambda_over_range, 3):
                    point = {
                        'lambda_crps': lambda_crps,
                        'lambda_under': lambda_under,
                        'lambda_over': lambda_over,
                        'epsilon': 0.1,  # 固定值
                        'spatial_corr': 100  # 固定值
                    }
                    grid_points.append(point)
        
        # 評估所有點
        for i, params in enumerate(grid_points):
            score = self.objective_function(params)
            self.history.append({'params': params, 'score': score})
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                print(f"   ✓ 新最佳: λ={params['lambda_crps']:.2f}, score={score:.4f}")
        
        return self._compile_results()
    
    def _bayesian_optimization(self, 
                              n_iterations: int,
                              n_initial_points: int) -> Dict[str, Any]:
        """貝葉斯優化"""
        print("🧮 執行貝葉斯優化...")
        
        # 初始隨機探索
        for i in range(n_initial_points):
            params = self.search_space.sample_random_point()
            score = self.objective_function(params)
            self._update_gp_model(params, score)
            print(f"   初始探索 {i+1}/{n_initial_points}: score={score:.4f}")
        
        # 貝葉斯優化主循環
        for i in range(n_iterations - n_initial_points):
            # 找下一個最有希望的點
            next_params = self._suggest_next_point()
            
            # 評估
            score = self.objective_function(next_params)
            self._update_gp_model(next_params, score)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = next_params
                print(f"   ✓ 迭代 {i+1}: 新最佳 λ={next_params['lambda_crps']:.2f}, score={score:.4f}")
        
        return self._compile_results()
    
    def _evolutionary_optimization(self, n_iterations: int) -> Dict[str, Any]:
        """進化算法優化"""
        print("🧬 執行進化算法優化...")
        
        def objective_wrapper(x):
            params = {
                'lambda_crps': x[0],
                'lambda_under': x[1],
                'lambda_over': x[2],
                'epsilon': x[3],
                'spatial_corr': x[4]
            }
            return -self.objective_function(params)  # 最小化
        
        result = differential_evolution(
            objective_wrapper,
            bounds=self.search_space.get_bounds(),
            maxiter=n_iterations // 15,  # 每代15個個體
            popsize=15,
            disp=True,
            seed=42
        )
        
        self.best_params = {
            'lambda_crps': result.x[0],
            'lambda_under': result.x[1],
            'lambda_over': result.x[2],
            'epsilon': result.x[3],
            'spatial_corr': result.x[4]
        }
        self.best_score = -result.fun
        
        return self._compile_results()
    
    def _adaptive_optimization(self,
                             n_iterations: int,
                             n_initial_points: int) -> Dict[str, Any]:
        """
        自適應優化策略
        
        結合多種方法的優點：
        1. 初期：隨機探索
        2. 中期：貝葉斯優化
        3. 後期：局部精煉
        """
        print("🎨 執行自適應優化策略...")
        
        total_budget = n_iterations
        
        # Phase 1: 隨機探索 (20%)
        phase1_budget = int(0.2 * total_budget)
        print(f"\n階段1: 隨機探索 ({phase1_budget} 次)")
        
        for i in range(phase1_budget):
            params = self.search_space.sample_random_point()
            score = self.objective_function(params)
            self._update_gp_model(params, score)
        
        # Phase 2: 貝葉斯優化 (60%)
        phase2_budget = int(0.6 * total_budget)
        print(f"\n階段2: 貝葉斯優化 ({phase2_budget} 次)")
        
        for i in range(phase2_budget):
            next_params = self._suggest_next_point()
            score = self.objective_function(next_params)
            self._update_gp_model(next_params, score)
            
            if score > self.best_score:
                print(f"   ✓ 新最佳: score={score:.4f}")
        
        # Phase 3: 局部精煉 (20%)
        phase3_budget = total_budget - phase1_budget - phase2_budget
        print(f"\n階段3: 局部精煉 ({phase3_budget} 次)")
        
        if self.best_params:
            for i in range(phase3_budget):
                # 在最佳點附近搜索
                perturbed_params = self._local_perturbation(self.best_params)
                score = self.objective_function(perturbed_params)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = perturbed_params
                    print(f"   ✓ 精煉改進: score={score:.4f}")
        
        return self._compile_results()
    
    def _update_gp_model(self, params: Dict[str, float], score: float):
        """更新高斯過程模型"""
        # 轉換為數組
        x = np.array([
            params['lambda_crps'],
            params['lambda_under'],
            params['lambda_over'],
            params['epsilon'],
            params['spatial_corr']
        ])
        
        self.X_observed.append(x)
        self.y_observed.append(score)
        
        # 記錄歷史
        self.history.append({'params': params, 'score': score})
        
        # 更新最佳
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
        
        # 訓練GP
        if len(self.X_observed) > 1:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp_model.fit(X, y)
    
    def _suggest_next_point(self) -> Dict[str, float]:
        """使用獲取函數建議下一個點"""
        if len(self.X_observed) < 2:
            return self.search_space.sample_random_point()
        
        # 使用Expected Improvement獲取函數
        best_score = np.max(self.y_observed)
        
        def acquisition_function(x):
            mu, sigma = self.gp_model.predict(x.reshape(1, -1), return_std=True)
            
            # Expected Improvement
            if sigma > 0:
                Z = (mu - best_score - 0.01) / sigma
                ei = sigma * (Z * self._norm_cdf(Z) + self._norm_pdf(Z))
            else:
                ei = 0
            
            return -ei[0]  # 最小化
        
        # 優化獲取函數
        result = minimize(
            acquisition_function,
            x0=self.X_observed[-1],
            bounds=self.search_space.get_bounds(),
            method='L-BFGS-B'
        )
        
        return {
            'lambda_crps': result.x[0],
            'lambda_under': result.x[1],
            'lambda_over': result.x[2],
            'epsilon': result.x[3],
            'spatial_corr': result.x[4]
        }
    
    def _local_perturbation(self, params: Dict[str, float], scale: float = 0.1) -> Dict[str, float]:
        """局部擾動"""
        perturbed = {}
        for key, value in params.items():
            if key == 'lambda_crps':
                # 對數尺度擾動
                log_value = np.log10(value)
                perturbed_log = log_value + np.random.normal(0, scale)
                perturbed[key] = 10 ** perturbed_log
            else:
                # 線性擾動
                perturbed[key] = value * (1 + np.random.normal(0, scale))
        
        # 確保在邊界內
        perturbed['lambda_crps'] = np.clip(
            perturbed['lambda_crps'], 
            *self.search_space.lambda_crps_range
        )
        perturbed['lambda_under'] = np.clip(
            perturbed['lambda_under'],
            *self.search_space.lambda_under_range
        )
        perturbed['lambda_over'] = np.clip(
            perturbed['lambda_over'],
            *self.search_space.lambda_over_range
        )
        
        return perturbed
    
    def _norm_cdf(self, z):
        """標準正態CDF"""
        from scipy.stats import norm
        return norm.cdf(z)
    
    def _norm_pdf(self, z):
        """標準正態PDF"""
        from scipy.stats import norm
        return norm.pdf(z)
    
    def _compile_results(self) -> Dict[str, Any]:
        """編譯優化結果"""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_evaluations': len(self.history),
            'history': self.history,
            'convergence_curve': [h['score'] for h in self.history],
            'strategy': self.strategy
        }
    
    def plot_convergence(self):
        """繪製收斂曲線"""
        try:
            import matplotlib.pyplot as plt
            
            scores = [h['score'] for h in self.history]
            best_scores = np.maximum.accumulate(scores)
            
            plt.figure(figsize=(10, 6))
            plt.plot(scores, 'o-', alpha=0.3, label='Score')
            plt.plot(best_scores, 'r-', linewidth=2, label='Best Score')
            plt.xlabel('Iteration')
            plt.ylabel('Score')
            plt.title(f'Hyperparameter Optimization Convergence ({self.strategy})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")

class CrossValidatedHyperparameterSearch:
    """
    交叉驗證的超參數搜索
    
    使用k-fold交叉驗證確保超參數的穩健性
    """
    
    def __init__(self,
                 base_optimizer: AdaptiveHyperparameterOptimizer,
                 n_folds: int = 5):
        """
        初始化交叉驗證搜索
        
        Parameters:
        -----------
        base_optimizer : AdaptiveHyperparameterOptimizer
            基礎優化器
        n_folds : int
            交叉驗證折數
        """
        self.base_optimizer = base_optimizer
        self.n_folds = n_folds
        self.cv_results = []
    
    def search(self, 
              data: np.ndarray,
              labels: np.ndarray,
              n_iterations: int = 50) -> Dict[str, Any]:
        """
        執行交叉驗證搜索
        
        Parameters:
        -----------
        data : np.ndarray
            特徵數據
        labels : np.ndarray
            標籤數據
        n_iterations : int
            每折的優化迭代次數
            
        Returns:
        --------
        Dict[str, Any]
            最佳超參數和CV結果
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        print(f"🔄 執行{self.n_folds}折交叉驗證超參數搜索...")
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            print(f"\n折 {fold+1}/{self.n_folds}:")
            
            # 分割數據
            X_train, X_val = data[train_idx], data[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # 定義本折的目標函數
            def fold_objective(params):
                # 這裡應該調用實際的模型訓練和評估
                # 返回驗證集上的分數
                pass  # 實際實現需要根據具體模型
            
            # 優化
            self.base_optimizer.objective_function = fold_objective
            result = self.base_optimizer.optimize(n_iterations)
            
            fold_results.append(result)
        
        # 彙總結果
        self.cv_results = fold_results
        
        # 選擇最穩健的超參數（平均分數最高）
        avg_scores = {}
        for params_key in fold_results[0]['best_params'].keys():
            values = [r['best_params'][params_key] for r in fold_results]
            avg_scores[params_key] = np.mean(values)
        
        return {
            'best_params': avg_scores,
            'cv_scores': [r['best_score'] for r in fold_results],
            'mean_score': np.mean([r['best_score'] for r in fold_results]),
            'std_score': np.std([r['best_score'] for r in fold_results]),
            'fold_results': fold_results
        }