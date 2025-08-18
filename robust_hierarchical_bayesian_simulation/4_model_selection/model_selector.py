#!/usr/bin/env python3
"""
Model Selector with Hyperparameter Optimization
模型海選與超參數優化器

實現雙層循環策略：
- 外層：遍歷所有模型架構 (prior, likelihood, ε)
- 內層：為每個模型尋找最佳超參數 λ

Author: Research Team
Date: 2025-01-17
Version: 2.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import warnings
import time
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging

# Import VI components
from .basis_risk_vi import BasisRiskAwareVI, DifferentiableCRPS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelCandidate:
    """模型候選人配置"""
    model_id: str
    prior_type: str  # 'informative', 'weakly_informative', 'vague'
    likelihood_family: str  # 'normal', 'student_t', 'laplace'
    epsilon_contamination: float  # ε-contamination level
    spatial_effects: bool = False
    hierarchy_levels: int = 2
    
    def __hash__(self):
        return hash(self.model_id)
    
    def __str__(self):
        return f"{self.model_id}: {self.prior_type}_{self.likelihood_family}_ε{self.epsilon_contamination:.2f}"

@dataclass
class HyperparameterConfig:
    """超參數配置"""
    lambda_crps: float  # CRPS權重
    lambda_under: float = 2.0  # 不足覆蓋懲罰
    lambda_over: float = 0.5  # 過度賠付懲罰
    vi_learning_rate: float = 0.01
    vi_iterations: int = 5000
    vi_batch_size: int = 32
    
    def to_dict(self):
        return {
            'λ_crps': self.lambda_crps,
            'λ_under': self.lambda_under,
            'λ_over': self.lambda_over,
            'lr': self.vi_learning_rate,
            'iters': self.vi_iterations
        }

@dataclass
class ModelSelectionResult:
    """模型選擇結果"""
    model: ModelCandidate
    best_hyperparams: HyperparameterConfig
    best_score: float  # L_BR score
    convergence_time: float
    validation_metrics: Dict[str, float]
    posterior_samples: Optional[np.ndarray] = None
    rank: Optional[int] = None
    
    def summary(self) -> Dict[str, Any]:
        return {
            'model_id': self.model.model_id,
            'score': self.best_score,
            'best_λ': self.best_hyperparams.lambda_crps,
            'time': self.convergence_time,
            'rank': self.rank
        }

class ModelSelectorWithHyperparamOptimization:
    """
    模型海選與超參數優化器
    
    實現論文中的雙層循環策略：
    1. 外層：遍歷模型架構
    2. 內層：優化超參數
    """
    
    def __init__(self, 
                 n_jobs: int = 4,
                 verbose: bool = True,
                 save_results: bool = True,
                 output_dir: str = "results/model_selection"):
        """
        初始化模型選擇器
        
        Parameters:
        -----------
        n_jobs : int
            並行處理的工作數
        verbose : bool
            是否顯示詳細信息
        save_results : bool
            是否保存結果
        output_dir : str
            輸出目錄
        """
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.save_results = save_results
        self.output_dir = Path(output_dir)
        
        if save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.leaderboard = {}
        self.full_results = []
        self.optimization_history = []
        
    def generate_model_candidates(self) -> List[ModelCandidate]:
        """
        生成所有模型候選
        
        Returns:
        --------
        List[ModelCandidate]
            模型候選列表
        """
        candidates = []
        
        # Prior types
        priors = ['informative', 'weakly_informative', 'vague']
        
        # Likelihood families
        likelihoods = ['normal', 'student_t', 'laplace']
        
        # Epsilon contamination levels
        epsilons = [0.05, 0.10, 0.15]
        
        # Spatial effects
        spatial_options = [False, True]
        
        model_id = 0
        for prior in priors:
            for likelihood in likelihoods:
                for epsilon in epsilons:
                    for spatial in spatial_options:
                        model_id += 1
                        candidate = ModelCandidate(
                            model_id=f"M{model_id:03d}",
                            prior_type=prior,
                            likelihood_family=likelihood,
                            epsilon_contamination=epsilon,
                            spatial_effects=spatial
                        )
                        candidates.append(candidate)
        
        if self.verbose:
            print(f"📊 生成了 {len(candidates)} 個模型候選")
            
        return candidates
    
    def generate_hyperparameter_grid(self, 
                                    lambda_values: Optional[List[float]] = None) -> List[HyperparameterConfig]:
        """
        生成超參數網格
        
        Parameters:
        -----------
        lambda_values : List[float], optional
            CRPS權重候選值
            
        Returns:
        --------
        List[HyperparameterConfig]
            超參數配置列表
        """
        if lambda_values is None:
            lambda_values = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
        
        configs = []
        
        # 基本網格
        for lambda_crps in lambda_values:
            # 標準配置
            configs.append(HyperparameterConfig(
                lambda_crps=lambda_crps,
                lambda_under=2.0,
                lambda_over=0.5
            ))
            
            # 激進配置（更重視不足覆蓋）
            configs.append(HyperparameterConfig(
                lambda_crps=lambda_crps,
                lambda_under=5.0,
                lambda_over=0.2
            ))
            
            # 保守配置（更重視過度賠付）
            configs.append(HyperparameterConfig(
                lambda_crps=lambda_crps,
                lambda_under=1.0,
                lambda_over=1.0
            ))
        
        if self.verbose:
            print(f"🔧 生成了 {len(configs)} 個超參數配置")
            
        return configs
    
    def run_model_selection(self,
                           data: Dict[str, np.ndarray],
                           candidates: Optional[List[ModelCandidate]] = None,
                           hyperparameter_grid: Optional[List[HyperparameterConfig]] = None,
                           top_k: int = 5) -> List[ModelSelectionResult]:
        """
        執行完整的模型海選與超參數優化
        
        Parameters:
        -----------
        data : Dict[str, np.ndarray]
            包含訓練和驗證數據
        candidates : List[ModelCandidate], optional
            模型候選列表
        hyperparameter_grid : List[HyperparameterConfig], optional
            超參數網格
        top_k : int
            返回前k個最佳模型
            
        Returns:
        --------
        List[ModelSelectionResult]
            前k個最佳模型的結果
        """
        print("🏁 開始模型海選與超參數優化")
        print("=" * 60)
        
        start_time = time.time()
        
        # Generate candidates if not provided
        if candidates is None:
            candidates = self.generate_model_candidates()
        
        # Generate hyperparameter grid if not provided
        if hyperparameter_grid is None:
            hyperparameter_grid = self.generate_hyperparameter_grid()
        
        # Run optimization for each model
        if self.n_jobs > 1:
            results = self._parallel_optimization(data, candidates, hyperparameter_grid)
        else:
            results = self._sequential_optimization(data, candidates, hyperparameter_grid)
        
        # Rank results
        self._rank_results(results)
        
        # Generate leaderboard
        self._generate_leaderboard(results)
        
        # Save results
        if self.save_results:
            self._save_results(results)
        
        total_time = time.time() - start_time
        print(f"\n✅ 模型海選完成！總時間: {total_time:.2f} 秒")
        
        # Return top k models
        return results[:top_k]
    
    def _optimize_single_model(self,
                              model: ModelCandidate,
                              data: Dict[str, np.ndarray],
                              hyperparameter_grid: List[HyperparameterConfig]) -> ModelSelectionResult:
        """
        為單一模型優化超參數
        
        Parameters:
        -----------
        model : ModelCandidate
            模型候選
        data : Dict[str, np.ndarray]
            數據
        hyperparameter_grid : List[HyperparameterConfig]
            超參數網格
            
        Returns:
        --------
        ModelSelectionResult
            優化結果
        """
        if self.verbose:
            print(f"\n🔍 優化模型: {model}")
        
        best_score = -np.inf
        best_config = None
        best_metrics = {}
        
        start_time = time.time()
        
        # Inner loop: hyperparameter optimization
        for config in hyperparameter_grid:
            try:
                # Run VI with current hyperparameters
                score, metrics = self._run_vi_with_hyperparams(model, config, data)
                
                # Track history
                self.optimization_history.append({
                    'model_id': model.model_id,
                    'config': config.to_dict(),
                    'score': score,
                    'metrics': metrics
                })
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_config = config
                    best_metrics = metrics
                    
                    if self.verbose:
                        print(f"   ✓ 新最佳: λ={config.lambda_crps:.1f}, score={score:.4f}")
                        
            except Exception as e:
                logger.warning(f"Failed to optimize {model} with {config}: {e}")
                continue
        
        convergence_time = time.time() - start_time
        
        return ModelSelectionResult(
            model=model,
            best_hyperparams=best_config,
            best_score=best_score,
            convergence_time=convergence_time,
            validation_metrics=best_metrics
        )
    
    def _run_vi_with_hyperparams(self,
                                model: ModelCandidate,
                                config: HyperparameterConfig,
                                data: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """
        使用指定超參數運行VI
        
        Parameters:
        -----------
        model : ModelCandidate
            模型配置
        config : HyperparameterConfig
            超參數配置
        data : Dict[str, np.ndarray]
            數據
            
        Returns:
        --------
        Tuple[float, Dict[str, float]]
            (L_BR分數, 驗證指標)
        """
        # Initialize VI trainer
        vi_trainer = BasisRiskAwareVI(
            prior_type=model.prior_type,
            likelihood_family=model.likelihood_family,
            epsilon_contamination=model.epsilon_contamination,
            lambda_crps=config.lambda_crps,
            lambda_under=config.lambda_under,
            lambda_over=config.lambda_over,
            learning_rate=config.vi_learning_rate,
            n_iterations=config.vi_iterations
        )
        
        # Train
        results = vi_trainer.fit(
            X=data['X_train'],
            y=data['y_train'],
            X_val=data.get('X_val'),
            y_val=data.get('y_val')
        )
        
        # Calculate L_BR score
        L_BR_score = results['final_elbo'] - config.lambda_crps * results['final_crps']
        
        # Validation metrics
        metrics = {
            'elbo': results['final_elbo'],
            'crps': results['final_crps'],
            'kl_divergence': results.get('kl_divergence', 0),
            'basis_risk': results.get('basis_risk', 0)
        }
        
        return L_BR_score, metrics
    
    def _parallel_optimization(self,
                             data: Dict[str, np.ndarray],
                             candidates: List[ModelCandidate],
                             hyperparameter_grid: List[HyperparameterConfig]) -> List[ModelSelectionResult]:
        """並行優化所有模型"""
        print(f"⚡ 使用 {self.n_jobs} 個進程並行優化...")
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for model in candidates:
                future = executor.submit(
                    self._optimize_single_model,
                    model, data, hyperparameter_grid
                )
                futures.append(future)
            
            results = [future.result() for future in futures]
        
        return results
    
    def _sequential_optimization(self,
                                data: Dict[str, np.ndarray],
                                candidates: List[ModelCandidate],
                                hyperparameter_grid: List[HyperparameterConfig]) -> List[ModelSelectionResult]:
        """順序優化所有模型"""
        print("🔄 順序優化模型...")
        
        results = []
        for i, model in enumerate(candidates):
            print(f"\n進度: {i+1}/{len(candidates)}")
            result = self._optimize_single_model(model, data, hyperparameter_grid)
            results.append(result)
        
        return results
    
    def _rank_results(self, results: List[ModelSelectionResult]):
        """為結果排名"""
        # Sort by score (descending)
        results.sort(key=lambda x: x.best_score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1
    
    def _generate_leaderboard(self, results: List[ModelSelectionResult]):
        """生成排行榜"""
        print("\n🏆 模型排行榜")
        print("=" * 60)
        
        leaderboard_data = []
        for result in results[:10]:  # Top 10
            row = {
                'Rank': result.rank,
                'Model': result.model.model_id,
                'Prior': result.model.prior_type,
                'Likelihood': result.model.likelihood_family,
                'ε': result.model.epsilon_contamination,
                'Best λ': result.best_hyperparams.lambda_crps,
                'Score': f"{result.best_score:.4f}",
                'Time': f"{result.convergence_time:.1f}s"
            }
            leaderboard_data.append(row)
        
        df = pd.DataFrame(leaderboard_data)
        print(df.to_string(index=False))
        
        self.leaderboard = df
    
    def _save_results(self, results: List[ModelSelectionResult]):
        """保存結果"""
        # Save full results as JSON
        results_dict = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_models': len(results),
            'results': [r.summary() for r in results],
            'optimization_history': self.optimization_history
        }
        
        with open(self.output_dir / 'model_selection_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # Save leaderboard as CSV
        if hasattr(self, 'leaderboard'):
            self.leaderboard.to_csv(
                self.output_dir / 'leaderboard.csv', 
                index=False
            )
        
        print(f"\n💾 結果已保存至: {self.output_dir}")
    
    def get_top_models_for_mcmc(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        獲取需要MCMC驗證的頂尖模型
        
        Parameters:
        -----------
        top_k : int
            返回前k個模型
            
        Returns:
        --------
        List[Dict[str, Any]]
            頂尖模型配置
        """
        if not self.full_results:
            raise ValueError("需要先運行模型選擇")
        
        top_models = []
        for result in self.full_results[:top_k]:
            config = {
                'model_id': result.model.model_id,
                'model_config': result.model,
                'best_hyperparams': result.best_hyperparams,
                'vi_score': result.best_score,
                'posterior_init': result.posterior_samples  # VI結果作為MCMC初始化
            }
            top_models.append(config)
        
        return top_models