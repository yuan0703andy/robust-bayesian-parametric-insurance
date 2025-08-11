#!/usr/bin/env python3
"""
計算效率優化器 (Computational Efficiency Optimizer)
大幅提升參數保險分析的計算性能

本模組回應考量點四：計算效率 (Computational Efficiency)
- 使用 NumPy 向量化操作取代多重迴圈
- 實施矩陣/向量運算來批量計算基差風險
- 優化記憶體使用和數據結構
- 提供性能基準測試和比較

核心優化策略：
1. 向量化基差風險計算 (Vectorized Basis Risk Computation)
2. 批量產品評估 (Batch Product Evaluation)
3. 並行化 Monte Carlo 模擬 (Parallel Monte Carlo Simulation)
4. 快取和記憶體優化 (Caching and Memory Optimization)
5. 稀疏矩陣運算 (Sparse Matrix Operations)

性能目標：實現 10-100x 的速度提升

Author: Research Team
Date: 2025-01-10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
from pathlib import Path
import time
from numba import jit, vectorize, cuda
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from scipy.sparse import csr_matrix
from functools import lru_cache

# 導入基礎模組
from skill_scores.basis_risk_functions import (
    BasisRiskCalculator, 
    BasisRiskType
)

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

@dataclass
class PerformanceConfig:
    """性能優化配置"""
    use_vectorization: bool = True      # 啟用向量化
    use_numba: bool = True              # 啟用 Numba JIT 編譯
    use_multiprocessing: bool = True    # 啟用多進程
    use_caching: bool = True            # 啟用快取
    n_workers: int = 4                  # 工作進程數
    chunk_size: int = 1000              # 批次大小
    memory_limit: str = "4GB"           # 記憶體限制

@dataclass
class BenchmarkResult:
    """基準測試結果"""
    method_name: str
    execution_time: float
    memory_usage: float
    speedup_ratio: float
    accuracy_loss: float
    
class VectorizedBasisRiskCalculator:
    """向量化基差風險計算器"""
    
    def __init__(self, config: PerformanceConfig):
        """
        初始化向量化計算器
        
        Parameters:
        -----------
        config : PerformanceConfig
            性能配置
        """
        self.config = config
    
    @staticmethod
    @jit(nopython=True, parallel=True)  # Numba 加速
    def _vectorized_weighted_asymmetric_risk(actual_losses: np.ndarray,
                                           payouts: np.ndarray,
                                           w_under: float = 2.0,
                                           w_over: float = 0.5) -> np.ndarray:
        """
        向量化計算加權不對稱基差風險
        
        Parameters:
        -----------
        actual_losses : np.ndarray
            實際損失向量
        payouts : np.ndarray
            賠付向量
        w_under, w_over : float
            權重參數
            
        Returns:
        --------
        np.ndarray
            基差風險向量
        """
        
        # 向量化計算不足覆蓋和過度覆蓋
        under_coverage = np.maximum(0, actual_losses - payouts)
        over_coverage = np.maximum(0, payouts - actual_losses)
        
        # 向量化加權計算
        weighted_risks = w_under * under_coverage + w_over * over_coverage
        
        return weighted_risks
    
    def calculate_batch_basis_risk(self,
                                 actual_losses: np.ndarray,
                                 hazard_indices: np.ndarray,
                                 trigger_thresholds: np.ndarray,
                                 payout_amounts: np.ndarray,
                                 w_under: float = 2.0,
                                 w_over: float = 0.5) -> np.ndarray:
        """
        批量計算多個產品的基差風險
        
        取代三重迴圈：
        原始: for product in products:
                for sample in posterior_samples:
                  for event in events:
        
        優化: 使用廣播和向量化一次計算所有組合
        
        Parameters:
        -----------
        actual_losses : np.ndarray
            實際損失 (n_scenarios,)
        hazard_indices : np.ndarray
            災害指標 (n_scenarios,)
        trigger_thresholds : np.ndarray
            觸發閾值 (n_products,)
        payout_amounts : np.ndarray
            賠付金額 (n_products,)
        w_under, w_over : float
            權重參數
            
        Returns:
        --------
        np.ndarray
            基差風險矩陣 (n_products, n_scenarios)
        """
        
        n_products = len(trigger_thresholds)
        n_scenarios = len(actual_losses)
        
        # 使用廣播創建賠付矩陣 (避免顯式迴圈)
        # shape: (n_products, n_scenarios)
        hazard_matrix = hazard_indices[np.newaxis, :]  # (1, n_scenarios)
        trigger_matrix = trigger_thresholds[:, np.newaxis]  # (n_products, 1)
        payout_matrix = payout_amounts[:, np.newaxis]  # (n_products, 1)
        
        # 向量化計算觸發條件 (廣播比較)
        triggered = hazard_matrix >= trigger_matrix  # (n_products, n_scenarios)
        
        # 向量化計算賠付 (條件賠付)
        payouts_matrix = np.where(triggered, payout_matrix, 0)  # (n_products, n_scenarios)
        
        # 廣播實際損失
        losses_matrix = actual_losses[np.newaxis, :]  # (1, n_scenarios)
        losses_matrix = np.broadcast_to(losses_matrix, (n_products, n_scenarios))
        
        # 向量化計算基差風險
        if self.config.use_numba:
            # 使用 Numba 加速的版本
            basis_risks = np.zeros((n_products, n_scenarios))
            for i in range(n_products):
                basis_risks[i, :] = self._vectorized_weighted_asymmetric_risk(
                    losses_matrix[i, :], payouts_matrix[i, :], w_under, w_over
                )
        else:
            # 純 NumPy 版本
            under_coverage = np.maximum(0, losses_matrix - payouts_matrix)
            over_coverage = np.maximum(0, payouts_matrix - losses_matrix)
            basis_risks = w_under * under_coverage + w_over * over_coverage
        
        return basis_risks
    
    def calculate_portfolio_basis_risk_vectorized(self,
                                                actual_losses: np.ndarray,
                                                hazard_indices: np.ndarray, 
                                                product_parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        向量化計算投資組合基差風險統計
        
        Parameters:
        -----------
        actual_losses : np.ndarray
            實際損失
        hazard_indices : np.ndarray
            災害指標
        product_parameters : Dict[str, np.ndarray]
            產品參數字典 {'trigger_thresholds': array, 'payout_amounts': array}
            
        Returns:
        --------
        Dict[str, np.ndarray]
            統計結果字典
        """
        
        # 批量計算基差風險矩陣
        basis_risk_matrix = self.calculate_batch_basis_risk(
            actual_losses,
            hazard_indices,
            product_parameters['trigger_thresholds'],
            product_parameters['payout_amounts']
        )
        
        # 向量化計算統計量
        stats = {
            'mean_basis_risk': np.mean(basis_risk_matrix, axis=1),      # (n_products,)
            'std_basis_risk': np.std(basis_risk_matrix, axis=1),        # (n_products,)
            'max_basis_risk': np.max(basis_risk_matrix, axis=1),        # (n_products,)
            'min_basis_risk': np.min(basis_risk_matrix, axis=1),        # (n_products,)
            'median_basis_risk': np.median(basis_risk_matrix, axis=1),  # (n_products,)
            'percentile_95': np.percentile(basis_risk_matrix, 95, axis=1),  # (n_products,)
            'percentile_5': np.percentile(basis_risk_matrix, 5, axis=1)     # (n_products,)
        }
        
        return stats

class ParallelMonteCarloEngine:
    """並行 Monte Carlo 模擬引擎"""
    
    def __init__(self, config: PerformanceConfig):
        """
        初始化並行引擎
        
        Parameters:
        -----------
        config : PerformanceConfig
            性能配置
        """
        self.config = config
        self.n_workers = config.n_workers or mp.cpu_count()
    
    def parallel_posterior_sampling(self,
                                  sample_function: Any,
                                  n_samples: int,
                                  **kwargs) -> np.ndarray:
        """
        並行後驗樣本生成
        
        Parameters:
        -----------
        sample_function : Callable
            採樣函數
        n_samples : int
            樣本數量
        **kwargs : dict
            採樣函數參數
            
        Returns:
        --------
        np.ndarray
            後驗樣本
        """
        
        if not self.config.use_multiprocessing or n_samples < 100:
            # 小樣本時不使用並行
            return sample_function(n_samples, **kwargs)
        
        # 將任務分配給多個進程
        chunk_size = max(1, n_samples // self.n_workers)
        chunks = [chunk_size] * (self.n_workers - 1)
        chunks.append(n_samples - sum(chunks))  # 最後一塊包含餘數
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for chunk in chunks:
                if chunk > 0:
                    future = executor.submit(sample_function, chunk, **kwargs)
                    futures.append(future)
            
            # 收集結果
            results = []
            for future in futures:
                results.append(future.result())
        
        # 合併結果
        return np.concatenate(results, axis=0)
    
    def parallel_basis_risk_computation(self,
                                      actual_losses: np.ndarray,
                                      hazard_indices: np.ndarray,
                                      product_parameters: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        並行計算多個產品的基差風險
        
        Parameters:
        -----------
        actual_losses : np.ndarray
            實際損失
        hazard_indices : np.ndarray
            災害指標
        product_parameters : List[Dict[str, float]]
            產品參數列表
            
        Returns:
        --------
        List[Dict[str, float]]
            基差風險結果
        """
        
        if not self.config.use_multiprocessing or len(product_parameters) < 10:
            # 少量產品時不使用並行
            calc = VectorizedBasisRiskCalculator(self.config)
            results = []
            
            for params in product_parameters:
                triggers = np.array([params['trigger_threshold']])
                payouts = np.array([params['payout_amount']])
                
                stats = calc.calculate_portfolio_basis_risk_vectorized(
                    actual_losses, hazard_indices,
                    {'trigger_thresholds': triggers, 'payout_amounts': payouts}
                )
                
                results.append({
                    'mean_basis_risk': stats['mean_basis_risk'][0],
                    'std_basis_risk': stats['std_basis_risk'][0],
                    'trigger_rate': np.mean(hazard_indices >= params['trigger_threshold'])
                })
            
            return results
        
        # 使用並行處理
        def compute_product_risk(params):
            calc = VectorizedBasisRiskCalculator(self.config)
            triggers = np.array([params['trigger_threshold']])
            payouts = np.array([params['payout_amount']])
            
            stats = calc.calculate_portfolio_basis_risk_vectorized(
                actual_losses, hazard_indices,
                {'trigger_thresholds': triggers, 'payout_amounts': payouts}
            )
            
            return {
                'mean_basis_risk': stats['mean_basis_risk'][0],
                'std_basis_risk': stats['std_basis_risk'][0],
                'trigger_rate': np.mean(hazard_indices >= params['trigger_threshold'])
            }
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(compute_product_risk, product_parameters))
        
        return results

class MemoryOptimizedCache:
    """記憶體優化快取系統"""
    
    def __init__(self, config: PerformanceConfig):
        """
        初始化快取系統
        
        Parameters:
        -----------
        config : PerformanceConfig
            性能配置
        """
        self.config = config
        self.cache = {}
        self.access_count = {}
        self.max_cache_size = 1000  # 最大快取條目數
    
    @lru_cache(maxsize=128)
    def cached_basis_risk_calculation(self,
                                    trigger_threshold: float,
                                    payout_amount: float,
                                    losses_hash: int,
                                    indices_hash: int) -> float:
        """
        帶快取的基差風險計算
        
        Parameters:
        -----------
        trigger_threshold, payout_amount : float
            產品參數
        losses_hash, indices_hash : int
            數據雜湊值 (用於快取鍵)
            
        Returns:
        --------
        float
            基差風險
        """
        
        # 這裡應該包含實際的計算邏輯
        # 由於輸入是雜湊值，實際實現中需要重新設計
        cache_key = (trigger_threshold, payout_amount, losses_hash, indices_hash)
        
        if cache_key in self.cache:
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            return self.cache[cache_key]
        
        # 模擬計算 (實際應用中替換為真實計算)
        result = trigger_threshold * payout_amount * 0.001
        
        # 存入快取
        if len(self.cache) >= self.max_cache_size:
            self._evict_least_used()
        
        self.cache[cache_key] = result
        self.access_count[cache_key] = 1
        
        return result
    
    def _evict_least_used(self):
        """移除最少使用的快取條目"""
        if not self.access_count:
            return
        
        # 找出使用次數最少的條目
        least_used_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        
        del self.cache[least_used_key]
        del self.access_count[least_used_key]

class SparseMatrixOptimizer:
    """稀疏矩陣優化器"""
    
    def __init__(self, config: PerformanceConfig):
        """
        初始化稀疏矩陣優化器
        
        Parameters:
        -----------
        config : PerformanceConfig
            性能配置
        """
        self.config = config
    
    def create_sparse_payout_matrix(self,
                                  hazard_indices: np.ndarray,
                                  trigger_thresholds: np.ndarray,
                                  payout_amounts: np.ndarray) -> csr_matrix:
        """
        創建稀疏賠付矩陣
        
        對於大多數災害事件，多數產品不會觸發，因此矩陣非常稀疏
        使用 CSR (Compressed Sparse Row) 格式可大幅節省記憶體
        
        Parameters:
        -----------
        hazard_indices : np.ndarray
            災害指標 (n_scenarios,)
        trigger_thresholds : np.ndarray  
            觸發閾值 (n_products,)
        payout_amounts : np.ndarray
            賠付金額 (n_products,)
            
        Returns:
        --------
        csr_matrix
            稀疏賠付矩陣 (n_products, n_scenarios)
        """
        
        n_products = len(trigger_thresholds)
        n_scenarios = len(hazard_indices)
        
        # 找出所有觸發的 (product, scenario) 對
        rows, cols, data = [], [], []
        
        for i, (trigger, payout) in enumerate(zip(trigger_thresholds, payout_amounts)):
            # 找出觸發此產品的情境
            triggered_scenarios = np.where(hazard_indices >= trigger)[0]
            
            # 添加到稀疏矩陣數據
            rows.extend([i] * len(triggered_scenarios))
            cols.extend(triggered_scenarios)
            data.extend([payout] * len(triggered_scenarios))
        
        # 創建 CSR 稀疏矩陣
        sparse_payouts = csr_matrix(
            (data, (rows, cols)), 
            shape=(n_products, n_scenarios)
        )
        
        return sparse_payouts
    
    def sparse_basis_risk_calculation(self,
                                    actual_losses: np.ndarray,
                                    sparse_payouts: csr_matrix,
                                    w_under: float = 2.0,
                                    w_over: float = 0.5) -> np.ndarray:
        """
        使用稀疏矩陣計算基差風險
        
        Parameters:
        -----------
        actual_losses : np.ndarray
            實際損失
        sparse_payouts : csr_matrix
            稀疏賠付矩陣
        w_under, w_over : float
            權重參數
            
        Returns:
        --------
        np.ndarray
            平均基差風險 (n_products,)
        """
        
        n_products, n_scenarios = sparse_payouts.shape
        
        # 將稀疏矩陣轉換為密集矩陣進行計算 (小規模時)
        # 大規模時需要更複雜的稀疏運算
        if n_products * n_scenarios < 1e6:
            payouts_dense = sparse_payouts.toarray()
            
            # 廣播實際損失
            losses_matrix = np.broadcast_to(
                actual_losses[np.newaxis, :], (n_products, n_scenarios)
            )
            
            # 向量化計算基差風險
            under_coverage = np.maximum(0, losses_matrix - payouts_dense)
            over_coverage = np.maximum(0, payouts_dense - losses_matrix)
            
            basis_risks = w_under * under_coverage + w_over * over_coverage
            return np.mean(basis_risks, axis=1)
        
        else:
            # 大規模稀疏矩陣運算 (可進一步優化)
            mean_risks = np.zeros(n_products)
            
            for i in range(n_products):
                product_payouts = sparse_payouts.getrow(i).toarray().flatten()
                
                under_coverage = np.maximum(0, actual_losses - product_payouts)
                over_coverage = np.maximum(0, product_payouts - actual_losses)
                
                mean_risks[i] = np.mean(w_under * under_coverage + w_over * over_coverage)
            
            return mean_risks

class PerformanceBenchmarker:
    """性能基準測試器"""
    
    def __init__(self):
        """初始化基準測試器"""
        self.results = []
    
    def benchmark_calculation_methods(self,
                                    actual_losses: np.ndarray,
                                    hazard_indices: np.ndarray,
                                    product_parameters: Dict[str, np.ndarray],
                                    n_runs: int = 5) -> List[BenchmarkResult]:
        """
        基準測試不同計算方法的性能
        
        Parameters:
        -----------
        actual_losses : np.ndarray
            實際損失
        hazard_indices : np.ndarray
            災害指標
        product_parameters : Dict[str, np.ndarray]
            產品參數
        n_runs : int
            測試運行次數
            
        Returns:
        --------
        List[BenchmarkResult]
            基準測試結果
        """
        
        print("🚀 執行性能基準測試...")
        
        methods = {
            'naive_loops': self._naive_loop_method,
            'vectorized': self._vectorized_method,
            'vectorized_numba': self._vectorized_numba_method,
            'sparse_matrix': self._sparse_matrix_method
        }
        
        baseline_time = None
        results = []
        
        for method_name, method_func in methods.items():
            print(f"  測試方法: {method_name}")
            
            # 多次運行取平均
            times = []
            memory_usages = []
            
            for run in range(n_runs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = method_func(actual_losses, hazard_indices, product_parameters)
                    
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    execution_time = end_time - start_time
                    memory_usage = end_memory - start_memory
                    
                    times.append(execution_time)
                    memory_usages.append(memory_usage)
                    
                except Exception as e:
                    print(f"    方法 {method_name} 執行失敗: {e}")
                    continue
            
            if times:
                avg_time = np.mean(times)
                avg_memory = np.mean(memory_usages)
                
                # 計算加速比
                if baseline_time is None:
                    baseline_time = avg_time
                    speedup_ratio = 1.0
                else:
                    speedup_ratio = baseline_time / avg_time
                
                benchmark_result = BenchmarkResult(
                    method_name=method_name,
                    execution_time=avg_time,
                    memory_usage=avg_memory,
                    speedup_ratio=speedup_ratio,
                    accuracy_loss=0.0  # 簡化，實際需要計算準確度差異
                )
                
                results.append(benchmark_result)
                
                print(f"    平均時間: {avg_time:.4f}s")
                print(f"    記憶體使用: {avg_memory:.2f}MB")
                print(f"    加速比: {speedup_ratio:.2f}x")
        
        self.results = results
        return results
    
    def _naive_loop_method(self,
                          actual_losses: np.ndarray,
                          hazard_indices: np.ndarray,
                          product_parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """原始的多重迴圈方法 (作為基準)"""
        
        triggers = product_parameters['trigger_thresholds']
        payouts = product_parameters['payout_amounts']
        
        n_products = len(triggers)
        n_scenarios = len(actual_losses)
        
        mean_risks = np.zeros(n_products)
        basis_risk_calc = BasisRiskCalculator()
        
        # 三重迴圈 (低效)
        for i in range(n_products):
            risks = []
            for j in range(n_scenarios):
                # 計算賠付
                payout = payouts[i] if hazard_indices[j] >= triggers[i] else 0.0
                
                # 計算基差風險
                risk = basis_risk_calc.calculate_weighted_asymmetric_basis_risk(
                    actual_losses[j], payout, w_under=2.0, w_over=0.5
                )
                risks.append(risk)
            
            mean_risks[i] = np.mean(risks)
        
        return {'mean_basis_risk': mean_risks}
    
    def _vectorized_method(self,
                          actual_losses: np.ndarray,
                          hazard_indices: np.ndarray,
                          product_parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """向量化方法"""
        
        config = PerformanceConfig(use_numba=False)
        calc = VectorizedBasisRiskCalculator(config)
        
        return calc.calculate_portfolio_basis_risk_vectorized(
            actual_losses, hazard_indices, product_parameters
        )
    
    def _vectorized_numba_method(self,
                                actual_losses: np.ndarray,
                                hazard_indices: np.ndarray,
                                product_parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """向量化 + Numba 方法"""
        
        config = PerformanceConfig(use_numba=True)
        calc = VectorizedBasisRiskCalculator(config)
        
        return calc.calculate_portfolio_basis_risk_vectorized(
            actual_losses, hazard_indices, product_parameters
        )
    
    def _sparse_matrix_method(self,
                             actual_losses: np.ndarray,
                             hazard_indices: np.ndarray,
                             product_parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """稀疏矩陣方法"""
        
        config = PerformanceConfig()
        optimizer = SparseMatrixOptimizer(config)
        
        # 創建稀疏賠付矩陣
        sparse_payouts = optimizer.create_sparse_payout_matrix(
            hazard_indices,
            product_parameters['trigger_thresholds'],
            product_parameters['payout_amounts']
        )
        
        # 計算基差風險
        mean_risks = optimizer.sparse_basis_risk_calculation(
            actual_losses, sparse_payouts
        )
        
        return {'mean_basis_risk': mean_risks}
    
    def _get_memory_usage(self) -> float:
        """獲取當前記憶體使用量 (MB)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0  # 如果沒有 psutil，返回 0
    
    def visualize_benchmark_results(self, output_dir: str = "results") -> None:
        """視覺化基準測試結果"""
        
        if not self.results:
            print("無基準測試結果可視覺化")
            return
        
        print("📊 生成性能基準測試視覺化...")
        
        # 準備數據
        methods = [r.method_name for r in self.results]
        times = [r.execution_time for r in self.results]
        speedups = [r.speedup_ratio for r in self.results]
        memories = [r.memory_usage for r in self.results]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('計算效率優化性能基準測試', fontsize=16, fontweight='bold')
        
        # 1. 執行時間比較
        ax1 = axes[0]
        bars1 = ax1.bar(methods, times, color=['red', 'blue', 'green', 'orange'])
        ax1.set_ylabel('執行時間 (秒)')
        ax1.set_title('執行時間比較')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加數值標籤
        for bar, time_val in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # 2. 加速比比較
        ax2 = axes[1]
        bars2 = ax2.bar(methods, speedups, color=['red', 'blue', 'green', 'orange'])
        ax2.set_ylabel('加速比 (倍)')
        ax2.set_title('相對加速比')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # 添加數值標籤
        for bar, speedup in zip(bars2, speedups):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup:.1f}x', ha='center', va='bottom')
        
        # 3. 記憶體使用比較
        ax3 = axes[2]
        bars3 = ax3.bar(methods, memories, color=['red', 'blue', 'green', 'orange'])
        ax3.set_ylabel('記憶體使用 (MB)')
        ax3.set_title('記憶體使用比較')
        ax3.tick_params(axis='x', rotation=45)
        
        # 添加數值標籤
        for bar, memory in zip(bars3, memories):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{memory:.1f}MB', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存圖表
        Path(output_dir).mkdir(exist_ok=True)
        output_file = Path(output_dir) / "computational_efficiency_benchmark.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 性能基準測試視覺化已保存: {output_file}")

class ComputationalEfficiencyFramework:
    """計算效率優化整合框架"""
    
    def __init__(self, config: PerformanceConfig = None):
        """
        初始化計算效率框架
        
        Parameters:
        -----------
        config : PerformanceConfig
            性能配置
        """
        self.config = config or PerformanceConfig()
        
        # 初始化組件
        self.vectorized_calc = VectorizedBasisRiskCalculator(self.config)
        self.parallel_engine = ParallelMonteCarloEngine(self.config)
        self.cache = MemoryOptimizedCache(self.config)
        self.sparse_optimizer = SparseMatrixOptimizer(self.config)
        self.benchmarker = PerformanceBenchmarker()
    
    def optimize_parametric_insurance_analysis(self,
                                             actual_losses: np.ndarray,
                                             hazard_indices: np.ndarray,
                                             n_products: int = 1000,
                                             run_benchmark: bool = True) -> Dict[str, Any]:
        """
        執行優化的參數保險分析
        
        Parameters:
        -----------
        actual_losses : np.ndarray
            實際損失數據
        hazard_indices : np.ndarray
            災害指標數據
        n_products : int
            分析的產品數量
        run_benchmark : bool
            是否執行性能基準測試
            
        Returns:
        --------
        Dict[str, Any]
            優化分析結果
        """
        
        print("⚡ 執行優化的參數保險分析...")
        print("=" * 60)
        
        # 生成產品參數
        print(f"📦 生成 {n_products} 個候選產品...")
        
        np.random.seed(42)
        trigger_range = (np.percentile(hazard_indices, 60), np.percentile(hazard_indices, 95))
        payout_range = (np.percentile(actual_losses, 20), np.percentile(actual_losses, 80))
        
        trigger_thresholds = np.random.uniform(trigger_range[0], trigger_range[1], n_products)
        payout_amounts = np.random.uniform(payout_range[0], payout_range[1], n_products)
        
        product_parameters = {
            'trigger_thresholds': trigger_thresholds,
            'payout_amounts': payout_amounts
        }
        
        # 執行基準測試
        benchmark_results = None
        if run_benchmark:
            print("🚀 執行性能基準測試...")
            
            # 使用較小的子集進行基準測試
            test_size = min(100, n_products)
            test_params = {
                'trigger_thresholds': trigger_thresholds[:test_size],
                'payout_amounts': payout_amounts[:test_size]
            }
            
            benchmark_results = self.benchmarker.benchmark_calculation_methods(
                actual_losses, hazard_indices, test_params
            )
            
            # 視覺化基準測試結果
            self.benchmarker.visualize_benchmark_results()
        
        # 執行優化分析
        print("⚡ 執行向量化優化分析...")
        start_time = time.time()
        
        optimized_stats = self.vectorized_calc.calculate_portfolio_basis_risk_vectorized(
            actual_losses, hazard_indices, product_parameters
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 找出最佳產品
        best_product_idx = np.argmin(optimized_stats['mean_basis_risk'])
        
        results = {
            'analysis_summary': {
                'n_products_analyzed': n_products,
                'n_scenarios': len(actual_losses),
                'total_execution_time': execution_time,
                'products_per_second': n_products / execution_time
            },
            
            'optimal_product': {
                'index': int(best_product_idx),
                'trigger_threshold': trigger_thresholds[best_product_idx],
                'payout_amount': payout_amounts[best_product_idx],
                'mean_basis_risk': optimized_stats['mean_basis_risk'][best_product_idx],
                'std_basis_risk': optimized_stats['std_basis_risk'][best_product_idx],
                'max_basis_risk': optimized_stats['max_basis_risk'][best_product_idx]
            },
            
            'performance_statistics': optimized_stats,
            'benchmark_results': benchmark_results
        }
        
        print(f"✅ 優化分析完成!")
        print(f"   分析產品數: {n_products:,}")
        print(f"   執行時間: {execution_time:.3f} 秒")
        print(f"   分析速度: {n_products/execution_time:.0f} 產品/秒")
        print(f"   最佳產品基差風險: {results['optimal_product']['mean_basis_risk']:.2e}")
        
        return results

def main():
    """主函數示例"""
    
    print("⚡ 計算效率優化框架示例...")
    print("=" * 80)
    
    # 生成大規模測試數據
    print("📊 生成大規模測試數據...")
    
    np.random.seed(42)
    n_scenarios = 5000   # 5000 個損失情境
    n_products = 2000    # 2000 個候選產品
    
    # 生成損失數據 (混合分佈)
    normal_losses = np.random.lognormal(np.log(5e7), 0.8, int(0.8 * n_scenarios))
    extreme_losses = np.random.lognormal(np.log(2e8), 1.0, int(0.2 * n_scenarios))
    actual_losses = np.concatenate([normal_losses, extreme_losses])
    np.random.shuffle(actual_losses)
    
    # 生成災害指標
    hazard_indices = np.random.gamma(2, 25, n_scenarios)
    
    print(f"   損失情境數: {n_scenarios:,}")
    print(f"   候選產品數: {n_products:,}")
    print(f"   計算複雜度: {n_scenarios * n_products:,} 次基差風險計算")
    print(f"   損失範圍: {actual_losses.min():.2e} - {actual_losses.max():.2e}")
    
    # 創建優化框架
    config = PerformanceConfig(
        use_vectorization=True,
        use_numba=True,
        use_multiprocessing=True,
        n_workers=4
    )
    
    framework = ComputationalEfficiencyFramework(config)
    
    # 執行優化分析
    results = framework.optimize_parametric_insurance_analysis(
        actual_losses, hazard_indices, n_products, run_benchmark=True
    )
    
    # 輸出性能總結
    print("\n" + "=" * 80)
    print("🏆 性能優化總結:")
    print("=" * 80)
    
    analysis_summary = results['analysis_summary']
    print(f"✅ 成功分析: {analysis_summary['n_products_analyzed']:,} 個產品")
    print(f"⏱️  總執行時間: {analysis_summary['total_execution_time']:.3f} 秒")
    print(f"🚀 分析速度: {analysis_summary['products_per_second']:.0f} 產品/秒")
    
    # 基準測試結果
    if results['benchmark_results']:
        print(f"\n📊 性能提升:")
        
        baseline = results['benchmark_results'][0]  # naive_loops 方法
        best_optimized = max(results['benchmark_results'], key=lambda x: x.speedup_ratio)
        
        print(f"   基準方法 ({baseline.method_name}): {baseline.execution_time:.3f}s")
        print(f"   最佳優化方法 ({best_optimized.method_name}): {best_optimized.execution_time:.3f}s")
        print(f"   🔥 總加速比: {best_optimized.speedup_ratio:.1f}x")
        
        # 預估大規模分析的時間節省
        full_scale_baseline = baseline.execution_time * (n_products / 100)  # 基準測試只用了100個產品
        full_scale_optimized = best_optimized.execution_time * (n_products / 100)
        time_saved = full_scale_baseline - full_scale_optimized
        
        print(f"   💰 預估全規模分析時間節省: {time_saved:.1f} 秒")
    
    # 最佳產品結果
    optimal = results['optimal_product']
    print(f"\n🎯 最佳產品:")
    print(f"   觸發閾值: {optimal['trigger_threshold']:.2f}")
    print(f"   賠付金額: {optimal['payout_amount']:.2e}")
    print(f"   基差風險: {optimal['mean_basis_risk']:.2e}")
    
    print(f"\n📁 結果已保存在: results/")
    print("🎉 計算效率優化完成！")

if __name__ == "__main__":
    main()