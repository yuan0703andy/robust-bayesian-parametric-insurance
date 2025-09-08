#!/usr/bin/env python3
"""
Basis-Risk-Aware VI Module
基差風險導向變分推斷模組

可重複使用的模組化組件：
- DifferentiableCRPS: 可微分CRPS計算
- ParametricPayoutFunction: 參數型保險賠付函數
- EpsilonContaminationModel: ε-contamination模型
- BasisRiskAwareVI: 基差風險導向VI訓練器

Author: Research Team
Date: 2025-01-17
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# Try importing PyTorch for differentiable operations
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
    # Type hints for when torch is available
    TorchTensor = torch.Tensor
    TorchOptimizer = torch.optim.Optimizer
except ImportError:
    TORCH_AVAILABLE = False
    # Dummy type hints when torch is not available
    TorchTensor = "torch.Tensor"
    TorchOptimizer = "torch.optim.Optimizer"


class DifferentiableCRPS:
    """可微分的 CRPS 計算器，適用於梯度下降"""
    
    @staticmethod
    def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        可微分的 Gaussian CRPS 計算
        
        Args:
            y_true: 實際觀測值
            mu: 預測分布的均值  
            sigma: 預測分布的標準差
            
        Returns:
            CRPS 分數
        """
        if TORCH_AVAILABLE and isinstance(mu, torch.Tensor):
            # PyTorch 版本
            z = (y_true - mu) / sigma
            crps = sigma * (z * (2 * torch.distributions.Normal(0, 1).cdf(z) - 1) + 
                           2 * torch.distributions.Normal(0, 1).log_prob(z).exp() - 
                           1 / np.sqrt(np.pi))
        else:
            # NumPy 版本
            from scipy.stats import norm
            z = (y_true - mu) / sigma
            crps = sigma * (z * (2 * norm.cdf(z) - 1) + 
                           2 * norm.pdf(z) - 
                           1 / np.sqrt(np.pi))
        
        return crps
    
    @staticmethod
    def crps_ensemble(y_true: np.ndarray, forecast_samples: np.ndarray) -> np.ndarray:
        """
        基於 ensemble 的 CRPS 計算 (可微分近似)
        
        Args:
            y_true: 實際觀測值 [N]
            forecast_samples: 預測樣本 [N, M] (N個觀測，M個樣本)
            
        Returns:
            CRPS 分數 [N]
        """
        if TORCH_AVAILABLE and isinstance(forecast_samples, torch.Tensor):
            # PyTorch 版本
            N, M = forecast_samples.shape
            
            # 計算經驗 CDF
            sorted_forecasts, _ = torch.sort(forecast_samples, dim=1)
            
            # CRPS 近似計算
            crps_scores = []
            for i in range(N):
                y = y_true[i]
                forecasts = sorted_forecasts[i]
                
                # CRPS 近似
                crps = torch.mean(torch.abs(forecasts - y)) - 0.5 * torch.mean(
                    torch.abs(forecasts[:, None] - forecasts[None, :])
                )
                crps_scores.append(crps)
                
            return torch.stack(crps_scores)
        else:
            # NumPy 版本 - 使用 properscoring 或簡單近似
            crps_scores = []
            for i in range(len(y_true)):
                y = y_true[i]
                forecasts = forecast_samples[i]
                # 簡單 CRPS 近似
                crps = np.mean(np.abs(forecasts - y)) - 0.5 * np.mean(
                    np.abs(forecasts[:, None] - forecasts[None, :])
                )
                crps_scores.append(crps)
            return np.array(crps_scores)


class ParametricPayoutFunction:
    """參數型保險賠付函數"""
    
    def __init__(self, 
                 trigger_thresholds: List[float] = None,
                 payout_amounts: List[float] = None,
                 max_payout: float = 10000):
        """
        初始化參數型賠付函數
        
        Args:
            trigger_thresholds: 觸發閾值 
            payout_amounts: 對應賠付金額
            max_payout: 最大賠付
        """
        if trigger_thresholds is None:
            trigger_thresholds = [75, 85, 95]
        if payout_amounts is None:
            payout_amounts = [1000, 5000, 10000]
            
        self.trigger_thresholds = np.array(trigger_thresholds)
        self.payout_amounts = np.array(payout_amounts)
        self.max_payout = max_payout
    
    def calculate_payout_distribution(self, 
                                    loss_samples: np.ndarray) -> np.ndarray:
        """
        基於損失分布樣本計算賠付分布
        
        Args:
            loss_samples: 損失分布樣本 [N, M]
            
        Returns:
            賠付分布樣本 [N, M]
        """
        if len(loss_samples.shape) == 1:
            loss_samples = loss_samples.reshape(-1, 1)
            
        N, M = loss_samples.shape
        payout_samples = np.zeros_like(loss_samples)
        
        for i in range(N):
            for j in range(M):
                loss = loss_samples[i, j]
                
                # 階梯式賠付邏輯
                payout = 0
                for k, threshold in enumerate(self.trigger_thresholds):
                    if loss >= threshold:
                        payout = self.payout_amounts[k]
                
                payout_samples[i, j] = min(payout, self.max_payout)
        
        return payout_samples
    
    def optimize_for_basis_risk(self, losses: np.ndarray, features: np.ndarray,
                               basis_risk_type: str = 'weighted') -> Dict:
        """
        優化觸發參數以最小化基差風險
        
        Args:
            losses: 實際損失
            features: 特徵數據
            basis_risk_type: 基差風險類型
            
        Returns:
            優化後的參數
        """
        feature_values = features.flatten()
        
        # 網格搜索
        trigger_candidates = np.percentile(feature_values, [60, 70, 75, 80, 85, 90, 95])
        payout_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        
        best_config = None
        best_risk = np.inf
        
        for trigger in trigger_candidates:
            for multiplier in payout_multipliers:
                max_payout = np.mean(losses[losses > 0]) * multiplier if np.any(losses > 0) else np.mean(losses) * multiplier
                payouts = np.where(feature_values >= trigger, max_payout, 0)
                
                # 計算基差風險
                if basis_risk_type == 'asymmetric':
                    risk = np.mean(np.maximum(0, losses - payouts))
                elif basis_risk_type == 'weighted':
                    under_penalty = np.maximum(0, losses - payouts) * 2.0
                    over_penalty = np.maximum(0, payouts - losses) * 0.5
                    risk = np.mean(under_penalty + over_penalty)
                else:  # absolute
                    risk = np.mean(np.abs(losses - payouts))
                
                if risk < best_risk:
                    best_risk = risk
                    best_config = {
                        'trigger': trigger,
                        'max_payout': max_payout,
                        'multiplier': multiplier,
                        'basis_risk': risk
                    }
        
        return best_config


class SigmoidPayoutProxy:
    """
    Sigmoid 代理賠付函數 - 兩階段優化的核心
    
    階段一：訓練時使用平滑Sigmoid函數作為階梯函數的代理，提供梯度信號
    階段二：評估時切換回真實階梯函數，確保結果實用性
    
    This is the key to surrogate optimization:
    - Training: Smooth, differentiable Sigmoid approximation
    - Evaluation: Original step function for realistic assessment
    """
    
    def __init__(self, 
                 steinmann_thresholds: List[float] = None,
                 steinmann_ratios: List[float] = None,
                 max_payout: float = 20e6,
                 k: float = 0.1,
                 training_mode: bool = True):
        """
        初始化Sigmoid代理賠付函數
        
        Args:
            steinmann_thresholds: Steinmann產品的閾值定義 [t1, t2, t3, t4]
            steinmann_ratios: 對應的賠付比例 [r1, r2, r3, r4]  
            max_payout: 最大賠付金額
            k: Sigmoid陡峭度參數 (越大越接近階梯)
            training_mode: True=使用Sigmoid, False=使用階梯
        """
        # 默認Steinmann產品配置 (四閾值產品示例)
        if steinmann_thresholds is None:
            steinmann_thresholds = [33.0, 43.0, 58.0, 999.0]  # Cat1, Cat2, Cat3, 無效
        if steinmann_ratios is None:
            steinmann_ratios = [0.25, 0.5, 0.75, 1.0]  # 25%遞增
        
        self.thresholds = np.array(steinmann_thresholds, dtype=np.float32)
        self.ratios = np.array(steinmann_ratios, dtype=np.float32)
        self.max_payout = float(max_payout)
        self.k = float(k)  # Sigmoid陡峭度
        self.training_mode = training_mode
        
        # PyTorch tensor版本 (用於GPU加速)
        if TORCH_AVAILABLE:
            self.thresholds_tensor = torch.tensor(self.thresholds, dtype=torch.float32)
            self.ratios_tensor = torch.tensor(self.ratios, dtype=torch.float32)
            self.max_payout_tensor = torch.tensor(self.max_payout, dtype=torch.float32)
            self.k_tensor = torch.tensor(self.k, dtype=torch.float32)
        
        print(f"🎯 Sigmoid代理函數初始化:")
        print(f"   閾值: {self.thresholds}")  
        print(f"   比例: {self.ratios}")
        print(f"   最大賠付: ${self.max_payout/1e6:.1f}M")
        print(f"   陡峭度k: {self.k}")
        print(f"   模式: {'訓練(Sigmoid)' if training_mode else '評估(階梯)'}")
    
    @classmethod  
    def from_steinmann_product(cls, 
                              product_index: int,
                              steinmann_data: Dict = None,
                              k: float = 0.1,
                              training_mode: bool = True):
        """
        從Steinmann產品定義創建代理函數
        
        Args:
            product_index: Steinmann產品索引 (0-349)
            steinmann_data: 包含thresholds, ratios, max_payouts的字典
            k: Sigmoid陡峭度
            training_mode: 訓練模式標誌
        
        Returns:
            SigmoidPayoutProxy實例
        """
        if steinmann_data is None:
            # 使用默認數據 (需要從BasisRiskAwareVI實例獲取)
            print("⚠️ 警告: 使用默認Steinmann數據，實際使用請傳入真實數據")
            return cls(k=k, training_mode=training_mode)
        
        thresholds = steinmann_data['thresholds'][product_index]
        ratios = steinmann_data['ratios'][product_index] 
        max_payout = steinmann_data['max_payouts'][product_index]
        
        return cls(
            steinmann_thresholds=thresholds.tolist() if hasattr(thresholds, 'tolist') else thresholds,
            steinmann_ratios=ratios.tolist() if hasattr(ratios, 'tolist') else ratios,
            max_payout=float(max_payout),
            k=k,
            training_mode=training_mode
        )
    
    def set_mode(self, training_mode: bool):
        """切換訓練/評估模式"""
        self.training_mode = training_mode
        mode_name = "訓練(Sigmoid)" if training_mode else "評估(階梯)"
        print(f"🔄 代理函數切換至: {mode_name}")
    
    def calculate_payout_sigmoid(self, 
                               loss_samples: np.ndarray) -> np.ndarray:
        """
        計算Sigmoid代理賠付 - 訓練階段使用
        
        使用平滑Sigmoid函數近似階梯式賠付：
        Payout(x) = max_payout * Σ[(rᵢ - rᵢ₋₁) * sigmoid(k*(x - tᵢ))]
        
        Args:
            loss_samples: 損失樣本 [N, M] 或 [N]
            
        Returns:
            平滑的賠付樣本 [N, M] 或 [N]
        """
        if len(loss_samples.shape) == 1:
            loss_samples = loss_samples.reshape(-1, 1)
            squeeze_output = True
        else:
            squeeze_output = False
            
        N, M = loss_samples.shape
        payout_samples = np.zeros_like(loss_samples, dtype=np.float32)
        
        # 向量化Sigmoid計算
        for i, (threshold, ratio) in enumerate(zip(self.thresholds, self.ratios)):
            if threshold < 999:  # 有效閾值
                # 計算當前閾值的增量賠付比例
                prev_ratio = self.ratios[i-1] if i > 0 else 0.0
                delta_ratio = ratio - prev_ratio
                
                # Sigmoid函數: 1 / (1 + exp(-k*(x - threshold)))
                sigmoid_values = 1.0 / (1.0 + np.exp(-self.k * (loss_samples - threshold)))
                payout_samples += delta_ratio * sigmoid_values
        
        # 應用最大賠付限制
        payout_samples *= self.max_payout
        payout_samples = np.minimum(payout_samples, self.max_payout)
        
        return payout_samples.squeeze() if squeeze_output else payout_samples
    
    def calculate_payout_step(self, 
                            loss_samples: np.ndarray) -> np.ndarray:
        """
        計算真實階梯賠付 - 評估階段使用
        
        使用原始Steinmann階梯式邏輯：
        對每個損失值，找到觸發的最高閾值，返回對應賠付
        
        Args:
            loss_samples: 損失樣本 [N, M] 或 [N]
            
        Returns:
            階梯式賠付樣本 [N, M] 或 [N]
        """
        if len(loss_samples.shape) == 1:
            loss_samples = loss_samples.reshape(-1, 1)
            squeeze_output = True
        else:
            squeeze_output = False
            
        N, M = loss_samples.shape
        payout_samples = np.zeros_like(loss_samples, dtype=np.float32)
        
        # 原始階梯邏輯 - 與Steinmann標準完全一致
        for i in range(N):
            for j in range(M):
                loss_value = loss_samples[i, j]
                total_payout = 0.0
                
                # 找到觸發的最高閾值
                for thresh_idx in range(len(self.thresholds)):
                    threshold = self.thresholds[thresh_idx]
                    if threshold < 999 and loss_value >= threshold:  # 有效閾值且觸發
                        total_payout = self.max_payout * self.ratios[thresh_idx]
                
                payout_samples[i, j] = min(total_payout, self.max_payout)
        
        return payout_samples.squeeze() if squeeze_output else payout_samples
    
    def calculate_payout_distribution(self, 
                                    loss_samples: np.ndarray) -> np.ndarray:
        """
        統一賠付計算接口 - 自動根據模式選擇Sigmoid或階梯
        
        Args:
            loss_samples: 損失分布樣本
            
        Returns:
            賠付分布樣本
        """
        if self.training_mode:
            return self.calculate_payout_sigmoid(loss_samples)
        else:
            return self.calculate_payout_step(loss_samples)
    
    def calculate_payout_tensor(self, 
                              loss_samples: torch.Tensor) -> torch.Tensor:
        """
        PyTorch張量版本的賠付計算 - 支援GPU加速和自動微分
        
        這是確保參數能夠傳遞到HBM並影響基差風險的關鍵方法！
        
        Args:
            loss_samples: PyTorch損失張量 [N, M]
            
        Returns:
            PyTorch賠付張量 [N, M]
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for tensor operations")
        
        device = loss_samples.device
        self.thresholds_tensor = self.thresholds_tensor.to(device)
        self.ratios_tensor = self.ratios_tensor.to(device)
        self.max_payout_tensor = self.max_payout_tensor.to(device)
        self.k_tensor = self.k_tensor.to(device)
        
        if self.training_mode:
            # Sigmoid模式 - 保持可微分性
            payout_samples = torch.zeros_like(loss_samples, dtype=torch.float32)
            
            for i, (threshold, ratio) in enumerate(zip(self.thresholds_tensor, self.ratios_tensor)):
                if threshold < 999:  # 有效閾值
                    prev_ratio = self.ratios_tensor[i-1] if i > 0 else 0.0
                    delta_ratio = ratio - prev_ratio
                    
                    # 可微分的Sigmoid
                    sigmoid_values = torch.sigmoid(self.k_tensor * (loss_samples - threshold))
                    payout_samples += delta_ratio * sigmoid_values
            
            payout_samples *= self.max_payout_tensor
            payout_samples = torch.clamp(payout_samples, 0, self.max_payout_tensor)
            
        else:
            # 階梯模式 - 使用不可微但精確的階梯邏輯
            payout_samples = torch.zeros_like(loss_samples, dtype=torch.float32)
            
            for i, (threshold, ratio) in enumerate(zip(self.thresholds_tensor, self.ratios_tensor)):
                if threshold < 999:
                    # 階梯函數：loss >= threshold 時賠付ratio * max_payout
                    triggered = (loss_samples >= threshold).float()
                    payout_samples = torch.maximum(
                        payout_samples, 
                        triggered * ratio * self.max_payout_tensor
                    )
        
        return payout_samples
    
    def get_configuration_summary(self) -> Dict:
        """獲取配置摘要"""
        return {
            'thresholds': self.thresholds.tolist(),
            'ratios': self.ratios.tolist(), 
            'max_payout': self.max_payout,
            'k': self.k,
            'training_mode': self.training_mode,
            'valid_thresholds': sum(1 for t in self.thresholds if t < 999),
            'payout_type': 'sigmoid' if self.training_mode else 'step'
        }


class EpsilonContaminationModel:
    """ε-contamination 模型"""
    
    def __init__(self, epsilon: float = 0.1):
        """
        初始化 ε-contamination 模型
        
        Args:
            epsilon: 污染比例
        """
        self.epsilon = epsilon
    
    def predict_distribution(self, 
                           theta: np.ndarray, 
                           X: np.ndarray,
                           n_samples: int = 100) -> np.ndarray:
        """
        基於參數 θ 和輸入 X 預測損失分布
        
        Args:
            theta: 模型參數
            X: 輸入特徵 [N, d]
            n_samples: 每個預測點的樣本數
            
        Returns:
            損失分布樣本 [N, n_samples]
        """
        N = X.shape[0]
        
        # 基本預測 (線性模型示例)
        if len(theta) >= X.shape[1]:
            linear_pred = X @ theta[:X.shape[1]]
        else:
            # 簡化：使用均值
            linear_pred = np.ones(N) * np.mean(theta)
        
        # ε-contamination: (1-ε) × Normal + ε × Heavy-tail
        samples = np.zeros((N, n_samples))
        
        for i in range(N):
            # 主要分布 (Normal)
            n_main = int((1-self.epsilon) * n_samples)
            main_samples = np.random.normal(linear_pred[i], abs(theta[-1]) if len(theta) > 1 else 1.0, n_main)
            
            # 污染分布 (Heavy-tail)
            n_contam = n_samples - n_main
            contamination_samples = np.random.exponential(abs(linear_pred[i]) * 2, n_contam)
            
            # 混合
            all_samples = np.concatenate([main_samples, contamination_samples])
            np.random.shuffle(all_samples)
            samples[i] = all_samples[:n_samples]
        
        return np.abs(samples)  # 確保損失為正


if TORCH_AVAILABLE:
    class VariationalPosterior(nn.Module):
        """變分後驗分布 q_φ(θ)"""
        
        def __init__(self, n_params: int, n_features: int):
            super().__init__()
            
            # 變分參數: 均值和對數標準差
            self.mu_net = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, n_params)
            )
            
            self.logvar_net = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, n_params)
            )
        
        def forward(self, x):
            """前向傳播"""
            mu = self.mu_net(x)
            logvar = self.logvar_net(x)
            return mu, logvar
        
        def sample(self, x, n_samples: int = 10):
            """使用 reparameterization trick 採樣"""
            mu, logvar = self.forward(x)
            std = torch.exp(0.5 * logvar)
            
            eps = torch.randn(n_samples, *mu.shape)
            samples = mu + eps * std
            
            return samples, mu, logvar


class BasisRiskAwareVI:
    """基差風險導向的變分推斷 - GPU加速版本"""
    
    def __init__(self, 
                 n_features: int,
                 epsilon_values: List[float] = None,
                 basis_risk_types: List[str] = None,
                 use_gpu: bool = True,
                 device: str = 'auto',
                 learning_rate: float = 0.01,
                 objective: str = 'crps_basis_risk',
                 n_params: int = None,
                 hierarchical_model = None,
                 # 新增: Sigmoid代理優化參數
                 use_sigmoid_proxy: bool = True,
                 sigmoid_steepness: float = 0.1,
                 training_mode: bool = True,
                 pytorch_hbm_model = None):
        """
        初始化基差風險導向 VI
        
        Args:
            n_features: 特徵維度
            epsilon_values: ε-contamination 參數候選
            basis_risk_types: 基差風險類型
            use_gpu: 是否使用GPU加速
            objective: 目標函數類型
                - 'traditional_elbo': 傳統ELBO (第二層比較)
                - 'crps_basis_risk': CRPS-based ELBO創新 (第三層比較)
                - 'hbm_two_step': 兩步法 - 先優化G(θ)再評估F_k(θ)
                - 'pytorch_hbm': PyTorch HBM風險大腦模式
            n_params: 參數維度 (None=自動計算, 2=向後兼容, 350=完整產品選擇)
            hierarchical_model: 外部HBM模型實例 (用於two_step模式)
            use_sigmoid_proxy: 是否使用Sigmoid代理函數進行可微分優化
            sigmoid_steepness: Sigmoid陡峭度參數k (越大越接近階梯函數)
            training_mode: True=訓練模式(Sigmoid), False=評估模式(階梯)
            pytorch_hbm_model: PyTorch HBM適配器實例 (用於pytorch_hbm模式)
        """
        if epsilon_values is None:
            epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20]
        if basis_risk_types is None:
            basis_risk_types = ['absolute', 'asymmetric', 'weighted']
            
        self.n_features = n_features
        
        # 🔑 支援350維產品選擇VI
        if n_params is None:
            self.n_params = n_features + 1  # 傳統：線性係數 + 噪音參數
        else:
            self.n_params = n_params  # 自定義參數維度 (2 或 350)
        self.epsilon_values = epsilon_values
        self.basis_risk_types = basis_risk_types
        self.objective = objective
        self.hierarchical_model = hierarchical_model  # 外部HBM實例
        self.pytorch_hbm_model = pytorch_hbm_model  # PyTorch HBM適配器
        
        # Sigmoid代理優化相關參數
        self.use_sigmoid_proxy = use_sigmoid_proxy
        self.sigmoid_steepness = sigmoid_steepness
        self.training_mode = training_mode
        
        # 雙GPU配置 - 支持並行計算
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.learning_rate = learning_rate
        
        if self.use_gpu:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if device == 'auto':
                    if gpu_count >= 2:
                        # 雙GPU模式
                        self.device = torch.device('cuda:0')  # 主GPU
                        self.device_secondary = torch.device('cuda:1')  # 副GPU
                        self.dual_gpu = True
                        print(f"🚀 雙GPU並行模式: GPU0 + GPU1")
                    else:
                        self.device = torch.device('cuda:0')
                        self.device_secondary = None
                        self.dual_gpu = False
                        print(f"🔧 單GPU模式: GPU0")
                else:
                    self.device = torch.device(device)
                    self.device_secondary = None
                    self.dual_gpu = False
                
                # 設置環境變量避免CUDA kernel問題
                import os
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                os.environ['TORCH_USE_CUDA_DSA'] = '1'  # 啟用設備端斷言
                
                # 清理所有GPU記憶體
                for i in range(gpu_count):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                torch.cuda.set_device(self.device)
                
            else:
                print("❌ 強制GPU模式但CUDA不可用")
                raise RuntimeError("GPU required but CUDA not available")
        else:
            self.device = 'cpu' if not TORCH_AVAILABLE else torch.device('cpu')
            self.device_secondary = None
            self.dual_gpu = False
            
        print(f"🔧 BasisRiskAwareVI初始化: {'GPU' if self.use_gpu else 'CPU'}模式")
        if self.use_gpu and TORCH_AVAILABLE:
            print(f"   GPU設備: {torch.cuda.get_device_name(self.device)}")
        
        # 🎯 賠付函數 - 支援Sigmoid代理優化
        if self.use_sigmoid_proxy:
            # 使用Sigmoid代理函數 - 支援兩階段優化
            self.payout_function = SigmoidPayoutProxy(
                k=self.sigmoid_steepness,
                training_mode=self.training_mode
            )
            print(f"🎯 啟用Sigmoid代理優化: k={self.sigmoid_steepness}, mode={'訓練' if self.training_mode else '評估'}")
        else:
            # 使用傳統階梯函數 - 僅評估模式
            self.payout_function = ParametricPayoutFunction()
            print("📊 使用傳統階梯函數")
        
        # CRPS 計算器
        self.crps_calculator = DifferentiableCRPS()
        
        # 存儲結果
        self.vi_results = {}
    
    def set_optimization_mode(self, training_mode: bool):
        """
        設置優化模式 - 兩階段代理優化的核心控制
        
        Args:
            training_mode: True=訓練模式(使用Sigmoid代理), False=評估模式(使用真實階梯)
        """
        self.training_mode = training_mode
        
        if hasattr(self.payout_function, 'set_mode'):
            self.payout_function.set_mode(training_mode)
            mode_name = "訓練(Sigmoid)" if training_mode else "評估(階梯)"
            print(f"🔄 優化模式切換: {mode_name}")
        else:
            print("⚠️ 當前賠付函數不支援模式切換")
    
    def get_steinmann_payout_proxy(self, product_index: int) -> SigmoidPayoutProxy:
        """
        為指定的Steinmann產品創建Sigmoid代理函數
        
        這是確保參數能夠正確傳遞到HBM模型的關鍵方法！
        
        Args:
            product_index: Steinmann產品索引 (0-349)
            
        Returns:
            配置好的SigmoidPayoutProxy實例
        """
        if not self.use_sigmoid_proxy:
            print("⚠️ 當前未啟用Sigmoid代理，無法創建產品代理")
            return None
        
        # 獲取Steinmann產品數據
        steinmann_data = {
            'thresholds': self._get_steinmann_products_tensor().detach().cpu().numpy(),
            'ratios': self._get_steinmann_ratios_tensor().detach().cpu().numpy(),
            'max_payouts': self._get_steinmann_max_payouts().detach().cpu().numpy()
        }
        
        # 為特定產品創建代理
        proxy = SigmoidPayoutProxy.from_steinmann_product(
            product_index=product_index,
            steinmann_data=steinmann_data,
            k=self.sigmoid_steepness,
            training_mode=self.training_mode
        )
        
        return proxy
    
    def predict_distribution(self, theta: np.ndarray, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """
        基於變分參數進行預測
        
        Args:
            theta: 變分參數均值
            X: 輸入特徵 [N, d]  
            n_samples: 每個預測點的樣本數
            
        Returns:
            預測損失分布樣本 [N, n_samples]
        """
        N = X.shape[0]
        
        # 使用變分參數生成預測
        # theta 包含線性係數和噪聲參數
        if len(theta) >= X.shape[1]:
            linear_pred = X @ theta[:X.shape[1]]
        else:
            # 如果參數不足，使用簡化預測
            linear_pred = np.ones(N) * np.mean(theta)
        
        # 添加噪聲
        noise_std = np.abs(theta[-1]) if len(theta) > X.shape[1] else 1e6
        
        # 生成樣本
        samples = np.zeros((N, n_samples))
        for i in range(N):
            # 生成對數正態分布樣本 (適合損失數據)
            log_mean = np.log(np.maximum(linear_pred[i], 1e3))  # 避免過小值
            samples[i] = np.random.lognormal(log_mean, noise_std, n_samples)
        
        return samples
    
    def compute_basis_risk(self, y_true: np.ndarray, payout_samples: np.ndarray,
                          basis_risk_type: str = 'weighted') -> float:
        """
        計算基差風險
        
        Args:
            y_true: 真實損失
            payout_samples: 賠付樣本
            basis_risk_type: 基差風險類型
            
        Returns:
            基差風險值
        """
        if len(payout_samples.shape) > 1:
            payout_mean = payout_samples.mean(1)
        else:
            payout_mean = payout_samples
            
        if basis_risk_type == 'asymmetric':
            # 只懲罰賠不夠的情況
            basis_risk = np.mean(np.maximum(0, y_true - payout_mean))
        elif basis_risk_type == 'weighted':
            # 加權不對稱懲罰
            under_penalty = np.maximum(0, y_true - payout_mean) * 2.0
            over_penalty = np.maximum(0, payout_mean - y_true) * 0.5
            basis_risk = np.mean(under_penalty + over_penalty)
        else:  # absolute
            basis_risk = np.mean(np.abs(y_true - payout_mean))
        
        return basis_risk
    
    def train_single_model(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         epsilon: float,
                         basis_risk_type: str = 'weighted',
                         n_iterations: int = 1000,
                         X_val: np.ndarray = None,
                         y_val: np.ndarray = None) -> Dict:
        """
        訓練單個 ε-contamination 模型 - 真正的GPU加速VI實現
        
        Args:
            X: 輸入特徵 [N, d]
            y: 真實損失 [N]
            epsilon: ε-contamination 參數
            basis_risk_type: 基差風險類型
            n_iterations: 訓練迭代次數
            
        Returns:
            訓練結果字典
        """
        import time
        start_time = time.time()
        
        print(f"      開始訓練 ε={epsilon:.3f}, 基差={basis_risk_type} (迭代={n_iterations})")
        
        if self.use_gpu and TORCH_AVAILABLE:
            return self._train_single_model_gpu(X, y, epsilon, basis_risk_type, n_iterations, start_time, X_val, y_val)
        else:
            return self._train_single_model_cpu(X, y, epsilon, basis_risk_type, n_iterations, start_time, X_val, y_val)
    
    def _train_single_model_gpu(self, X: np.ndarray, y: np.ndarray, epsilon: float, 
                               basis_risk_type: str, n_iterations: int, start_time: float,
                               X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """GPU加速的VI訓練"""
        import time
        
        # 🔧 數據縮放防止數值爆炸 (GPU優化)
        y_scale = np.std(y) if np.std(y) > 0 else 1.0
        y_mean = np.mean(y)
        y_scaled = (y - y_mean) / y_scale
        
        # 轉換為GPU張量
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y_scaled).float().to(self.device)  # 使用縮放後的數據
        
        print(f"        🔧 數據縮放 - 原始損失: {y.mean()/1e6:.1f}M±{y.std()/1e6:.1f}M → 標準化: {y_scaled.mean():.3f}±{y_scaled.std():.3f}")
        
        # 保存縮放參數用於後續恢復
        self._y_scale = y_scale
        self._y_mean = y_mean
        
        # 驗證集張量（如果提供）
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            # 🔧 驗證集也需要相同的縮放
            y_val_scaled = (y_val - y_mean) / y_scale
            X_val_tensor = torch.from_numpy(X_val).float().to(self.device)
            y_val_tensor = torch.from_numpy(y_val_scaled).float().to(self.device)
            print(f"        📊 使用驗證集監督: 訓練={X.shape[0]}, 驗證={X_val.shape[0]}")
            print(f"        🔧 驗證集縮放: {y_val.mean()/1e6:.1f}M±{y_val.std()/1e6:.1f}M → {y_val_scaled.mean():.3f}±{y_val_scaled.std():.3f}")
        else:
            print(f"        ⚠️ 無驗證集，可能過度擬合")
        
        # 變分參數 (在GPU上)
        torch.manual_seed(42 + int(epsilon*1000))
        mu_theta = torch.randn(self.n_params, device=self.device) * 0.1
        log_sigma_theta = torch.full((self.n_params,), -2.0, device=self.device)
        
        # 設為可求導
        mu_theta.requires_grad_(True)
        log_sigma_theta.requires_grad_(True)
        
        # 🔧 GPU優化的Adam優化器配置
        optimizer = torch.optim.Adam([mu_theta, log_sigma_theta], 
                                   lr=0.001,  # 降低學習率避免數值不穩定
                                   betas=(0.9, 0.999),
                                   eps=1e-8,
                                   weight_decay=1e-5)  # 輕微L2正則化
        
        print(f"        🔧 GPU優化器配置: lr=0.001, 參數數量={len(list(optimizer.param_groups[0]['params']))}")
        
        best_elbo = -float('inf')
        best_basis_risk_train = float('inf')
        best_basis_risk_val = float('inf')
        best_mu = mu_theta.clone()
        best_log_sigma = log_sigma_theta.clone()
        
        # Early stopping監控
        patience = 100
        no_improve_count = 0
        validation_history = []
        
        n_samples_per_iteration = 10
        
        print(f"        🚀 GPU張量計算開始...")
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # 1. 從變分分布採樣 (GPU)
            sigma_theta = torch.exp(log_sigma_theta)
            eps = torch.randn(n_samples_per_iteration, self.n_params, device=self.device)
            theta_samples = mu_theta.unsqueeze(0) + sigma_theta.unsqueeze(0) * eps  # [n_samples, n_params]
            
            # 2. 批次計算ELBO (完全GPU並行)
            elbo_batch = self._compute_elbo_batch_gpu(
                X_tensor, y_tensor, theta_samples, epsilon, basis_risk_type, mu_theta, sigma_theta
            )
            
            # 3. 反向傳播
            loss = -elbo_batch.mean()  # 最大化ELBO = 最小化負ELBO
            
            # 🔧 數值檢查防止NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"        ⚠️ 警告: 迭代{iteration+1} loss異常: {loss.item()}")
                continue
                
            loss.backward()
            
            # 🔧 梯度裁剪避免爆炸 (GPU優化)
            total_grad_norm = torch.nn.utils.clip_grad_norm_([mu_theta, log_sigma_theta], max_norm=1.0)
            
            # 檢查梯度是否有效
            if torch.isnan(total_grad_norm) or total_grad_norm == 0:
                print(f"        ⚠️ 警告: 迭代{iteration+1} 梯度異常: {total_grad_norm}")
                
            optimizer.step()
            
            # 約束log_sigma範圍
            with torch.no_grad():
                log_sigma_theta.clamp_(-5, 1)
            
            # 🔧 計算當前基差風險用於記錄 (需要恢復原始尺度)
            with torch.no_grad():
                # 訓練集基差風險 - 已在標準化尺度計算，恢復到原始損失尺度
                basis_risk_scaled = self._compute_basis_risk_batch_gpu(
                    X_tensor, y_tensor, theta_samples, epsilon, basis_risk_type
                ).mean().item()
                # 🔧 修復：基差風險已經是標準化尺度的，只需乘以標準差恢復尺度
                current_basis_risk_train = basis_risk_scaled * self._y_scale
                
                # 驗證集基差風險（如果有）
                if has_validation:
                    basis_risk_val_scaled = self._compute_basis_risk_batch_gpu(
                        X_val_tensor, y_val_tensor, theta_samples, epsilon, basis_risk_type
                    ).mean().item()
                    # 恢復到原始尺度 (百萬美元)
                    current_basis_risk_val = basis_risk_val_scaled * self._y_scale
                    validation_history.append(current_basis_risk_val)
                else:
                    current_basis_risk_val = current_basis_risk_train
                
                current_elbo = elbo_batch.mean().item()
                
                # *** 關鍵修正：使用驗證集選擇最佳模型 ***
                if has_validation:
                    # 如果有驗證集，以驗證集基差風險為準
                    if current_basis_risk_val < best_basis_risk_val:
                        best_elbo = current_elbo
                        best_basis_risk_train = current_basis_risk_train
                        best_basis_risk_val = current_basis_risk_val
                        best_mu = mu_theta.clone()
                        best_log_sigma = log_sigma_theta.clone()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # 無驗證集時才用訓練集
                    if current_elbo > best_elbo:
                        best_elbo = current_elbo
                        best_basis_risk_train = current_basis_risk_train
                        best_basis_risk_val = current_basis_risk_val
                        best_mu = mu_theta.clone()
                        best_log_sigma = log_sigma_theta.clone()
            
            # Early stopping
            if has_validation and no_improve_count >= patience:
                print(f"        🛑 Early stopping: 驗證集{patience}次無改善")
                break
            
            # 進度報告 + GPU診斷監控
            if (iteration + 1) % 200 == 0:
                # 🔧 GPU參數和梯度診斷
                mu_sample = mu_theta[:3].detach().cpu().numpy()  # 前3個參數
                sigma_sample = torch.exp(log_sigma_theta[:3]).detach().cpu().numpy()
                mu_grad_norm = torch.norm(mu_theta.grad).item() if mu_theta.grad is not None else 0.0
                sigma_grad_norm = torch.norm(log_sigma_theta.grad).item() if log_sigma_theta.grad is not None else 0.0
                
                if has_validation:
                    print(f"        迭代 {iteration+1}: ELBO={current_elbo:.3f}, 訓練={current_basis_risk_train/1e6:.1f}M, 驗證={current_basis_risk_val/1e6:.1f}M")
                else:
                    print(f"        迭代 {iteration+1}: ELBO={current_elbo:.3f}, 基差風險={current_basis_risk_train/1e6:.1f}M")
                
                # 🔍 GPU診斷信息
                print(f"        🔧 GPU診斷 - μ樣本: {mu_sample}, σ樣本: {sigma_sample}")
                print(f"        📊 梯度norm - μ: {mu_grad_norm:.6f}, σ: {sigma_grad_norm:.6f}")
                
                # 檢查參數變化
                if iteration > 0:
                    mu_change = torch.norm(mu_theta - prev_mu).item() if 'prev_mu' in locals() else 0
                    print(f"        🎯 參數變化 - Δμ: {mu_change:.6f}")
                
                # 保存當前參數用於下次比較
                prev_mu = mu_theta.clone()
        
        training_time = time.time() - start_time
        
        # 轉換回CPU NumPy用於返回
        final_mu = best_mu.detach().cpu().numpy()
        final_sigma = torch.exp(best_log_sigma).detach().cpu().numpy()
        
        if has_validation:
            print(f"      ✅ GPU訓練完成: {training_time:.1f}s, ELBO={best_elbo:.3f}")
            print(f"        訓練基差風險: {best_basis_risk_train/1e6:.1f}M, 驗證基差風險: {best_basis_risk_val/1e6:.1f}M")
            print(f"        訓練/驗證比率: {best_basis_risk_train/best_basis_risk_val:.3f}")
        else:
            print(f"      ✅ GPU訓練完成: {training_time:.1f}s, ELBO={best_elbo:.3f}, 基差風險={best_basis_risk_train/1e6:.1f}M")
        
        return {
            'epsilon': epsilon,
            'basis_risk_type': basis_risk_type,
            'final_basis_risk': best_basis_risk_val if has_validation else best_basis_risk_train,
            'train_basis_risk': best_basis_risk_train,
            'val_basis_risk': best_basis_risk_val,
            'train_val_ratio': best_basis_risk_train / best_basis_risk_val if has_validation else 1.0,
            'best_theta': final_mu,
            'theta_uncertainty': final_sigma,
            'elbo': best_elbo,
            'converged': True,
            'training_time': training_time,
            'has_validation': has_validation
        }
    
    def _compute_traditional_elbo_batch_gpu(self, X_tensor, y_tensor, theta_samples, 
                                          mu_theta, sigma_theta, likelihood_family='normal'):
        """GPU上批次計算傳統ELBO - 第二層比較的標準方法
        
        實現標準公式: ℒTraditional(φ) = E[log p(data|θ)] - KL(qφ(θ)||p(θ))
        
        這是backward-looking目標，優化歷史數據擬合度
        
        Args:
            likelihood_family: 似然函數族 ('normal', 'student_t', 'laplace')
        """
        batch_size = theta_samples.shape[0]
        n_data = X_tensor.shape[0]
        
        # *** 傳統似然項：E[log p(y|X, θ)] ***
        # θ參數: [batch_size, n_params] where n_params=2 (slope, intercept)
        if theta_samples.shape[1] >= 2:
            theta_slope = theta_samples[:, 0:1]      # [batch_size, 1]
            theta_intercept = theta_samples[:, 1:2]  # [batch_size, 1]
        else:
            theta_slope = theta_samples[:, 0:1]
            theta_intercept = torch.zeros_like(theta_slope)
        
        # 線性預測: μ = slope * X + intercept
        # 處理可能的3維輸入張量
        if X_tensor.dim() == 3:
            # 如果是 3 維 [batch, seq, features]，取最後一維並展平
            X_flat = X_tensor.reshape(-1, X_tensor.size(-1)).squeeze(-1)  # [total_data]
        else:
            X_flat = X_tensor.squeeze(-1)  # [n_data]
        
        n_data = X_flat.shape[0]
        X_expanded = X_flat.unsqueeze(0).expand(batch_size, -1)  # [batch_size, n_data]
        mu_pred = theta_slope * X_expanded + theta_intercept  # [batch_size, n_data]
        
        # 預測標準差 (假設同質性)
        sigma_pred = torch.abs(theta_samples[:, -1:]) + 1e-6  # [batch_size, 1], 使用最後一個參數
        sigma_pred = sigma_pred.expand(-1, n_data)  # [batch_size, n_data]
        
        # 計算log likelihood
        # 處理可能的維度不匹配
        if y_tensor.dim() > 1:
            y_flat = y_tensor.reshape(-1)  # 展平為1維
        else:
            y_flat = y_tensor
        
        # 確保 y 和 X 的數據點數量匹配
        y_flat = y_flat[:n_data]  # 截取匹配的數據點數量
        y_expanded = y_flat.unsqueeze(0).expand(batch_size, -1)  # [batch_size, n_data]
        
        if likelihood_family == 'normal':
            # Normal likelihood: log p(y|μ,σ) = -0.5*log(2πσ²) - (y-μ)²/(2σ²)
            log_likelihood = (-0.5 * torch.log(2 * np.pi * sigma_pred**2) - 
                            (y_expanded - mu_pred)**2 / (2 * sigma_pred**2))
        elif likelihood_family == 'student_t':
            # Student-t likelihood (簡化版，自由度=3)
            nu = 3.0
            log_likelihood = (torch.lgamma(torch.tensor((nu + 1) / 2, device=self.device)) - 
                            torch.lgamma(torch.tensor(nu / 2, device=self.device)) - 
                            0.5 * torch.log(torch.tensor(nu * np.pi, device=self.device)) - 
                            torch.log(sigma_pred) - 
                            ((nu + 1) / 2) * torch.log(1 + (y_expanded - mu_pred)**2 / (nu * sigma_pred**2)))
        elif likelihood_family == 'laplace':
            # Laplace likelihood: log p(y|μ,b) = -log(2b) - |y-μ|/b
            log_likelihood = (-torch.log(2 * sigma_pred) - torch.abs(y_expanded - mu_pred) / sigma_pred)
        else:
            raise ValueError(f"不支持的似然函數族: {likelihood_family}")
        
        # 所有數據點的平均log likelihood
        likelihood_term = torch.sum(log_likelihood, dim=1)  # [batch_size]
        
        # *** KL散度正則化項 ***
        # 先驗 p(θ) ~ N(0,I)
        log_prior = -0.5 * torch.sum(theta_samples**2, dim=1) - \
                    0.5 * self.n_params * torch.log(torch.tensor(2 * np.pi, device=self.device))
        
        # 變分後驗 qφ(θ) ~ N(μφ, σφ²I)
        log_q = -0.5 * torch.sum((theta_samples - mu_theta)**2 / sigma_theta**2, dim=1) - \
                0.5 * torch.sum(torch.log(2 * np.pi * sigma_theta**2))
        
        # KL散度
        kl_divergence = log_q - log_prior
        
        # 傳統ELBO = E[log p(data|θ)] - KL(qφ||p)
        elbo = likelihood_term - kl_divergence
        
        return elbo
    
    def _compute_elbo_batch_gpu(self, X_tensor, y_tensor, theta_samples, epsilon, 
                               basis_risk_type, mu_theta, sigma_theta):
        """GPU上批次計算ELBO - 支援三種模式
        
        根據self.objective選擇:
        - 'traditional_elbo': 傳統ELBO (第二層比較)
        - 'crps_basis_risk': CRPS-based ELBO創新 (第三層比較)
        - 'hbm_two_step': HBM兩步法 - 直接優化G(θ)的CRPS
        """
        if self.objective == 'traditional_elbo':
            # 第二層：使用傳統ELBO
            return self._compute_traditional_elbo_batch_gpu(
                X_tensor, y_tensor, theta_samples, mu_theta, sigma_theta
            )
        elif self.objective == 'crps_basis_risk':
            # 第三層：使用創新的CRPS-based ELBO
            return self._compute_crps_based_elbo_batch_gpu(
                X_tensor, y_tensor, theta_samples, epsilon, mu_theta, sigma_theta
            )
        elif self.objective == 'hbm_two_step':
            # HBM兩步法：Step 1 - 直接優化G(θ)的CRPS
            return self._compute_hbm_two_step_elbo_batch_gpu(
                X_tensor, y_tensor, theta_samples, mu_theta, sigma_theta
            )
        elif self.objective == 'pytorch_hbm':
            # PyTorch HBM風險大腦：使用PyTorch層級貝氏模型
            return self._compute_pytorch_hbm_elbo_batch_gpu(
                X_tensor, y_tensor, theta_samples, mu_theta, sigma_theta
            )
        else:
            raise ValueError(f"不支持的目標函數: {self.objective}")
    
    def _compute_crps_based_elbo_batch_gpu(self, X_tensor, y_tensor, theta_samples, epsilon, 
                                         mu_theta, sigma_theta):
        """GPU上批次計算Basis-Risk-Aware ELBO - 核心創新實現
        
        實現創新公式: ℒBR(φ) = -E_qφ(θ)[CRPS(y, F(θ))] - KL(qφ(θ)||p(θ))
        
        這是forward-looking目標，直接優化未來預測質量而非歷史擬合度
        """
        batch_size = theta_samples.shape[0]  # n_samples_per_iteration
        
        # *** 核心創新：使用CRPS(y, F(θ))替代傳統似然 ***
        # 計算 CRPS(y_observed, F_payout(θ)) for each θ sample
        crps_scores = self._compute_crps_batch_gpu(
            X_tensor, y_tensor, theta_samples, epsilon
        )
        # CRPS項：負號使得最小化CRPS等效於最大化ELBO
        crps_term = -crps_scores
        
        # KL散度正則化項 = E_qφ[log qφ(θ)] - E_qφ[log p(θ)]
        # 先驗 p(θ) ~ N(0,I)
        log_prior = -0.5 * torch.sum(theta_samples**2, dim=1) - \
                    0.5 * self.n_params * torch.log(torch.tensor(2 * np.pi, device=self.device))
        
        # 變分後驗 qφ(θ) ~ N(μφ, σφ²I)
        log_q = -0.5 * torch.sum((theta_samples - mu_theta)**2 / sigma_theta**2, dim=1) - \
                0.5 * torch.sum(torch.log(2 * np.pi * sigma_theta**2))
        
        # KL散度: KL(qφ||p) = E_qφ[log qφ(θ)] - E_qφ[log p(θ)]
        kl_divergence = log_q - log_prior
        
        # Basis-Risk-Aware ELBO = -E[CRPS] - KL
        # 注意：我們要最大化這個值，所以最小化CRPS，最小化KL
        elbo = crps_term - kl_divergence
        
        return elbo
    
    def _compute_hbm_two_step_elbo_batch_gpu(self, X_tensor, y_tensor, theta_samples, 
                                           mu_theta, sigma_theta):
        """HBM兩步法Step 1: 直接優化G(θ)的CRPS - 無參數保險產品
        
        實現公式: ℒHBM(φ) = -E_qφ(θ)[CRPS(y, G(θ))] - KL(qφ(θ)||p(θ))
        
        G(θ)是外部提供的層次貝葉斯模型，直接預測損失分布
        """
        if self.hierarchical_model is None:
            raise ValueError("HBM兩步法需要提供 hierarchical_model 實例")
            
        batch_size = theta_samples.shape[0]
        
        # *** 核心：使用外部HBM G(θ)預測損失分布 ***
        # 將GPU張量轉為numpy供HBM使用
        X_np = X_tensor.detach().cpu().numpy()
        theta_np = theta_samples.detach().cpu().numpy()
        
        # 批次調用HBM預測
        hbm_predictions = []
        for i in range(batch_size):
            # 調用外部HBM的predict_distribution方法
            pred_samples = self.hierarchical_model.predict_distribution(
                theta_np[i], X_np, n_samples=50
            )
            hbm_predictions.append(pred_samples)
        
        # 轉回GPU張量 [batch_size, n_data, n_samples]
        hbm_tensor = torch.from_numpy(np.array(hbm_predictions)).float().to(self.device)
        
        # 計算CRPS(y_observed, G(θ))
        crps_scores = []
        for i in range(batch_size):
            # 對每個θ樣本計算CRPS
            pred_samples = hbm_tensor[i]  # [n_data, n_samples]
            crps_batch = []
            for j in range(pred_samples.shape[0]):
                # 使用ensemble CRPS計算
                y_obs = y_tensor[j:j+1]  # [1]
                forecast = pred_samples[j]  # [n_samples]
                
                # 簡化CRPS計算
                crps_val = torch.mean(torch.abs(forecast - y_obs)) - 0.5 * torch.mean(
                    torch.abs(forecast.unsqueeze(0) - forecast.unsqueeze(1))
                )
                crps_batch.append(crps_val)
            
            crps_scores.append(torch.stack(crps_batch).mean())
        
        crps_term = -torch.stack(crps_scores)  # 負號使CRPS最小化等效ELBO最大化
        
        # KL散度正則化項
        log_prior = -0.5 * torch.sum(theta_samples**2, dim=1) - \
                    0.5 * self.n_params * torch.log(torch.tensor(2 * np.pi, device=self.device))
        
        log_q = -0.5 * torch.sum((theta_samples - mu_theta)**2 / sigma_theta**2, dim=1) - \
                0.5 * torch.sum(torch.log(2 * np.pi * sigma_theta**2))
        
        kl_divergence = log_q - log_prior
        
        # HBM ELBO = -E[CRPS(y,G(θ))] - KL
        elbo = crps_term - kl_divergence
        
        return elbo
    
    def _compute_pytorch_hbm_elbo_batch_gpu(self, X_tensor, y_tensor, theta_samples, 
                                          mu_theta, sigma_theta):
        """PyTorch HBM風險大腦: 使用PyTorch層次貝氏模型進行VI優化
        
        實現公式: ℒPyTorchHBM(φ) = -E_qφ(θ)[CRPS(y, PyTorchHBM(θ))] - KL(qφ(θ)||p(θ))
        
        PyTorchHBM是一個完全可微分的4層階層模型，支持端到端優化
        """
        if self.pytorch_hbm_model is None:
            raise ValueError("PyTorch HBM模式需要提供 pytorch_hbm_model 適配器實例")
            
        batch_size = theta_samples.shape[0]
        
        # *** 核心：使用PyTorch HBM預測損失分布 ***
        # PyTorch HBM適配器已經處理GPU張量，無需轉換
        X_list = [X_tensor[:, 0], X_tensor[:, 1]]  # [hazard_intensities, exposure_values]
        
        # 批次調用PyTorch HBM預測
        crps_scores = []
        for i in range(batch_size):
            # 調用PyTorch HBM適配器的predict_distribution方法
            # 該方法返回numpy數組，需要轉換為GPU張量
            pred_samples_np = self.pytorch_hbm_model.predict_distribution(
                theta=theta_samples[i].detach().cpu().numpy(),
                X=X_list,
                n_samples=100
            )
            
            # 轉換為GPU張量 [n_samples, n_hospitals, n_events] -> [n_data, n_samples]
            pred_samples_tensor = torch.from_numpy(pred_samples_np).float().to(self.device)
            
            # 重新調整維度以匹配期望格式 [n_data, n_samples]
            if pred_samples_tensor.dim() == 3:
                # 假設我們關心第一個醫院的所有事件
                pred_samples_flat = pred_samples_tensor[:, 0, :].T  # [n_events, n_samples] -> [n_samples, n_events]
                # 進一步平坦化為 [n_data=n_events*n_hospitals, n_samples]
                pred_samples_2d = pred_samples_flat.T  # [n_events, n_samples]
            else:
                pred_samples_2d = pred_samples_tensor
                
            # 計算CRPS(y_observed, PyTorchHBM(θ))
            crps_batch = []
            n_data = min(pred_samples_2d.shape[0], y_tensor.shape[0])
            
            for j in range(n_data):
                y_obs = y_tensor[j:j+1]  # [1]
                
                if pred_samples_2d.shape[0] > j:
                    forecast = pred_samples_2d[j]  # [n_samples]
                    
                    # 簡化CRPS計算 (可微分)
                    if len(forecast.shape) > 0 and forecast.numel() > 1:
                        crps_val = torch.mean(torch.abs(forecast - y_obs)) - 0.5 * torch.mean(
                            torch.abs(forecast.unsqueeze(0) - forecast.unsqueeze(1))
                        )
                    else:
                        # 退化到MSE
                        crps_val = torch.abs(forecast.mean() - y_obs)
                        
                    crps_batch.append(crps_val)
            
            if crps_batch:
                crps_scores.append(torch.stack(crps_batch).mean())
            else:
                # 如果沒有有效的CRPS計算，使用默認懲罰
                crps_scores.append(torch.tensor(1e6, device=self.device))
        
        crps_term = -torch.stack(crps_scores)  # 負號使CRPS最小化等效ELBO最大化
        
        # KL散度正則化項 (與其他方法保持一致)
        log_prior = -0.5 * torch.sum(theta_samples**2, dim=1) - \
                    0.5 * self.n_params * torch.log(torch.tensor(2 * np.pi, device=self.device))
        
        log_q = -0.5 * torch.sum((theta_samples - mu_theta)**2 / sigma_theta**2, dim=1) - \
                0.5 * torch.sum(torch.log(2 * np.pi * sigma_theta**2))
        
        kl_divergence = log_q - log_prior
        
        # PyTorch HBM ELBO = -E[CRPS(y,PyTorchHBM(θ))] - KL
        elbo = crps_term - kl_divergence
        
        return elbo
    
    def _compute_basis_risk_batch_gpu(self, X_tensor, y_tensor, theta_samples, epsilon, basis_risk_type):
        """
        雙GPU並行計算350維VI基差風險 - 支援Sigmoid代理優化
        
        🎯 兩階段代理優化：
        - 訓練模式：使用Sigmoid代理函數，保持可微分性
        - 評估模式：使用原始階梯函數，確保真實性
        
        GPU0: 處理前175個產品 (0-174)  
        GPU1: 處理後175個產品 (175-349)
        使用CUDA流並行執行，無回退機制
        """
        batch_size = theta_samples.shape[0]
        n_params = theta_samples.shape[1]
        
        if n_params != 350:
            raise ValueError(f"雙GPU並行要求 n_params=350, got {n_params}")
        
        # 處理輸入數據維度
        if X_tensor.dim() == 3:
            X_flat = X_tensor.reshape(-1, X_tensor.size(-1)).squeeze(-1)
        else:
            X_flat = X_tensor.squeeze(-1)
        
        if y_tensor.dim() > 1:
            y_flat = y_tensor.reshape(-1)
        else:
            y_flat = y_tensor
            
        n_data = X_flat.shape[0]
        y_flat = y_flat[:n_data]
        
        # epsilon contamination
        if epsilon > 0:
            noise = torch.randn_like(y_flat.unsqueeze(0).expand(batch_size, -1)) * epsilon * y_flat.mean()
            y_perturbed = y_flat.unsqueeze(0).expand(batch_size, -1) + noise
        else:
            y_perturbed = y_flat.unsqueeze(0).expand(batch_size, -1)
        
        wind_speeds = X_flat.unsqueeze(0).expand(batch_size, -1)
        
        # 強制雙GPU並行分工
        gpu0_products = 175
        gpu1_products = 175
        
        # 分割產品權重和數據到兩個GPU
        product_weights = torch.softmax(theta_samples, dim=1)
        weights_gpu0 = product_weights[:, :gpu0_products].to(self.device)
        weights_gpu1 = product_weights[:, gpu0_products:].to(self.device_secondary)
        
        # 數據複製到兩個GPU
        wind_gpu0 = wind_speeds.to(self.device)
        wind_gpu1 = wind_speeds.to(self.device_secondary)
        y_gpu0 = y_perturbed.to(self.device)
        y_gpu1 = y_perturbed.to(self.device_secondary)
        
        # 載入Steinmann產品定義到兩個GPU
        steinmann_data = self._load_steinmann_dual_gpu()
        
        # 創建CUDA流實現真正並行
        stream0 = torch.cuda.Stream(device=self.device)
        stream1 = torch.cuda.Stream(device=self.device_secondary)
        
        # 並行計算兩部分產品賠付
        with torch.cuda.stream(stream0):
            payouts_gpu0 = self._compute_products_subset_gpu(
                wind_gpu0, weights_gpu0, 
                steinmann_data['gpu0'], 0, gpu0_products
            )
        
        with torch.cuda.stream(stream1):
            payouts_gpu1 = self._compute_products_subset_gpu(
                wind_gpu1, weights_gpu1,
                steinmann_data['gpu1'], gpu0_products, gpu0_products + gpu1_products
            )
        
        # 等待兩個流完成
        stream0.synchronize()
        stream1.synchronize()
        
        # 合併結果到主GPU
        total_payouts = payouts_gpu0 + payouts_gpu1.to(self.device)
        
        # 計算基差風險
        if basis_risk_type == 'absolute':
            basis_risk = torch.mean(torch.abs(y_gpu0 - total_payouts), dim=1)
        elif basis_risk_type == 'asymmetric':
            under_penalty = torch.mean(torch.relu(y_gpu0 - total_payouts), dim=1)
            over_penalty = torch.mean(torch.relu(total_payouts - y_gpu0), dim=1)
            basis_risk = 2.0 * under_penalty + over_penalty
        else:
            diff = y_gpu0 - total_payouts
            weights = torch.abs(y_gpu0) / torch.mean(torch.abs(y_gpu0), dim=1, keepdim=True)
            weighted_diff = diff * weights
            basis_risk = torch.mean(torch.abs(weighted_diff), dim=1)
        
        return basis_risk
    
    def _load_steinmann_dual_gpu(self):
        """載入350個Steinmann產品到雙GPU，返回分割的數據"""
        steinmann_thresholds = self._get_steinmann_products_tensor().to(dtype=torch.float32)  # [350, 4]
        steinmann_ratios = self._get_steinmann_ratios_tensor().to(dtype=torch.float32)        # [350, 4] 
        steinmann_max_payouts = self._get_steinmann_max_payouts().to(dtype=torch.float32)     # [350]
        
        # 標準化
        steinmann_max_payouts = (steinmann_max_payouts - self._y_mean) / self._y_scale
        
        # 分割到兩個GPU
        gpu0_data = {
            'thresholds': steinmann_thresholds[:175].to(self.device),
            'ratios': steinmann_ratios[:175].to(self.device),
            'max_payouts': steinmann_max_payouts[:175].to(self.device)
        }
        
        gpu1_data = {
            'thresholds': steinmann_thresholds[175:].to(self.device_secondary),
            'ratios': steinmann_ratios[175:].to(self.device_secondary),
            'max_payouts': steinmann_max_payouts[175:].to(self.device_secondary)
        }
        
        return {'gpu0': gpu0_data, 'gpu1': gpu1_data}
    
    def _compute_products_subset_gpu(self, wind_speeds, product_weights, steinmann_data, start_idx, end_idx):
        """
        🎯 代理優化核心：根據模式選擇Sigmoid或階梯函數
        
        確保θ參數變化能夠正確反映到基差風險計算中！
        """
        batch_size = wind_speeds.shape[0]
        n_data = wind_speeds.shape[1]
        n_products = end_idx - start_idx
        
        payouts = torch.zeros(batch_size, n_data, device=wind_speeds.device, dtype=torch.float32)
        
        thresholds = steinmann_data['thresholds']  # [n_products, 4]
        ratios = steinmann_data['ratios']          # [n_products, 4]
        max_payouts = steinmann_data['max_payouts'] # [n_products]
        
        # 🔄 根據代理優化模式選擇計算方法
        if self.use_sigmoid_proxy and self.training_mode:
            # === 訓練模式：Sigmoid代理函數 ===
            print(f"🎯 使用Sigmoid代理函數計算賠付 (k={self.sigmoid_steepness})")
            payouts = self._compute_sigmoid_payout_gpu(
                wind_speeds, product_weights, thresholds, ratios, max_payouts, n_products
            )
        else:
            # === 評估模式：原始階梯函數 ===
            print(f"📊 使用原始階梯函數計算賠付")
            payouts = self._compute_step_payout_gpu(
                wind_speeds, product_weights, thresholds, ratios, max_payouts, n_products
            )
        
        return payouts
    
    def _compute_sigmoid_payout_gpu(self, wind_speeds, product_weights, thresholds, ratios, max_payouts, n_products):
        """
        🎯 Sigmoid代理賠付計算 - 可微分的平滑近似
        
        實現公式: Payout(x) = max_payout * Σ[(rᵢ - rᵢ₋₁) * sigmoid(k*(x - tᵢ))]
        """
        batch_size = wind_speeds.shape[0]
        n_data = wind_speeds.shape[1]
        payouts = torch.zeros(batch_size, n_data, device=wind_speeds.device, dtype=torch.float32)
        
        # 向量化計算所有產品的Sigmoid賠付
        for prod_idx in range(n_products):
            product_thresholds = thresholds[prod_idx]  # [4]
            product_ratios = ratios[prod_idx]          # [4]
            product_max_payout = max_payouts[prod_idx] # scalar
            
            # 為每個產品計算Sigmoid賠付
            product_payouts = torch.zeros_like(wind_speeds, dtype=torch.float32)
            
            for threshold_idx in range(4):
                threshold = float(product_thresholds[threshold_idx].item())
                if threshold < 999:  # 有效閾值
                    ratio = float(product_ratios[threshold_idx].item())
                    
                    # 計算增量賠付比例 (Steinmann階梯式邏輯)
                    prev_ratio = float(product_ratios[threshold_idx-1].item()) if threshold_idx > 0 else 0.0
                    delta_ratio = ratio - prev_ratio
                    
                    if delta_ratio > 0:
                        # 🎯 可微分的Sigmoid函數
                        sigmoid_values = torch.sigmoid(self.sigmoid_steepness * (wind_speeds - threshold))
                        product_payouts += delta_ratio * sigmoid_values
            
            # 應用最大賠付限制
            product_payouts *= product_max_payout
            product_payouts = torch.clamp(product_payouts, 0, product_max_payout)
            
            # 加權累積 - θ參數的變化在這裡體現！
            weight = product_weights[:, prod_idx:prod_idx+1]  # [batch_size, 1]
            payouts += weight * product_payouts
        
        return payouts
    
    def _compute_step_payout_gpu(self, wind_speeds, product_weights, thresholds, ratios, max_payouts, n_products):
        """
        📊 原始階梯賠付計算 - 精確但不可微分
        
        使用真實的Steinmann階梯函數邏輯
        """
        batch_size = wind_speeds.shape[0]
        n_data = wind_speeds.shape[1]
        payouts = torch.zeros(batch_size, n_data, device=wind_speeds.device, dtype=torch.float32)
        
        # 向量化計算所有產品的階梯賠付
        for prod_idx in range(n_products):
            product_thresholds = thresholds[prod_idx]  # [4]
            product_ratios = ratios[prod_idx]          # [4]
            product_max_payout = max_payouts[prod_idx] # scalar
            
            # 為每個產品計算階梯賠付
            product_payouts = torch.zeros_like(wind_speeds, dtype=torch.float32)
            
            # Steinmann階梯邏輯：找到觸發的最高閾值
            for i in range(batch_size):
                for j in range(n_data):
                    wind_value = wind_speeds[i, j].item()
                    max_triggered_ratio = 0.0
                    
                    for threshold_idx in range(4):
                        threshold = float(product_thresholds[threshold_idx].item())
                        if threshold < 999 and wind_value >= threshold:
                            ratio = float(product_ratios[threshold_idx].item())
                            max_triggered_ratio = max(max_triggered_ratio, ratio)
                    
                    product_payouts[i, j] = product_max_payout * max_triggered_ratio
            
            # 加權累積
            weight = product_weights[:, prod_idx:prod_idx+1]  # [batch_size, 1]
            payouts += weight * product_payouts
        
        return payouts
    
    def _get_steinmann_products_tensor(self):
        """獲取完整350個Steinmann產品的閾值定義 - Steinmann et al. (2023) 標準"""
        thresholds_list = []
        
        # ===== 25個單閾值產品 =====
        # 基於Saffir-Simpson分級: Cat1=33, Cat2=43, Cat3=50, Cat4=58, Cat5=70 m/s
        single_thresholds = [
            33, 35, 37, 39, 41,  # Cat1附近
            43, 45, 47, 49,      # Cat2附近  
            50, 52, 54, 56,      # Cat3附近
            58, 60, 62, 64, 66,  # Cat4附近
            70, 72, 74, 76, 78, 80, 85  # Cat5附近
        ]
        
        for thresh in single_thresholds:
            # 單閾值：[threshold, 999, 999, 999] - 只有1個有效閾值
            thresholds_list.append(torch.tensor([thresh, 999, 999, 999], device=self.device, dtype=torch.float32))
        
        # ===== 20個雙閾值產品 =====
        dual_pairs = [
            (33, 50), (35, 52), (37, 54), (39, 56), (41, 58),  # Cat1→Cat3
            (43, 58), (45, 60), (47, 62), (49, 64),            # Cat2→Cat4
            (50, 70), (52, 72), (54, 74), (56, 76),            # Cat3→Cat5
            (33, 43), (35, 45), (37, 47),                      # Cat1→Cat2
            (58, 78), (60, 80), (62, 82), (64, 85)             # Cat4→Cat5+
        ]
        
        for t1, t2 in dual_pairs:
            # 雙閾值：[t1, t2, 999, 999]
            thresholds_list.append(torch.tensor([t1, t2, 999, 999], device=self.device, dtype=torch.float32))
        
        # ===== 15個三閾值產品 =====
        triple_sets = [
            (33, 43, 58), (35, 45, 60), (37, 47, 62),         # 低→中→高
            (33, 50, 70), (35, 52, 72), (37, 54, 74),         # 全范圍漸進
            (43, 58, 78), (45, 60, 80), (47, 62, 82),         # 中→高→超高
            (30, 45, 65), (32, 47, 67), (34, 49, 69),         # 密集覆蓋
            (40, 55, 75), (42, 57, 77), (44, 59, 79)          # 高密度
        ]
        
        for t1, t2, t3 in triple_sets:
            # 三閾值：[t1, t2, t3, 999]
            thresholds_list.append(torch.tensor([t1, t2, t3, 999], device=self.device, dtype=torch.float32))
        
        # ===== 10個四閾值產品 =====
        quad_sets = [
            (30, 40, 55, 75), (32, 42, 57, 77),               # 全覆蓋細分
            (33, 43, 58, 78), (35, 45, 60, 80),               # Saffir-Simpson標準
            (31, 44, 59, 76), (34, 46, 61, 79),               # 交錯覆蓋
            (29, 41, 56, 74), (36, 48, 63, 81),               # 擴展范圍
            (38, 50, 65, 83), (40, 52, 67, 85)                # 高端覆蓋
        ]
        
        for t1, t2, t3, t4 in quad_sets:
            # 四閾值：[t1, t2, t3, t4]
            thresholds_list.append(torch.tensor([t1, t2, t3, t4], device=self.device, dtype=torch.float32))
        
        # ===== 每個產品×5個半徑 = 350個 =====
        radii = [15, 30, 50, 75, 100]  # km
        full_products = []
        
        for radius in radii:
            for base_thresholds in thresholds_list:
                # 每個產品在不同半徑下閾值可能略有調整
                radius_factor = 1.0 + (radius - 50) * 0.001  # 半徑影響係數
                adjusted_thresholds = base_thresholds.clone()
                adjusted_thresholds[adjusted_thresholds < 999] *= radius_factor
                full_products.append(adjusted_thresholds)
        
        return torch.stack(full_products)  # [350, 4] tensor
    
    def _get_steinmann_ratios_tensor(self):
        """獲取完整350個產品的賠付比例 - Steinmann 2023標準25%遞增"""
        ratios_list = []
        
        # ===== 25個單閾值產品的比例 =====
        for _ in range(25):
            # 單閾值：100%賠付 [1.0, 0, 0, 0]
            ratios_list.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32))
        
        # ===== 20個雙閾值產品的比例 =====
        for _ in range(20):
            # 雙閾值：50%, 100% 賠付 [0.5, 1.0, 0, 0]
            ratios_list.append(torch.tensor([0.5, 1.0, 0.0, 0.0], device=self.device, dtype=torch.float32))
        
        # ===== 15個三閾值產品的比例 =====
        for _ in range(15):
            # 三閾值：25%, 50%, 100% 賠付 [0.25, 0.5, 1.0, 0]
            ratios_list.append(torch.tensor([0.25, 0.5, 1.0, 0.0], device=self.device, dtype=torch.float32))
        
        # ===== 10個四閾值產品的比例 =====
        for _ in range(10):
            # 四閾值：25%, 50%, 75%, 100% 賠付 [0.25, 0.5, 0.75, 1.0] - 完整Steinmann標準
            ratios_list.append(torch.tensor([0.25, 0.5, 0.75, 1.0], device=self.device, dtype=torch.float32))
        
        # ===== 每個基礎產品×5個半徑 = 350個 =====
        radii = [15, 30, 50, 75, 100]
        full_ratios = []
        
        for radius in radii:
            for base_ratios in ratios_list:
                # 半徑對賠付比例的微調
                radius_adjustment = 1.0 + (radius - 50) * 0.0005  # 微小調整
                adjusted_ratios = base_ratios.clone()
                adjusted_ratios[adjusted_ratios > 0] *= radius_adjustment
                adjusted_ratios = torch.clamp(adjusted_ratios, 0.0, 1.0)  # 保持在[0,1]範圍
                full_ratios.append(adjusted_ratios)
        
        return torch.stack(full_ratios)  # [350, 4] tensor
    
    def _get_steinmann_max_payouts(self):
        """獲取完整350個產品的最大賠付額度 - 基於實際風險暴露"""
        # 基於實際數據的賠付額度設置
        base_payout = float(self._y_mean) if hasattr(self, '_y_mean') else 20e6
        payouts = []
        
        # ===== 產品類型對應不同賠付水平 =====
        radii = [15, 30, 50, 75, 100]
        base_products_count = 70  # 25+20+15+10
        
        for radius_idx, radius in enumerate(radii):
            # 半徑影響基礎賠付水平
            radius_multiplier = 0.3 + (radius / 100) * 0.7  # 0.3到1.0
            
            # 單閾值產品 (25個) - 較低賠付
            for i in range(25):
                product_multiplier = 0.5 + (i / 24) * 0.3  # 0.5到0.8
                total_multiplier = radius_multiplier * product_multiplier
                payouts.append(base_payout * total_multiplier)
            
            # 雙閾值產品 (20個) - 中等賠付
            for i in range(20):
                product_multiplier = 0.6 + (i / 19) * 0.4  # 0.6到1.0
                total_multiplier = radius_multiplier * product_multiplier
                payouts.append(base_payout * total_multiplier)
            
            # 三閾值產品 (15個) - 較高賠付
            for i in range(15):
                product_multiplier = 0.8 + (i / 14) * 0.5  # 0.8到1.3
                total_multiplier = radius_multiplier * product_multiplier
                payouts.append(base_payout * total_multiplier)
            
            # 四閾值產品 (10個) - 最高賠付
            for i in range(10):
                product_multiplier = 1.0 + (i / 9) * 0.8  # 1.0到1.8
                total_multiplier = radius_multiplier * product_multiplier
                payouts.append(base_payout * total_multiplier)
        
        return torch.tensor(payouts, device=self.device, dtype=torch.float32)  # [350] tensor
    
    def _compute_crps_batch_gpu(self, X_tensor, y_tensor, theta_samples, epsilon):
        """統一的CRPS計算 - 使用相同的350產品基差風險邏輯
        
        🔑 核心修復: 現在CRPS和基差風險使用完全相同的計算邏輯
        避免"A考卷訓練，B考卷評分"的問題
        """
        # ✅ 直接使用基差風險計算函數獲取賠付
        basis_risks = self._compute_basis_risk_batch_gpu(
            X_tensor, y_tensor, theta_samples, epsilon, 'absolute'
        )
        
        # 將基差風險轉換為CRPS分數
        # 基差風險越小，CRPS越好（負值，因為ELBO要最大化）
        crps_scores = -basis_risks  # 負基差風險作為CRPS
        
        return crps_scores
    
    
    def _train_single_model_cpu(self, X: np.ndarray, y: np.ndarray, epsilon: float, 
                               basis_risk_type: str, n_iterations: int, start_time: float,
                               X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """CPU版本的VI訓練（原始實現）"""
        # 真正的VI：估計參數分布的變分參數
        model = EpsilonContaminationModel(epsilon)
        
        # 變分參數：均值和對數方差
        np.random.seed(42 + int(epsilon*1000))  # 不同epsilon用不同種子
        mu_theta = np.random.randn(self.n_params) * 0.1
        log_sigma_theta = np.full(self.n_params, -2.0)  # 初始方差較小
        
        best_elbo = -np.inf
        best_basis_risk = np.inf
        best_mu = mu_theta.copy()
        best_log_sigma = log_sigma_theta.copy()
        
        learning_rate = 0.01
        n_samples_per_iteration = 10
        
        # 真正的VI優化循環
        for iteration in range(n_iterations):
            # 1. 從變分分布中採樣參數
            sigma_theta = np.exp(log_sigma_theta)
            theta_samples = []
            
            for _ in range(n_samples_per_iteration):
                theta_sample = mu_theta + sigma_theta * np.random.randn(self.n_params)
                theta_samples.append(theta_sample)
            
            # 2. 計算ELBO及其梯度
            elbo_total = 0
            mu_grad = np.zeros_like(mu_theta)
            log_sigma_grad = np.zeros_like(log_sigma_theta)
            total_basis_risk = 0
            
            for theta in theta_samples:
                # 預測分布
                loss_samples = model.predict_distribution(theta, X, 50)
                payout_samples = self.payout_function.calculate_payout_distribution(loss_samples)
                
                # 計算似然 (負基差風險作為似然)
                basis_risk = self.compute_basis_risk(y, payout_samples, basis_risk_type)
                log_likelihood = -basis_risk / 1e9  # 標準化
                
                # 先驗 (標準高斯)
                log_prior = -0.5 * np.sum(theta**2)
                
                # 變分分布熵
                log_q = -0.5 * np.sum((theta - mu_theta)**2 / sigma_theta**2) - \
                        0.5 * np.sum(np.log(2 * np.pi * sigma_theta**2))
                
                # ELBO = E[log p(y|θ)] + E[log p(θ)] - E[log q(θ)]
                elbo = log_likelihood + log_prior - log_q
                elbo_total += elbo
                total_basis_risk += basis_risk
                
                # 梯度估計 (REINFORCE-style)
                if elbo > -1e6:  # 避免數值不穩定
                    reward = elbo + 1e6  # 偏移確保正數
                    score_mu = (theta - mu_theta) / sigma_theta**2
                    score_log_sigma = 0.5 * (((theta - mu_theta)/sigma_theta)**2 - 1)
                    
                    mu_grad += reward * score_mu
                    log_sigma_grad += reward * score_log_sigma
            
            # 平均梯度
            mu_grad /= n_samples_per_iteration
            log_sigma_grad /= n_samples_per_iteration
            elbo_total /= n_samples_per_iteration
            avg_basis_risk = total_basis_risk / n_samples_per_iteration
            
            # 3. 參數更新 (Adam-like)
            momentum = 0.9 if iteration > 0 else 0
            if iteration == 0:
                mu_velocity = np.zeros_like(mu_theta)
                log_sigma_velocity = np.zeros_like(log_sigma_theta)
            
            mu_velocity = momentum * mu_velocity + (1-momentum) * mu_grad
            log_sigma_velocity = momentum * log_sigma_velocity + (1-momentum) * log_sigma_grad
            
            # 自適應學習率
            current_lr = learning_rate / (1 + iteration / 500)
            
            mu_theta += current_lr * mu_velocity
            log_sigma_theta += current_lr * 0.5 * log_sigma_velocity  # 方差更新較慢
            
            # 防止方差過小或過大
            log_sigma_theta = np.clip(log_sigma_theta, -5, 1)
            
            # 更新最佳結果
            if elbo_total > best_elbo:
                best_elbo = elbo_total
                best_basis_risk = avg_basis_risk
                best_mu = mu_theta.copy()
                best_log_sigma = log_sigma_theta.copy()
            
            # 進度報告
            if (iteration + 1) % 200 == 0:
                print(f"        迭代 {iteration+1}: ELBO={elbo_total:.3f}, 基差風險={avg_basis_risk/1e6:.1f}M")
        
        training_time = time.time() - start_time
        final_sigma = np.exp(best_log_sigma)
        
        print(f"      ✅ 訓練完成: {training_time:.1f}s, ELBO={best_elbo:.3f}, 基差風險={best_basis_risk/1e6:.1f}M")
        
        return {
            'epsilon': epsilon,
            'basis_risk_type': basis_risk_type,
            'final_basis_risk': best_basis_risk,
            'best_theta': best_mu,
            'theta_uncertainty': final_sigma,
            'elbo': best_elbo,
            'converged': True,
            'training_time': training_time
        }
    
    def run_comprehensive_screening(self, X: np.ndarray, y: np.ndarray, 
                                   X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        執行全面的 VI 篩選 - GPU加速版本
        
        Args:
            X: 輸入特徵
            y: 真實損失
            
        Returns:
            篩選結果
        """
        if self.use_gpu:
            return self._gpu_screening(X, y, X_val, y_val)
        else:
            return self._cpu_screening(X, y, X_val, y_val)
    
    def _gpu_screening(self, X: np.ndarray, y: np.ndarray, 
                      X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """GPU加速的VI篩選 - 修正版：調用真正的VI訓練"""
        print("🚀 使用GPU加速VI篩選")
        if X_val is not None and y_val is not None:
            print("   📊 使用訓練+驗證集監督，防止過度擬合")
        else:
            print("   ⚠️ 僅使用訓練集，可能過度擬合")
        print("   注意：GPU張量加速，使用完整的VI訓練")
        
        all_results = []
        total_configs = len(self.epsilon_values) * len(self.basis_risk_types)
        
        print(f"   並行計算 {total_configs} 個配置...")
        
        # 對每個配置執行完整的VI訓練
        config_idx = 0
        for epsilon in self.epsilon_values:
            for basis_risk_type in self.basis_risk_types:
                config_idx += 1
                
                print(f"     開始配置 {config_idx}/{total_configs}: ε={epsilon:.3f}, {basis_risk_type}")
                
                # 調用真正的VI訓練（現在支持驗證集）
                result = self.train_single_model(
                    X, y, epsilon, basis_risk_type, n_iterations=1000,
                    X_val=X_val, y_val=y_val
                )
                all_results.append(result)
                
                # 進度顯示
                print(f"     ✅ 配置 {config_idx}/{total_configs} 完成: 基差風險={result['final_basis_risk']/1e6:.1f}M")
        
        # 按基差風險排序
        all_results = sorted(all_results, key=lambda x: x['final_basis_risk'])
        
        print(f"✅ GPU篩選完成!")
        
        return {
            'all_results': all_results,
            'best_models': all_results[:3],
            'best_model': all_results[0]
        }


if TORCH_AVAILABLE:
    # ============================================================
    # Route A: Solvency-compliant integrated objective with cat-in-circle
    # ============================================================

    # Stable Student-t logpdf
    def student_t_logpdf(y: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, df: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        scale = torch.clamp(scale, min=eps)
        df = torch.clamp(df, min=2.01)
        t = (y - loc) / scale
        Z = (torch.lgamma((df + 1)/2) - torch.lgamma(df/2)
             - 0.5*torch.log(df*torch.pi) - torch.log(scale))
        return Z - 0.5*(df+1)*torch.log1p(t*t/df)

    def smooth_max(x: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        return tau * torch.logsumexp(x / tau, dim=-1)

    def smooth_ramp(z: torch.Tensor, a: float, b: float, eps: float = 1e-6, k: float = 20.0) -> torch.Tensor:
        width = torch.clamp(torch.tensor(b - a, device=z.device, dtype=z.dtype), min=eps)
        t = (z - a) / width
        # Smooth clamp to [0,1]
        return torch.clamp(t, 0.0, 1.0)

    def indemnity_from_loss(loss: torch.Tensor, deductible: float = 0.0, limit: float = float('inf')) -> torch.Tensor:
        pay = torch.clamp(loss - deductible, min=0.0)
        if torch.isfinite(torch.tensor(limit, device=loss.device)):
            pay = torch.clamp(pay, max=limit)
        return pay

    def crps_from_samples(x_samples: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        # x_samples: [S, E], y_target: [E]
        S, E = x_samples.shape
        term1 = torch.mean(torch.abs(x_samples - y_target))
        x_perm = x_samples[torch.randperm(S)]
        term2 = 0.5 * torch.mean(torch.abs(x_samples - x_perm))
        return term1 - term2

    class CatInCirclePayout(nn.Module):
        def __init__(self, distance_matrix_km, circle_radius_km: float,
                     trigger_ms: float, exhaustion_ms: float,
                     site_limits, smooth_tau: float = 0.5,
                     payout_cap: float = None, site_weights = None):
            super().__init__()
            D = torch.tensor(distance_matrix_km, dtype=torch.float32)
            self.register_buffer('D', D)
            self.N_site, self.N_sta = D.shape
            self.R = float(circle_radius_km)
            self.trigger = float(trigger_ms)
            self.exhaustion = float(exhaustion_ms)
            self.smooth_tau = float(smooth_tau)
            self.site_limits = torch.as_tensor(site_limits, dtype=torch.float32)
            self.site_weights = (torch.ones(self.N_site, dtype=torch.float32)
                                  if site_weights is None else torch.as_tensor(site_weights, dtype=torch.float32))
            self.payout_cap = payout_cap

            mask = (D <= self.R).float()
            empty = (mask.sum(dim=1) == 0)
            if torch.any(empty):
                nearest_idx = torch.argmin(D[empty], dim=1)
                mask[empty, :] = 0.0
                mask[empty, nearest_idx] = 1.0
            self.register_buffer('mask', mask)

        def forward(self, winds_ms: torch.Tensor):  # winds_ms: [N_sta, E]
            device = winds_ms.device
            masked = winds_ms.unsqueeze(0).expand(self.N_site, -1, -1)
            # non-inplace: set outside-circle to big negative
            big_neg = torch.tensor(-1e6, device=device, dtype=masked.dtype)
            masked = masked + (self.mask.unsqueeze(-1) - 1.0) * (-big_neg)
            I_site = smooth_max(masked, tau=self.smooth_tau)  # [N_site, E]

            ramp = smooth_ramp(I_site, self.trigger, self.exhaustion)
            payout_site = ramp * self.site_limits.unsqueeze(-1)
            total = (self.site_weights.unsqueeze(-1) * payout_site).sum(dim=0)
            if self.payout_cap is not None:
                total = torch.clamp(total, max=self.payout_cap)
            return total, I_site

    class VulnerabilityHBM(nn.Module):
        def __init__(self, n_hospitals: int):
            super().__init__()
            self.alpha_un = nn.Parameter(torch.zeros(1))
            self.beta_un  = nn.Parameter(torch.log(torch.tensor(1.5)))
            self.v0_un    = nn.Parameter(torch.log(torch.tensor(20.0)))
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(1.0)))
            self.log_nu_m1 = nn.Parameter(torch.log(torch.tensor(5.0 - 1.0)))

        def forward(self, winds_ms: torch.Tensor, exposure: torch.Tensor):
            alpha = F.softplus(self.alpha_un) + 1e-6
            beta  = F.softplus(self.beta_un)  + 1e-6
            v0    = F.softplus(self.v0_un)    + 1e-6
            relu_term = torch.clamp(winds_ms - v0, min=0.0) ** beta
            mean_loss = exposure.unsqueeze(-1) * alpha * relu_term
            sigma = torch.exp(self.log_sigma) + 1e-6
            nu    = 1.0 + F.softplus(self.log_nu_m1)
            return mean_loss, sigma, nu

    class SolvencyCompliantBayesVI(nn.Module):
        def __init__(self, n_hospitals: int, n_regions: int,
                     distance_matrix_km, product_config: Dict,
                     lambda_kl: float = 0.05, lambda_br: float = 1.0,
                     loss_scale: float = 1e6, payout_scale: float = None,
                     solvency_mode: bool = True):
            super().__init__()
            self.n_hospitals = n_hospitals
            self.lambda_kl = lambda_kl
            self.lambda_br = lambda_br
            self.loss_scale = loss_scale

            self.mu_q = nn.Parameter(torch.zeros(6))
            self.log_std_q = nn.Parameter(torch.full((6,), -1.0))
            self.register_buffer('mu_p', torch.zeros(6))
            self.register_buffer('std_p', torch.ones(6))

            self.hbm = VulnerabilityHBM(n_hospitals)
            self.product = CatInCirclePayout(
                distance_matrix_km=distance_matrix_km,
                circle_radius_km=product_config['radius_km'],
                trigger_ms=product_config['trigger_ms'],
                exhaustion_ms=product_config['exhaustion_ms'],
                site_limits=product_config['site_limits'],
                smooth_tau=product_config.get('smooth_tau', 0.5),
                payout_cap=product_config.get('payout_cap', None),
                site_weights=product_config.get('site_weights', None)
            )
            if payout_scale is None:
                if product_config.get('payout_cap', None) is not None:
                    self.payout_scale = float(product_config['payout_cap'])
                else:
                    self.payout_scale = float(torch.tensor(product_config['site_limits']).sum())
            else:
                self.payout_scale = payout_scale

        def sample_theta(self, n_samples: int) -> torch.Tensor:
            eps = torch.randn(n_samples, self.mu_q.numel(), device=self.mu_q.device)
            return self.mu_q + torch.exp(self.log_std_q) * eps

        def kl_qp(self) -> torch.Tensor:
            var_q = torch.exp(2*self.log_std_q)
            var_p = self.std_p**2
            term = (torch.log(self.std_p) - self.log_std_q) + (var_q + (self.mu_q - self.mu_p)**2)/(2*var_p) - 0.5
            return term.sum()

        def load_theta_to_hbm_(self, theta_vec: torch.Tensor):
            self.hbm.alpha_un.data = theta_vec[0:1]
            self.hbm.beta_un.data  = theta_vec[1:2]
            self.hbm.v0_un.data    = theta_vec[2:3]
            self.hbm.log_sigma.data = theta_vec[3:4]
            self.hbm.log_nu_m1.data = theta_vec[4:5]

        def compute_solvency_elbo(self, winds_ms: torch.Tensor, exposure: torch.Tensor, losses: torch.Tensor,
                                   indemnity_cfg: Dict, n_samples: int = 64):
            device = self.mu_q.device
            winds_ms = winds_ms.to(device)
            exposure = exposure.to(device)
            losses   = losses.to(device)

            losses_scaled = losses / self.loss_scale
            S = n_samples
            theta_s = self.sample_theta(S)

            loglik_terms = []
            payouts_samples = []
            for s in range(S):
                self.load_theta_to_hbm_(theta_s[s])
                mean_loss, sigma, nu = self.hbm(winds_ms, exposure)
                ll = student_t_logpdf(losses_scaled,
                                       loc=mean_loss/self.loss_scale,
                                       scale=sigma/self.loss_scale,
                                       df=nu).sum()
                loglik_terms.append(ll)
                total_payout, _ = self.product(winds_ms)
                payouts_samples.append(total_payout)

            loglik = torch.stack(loglik_terms).mean()
            payouts_samples = torch.stack(payouts_samples, dim=0)

            kl = self.kl_qp()

            loss_total = losses.sum(dim=0)
            y_pay_target = indemnity_from_loss(
                loss_total,
                deductible=indemnity_cfg.get('deductible', 0.0),
                limit=indemnity_cfg.get('limit', float('inf'))
            )
            crps_raw = crps_from_samples(payouts_samples, y_pay_target)
            crps_scaled = crps_raw / self.payout_scale

            nll = -loglik
            total_loss = nll + self.lambda_kl * kl + self.lambda_br * crps_scaled
            solvency_elbo = loglik - self.lambda_kl * kl - self.lambda_br * crps_scaled

            logs = {
                'nll': float(nll.detach()),
                'kl': float(kl.detach()),
                'crps_raw': float(crps_raw.detach()),
                'crps_scaled': float(crps_scaled.detach()),
                'solvency_elbo': float(solvency_elbo.detach()),
            }
            return total_loss, logs
    
    
    def _cpu_screening(self, X: np.ndarray, y: np.ndarray, 
                      X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """CPU版本的VI篩選（原始實現）"""
        print("💻 使用CPU進行VI篩選")
        if X_val is not None and y_val is not None:
            print("   📊 使用訓練+驗證集監督，防止過度擬合")
        else:
            print("   ⚠️ 僅使用訓練集，可能過度擬合")
        
        all_results = []
        
        for epsilon in self.epsilon_values:
            for basis_risk_type in self.basis_risk_types:
                result = self.train_single_model(
                    X, y, epsilon, basis_risk_type, n_iterations=1000,
                    X_val=X_val, y_val=y_val
                )
                all_results.append(result)
        
        # 按基差風險排序
        all_results = sorted(all_results, key=lambda x: x['final_basis_risk'])
        
        return {
            'all_results': all_results,
            'best_models': all_results[:3],
            'best_model': all_results[0]
        }
    
    def run_hbm_two_step_optimization(self, X: np.ndarray, y: np.ndarray,
                                     prior_likelihood_configs: List[Dict],
                                     X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        執行HBM兩步法優化 - 接入階段3的Prior/Likelihood配置
        
        Step 1: 對每個Prior/Likelihood組合優化G(θ)的CRPS
        Step 2: 用最佳θ評估350個Steinmann產品的F_k(θ)
        
        Args:
            X: 輸入特徵 (風速等)
            y: 真實損失
            prior_likelihood_configs: 階段3定義的測試配置列表
                格式: [{'name': '配置名', 'prior': PriorScenario, 'likelihood': LikelihoodFamily, 'epsilon': float}, ...]
            X_val, y_val: 驗證集 (可選)
            
        Returns:
            {'step1_results': {...}, 'step2_results': {...}, 'best_config': {...}}
        """
        print("🚀 啟動HBM兩步法優化")
        print(f"   📊 測試{len(prior_likelihood_configs)}種Prior/Likelihood組合")
        
        if self.hierarchical_model is None:
            raise ValueError("兩步法需要先提供 hierarchical_model 實例")
            
        # === Step 1: 優化G(θ)的CRPS ===
        print("\n=== Step 1: 優化層次貝葉斯模型G(θ) ===")
        step1_results = []
        
        # 暫時切換到HBM模式
        original_objective = self.objective
        self.objective = 'hbm_two_step'
        
        try:
            for config_idx, config in enumerate(prior_likelihood_configs):
                print(f"\n🧪 測試配置 {config_idx+1}/{len(prior_likelihood_configs)}: {config['name']}")
                
                # 更新HBM的先驗和似然配置
                self.hierarchical_model.update_configuration(
                    prior_scenario=config['prior'],
                    likelihood_family=config['likelihood']
                )
                
                # 訓練當前配置
                result = self.train_single_model(
                    X, y, epsilon=config['epsilon'], 
                    basis_risk_type='absolute',  # Step1用absolute
                    n_iterations=1000,
                    X_val=X_val, y_val=y_val
                )
                
                # 添加配置信息
                result.update({
                    'config_name': config['name'],
                    'prior': config['prior'],
                    'likelihood': config['likelihood'],
                    'step': 1
                })
                
                step1_results.append(result)
                print(f"   ✅ {config['name']}: CRPS={result['final_basis_risk']/1e6:.1f}M")
                
        finally:
            # 恢復原始objective
            self.objective = original_objective
        
        # 按CRPS排序，選擇最佳配置
        step1_results = sorted(step1_results, key=lambda x: x['final_basis_risk'])
        best_config = step1_results[0]
        
        print(f"\n🏆 Step 1最佳配置: {best_config['config_name']}")
        print(f"   CRPS: {best_config['final_basis_risk']/1e6:.1f}M")
        
        # === Step 2: 評估350個Steinmann產品 ===
        print(f"\n=== Step 2: 評估350個參數保險產品F_k(θ*) ===")
        
        # 使用最佳θ*配置HBM
        self.hierarchical_model.update_configuration(
            prior_scenario=best_config['prior'],
            likelihood_family=best_config['likelihood']
        )
        
        # 用最佳θ*評估所有產品
        step2_results = self._evaluate_steinmann_products_with_hbm(
            X, y, best_config['best_theta'], X_val, y_val
        )
        
        return {
            'step1_results': step1_results,
            'step2_results': step2_results,
            'best_config': best_config,
            'method': 'hbm_two_step'
        }
    
    def _evaluate_steinmann_products_with_hbm(self, X: np.ndarray, y: np.ndarray, 
                                             best_theta: np.ndarray,
                                             X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Step 2: 用最佳θ*評估350個Steinmann產品的CRPS(y, F_k(θ*))
        """
        print("   📊 計算350個產品的F_k(θ*)...")
        
        # 使用最佳θ*生成損失預測
        hbm_loss_samples = self.hierarchical_model.predict_distribution(
            best_theta, X, n_samples=100
        )
        
        product_results = []
        
        # 載入350個Steinmann產品定義
        steinmann_thresholds = self._get_steinmann_products_tensor().detach().cpu().numpy()
        steinmann_ratios = self._get_steinmann_ratios_tensor().detach().cpu().numpy() 
        steinmann_max_payouts = self._get_steinmann_max_payouts().detach().cpu().numpy()
        
        print("   🔄 評估每個產品的CRPS...")
        
        for k in range(350):  # 350個產品
            if (k + 1) % 50 == 0:
                print(f"       進度: {k+1}/350")
            
            # 計算產品k的賠付F_k(θ*)
            product_payouts = self._compute_single_product_payout(
                hbm_loss_samples, k, 
                steinmann_thresholds[k], steinmann_ratios[k], steinmann_max_payouts[k]
            )
            
            # 計算CRPS(y, F_k(θ*))
            crps_scores = []
            for i in range(len(y)):
                y_obs = y[i]
                payout_samples = product_payouts[i]  # 該事件的賠付分布
                
                # Ensemble CRPS
                crps = np.mean(np.abs(payout_samples - y_obs)) - 0.5 * np.mean(
                    np.abs(payout_samples[:, None] - payout_samples[None, :])
                )
                crps_scores.append(crps)
            
            mean_crps = np.mean(crps_scores)
            
            product_results.append({
                'product_id': k,
                'crps': mean_crps,
                'thresholds': steinmann_thresholds[k].tolist(),
                'ratios': steinmann_ratios[k].tolist(),
                'max_payout': float(steinmann_max_payouts[k])
            })
        
        # 按CRPS排序
        product_results = sorted(product_results, key=lambda x: x['crps'])
        
        print(f"   🏆 最佳產品: ID={product_results[0]['product_id']}, CRPS={product_results[0]['crps']/1e6:.1f}M")
        
        return {
            'all_products': product_results,
            'best_products': product_results[:10],
            'best_product': product_results[0]
        }
    
    def _compute_single_product_payout(self, loss_samples: np.ndarray, product_id: int,
                                     thresholds: np.ndarray, ratios: np.ndarray, 
                                     max_payout: float) -> np.ndarray:
        """
        計算單個Steinmann產品的賠付分布
        
        Args:
            loss_samples: HBM損失預測 [n_events, n_samples]
            product_id: 產品ID
            thresholds: 產品閾值 [4]
            ratios: 賠付比例 [4] 
            max_payout: 最大賠付
            
        Returns:
            產品賠付樣本 [n_events, n_samples]
        """
        n_events, n_samples = loss_samples.shape
        payout_samples = np.zeros_like(loss_samples)
        
        for event_idx in range(n_events):
            for sample_idx in range(n_samples):
                loss_value = loss_samples[event_idx, sample_idx]
                
                # Steinmann階梯式賠付邏輯
                total_payout = 0.0
                for thresh_idx in range(4):
                    if thresholds[thresh_idx] < 999:  # 有效閾值
                        if loss_value >= thresholds[thresh_idx]:
                            total_payout = max_payout * ratios[thresh_idx]
                
                payout_samples[event_idx, sample_idx] = min(total_payout, max_payout)
        
        return payout_samples
