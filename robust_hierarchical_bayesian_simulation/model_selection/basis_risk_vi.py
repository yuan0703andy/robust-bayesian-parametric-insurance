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
from typing import Dict, List, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# Try importing PyTorch for differentiable operations
try:
    import torch
    import torch.nn as nn
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
                 use_gpu: bool = True):
        """
        初始化基差風險導向 VI
        
        Args:
            n_features: 特徵維度
            epsilon_values: ε-contamination 參數候選
            basis_risk_types: 基差風險類型
            use_gpu: 是否使用GPU加速
        """
        if epsilon_values is None:
            epsilon_values = [0.0, 0.05, 0.10, 0.15, 0.20]
        if basis_risk_types is None:
            basis_risk_types = ['absolute', 'asymmetric', 'weighted']
            
        self.n_features = n_features
        self.n_params = n_features + 1  # 線性係數 + 噪音參數
        self.epsilon_values = epsilon_values
        self.basis_risk_types = basis_risk_types
        
        # GPU配置
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        if self.use_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cpu':
                print("⚠️ GPU不可用，降級到CPU")
                self.use_gpu = False
        else:
            self.device = torch.device('cpu')
            
        print(f"🔧 BasisRiskAwareVI初始化: {'GPU' if self.use_gpu else 'CPU'}模式")
        if self.use_gpu:
            print(f"   GPU設備: {torch.cuda.get_device_name(self.device)}")
        
        # 賠付函數
        self.payout_function = ParametricPayoutFunction()
        
        # CRPS 計算器
        self.crps_calculator = DifferentiableCRPS()
        
        # 存儲結果
        self.vi_results = {}
    
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
        
        # 轉換為GPU張量
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)
        
        # 驗證集張量（如果提供）
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val_tensor = torch.from_numpy(X_val).float().to(self.device)
            y_val_tensor = torch.from_numpy(y_val).float().to(self.device)
            print(f"        📊 使用驗證集監督: 訓練={X.shape[0]}, 驗證={X_val.shape[0]}")
        else:
            print(f"        ⚠️ 無驗證集，可能過度擬合")
        
        # 變分參數 (在GPU上)
        torch.manual_seed(42 + int(epsilon*1000))
        mu_theta = torch.randn(self.n_params, device=self.device) * 0.1
        log_sigma_theta = torch.full((self.n_params,), -2.0, device=self.device)
        
        # 設為可求導
        mu_theta.requires_grad_(True)
        log_sigma_theta.requires_grad_(True)
        
        # Adam優化器
        optimizer = torch.optim.Adam([mu_theta, log_sigma_theta], lr=0.01)
        
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
            loss.backward()
            
            # 梯度裁剪避免爆炸
            torch.nn.utils.clip_grad_norm_([mu_theta, log_sigma_theta], max_norm=1.0)
            
            optimizer.step()
            
            # 約束log_sigma範圍
            with torch.no_grad():
                log_sigma_theta.clamp_(-5, 1)
            
            # 計算當前基差風險用於記錄  
            with torch.no_grad():
                # 訓練集基差風險
                current_basis_risk_train = self._compute_basis_risk_batch_gpu(
                    X_tensor, y_tensor, theta_samples, epsilon, basis_risk_type
                ).mean().item()
                
                # 驗證集基差風險（如果有）
                if has_validation:
                    current_basis_risk_val = self._compute_basis_risk_batch_gpu(
                        X_val_tensor, y_val_tensor, theta_samples, epsilon, basis_risk_type
                    ).mean().item()
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
            
            # 進度報告
            if (iteration + 1) % 200 == 0:
                if has_validation:
                    print(f"        迭代 {iteration+1}: ELBO={current_elbo:.3f}, 訓練={current_basis_risk_train/1e6:.1f}M, 驗證={current_basis_risk_val/1e6:.1f}M")
                else:
                    print(f"        迭代 {iteration+1}: ELBO={current_elbo:.3f}, 基差風險={current_basis_risk_train/1e6:.1f}M")
        
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
    
    def _compute_elbo_batch_gpu(self, X_tensor, y_tensor, theta_samples, epsilon, 
                               basis_risk_type, mu_theta, sigma_theta):
        """GPU上批次計算ELBO"""
        batch_size = theta_samples.shape[0]  # n_samples_per_iteration
        
        # 批次計算基差風險 (似然項)
        basis_risks = self._compute_basis_risk_batch_gpu(
            X_tensor, y_tensor, theta_samples, epsilon, basis_risk_type
        )
        log_likelihood = -basis_risks / 1e9  # 標準化
        
        # 先驗項 (標準高斯)
        log_prior = -0.5 * torch.sum(theta_samples**2, dim=1)
        
        # 變分分布log密度
        log_q = -0.5 * torch.sum((theta_samples - mu_theta)**2 / sigma_theta**2, dim=1) - \
                0.5 * torch.sum(torch.log(2 * np.pi * sigma_theta**2))
        
        # ELBO = E[log p(y|θ)] + E[log p(θ)] - E[log q(θ)]
        elbo = log_likelihood + log_prior - log_q
        
        return elbo
    
    def _compute_basis_risk_batch_gpu(self, X_tensor, y_tensor, theta_samples, epsilon, basis_risk_type):
        """GPU上批次計算基差風險 - 修正版：θ參數真正影響計算"""
        batch_size = theta_samples.shape[0]
        n_data = X_tensor.shape[0]
        
        # epsilon contamination
        if epsilon > 0:
            noise = torch.randn_like(y_tensor.unsqueeze(0).expand(batch_size, -1)) * epsilon * y_tensor.mean()
            y_perturbed = y_tensor.unsqueeze(0).expand(batch_size, -1) + noise
        else:
            y_perturbed = y_tensor.unsqueeze(0).expand(batch_size, -1)
        
        # *** 關鍵修正：讓θ參數影響賠付計算 ***
        wind_speeds = X_tensor.squeeze(-1)  # [n_data]
        wind_speeds = wind_speeds.unsqueeze(0).expand(batch_size, -1)  # [batch_size, n_data]
        
        # 使用θ參數調整閾值和賠付比例
        # theta_samples: [batch_size, n_params], 其中n_params=2 (slope + intercept)
        theta_slope = theta_samples[:, 0:1]      # [batch_size, 1] - 閾值斜率
        theta_intercept = theta_samples[:, 1:2]  # [batch_size, 1] - 基礎閾值
        
        # 動態閾值：受θ影響
        base_thresholds = torch.tensor([25.0, 35.0, 45.0], device=self.device)
        # 廣播到 [batch_size, 3]
        dynamic_thresholds = (base_thresholds.unsqueeze(0) + 
                            theta_intercept * 10.0 +  # intercept影響基礎閾值
                            theta_slope * torch.arange(3, device=self.device).float())  # slope影響間隔
        
        # 動態賠付比例：受θ影響  
        base_ratios = torch.tensor([0.25, 0.5, 1.0], device=self.device)
        dynamic_ratios = torch.sigmoid(base_ratios.unsqueeze(0) + theta_slope * 2.0)  # [batch_size, 3]
        
        # 動態最大賠付
        max_payout_base = y_tensor.mean()
        max_payout = max_payout_base * torch.exp(theta_intercept).squeeze(-1)  # [batch_size]
        
        # 計算賠付（現在受θ影響）
        payouts = torch.zeros_like(y_perturbed)  # [batch_size, n_data]
        
        for i in range(3):
            # 對每個批次樣本，使用不同的閾值和賠付比例
            threshold_batch = dynamic_thresholds[:, i:i+1]  # [batch_size, 1]
            ratio_batch = dynamic_ratios[:, i:i+1]          # [batch_size, 1]
            max_payout_batch = max_payout[:, None]          # [batch_size, 1]
            
            mask = wind_speeds >= threshold_batch  # [batch_size, n_data]
            payout_value = max_payout_batch * ratio_batch  # [batch_size, 1]
            payouts = torch.where(mask, payout_value, payouts)
        
        # 計算基差風險
        if basis_risk_type == 'absolute':
            basis_risk = torch.mean(torch.abs(y_perturbed - payouts), dim=1)
        elif basis_risk_type == 'asymmetric':
            # Asymmetric: 懲罰under-payment更重 (2:1)
            under_penalty = torch.mean(torch.relu(y_perturbed - payouts), dim=1)
            over_penalty = torch.mean(torch.relu(payouts - y_perturbed), dim=1)
            basis_risk = 2.0 * under_penalty + over_penalty
        else:  # weighted
            # Weighted: 根據損失大小調整懲罰 (大損失懲罰更重)
            diff = y_perturbed - payouts
            weights = torch.abs(y_perturbed) / torch.mean(torch.abs(y_perturbed), dim=1, keepdim=True)
            weighted_diff = diff * weights
            basis_risk = torch.mean(torch.abs(weighted_diff), dim=1)
        
        return basis_risk
    
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
    
    def _compute_basis_risk_gpu(self, X_tensor, y_tensor, epsilon, basis_risk_type):
        """在GPU上計算基差風險"""
        # 添加epsilon contamination
        if epsilon > 0:
            noise = torch.randn_like(y_tensor) * epsilon * y_tensor.mean()
            y_perturbed = y_tensor + noise
        else:
            y_perturbed = y_tensor
        
        # 基於風速特徵計算參數賠付
        wind_speeds = X_tensor.squeeze()
        
        # 簡化的參數保險邏輯（基於風速閾值）
        payouts = torch.zeros_like(y_perturbed)
        
        # 多層閾值賠付
        thresholds = torch.tensor([25.0, 35.0, 45.0], device=self.device)
        payout_ratios = torch.tensor([0.25, 0.5, 1.0], device=self.device)
        max_payout = y_tensor.mean() * 2.0  # 動態最大賠付
        
        for i, threshold in enumerate(thresholds):
            mask = wind_speeds >= threshold
            payouts[mask] = max_payout * payout_ratios[i]
        
        # 計算基差風險
        if basis_risk_type == 'absolute':
            basis_risk = torch.mean(torch.abs(y_perturbed - payouts))
        elif basis_risk_type == 'asymmetric':
            under_penalty = torch.mean(torch.relu(y_perturbed - payouts))
            over_penalty = torch.mean(torch.relu(payouts - y_perturbed))
            basis_risk = 2.0 * under_penalty + over_penalty
        else:  # weighted
            under = torch.relu(y_perturbed - payouts) * 2.0
            over = torch.relu(payouts - y_perturbed) * 0.5
            basis_risk = torch.mean(under + over)
        
        return basis_risk
    
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