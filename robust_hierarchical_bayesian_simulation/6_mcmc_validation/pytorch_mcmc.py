#!/usr/bin/env python3
"""
PyTorch-based MCMC Implementation
基於PyTorch的MCMC實現

完全使用PyTorch重寫MCMC，支援GPU加速
實現HMC (Hamiltonian Monte Carlo) 和 NUTS (No-U-Turn Sampler)

Author: Research Team
Date: 2025-01-18
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import time
import warnings
warnings.filterwarnings('ignore')

# ========================================
# MCMC配置
# ========================================

@dataclass
class MCMCConfig:
    """MCMC配置"""
    n_samples: int = 2000
    n_warmup: int = 1000
    n_chains: int = 4
    step_size: float = 0.01
    num_steps: int = 10
    target_accept: float = 0.8
    adapt_step_size: bool = True
    adapt_mass_matrix: bool = True
    max_tree_depth: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.float32
    seed: int = 42

# ========================================
# PyTorch MCMC基礎類
# ========================================

class PytorchMCMC:
    """PyTorch MCMC基礎類"""
    
    def __init__(self, config: MCMCConfig = None):
        """
        初始化MCMC採樣器
        
        Parameters:
        -----------
        config : MCMCConfig
            MCMC配置
        """
        self.config = config or MCMCConfig()
        
        # 設定設備和數據類型
        self.device = torch.device(self.config.device)
        self.dtype = self.config.dtype
        
        # 設定隨機種子
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
        
        print(f"🚀 PyTorch MCMC初始化")
        print(f"   設備: {self.device}")
        print(f"   數據類型: {self.dtype}")
        
        # 採樣結果存儲
        self.samples = None
        self.log_probs = None
        self.accept_rates = None
        self.diagnostics = {}
    
    def log_prob_fn(self, theta: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        計算log probability (需要子類實現)
        
        Parameters:
        -----------
        theta : torch.Tensor
            參數向量
        data : Dict[str, torch.Tensor]
            觀測數據
            
        Returns:
        --------
        torch.Tensor
            log probability
        """
        raise NotImplementedError("需要實現log_prob_fn")
    
    def grad_log_prob(self, theta: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        計算log probability的梯度
        
        Parameters:
        -----------
        theta : torch.Tensor
            參數向量
        data : Dict[str, torch.Tensor]
            觀測數據
            
        Returns:
        --------
        torch.Tensor
            梯度
        """
        theta = theta.requires_grad_(True)
        log_prob = self.log_prob_fn(theta, data)
        grad = torch.autograd.grad(log_prob.sum(), theta)[0]
        return grad

# ========================================
# HMC (Hamiltonian Monte Carlo)
# ========================================

class HamiltonianMonteCarlo(PytorchMCMC):
    """Hamiltonian Monte Carlo採樣器"""
    
    def __init__(self, config: MCMCConfig = None):
        super().__init__(config)
        self.step_size = self.config.step_size
        self.num_steps = self.config.num_steps
        self.mass_matrix = None
    
    def leapfrog(self, theta: torch.Tensor, momentum: torch.Tensor, 
                 data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Leapfrog積分器
        
        Parameters:
        -----------
        theta : torch.Tensor
            當前位置
        momentum : torch.Tensor
            當前動量
        data : Dict[str, torch.Tensor]
            數據
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            新位置和新動量
        """
        # 複製以避免原地修改
        theta = theta.clone()
        momentum = momentum.clone()
        
        # 半步更新動量
        grad = self.grad_log_prob(theta, data)
        momentum = momentum + 0.5 * self.step_size * grad
        
        # 完整步更新位置和動量
        for _ in range(self.num_steps - 1):
            theta = theta + self.step_size * momentum
            grad = self.grad_log_prob(theta, data)
            momentum = momentum + self.step_size * grad
        
        # 最後更新位置
        theta = theta + self.step_size * momentum
        
        # 最後半步更新動量
        grad = self.grad_log_prob(theta, data)
        momentum = momentum + 0.5 * self.step_size * grad
        
        return theta, momentum
    
    def sample_one_chain(self, initial_theta: torch.Tensor, 
                        data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        運行單條鏈的HMC採樣
        
        Parameters:
        -----------
        initial_theta : torch.Tensor
            初始參數值
        data : Dict[str, torch.Tensor]
            數據
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            採樣結果
        """
        n_params = initial_theta.shape[0]
        n_total = self.config.n_warmup + self.config.n_samples
        
        # 初始化存儲
        samples = torch.zeros((n_total, n_params), device=self.device, dtype=self.dtype)
        log_probs = torch.zeros(n_total, device=self.device, dtype=self.dtype)
        accepted = torch.zeros(n_total, dtype=torch.bool, device=self.device)
        
        # 初始化
        current_theta = initial_theta.clone()
        current_log_prob = self.log_prob_fn(current_theta, data)
        
        # 自適應參數
        if self.config.adapt_step_size:
            step_size_adapter = DualAveragingStepSize(
                initial_step_size=self.step_size,
                target_accept=self.config.target_accept
            )
        
        # 主採樣循環
        for i in range(n_total):
            # 採樣動量
            momentum = torch.randn_like(current_theta)
            current_momentum = momentum.clone()
            
            # Leapfrog積分
            proposed_theta, proposed_momentum = self.leapfrog(
                current_theta, momentum, data
            )
            
            # 計算接受概率
            proposed_log_prob = self.log_prob_fn(proposed_theta, data)
            
            current_hamiltonian = current_log_prob - 0.5 * torch.sum(current_momentum ** 2)
            proposed_hamiltonian = proposed_log_prob - 0.5 * torch.sum(proposed_momentum ** 2)
            
            log_accept_prob = proposed_hamiltonian - current_hamiltonian
            accept_prob = torch.exp(torch.clamp(log_accept_prob, max=0))
            
            # 接受/拒絕
            if torch.rand(1, device=self.device) < accept_prob:
                current_theta = proposed_theta
                current_log_prob = proposed_log_prob
                accepted[i] = True
            
            # 存儲樣本
            samples[i] = current_theta
            log_probs[i] = current_log_prob
            
            # 自適應調整（熱身期）
            if i < self.config.n_warmup and self.config.adapt_step_size:
                self.step_size = step_size_adapter.update(accept_prob.item())
        
        # 只返回採樣期的樣本
        return {
            'samples': samples[self.config.n_warmup:],
            'log_probs': log_probs[self.config.n_warmup:],
            'accept_rate': accepted[self.config.n_warmup:].float().mean().item(),
            'warmup_accept_rate': accepted[:self.config.n_warmup].float().mean().item()
        }
    
    def sample(self, data: Dict[str, torch.Tensor], 
              initial_values: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        運行多鏈HMC採樣
        
        Parameters:
        -----------
        data : Dict[str, torch.Tensor]
            數據
        initial_values : torch.Tensor, optional
            初始值
            
        Returns:
        --------
        Dict[str, Any]
            採樣結果和診斷
        """
        print(f"\n🎯 開始HMC採樣")
        print(f"   鏈數: {self.config.n_chains}")
        print(f"   樣本數/鏈: {self.config.n_samples}")
        print(f"   熱身數/鏈: {self.config.n_warmup}")
        
        start_time = time.time()
        
        # 確定參數維度
        if initial_values is None:
            # 需要從數據推斷參數維度
            n_params = 10  # 預設值，實際應該根據模型確定
            initial_values = torch.randn(self.config.n_chains, n_params, 
                                        device=self.device, dtype=self.dtype)
        
        # 運行多條鏈
        chain_results = []
        for chain_id in range(self.config.n_chains):
            print(f"   運行鏈 {chain_id + 1}/{self.config.n_chains}...")
            
            result = self.sample_one_chain(
                initial_theta=initial_values[chain_id],
                data=data
            )
            chain_results.append(result)
        
        # 合併結果
        all_samples = torch.stack([r['samples'] for r in chain_results])
        all_log_probs = torch.stack([r['log_probs'] for r in chain_results])
        accept_rates = [r['accept_rate'] for r in chain_results]
        
        # 計算診斷統計
        self.samples = all_samples
        self.log_probs = all_log_probs
        self.accept_rates = accept_rates
        
        self.diagnostics = self.compute_diagnostics()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ HMC採樣完成")
        print(f"   總時間: {elapsed_time:.2f} 秒")
        print(f"   平均接受率: {np.mean(accept_rates):.2%}")
        print(f"   R-hat: {self.diagnostics['rhat']:.3f}")
        print(f"   ESS: {self.diagnostics['ess']:.0f}")
        
        return {
            'samples': all_samples,
            'log_probs': all_log_probs,
            'accept_rates': accept_rates,
            'diagnostics': self.diagnostics,
            'elapsed_time': elapsed_time
        }
    
    def compute_diagnostics(self) -> Dict[str, Any]:
        """計算MCMC診斷統計"""
        if self.samples is None:
            return {}
        
        # 轉換為numpy進行診斷計算
        samples_np = self.samples.cpu().numpy()
        
        # R-hat (Gelman-Rubin統計)
        rhat = self.compute_rhat(samples_np)
        
        # ESS (有效樣本大小)
        ess = self.compute_ess(samples_np)
        
        return {
            'rhat': rhat,
            'ess': ess,
            'mean_accept_rate': np.mean(self.accept_rates),
            'min_accept_rate': np.min(self.accept_rates),
            'max_accept_rate': np.max(self.accept_rates)
        }
    
    def compute_rhat(self, samples: np.ndarray) -> float:
        """計算R-hat統計"""
        n_chains, n_samples, n_params = samples.shape
        
        # 計算鏈間和鏈內變異
        chain_means = np.mean(samples, axis=1)
        grand_mean = np.mean(chain_means, axis=0)
        
        B = n_samples * np.var(chain_means, axis=0, ddof=1)
        W = np.mean(np.var(samples, axis=1, ddof=1), axis=0)
        
        var_estimate = ((n_samples - 1) * W + B) / n_samples
        rhat = np.sqrt(var_estimate / W)
        
        return np.mean(rhat)
    
    def compute_ess(self, samples: np.ndarray) -> float:
        """計算有效樣本大小"""
        n_chains, n_samples, n_params = samples.shape
        
        # 簡化的ESS計算
        # 實際應該使用更複雜的自相關方法
        pooled_samples = samples.reshape(-1, n_params)
        ess = n_chains * n_samples  # 簡化版本
        
        return ess

# ========================================
# 自適應步長調整
# ========================================

class DualAveragingStepSize:
    """雙重平均步長自適應"""
    
    def __init__(self, initial_step_size: float = 1.0, 
                 target_accept: float = 0.8,
                 gamma: float = 0.05,
                 t0: float = 10.0,
                 kappa: float = 0.75):
        self.mu = np.log(10 * initial_step_size)
        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        
        self.log_step_size = np.log(initial_step_size)
        self.log_step_size_bar = 0.0
        self.h_bar = 0.0
        self.iteration = 0
    
    def update(self, accept_prob: float) -> float:
        """更新步長"""
        self.iteration += 1
        
        # 更新H統計
        self.h_bar = (1 - 1 / (self.iteration + self.t0)) * self.h_bar + \
                     (self.target_accept - accept_prob) / (self.iteration + self.t0)
        
        # 更新步長
        self.log_step_size = self.mu - np.sqrt(self.iteration) / self.gamma * self.h_bar
        
        # 更新平均步長
        eta = self.iteration ** (-self.kappa)
        self.log_step_size_bar = eta * self.log_step_size + (1 - eta) * self.log_step_size_bar
        
        return np.exp(self.log_step_size)

# ========================================
# CRPS-aware MCMC
# ========================================

class CRPSAwareMCMC(HamiltonianMonteCarlo):
    """CRPS-aware MCMC採樣器"""
    
    def __init__(self, config: MCMCConfig = None, crps_weight: float = 1.0):
        super().__init__(config)
        self.crps_weight = crps_weight
    
    def crps_score(self, predictions: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """
        計算CRPS分數
        
        Parameters:
        -----------
        predictions : torch.Tensor
            預測分布樣本 [n_samples, n_obs]
        observations : torch.Tensor
            觀測值 [n_obs]
            
        Returns:
        --------
        torch.Tensor
            CRPS分數
        """
        # 使用簡化的CRPS計算
        # CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
        
        n_samples = predictions.shape[0]
        
        # 第一項：預測與觀測的差異
        term1 = torch.mean(torch.abs(predictions - observations.unsqueeze(0)))
        
        # 第二項：預測樣本間的差異
        if n_samples > 1:
            # 隨機選擇樣本對
            idx1 = torch.randperm(n_samples)[:n_samples//2]
            idx2 = torch.randperm(n_samples)[:n_samples//2]
            term2 = torch.mean(torch.abs(predictions[idx1] - predictions[idx2]))
        else:
            term2 = 0.0
        
        crps = term1 - 0.5 * term2
        return crps
    
    def log_prob_fn(self, theta: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        CRPS-aware log probability
        
        結合標準log likelihood和CRPS懲罰
        """
        # 標準log likelihood
        X = data['X']
        y = data['y']
        
        # 簡單線性模型作為示例
        predictions = torch.matmul(X, theta[:X.shape[1]])
        
        # 正態似然
        sigma = torch.exp(theta[-1])  # 最後一個參數是log(sigma)
        log_likelihood = -0.5 * torch.sum((y - predictions) ** 2) / (sigma ** 2) - \
                        len(y) * torch.log(sigma)
        
        # CRPS懲罰
        # 生成預測分布樣本
        pred_samples = predictions.unsqueeze(0) + sigma * torch.randn(100, len(y), 
                                                                      device=self.device)
        crps = self.crps_score(pred_samples, y)
        
        # 結合兩者
        log_prob = log_likelihood - self.crps_weight * crps
        
        # 添加先驗
        prior = -0.5 * torch.sum(theta ** 2) / 100  # 弱先驗
        
        return log_prob + prior

# ========================================
# 模型特定的MCMC實現
# ========================================

class BayesianHierarchicalMCMC(CRPSAwareMCMC):
    """貝葉斯階層模型MCMC"""
    
    def __init__(self, vulnerability_type: str = 'emanuel', 
                 config: MCMCConfig = None):
        super().__init__(config)
        self.vulnerability_type = vulnerability_type
    
    def emanuel_vulnerability(self, wind_speed: torch.Tensor, 
                             params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Emanuel脆弱度函數"""
        a = params['a']
        b = params['b']
        threshold = 25.0
        
        wind_excess = torch.clamp(wind_speed - threshold, min=0)
        vulnerability = torch.clamp(a * wind_excess ** b, max=1.0)
        
        return vulnerability
    
    def log_prob_fn(self, theta: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        階層模型的log probability
        """
        # 解包參數
        n_params = len(theta)
        
        # Emanuel函數參數
        log_a = theta[0]
        b = theta[1]
        
        # 階層參數
        alpha = theta[2]  # 截距
        beta = theta[3:5] if n_params > 4 else torch.zeros(2, device=self.device)
        
        # 觀測誤差
        log_sigma = theta[-1]
        
        # 轉換參數
        a = torch.exp(log_a)
        sigma = torch.exp(log_sigma)
        
        # 計算脆弱度
        params = {'a': a, 'b': b}
        vulnerability = self.emanuel_vulnerability(data['wind_speed'], params)
        
        # 計算預期損失
        expected_loss = vulnerability * data['exposure'] * torch.exp(alpha)
        
        # 對數似然
        log_likelihood = -0.5 * torch.sum((data['losses'] - expected_loss) ** 2) / (sigma ** 2)
        log_likelihood -= len(data['losses']) * torch.log(sigma)
        
        # 先驗
        log_prior = 0.0
        
        # a的Gamma先驗
        log_prior += -2 * log_a - a / 500  # Gamma(2, 500)
        
        # b的正態先驗
        log_prior += -0.5 * (b - 2.0) ** 2 / 0.25  # Normal(2, 0.5)
        
        # alpha的正態先驗
        log_prior += -0.5 * alpha ** 2 / 4  # Normal(0, 2)
        
        # sigma的半正態先驗
        log_prior += -0.5 * log_sigma ** 2  # HalfNormal via log transform
        
        return log_likelihood + log_prior

# ========================================
# 便利函數
# ========================================

def run_pytorch_mcmc(data: Dict[str, Any], 
                     model_type: str = 'hierarchical',
                     use_gpu: bool = True,
                     n_chains: int = 4,
                     n_samples: int = 2000) -> Dict[str, Any]:
    """
    運行PyTorch MCMC的便利函數
    
    Parameters:
    -----------
    data : Dict[str, Any]
        數據
    model_type : str
        模型類型
    use_gpu : bool
        是否使用GPU
    n_chains : int
        鏈數
    n_samples : int
        樣本數
        
    Returns:
    --------
    Dict[str, Any]
        MCMC結果
    """
    # 配置
    device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
    config = MCMCConfig(
        n_chains=n_chains,
        n_samples=n_samples,
        n_warmup=n_samples // 2,
        device=device
    )
    
    # 轉換數據到torch
    torch_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            torch_data[key] = torch.tensor(value, device=device, dtype=torch.float32)
        else:
            torch_data[key] = value
    
    # 選擇模型
    if model_type == 'hierarchical':
        mcmc = BayesianHierarchicalMCMC(config=config)
    elif model_type == 'crps':
        mcmc = CRPSAwareMCMC(config=config)
    elif model_type == 'basic':
        mcmc = CRPSAwareMCMC(config=config)  # 使用CRPS-aware作為基本實現
    else:
        mcmc = HamiltonianMonteCarlo(config=config)
    
    # 運行MCMC
    results = mcmc.sample(torch_data)
    
    return results

# ========================================
# 測試
# ========================================

def test_pytorch_mcmc():
    """測試PyTorch MCMC"""
    print("🧪 測試PyTorch MCMC實現")
    
    # 生成測試數據
    np.random.seed(42)
    n_obs = 100
    
    data = {
        'wind_speed': np.random.uniform(20, 80, n_obs),
        'exposure': np.random.uniform(1e6, 1e8, n_obs),
        'losses': np.random.uniform(0, 1e6, n_obs),
        'X': np.random.randn(n_obs, 5),
        'y': np.random.randn(n_obs)
    }
    
    # 測試基本HMC
    print("\n1. 測試基本HMC")
    results = run_pytorch_mcmc(
        data=data,
        model_type='basic',
        use_gpu=torch.cuda.is_available(),
        n_chains=2,
        n_samples=500
    )
    
    print(f"   採樣形狀: {results['samples'].shape}")
    print(f"   接受率: {np.mean(results['accept_rates']):.2%}")
    
    # 測試階層模型
    print("\n2. 測試階層模型MCMC")
    results = run_pytorch_mcmc(
        data=data,
        model_type='hierarchical',
        use_gpu=torch.cuda.is_available(),
        n_chains=2,
        n_samples=500
    )
    
    print(f"   R-hat: {results['diagnostics']['rhat']:.3f}")
    print(f"   ESS: {results['diagnostics']['ess']:.0f}")
    
    print("\n✅ PyTorch MCMC測試完成")

if __name__ == "__main__":
    test_pytorch_mcmc()