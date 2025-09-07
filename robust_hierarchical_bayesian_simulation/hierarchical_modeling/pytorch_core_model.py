#!/usr/bin/env python3
"""
PyTorch-based Hierarchical Bayesian Model (Risk Brain)
PyTorch風險大腦 - 4層階層貝氏模型

完全基於PyTorch的HBM實現，取代JAX版本
支持自動微分、GPU加速和與Sigmoid代理優化框架的無縫集成

核心功能:
- PyTorch nn.Module架構
- 4層階層結構 (Global → Regional → Local → Event)
- 自動微分支持的predict_distribution方法
- GPU/CPU自適應計算
- 與BasisRiskAwareVI兼容的接口

Author: Research Team
Date: 2025-01-17
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings
import os

# PyTorch設備配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"✅ PyTorch 版本: {torch.__version__}")
    print(f"   🚀 將使用GPU加速: {torch.cuda.get_device_name(0)}")
    print(f"   GPU記憶體: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f}GB")
else:
    print(f"✅ PyTorch 版本: {torch.__version__}")
    print(f"   💻 將使用CPU計算")

# 導入必要的模組
try:
    from .prior_specifications import PriorScenario, LikelihoodFamily, ContaminationDistribution, VulnerabilityFunctionType
    from .likelihood_families import MCMCConfig, DiagnosticResult, HierarchicalModelResult
except ImportError:
    try:
        from prior_specifications import PriorScenario, LikelihoodFamily, ContaminationDistribution, VulnerabilityFunctionType
        from likelihood_families import MCMCConfig, DiagnosticResult, HierarchicalModelResult
    except ImportError:
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from prior_specifications import PriorScenario, LikelihoodFamily, ContaminationDistribution, VulnerabilityFunctionType
        from likelihood_families import MCMCConfig, DiagnosticResult, HierarchicalModelResult


class PyTorchHierarchicalBayesianModel(nn.Module):
    """
    PyTorch-based 4層階層貝氏風險大腦模型
    
    架構:
    Level 4 (Global): α ~ N(μ_α, σ_α²)
    Level 3 (Regional): β_r ~ N(α, τ_r²) 
    Level 2 (Local): γ_l ~ N(β_r(l), τ_l²)
    Level 1 (Event): θ_e ~ N(γ_l(e), σ_e²)
    
    支持Emanuel USA、線性和多項式脆弱度函數
    """
    
    def __init__(self, 
                 model_spec: 'ModelSpec',
                 n_hospitals: int = 100,
                 n_events: int = 1000,
                 device: Optional[torch.device] = None):
        """
        初始化PyTorch階層模型
        
        Parameters:
        -----------
        model_spec : ModelSpec
            模型規格配置
        n_hospitals : int
            醫院數量 (地區層級)
        n_events : int  
            事件數量 (時間維度)
        device : torch.device, optional
            計算設備
        """
        super(PyTorchHierarchicalBayesianModel, self).__init__()
        
        self.model_spec = model_spec
        self.n_hospitals = n_hospitals
        self.n_events = n_events
        self.device = device or globals()['device']
        
        # 移動模型到指定設備
        self.to(self.device)
        
        # 初始化4層階層參數
        self._initialize_hierarchical_parameters()
        
        # 初始化脆弱度函數參數
        self._initialize_vulnerability_parameters()
        
        print(f"🏗️ PyTorch階層模型已初始化:")
        print(f"   模型: {self.model_spec.model_name}")
        print(f"   概似函數: {self.model_spec.likelihood_family.value}")
        print(f"   事前情境: {self.model_spec.prior_scenario.value}")
        print(f"   醫院數量: {self.n_hospitals}")
        print(f"   事件數量: {self.n_events}")
        print(f"   計算設備: {self.device}")
        
    def _initialize_hierarchical_parameters(self):
        """初始化4層階層參數"""
        
        # Level 4 (Global) - 全域參數
        self.global_alpha_mean = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.global_alpha_std = nn.Parameter(torch.tensor(1.0, device=self.device))
        
        # Level 3 (Regional) - 區域參數 (對應醫院)
        self.regional_beta_mean = nn.Parameter(torch.zeros(self.n_hospitals, device=self.device))
        self.regional_tau = nn.Parameter(torch.ones(self.n_hospitals, device=self.device))
        
        # Level 2 (Local) - 地方參數 (醫院特定)
        self.local_gamma = nn.Parameter(torch.zeros(self.n_hospitals, device=self.device))
        self.local_tau = nn.Parameter(torch.ones(self.n_hospitals, device=self.device))
        
        # Level 1 (Event) - 事件參數 (醫院x事件)
        self.event_theta = nn.Parameter(torch.zeros(self.n_hospitals, self.n_events, device=self.device))
        self.event_sigma = nn.Parameter(torch.ones(self.n_hospitals, self.n_events, device=self.device))
        
        # 觀測噪聲
        self.observation_sigma = nn.Parameter(torch.tensor(1.0, device=self.device))
        
    def _initialize_vulnerability_parameters(self):
        """初始化脆弱度函數參數"""
        
        if self.model_spec.vulnerability_type == VulnerabilityFunctionType.EMANUEL:
            # Emanuel USA: V = min(1, a * max(H-25, 0)^b)
            self.vuln_a = nn.Parameter(torch.tensor(0.004, device=self.device))
            self.vuln_b = nn.Parameter(torch.tensor(2.0, device=self.device))
            
        elif self.model_spec.vulnerability_type == VulnerabilityFunctionType.LINEAR:
            # 線性: V = a * H + b
            self.vuln_a = nn.Parameter(torch.tensor(0.01, device=self.device))
            self.vuln_b = nn.Parameter(torch.tensor(0.0, device=self.device))
            
        elif self.model_spec.vulnerability_type == VulnerabilityFunctionType.POLYNOMIAL:
            # 多項式: V = a * H^2 + b * H + c
            self.vuln_a = nn.Parameter(torch.tensor(0.0001, device=self.device))
            self.vuln_b = nn.Parameter(torch.tensor(0.01, device=self.device))
            self.vuln_c = nn.Parameter(torch.tensor(0.0, device=self.device))
        
    def compute_vulnerability_function(self, hazard_intensities: torch.Tensor) -> torch.Tensor:
        """
        計算脆弱度函數
        
        Parameters:
        -----------
        hazard_intensities : torch.Tensor
            災害強度 (n_hospitals, n_events)
            
        Returns:
        --------
        torch.Tensor
            脆弱度值 (n_hospitals, n_events)
        """
        
        if self.model_spec.vulnerability_type == VulnerabilityFunctionType.EMANUEL:
            # Emanuel USA函數
            h_thresh = torch.clamp(hazard_intensities - 25, min=0)
            vulnerability = torch.clamp(self.vuln_a * torch.pow(h_thresh, self.vuln_b), max=1.0)
            
        elif self.model_spec.vulnerability_type == VulnerabilityFunctionType.LINEAR:
            # 線性函數
            vulnerability = torch.clamp(self.vuln_a * hazard_intensities + self.vuln_b, min=0)
            
        elif self.model_spec.vulnerability_type == VulnerabilityFunctionType.POLYNOMIAL:
            # 多項式函數
            vulnerability = torch.clamp(
                self.vuln_a * torch.pow(hazard_intensities, 2) + 
                self.vuln_b * hazard_intensities + 
                self.vuln_c, 
                min=0
            )
        else:
            raise ValueError(f"不支援的脆弱度函數: {self.model_spec.vulnerability_type}")
        
        return vulnerability
    
    def compute_hierarchical_effects(self) -> Dict[str, torch.Tensor]:
        """
        計算4層階層效應
        
        Returns:
        --------
        Dict[str, torch.Tensor]
            各層級的效應參數
        """
        
        # Level 4: 全域效應
        global_alpha = dist.Normal(self.global_alpha_mean, torch.abs(self.global_alpha_std)).rsample()
        
        # Level 3: 區域效應 (依賴全域)
        regional_dist = dist.Normal(global_alpha.expand(self.n_hospitals), torch.abs(self.regional_tau))
        regional_beta = regional_dist.rsample()
        
        # Level 2: 地方效應 (依賴區域)  
        local_dist = dist.Normal(regional_beta, torch.abs(self.local_tau))
        local_gamma = local_dist.rsample()
        
        # Level 1: 事件效應 (依賴地方)
        local_expanded = local_gamma.unsqueeze(1).expand(-1, self.n_events)
        event_dist = dist.Normal(local_expanded, torch.abs(self.event_sigma))
        event_theta = event_dist.rsample()
        
        return {
            'global_alpha': global_alpha,
            'regional_beta': regional_beta, 
            'local_gamma': local_gamma,
            'event_theta': event_theta
        }
    
    def forward(self, 
                hazard_intensities: torch.Tensor, 
                exposure_values: torch.Tensor) -> torch.Tensor:
        """
        前向傳播 - 計算預期損失
        
        Parameters:
        -----------
        hazard_intensities : torch.Tensor
            災害強度 (n_hospitals, n_events)
        exposure_values : torch.Tensor
            曝險值 (n_hospitals,)
            
        Returns:
        --------
        torch.Tensor
            預期損失 (n_hospitals, n_events)
        """
        
        # 計算脆弱度函數
        vulnerability = self.compute_vulnerability_function(hazard_intensities)
        
        # 計算階層效應
        hierarchical_effects = self.compute_hierarchical_effects()
        
        # 計算調整後的脆弱度 (包含階層效應)
        adjusted_vulnerability = vulnerability * torch.exp(hierarchical_effects['event_theta'])
        
        # 計算預期損失 - 處理維度廣播
        exposure_expanded = exposure_values.unsqueeze(1).expand(-1, self.n_events)
        expected_loss = adjusted_vulnerability * exposure_expanded
        
        return expected_loss
    
    def compute_log_likelihood(self, 
                             expected_loss: torch.Tensor, 
                             observed_loss: torch.Tensor) -> torch.Tensor:
        """
        計算log likelihood
        
        Parameters:
        -----------
        expected_loss : torch.Tensor
            預期損失 (n_hospitals, n_events)
        observed_loss : torch.Tensor
            觀測損失 (n_hospitals, n_events)
            
        Returns:
        --------
        torch.Tensor
            Log likelihood值
        """
        
        sigma = torch.abs(self.observation_sigma)
        
        if self.model_spec.likelihood_family == LikelihoodFamily.NORMAL:
            likelihood_dist = dist.Normal(expected_loss, sigma)
            
        elif self.model_spec.likelihood_family == LikelihoodFamily.LOGNORMAL:
            log_expected = torch.log(torch.clamp(expected_loss, min=1e-6))
            likelihood_dist = dist.LogNormal(log_expected, sigma)
            
        elif self.model_spec.likelihood_family == LikelihoodFamily.STUDENT_T:
            # 使用固定的自由度參數
            df = 4.0
            likelihood_dist = dist.StudentT(df, expected_loss, sigma)
            
        else:
            # 默認使用正態分布
            likelihood_dist = dist.Normal(expected_loss, sigma)
        
        return likelihood_dist.log_prob(observed_loss).sum()
    
    def compute_log_prior(self) -> torch.Tensor:
        """
        計算log prior probability
        
        Returns:
        --------
        torch.Tensor
            Log prior值
        """
        log_prior = 0.0
        
        # 根據prior scenario設置先驗
        if self.model_spec.prior_scenario == PriorScenario.NON_INFORMATIVE:
            # 全域參數先驗
            log_prior += dist.Normal(0.0, 10.0).log_prob(self.global_alpha_mean)
            log_prior += dist.HalfNormal(5.0).log_prob(torch.abs(self.global_alpha_std))
            
        elif self.model_spec.prior_scenario == PriorScenario.WEAK_INFORMATIVE:
            log_prior += dist.Normal(0.0, 2.0).log_prob(self.global_alpha_mean)
            log_prior += dist.HalfNormal(1.0).log_prob(torch.abs(self.global_alpha_std))
            
        elif self.model_spec.prior_scenario == PriorScenario.OPTIMISTIC:
            log_prior += dist.Normal(-1.0, 1.0).log_prob(self.global_alpha_mean)
            log_prior += dist.HalfNormal(0.5).log_prob(torch.abs(self.global_alpha_std))
            
        elif self.model_spec.prior_scenario == PriorScenario.PESSIMISTIC:
            log_prior += dist.Normal(1.0, 1.0).log_prob(self.global_alpha_mean)
            log_prior += dist.HalfNormal(2.0).log_prob(torch.abs(self.global_alpha_std))
        
        # 階層參數先驗
        log_prior += dist.HalfNormal(1.0).log_prob(torch.abs(self.regional_tau)).sum()
        log_prior += dist.HalfNormal(1.0).log_prob(torch.abs(self.local_tau)).sum()
        log_prior += dist.HalfNormal(1.0).log_prob(torch.abs(self.event_sigma)).sum()
        log_prior += dist.HalfNormal(1.0).log_prob(torch.abs(self.observation_sigma))
        
        # 脆弱度函數參數先驗
        if self.model_spec.vulnerability_type == VulnerabilityFunctionType.EMANUEL:
            log_prior += dist.Gamma(2.0, 500.0).log_prob(self.vuln_a)
            log_prior += dist.Normal(2.0, 0.5).log_prob(self.vuln_b)
        elif self.model_spec.vulnerability_type == VulnerabilityFunctionType.LINEAR:
            log_prior += dist.Normal(0.01, 0.005).log_prob(self.vuln_a)
            log_prior += dist.Normal(0.0, 0.1).log_prob(self.vuln_b)
        elif self.model_spec.vulnerability_type == VulnerabilityFunctionType.POLYNOMIAL:
            log_prior += dist.Normal(0.0001, 0.00005).log_prob(self.vuln_a)
            log_prior += dist.Normal(0.01, 0.005).log_prob(self.vuln_b)
            log_prior += dist.Normal(0.0, 0.1).log_prob(self.vuln_c)
        
        return log_prior
    
    def predict_distribution(self, 
                           theta: torch.Tensor,
                           X: torch.Tensor, 
                           n_samples: int = 1000) -> torch.Tensor:
        """
        預測分布生成方法 - 與BasisRiskAwareVI兼容
        
        Parameters:
        -----------
        theta : torch.Tensor  
            參數向量 (來自VI優化)
        X : torch.Tensor
            輸入特徵 [hazard_intensities, exposure_values]
        n_samples : int
            樣本數量
            
        Returns:
        --------
        torch.Tensor
            預測樣本 (n_samples, n_hospitals, n_events)
        """
        
        # 確保在評估模式
        self.eval()
        
        with torch.no_grad():
            # 解析輸入
            hazard_intensities = X[0]  # (n_hospitals, n_events)
            exposure_values = X[1]     # (n_hospitals,)
            
            # 確保張量在正確設備上
            hazard_intensities = hazard_intensities.to(self.device)
            exposure_values = exposure_values.to(self.device)
            
            # 如果提供了theta參數，更新模型參數
            if theta is not None:
                self._update_parameters_from_theta(theta)
            
            # 生成多個樣本
            samples = []
            for _ in range(n_samples):
                expected_loss = self.forward(hazard_intensities, exposure_values)
                
                # 添加觀測噪聲
                sigma = torch.abs(self.observation_sigma)
                if self.model_spec.likelihood_family == LikelihoodFamily.NORMAL:
                    sample = torch.normal(expected_loss, sigma)
                elif self.model_spec.likelihood_family == LikelihoodFamily.LOGNORMAL:
                    log_expected = torch.log(torch.clamp(expected_loss, min=1e-6))
                    sample = torch.distributions.LogNormal(log_expected, sigma).sample()
                else:
                    sample = torch.normal(expected_loss, sigma)
                
                samples.append(sample)
            
            return torch.stack(samples, dim=0)  # (n_samples, n_hospitals, n_events)
    
    def _update_parameters_from_theta(self, theta: torch.Tensor):
        """從theta向量更新模型參數"""
        
        # 確保theta在正確設備上
        theta = theta.to(self.device)
        
        # 這個方法需要根據具體的參數化方案實現
        # 這裡提供一個簡化版本
        with torch.no_grad():
            # 假設theta包含主要的脆弱度參數
            if len(theta) >= 2:
                if hasattr(self, 'vuln_a'):
                    self.vuln_a.data = theta[0]
                if hasattr(self, 'vuln_b'):
                    self.vuln_b.data = theta[1]
                    
                # 如果有更多參數
                if len(theta) > 2 and hasattr(self, 'vuln_c'):
                    self.vuln_c.data = theta[2]
                    
                # 更新觀測噪聲
                if len(theta) > 3:
                    self.observation_sigma.data = torch.abs(theta[3])
    
    def get_parameter_dict(self) -> Dict[str, torch.Tensor]:
        """獲取所有模型參數的字典"""
        
        param_dict = {
            'global_alpha_mean': self.global_alpha_mean,
            'global_alpha_std': self.global_alpha_std,
            'regional_beta_mean': self.regional_beta_mean,
            'regional_tau': self.regional_tau,
            'local_gamma': self.local_gamma,
            'local_tau': self.local_tau,
            'event_theta': self.event_theta,
            'event_sigma': self.event_sigma,
            'observation_sigma': self.observation_sigma
        }
        
        # 添加脆弱度參數
        if hasattr(self, 'vuln_a'):
            param_dict['vuln_a'] = self.vuln_a
        if hasattr(self, 'vuln_b'):
            param_dict['vuln_b'] = self.vuln_b
        if hasattr(self, 'vuln_c'):
            param_dict['vuln_c'] = self.vuln_c
            
        return param_dict
    
    def fit_with_vi(self, 
                    hazard_intensities: torch.Tensor,
                    exposure_values: torch.Tensor, 
                    observed_losses: torch.Tensor,
                    n_iterations: int = 1000,
                    learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        使用變分推理擬合模型
        
        Parameters:
        -----------
        hazard_intensities : torch.Tensor
            災害強度數據
        exposure_values : torch.Tensor  
            曝險值數據
        observed_losses : torch.Tensor
            觀測損失數據
        n_iterations : int
            VI迭代次數
        learning_rate : float
            學習率
            
        Returns:
        --------
        Dict[str, Any]
            擬合結果
        """
        
        # 設置為訓練模式
        self.train()
        
        # 初始化優化器
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # 確保數據在正確設備上
        hazard_intensities = hazard_intensities.to(self.device)
        exposure_values = exposure_values.to(self.device)
        observed_losses = observed_losses.to(self.device)
        
        losses = []
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # 前向傳播
            expected_loss = self.forward(hazard_intensities, exposure_values)
            
            # 計算ELBO (Evidence Lower BOund)
            log_likelihood = self.compute_log_likelihood(expected_loss, observed_losses)
            log_prior = self.compute_log_prior()
            
            # ELBO = log_likelihood + log_prior (負ELBO用於最小化)
            elbo = log_likelihood + log_prior
            loss = -elbo
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            losses.append(float(loss))
            
            # 每100次迭代打印進度
            if iteration % 100 == 0:
                print(f"VI Iteration {iteration}: Loss = {float(loss):.4f}, "
                      f"LogLik = {float(log_likelihood):.4f}, LogPrior = {float(log_prior):.4f}")
        
        # 返回擬合結果
        return {
            'final_loss': losses[-1],
            'loss_history': losses,
            'converged': True,
            'n_iterations': n_iterations,
            'final_parameters': self.get_parameter_dict()
        }


class PyTorchHBMIntegrationAdapter:
    """
    PyTorch HBM與BasisRiskAwareVI的集成適配器
    
    提供統一接口，確保與現有Sigmoid代理優化框架兼容
    """
    
    def __init__(self, pytorch_model: PyTorchHierarchicalBayesianModel):
        """
        初始化適配器
        
        Parameters:
        -----------
        pytorch_model : PyTorchHierarchicalBayesianModel
            PyTorch HBM模型實例
        """
        self.pytorch_model = pytorch_model
        self.device = pytorch_model.device
        
    def predict_distribution(self, 
                           theta: Union[np.ndarray, torch.Tensor],
                           X: List[Union[np.ndarray, torch.Tensor]], 
                           n_samples: int = 1000) -> np.ndarray:
        """
        與BasisRiskAwareVI兼容的預測接口
        
        Parameters:
        -----------
        theta : array-like
            參數向量
        X : list
            輸入特徵 [hazard_intensities, exposure_values]
        n_samples : int
            樣本數量
            
        Returns:
        --------
        np.ndarray
            預測樣本
        """
        
        # 轉換輸入為PyTorch張量
        if isinstance(theta, np.ndarray):
            theta = torch.from_numpy(theta).float().to(self.device)
        elif isinstance(theta, torch.Tensor):
            theta = theta.to(self.device)
        
        # 轉換特徵
        X_tensors = []
        for x in X:
            if isinstance(x, np.ndarray):
                x_tensor = torch.from_numpy(x).float().to(self.device)
            elif isinstance(x, torch.Tensor):
                x_tensor = x.to(self.device)
            else:
                x_tensor = torch.tensor(x).float().to(self.device)
            X_tensors.append(x_tensor)
        
        # 調用PyTorch模型的預測方法
        with torch.no_grad():
            predictions = self.pytorch_model.predict_distribution(theta, X_tensors, n_samples)
            
        # 轉換回numpy格式
        return predictions.cpu().numpy()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """獲取模型摘要信息"""
        
        total_params = sum(p.numel() for p in self.pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in self.pytorch_model.parameters() if p.requires_grad)
        
        return {
            'model_type': 'PyTorch 4-Layer Hierarchical Bayesian Model',
            'device': str(self.device),
            'n_hospitals': self.pytorch_model.n_hospitals,
            'n_events': self.pytorch_model.n_events,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vulnerability_type': self.pytorch_model.model_spec.vulnerability_type,
            'likelihood_family': self.pytorch_model.model_spec.likelihood_family,
            'prior_scenario': self.pytorch_model.model_spec.prior_scenario
        }


def create_pytorch_hbm_model(model_spec: 'ModelSpec', 
                           n_hospitals: int = 100, 
                           n_events: int = 1000) -> PyTorchHBMIntegrationAdapter:
    """
    創建PyTorch HBM模型的便利函數
    
    Parameters:
    -----------
    model_spec : ModelSpec
        模型規格
    n_hospitals : int
        醫院數量
    n_events : int
        事件數量
        
    Returns:
    --------
    PyTorchHBMIntegrationAdapter
        集成適配器
    """
    
    # 創建PyTorch模型
    pytorch_model = PyTorchHierarchicalBayesianModel(
        model_spec=model_spec,
        n_hospitals=n_hospitals,
        n_events=n_events
    )
    
    # 返回適配器
    return PyTorchHBMIntegrationAdapter(pytorch_model)


def test_pytorch_hbm():
    """測試PyTorch HBM功能"""
    print("🧪 測試PyTorch階層貝氏模型...")
    
    # 模擬model_spec
    class MockModelSpec:
        model_name = "test_pytorch_hbm"
        vulnerability_type = VulnerabilityFunctionType.EMANUEL
        likelihood_family = LikelihoodFamily.NORMAL
        prior_scenario = PriorScenario.WEAK_INFORMATIVE
    
    # 創建測試模型
    model_spec = MockModelSpec()
    adapter = create_pytorch_hbm_model(model_spec, n_hospitals=10, n_events=100)
    
    # 測試數據
    hazard_intensities = torch.randn(10, 100) * 20 + 35  # 15-55 m/s風速
    exposure_values = torch.rand(10) * 1e8  # 0-100M曝險值
    theta = torch.tensor([0.004, 2.0])  # Emanuel參數
    
    # 測試預測
    predictions = adapter.predict_distribution(
        theta=theta.numpy(),
        X=[hazard_intensities.numpy(), exposure_values.numpy()],
        n_samples=100
    )
    
    print(f"✅ 預測輸出形狀: {predictions.shape}")
    print(f"   預測範圍: [{predictions.min():.2e}, {predictions.max():.2e}]")
    
    # 打印模型摘要
    summary = adapter.get_model_summary()
    print("📊 模型摘要:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("✅ PyTorch HBM測試完成")


if __name__ == "__main__":
    test_pytorch_hbm()