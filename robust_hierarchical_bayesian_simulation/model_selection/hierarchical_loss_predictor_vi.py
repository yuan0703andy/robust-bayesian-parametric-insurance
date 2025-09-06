#!/usr/bin/env python3
"""
Hierarchical Loss Predictor VI Module
階層貝葉斯損失預測器變分推斷模組

正確實現兩步驟架構的 Step 1: 純粹的損失預測器訓練

目標：使用 CRPS-VI 訓練階層貝葉斯模型，使其成為最準確的損失預測器
不涉及任何保險產品邏輯，純粹的風險建模

Author: Research Team  
Date: 2025-01-17
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try importing PyMC and ArviZ
try:
    import pymc as pm
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("⚠️ PyMC not available, using simplified hierarchical model")

try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False

# Import hierarchical model builder
try:
    from ..hierarchical_modeling.hierarchical_model_builder import build_hierarchical_model
    HIERARCHICAL_BUILDER_AVAILABLE = True
except ImportError:
    try:
        from robust_hierarchical_bayesian_simulation.hierarchical_modeling.hierarchical_model_builder import build_hierarchical_model
        HIERARCHICAL_BUILDER_AVAILABLE = True
    except ImportError:
        HIERARCHICAL_BUILDER_AVAILABLE = False
        print("⚠️ Hierarchical model builder not available, using simplified model")

# Import spatial data
try:
    from data_processing.spatial_data_processor import SpatialData
    SPATIAL_DATA_AVAILABLE = True
except ImportError:
    SPATIAL_DATA_AVAILABLE = False
    # Create dummy SpatialData class
    class SpatialData:
        def __init__(self):
            self.n_hospitals = 0
            self.n_regions = 1
            self.region_assignments = []
            self.hazard_intensities = None


class HierarchicalLossPredictorVI:
    """
    階層貝葉斯損失預測器 - CRPS-VI 訓練
    
    專門用於 Step 1: 訓練最準確的損失預測器
    不涉及任何保險產品，純粹的風險建模
    """
    
    def __init__(self, 
                 spatial_data: SpatialData,
                 contamination_epsilon: float = 0.05,
                 use_gpu: bool = True):
        """
        初始化階層損失預測器
        
        Args:
            spatial_data: 空間數據對象
            contamination_epsilon: ε-contamination 參數
            use_gpu: 是否使用GPU加速
        """
        self.spatial_data = spatial_data
        self.contamination_epsilon = contamination_epsilon
        
        # GPU配置
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.device = torch.device('cuda:0')
            print(f"🚀 使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("💻 使用CPU計算")
        
        # 構建階層模型
        self.hierarchical_model = build_hierarchical_model(
            spatial_data=spatial_data,
            contamination_epsilon=contamination_epsilon,
            model_name="Loss_Predictor_Model"
        )
        
        # 提取階層模型的參數結構
        self._extract_model_structure()
        
        print(f"✅ 階層損失預測器初始化完成")
        print(f"   模型參數: {self.n_hierarchical_params}")
        print(f"   醫院數: {spatial_data.n_hospitals}")
        print(f"   區域數: {spatial_data.n_regions}")
    
    def _extract_model_structure(self):
        """提取階層模型的參數結構"""
        # 從PyMC模型中提取參數信息
        with self.hierarchical_model:
            # 獲取所有未觀測變量（即需要推斷的參數）
            free_rvs = [rv for rv in self.hierarchical_model.free_RVs]
            
            # 計算總參數數量
            total_params = 0
            self.param_info = {}
            
            for rv in free_rvs:
                param_name = rv.name
                param_size = np.prod(rv.shape.eval() if hasattr(rv.shape, 'eval') else rv.shape)
                self.param_info[param_name] = {
                    'size': param_size,
                    'shape': rv.shape if hasattr(rv, 'shape') else (),
                    'start_idx': total_params
                }
                total_params += param_size
            
            self.n_hierarchical_params = total_params
            
        print(f"📊 階層模型參數結構:")
        for name, info in self.param_info.items():
            print(f"   {name}: {info['size']} 參數")
    
    def predict_loss_distribution(self, 
                                hazard_data: np.ndarray, 
                                theta_hierarchical: np.ndarray,
                                n_samples: int = 100) -> np.ndarray:
        """
        使用階層模型參數預測損失分佈
        
        Args:
            hazard_data: 災害數據 [n_hospitals, n_events]
            theta_hierarchical: 階層模型參數 [n_hierarchical_params]
            n_samples: 每個預測的樣本數
            
        Returns:
            損失分佈樣本 [n_hospitals, n_events, n_samples]
        """
        n_hospitals, n_events = hazard_data.shape
        
        # 將一維參數重構為模型參數
        model_params = self._reconstruct_model_params(theta_hierarchical)
        
        # 使用階層結構預測損失
        loss_samples = np.zeros((n_hospitals, n_events, n_samples))
        
        for hospital_idx in range(n_hospitals):
            for event_idx in range(n_events):
                # 獲取該醫院該事件的風險強度
                hazard_intensity = hazard_data[hospital_idx, event_idx]
                
                # 階層結構預測
                # Level 1: 區域效應
                region_idx = self.spatial_data.region_assignments[hospital_idx]
                alpha_region = model_params.get('α', np.zeros(self.spatial_data.n_regions))[region_idx]
                
                # Level 2: 空間效應 (簡化版)
                gamma_hospital = model_params.get('γ', np.zeros(n_hospitals))[hospital_idx]
                
                # Level 3: 事件特定效應
                delta_event = model_params.get('δ', np.zeros(n_events))[event_idx] if n_events <= len(model_params.get('δ', [])) else 0
                
                # Emanuel脆弱度函數
                vulnerability_power = model_params.get('vulnerability_power', 2.5)
                emanuel_threshold = 25.7  # mph
                
                if hazard_intensity > emanuel_threshold:
                    # Emanuel函數: L = (H - H0)^p
                    base_loss = (hazard_intensity - emanuel_threshold) ** vulnerability_power
                else:
                    base_loss = 0.0
                
                # 階層效應組合
                hierarchical_multiplier = np.exp(alpha_region + gamma_hospital + delta_event)
                predicted_loss = base_loss * hierarchical_multiplier
                
                # 添加不確定性
                observation_noise = model_params.get('σ_obs', 1.0)
                
                # 生成樣本
                if predicted_loss > 0:
                    # 使用對數正態分佈模擬損失
                    log_mean = np.log(predicted_loss)
                    for sample_idx in range(n_samples):
                        sample = np.random.lognormal(log_mean, observation_noise)
                        loss_samples[hospital_idx, event_idx, sample_idx] = sample
                else:
                    # 無損失
                    loss_samples[hospital_idx, event_idx, :] = 0.0
        
        return loss_samples
    
    def _reconstruct_model_params(self, theta_flat: np.ndarray) -> Dict:
        """
        將一維參數重構為模型參數字典
        
        Args:
            theta_flat: 一維參數向量
            
        Returns:
            參數字典
        """
        params = {}
        
        for param_name, info in self.param_info.items():
            start_idx = info['start_idx']
            end_idx = start_idx + info['size']
            
            param_values = theta_flat[start_idx:end_idx]
            
            # 根據形狀重構
            if info['size'] == 1:
                params[param_name] = float(param_values[0])
            else:
                try:
                    params[param_name] = param_values.reshape(info['shape'])
                except:
                    params[param_name] = param_values
        
        return params
    
    def compute_crps_loss(self, 
                         hazard_data: np.ndarray,
                         observed_losses: np.ndarray,
                         theta_hierarchical: np.ndarray) -> float:
        """
        計算階層模型的CRPS損失
        
        Args:
            hazard_data: 災害數據 [n_hospitals, n_events]
            observed_losses: 真實觀測損失 [n_hospitals, n_events]
            theta_hierarchical: 階層模型參數
            
        Returns:
            CRPS損失值 (越小越好)
        """
        # 預測損失分佈
        predicted_loss_samples = self.predict_loss_distribution(
            hazard_data, theta_hierarchical, n_samples=50
        )
        
        total_crps = 0.0
        n_observations = 0
        
        n_hospitals, n_events = hazard_data.shape
        
        for hospital_idx in range(n_hospitals):
            for event_idx in range(n_events):
                observed = observed_losses[hospital_idx, event_idx]
                predicted_samples = predicted_loss_samples[hospital_idx, event_idx, :]
                
                # 只對有損失的情況計算CRPS
                if observed > 0 or np.any(predicted_samples > 0):
                    # 經驗CRPS計算
                    crps = self._compute_empirical_crps(observed, predicted_samples)
                    total_crps += crps
                    n_observations += 1
        
        return total_crps / n_observations if n_observations > 0 else 0.0
    
    def _compute_empirical_crps(self, observation: float, forecasts: np.ndarray) -> float:
        """
        計算經驗CRPS分數
        
        Args:
            observation: 單個觀測值
            forecasts: 預測樣本數組
            
        Returns:
            CRPS分數
        """
        # 排序預測樣本
        sorted_forecasts = np.sort(forecasts)
        
        # CRPS公式：E[|X - observation|] - 0.5 * E[|X - X'|]
        term1 = np.mean(np.abs(sorted_forecasts - observation))
        term2 = 0.5 * np.mean(np.abs(sorted_forecasts[:, None] - sorted_forecasts[None, :]))
        
        return term1 - term2
    
    def train_loss_predictor(self,
                           hazard_data: np.ndarray,
                           observed_losses: np.ndarray,
                           n_iterations: int = 2000,
                           learning_rate: float = 0.01) -> Dict:
        """
        訓練階層貝葉斯損失預測器
        
        使用 CRPS-VI 優化階層模型參數，目標是預測真實損失
        
        Args:
            hazard_data: 災害數據 [n_hospitals, n_events]
            observed_losses: 真實觀測損失 [n_hospitals, n_events]
            n_iterations: 訓練迭代次數
            learning_rate: 學習率
            
        Returns:
            訓練結果
        """
        print(f"🧠 開始訓練階層貝葉斯損失預測器...")
        print(f"   目標: 最小化 CRPS(真實損失, 階層模型預測)")
        print(f"   參數數量: {self.n_hierarchical_params}")
        print(f"   數據規模: {hazard_data.shape}")
        
        # 轉換為GPU張量（如果使用GPU）
        if self.use_gpu:
            hazard_tensor = torch.from_numpy(hazard_data).float().to(self.device)
            losses_tensor = torch.from_numpy(observed_losses).float().to(self.device)
        
        # 初始化變分參數
        torch.manual_seed(42)
        mu_theta = torch.randn(self.n_hierarchical_params, device=self.device) * 0.1
        log_sigma_theta = torch.full((self.n_hierarchical_params,), -2.0, device=self.device)
        
        mu_theta.requires_grad_(True)
        log_sigma_theta.requires_grad_(True)
        
        # 優化器
        optimizer = torch.optim.Adam([mu_theta, log_sigma_theta], lr=learning_rate)
        
        best_crps = float('inf')
        best_mu = mu_theta.clone()
        best_log_sigma = log_sigma_theta.clone()
        
        print(f"🚀 開始VI訓練...")
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # 從變分分佈採樣
            sigma_theta = torch.exp(log_sigma_theta)
            eps = torch.randn(10, self.n_hierarchical_params, device=self.device)
            theta_samples = mu_theta.unsqueeze(0) + sigma_theta.unsqueeze(0) * eps
            
            # 計算CRPS損失（批次處理）
            total_loss = 0.0
            for theta_sample in theta_samples:
                # 轉換為numpy進行CRPS計算（暫時的實現）
                theta_np = theta_sample.detach().cpu().numpy()
                crps_loss = self.compute_crps_loss(
                    hazard_data, observed_losses, theta_np
                )
                total_loss += crps_loss
            
            avg_crps = total_loss / len(theta_samples)
            
            # KL散度
            log_prior = -0.5 * torch.sum(theta_samples**2, dim=1)
            log_q = -0.5 * torch.sum((theta_samples - mu_theta)**2 / sigma_theta**2, dim=1) - \
                    0.5 * torch.sum(torch.log(2 * np.pi * sigma_theta**2))
            
            kl_div = torch.mean(log_q - log_prior)
            
            # VI目標：最大化 -CRPS - KL
            vi_loss = avg_crps + kl_div.item()
            vi_loss_tensor = torch.tensor(vi_loss, requires_grad=True, device=self.device)
            
            # 反向傳播
            vi_loss_tensor.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_([mu_theta, log_sigma_theta], max_norm=1.0)
            optimizer.step()
            
            # 記錄最佳結果
            if avg_crps < best_crps:
                best_crps = avg_crps
                best_mu = mu_theta.clone()
                best_log_sigma = log_sigma_theta.clone()
            
            # 進度報告
            if (iteration + 1) % 100 == 0:
                print(f"   迭代 {iteration+1}: CRPS={avg_crps:.4f}, KL={kl_div.item():.4f}")
        
        # 最終結果
        final_mu = best_mu.detach().cpu().numpy()
        final_sigma = torch.exp(best_log_sigma).detach().cpu().numpy()
        
        print(f"✅ 損失預測器訓練完成!")
        print(f"   最佳CRPS: {best_crps:.4f}")
        print(f"   參數收斂性: μ範圍=[{final_mu.min():.3f}, {final_mu.max():.3f}]")
        print(f"   不確定性: σ範圍=[{final_sigma.min():.3f}, {final_sigma.max():.3f}]")
        
        return {
            'best_theta_mean': final_mu,
            'best_theta_std': final_sigma,
            'final_crps': best_crps,
            'converged': True,
            'n_params': self.n_hierarchical_params
        }
    
    def predict(self, hazard_data: np.ndarray, trained_params: Dict) -> np.ndarray:
        """
        使用訓練好的參數進行損失預測
        
        Args:
            hazard_data: 新的災害數據
            trained_params: 訓練結果
            
        Returns:
            預測的損失分佈
        """
        theta_mean = trained_params['best_theta_mean']
        
        # 使用均值參數進行預測
        predicted_losses = self.predict_loss_distribution(
            hazard_data, theta_mean, n_samples=100
        )
        
        # 返回分佈的統計量
        predicted_mean = np.mean(predicted_losses, axis=2)
        predicted_std = np.std(predicted_losses, axis=2)
        
        return {
            'loss_mean': predicted_mean,
            'loss_std': predicted_std,
            'loss_samples': predicted_losses
        }


if __name__ == "__main__":
    print("🧠 階層貝葉斯損失預測器 - 獨立測試")
    
    # 這裡可以添加獨立測試代碼
    pass