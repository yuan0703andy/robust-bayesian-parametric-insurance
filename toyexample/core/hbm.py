"""Hierarchical Bayesian Model moved from toy_example_complete.py."""
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, LogNormal
from torch.nn import functional as F
from typing import Tuple, List

# %%
# ============================================================================
# 3. 四層階層貝氏模型（PyTorch 實現）
# ============================================================================

# 檢查PyTorch可用性
print("🧠 開始定義四層階層貝氏模型...")

## 移除早期占位的 PriorLikelihoodProcessor，保留後續完整實現
class DifferentiableHierarchicalBayesianModel(nn.Module):
        """可微分的4層階層貝氏模型 - 「風險大腦」"""
        
        def __init__(self, n_hospitals: int, n_regions: int, n_events: int,
                     distance_matrix: np.ndarray, verbose: bool = False):
            super().__init__()
            
            self.n_hospitals = n_hospitals
            self.n_regions = n_regions  
            self.n_events = n_events
            self.verbose = verbose
            
            # 註冊距離矩陣為不可訓練參數
            self.register_buffer('distance_matrix', 
                               torch.tensor(distance_matrix, dtype=torch.float32))
        
        def forward(self, hazard_intensities: torch.Tensor, 
                    exposure_values: torch.Tensor,
                    theta_samples: torch.Tensor,
                    region_assignments: torch.Tensor = None) -> Dict[str, torch.Tensor]:
            """
            四層階層模型前向傳播 - 改進版本支持真實區域分配
            
            Args:
                hazard_intensities: (n_hospitals, n_events)
                exposure_values: (n_hospitals,) 
                theta_samples: (n_samples, n_params) - VI採樣的參數
                region_assignments: (n_hospitals,) - 每家醫院的區域分配 (可選)
                
            Returns:
                損失預測分佈 G(θ) 的參數
            """
            batch_size = theta_samples.shape[0]
            
            # 解析HBM參數 (假設特定的參數順序)
            params = self._parse_theta_samples(theta_samples)
            
            # Level 4: 超參數層 - 已在theta_samples中
            
            # 確保區域分配存在且類型/設備一致
            if region_assignments is None:
                # 預設採用循環分配，避免全部歸到同一區域造成偏置
                region_assignments = (torch.arange(self.n_hospitals, device=hazard_intensities.device) % self.n_regions).long()
                if self.verbose:
                    print(f"⚠️ 使用預設區域分配 (循環分配到 {self.n_regions} 個區域)")
            else:
                region_assignments = region_assignments.to(hazard_intensities.device).long()
            # Level 3: 參數層
            region_effects, individual_effects, spatial_effects = self._compute_level3_effects(
                params, batch_size
            )
            
            # Level 2: 過程層 - 位置特定脆弱度參數
            vulnerability_params = self._compute_vulnerability_parameters(
                region_effects, individual_effects, spatial_effects, params, region_assignments
            )
            
            # Level 1: 觀測層 - 損失預測
            loss_distribution_params = self._compute_loss_predictions(
                hazard_intensities, exposure_values, vulnerability_params, params, region_assignments
            )
            
            return loss_distribution_params
        
        def _parse_theta_samples(self, theta_samples: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            解析VI採樣的參數 - 改進版本支持更靈活的a, b, v₀學習
            
            擴展參數空間:
            - theta[0-3]: 層次效應參數 (log σ_α, log σ_β, log σ_δ, log ρ_spatial)
            - theta[4]: log(vulnerability_a) - Emanuel函數的風險係數  
            - theta[5]: log(vulnerability_b) - Emanuel函數的指數
            - theta[6]: log(sigma_obs_base) - 基礎觀測誤差 (全域)
            - theta[7]: log(v_threshold) - 可學習的閾值風速v₀
            - theta[8]: log(sigma_obs_scale) - 觀測誤差異質性縮放參數
            - theta[9+]: 異質觀測誤差 (可選，每家醫院)
            """
            # 基礎參數 (使用softplus確保正值)
            parsed_params = {
                # 這三個是標準差，維持 softplus 沒問題
                'sigma_alpha': F.softplus(theta_samples[:, 0]),
                'sigma_gamma': F.softplus(theta_samples[:, 1]),
                'sigma_delta': F.softplus(theta_samples[:, 2]),
                # 範圍參數保持正值
                'rho_spatial': torch.clamp(F.softplus(theta_samples[:, 3]), min=1e-3),

                # 與先驗一致：這些參數在先驗是「log 空間」，所以用 exp 還原
                'vulnerability_a': torch.clamp(torch.exp(theta_samples[:, 4]), max=1.0),
                'vulnerability_b': torch.clamp(torch.exp(theta_samples[:, 5]), max=5.0),

                # 觀測誤差的基準尺度用 exp（亦可用 softplus，但要與先驗一致）
                'sigma_obs_base': torch.exp(theta_samples[:, 6]),
            }
            if theta_samples.shape[1] > 7:
                parsed_params['v_threshold'] = torch.exp(theta_samples[:, 7])
            if theta_samples.shape[1] > 8:
                parsed_params['sigma_obs_scale'] = torch.exp(theta_samples[:, 8])
                
            # 構建異質觀測誤差
            parsed_params['sigma_obs'] = self._compute_heteroscedastic_sigma_obs(
                parsed_params.get('sigma_obs_base', torch.tensor(1e6)),
                parsed_params.get('sigma_obs_scale', torch.tensor(1.0)),
                theta_samples
            )
            
            return parsed_params
        
        def _compute_heteroscedastic_sigma_obs(self, sigma_obs_base: torch.Tensor,
                                             sigma_obs_scale: torch.Tensor,
                                             theta_samples: torch.Tensor) -> torch.Tensor:
            """
            計算異質觀測誤差 - 每家醫院可能有不同的觀測誤差
            
            採用分層結構:
            σ_obs_i = σ_obs_base * exp(σ_obs_scale * η_i)
            其中 η_i ~ N(0,1) 是醫院特定的隨機效應
            
            這比完全獨立的醫院誤差更高效，同時保持異質性
            """
            batch_size = theta_samples.shape[0]
            
            # 如果參數足夠支持異質誤差 (9 + n_hospitals個參數)
            min_params_for_heteroscedastic = 9 + self.n_hospitals
            
            if theta_samples.shape[1] >= min_params_for_heteroscedastic:
                # 使用完全異質觀測誤差 (每家醫院獨立)
                hospital_noise_params = theta_samples[:, 9:9+self.n_hospitals]
                hospital_multipliers = torch.exp(sigma_obs_scale.unsqueeze(1) * 
                                               torch.tanh(hospital_noise_params))  # 使用tanh限制範圍
                
                # 每家醫院的誤差: σ_obs_i = σ_obs_base * multiplier_i
                sigma_obs_heteroscedastic = sigma_obs_base.unsqueeze(1) * hospital_multipliers
                
                if self.verbose:
                    print(f"✅ 使用完全異質觀測誤差 - {self.n_hospitals}家醫院獨立")
                return sigma_obs_heteroscedastic  # (batch_size, n_hospitals)
                
            elif theta_samples.shape[1] >= 9:
                # 使用基於區域的異質誤差（每個樣本一組隨機效應）
                hospital_random_effects = torch.randn(batch_size, self.n_hospitals, device=theta_samples.device)
                hospital_multipliers = torch.exp(sigma_obs_scale.unsqueeze(1) * hospital_random_effects)
                sigma_obs_heteroscedastic = sigma_obs_base.unsqueeze(1) * hospital_multipliers
                if self.verbose:
                    print(f"✅ 使用基於區域的異質觀測誤差 - {self.n_hospitals}家醫院")
                return sigma_obs_heteroscedastic  # (batch_size, n_hospitals)
            
            else:
                # 回退到同質觀測誤差
                if self.verbose:
                    print("⚠️ 參數不足，使用同質觀測誤差")
                return sigma_obs_base.unsqueeze(1).expand(batch_size, self.n_hospitals)
        
        def _compute_level3_effects(self, params: Dict[str, torch.Tensor], 
                                  batch_size: int) -> Tuple[torch.Tensor, ...]:
            """Level 3: 計算參數層效應"""
            
            # 區域平均效應 α_r ~ N(0, σ_α²)  
            device = params['sigma_alpha'].device
            region_effects = torch.randn(batch_size, self.n_regions, device=device) * params['sigma_alpha'].unsqueeze(1)
            
            # 非結構化個體隨機效應 γ_i ~ N(0, σ_γ²)
            individual_effects = torch.randn(batch_size, self.n_hospitals, device=device) * params['sigma_gamma'].unsqueeze(1)
            
            # 空間結構化隨機效應 δ_i ~ MVN(0, Σ_spatial)
            spatial_effects = self._sample_spatial_effects(params, batch_size)
            
            return region_effects, individual_effects, spatial_effects
    
        def _sample_spatial_effects(self, params: Dict[str, torch.Tensor],
                                   batch_size: int) -> torch.Tensor:
            """
            無協方差分解版的空間效應採樣：
            δ = σ_δ * W_norm @ ε，其中 W_ij = exp(-d_ij / ρ)，再做
              1) 列和歸一 (row-normalize)、
              2) 每列 L2 規範化，令 Var(δ_i) ≈ σ_δ^2。
            避免 Cholesky / MVN，數值穩定且可微。
            返回: (batch_size, n_hospitals)
            """
            device = self.distance_matrix.device
            D = self.distance_matrix  # (H, H)
            H = self.n_hospitals

            # 參數保護
            sigma_delta = torch.clamp(params['sigma_delta'], min=1e-6)               # (B,)
            rho_spatial = torch.clamp(params['rho_spatial'], min=1e-3, max=1e4)      # (B,)

            effects = []
            for b in range(batch_size):
                rho = rho_spatial[b]
                # 距離衰減權重
                W = torch.exp(- D / rho).to(device)
                # 主對角保底（自身權重）- 非就地操作以保留梯度
                I = torch.eye(H, device=device, dtype=W.dtype)
                W = W * (1.0 - I) + I
                # 列和歸一化
                row_sum = W.sum(dim=1, keepdim=True).clamp_min(1e-8)
                Wn = W / row_sum
                # 每列 L2 規範化
                l2 = torch.sqrt((Wn ** 2).sum(dim=1, keepdim=True)).clamp_min(1e-8)
                Wn = Wn / l2
                # 抽 i.i.d 噪聲並平滑
                eps = torch.randn(H, device=device)
                delta = sigma_delta[b] * (Wn @ eps)
                # 清理數值
                delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                effects.append(delta)

            return torch.stack(effects)
        
        def _compute_matern_covariance(self, distance_matrix: torch.Tensor, 
                                     sigma_delta: torch.Tensor, 
                                     rho_spatial: torch.Tensor, 
                                     nu: float = 1.5) -> torch.Tensor:
            """
            計算Matern協方差矩陣（ν=1.5）- 數值穩定版。
            - 對 ρ 做下限夾取，避免除以 0
            - 對 (1+d)*exp(-d) 使用安全計算，避免 inf*0 → NaN
            - 清理 NaN/Inf，並加 nugget 提升正定性
            """
            # 保護 ρ 範圍
            rho = torch.clamp(
                rho_spatial,
                min=torch.tensor(1e-3, device=distance_matrix.device, dtype=distance_matrix.dtype),
                max=torch.tensor(1e4, device=distance_matrix.device, dtype=distance_matrix.dtype)
            )
            # 標準化距離: d_norm = √3 * d / ρ
            d_norm = (np.sqrt(3.0) * distance_matrix / rho)
            # 安全計算 (1 + d_norm) * exp(-d_norm)
            matern_kernel = torch.zeros_like(d_norm)
            small_mask = d_norm < 50  # 大於此值時趨近0
            if small_mask.any():
                d_small = d_norm[small_mask]
                matern_kernel[small_mask] = (1.0 + d_small) * torch.exp(-d_small)
            # 清理 NaN/Inf
            matern_kernel = torch.nan_to_num(matern_kernel, nan=0.0, posinf=0.0, neginf=0.0)
            cov_matrix = (sigma_delta**2) * matern_kernel
            # 加 nugget 確保正定
            nugget = torch.clamp(sigma_delta**2 * 1e-4, min=1e-8, max=1e-3)
            cov_matrix = cov_matrix + nugget * torch.eye(self.n_hospitals, device=cov_matrix.device)
            return cov_matrix
        
        def _sample_from_covariance(self, cov_matrix: torch.Tensor, 
                                   sigma_delta: torch.Tensor) -> torch.Tensor:
            """
            從協方差矩陣採樣，具有多種fallback機制；加入 NaN/Inf 清理與對角保護。
            """
            # 清理 NaN/Inf 並對角保護
            cov_matrix = torch.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            eps_pd = torch.clamp(sigma_delta**2 * 1e-4, min=1e-8, max=1e-3)
            cov_matrix = cov_matrix + eps_pd * torch.eye(self.n_hospitals, device=cov_matrix.device)
            try:
                mvn = torch.distributions.MultivariateNormal(
                    torch.zeros(self.n_hospitals, device=cov_matrix.device), 
                    covariance_matrix=cov_matrix
                )
                return mvn.sample()
            except RuntimeError:
                try:
                    # 第二步：增大 nugget
                    nugget_enhanced = torch.clamp(sigma_delta**2 * 1e-2, min=1e-6, max=1e-1)
                    cov_matrix_stable = cov_matrix + nugget_enhanced * torch.eye(
                        self.n_hospitals, device=cov_matrix.device
                    )
                    cov_matrix_stable = torch.nan_to_num(cov_matrix_stable, nan=0.0, posinf=0.0, neginf=0.0)
                    mvn = torch.distributions.MultivariateNormal(
                        torch.zeros(self.n_hospitals, device=cov_matrix.device), 
                        covariance_matrix=cov_matrix_stable
                    )
                    return mvn.sample()
                except RuntimeError:
                    # 第三步：使用對角近似（保護對角）
                    diag = torch.clamp(torch.diag(cov_matrix), min=1e-12)
                    diagonal_std = torch.sqrt(diag)
                    diagonal_std = torch.nan_to_num(diagonal_std, nan=0.0, posinf=0.0, neginf=0.0)
                    return torch.randn(self.n_hospitals, device=sigma_delta.device) * diagonal_std
    
        def _compute_vulnerability_parameters(self, region_effects: torch.Tensor,
                                            individual_effects: torch.Tensor,
                                            spatial_effects: torch.Tensor,
                                            params: Dict[str, torch.Tensor],
                                            region_assignments: torch.Tensor) -> torch.Tensor:
            """
            Level 2: 計算位置特定脆弱度參數 - 改進版本支持真實區域分配
            
            Args:
                region_assignments: (n_hospitals,) 每家醫院的區域分配
                                  如果為None，則使用K-means聚類生成的預設分配
            """
            batch_size = region_effects.shape[0]
            # 驗證 region_assignments
            region_assignments = region_assignments.to(region_effects.device).long()
            if region_assignments.numel() != self.n_hospitals:
                raise RuntimeError(
                    f"region_assignments 長度錯誤: 得到 {region_assignments.numel()}，預期 {self.n_hospitals}。"
                )
            if torch.min(region_assignments) < 0 or torch.max(region_assignments) >= self.n_regions:
                raise RuntimeError(
                    f"region_assignments 值域超界: 允許 [0, {self.n_regions-1}]，實際最小={int(torch.min(region_assignments))} 最大={int(torch.max(region_assignments))}。"
                )
            # 將區域效應一次性映射到醫院層: (batch, n_regions) -> (batch, n_hospitals)
            index = region_assignments.unsqueeze(0).expand(batch_size, -1)
            region_effects_mapped = torch.gather(region_effects, 1, index)
            # 確保 spatial 與 individual 已是 (batch, n_hospitals)
            if spatial_effects.shape != (batch_size, self.n_hospitals):
                raise RuntimeError(
                    f"spatial_effects 維度錯誤: 得到 {tuple(spatial_effects.shape)}，預期 ({batch_size}, {self.n_hospitals})。"
                )
            if individual_effects.shape != (batch_size, self.n_hospitals):
                raise RuntimeError(
                    f"individual_effects 維度錯誤: 得到 {tuple(individual_effects.shape)}，預期 ({batch_size}, {self.n_hospitals})。"
                )
            # β_i = α_{r(i)} + δ_i + γ_i
            vulnerability_params = region_effects_mapped + spatial_effects + individual_effects
            # 數值保護：限制線性項幅度，避免 exp(β) 溢位
            vulnerability_params = torch.clamp(vulnerability_params, min=-10.0, max=10.0)
            return vulnerability_params
    
        def _compute_loss_predictions(self, hazard_intensities: torch.Tensor,
                                        exposure_values: torch.Tensor,
                                        vulnerability_params: torch.Tensor,
                                        params: Dict[str, torch.Tensor],
                                        region_assignments: torch.Tensor = None) -> Dict[str, torch.Tensor]:
            """
            Level 1: 計算損失預測分佈參數
            
            使用Emanuel USA函數加上層次修正:
            V(v; β_i) = a * [(v-v₀)/v₀]^b * exp(β_i)
            其中β_i是位置特定的層次效應參數
            """
            batch_size = vulnerability_params.shape[0]
            
            # Emanuel USA 校準參數 - 改進版本支持可學習的v₀
            vulnerability_a = params['vulnerability_a']      # ~0.0039
            vulnerability_b = params['vulnerability_b']      # ~2.04
            
            # 使用可學習的閾值風速v₀ (如果可用)，否則使用Emanuel預設值
            if 'v_threshold' in params:
                v_threshold = params['v_threshold']  # 可學習的閾值風速 (m/s)
            else:
                v_threshold = torch.full_like(vulnerability_a, 25.7)  # Emanuel預設閾值風速
            
            # 對齊 hazard 至醫院層級
            if hazard_intensities.shape[0] == self.n_hospitals:
                hi_by_hospital = hazard_intensities
            elif hazard_intensities.shape[0] == self.n_regions:
                if region_assignments is None:
                    raise RuntimeError(
                        "hazard_intensities 為區域層 (n_regions x n_events)，但未提供 region_assignments。\n"
                        "請在訓練/評估時傳入 spatial_data 以提供 region_assignments，或改用醫院層 (n_hospitals x n_events) 風險矩陣。"
                    )
                hi_by_hospital = hazard_intensities.index_select(0, region_assignments.to(hazard_intensities.device))
            else:
                raise ValueError(
                    f"hazard_intensities 維度 {tuple(hazard_intensities.shape)} 不匹配 n_hospitals={self.n_hospitals} 或 n_regions={self.n_regions}"
                )

            # 確保 exposure_values 形狀為 (n_hospitals, 1) 以利廣播
            if exposure_values.dim() == 1:
                exposure_values = exposure_values.view(-1, 1)

            # 計算標準化風速超量 - 支持批次不同的v₀值
            normalized_excess_batch = []
            for b in range(batch_size):
                v_thresh_b = v_threshold[b] if v_threshold.dim() > 0 else v_threshold
                hazard_excess = torch.clamp(hi_by_hospital - v_thresh_b, min=0.0)
                normalized_excess = hazard_excess / v_thresh_b
                normalized_excess_batch.append(normalized_excess)
            
            # 批次計算損失期望 - 使用每個批次特定的normalized_excess
            mu_loss_batch = []
            vulnerability_batch = []
            for b in range(batch_size):
                # 使用該批次的標準化風速超量
                normalized_excess_b = normalized_excess_batch[b]
                
                # 數值安全版：在 log 域相加，並夾範圍避免溢出/NaN
                eps_v = 1e-30

                # log base: a + b*log(excess)，excess=0 時 -> log(eps) 使得值趨近 0 而不 NaN
                log_base = torch.log(torch.clamp(normalized_excess_b, min=eps_v)) * vulnerability_b[b] \
                        + torch.log(torch.clamp(vulnerability_a[b], min=eps_v))

                # 夾住層級效應，避免 exp 爆炸
                beta = torch.clamp(vulnerability_params[b], -12.0, 12.0).unsqueeze(1)

                # log v = log_base + beta；再夾到 ≤0，確保 vulnerability ≤ 1
                log_v = torch.clamp(log_base + beta, max=0.0, min=-60.0)
                vulnerability = torch.exp(log_v)
                
                # 期望損失 = V(v; β_i) × E_i
                expected_loss = vulnerability * exposure_values
                mu_loss_batch.append(expected_loss)
                vulnerability_batch.append(vulnerability)
            
            mu_loss = torch.stack(mu_loss_batch)  # (batch_size, n_hospitals, n_events)
            vulnerability_all = torch.stack(vulnerability_batch)  # (batch_size, n_hospitals, n_events)
            
            # 對數正態分佈參數化: Loss ~ LogNormal(μ_log, σ_log)
            # 確保數值穩定性
            mu_loss_clamped = torch.clamp(mu_loss, min=1e3)  # 最小損失 $1K
            
            # === 用 CV 公式把「美元尺度的不確定度」轉成 LogNormal 的 sigma_log ===
            sigma_obs = params['sigma_obs']  # 可能為 (batch, H) 或 (batch,)
            if sigma_obs.dim() == 2:
                std_loss = sigma_obs.unsqueeze(2).expand_as(mu_loss_clamped)     # (batch, H, E)
            else:
                std_loss = sigma_obs.unsqueeze(1).unsqueeze(2).expand_as(mu_loss_clamped)  # (batch, H, E)
            m = torch.clamp(mu_loss_clamped, min=1e3)
            cv2 = (std_loss / m) ** 2
            cv2 = torch.nan_to_num(cv2, nan=0.0, posinf=1e6, neginf=0.0)
            cv2 = torch.clamp(cv2, min=0.0, max=1e6)
            sigma_log = torch.sqrt(torch.clamp(torch.log1p(cv2), min=1e-12))
            sigma_log = torch.nan_to_num(sigma_log, nan=1e-3, posinf=2.5, neginf=1e-3)
            sigma_log = torch.clamp(sigma_log, 1e-3, 2.5)
            # 進一步保護 mu_log（避免極端時的 NaN/Inf）
            mu_log = torch.log(mu_loss_clamped) - 0.5 * (sigma_log ** 2)
            mu_log = torch.nan_to_num(mu_log, nan=7.0, posinf=20.0, neginf=-60.0)
            
            return {
                'mu_log': mu_log,         # 對數正態分佈的位置參數
                'sigma_log': sigma_log,   # 對數正態分佈的尺度參數
                'mu_loss': mu_loss,       # 原始損失期望 (用於分析)
                'vulnerability': vulnerability_all,  # 脆弱度值 (用於診斷，含batch)
                'sigma_obs': params.get('sigma_obs', torch.exp(sigma_log))  # 傳遞觀測誤差（供NORMAL/Student-t使用）
            }

print("✅ 四層階層貝氏模型定義完成")