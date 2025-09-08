from .model import UnifiedEndToEndVIModel
from utils.gpu_setup import device, USE_MULTI_GPU, GPU_DEVICES
from components.payout import _indemnity_from_loss
from typing import Dict, List, Any
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, LogNormal, StudentT
from enum import Enum
from typing import Optional
from components.config import PriorScenario, LikelihoodFamily
from .data import SimulatedSpatialData


class EndToEndTrainer:
    """端到端訓練器 - GPU-Accelerated Version"""
        
    def __init__(self, model: UnifiedEndToEndVIModel, learning_rate: float = 3e-5,
                    enable_multi_gpu: bool = False, verbose: bool = False,
                    log_every: int = 20):
        self.original_model = model
        # 暫時停用多卡，避免DataParallel沿H維切片導致維度不一致
        self.enable_multi_gpu = False
        self.verbose = verbose
        
        # Warmup 參數
        self.log_every = int(log_every)
        self.beta_kl_max = 1.0      # KL 最大權重
        self.beta_warmup_epochs = 100
        self.lambda_crps_max = 5.0  # CRPS 最大權重（調整到決策量級）
        self.lambda_warmup_epochs = 50
        
        # GPU配置和模型設置
        if torch.cuda.is_available():
            # 移動模型到主GPU
            self.model = model.to(f'cuda:{GPU_DEVICES[0]}')
            
            # 配置多GPU DataParallel
            if self.enable_multi_gpu and len(GPU_DEVICES) >= 2:
                if self.verbose:
                    print(f"🚀 配置DataParallel: 使用GPU {GPU_DEVICES}")
                self.model = nn.DataParallel(self.model, device_ids=GPU_DEVICES)
                self.device = f'cuda:{GPU_DEVICES[0]}'
            else:
                self.device = f'cuda:{GPU_DEVICES[0]}'
                if self.verbose:
                    print(f"🔧 單GPU模式: 使用GPU {GPU_DEVICES[0]}")
        else:
            self.model = model
            self.device = 'cpu'
            if self.verbose:
                print("💻 CPU模式: GPU不可用")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
        self.loss_history = []
        
        # GPU性能監控
        self.gpu_memory_usage = []
        self.training_times = []
        
    def _should_log(self, epoch_idx: int) -> bool:
        """判斷是否應該輸出日誌"""
        return self.verbose and (epoch_idx % self.log_every == 0 or epoch_idx <= 3)
    
    def _as_scalar(self, x):
        """確保輸入是標量tensor"""
        if isinstance(x, torch.Tensor) and x.ndim == 0:
            return x
        elif hasattr(x, 'mean'):
            return x.mean()
        else:
            return torch.as_tensor(x, device=self.device)
        
        
    def _anneal_tau(self, payout_fn):
        """簡單線性退火：1→0.1；你可依需要調整里程碑。"""
        # 以目前已完成的 epoch 數推估下一個 tau
        epoch = len(self.loss_history) + 1
        if epoch < 50:
            tau = 0.8
        elif epoch > 150:
            tau = 0.1
        else:
            tau = 0.8 - 0.7 * (epoch - 50) / 100.0
        if hasattr(payout_fn, "set_tau"):
            payout_fn.set_tau(float(tau))
    
    def _parametric_payout_hard(self, base_model, hazard_intensities: torch.Tensor) -> torch.Tensor:
        """
        事件層參數型賠付（硬條款）：用 cat-in-circle 的「硬 max + 硬 ramp」
        返回 shape: [E]
        """
        pt = base_model.param_target
        if pt is None:
            # 沒有參數型條款就回傳 0（或你可改成其它後備）
            return hazard_intensities.sum(dim=0).new_zeros(hazard_intensities.shape[1])

        # winds_ms: [H, E]
        winds_ms = hazard_intensities
        H, E = winds_ms.shape
        # 遮罩：只保留圓內站點，其餘置極小
        masked = winds_ms.unsqueeze(0).expand(H, -1, -1).clone()     # [H, H, E]
        mask3 = pt.mask.unsqueeze(-1).bool()                         # [H, H, 1]
        big_neg = torch.tensor(-1e9, device=winds_ms.device, dtype=winds_ms.dtype)
        masked = masked.masked_fill(~mask3, big_neg)

        # 硬 max：每個 site 對圓內的站點取最大風速
        I_site = masked.max(dim=1).values                            # [H, E]

        # 硬 ramp：觸發→耗盡 線性，夾到 [0,1]
        width = max(pt.exhaustion - pt.trigger, 1e-6)
        ramp = torch.clamp((I_site - pt.trigger) / width, 0.0, 1.0)  # [H, E]

        # site 賠付與事件總賠付
        payout_site = ramp * pt.site_limits.unsqueeze(-1)            # [H, E]
        total = (pt.site_weights.unsqueeze(-1) * payout_site).sum(dim=0)  # [E]
        if pt.payout_cap is not None:
            total = torch.clamp(total, max=pt.payout_cap)
        return total


    def _indemnity_hard(self, base_model, observed_losses: torch.Tensor) -> torch.Tensor:
        """
        事件層硬條款理賠：把觀測損失丟入「分層線性」硬 ramp，沿醫院加總 → [E]
        金額單位：與 product_config 保持一致（美元）
        """
        pf = base_model.payout_function
        thr    = pf.thresholds    # [T], 美元
        ratios = pf.ratios        # [T], 累積比例

        # widths：若 payout function 有就用；否則以相鄰 thr 差估，最後一段沿用倒數第二段寬
        if hasattr(pf, "widths"):
            widths = pf.widths
        else:
            d = torch.diff(thr)
            last = d[-1] if d.numel() > 0 else torch.tensor(1.0, device=thr.device, dtype=thr.dtype)
            widths = torch.cat([d, last.view(1)])

        # 標準化到 [1,H,E]
        if observed_losses.dim() == 1:          # [E]
            loss_obs = observed_losses.view(1, 1, -1)
        elif observed_losses.dim() == 2:        # [H,E]
            loss_obs = observed_losses.unsqueeze(0)
        elif observed_losses.dim() == 3:        # [1,H,E] 或 [B,H,E]
            loss_obs = observed_losses[:1]
        else:
            raise ValueError("observed_losses 需為 [E]、[H,E] 或 [1,H,E]")

        payout_ratio = torch.zeros_like(loss_obs)
        prev = 0.0
        for i in range(len(thr)):
            dr = float(ratios[i].item() - prev)
            if dr <= 0:
                prev = float(ratios[i]); continue
            t = float(thr[i].item())
            w = float(widths[i].item())
            r = (loss_obs - t) / max(w, 1e-6)
            ramp = torch.clamp(r, 0.0, 1.0)        # 硬線性 0→1
            payout_ratio = payout_ratio + dr * ramp
            prev = float(ratios[i])

        payout = payout_ratio * pf.max_payout      # [1,H,E]（美元）
        return payout.sum(dim=1).squeeze(0)        # → [E]

    
    def train_epoch(self, hazard_intensities: torch.Tensor,
                   exposure_values: torch.Tensor,
                   observed_losses: torch.Tensor,
                   n_samples: int = 10,
                   spatial_data: 'SimulatedSpatialData' = None) -> Dict[str, float]:
        """GPU加速的訓練epoch"""
        
        start_time = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        
        # 將數據移動到主GPU
        hazard_intensities = hazard_intensities.to(self.device)
        exposure_values = exposure_values.to(self.device)
        observed_losses = observed_losses.to(self.device)
        
        # 監控GPU記憶體使用
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = [torch.cuda.memory_allocated(i) / 1e6 for i in GPU_DEVICES]
        
        
        # === Warmup 權重計算 ===
        epoch_idx = len(self.loss_history) + 1
        beta_kl = min(1.0, epoch_idx / self.beta_warmup_epochs) * self.beta_kl_max
        lambda_crps = min(1.0, epoch_idx / self.lambda_warmup_epochs) * self.lambda_crps_max
        
        # === 訓練期：用平滑近似（軟條款） ===
        base_model = self.model.module if hasattr(self.model, 'module') else self.model
        base_model.payout_function.train()
        base_model.payout_function.eval_hard = False  # 訓練用「軟」條款
        self._anneal_tau(base_model.payout_function)  # 依 epoch 退火 tau
        
        # 計算損失（DP使用forward可並行）
        if self.enable_multi_gpu:
            outputs = self.model(
                hazard_intensities, exposure_values, observed_losses, n_samples, spatial_data
            )
            # DataParallel 將返回張量堆疊（num_devices,)
            total_loss, elbo, crps_term, kl_div = outputs
            elbo_scalar = self._as_scalar(elbo)
            crps_scalar = self._as_scalar(crps_term)
            kl_scalar = self._as_scalar(kl_div)
            
            # 正確的 KL 權重：ELBO 已包含 KL，用 (beta_kl - 1.0) 調整
            loss_corrected = -elbo_scalar + (beta_kl - 1.0) * kl_scalar + lambda_crps * crps_scalar
            total_loss_scalar = self._as_scalar(loss_corrected)
            
            # CRPS 數值健康性檢查
            assert torch.isfinite(crps_scalar), "CRPS NaN/Inf detected"
            if not (0.0 <= float(crps_scalar) <= 2.5):
                print(f"⚠️ CRPS(u) out-of-range: {float(crps_scalar):.3f}")
                
            # 多GPU路徑也需要統一的日誌打印（如果需要的話）
            if self._should_log(epoch_idx):
                loss_value = float(total_loss_scalar.detach().cpu())
                elbo_value = float(elbo_scalar.detach().cpu())
                kl_value = float(kl_scalar.detach().cpu())
                crps_value = float(crps_scalar.detach().cpu())
                print(f"[Multi-GPU Epoch {epoch_idx:3d}] loss={loss_value:.3f} | ELBO={elbo_value:+.3f} "
                      f"| KL={kl_value:.3f}(β={beta_kl:.2f}) | CRPS(u)={crps_value:.3f}(λ={lambda_crps:.2f})")
        else:
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            total_loss, elbo, crps_term_orig, kl_div = base_model(
                hazard_intensities, exposure_values, observed_losses, n_samples, spatial_data
            )
            
            # 用穩定的 MC-CRPS unitless 覆蓋（避免解析式爆數）
            with torch.no_grad():
                cap = float(getattr(base_model, 'payout_scale', 1.0))
                # 事件層真值（硬條款）
                y_indemn_evt = self._indemnity_hard(base_model, observed_losses)  # [E]
                # 從 predictive 抽樣
                S = 32
                # 確保 region_assignments 是正確的格式
                region_assignments = None
                if spatial_data is not None and hasattr(spatial_data, 'region_assignments'):
                    region_assignments = spatial_data.region_assignments
                    if not isinstance(region_assignments, torch.Tensor):
                        region_assignments = torch.tensor(region_assignments, dtype=torch.long, device=hazard_intensities.device)
                
                pred_samples = base_model._sample_total_payout_from_loss(
                    base_model.hbm(hazard_intensities, exposure_values, 
                                   base_model._sample_theta(n_samples), 
                                   region_assignments), 
                    n_pred_samples=S
                )  # [S,E] 金額
                crps_u_mc = base_model._mc_crps_unitless(pred_samples, y_indemn_evt, cap).mean()  # 單位化的 mean
                crps_term = crps_u_mc
            
            # 確保所有組件是標量
            elbo_scalar = self._as_scalar(elbo)
            crps_scalar = self._as_scalar(crps_term)
            kl_scalar = self._as_scalar(kl_div)
            
            # 正確的 KL 權重：ELBO 已包含 KL，用 (beta_kl - 1.0) 調整
            # beta_kl=0: 純重構損失, beta_kl=1: 純VI目標
            loss_corrected = -elbo_scalar + (beta_kl - 1.0) * kl_scalar + lambda_crps * crps_scalar
            total_loss_scalar = self._as_scalar(loss_corrected)
        
        # CRPS 數值健康性檢查
        assert torch.isfinite(crps_scalar), "CRPS NaN/Inf detected"
        if not (0.0 <= float(crps_scalar) <= 2.5):
            print(f"⚠️ CRPS(u) out-of-range: {float(crps_scalar):.3f}")
        
        # 確保損失是標量用於反向傳播
        assert total_loss_scalar.ndim == 0, f"Loss must be scalar, got shape {total_loss_scalar.shape}"
        
        # 反向傳播 
        total_loss_scalar.backward()
        
        # 梯度裁剪
        # 1) 梯度裁剪 + 非有限梯度置零
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    bad = ~torch.isfinite(p.grad)
                    if bad.any():
                        p.grad[bad] = 0.0

        
        # 參數更新
        self.optimizer.step()
        
        with torch.no_grad():
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            # log_sigma_theta 與 mu_theta 約束在合理範圍
            base_model.log_sigma_theta.clamp_(-10.0, 10.0)
            base_model.mu_theta.clamp_(-20.0, 20.0)
            
            # 對 σ_obs 參數（第6個參數，log_σ_obs）加更強約束
            # 限制 log_σ_obs 在 [-5, 5] 範圍，對應 σ_obs ∈ [0.007M, 148M] 
            if base_model.mu_theta.numel() >= 7:
                base_model.mu_theta[6].clamp_(-5.0, 5.0)
                if base_model.log_sigma_theta.numel() >= 7:
                    base_model.log_sigma_theta[6].clamp_(-3.0, 1.0)  # 更小的變異性
            
        # 記錄性能指標
        epoch_time = time.time() - start_time
        self.training_times.append(epoch_time)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_after = [torch.cuda.memory_allocated(i) / 1e6 for i in GPU_DEVICES]
            self.gpu_memory_usage.append({
                'before': memory_before,
                'after': memory_after,
                'peak': [torch.cuda.max_memory_allocated(i) / 1e6 for i in GPU_DEVICES]
            })
        
        # 參數型基差風險（事件層）：| parametric(hazard, hard) - indemnity(loss, hard) |
        with torch.no_grad():
            base_model = self.model.module if hasattr(self.model, 'module') else self.model

            # 1) 參數型賠付（硬 max / 硬 ramp）
            if base_model.param_target is not None:
                y_parametric = self._parametric_payout_hard(base_model, hazard_intensities)  # [E]
            else:
                # 後備：用損失總額的硬條款理賠當作「參數型」(讓指標不為常數)
                y_parametric = self._indemnity_hard(base_model, observed_losses)  # [E]

            # 2) 觀測損失的「實際理賠」（硬條款）
            y_indemn = self._indemnity_hard(base_model, observed_losses)                     # [E]

            # 添加輕量偵錯信息（每40個epoch打印一次）
            nz_param  = (y_parametric > 0).float().mean().item()
            nz_indemn = (y_indemn     > 0).float().mean().item()
            
            # 使用統一的日誌節流 - 確保loss計算與打印一致
            if self._should_log(epoch_idx):
                # 用同一組標量變數計算並打印loss，確保一致性
                loss_value = float(total_loss_scalar.detach().cpu())
                elbo_value = float(elbo_scalar.detach().cpu())
                kl_value = float(kl_scalar.detach().cpu())
                crps_value = float(crps_scalar.detach().cpu())
                
                # 驗證loss計算公式一致性
                expected_loss = -elbo_value + (beta_kl - 1.0) * kl_value + lambda_crps * crps_value
                if abs(loss_value - expected_loss) > 1e-3:
                    print(f"⚠️ Loss不一致: 實際={loss_value:.3f} vs 期望={expected_loss:.3f}")
                
                print(f"[Epoch {epoch_idx:3d}] loss={loss_value:.3f} | ELBO={elbo_value:+.3f} "
                      f"| KL={kl_value:.3f}(β={beta_kl:.2f}) | CRPS(u)={crps_value:.3f}(λ={lambda_crps:.2f})")
                print(f"   觸發率 param={nz_param:.3f} indemn={nz_indemn:.3f} | "
                      f"均值 param=${y_parametric.mean().item()/1e6:.1f}M indemn=${y_indemn.mean().item()/1e6:.1f}M")

            trad_basis = torch.mean(torch.abs(y_parametric - y_indemn)).item()

        # 記錄損失與指標
        losses = {
            'total_loss': total_loss_scalar.item(),
            'elbo': elbo_scalar.item(),
            'crps_term': crps_scalar.item(),
            'kl_term': kl_scalar.item(),
            'trad_basis': trad_basis
        }
        losses['epoch_time'] = epoch_time
        self.loss_history.append(losses)
        
        return losses
    
    def _multi_gpu_forward(self, hazard_intensities: torch.Tensor,
                          exposure_values: torch.Tensor, 
                          observed_losses: torch.Tensor,
                          n_samples: int,
                          spatial_data: 'SimulatedSpatialData' = None) -> Dict[str, torch.Tensor]:
        """多GPU並行前向傳播"""
        
        # 將數據分割到不同GPU
        batch_size = hazard_intensities.shape[0]
        chunk_size = batch_size // len(GPU_DEVICES)
        
        # 使用 DataParallel 的 forward 進行並行計算，並聚合結果
        outputs = self.model(
            hazard_intensities, exposure_values, observed_losses, n_samples, spatial_data
        )
        total_loss, elbo, crps_term, kl_div = outputs
        return {
            'total_loss': total_loss.mean(),
            'elbo': elbo.mean(),
            'crps_term': crps_term.mean(),
            'kl_term': kl_div.mean()
        }
    
    def parallel_mcmc_sampling(self, n_total_samples: int, 
                              chunk_size: int = None) -> torch.Tensor:
        """並行MCMC採樣 - 分散到多個GPU"""
        
        if not self.enable_multi_gpu:
            # 單GPU或CPU模式
            if hasattr(self.model, 'module'):
                return self.model.module._sample_theta(n_total_samples)
            else:
                return self.model._sample_theta(n_total_samples)
        
        if chunk_size is None:
            chunk_size = n_total_samples // len(GPU_DEVICES)
        
        print(f"🔄 並行MCMC採樣: {n_total_samples}個樣本分散到{len(GPU_DEVICES)}個GPU")
        
        # 分配採樣任務到不同GPU
        sample_chunks = []
        remaining_samples = n_total_samples
        
        for i, gpu_id in enumerate(GPU_DEVICES):
            # 計算此GPU的採樣數量
            if i == len(GPU_DEVICES) - 1:
                # 最後一個GPU處理剩餘樣本
                n_samples_gpu = remaining_samples
            else:
                n_samples_gpu = min(chunk_size, remaining_samples)
            
            with torch.cuda.device(gpu_id):
                # 在特定GPU上進行採樣
                model_for_sampling = self.model.module if hasattr(self.model, 'module') else self.model
                
                # 使用該GPU上的變分參數進行採樣
                mu_theta_gpu = model_for_sampling.mu_theta.to(f'cuda:{gpu_id}')
                log_sigma_theta_gpu = model_for_sampling.log_sigma_theta.to(f'cuda:{gpu_id}')
                sigma_theta_gpu = torch.exp(log_sigma_theta_gpu)
                
                # 重參數化採樣
                epsilon = torch.randn(n_samples_gpu, model_for_sampling.n_hbm_params, 
                                    device=f'cuda:{gpu_id}')
                theta_samples_gpu = (mu_theta_gpu.unsqueeze(0) + 
                                   sigma_theta_gpu.unsqueeze(0) * epsilon)
                
                sample_chunks.append(theta_samples_gpu)
                remaining_samples -= n_samples_gpu
                
            print(f"  GPU {gpu_id}: {n_samples_gpu}個樣本")
        
        # 將所有GPU結果聚合到主GPU
        all_samples = torch.cat([chunk.to(self.device) for chunk in sample_chunks], dim=0)
        
        print(f"✅ 並行採樣完成: {all_samples.shape[0]}個總樣本")
        return all_samples
        
    def evaluate(self, hazard_intensities: torch.Tensor,
                exposure_values: torch.Tensor,
                observed_losses: torch.Tensor,
                n_samples: int = 50,
                spatial_data: 'SimulatedSpatialData' = None) -> Dict[str, float]:
        """GPU加速的模型評估（硬條款 + 正確的 basis risk 指標）"""
        self.model.eval()

        # 移動數據到GPU
        hazard_intensities = hazard_intensities.to(self.device)
        exposure_values    = exposure_values.to(self.device)
        observed_losses    = observed_losses.to(self.device)

        # 小工具：把任何張量壓成 0 維（預設用 mean），確保能 .item()
        def _as_scalar_eval(x):
            if torch.is_tensor(x):
                return x if x.ndim == 0 else x.mean()
            # 不是 tensor 也轉成張量，保持型別一致
            return torch.as_tensor(x, device=self.device)

        with torch.no_grad():
            base_model = self.model.module if hasattr(self.model, 'module') else self.model

            # 前向（通常會回傳每事件的向量）
            total_loss, elbo, crps_term, kl_div = base_model(
                hazard_intensities, exposure_values, observed_losses, n_samples, spatial_data
            )

            # === 硬條款：事件層基差 ===
            if base_model.param_target is not None:
                y_parametric = self._parametric_payout_hard(base_model, hazard_intensities)  # [E]
            else:
                # ✅ 修正：不能 sum(dim=0) 變成 [E] 再丟 _indemnity_hard
                #   這裡直接用 [H,E] 的 observed_losses
                y_parametric = self._indemnity_hard(base_model, observed_losses)             # [E]

            y_indemn = self._indemnity_hard(base_model, observed_losses)                     # [E]
            trad_basis = torch.mean(torch.abs(y_parametric - y_indemn))                      # scalar

            # 添加輕量偵錯信息（評估時總是打印）
            nz_param  = (y_parametric > 0).float().mean().item()
            nz_indemn = (y_indemn     > 0).float().mean().item()
            print(f"[Eval] 觸發率 param={nz_param:.3f} indemn={nz_indemn:.3f} | "
                  f"均值 param=${y_parametric.mean().item()/1e6:.1f}M indemn=${y_indemn.mean().item()/1e6:.1f}M")

            # 先在 GPU 上把它們壓成 0 維張量
            total_loss_s = _as_scalar_eval(total_loss)
            elbo_s       = _as_scalar_eval(elbo)
            crps_s       = _as_scalar_eval(crps_term)
            kl_s         = _as_scalar_eval(kl_div)
            trad_s       = _as_scalar_eval(trad_basis)

            loss_dict = {
                'total_loss': total_loss_s,
                'elbo': elbo_s,
                'crps_term': crps_s,
                'kl_term': kl_s,
                'trad_basis': trad_s
            }

        # 轉成 Python float 回傳（此時皆為 0 維）
        return {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()}

    def evaluate_with_metrics(self, hazard_intensities: torch.Tensor,
                              exposure_values: torch.Tensor,
                              observed_losses: torch.Tensor,
                              n_samples: int = 30,
                              spatial_data: 'SimulatedSpatialData' = None,
                              n_pred_samples: int = 100) -> Dict[str, float]:
        """計算CRPS與robust log-likelihood（評估用，不影響訓練）。"""
        self.model.eval()
        hazard_intensities = hazard_intensities.to(self.device)
        exposure_values = exposure_values.to(self.device)
        observed_losses = observed_losses.to(self.device)
        with torch.no_grad():
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            metrics = base_model.evaluate_metrics(
                hazard_intensities, exposure_values, observed_losses,
                n_samples=n_samples, spatial_data=spatial_data,
                n_pred_samples=n_pred_samples
            )
        # to cpu scalars
        return {k: (v.item() if hasattr(v, 'item') else v) for k, v in metrics.items()}
    
    def summarize_hbm(self, param_names: Optional[List[str]] = None, n_samples: int = 2000):
        """生成HBM後驗參數摘要"""
        base = self.model.module if hasattr(self.model, 'module') else self.model
        with torch.no_grad():
            mu = base.mu_theta.detach().cpu()              # [P]
            sigma = torch.exp(base.log_sigma_theta).detach().cpu()
            P = mu.numel()
            if not param_names or len(param_names) != P:
                param_names = [f"θ{j}" for j in range(P)]
            eps = torch.randn(n_samples, P)
            theta = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # [S,P]
            mean  = theta.mean(0).numpy()
            std   = theta.std(0, unbiased=True).numpy()
            q05   = theta.quantile(0.05, dim=0).numpy()
            q50   = theta.quantile(0.50, dim=0).numpy()
            q95   = theta.quantile(0.95, dim=0).numpy()
        rows = []
        for j, name in enumerate(param_names):
            rows.append({
                "name": name, "mean": float(mean[j]), "std": float(std[j]),
                "q05": float(q05[j]), "q50": float(q50[j]), "q95": float(q95[j])
            })
        # 少印：只在 verbose 時印；否則回傳讓上層自己處理（寫檔/表格）
        if self.verbose:
            print("\nHBM posterior summary:")
            for r in rows:
                print(f" - {r['name']:>8s}: mean={r['mean']:+.3f} ±{r['std']:.3f} | "
                      f"[{r['q05']:+.3f}, {r['q50']:+.3f}, {r['q95']:+.3f}]")
        return rows

    def get_performance_stats(self) -> Dict[str, Any]:
        """獲取性能統計"""
        stats = {
            'multi_gpu_enabled': self.enable_multi_gpu,
            'device': self.device,
            'gpu_devices': GPU_DEVICES if torch.cuda.is_available() else None,
            'avg_epoch_time': np.mean(self.training_times) if self.training_times else 0,
            'total_epochs': len(self.training_times)
        }
        
        if self.gpu_memory_usage and torch.cuda.is_available():
            latest_memory = self.gpu_memory_usage[-1]
            stats['gpu_memory_mb'] = {
                'peak': latest_memory['peak'],
                'current': latest_memory['after']
            }
        
        return stats

print("✅ 端到端訓練器定義完成")