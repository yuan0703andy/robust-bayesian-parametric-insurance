from .model import UnifiedEndToEndVIModel
from utils.gpu_setup import device, USE_MULTI_GPU, GPU_DEVICES
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
        
    def __init__(self, model: UnifiedEndToEndVIModel, learning_rate: float = 0.0001,
                    enable_multi_gpu: bool = False, verbose: bool = False):
        self.original_model = model
        # 暫時停用多卡，避免DataParallel沿H維切片導致維度不一致
        self.enable_multi_gpu = False
        self.verbose = verbose
        
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
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_history = []
        
        # GPU性能監控
        self.gpu_memory_usage = []
        self.training_times = []
        
        
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
            total_loss_scalar = total_loss.mean()
            elbo_scalar = elbo.mean()
            crps_scalar = crps_term.mean()
            kl_scalar = kl_div.mean()
        else:
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            total_loss, elbo, crps_term, kl_div = base_model(
                hazard_intensities, exposure_values, observed_losses, n_samples, spatial_data
            )
            total_loss_scalar, elbo_scalar, crps_scalar, kl_scalar = total_loss, elbo, crps_term, kl_div
        
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

            # 添加輕量偵錯信息（可選）
            nz_param  = (y_parametric > 0).float().mean().item()
            nz_indemn = (y_indemn     > 0).float().mean().item()
            # print(f"[debug] trigger% param={nz_param:.3f} indemn={nz_indemn:.3f}, "
            #       f"param(mean)={y_parametric.mean().item():.1f}, indemn(mean)={y_indemn.mean().item():.1f}")

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

        with torch.no_grad():
            base_model = self.model.module if hasattr(self.model, 'module') else self.model

            # 前向（照舊）
            total_loss, elbo, crps_term, kl_div = base_model(
                hazard_intensities, exposure_values, observed_losses, n_samples, spatial_data
            )

            # 顯式硬條款（我們的 hard payout 是在 trainer 端自己算，不靠 model 裡的平滑）
            # → 不需要再設 eval_hard flag；保持 model 的平滑流程只用於訓練/CRPS
            # 正確的事件層基差風險：
            if base_model.param_target is not None:
                y_parametric = self._parametric_payout_hard(base_model, hazard_intensities)  # [E]
            else:
                y_parametric = self._indemnity_hard(base_model, observed_losses)  # 後備

            y_indemn = self._indemnity_hard(base_model, observed_losses)                     # [E]

            # 添加輕量偵錯信息（可選）
            nz_param  = (y_parametric > 0).float().mean().item()
            nz_indemn = (y_indemn     > 0).float().mean().item()
            # print(f"[debug eval] trigger% param={nz_param:.3f} indemn={nz_indemn:.3f}, "
            #       f"param(mean)={y_parametric.mean().item():.1f}, indemn(mean)={y_indemn.mean().item():.1f}")

            trad_basis = torch.mean(torch.abs(y_parametric - y_indemn))

            loss_dict = {
                'total_loss': total_loss,
                'elbo': elbo,
                'crps_term': crps_term,
                'kl_term': kl_div,
                'trad_basis': trad_basis
            }

        # to cpu scalars
        return {k: (v.item() if hasattr(v, 'item') else v) for k, v in loss_dict.items()}

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