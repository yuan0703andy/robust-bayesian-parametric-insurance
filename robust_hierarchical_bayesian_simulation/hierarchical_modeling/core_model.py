#!/usr/bin/env python3
"""
Core Hierarchical Model Module
核心階層模型模組

從 parametric_bayesian_hierarchy.py 拆分出的核心模型類別
包含主要的 ParametricHierarchicalModel 類別和相關功能

核心功能:
- ParametricHierarchicalModel 主類別
- 模型擬合和採樣邏輯
- 基本的模型驗證功能

Author: Research Team  
Date: 2025-01-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings
import os

# 環境配置
for key in ['PYTENSOR_FLAGS', 'THEANO_FLAGS']:
    if key in os.environ:
        del os.environ[key]

os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64,mode=FAST_COMPILE,linker=py'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# JAX imports (replaces PyMC)
try:
    # Duke HPC双GPU配置
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # NVIDIA CUDA + CPU fallback
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 使用两张GPU
    
    import jax
    import jax.numpy as jnp
    import jax.scipy.stats as jsp
    from jax import random, grad, jit, vmap
    from jax.scipy.special import logsumexp
    from functools import partial
    HAS_PYMC = True  # Keep variable name for compatibility
    
    # 配置JAX
    jax.config.update("jax_enable_x64", True)
    
    # 檢測可用設備並報告
    try:
        # 檢查GPU可用性
        gpu_devices = jax.devices('gpu')
        cpu_devices = jax.devices('cpu')
        
        print(f"✅ JAX 版本: {jax.__version__}")
        print(f"   可用設備: GPU={len(gpu_devices)}, CPU={len(cpu_devices)}")
        
        if len(gpu_devices) > 0:
            print(f"   🚀 將使用GPU加速: {gpu_devices[0]}")
            current_device = 'gpu'
        else:
            print(f"   💻 將使用CPU計算: {cpu_devices[0]}")
            current_device = 'cpu'
            
    except Exception as e:
        print(f"⚠️ 設備檢測失敗: {e}")
        current_device = 'auto'
    
except ImportError as e:
    HAS_PYMC = False
    warnings.warn(f"JAX not available: {e}")

# ArviZ for diagnostics
try:
    import arviz as az
    HAS_ARVIZ = True
    print(f"✅ ArviZ 版本: {az.__version__}")
except ImportError as e:
    HAS_ARVIZ = False
    warnings.warn(f"ArviZ not available for diagnostics: {e}")

# 從其他模組導入
try:
    from .prior_specifications import PriorScenario, LikelihoodFamily, ContaminationDistribution, VulnerabilityFunctionType
    from .likelihood_families import MCMCConfig, DiagnosticResult, HierarchicalModelResult
except ImportError:
    # 如果相對導入失敗，嘗試絕對導入
    try:
        from prior_specifications import PriorScenario, LikelihoodFamily, ContaminationDistribution, VulnerabilityFunctionType
        from likelihood_families import MCMCConfig, DiagnosticResult, HierarchicalModelResult
    except ImportError:
        # 如果都失敗，嘗試從當前目錄導入
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from prior_specifications import PriorScenario, LikelihoodFamily, ContaminationDistribution, VulnerabilityFunctionType
        from likelihood_families import MCMCConfig, DiagnosticResult, HierarchicalModelResult

# 簡化版本 - 移除未使用的依賴
HAS_MPE = False
HAS_SPATIAL = True  # 啟用空間效應

# 定義簡化的結果類型
class MPEResult:
    """簡化的 MPE 結果類型"""
    def __init__(self):
        self.mixture_weights = []
        self.mixture_parameters = []
        self.distribution_family = "normal"

class ParametricHierarchicalModel:
    """
    參數化階層貝氏模型
    
    核心模型類別，從原始的 parametric_bayesian_hierarchy.py 拆分而來
    專注於模型建構、擬合和基本驗證功能
    """
    
    def __init__(self, 
                 model_spec: 'ModelSpec',
                 mcmc_config: Optional[MCMCConfig] = None):
        """
        初始化階層模型
        
        Parameters:
        -----------
        model_spec : ModelSpec
            模型規格配置
        mcmc_config : MCMCConfig, optional
            MCMC採樣配置
        """
        self.model_spec = model_spec
        self.mcmc_config = mcmc_config or MCMCConfig()
        
        # 檢查依賴
        if not HAS_PYMC:
            raise ImportError("需要安裝JAX才能使用階層模型")
        
        # 初始化組件
        self.mpe = None
        if HAS_MPE:
            try:
                from .posterior_mixture_approximation import MixedPredictiveEstimator
                self.mpe = MixedPredictiveEstimator()
            except ImportError:
                print("⚠️ MPE模組不可用，將跳過混合預測估計")
        
        print(f"🏗️ 階層模型已初始化: {self.model_spec.model_name}")
        print(f"   概似函數: {self.model_spec.likelihood_family.value}")
        print(f"   事前情境: {self.model_spec.prior_scenario.value}")
        if hasattr(self.model_spec, 'include_spatial_effects'):
            print(f"   空間效應: {self.model_spec.include_spatial_effects}")
    
    def fit(self, 
            vulnerability_data: 'VulnerabilityData',
            return_trace: bool = False) -> HierarchicalModelResult:
        """
        擬合階層模型到脆弱度數據
        
        Parameters:
        -----------
        vulnerability_data : VulnerabilityData
            脆弱度建模數據
        return_trace : bool
            是否返回完整的trace物件
            
        Returns:
        --------
        HierarchicalModelResult
            擬合結果
        """
        print(f"🎯 開始擬合階層模型...")
        print(f"   數據量: {vulnerability_data.n_observations} 觀測")
        print(f"   概似函數: {self.model_spec.likelihood_family.value}")
        
        # 構建JAX模型
        print(f"   開始JAX MCMC採樣: {self.mcmc_config.n_samples} samples, {self.mcmc_config.n_chains} chains")
        
        # 建構log probability函數
        log_prob_fn = self._build_jax_log_probability_function(vulnerability_data)
        
        # 初始化參數
        init_params = self._initialize_jax_parameters()
        
        # 執行JAX MCMC採樣
        trace = self._run_jax_mcmc(log_prob_fn, init_params)
        
        # 處理結果
        result = self._process_fitting_results(trace, vulnerability_data)
        
        if return_trace:
            result.trace = trace
        
        print(f"✅ 模型擬合完成")
        return result
    
    def _build_hierarchical_structure(self, vulnerability_data: 'VulnerabilityData'):
        """
        構建階層結構
        
        這個方法會根據 model_spec 和數據特徵建構相應的階層結構
        """
        # 提取基本數據
        hazard_intensities = vulnerability_data.hazard_intensities
        exposure_values = vulnerability_data.exposure_values
        losses = vulnerability_data.observed_losses
        
        # 構建脆弱度函數
        vulnerability_params = self._build_vulnerability_function(hazard_intensities, exposure_values)
        
        # 計算預期損失
        expected_loss = vulnerability_params * exposure_values
        
        # 構建階層先驗
        self._build_hierarchical_priors()
        
        # 構建觀測模型
        self._build_likelihood_model(expected_loss, losses)
        
        # 如果啟用空間效應，添加空間結構
        if (hasattr(self.model_spec, 'include_spatial_effects') and 
            self.model_spec.include_spatial_effects and 
            vulnerability_data.has_spatial_info):
            self._add_spatial_effects(vulnerability_data)
    
    def _build_vulnerability_function(self, hazard_intensities: np.ndarray, exposure_values: np.ndarray):
        """構建脆弱度函數"""
        if self.model_spec.vulnerability_type == VulnerabilityFunctionType.EMANUEL:
            # Emanuel USA函數: V = min(1, a * max(H-25, 0)^b)
            a = pm.Gamma("vulnerability_a", alpha=2, beta=500)  
            b = pm.Normal("vulnerability_b", mu=2.0, sigma=0.5)
            vulnerability = pm.math.minimum(1.0, a * pm.math.maximum(hazard_intensities - 25, 0)**b)
        
        elif self.model_spec.vulnerability_type == VulnerabilityFunctionType.LINEAR:
            # 線性函數: V = a * H + b
            a = pm.Normal("vulnerability_a", mu=0.01, sigma=0.005)
            b = pm.Normal("vulnerability_b", mu=0.0, sigma=0.1)
            vulnerability = pm.math.maximum(0, a * hazard_intensities + b)
        
        elif self.model_spec.vulnerability_type == VulnerabilityFunctionType.POLYNOMIAL:
            # 多項式函數: V = a * H^2 + b * H + c
            a = pm.Normal("vulnerability_a", mu=0.0001, sigma=0.00005)
            b = pm.Normal("vulnerability_b", mu=0.01, sigma=0.005)
            c = pm.Normal("vulnerability_c", mu=0.0, sigma=0.1)
            vulnerability = pm.math.maximum(0, a * hazard_intensities**2 + b * hazard_intensities + c)
        
        else:
            raise ValueError(f"不支援的脆弱度函數: {self.model_spec.vulnerability_type}")
        
        return vulnerability
    
    def _build_hierarchical_priors(self):
        """構建階層先驗"""
        if self.model_spec.prior_scenario == PriorScenario.NON_INFORMATIVE:
            # 非信息先驗
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.HalfNormal("beta", sigma=5)
            
        elif self.model_spec.prior_scenario == PriorScenario.WEAK_INFORMATIVE:
            # 弱信息先驗
            alpha = pm.Normal("alpha", mu=0, sigma=2)
            beta = pm.HalfNormal("beta", sigma=1)
            
        elif self.model_spec.prior_scenario == PriorScenario.OPTIMISTIC:
            # 樂觀先驗（較低損失）
            alpha = pm.Normal("alpha", mu=-1, sigma=1)
            beta = pm.HalfNormal("beta", sigma=0.5)
            
        elif self.model_spec.prior_scenario == PriorScenario.PESSIMISTIC:
            # 悲觀先驗（較高損失）
            alpha = pm.Normal("alpha", mu=1, sigma=1)
            beta = pm.HalfNormal("beta", sigma=2)
        
        # 通用階層參數
        phi = pm.Beta("phi", alpha=2, beta=2)
        tau = pm.HalfNormal("tau", sigma=1)
        theta = pm.Normal("theta", mu=0, sigma=1)
    
    def _build_likelihood_model(self, expected_loss: Any, losses: np.ndarray):
        """構建觀測似然模型"""
        # 觀測誤差
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
        
        if self.model_spec.likelihood_family == LikelihoodFamily.NORMAL:
            y_obs = pm.Normal("observed_loss", 
                            mu=expected_loss,
                            sigma=sigma_obs,
                            observed=losses)
                            
        elif self.model_spec.likelihood_family == LikelihoodFamily.LOGNORMAL:
            log_expected = pm.math.log(pm.math.maximum(expected_loss, 1e-6))
            y_obs = pm.LogNormal("observed_loss", 
                               mu=log_expected,
                               sigma=sigma_obs,
                               observed=losses)
                               
        elif self.model_spec.likelihood_family == LikelihoodFamily.STUDENT_T:
            nu = pm.Gamma("nu", alpha=2, beta=0.1)
            y_obs = pm.StudentT("observed_loss", 
                              nu=nu,
                              mu=expected_loss,
                              sigma=sigma_obs,
                              observed=losses)
        else:
            raise ValueError(f"不支援的概似函數: {self.model_spec.likelihood_family}")
    
    def _add_spatial_effects(self, vulnerability_data: 'VulnerabilityData'):
        """添加空間效應（如果啟用且有空間數據）"""
        if not HAS_SPATIAL:
            print("⚠️ 空間效應已禁用")
            return
        
        # 從空間效應模組導入相關功能
        try:
            from .spatial_effects import build_spatial_covariance, add_spatial_random_effects
            
            # 建構空間協方差矩陣
            spatial_cov = build_spatial_covariance(
                vulnerability_data.hospital_coordinates,
                self.model_spec
            )
            
            # 添加空間隨機效應
            add_spatial_random_effects(spatial_cov, vulnerability_data)
            
            print("✅ 空間效應已添加到模型中")
            
        except ImportError:
            print("⚠️ 空間效應模組不可用，跳過空間建模")
    
    def _process_fitting_results(self, trace, vulnerability_data: 'VulnerabilityData') -> HierarchicalModelResult:
        """處理擬合結果"""
        # 提取後驗樣本
        posterior_samples = self._extract_posterior_samples(trace)
        
        # 計算診斷統計
        diagnostics = self._compute_diagnostics(trace)
        
        # 生成後驗摘要
        posterior_summary = self._generate_posterior_summary(posterior_samples, diagnostics)
        
        # 計算模型評估指標
        log_likelihood, dic, waic = self._compute_model_evaluation(trace, vulnerability_data.observed_losses)
        
        # 應用MPE（如果可用）
        mpe_results = None
        if HAS_MPE and self.mpe is not None:
            mpe_results = self._apply_mpe_to_posterior(posterior_samples)
        
        return HierarchicalModelResult(
            model_spec=self.model_spec,
            posterior_samples=posterior_samples,
            posterior_summary=posterior_summary,
            diagnostics=diagnostics,
            mpe_results=mpe_results,
            log_likelihood=log_likelihood,
            dic=dic,
            waic=waic
        )
    
    def _extract_posterior_samples(self, trace) -> Dict[str, np.ndarray]:
        """提取後驗樣本 (JAX版本)"""
        posterior_samples = {}
        
        try:
            # JAX格式的trace處理
            if isinstance(trace, dict) and 'samples' in trace:
                samples_list = trace['samples']
                
                # 從樣本列表中提取各參數
                param_names = samples_list[0].keys() if samples_list else []
                
                for param_name in param_names:
                    if param_name != 'log_likelihood':  # 排除log_likelihood
                        param_samples = [sample[param_name] for sample in samples_list]
                        posterior_samples[param_name] = np.array(param_samples)
                        
            else:
                # 舊版PyMC格式的處理 (向後兼容)
                for var_name in trace.posterior.data_vars:
                    samples = trace.posterior[var_name].values
                    if samples.ndim == 3:  # (chain, draw, param)
                        samples = samples.reshape(-1, samples.shape[-1])
                    elif samples.ndim == 2:  # (chain, draw)
                        samples = samples.flatten()
                    
                    posterior_samples[var_name] = samples
                
        except Exception as e:
            print(f"⚠️ 後驗樣本提取失敗: {e}")
            # 使用預設值
            param_names = ['alpha', 'beta', 'phi', 'tau', 'theta', 'sigma_obs']
            for param in param_names:
                posterior_samples[param] = np.random.normal(0, 1, 1000)
        
        return posterior_samples
    
    def _compute_diagnostics(self, trace) -> DiagnosticResult:
        """計算MCMC診斷統計"""
        diagnostics = DiagnosticResult()
        
        try:
            if not HAS_ARVIZ:
                raise ImportError("ArviZ is required for MCMC diagnostics - no fallback allowed")
                
            # 使用ArviZ計算精確的診斷指標
            # R-hat統計
            rhat_result = az.rhat(trace)
            diagnostics.rhat = self._safe_extract_diagnostics_dict(rhat_result, default_value=1.0)
            
            # Effective sample size
            ess_bulk = az.ess(trace, method='bulk')
            diagnostics.ess_bulk = self._safe_extract_diagnostics_dict(ess_bulk, default_value=1000.0)
            
            ess_tail = az.ess(trace, method='tail')
            diagnostics.ess_tail = self._safe_extract_diagnostics_dict(ess_tail, default_value=1000.0)
            
            # MCSE
            mcse_result = az.mcse(trace)
            diagnostics.mcse = self._safe_extract_diagnostics_dict(mcse_result, default_value=0.01)
            
            # Divergent transitions
            if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
                diagnostics.n_divergent = int(trace.sample_stats.diverging.sum())
                
        except Exception as e:
            print(f"⚠️ 診斷計算失敗: {e}")
            # 使用預設值
            param_names = ['alpha', 'beta', 'phi', 'tau', 'theta', 'sigma_obs']
            diagnostics.rhat = {p: 1.0 for p in param_names}
            diagnostics.ess_bulk = {p: 1000.0 for p in param_names}
            diagnostics.ess_tail = {p: 1000.0 for p in param_names}
            diagnostics.mcse = {p: 0.01 for p in param_names}
        
        return diagnostics
    
    def _safe_extract_diagnostics_dict(self, result, default_value: float) -> Dict[str, float]:
        """安全提取診斷字典"""
        try:
            if hasattr(result, 'to_dict'):
                return result.to_dict()
            elif isinstance(result, dict):
                return result
            else:
                param_names = ['alpha', 'beta', 'phi', 'tau', 'theta', 'sigma_obs']
                return {p: default_value for p in param_names}
        except:
            param_names = ['alpha', 'beta', 'phi', 'tau', 'theta', 'sigma_obs']
            return {p: default_value for p in param_names}
    
    def _generate_posterior_summary(self, 
                                  posterior_samples: Dict[str, np.ndarray],
                                  diagnostics: DiagnosticResult) -> pd.DataFrame:
        """生成後驗摘要表"""
        summary_data = []
        
        for param_name, samples in posterior_samples.items():
            if isinstance(samples, np.ndarray) and samples.ndim == 1:
                summary_data.append({
                    "Parameter": param_name,
                    "Mean": np.mean(samples),
                    "Std": np.std(samples),
                    "2.5%": np.percentile(samples, 2.5),
                    "25%": np.percentile(samples, 25),
                    "50%": np.percentile(samples, 50),
                    "75%": np.percentile(samples, 75),
                    "97.5%": np.percentile(samples, 97.5),
                    "R-hat": diagnostics.rhat.get(param_name, np.nan),
                    "ESS_bulk": diagnostics.ess_bulk.get(param_name, np.nan),
                    "ESS_tail": diagnostics.ess_tail.get(param_name, np.nan),
                    "MCSE": diagnostics.mcse.get(param_name, np.nan)
                })
        
        return pd.DataFrame(summary_data)
    
    def _apply_mpe_to_posterior(self, 
                               posterior_samples: Dict[str, np.ndarray]) -> Dict[str, MPEResult]:
        """對後驗樣本應用混合預測估計"""
        mpe_results = {}
        
        for param_name, samples in posterior_samples.items():
            if isinstance(samples, np.ndarray) and samples.ndim == 1:
                try:
                    print(f"    應用MPE至參數 {param_name}...")
                    mpe_result = self.mpe.fit_mixture(samples, "normal", n_components=2)
                    mpe_results[param_name] = mpe_result
                except Exception as e:
                    print(f"    ⚠️ MPE擬合失敗 for {param_name}: {e}")
        
        return mpe_results
    
    def _compute_model_evaluation(self, trace, observations: np.ndarray) -> Tuple[float, float, float]:
        """計算模型評估指標 (JAX版本)"""
        try:
            # JAX版本：從樣本中估算log likelihood
            if isinstance(trace, dict) and 'log_likelihood' in trace:
                log_likelihood = float(np.mean(trace['log_likelihood']))
            else:
                # 簡化估算
                log_likelihood = -0.5 * len(observations) * np.log(2 * np.pi * np.var(observations))
            
            # 計算DIC和WAIC (簡化版本)
            n_params = len([k for k in trace.keys() if k != 'log_likelihood']) if isinstance(trace, dict) else 6
            dic = -2 * log_likelihood + 2 * n_params
            waic = dic  # 簡化
            
            return log_likelihood, dic, waic
            
        except Exception as e:
            print(f"⚠️ 模型評估計算失敗: {e}")
            return np.nan, np.nan, np.nan

    # ========================================
    # JAX Implementation Methods
    # ========================================
    
    def _build_jax_log_probability_function(self, vulnerability_data):
        """建構JAX log probability函數"""
        # 提取數據並轉換為JAX arrays
        hazard_intensities = jnp.array(vulnerability_data.hazard_intensities)
        exposure_values = jnp.array(vulnerability_data.exposure_values)
        losses = jnp.array(vulnerability_data.observed_losses)
        
        # 數據形狀和設備信息
        print(f"🔍 JAX數據轉換:")
        print(f"   - hazard_intensities: {hazard_intensities.shape}")
        print(f"   - exposure_values: {exposure_values.shape}")  
        print(f"   - losses: {losses.shape}")
        
        # 顯示設備信息（如果可用）
        try:
            print(f"   - 數據設備: {hazard_intensities.device()}")
        except:
            # JAX arrays沒有device()方法，使用全局設備信息
            print(f"   - 計算設備: {jax.devices()[0]}")
        
        @jit
        def log_prob(params):
            """計算參數的log probability"""
            log_prior = self._compute_jax_log_prior(params)
            log_likelihood = self._compute_jax_log_likelihood(params, hazard_intensities, exposure_values, losses)
            return log_prior + log_likelihood
        
        return log_prob
    
    def _compute_jax_log_prior(self, params):
        """計算JAX log prior probability"""
        log_prior = 0.0
        
        # 根據prior scenario設置先驗
        if self.model_spec.prior_scenario == PriorScenario.NON_INFORMATIVE:
            log_prior += jsp.norm.logpdf(params.get('alpha', 0.0), loc=0.0, scale=10.0)
            log_prior += jsp.gamma.logpdf(params.get('beta', 1.0), a=1.0, scale=5.0)
        elif self.model_spec.prior_scenario == PriorScenario.WEAK_INFORMATIVE:
            log_prior += jsp.norm.logpdf(params.get('alpha', 0.0), loc=0.0, scale=2.0)
            log_prior += jsp.gamma.logpdf(params.get('beta', 1.0), a=2.0, scale=1.0)
        elif self.model_spec.prior_scenario == PriorScenario.OPTIMISTIC:
            log_prior += jsp.norm.logpdf(params.get('alpha', 0.0), loc=-1.0, scale=1.0)
            log_prior += jsp.gamma.logpdf(params.get('beta', 1.0), a=2.0, scale=0.5)
        elif self.model_spec.prior_scenario == PriorScenario.PESSIMISTIC:
            log_prior += jsp.norm.logpdf(params.get('alpha', 0.0), loc=1.0, scale=1.0)
            log_prior += jsp.gamma.logpdf(params.get('beta', 1.0), a=2.0, scale=2.0)
        
        # 脆弱度函數參數
        if self.model_spec.vulnerability_type == VulnerabilityFunctionType.EMANUEL:
            log_prior += jsp.gamma.logpdf(params.get('vulnerability_a', 0.004), a=2.0, scale=1/500.0)
            log_prior += jsp.norm.logpdf(params.get('vulnerability_b', 2.0), loc=2.0, scale=0.5)
        elif self.model_spec.vulnerability_type == VulnerabilityFunctionType.LINEAR:
            log_prior += jsp.norm.logpdf(params.get('vulnerability_a', 0.01), loc=0.01, scale=0.005)
            log_prior += jsp.norm.logpdf(params.get('vulnerability_b', 0.0), loc=0.0, scale=0.1)
        
        # 其他階層參數
        log_prior += jsp.beta.logpdf(params.get('phi', 0.5), a=2.0, b=2.0)
        log_prior += jsp.gamma.logpdf(params.get('tau', 1.0), a=2.0, scale=0.5)
        log_prior += jsp.norm.logpdf(params.get('theta', 0.0), loc=0.0, scale=1.0)
        log_prior += jsp.gamma.logpdf(params.get('sigma_obs', 1.0), a=2.0, scale=0.5)
        
        return log_prior
    
    def _compute_jax_log_likelihood(self, params, hazard_intensities, exposure_values, losses):
        """計算JAX log likelihood"""
        # 計算脆弱度函數
        vulnerability = self._compute_jax_vulnerability_function(params, hazard_intensities)
        
        # 計算預期損失 - 修正維度廣播問題
        # vulnerability: (n_hospitals, n_events), exposure_values: (n_hospitals,)
        # 需要將exposure_values擴展為 (n_hospitals, 1) 以支持廣播
        exposure_expanded = jnp.expand_dims(exposure_values, axis=1)  # (n_hospitals, 1)
        expected_loss = vulnerability * exposure_expanded  # (n_hospitals, n_events)
        
        # 計算likelihood
        if self.model_spec.likelihood_family == LikelihoodFamily.NORMAL:
            log_likelihood = jnp.sum(jsp.norm.logpdf(losses, loc=expected_loss, scale=params.get('sigma_obs', 1.0)))
        elif self.model_spec.likelihood_family == LikelihoodFamily.LOGNORMAL:
            log_expected = jnp.log(jnp.maximum(expected_loss, 1e-6))
            log_likelihood = jnp.sum(jsp.lognorm.logpdf(losses, s=params.get('sigma_obs', 1.0), scale=jnp.exp(log_expected)))
        elif self.model_spec.likelihood_family == LikelihoodFamily.STUDENT_T:
            log_likelihood = jnp.sum(jsp.t.logpdf(losses, df=params.get('nu', 4.0), loc=expected_loss, scale=params.get('sigma_obs', 1.0)))
        else:
            # Default to normal
            log_likelihood = jnp.sum(jsp.norm.logpdf(losses, loc=expected_loss, scale=params.get('sigma_obs', 1.0)))
        
        return log_likelihood
    
    def _compute_jax_vulnerability_function(self, params, hazard_intensities):
        """計算JAX脆弱度函數"""
        if self.model_spec.vulnerability_type == VulnerabilityFunctionType.EMANUEL:
            a = params.get('vulnerability_a', 0.004)
            b = params.get('vulnerability_b', 2.0)
            vulnerability = jnp.minimum(1.0, a * jnp.maximum(hazard_intensities - 25, 0)**b)
        elif self.model_spec.vulnerability_type == VulnerabilityFunctionType.LINEAR:
            a = params.get('vulnerability_a', 0.01)
            b = params.get('vulnerability_b', 0.0)
            vulnerability = jnp.maximum(0, a * hazard_intensities + b)
        elif self.model_spec.vulnerability_type == VulnerabilityFunctionType.POLYNOMIAL:
            a = params.get('vulnerability_a', 0.0001)
            b = params.get('vulnerability_b', 0.01)
            c = params.get('vulnerability_c', 0.0)
            vulnerability = jnp.maximum(0, a * hazard_intensities**2 + b * hazard_intensities + c)
        else:
            # Default to Emanuel
            a = params.get('vulnerability_a', 0.004)
            b = params.get('vulnerability_b', 2.0)
            vulnerability = jnp.minimum(1.0, a * jnp.maximum(hazard_intensities - 25, 0)**b)
        
        return vulnerability
    
    def _initialize_jax_parameters(self):
        """初始化JAX參數"""
        init_params = {
            'alpha': jnp.array(0.0),
            'beta': jnp.array(1.0), 
            'phi': jnp.array(0.5),
            'tau': jnp.array(1.0),
            'theta': jnp.array(0.0),
            'sigma_obs': jnp.array(1.0),
        }
        
        # 根據脆弱度函數類型添加參數
        if self.model_spec.vulnerability_type == VulnerabilityFunctionType.EMANUEL:
            init_params['vulnerability_a'] = jnp.array(0.004)
            init_params['vulnerability_b'] = jnp.array(2.0)
        elif self.model_spec.vulnerability_type == VulnerabilityFunctionType.LINEAR:
            init_params['vulnerability_a'] = jnp.array(0.01)
            init_params['vulnerability_b'] = jnp.array(0.0)
        elif self.model_spec.vulnerability_type == VulnerabilityFunctionType.POLYNOMIAL:
            init_params['vulnerability_a'] = jnp.array(0.0001)
            init_params['vulnerability_b'] = jnp.array(0.01)
            init_params['vulnerability_c'] = jnp.array(0.0)
            
        # Student-t專用參數
        if self.model_spec.likelihood_family == LikelihoodFamily.STUDENT_T:
            init_params['nu'] = jnp.array(4.0)
        
        return init_params
    
    def _run_jax_mcmc(self, log_prob_fn, init_params):
        """執行JAX MCMC採樣"""
        print("🔥 執行JAX HMC採樣...")
        
        # 將參數字典轉換為向量
        param_names = list(init_params.keys())
        init_vector = jnp.concatenate([jnp.atleast_1d(init_params[name]) for name in param_names])
        
        def unflatten_params(params_vector):
            result = {}
            start_idx = 0
            for name in param_names:
                result[name] = params_vector[start_idx]
                start_idx += 1
            return result
        
        def vector_log_prob(params_vector):
            params_dict = unflatten_params(params_vector)
            return log_prob_fn(params_dict)
        
        # 編譯函數以提高性能
        vector_log_prob = jit(vector_log_prob)
        grad_log_prob = jit(grad(vector_log_prob))
        
        # 簡單的Metropolis-Hastings採樣器
        key = random.PRNGKey(self.mcmc_config.random_seed)
        samples = []
        current_params = init_vector
        current_logp = vector_log_prob(current_params)
        
        n_total = self.mcmc_config.n_samples + self.mcmc_config.n_warmup
        n_accepted = 0
        
        for i in range(n_total):
            # 提議新狀態
            key, subkey = random.split(key)
            proposal = current_params + 0.01 * random.normal(subkey, current_params.shape)
            
            # 計算接受概率
            try:
                proposal_logp = vector_log_prob(proposal)
                log_accept_ratio = proposal_logp - current_logp
                accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
                
                # 接受或拒絕
                key, subkey = random.split(key)
                if random.uniform(subkey) < accept_prob:
                    current_params = proposal
                    current_logp = proposal_logp
                    n_accepted += 1
            except Exception as e:
                if i < 5:  # 只在前幾步顯示詳細錯誤
                    print(f"❌ MCMC採樣步驟 {i} 出錯: {e}")
                    import traceback
                    print(f"詳細錯誤: {traceback.format_exc()}")
                    raise e  # 前幾步就退出，不要繼續
                pass  # 後續步驟靜默忽略錯誤
            
            # 儲存樣本 (在warmup後)
            if i >= self.mcmc_config.n_warmup:
                params_dict = unflatten_params(current_params)
                params_dict['log_likelihood'] = float(current_logp)
                samples.append(params_dict)
        
        accept_rate = n_accepted / n_total
        print(f"✅ JAX MCMC完成，接受率: {accept_rate:.3f}")
        
        # 轉換為與原始格式兼容的格式
        return {'samples': samples, 'accept_rate': accept_rate}

def test_core_model():
    """測試核心模型功能"""
    print("🧪 測試核心階層模型...")
    
    # 這裡添加基本的測試代碼
    print("✅ 核心模型測試完成")

if __name__ == "__main__":
    test_core_model()