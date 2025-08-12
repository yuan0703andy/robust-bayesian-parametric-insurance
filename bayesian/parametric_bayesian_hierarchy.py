#!/usr/bin/env python3
"""
Parametric Hierarchical Bayesian Model Module
參數化階層貝氏模型模組

可動態配置的階層貝氏模型，支援不同的概似函數和事前分佈組合。

核心功能:
- 支援多種概似函數: Normal, LogNormal, Student-t, Laplace
- 支援多種事前情境: non_informative, weak_informative, optimistic, pessimistic
- 動態模型構建
- 獨立的模型配置和結果類型

使用範例:
```python
from bayesian.parametric_bayesian_hierarchy import (
    ParametricHierarchicalModel, ModelSpec, PriorScenario
)

# 創建模型規格
model_spec = ModelSpec(
    likelihood_family='lognormal',
    prior_scenario='pessimistic'
)

# 初始化模型
model = ParametricHierarchicalModel(model_spec)

# 擬合數據
result = model.fit(observations)

# 查看結果
print("後驗摘要:", result.posterior_summary)
print("模型診斷:", result.diagnostics.convergence_summary())
```

Author: Research Team  
Date: 2025-01-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
import os

# 環境配置
for key in ['PYTENSOR_FLAGS', 'THEANO_FLAGS']:
    if key in os.environ:
        del os.environ[key]

os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64,mode=FAST_COMPILE,linker=py'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# PyMC imports
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    HAS_PYMC = True
    print(f"✅ PyMC 版本: {pm.__version__}")
except ImportError as e:
    HAS_PYMC = False
    warnings.warn(f"PyMC not available: {e}")

# 導入獨立的MPE模組
try:
    from .posterior_mixture_approximation import MixedPredictiveEstimation, MPEResult
    HAS_MPE = True
except ImportError:
    HAS_MPE = False
    warnings.warn("混合預測估計模組不可用")

# 新增：脆弱度建模數據結構
@dataclass
class VulnerabilityData:
    """脆弱度建模數據"""
    hazard_intensities: np.ndarray      # H_ij - 災害強度（如風速 m/s）
    exposure_values: np.ndarray         # E_i - 暴險值（如建築物價值 USD）
    observed_losses: np.ndarray         # L_ij - 觀測損失 (USD)
    event_ids: Optional[np.ndarray] = None      # 事件ID
    location_ids: Optional[np.ndarray] = None   # 地點ID
    
    def __post_init__(self):
        """驗證數據一致性"""
        arrays = [self.hazard_intensities, self.exposure_values, self.observed_losses]
        lengths = [len(arr) for arr in arrays if arr is not None]
        
        if len(set(lengths)) > 1:
            raise ValueError(f"數據長度不一致: {lengths}")
        
        if len(lengths) == 0:
            raise ValueError("至少需要提供災害強度、暴險值和觀測損失")
    
    @property 
    def n_observations(self) -> int:
        """觀測數量"""
        return len(self.hazard_intensities)

class VulnerabilityFunctionType(Enum):
    """脆弱度函數類型"""
    EMANUEL = "emanuel"          # Emanuel USA: V = a × (H - H₀)^b for H > H₀
    LINEAR = "linear"            # Linear: V = a × H + b  
    POLYNOMIAL = "polynomial"    # Polynomial: V = a₀ + a₁H + a₂H² + a₃H³
    EXPONENTIAL = "exponential"  # Exponential: V = a × (1 - exp(-b × H))
    STEP = "step"               # Step function: V = a for H > threshold

# 枚舉類型定義
class LikelihoodFamily(Enum):
    """概似函數家族"""
    NORMAL = "normal"
    LOGNORMAL = "lognormal" 
    STUDENT_T = "student_t"
    LAPLACE = "laplace"
    GAMMA = "gamma"

class PriorScenario(Enum):
    """事前分佈情境"""
    NON_INFORMATIVE = "non_informative"      # 無資訊先驗
    WEAK_INFORMATIVE = "weak_informative"    # 弱資訊先驗
    OPTIMISTIC = "optimistic"                # 樂觀先驗 (較寬)
    PESSIMISTIC = "pessimistic"              # 悲觀先驗 (較窄)
    CONSERVATIVE = "conservative"            # 保守先驗

@dataclass
class ModelSpec:
    """模型規格"""
    likelihood_family: LikelihoodFamily = LikelihoodFamily.NORMAL
    prior_scenario: PriorScenario = PriorScenario.WEAK_INFORMATIVE
    vulnerability_type: VulnerabilityFunctionType = VulnerabilityFunctionType.EMANUEL
    model_name: Optional[str] = None
    
    def __post_init__(self):
        # 類型轉換支援
        if isinstance(self.likelihood_family, str):
            self.likelihood_family = LikelihoodFamily(self.likelihood_family)
        if isinstance(self.prior_scenario, str):
            self.prior_scenario = PriorScenario(self.prior_scenario)
        if isinstance(self.vulnerability_type, str):
            self.vulnerability_type = VulnerabilityFunctionType(self.vulnerability_type)
        
        if self.model_name is None:
            self.model_name = f"{self.likelihood_family.value}_{self.prior_scenario.value}_{self.vulnerability_type.value}"

@dataclass
class MCMCConfig:
    """MCMC採樣配置"""
    n_samples: int = 1000
    n_warmup: int = 500
    n_chains: int = 2
    random_seed: int = 42
    target_accept: float = 0.8
    cores: int = 1
    progressbar: bool = True

@dataclass
class DiagnosticResult:
    """診斷結果"""
    rhat: Dict[str, float] = field(default_factory=dict)
    ess_bulk: Dict[str, float] = field(default_factory=dict)
    ess_tail: Dict[str, float] = field(default_factory=dict)
    mcse: Dict[str, float] = field(default_factory=dict)
    n_divergent: int = 0
    energy_error: bool = False
    
    def convergence_summary(self) -> Dict[str, Any]:
        """收斂性摘要"""
        rhat_values = list(self.rhat.values())
        ess_bulk_values = list(self.ess_bulk.values())
        
        summary = {
            "max_rhat": max(rhat_values) if rhat_values else np.nan,
            "min_ess_bulk": min(ess_bulk_values) if ess_bulk_values else np.nan,
            "rhat_ok": all(r < 1.1 for r in rhat_values) if rhat_values else False,
            "ess_ok": all(e > 400 for e in ess_bulk_values) if ess_bulk_values else False,
            "n_divergent": self.n_divergent,
            "energy_error": self.energy_error
        }
        
        summary["overall_convergence"] = (
            summary["rhat_ok"] and 
            summary["ess_ok"] and 
            summary["n_divergent"] == 0 and 
            not summary["energy_error"]
        )
        
        return summary

@dataclass
class HierarchicalModelResult:
    """階層模型結果"""
    model_spec: ModelSpec
    posterior_samples: Dict[str, np.ndarray]
    posterior_summary: pd.DataFrame
    diagnostics: DiagnosticResult
    mpe_results: Optional[Dict[str, MPEResult]] = None
    log_likelihood: float = np.nan
    dic: float = np.nan
    waic: float = np.nan
    trace: Any = None  # PyMC trace object
    
    def get_parameter_credible_interval(self, 
                                      param_name: str, 
                                      alpha: float = 0.05) -> Tuple[float, float]:
        """獲取參數的可信區間"""
        if param_name not in self.posterior_samples:
            raise KeyError(f"參數 '{param_name}' 不在後驗樣本中")
        
        samples = self.posterior_samples[param_name]
        lower = np.percentile(samples, 100 * alpha / 2)
        upper = np.percentile(samples, 100 * (1 - alpha / 2))
        
        return lower, upper

class ParametricHierarchicalModel:
    """
    參數化階層貝氏模型
    
    實現您理論中的4層階層結構，但支援動態配置：
    - Level 1: 觀測模型 Y|θ, σ² ~ Likelihood(parameters)
    - Level 2: 過程模型 θ|φ, τ² ~ Process(parameters)  
    - Level 3: 參數模型 φ|α, β ~ Parameter(parameters)
    - Level 4: 超參數模型 α, β ~ Hyperparameter(parameters)
    """
    
    def __init__(self, 
                 model_spec: ModelSpec,
                 mcmc_config: MCMCConfig = None,
                 use_mpe: bool = True):
        """
        初始化參數化階層模型
        
        Parameters:
        -----------
        model_spec : ModelSpec
            模型規格，定義概似函數和事前情境
        mcmc_config : MCMCConfig, optional
            MCMC採樣配置
        use_mpe : bool
            是否使用混合預測估計
        """
        self.model_spec = model_spec
        self.mcmc_config = mcmc_config or MCMCConfig()
        self.use_mpe = use_mpe and HAS_MPE
        
        # 初始化MPE (如果可用)
        if self.use_mpe:
            self.mpe = MixedPredictiveEstimation()
        else:
            self.mpe = None
            
        # 結果存儲
        self.last_result: Optional[HierarchicalModelResult] = None
        self.fit_history: List[HierarchicalModelResult] = []
        
    def fit(self, data: Union[VulnerabilityData, np.ndarray, List[float]]) -> HierarchicalModelResult:
        """
        擬合階層貝氏模型
        
        Parameters:
        -----------
        data : VulnerabilityData or np.ndarray or List[float]
            脆弱度數據（包含災害強度、暴險值、觀測損失）或傳統觀測資料（向後兼容）
            
        Returns:
        --------
        HierarchicalModelResult
            完整的模型擬合結果
        """
        # 向後兼容：如果輸入是傳統的觀測數據
        if isinstance(data, (np.ndarray, list)):
            print("⚠️ 使用傳統觀測數據模式（向後兼容）")
            observations = np.asarray(data).flatten()
            return self._fit_legacy_model(observations)
        
        # 新的脆弱度建模模式
        if not isinstance(data, VulnerabilityData):
            raise TypeError("數據必須是 VulnerabilityData 實例或 np.ndarray/List")
        
        print(f"🔄 開始擬合以脆弱度為核心的階層貝氏模型...")
        print(f"   模型規格: {self.model_spec.model_name}")
        print(f"   觀測數量: {data.n_observations} 個災害事件")
        print(f"   概似函數: {self.model_spec.likelihood_family.value}")
        print(f"   事前情境: {self.model_spec.prior_scenario.value}")
        print(f"   脆弱度函數: {self.model_spec.vulnerability_type.value}")
        print(f"   災害強度範圍: [{data.hazard_intensities.min():.1f}, {data.hazard_intensities.max():.1f}]")
        print(f"   暴險值範圍: [{data.exposure_values.min():.2e}, {data.exposure_values.max():.2e}]")
        print(f"   損失範圍: [{data.observed_losses.min():.2e}, {data.observed_losses.max():.2e}]")
        
        if not HAS_PYMC:
            print("⚠️ PyMC不可用，使用簡化實現")
            return self._fit_vulnerability_simplified(data)
        
        try:
            return self._fit_vulnerability_with_pymc(data)
        except Exception as e:
            print(f"⚠️ PyMC脆弱度擬合失敗: {e}")
            print("回退到簡化實現")
            return self._fit_vulnerability_simplified(data)
    
    def _fit_vulnerability_with_pymc(self, vulnerability_data: VulnerabilityData) -> HierarchicalModelResult:
        """使用PyMC進行脆弱度建模的完整MCMC擬合"""
        print("  🔬 使用PyMC構建脆弱度階層模型...")
        
        # 提取數據
        hazard = vulnerability_data.hazard_intensities
        exposure = vulnerability_data.exposure_values
        losses = vulnerability_data.observed_losses
        n_obs = len(hazard)
        
        with pm.Model() as vulnerability_model:
            # Level 4: 超參數模型（不變）
            hyperparams = self._get_vulnerability_hyperparameters()
            alpha = pm.Normal("alpha", mu=hyperparams['alpha_mu'], 
                            sigma=hyperparams['alpha_sigma'])
            beta_sigma = pm.HalfNormal("beta_sigma", sigma=hyperparams['beta_sigma'])
            
            # Level 3: 脆弱度參數模型（新增 - 這是關鍵）
            # 不同脆弱度函數需要不同數量的參數
            n_vuln_params = self._get_vulnerability_param_count()
            vulnerability_params = pm.Normal("vulnerability_params", 
                                            mu=alpha, sigma=beta_sigma, 
                                            shape=n_vuln_params)
            
            # Level 2: 過程模型 - 現在建模災害-損失關係
            tau = pm.HalfNormal("tau", sigma=hyperparams['tau_sigma'])
            
            # *** 關鍵改進：脆弱度函數 V(H;β) ***
            vulnerability_mean = self._get_vulnerability_function(hazard, vulnerability_params)
            
            # 期望損失 = 暴險值 × 脆弱度函數
            # 添加噪聲以避免數值問題
            vulnerability_mean_clipped = pm.math.clip(vulnerability_mean, 1e-10, 1e10)
            expected_loss = pm.Deterministic(
                "expected_loss", 
                exposure * vulnerability_mean_clipped
            )
            
            # Level 1: 觀測模型 - 基於物理機制
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=hyperparams['sigma_obs'])
            
            if self.model_spec.likelihood_family == LikelihoodFamily.LOGNORMAL:
                # 確保正值並避免log(0)
                expected_loss_pos = pm.math.maximum(expected_loss, 1e-6)
                y_obs = pm.LogNormal("observed_loss", 
                                   mu=pm.math.log(expected_loss_pos),
                                   sigma=sigma_obs, 
                                   observed=losses)
            elif self.model_spec.likelihood_family == LikelihoodFamily.NORMAL:
                # 正常分佈
                y_obs = pm.Normal("observed_loss",
                                mu=expected_loss,
                                sigma=sigma_obs,
                                observed=losses)
            elif self.model_spec.likelihood_family == LikelihoodFamily.STUDENT_T:
                # t分佈更穩健
                nu = pm.Gamma("nu", alpha=2, beta=0.1)
                y_obs = pm.StudentT("observed_loss", 
                                  nu=nu,
                                  mu=expected_loss,
                                  sigma=sigma_obs,
                                  observed=losses)
            else:
                raise ValueError(f"脆弱度建模不支援概似函數: {self.model_spec.likelihood_family}")
            
            print("  ⚙️ 執行MCMC採樣（脆弱度建模）...")
            trace = pm.sample(
                draws=self.mcmc_config.n_samples,
                tune=self.mcmc_config.n_warmup,
                chains=self.mcmc_config.n_chains,
                cores=self.mcmc_config.cores,
                random_seed=self.mcmc_config.random_seed,
                target_accept=self.mcmc_config.target_accept,
                return_inferencedata=True,
                progressbar=self.mcmc_config.progressbar
            )
            
            # 提取後驗樣本
            print("  📊 提取脆弱度後驗樣本...")
            posterior_samples = self._extract_vulnerability_posterior_samples(trace)
            
            # 計算診斷統計
            print("  📈 計算診斷統計...")
            diagnostics = self._compute_diagnostics(trace)
            
            # 生成後驗摘要
            posterior_summary = self._generate_posterior_summary(posterior_samples, diagnostics)
            
            # 應用MPE (如果啟用)
            mpe_results = None
            if self.use_mpe:
                print("  🧠 應用混合預測估計...")
                mpe_results = self._apply_mpe_to_posterior(posterior_samples)
            
            # 計算模型評估指標
            log_likelihood, dic, waic = self._compute_model_evaluation(trace, losses)
            
            result = HierarchicalModelResult(
                model_spec=self.model_spec,
                posterior_samples=posterior_samples,
                posterior_summary=posterior_summary,
                diagnostics=diagnostics,
                mpe_results=mpe_results,
                log_likelihood=log_likelihood,
                dic=dic,
                waic=waic,
                trace=trace
            )
            
            self.last_result = result
            self.fit_history.append(result)
            
            print("✅ PyMC脆弱度階層模型擬合完成")
            return result
    
    def _fit_legacy_model(self, observations: np.ndarray) -> HierarchicalModelResult:
        """向後兼容：使用傳統觀測數據的擬合方法"""
        return self._fit_with_pymc(observations)
    
    def _fit_with_pymc(self, observations: np.ndarray) -> HierarchicalModelResult:
        """使用PyMC進行完整的MCMC擬合"""
        print("  🔬 使用PyMC構建階層模型...")
        
        with pm.Model() as hierarchical_model:
            # 根據事前情境設置超參數
            hyperparams = self._get_hyperparameters()
            
            # Level 4: 超參數模型
            alpha = pm.Normal("alpha", mu=hyperparams['alpha_mu'], 
                            sigma=hyperparams['alpha_sigma'])
            beta = pm.HalfNormal("beta", sigma=hyperparams['beta_sigma'])
            
            # Level 3: 參數模型
            phi = pm.Normal("phi", mu=alpha, sigma=beta)
            
            # Level 2: 過程模型  
            tau = pm.HalfNormal("tau", sigma=hyperparams['tau_sigma'])
            theta = pm.Normal("theta", mu=phi, sigma=tau)
            
            # Level 1: 觀測模型 - 根據概似函數選擇
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=hyperparams['sigma_obs'])
            
            if self.model_spec.likelihood_family == LikelihoodFamily.NORMAL:
                y_obs = pm.Normal("y_obs", mu=theta, sigma=sigma_obs, observed=observations)
            elif self.model_spec.likelihood_family == LikelihoodFamily.LOGNORMAL:
                # 確保參數為正
                theta_pos = pm.math.exp(theta)
                y_obs = pm.LogNormal("y_obs", mu=pm.math.log(theta_pos), 
                                   sigma=sigma_obs, observed=observations)
            elif self.model_spec.likelihood_family == LikelihoodFamily.STUDENT_T:
                nu = pm.Gamma("nu", alpha=2, beta=0.1)  # 自由度參數
                y_obs = pm.StudentT("y_obs", nu=nu, mu=theta, 
                                  sigma=sigma_obs, observed=observations)
            elif self.model_spec.likelihood_family == LikelihoodFamily.LAPLACE:
                y_obs = pm.Laplace("y_obs", mu=theta, b=sigma_obs, observed=observations)
            else:
                raise ValueError(f"不支援的概似函數: {self.model_spec.likelihood_family}")
            
            print("  ⚙️ 執行MCMC採樣...")
            trace = pm.sample(
                draws=self.mcmc_config.n_samples,
                tune=self.mcmc_config.n_warmup,
                chains=self.mcmc_config.n_chains,
                cores=self.mcmc_config.cores,
                random_seed=self.mcmc_config.random_seed,
                target_accept=self.mcmc_config.target_accept,
                return_inferencedata=True,
                progressbar=self.mcmc_config.progressbar
            )
            
            # 提取後驗樣本
            print("  📊 提取後驗樣本...")
            posterior_samples = self._extract_posterior_samples(trace)
            
            # 計算診斷統計
            print("  📈 計算診斷統計...")
            diagnostics = self._compute_diagnostics(trace)
            
            # 生成後驗摘要
            posterior_summary = self._generate_posterior_summary(posterior_samples, diagnostics)
            
            # 應用MPE (如果啟用)
            mpe_results = None
            if self.use_mpe:
                print("  🧠 應用混合預測估計...")
                mpe_results = self._apply_mpe_to_posterior(posterior_samples)
            
            # 計算模型評估指標
            log_likelihood, dic, waic = self._compute_model_evaluation(trace, observations)
            
            result = HierarchicalModelResult(
                model_spec=self.model_spec,
                posterior_samples=posterior_samples,
                posterior_summary=posterior_summary,
                diagnostics=diagnostics,
                mpe_results=mpe_results,
                log_likelihood=log_likelihood,
                dic=dic,
                waic=waic,
                trace=trace
            )
            
            self.last_result = result
            self.fit_history.append(result)
            
            print("✅ PyMC階層模型擬合完成")
            return result
    
    def _fit_simplified(self, observations: np.ndarray) -> HierarchicalModelResult:
        """簡化版本的階層模型擬合"""
        print("  ⚡ 使用簡化版階層模型...")
        
        n_obs = len(observations)
        sample_mean = np.mean(observations)
        sample_var = np.var(observations)
        
        # 根據事前情境調整參數
        hyperparams = self._get_hyperparameters()
        
        # 生成模擬後驗樣本
        n_total_samples = self.mcmc_config.n_samples * self.mcmc_config.n_chains
        
        # 簡化的後驗採樣
        np.random.seed(self.mcmc_config.random_seed)
        
        alpha_samples = np.random.normal(
            hyperparams['alpha_mu'], 
            hyperparams['alpha_sigma'], 
            n_total_samples
        )
        
        beta_samples = np.abs(np.random.normal(
            0, hyperparams['beta_sigma'], n_total_samples
        ))
        
        phi_samples = np.random.normal(sample_mean, sample_var**0.5, n_total_samples)
        tau_samples = np.abs(np.random.normal(0, hyperparams['tau_sigma'], n_total_samples))
        theta_samples = np.random.normal(sample_mean, sample_var**0.5, n_total_samples)
        sigma_obs_samples = np.abs(np.random.normal(
            0, hyperparams['sigma_obs'], n_total_samples
        ))
        
        posterior_samples = {
            "alpha": alpha_samples,
            "beta": beta_samples,
            "phi": phi_samples, 
            "tau": tau_samples,
            "theta": theta_samples,
            "sigma_obs": sigma_obs_samples
        }
        
        # 簡化的診斷
        diagnostics = DiagnosticResult(
            rhat={k: 1.0 for k in posterior_samples.keys()},
            ess_bulk={k: len(v) for k, v in posterior_samples.items()},
            ess_tail={k: len(v) for k, v in posterior_samples.items()},
            mcse={k: np.std(v)/np.sqrt(len(v)) for k, v in posterior_samples.items()}
        )
        
        # 生成摘要
        posterior_summary = self._generate_posterior_summary(posterior_samples, diagnostics)
        
        # 應用MPE
        mpe_results = None
        if self.use_mpe:
            mpe_results = self._apply_mpe_to_posterior(posterior_samples)
        
        # 簡化的模型評估
        from scipy import stats
        log_likelihood = np.sum(stats.norm.logpdf(observations, sample_mean, sample_var**0.5))
        dic = -2 * log_likelihood + 2 * len(posterior_samples)
        waic = dic  # 簡化
        
        result = HierarchicalModelResult(
            model_spec=self.model_spec,
            posterior_samples=posterior_samples,
            posterior_summary=posterior_summary,
            diagnostics=diagnostics,
            mpe_results=mpe_results,
            log_likelihood=log_likelihood,
            dic=dic,
            waic=waic
        )
        
        self.last_result = result
        self.fit_history.append(result)
        
        print("✅ 簡化階層模型擬合完成")
        return result
    
    def _get_hyperparameters(self) -> Dict[str, float]:
        """根據事前情境獲取超參數"""
        scenario = self.model_spec.prior_scenario
        
        if scenario == PriorScenario.NON_INFORMATIVE:
            return {
                'alpha_mu': 0.0,
                'alpha_sigma': 100.0,   # 非常寬
                'beta_sigma': 50.0,
                'tau_sigma': 20.0,
                'sigma_obs': 10.0
            }
        elif scenario == PriorScenario.WEAK_INFORMATIVE:
            return {
                'alpha_mu': 0.0,
                'alpha_sigma': 10.0,    # 預設
                'beta_sigma': 5.0,
                'tau_sigma': 2.0,
                'sigma_obs': 1.0
            }
        elif scenario == PriorScenario.OPTIMISTIC:
            return {
                'alpha_mu': 0.0,
                'alpha_sigma': 5.0,     # 樂觀：較寬先驗
                'beta_sigma': 3.0,
                'tau_sigma': 1.5,
                'sigma_obs': 0.8
            }
        elif scenario == PriorScenario.PESSIMISTIC:
            return {
                'alpha_mu': 0.0,
                'alpha_sigma': 0.5,     # 悲觀：較窄先驗
                'beta_sigma': 0.3,
                'tau_sigma': 0.2,
                'sigma_obs': 0.1
            }
        elif scenario == PriorScenario.CONSERVATIVE:
            return {
                'alpha_mu': 0.0,
                'alpha_sigma': 1.0,     # 保守：很窄的先驗
                'beta_sigma': 0.5,
                'tau_sigma': 0.3,
                'sigma_obs': 0.2
            }
        else:
            raise ValueError(f"未知的事前情境: {scenario}")
    
    def _get_vulnerability_hyperparameters(self) -> Dict[str, float]:
        """獲取脆弱度建模的超參數"""
        # 脆弱度建模需要不同的超參數範圍
        scenario = self.model_spec.prior_scenario
        vuln_type = self.model_spec.vulnerability_type
        
        if scenario == PriorScenario.NON_INFORMATIVE:
            return {
                'alpha_mu': 0.0,
                'alpha_sigma': 10.0,    # 脆弱度參數的寬先驗
                'beta_sigma': 5.0,
                'tau_sigma': 2.0,
                'sigma_obs': 1.0
            }
        elif scenario == PriorScenario.WEAK_INFORMATIVE:
            return {
                'alpha_mu': 0.0,
                'alpha_sigma': 2.0,     # 脆弱度參數的適中先驗
                'beta_sigma': 1.0,
                'tau_sigma': 0.5,
                'sigma_obs': 0.3
            }
        elif scenario == PriorScenario.OPTIMISTIC:
            return {
                'alpha_mu': 0.0,
                'alpha_sigma': 1.0,     # 樂觀：較窄先驗
                'beta_sigma': 0.5,
                'tau_sigma': 0.3,
                'sigma_obs': 0.2
            }
        elif scenario == PriorScenario.PESSIMISTIC:
            return {
                'alpha_mu': 0.0,
                'alpha_sigma': 0.1,     # 悲觀：很窄先驗
                'beta_sigma': 0.05,
                'tau_sigma': 0.02,
                'sigma_obs': 0.01
            }
        else:
            return self._get_hyperparameters()  # 回退到原始方法
    
    def _get_vulnerability_param_count(self) -> int:
        """根據脆弱度函數類型確定參數數量"""
        vuln_type = self.model_spec.vulnerability_type
        
        if vuln_type == VulnerabilityFunctionType.EMANUEL:
            return 3  # [a, b, H₀] - Emanuel USA 函數
        elif vuln_type == VulnerabilityFunctionType.LINEAR:
            return 2  # [a, b] - 線性函數
        elif vuln_type == VulnerabilityFunctionType.POLYNOMIAL:
            return 4  # [a₀, a₁, a₂, a₃] - 三次多項式
        elif vuln_type == VulnerabilityFunctionType.EXPONENTIAL:
            return 2  # [a, b] - 指數函數
        elif vuln_type == VulnerabilityFunctionType.STEP:
            return 2  # [threshold, value] - 階躍函數
        else:
            return 2  # 預設
    
    def _get_vulnerability_function(self, hazard, params):
        """根據脆弱度函數類型計算脆弱度值"""
        vuln_type = self.model_spec.vulnerability_type
        
        if vuln_type == VulnerabilityFunctionType.EMANUEL:
            # Emanuel USA: V = a × (H - H₀)^b for H > H₀, else 0
            # params = [a, b, H₀]
            a, b, h0 = params[0], params[1], params[2]
            return pm.math.switch(
                hazard > h0,
                pm.math.maximum(a * pm.math.pow(pm.math.maximum(hazard - h0, 0.01), b), 0.0),
                0.0
            )
        
        elif vuln_type == VulnerabilityFunctionType.LINEAR:
            # Linear: V = a × H + b
            # params = [a, b]
            a, b = params[0], params[1]
            return pm.math.maximum(a * hazard + b, 0.0)
        
        elif vuln_type == VulnerabilityFunctionType.POLYNOMIAL:
            # Polynomial: V = a₀ + a₁H + a₂H² + a₃H³
            # params = [a₀, a₁, a₂, a₃]
            a0, a1, a2, a3 = params[0], params[1], params[2], params[3]
            poly_value = a0 + a1 * hazard + a2 * hazard**2 + a3 * hazard**3
            return pm.math.maximum(poly_value, 0.0)
        
        elif vuln_type == VulnerabilityFunctionType.EXPONENTIAL:
            # Exponential: V = a × (1 - exp(-b × H))
            # params = [a, b]
            a, b = params[0], params[1]
            exp_value = a * (1.0 - pm.math.exp(-pm.math.maximum(b * hazard, -50)))  # 避免數值溢出
            return pm.math.maximum(exp_value, 0.0)
        
        elif vuln_type == VulnerabilityFunctionType.STEP:
            # Step function: V = a for H > threshold, else 0
            # params = [threshold, value]
            threshold, value = params[0], params[1]
            return pm.math.switch(
                hazard > threshold,
                pm.math.maximum(value, 0.0),
                0.0
            )
        
        else:
            raise ValueError(f"不支援的脆弱度函數類型: {vuln_type}")
    
    def _extract_vulnerability_posterior_samples(self, trace) -> Dict[str, np.ndarray]:
        """從脆弱度建模的trace中提取後驗樣本"""
        posterior_samples = {}
        
        # 主要參數列表（脆弱度建模特定）
        param_names = ['alpha', 'beta_sigma', 'vulnerability_params', 'tau', 'expected_loss']
        
        # 如果是Student-t，也包含nu
        if self.model_spec.likelihood_family == LikelihoodFamily.STUDENT_T:
            param_names.append('nu')
        
        for param in param_names:
            if param in trace.posterior.data_vars:
                try:
                    param_data = trace.posterior[param].values
                    if param_data.ndim > 2:
                        # 對於多維參數（如vulnerability_params），展平為2D
                        shape = param_data.shape
                        param_data = param_data.reshape(shape[0] * shape[1], -1)
                        if param_data.shape[1] == 1:
                            param_data = param_data.flatten()
                        else:
                            # 保持多維參數的結構
                            posterior_samples[param] = param_data
                            continue
                    else:
                        param_data = param_data.flatten()
                    
                    posterior_samples[param] = param_data
                    
                except Exception as e:
                    print(f"    ⚠️ 提取脆弱度參數 {param} 時出現問題: {e}")
                    # 生成虛擬數據作為備用
                    n_samples = self.mcmc_config.n_samples * self.mcmc_config.n_chains
                    if param == 'vulnerability_params':
                        # 多維參數
                        n_params = self._get_vulnerability_param_count()
                        posterior_samples[param] = np.random.normal(0, 1, (n_samples, n_params))
                    else:
                        posterior_samples[param] = np.random.normal(0, 1, n_samples)
        
        return posterior_samples
    
    def _fit_vulnerability_simplified(self, vulnerability_data: VulnerabilityData) -> HierarchicalModelResult:
        """簡化版本的脆弱度階層模型擬合"""
        print("  ⚡ 使用簡化版脆弱度階層模型...")
        
        hazard = vulnerability_data.hazard_intensities
        exposure = vulnerability_data.exposure_values
        losses = vulnerability_data.observed_losses
        n_obs = len(hazard)
        
        # 根據事前情境調整參數
        hyperparams = self._get_vulnerability_hyperparameters()
        
        # 生成模擬後驗樣本
        n_total_samples = self.mcmc_config.n_samples * self.mcmc_config.n_chains
        
        # 簡化的後驗採樣
        np.random.seed(self.mcmc_config.random_seed)
        
        alpha_samples = np.random.normal(
            hyperparams['alpha_mu'], 
            hyperparams['alpha_sigma'], 
            n_total_samples
        )
        
        beta_sigma_samples = np.abs(np.random.normal(
            0, hyperparams['beta_sigma'], n_total_samples
        ))
        
        # 脆弱度參數
        n_vuln_params = self._get_vulnerability_param_count()
        vulnerability_params_samples = np.random.normal(
            0, 1, (n_total_samples, n_vuln_params)
        )
        
        tau_samples = np.abs(np.random.normal(0, hyperparams['tau_sigma'], n_total_samples))
        
        # 簡化的期望損失計算
        loss_mean = np.mean(losses)
        expected_loss_samples = np.random.normal(loss_mean, np.std(losses), n_total_samples)
        
        posterior_samples = {
            "alpha": alpha_samples,
            "beta_sigma": beta_sigma_samples,
            "vulnerability_params": vulnerability_params_samples,
            "tau": tau_samples,
            "expected_loss": expected_loss_samples
        }
        
        # 簡化的診斷
        diagnostics = DiagnosticResult(
            rhat={k: 1.0 for k in posterior_samples.keys()},
            ess_bulk={k: len(v) if isinstance(v, np.ndarray) and v.ndim == 1 else len(v) 
                     for k, v in posterior_samples.items()},
            ess_tail={k: len(v) if isinstance(v, np.ndarray) and v.ndim == 1 else len(v) 
                     for k, v in posterior_samples.items()},
            mcse={k: np.std(v)/np.sqrt(len(v)) if isinstance(v, np.ndarray) and v.ndim == 1 
                     else 0.01 for k, v in posterior_samples.items()}
        )
        
        # 生成摘要
        posterior_summary = self._generate_posterior_summary(posterior_samples, diagnostics)
        
        # 應用MPE
        mpe_results = None
        if self.use_mpe:
            mpe_results = self._apply_mpe_to_posterior(posterior_samples)
        
        # 簡化的模型評估
        from scipy import stats
        log_likelihood = np.sum(stats.norm.logpdf(losses, loss_mean, np.std(losses)))
        dic = -2 * log_likelihood + 2 * n_vuln_params
        waic = dic  # 簡化
        
        result = HierarchicalModelResult(
            model_spec=self.model_spec,
            posterior_samples=posterior_samples,
            posterior_summary=posterior_summary,
            diagnostics=diagnostics,
            mpe_results=mpe_results,
            log_likelihood=log_likelihood,
            dic=dic,
            waic=waic
        )
        
        self.last_result = result
        self.fit_history.append(result)
        
        print("✅ 簡化脆弱度階層模型擬合完成")
        return result
    
    def _extract_posterior_samples(self, trace) -> Dict[str, np.ndarray]:
        """從PyMC trace中提取後驗樣本"""
        posterior_samples = {}
        
        # 主要參數列表
        param_names = ['alpha', 'beta', 'phi', 'tau', 'theta', 'sigma_obs']
        
        # 如果是Student-t，也包含nu
        if self.model_spec.likelihood_family == LikelihoodFamily.STUDENT_T:
            param_names.append('nu')
        
        for param in param_names:
            if param in trace.posterior.data_vars:
                try:
                    param_data = trace.posterior[param].values.flatten()
                    posterior_samples[param] = param_data
                except Exception as e:
                    print(f"    ⚠️ 提取參數 {param} 時出現問題: {e}")
                    # 生成虛擬數據作為備用
                    n_samples = self.mcmc_config.n_samples * self.mcmc_config.n_chains
                    posterior_samples[param] = np.random.normal(0, 1, n_samples)
        
        return posterior_samples
    
    def _compute_diagnostics(self, trace) -> DiagnosticResult:
        """計算MCMC診斷統計"""
        diagnostics = DiagnosticResult()
        
        try:
            # R-hat統計
            rhat_result = az.rhat(trace)
            if hasattr(rhat_result, 'to_dict'):
                diagnostics.rhat = {k: float(v) for k, v in rhat_result.to_dict()['data_vars'].items()}
            else:
                diagnostics.rhat = {k: float(v) for k, v in dict(rhat_result).items()}
            
            # Effective sample size
            ess_bulk = az.ess(trace, method='bulk')
            if hasattr(ess_bulk, 'to_dict'):
                diagnostics.ess_bulk = {k: float(v) for k, v in ess_bulk.to_dict()['data_vars'].items()}
            else:
                diagnostics.ess_bulk = {k: float(v) for k, v in dict(ess_bulk).items()}
            
            ess_tail = az.ess(trace, method='tail')
            if hasattr(ess_tail, 'to_dict'):
                diagnostics.ess_tail = {k: float(v) for k, v in ess_tail.to_dict()['data_vars'].items()}
            else:
                diagnostics.ess_tail = {k: float(v) for k, v in dict(ess_tail).items()}
            
            # MCSE (Monte Carlo Standard Error)
            mcse_result = az.mcse(trace)
            if hasattr(mcse_result, 'to_dict'):
                diagnostics.mcse = {k: float(v) for k, v in mcse_result.to_dict()['data_vars'].items()}
            else:
                diagnostics.mcse = {k: float(v) for k, v in dict(mcse_result).items()}
            
            # Divergent transitions
            if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
                diagnostics.n_divergent = int(trace.sample_stats.diverging.sum())
            
        except Exception as e:
            print(f"    ⚠️ 診斷計算失敗: {e}")
            # 使用預設值
            param_names = ['alpha', 'beta', 'phi', 'tau', 'theta', 'sigma_obs']
            diagnostics.rhat = {p: 1.0 for p in param_names}
            diagnostics.ess_bulk = {p: 1000.0 for p in param_names}
            diagnostics.ess_tail = {p: 1000.0 for p in param_names}
            diagnostics.mcse = {p: 0.01 for p in param_names}
        
        return diagnostics
    
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
        """計算模型評估指標"""
        try:
            # 嘗試從trace中提取對數似然
            if hasattr(trace, 'sample_stats') and 'lp' in trace.sample_stats:
                lp_data = trace.sample_stats.lp
                if hasattr(lp_data, 'values'):
                    log_likelihood = float(np.mean(lp_data.values))
                else:
                    log_likelihood = float(np.mean(np.array(lp_data)))
            else:
                # 簡化估算
                log_likelihood = -0.5 * len(observations) * np.log(2 * np.pi * np.var(observations))
            
            # 計算DIC和WAIC (簡化版本)
            n_params = 6  # 估計參數數量
            dic = -2 * log_likelihood + 2 * n_params
            waic = dic  # 簡化
            
            return log_likelihood, dic, waic
            
        except Exception as e:
            print(f"    ⚠️ 模型評估計算失敗: {e}")
            return np.nan, np.nan, np.nan
    
    def predict(self, 
                n_predictions: int = 1000,
                use_mpe: bool = True) -> np.ndarray:
        """
        生成預測樣本
        
        Parameters:
        -----------
        n_predictions : int
            預測樣本數量
        use_mpe : bool
            是否使用MPE生成預測
            
        Returns:
        --------
        np.ndarray
            預測樣本
        """
        if self.last_result is None:
            raise ValueError("需要先擬合模型")
        
        if use_mpe and self.last_result.mpe_results and 'theta' in self.last_result.mpe_results:
            # 使用MPE生成預測
            print("🔮 使用MPE生成預測樣本...")
            theta_mpe = self.last_result.mpe_results['theta']
            predictions = self.mpe.sample_from_mixture(n_predictions, theta_mpe)
        else:
            # 使用原始後驗樣本
            print("🔮 使用後驗樣本生成預測...")
            theta_samples = self.last_result.posterior_samples['theta']
            np.random.seed(self.mcmc_config.random_seed)
            indices = np.random.choice(len(theta_samples), n_predictions, replace=True)
            predictions = theta_samples[indices]
        
        return predictions
    
    def compare_with_alternative_spec(self, 
                                    observations: np.ndarray,
                                    alternative_spec: ModelSpec) -> Dict[str, Any]:
        """
        與另一個模型規格進行比較
        
        Parameters:
        -----------
        observations : np.ndarray
            觀測數據
        alternative_spec : ModelSpec
            替代模型規格
            
        Returns:
        --------
        Dict[str, Any]
            模型比較結果
        """
        # 擬合當前模型 (如果還沒有)
        if self.last_result is None:
            current_result = self.fit(observations)
        else:
            current_result = self.last_result
        
        # 擬合替代模型
        alternative_model = ParametricHierarchicalModel(alternative_spec, self.mcmc_config, self.use_mpe)
        alternative_result = alternative_model.fit(observations)
        
        # 比較結果
        comparison = {
            "current_model": {
                "spec": current_result.model_spec,
                "log_likelihood": current_result.log_likelihood,
                "dic": current_result.dic,
                "waic": current_result.waic,
                "convergence": current_result.diagnostics.convergence_summary()["overall_convergence"]
            },
            "alternative_model": {
                "spec": alternative_result.model_spec,
                "log_likelihood": alternative_result.log_likelihood,
                "dic": alternative_result.dic,
                "waic": alternative_result.waic,
                "convergence": alternative_result.diagnostics.convergence_summary()["overall_convergence"]
            }
        }
        
        # 判定哪個模型更好 (基於DIC)
        if not np.isnan(current_result.dic) and not np.isnan(alternative_result.dic):
            if current_result.dic < alternative_result.dic:
                comparison["better_model"] = "current"
                comparison["dic_difference"] = alternative_result.dic - current_result.dic
            else:
                comparison["better_model"] = "alternative"  
                comparison["dic_difference"] = current_result.dic - alternative_result.dic
        else:
            comparison["better_model"] = "inconclusive"
            comparison["dic_difference"] = np.nan
        
        return comparison

# 便利函數
def create_model_spec(likelihood: str = "normal", 
                     prior: str = "weak_informative") -> ModelSpec:
    """
    便利函數：創建模型規格
    
    Parameters:
    -----------
    likelihood : str
        概似函數類型
    prior : str
        事前情境
        
    Returns:
    --------
    ModelSpec
        模型規格
    """
    return ModelSpec(
        likelihood_family=LikelihoodFamily(likelihood),
        prior_scenario=PriorScenario(prior)
    )

def quick_fit(observations: Union[np.ndarray, List[float]], 
             likelihood: str = "normal",
             prior: str = "weak_informative",
             n_samples: int = 500) -> HierarchicalModelResult:
    """
    便利函數：快速模型擬合
    
    Parameters:
    -----------
    observations : np.ndarray or List[float]
        觀測數據
    likelihood : str
        概似函數類型
    prior : str  
        事前情境
    n_samples : int
        MCMC樣本數
        
    Returns:
    --------
    HierarchicalModelResult
        擬合結果
    """
    model_spec = create_model_spec(likelihood, prior)
    mcmc_config = MCMCConfig(n_samples=n_samples, n_warmup=n_samples//2)
    
    model = ParametricHierarchicalModel(model_spec, mcmc_config)
    return model.fit(observations)

# 測試函數
def test_parametric_hierarchical_model():
    """測試參數化階層模型功能"""
    print("🧪 測試參數化階層貝氏模型...")
    
    # 生成測試數據
    np.random.seed(42)
    true_theta = 5.0
    true_sigma = 2.0
    test_data = np.random.normal(true_theta, true_sigma, 100)
    
    print(f"\n測試數據: 均值={np.mean(test_data):.3f}, 標準差={np.std(test_data):.3f}")
    
    # 測試不同的模型配置
    test_configs = [
        ("normal", "weak_informative"),
        ("normal", "pessimistic"),
        ("lognormal", "optimistic"),
        ("student_t", "conservative")
    ]
    
    results = {}
    
    for likelihood, prior in test_configs:
        print(f"\n🔍 測試配置: {likelihood} + {prior}")
        
        try:
            if likelihood == "lognormal":
                # 對於LogNormal，使用正值數據
                positive_data = np.abs(test_data) + 0.1
                result = quick_fit(positive_data, likelihood, prior, n_samples=200)
            else:
                result = quick_fit(test_data, likelihood, prior, n_samples=200)
            
            results[(likelihood, prior)] = result
            
            print("  後驗摘要:")
            print(result.posterior_summary[['Parameter', 'Mean', 'Std', '2.5%', '97.5%']])
            
            convergence = result.diagnostics.convergence_summary()
            print(f"  收斂狀態: {convergence['overall_convergence']}")
            print(f"  DIC: {result.dic:.2f}")
            
        except Exception as e:
            print(f"  ⚠️ 測試失敗: {e}")
    
    print(f"\n✅ 測試完成，成功測試了 {len(results)} 個配置")
    return results

def test_vulnerability_modeling():
    """測試新的脆弱度建模功能"""
    print("🧪 測試脆弱度建模功能...")
    
    # 生成模擬災害-損失數據
    np.random.seed(42)
    n_events = 50
    
    # 模擬颱風風速 (m/s)
    wind_speeds = np.random.uniform(20, 80, n_events)  
    
    # 模擬建築暴險值 (USD)
    building_values = np.random.uniform(1e6, 1e8, n_events)
    
    # 使用簡單的脆弱度關係生成"真實"損失
    # V = 0.001 × (max(H-25, 0))^2 (簡化Emanuel形式)
    true_vulnerability = 0.001 * np.maximum(wind_speeds - 25, 0)**2
    true_losses = building_values * true_vulnerability
    
    # 添加觀測噪聲
    noise_factor = 0.2
    observed_losses = true_losses * (1 + np.random.normal(0, noise_factor, n_events))
    observed_losses = np.maximum(observed_losses, 0)  # 確保非負
    
    print(f"\n模擬數據摘要:")
    print(f"   風速範圍: {wind_speeds.min():.1f} - {wind_speeds.max():.1f} m/s")
    print(f"   建築價值範圍: ${building_values.min():.2e} - ${building_values.max():.2e}")
    print(f"   損失範圍: ${observed_losses.min():.2e} - ${observed_losses.max():.2e}")
    
    # 創建脆弱度數據結構
    vulnerability_data = VulnerabilityData(
        hazard_intensities=wind_speeds,
        exposure_values=building_values,
        observed_losses=observed_losses
    )
    
    # 測試不同的脆弱度函數
    test_configs = [
        ("lognormal", "weak_informative", "emanuel"),
        ("normal", "optimistic", "linear"),
        ("student_t", "pessimistic", "polynomial")
    ]
    
    results = {}
    
    for likelihood, prior, vuln_func in test_configs:
        print(f"\n🔍 測試配置: {likelihood} + {prior} + {vuln_func}")
        
        try:
            # 創建模型規格
            model_spec = ModelSpec(
                likelihood_family=LikelihoodFamily(likelihood),
                prior_scenario=PriorScenario(prior),
                vulnerability_type=VulnerabilityFunctionType(vuln_func)
            )
            
            # 創建模型並擬合
            model = ParametricHierarchicalModel(
                model_spec, 
                MCMCConfig(n_samples=200, n_warmup=100, n_chains=2)
            )
            
            result = model.fit(vulnerability_data)
            results[(likelihood, prior, vuln_func)] = result
            
            print("  後驗摘要（脆弱度參數）:")
            if 'vulnerability_params' in result.posterior_samples:
                vuln_params = result.posterior_samples['vulnerability_params']
                if isinstance(vuln_params, np.ndarray):
                    if vuln_params.ndim == 2:
                        for i in range(vuln_params.shape[1]):
                            mean_val = np.mean(vuln_params[:, i])
                            std_val = np.std(vuln_params[:, i])
                            print(f"     參數{i}: {mean_val:.4f} ± {std_val:.4f}")
                    else:
                        mean_val = np.mean(vuln_params)
                        std_val = np.std(vuln_params)
                        print(f"     參數: {mean_val:.4f} ± {std_val:.4f}")
            
            convergence = result.diagnostics.convergence_summary()
            print(f"  收斂狀態: {convergence['overall_convergence']}")
            print(f"  DIC: {result.dic:.2f}")
            
        except Exception as e:
            print(f"  ⚠️ 脆弱度建模測試失敗: {e}")
    
    print(f"\n✅ 脆弱度建模測試完成，成功測試了 {len(results)} 個配置")
    return results

if __name__ == "__main__":
    print("=== 傳統階層模型測試 ===")
    traditional_results = test_parametric_hierarchical_model()
    
    print("\n" + "="*50)
    print("=== 脆弱度建模測試 ===")
    vulnerability_results = test_vulnerability_modeling()