#!/usr/bin/env python3
"""
Likelihood Families Module
似然函數族模組

從 parametric_bayesian_hierarchy.py 拆分出的似然函數和相關配置
包含MCMC配置、診斷結果和模型結果結構

核心功能:
- MCMC配置 (MCMCConfig)
- 診斷結果 (DiagnosticResult)
- 階層模型結果 (HierarchicalModelResult)
- 似然函數建構工具

Author: Research Team  
Date: 2025-01-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

# ========================================
# 配置類別
# ========================================

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
    model_spec: 'ModelSpec'
    posterior_samples: Dict[str, np.ndarray]
    posterior_summary: pd.DataFrame
    diagnostics: DiagnosticResult
    mpe_results: Optional[Dict[str, Any]] = None
    log_likelihood: float = np.nan
    dic: float = np.nan
    waic: float = np.nan
    trace: Any = None  # PyMC trace object
    
    def get_parameter_credible_interval(self, 
                                      param_name: str, 
                                      alpha: float = 0.05) -> Tuple[float, float]:
        """獲取參數的可信區間"""
        if param_name not in self.posterior_samples:
            raise ValueError(f"參數 '{param_name}' 不存在於後驗樣本中")
        
        samples = self.posterior_samples[param_name]
        if isinstance(samples, np.ndarray) and samples.ndim == 1:
            lower = np.percentile(samples, 100 * alpha / 2)
            upper = np.percentile(samples, 100 * (1 - alpha / 2))
            return lower, upper
        else:
            raise ValueError(f"參數 '{param_name}' 的樣本格式不正確")
    
    def get_parameter_summary(self, param_name: str) -> Dict[str, float]:
        """獲取參數的摘要統計"""
        if param_name not in self.posterior_samples:
            raise ValueError(f"參數 '{param_name}' 不存在於後驗樣本中")
        
        samples = self.posterior_samples[param_name]
        if isinstance(samples, np.ndarray) and samples.ndim == 1:
            return {
                "mean": np.mean(samples),
                "std": np.std(samples),
                "median": np.median(samples),
                "q025": np.percentile(samples, 2.5),
                "q975": np.percentile(samples, 97.5),
                "rhat": self.diagnostics.rhat.get(param_name, np.nan),
                "ess_bulk": self.diagnostics.ess_bulk.get(param_name, np.nan)
            }
        else:
            raise ValueError(f"參數 '{param_name}' 的樣本格式不正確")
    
    def model_comparison_metrics(self) -> Dict[str, float]:
        """模型比較指標"""
        return {
            "log_likelihood": self.log_likelihood,
            "dic": self.dic,
            "waic": self.waic,
            "n_parameters": len(self.posterior_samples),
            "n_observations": len(self.posterior_samples.get('observed_loss', [0]))
        }

# ========================================
# 似然函數建構工具
# ========================================

class LikelihoodBuilder:
    """似然函數建構器"""
    
    @staticmethod
    def build_normal_likelihood(mu, sigma, observed_data, name="likelihood"):
        """建構正態似然 (JAX版本)"""
        try:
            import jax.numpy as jnp
            import jax.scipy.stats as jsp
            # Return JAX log-pdf function instead of PyMC distribution
            def normal_logpdf():
                return jsp.norm.logpdf(jnp.array(observed_data), loc=mu, scale=sigma).sum()
            return normal_logpdf
        except ImportError:
            raise ImportError("需要JAX來建構似然函數")
    
    @staticmethod
    def build_lognormal_likelihood(mu, sigma, observed_data, name="likelihood"):
        """建構對數正態似然 (JAX版本)"""
        try:
            import jax.numpy as jnp
            import jax.scipy.stats as jsp
            
            def lognormal_logpdf():
                # 確保mu > 0 for lognormal
                mu_positive = jnp.maximum(mu, 1e-6)
                log_mu = jnp.log(mu_positive)
                return jsp.lognorm.logpdf(jnp.array(observed_data), s=sigma, scale=jnp.exp(log_mu)).sum()
            return lognormal_logpdf
        except ImportError:
            raise ImportError("需要JAX來建構似然函數")
    
    @staticmethod
    def build_student_t_likelihood(nu, mu, sigma, observed_data, name="likelihood"):
        """建構Student-t似然 (JAX版本)"""
        try:
            import jax.numpy as jnp
            import jax.scipy.stats as jsp
            
            def student_t_logpdf():
                return jsp.t.logpdf(jnp.array(observed_data), df=nu, loc=mu, scale=sigma).sum()
            return student_t_logpdf
        except ImportError:
            raise ImportError("需要JAX來建構似然函數")
    
    @staticmethod
    def build_gamma_likelihood(alpha, beta, observed_data, name="likelihood"):
        """建構Gamma似然 (JAX版本)"""
        try:
            import jax.numpy as jnp
            import jax.scipy.stats as jsp
            
            def gamma_logpdf():
                return jsp.gamma.logpdf(jnp.array(observed_data), a=alpha, scale=1/beta).sum()
            return gamma_logpdf
        except ImportError:
            raise ImportError("需要JAX來建構似然函數")
    
    @staticmethod
    def build_laplace_likelihood(mu, b, observed_data, name="likelihood"):
        """建構Laplace似然 (JAX版本)"""
        try:
            import jax.numpy as jnp
            
            def laplace_logpdf():
                return jnp.sum(-jnp.log(2 * b) - jnp.abs(jnp.array(observed_data) - mu) / b)
            return laplace_logpdf
        except ImportError:
            raise ImportError("需要JAX來建構似然函數")

class ContaminationMixture:
    """ε-contamination混合分布建構器"""
    
    @staticmethod
    def build_epsilon_contamination(base_likelihood, contamination_likelihood, 
                                  epsilon, observed_data, name="epsilon_mixture"):
        """
        建構ε-contamination混合似然
        
        f(y) = (1-ε)f₀(y|θ) + ε*q(y)
        
        Parameters:
        -----------
        base_likelihood : PyMC distribution
            基礎分布 f₀(y|θ)
        contamination_likelihood : PyMC distribution  
            污染分布 q(y)
        epsilon : float or PyMC random variable
            混合權重
        observed_data : array_like
            觀測數據
        name : str
            分布名稱
        """
        try:
            import jax.numpy as jnp
            from jax.scipy.special import logsumexp
            
            def epsilon_contamination_logpdf():
                # 計算對數似然（假設likelihood functions返回log-pdf值）
                base_logp = base_likelihood() if callable(base_likelihood) else base_likelihood
                contamination_logp = contamination_likelihood() if callable(contamination_likelihood) else contamination_likelihood
                
                # 混合對數似然 using log-sum-exp trick
                base_log_weight = jnp.log(1 - epsilon) + base_logp
                contamination_log_weight = jnp.log(epsilon) + contamination_logp
                
                mixture_logp = logsumexp(
                    jnp.stack([base_log_weight, contamination_log_weight]), 
                    axis=0
                )
                
                return jnp.sum(mixture_logp)
            
            return epsilon_contamination_logpdf
            
        except ImportError:
            raise ImportError("需要JAX來建構混合似然函數")

class GPDLikelihood:
    """Generalized Pareto Distribution 似然"""
    
    @staticmethod
    def build_gpd_likelihood(mu, sigma, xi, threshold, observed_data, name="gpd_likelihood"):
        """
        建構GPD似然（超過閾值的數據）
        
        Parameters:
        -----------
        mu : float or PyMC variable
            位置參數（閾值）
        sigma : float or PyMC variable
            尺度參數
        xi : float or PyMC variable
            形狀參數
        threshold : float
            閾值
        observed_data : array_like
            觀測數據
        name : str
            分布名稱
        """
        try:
            import jax.numpy as jnp
            import numpy as np
            
            def gpd_logpdf():
                # 過濾超過閾值的數據
                observed_array = jnp.array(observed_data)
                exceedances = observed_array[observed_array > threshold] - threshold
                
                if len(exceedances) == 0:
                    print(f"⚠️ 沒有數據超過閾值 {threshold}")
                    return jnp.array(0.0)
                
                # GPD log-pdf
                # log p(y) = -log(sigma) - (1 + 1/xi) * log(1 + xi * y / sigma)
                # for y > 0, sigma > 0, and 1 + xi * y / sigma > 0
                
                y_scaled = exceedances / sigma
                
                # 確保 1 + xi * y_scaled > 0
                condition = 1 + xi * y_scaled
                
                # GPD log probability
                logp = (-jnp.log(sigma) - 
                       (1 + 1/xi) * jnp.log(condition))
                
                # 只有當條件滿足時才計算
                valid_logp = jnp.where(condition > 0, logp, -jnp.inf)
                
                return jnp.sum(valid_logp)
            
            return gpd_logpdf
            
        except ImportError:
            raise ImportError("需要JAX來建構GPD似然函數")

# ========================================
# 脆弱度函數建構器
# ========================================

class VulnerabilityFunctionBuilder:
    """脆弱度函數建構器"""
    
    @staticmethod
    def build_emanuel_function(hazard_intensities, threshold=25.0):
        """
        建構Emanuel USA脆弱度函數
        
        V = min(1, a * max(H-threshold, 0)^b)
        
        Parameters:
        -----------
        hazard_intensities : array_like
            災害強度（風速）
        threshold : float
            閾值風速
        """
        try:
            import jax.numpy as jnp
            
            def emanuel_vulnerability(a, b):
                """Emanuel脆弱度函數"""
                wind_excess = jnp.maximum(jnp.array(hazard_intensities) - threshold, 0)
                return jnp.minimum(1.0, a * wind_excess**b)
            
            # 返回函數和參數映射
            return emanuel_vulnerability, {"param_names": ["vulnerability_a", "vulnerability_b"]}
            
        except ImportError:
            raise ImportError("需要JAX來建構脆弱度函數")
    
    @staticmethod 
    def build_linear_function(hazard_intensities):
        """
        建構線性脆弱度函數
        
        V = max(0, a * H + b)
        
        Parameters:
        -----------
        hazard_intensities : array_like
            災害強度
        """
        try:
            import jax.numpy as jnp
            
            def linear_vulnerability(a, b):
                """線性脆弱度函數"""
                return jnp.maximum(0, a * jnp.array(hazard_intensities) + b)
            
            # 返回函數和參數映射
            return linear_vulnerability, {"param_names": ["vulnerability_a", "vulnerability_b"]}
            
        except ImportError:
            raise ImportError("需要JAX來建構脆弱度函數")
    
    @staticmethod
    def build_polynomial_function(hazard_intensities, degree=2):
        """
        建構多項式脆弱度函數
        
        V = max(0, a * H^2 + b * H + c) for degree=2
        
        Parameters:
        -----------
        hazard_intensities : array_like
            災害強度
        degree : int
            多項式次數
        """
        try:
            import jax.numpy as jnp
            
            if degree == 2:
                def polynomial_vulnerability(a, b, c):
                    """二次多項式脆弱度函數"""
                    hazard_array = jnp.array(hazard_intensities)
                    return jnp.maximum(0, 
                        a * hazard_array**2 + 
                        b * hazard_array + c)
                
                return polynomial_vulnerability, {"param_names": ["vulnerability_a", "vulnerability_b", "vulnerability_c"]}
            
            else:
                raise NotImplementedError(f"多項式次數 {degree} 尚未實現")
                
        except ImportError:
            raise ImportError("需要JAX來建構脆弱度函數")

# ========================================
# 工具函數
# ========================================

def check_convergence(diagnostics: DiagnosticResult, 
                     strict: bool = True) -> Tuple[bool, List[str]]:
    """
    檢查MCMC收斂性
    
    Parameters:
    -----------
    diagnostics : DiagnosticResult
        診斷結果
    strict : bool
        是否使用嚴格標準
        
    Returns:
    --------
    Tuple[bool, List[str]]
        (是否收斂, 警告訊息列表)
    """
    warnings = []
    
    # R-hat檢查
    rhat_threshold = 1.01 if strict else 1.1
    bad_rhat = [k for k, v in diagnostics.rhat.items() if v > rhat_threshold]
    if bad_rhat:
        warnings.append(f"R-hat > {rhat_threshold}: {bad_rhat}")
    
    # ESS檢查
    ess_threshold = 400 if not strict else 1000
    bad_ess = [k for k, v in diagnostics.ess_bulk.items() if v < ess_threshold]
    if bad_ess:
        warnings.append(f"ESS < {ess_threshold}: {bad_ess}")
    
    # Divergent transitions
    if diagnostics.n_divergent > 0:
        warnings.append(f"發現 {diagnostics.n_divergent} 個divergent transitions")
    
    # Energy error
    if diagnostics.energy_error:
        warnings.append("發現energy transition問題")
    
    converged = len(warnings) == 0
    return converged, warnings

def recommend_mcmc_adjustments(diagnostics: DiagnosticResult) -> Dict[str, Any]:
    """
    根據診斷結果推薦MCMC調整
    
    Parameters:
    -----------
    diagnostics : DiagnosticResult
        診斷結果
        
    Returns:
    --------
    Dict[str, Any]
        推薦的調整配置
    """
    recommendations = {}
    
    # 檢查R-hat
    max_rhat = max(diagnostics.rhat.values()) if diagnostics.rhat else 1.0
    if max_rhat > 1.1:
        recommendations["n_warmup"] = "增加到 1000-2000"
        recommendations["n_samples"] = "增加到 2000-5000"
    
    # 檢查ESS
    min_ess = min(diagnostics.ess_bulk.values()) if diagnostics.ess_bulk else 1000
    if min_ess < 400:
        recommendations["n_samples"] = "增加樣本數量"
        recommendations["thinning"] = "考慮使用thinning"
    
    # 檢查divergent transitions
    if diagnostics.n_divergent > 0:
        recommendations["target_accept"] = "增加到 0.9-0.95"
        recommendations["step_size"] = "使用更小的step size"
    
    # 檢查energy error
    if diagnostics.energy_error:
        recommendations["target_accept"] = "增加target_accept"
        recommendations["model_reparameterization"] = "考慮重新參數化模型"
    
    return recommendations

def test_likelihood_families():
    """測試似然函數族功能"""
    print("🧪 測試似然函數族...")
    
    # 測試配置類別
    print("✅ 測試MCMC配置:")
    config = MCMCConfig(n_samples=2000, n_chains=4)
    print(f"   樣本數: {config.n_samples}, 鏈數: {config.n_chains}")
    
    # 測試診斷結果
    print("✅ 測試診斷結果:")
    diagnostics = DiagnosticResult()
    diagnostics.rhat = {"alpha": 1.05, "beta": 1.02}
    diagnostics.ess_bulk = {"alpha": 800, "beta": 1200}
    
    summary = diagnostics.convergence_summary()
    print(f"   收斂性: {summary['overall_convergence']}")
    
    # 測試收斂性檢查
    print("✅ 測試收斂性檢查:")
    converged, warnings = check_convergence(diagnostics)
    print(f"   收斂: {converged}, 警告: {len(warnings)}")
    
    print("✅ 似然函數族測試完成")

if __name__ == "__main__":
    test_likelihood_families()