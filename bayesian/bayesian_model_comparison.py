"""
Bayesian Model Comparison Framework (方法一)
貝葉斯模型比較框架

Implements the two-stage approach from bayesian_implement.md:
Stage 1: Fit multiple candidate models
Stage 2: Evaluate using CRPS and select the best model

This module provides three different model structures:
- Model A: Simple Log-Normal baseline
- Model B: Hierarchical Bayesian model  
- Model C: Alternative model with different predictors
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PyMC for Bayesian modeling
try:
    import pymc as pm
    import pytensor.tensor as pt
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    warnings.warn("PyMC not available, using simplified models")

# Import skill scores
try:
    from skill_scores import (
        calculate_crps, calculate_tss, calculate_edi
    )
    HAS_SKILL_SCORES = True
except ImportError:
    HAS_SKILL_SCORES = False
    warnings.warn("skill_scores module not available")

@dataclass
class ModelComparisonResult:
    """模型比較結果"""
    model_name: str
    model_type: str
    trace: Any  # PyMC trace object
    posterior_predictive: np.ndarray
    crps_score: float
    tss_score: float
    edi_score: float
    log_likelihood: float
    convergence_diagnostics: Dict[str, Any]

class BayesianModelComparison:
    """
    貝葉斯模型比較框架
    
    實現方法一：建立多個候選模型並用 Skill Scores 評估
    """
    
    def __init__(self,
                 n_samples: int = 2000,
                 n_chains: int = 4,
                 random_seed: int = 42):
        """
        初始化模型比較框架
        
        Parameters:
        -----------
        n_samples : int
            MCMC 採樣數
        n_chains : int
            MCMC 鏈數
        random_seed : int
            隨機種子
        """
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.random_seed = random_seed
        
        # 存儲結果
        self.models = {}
        self.traces = {}
        self.posterior_predictives = {}
        self.comparison_results = []
        
    def build_model_A_simple_lognormal(self, 
                                       observations: np.ndarray,
                                       covariates: Optional[np.ndarray] = None) -> Any:
        """
        模型 A: 簡單的對數正態分佈基準模型
        
        這是最基礎的模型，假設損失遵循對數正態分佈
        
        Parameters:
        -----------
        observations : np.ndarray
            觀測損失數據
        covariates : np.ndarray, optional
            協變量（如風速、降雨等）
            
        Returns:
        --------
        PyMC model object
        """
        
        if not HAS_PYMC:
            warnings.warn("PyMC not available, returning None")
            return None
            
        print("📊 建立模型 A: 簡單對數正態基準模型")
        
        with pm.Model() as model_A:
            # 數據轉換 - 避免零值
            obs_positive = np.maximum(observations, 1e-6)
            log_obs = np.log(obs_positive)
            
            # 簡單的先驗
            mu = pm.Normal('mu', mu=np.mean(log_obs), sigma=2)
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # 如果有協變量，加入簡單的線性關係
            if covariates is not None:
                beta = pm.Normal('beta', mu=0, sigma=1, shape=covariates.shape[1])
                mu_obs = mu + pm.math.dot(covariates, beta)
            else:
                mu_obs = mu
            
            # Likelihood - 對數正態分佈
            y_obs = pm.LogNormal('y_obs', mu=mu_obs, sigma=sigma, observed=observations)
            
        self.models['A_simple_lognormal'] = model_A
        return model_A
    
    def build_model_B_hierarchical(self,
                                   observations: np.ndarray,
                                   groups: Optional[np.ndarray] = None,
                                   covariates: Optional[np.ndarray] = None) -> Any:
        """
        模型 B: 階層貝葉斯模型（改進版）
        
        包含4層階層結構，處理群組效應
        
        Parameters:
        -----------
        observations : np.ndarray
            觀測損失數據
        groups : np.ndarray, optional
            群組標籤（如地區、事件類型）
        covariates : np.ndarray, optional
            協變量
            
        Returns:
        --------
        PyMC model object
        """
        
        if not HAS_PYMC:
            warnings.warn("PyMC not available, returning None")
            return None
            
        print("📊 建立模型 B: 階層貝葉斯模型")
        
        with pm.Model() as model_B:
            # 數據準備
            obs_positive = np.maximum(observations, 1e-6)
            log_obs = np.log(obs_positive)
            
            # Level 4: Hyperpriors (超參數)
            mu_alpha = pm.Normal('mu_alpha', mu=np.mean(log_obs), sigma=3)
            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
            
            # Level 3: Group-level parameters (群組參數)
            if groups is not None:
                n_groups = len(np.unique(groups))
                alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=n_groups)
                
                # Map groups to alpha values
                group_idx = pm.ConstantData('group_idx', groups)
                mu_group = alpha[group_idx]
            else:
                alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha)
                mu_group = alpha
            
            # Level 2: Individual-level parameters (個體參數)
            if covariates is not None:
                beta = pm.Normal('beta', mu=0, sigma=1, shape=covariates.shape[1])
                mu_individual = mu_group + pm.math.dot(covariates, beta)
            else:
                mu_individual = mu_group
            
            # Level 1: Observation model (觀測模型)
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
            
            # 使用 Gamma 分佈作為 likelihood (更適合損失數據)
            # 轉換參數到 Gamma 分佈的 alpha 和 beta
            mu_exp = pm.math.exp(mu_individual)
            alpha_gamma = mu_exp**2 / sigma_obs**2
            beta_gamma = mu_exp / sigma_obs**2
            
            y_obs = pm.Gamma('y_obs', alpha=alpha_gamma, beta=beta_gamma, observed=observations)
            
        self.models['B_hierarchical'] = model_B
        return model_B
    
    def build_model_C_alternative(self,
                                  observations: np.ndarray,
                                  wind_speed: Optional[np.ndarray] = None,
                                  rainfall: Optional[np.ndarray] = None,
                                  storm_surge: Optional[np.ndarray] = None) -> Any:
        """
        模型 C: 包含不同預測變數的替代模型
        
        使用特定的氣象變數作為預測因子
        
        Parameters:
        -----------
        observations : np.ndarray
            觀測損失數據
        wind_speed : np.ndarray, optional
            風速數據
        rainfall : np.ndarray, optional
            降雨數據
        storm_surge : np.ndarray, optional
            風暴潮數據
            
        Returns:
        --------
        PyMC model object
        """
        
        if not HAS_PYMC:
            warnings.warn("PyMC not available, returning None")
            return None
            
        print("📊 建立模型 C: 替代預測變數模型")
        
        with pm.Model() as model_C:
            # 數據準備
            obs_positive = np.maximum(observations, 1e-6)
            
            # 基礎截距
            intercept = pm.Normal('intercept', mu=np.log(np.mean(obs_positive)), sigma=2)
            
            # 預測變數效應
            mu = intercept
            
            if wind_speed is not None:
                # 風速的非線性效應 (平方項)
                beta_wind = pm.Normal('beta_wind', mu=0.1, sigma=0.05)
                beta_wind_sq = pm.Normal('beta_wind_sq', mu=0.01, sigma=0.005)
                wind_normalized = (wind_speed - np.mean(wind_speed)) / np.std(wind_speed)
                mu = mu + beta_wind * wind_normalized + beta_wind_sq * wind_normalized**2
            
            if rainfall is not None:
                # 降雨的對數效應
                beta_rain = pm.Normal('beta_rain', mu=0.05, sigma=0.02)
                rain_log = np.log(rainfall + 1)  # 加1避免log(0)
                rain_normalized = (rain_log - np.mean(rain_log)) / np.std(rain_log)
                mu = mu + beta_rain * rain_normalized
            
            if storm_surge is not None:
                # 風暴潮的閾值效應
                beta_surge = pm.Normal('beta_surge', mu=0.2, sigma=0.1)
                surge_threshold = pm.Normal('surge_threshold', mu=2, sigma=0.5)
                surge_effect = pm.math.switch(storm_surge > surge_threshold, 
                                             beta_surge * (storm_surge - surge_threshold), 
                                             0)
                mu = mu + surge_effect
            
            # 使用 Tweedie 分佈 (適合包含零的損失數據)
            # 簡化為 Gamma + 零膨脹
            p_zero = pm.Beta('p_zero', alpha=1, beta=9)  # 約10%零損失的先驗
            
            # 非零損失的 Gamma 分佈
            mu_positive = pm.math.exp(mu)
            dispersion = pm.HalfNormal('dispersion', sigma=1)
            
            # Zero-inflated Gamma
            # 這裡簡化處理，實際應使用 Mixture 或 ZeroInflatedGamma
            y_obs = pm.Gamma('y_obs', 
                            alpha=mu_positive/dispersion, 
                            beta=1/dispersion,
                            observed=observations)
            
        self.models['C_alternative'] = model_C
        return model_C
    
    def fit_all_models(self,
                       train_data: np.ndarray,
                       validation_data: np.ndarray,
                       **model_kwargs) -> List[ModelComparisonResult]:
        """
        擬合所有候選模型並評估
        
        Parameters:
        -----------
        train_data : np.ndarray
            訓練數據
        validation_data : np.ndarray
            驗證數據
        **model_kwargs : dict
            傳遞給模型建構函數的額外參數
            
        Returns:
        --------
        List[ModelComparisonResult]
            模型比較結果列表
        """
        
        print("🚀 開始方法一：模型擬合與比較")
        print("=" * 80)
        
        # Step 1: 建立候選模型
        print("\n📦 Step 1: 建立候選模型")
        model_A = self.build_model_A_simple_lognormal(train_data, 
                                                      model_kwargs.get('covariates'))
        model_B = self.build_model_B_hierarchical(train_data,
                                                  model_kwargs.get('groups'),
                                                  model_kwargs.get('covariates'))
        model_C = self.build_model_C_alternative(train_data,
                                                 model_kwargs.get('wind_speed'),
                                                 model_kwargs.get('rainfall'),
                                                 model_kwargs.get('storm_surge'))
        
        models = {
            'A_simple_lognormal': model_A,
            'B_hierarchical': model_B,
            'C_alternative': model_C
        }
        
        # Step 2: 擬合所有模型
        print("\n⚙️ Step 2: 使用 MCMC 擬合所有模型")
        
        for name, model in models.items():
            if model is None:
                continue
                
            print(f"\n  擬合 {name}...")
            
            try:
                with model:
                    # MCMC 採樣
                    trace = pm.sample(
                        draws=self.n_samples,
                        chains=self.n_chains,
                        random_seed=self.random_seed,
                        progressbar=True
                    )
                    
                    self.traces[name] = trace
                    
                    # Step 3: 生成後驗預測分佈
                    print(f"  生成後驗預測分佈...")
                    posterior_predictive = pm.sample_posterior_predictive(
                        trace,
                        random_seed=self.random_seed
                    )
                    
                    self.posterior_predictives[name] = posterior_predictive
                    
            except Exception as e:
                print(f"  ❌ 模型 {name} 擬合失敗: {e}")
                continue
        
        # Step 4: 計算 Skill Scores
        print("\n📊 Step 4: 計算並比較 Skill Scores")
        
        results = []
        
        for name in self.traces.keys():
            trace = self.traces[name]
            post_pred = self.posterior_predictives[name]
            
            # 提取預測樣本
            if 'y_obs' in post_pred.posterior_predictive:
                pred_samples = post_pred.posterior_predictive['y_obs'].values.flatten()
            else:
                print(f"  ⚠️ 模型 {name} 缺少預測樣本")
                continue
            
            # 計算 CRPS
            if HAS_SKILL_SCORES:
                # 為每個驗證點生成預測集合
                n_val = len(validation_data)
                n_samples = len(pred_samples) // n_val if len(pred_samples) >= n_val else len(pred_samples)
                
                if n_samples > 0:
                    pred_ensemble = pred_samples[:n_val*n_samples].reshape(n_val, n_samples)
                    crps = calculate_crps(validation_data, forecasts_ensemble=pred_ensemble)
                    
                    # 計算 TSS (需要二元化)
                    threshold = np.median(validation_data)
                    obs_binary = validation_data > threshold
                    pred_prob = np.mean(pred_ensemble > threshold, axis=1)
                    tss = calculate_tss(obs_binary, pred_prob)
                    
                    # 計算 EDI
                    edi = calculate_edi(validation_data, pred_ensemble)
                else:
                    crps = float('inf')
                    tss = -1
                    edi = 0
            else:
                # 簡化計算
                crps = np.mean(np.abs(pred_samples[:len(validation_data)] - validation_data))
                tss = 0
                edi = 0
            
            # 收集診斷信息
            diagnostics = self._get_convergence_diagnostics(trace)
            
            result = ModelComparisonResult(
                model_name=name,
                model_type=name.split('_')[1],
                trace=trace,
                posterior_predictive=pred_samples,
                crps_score=float(crps),
                tss_score=float(tss),
                edi_score=float(edi),
                log_likelihood=self._calculate_log_likelihood(trace),
                convergence_diagnostics=diagnostics
            )
            
            results.append(result)
            self.comparison_results.append(result)
        
        # Step 5: 選擇最佳模型
        print("\n🏆 Step 5: 根據 Skill Scores 選擇最佳模型")
        self._print_comparison_table(results)
        
        if results:
            best_model = min(results, key=lambda x: x.crps_score)
            print(f"\n✅ 最佳模型: {best_model.model_name}")
            print(f"   CRPS: {best_model.crps_score:.2e}")
            print(f"   TSS: {best_model.tss_score:.3f}")
            
        return results
    
    def _get_convergence_diagnostics(self, trace) -> Dict[str, Any]:
        """獲取收斂診斷"""
        diagnostics = {}
        
        try:
            import arviz as az
            
            # R-hat
            rhat = az.rhat(trace)
            diagnostics['rhat'] = {var: float(rhat[var].max()) for var in rhat.data_vars}
            
            # ESS
            ess = az.ess(trace)
            diagnostics['ess'] = {var: float(ess[var].min()) for var in ess.data_vars}
            
        except:
            diagnostics['rhat'] = {}
            diagnostics['ess'] = {}
        
        return diagnostics
    
    def _calculate_log_likelihood(self, trace) -> float:
        """計算對數似然"""
        try:
            import arviz as az
            loo = az.loo(trace)
            return float(loo.elpd_loo)
        except:
            return -np.inf
    
    def _print_comparison_table(self, results: List[ModelComparisonResult]):
        """列印比較表格"""
        
        if not results:
            print("沒有可比較的結果")
            return
        
        # 創建比較表
        comparison_data = []
        for r in results:
            comparison_data.append({
                '模型': r.model_name,
                'CRPS (越低越好)': f"{r.crps_score:.2e}",
                'TSS (越高越好)': f"{r.tss_score:.3f}",
                'EDI': f"{r.edi_score:.3f}"
            })
        
        df = pd.DataFrame(comparison_data)
        print("\n模型比較結果:")
        print(df.to_string(index=False))
    
    def get_best_model(self) -> Optional[ModelComparisonResult]:
        """獲取最佳模型"""
        if not self.comparison_results:
            return None
        
        return min(self.comparison_results, key=lambda x: x.crps_score)