"""
Mathematical Utilities
數學工具

Core mathematical functions for Bayesian analysis and CRPS calculations.
"""

import numpy as np
from scipy import stats
from scipy.special import erf
from typing import Union, List, Tuple, Optional, Any
import warnings

def crps_empirical(y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate empirical CRPS (Continuous Ranked Probability Score)
    計算經驗CRPS
    
    Parameters:
    -----------
    y_true : np.ndarray
        Observed values
    y_pred : np.ndarray
        Predicted ensemble (samples x predictions) or (predictions x samples)
        
    Returns:
    --------
    Union[float, np.ndarray]
        CRPS scores
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle different shapes - assume y_pred is (n_obs, n_samples)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
    
    if y_true.ndim == 0:
        y_true = np.array([y_true])
    
    # Ensure y_true matches first dimension of y_pred
    if len(y_true) != y_pred.shape[0]:
        if len(y_true) == 1:
            y_true = np.repeat(y_true, y_pred.shape[0])
        elif y_pred.shape[0] == 1:
            y_pred = np.repeat(y_pred, len(y_true), axis=0)
        else:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    crps_values = []
    
    for i in range(len(y_true)):
        obs = y_true[i]
        forecast = y_pred[i] if y_pred.ndim > 1 else y_pred
        
        # Sort forecasts
        forecast = np.sort(forecast)
        n = len(forecast)
        
        # Calculate CRPS using the empirical formula
        # CRPS = ∫ (F(x) - H(x-obs))² dx
        # Where F(x) is empirical CDF, H is Heaviside function
        
        # Vectorized calculation
        diff_obs = np.abs(forecast - obs)  # |x_i - obs|
        
        # Calculate weights for each forecast point
        weights = np.arange(1, n + 1) / n  # Empirical CDF values
        
        # CRPS calculation
        crps = np.mean(diff_obs) - 0.5 * np.mean(np.abs(forecast[:, np.newaxis] - forecast[np.newaxis, :]))
        
        crps_values.append(crps)
    
    return np.array(crps_values) if len(crps_values) > 1 else crps_values[0]

def crps_normal(observation: float, mean: float, std: float) -> float:
    """
    Calculate CRPS for normal distribution analytically
    解析計算正態分佈的CRPS
    
    Parameters:
    -----------
    observation : float
        Observed value
    mean : float
        Predicted mean
    std : float
        Predicted standard deviation
        
    Returns:
    --------
    float
        CRPS score
    """
    if std <= 0:
        return abs(observation - mean)
    
    # Standardize
    z = (observation - mean) / std
    
    # Analytical CRPS formula for normal distribution
    # CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
    phi_z = stats.norm.cdf(z)  # Φ(z)
    pdf_z = stats.norm.pdf(z)  # φ(z)
    
    crps = std * (z * (2 * phi_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))
    
    return crps

def effective_sample_size(x: np.ndarray, maxlag: Optional[int] = None) -> float:
    """
    Calculate effective sample size using autocorrelation
    使用自相關計算有效樣本大小
    
    Parameters:
    -----------
    x : np.ndarray
        MCMC samples
    maxlag : int, optional
        Maximum lag for autocorrelation calculation
        
    Returns:
    --------
    float
        Effective sample size
    """
    x = np.asarray(x)
    n = len(x)
    
    if maxlag is None:
        maxlag = min(n // 4, 200)
    
    # Calculate autocorrelation
    def autocorr(y, maxlag):
        y = y - np.mean(y)
        autocorr_full = np.correlate(y, y, mode='full')
        autocorr_full = autocorr_full[autocorr_full.size // 2:]
        autocorr_full = autocorr_full / autocorr_full[0]
        return autocorr_full[:maxlag + 1]
    
    autocorr_vals = autocorr(x, maxlag)
    
    # Find first negative autocorrelation
    first_negative = np.where(autocorr_vals < 0)[0]
    if len(first_negative) > 0:
        cutoff = first_negative[0]
    else:
        cutoff = len(autocorr_vals)
    
    # Calculate integrated autocorrelation time
    if cutoff > 1:
        tau_int = 1 + 2 * np.sum(autocorr_vals[1:cutoff])
    else:
        tau_int = 1.0
    
    # Effective sample size
    n_eff = n / max(tau_int, 1.0)
    
    return n_eff

def rhat_statistic(chains: List[np.ndarray]) -> float:
    """
    Calculate R-hat convergence diagnostic
    計算R-hat收斂診斷統計量
    
    Parameters:
    -----------
    chains : List[np.ndarray]
        List of MCMC chains
        
    Returns:
    --------
    float
        R-hat statistic
    """
    chains = [np.asarray(chain) for chain in chains]
    
    if len(chains) < 2:
        return 1.0
    
    # Number of chains and samples per chain
    m = len(chains)
    n = len(chains[0])
    
    # Chain means and overall mean
    chain_means = [np.mean(chain) for chain in chains]
    overall_mean = np.mean(chain_means)
    
    # Between-chain variance
    B = n * np.var(chain_means, ddof=1)
    
    # Within-chain variance
    chain_vars = [np.var(chain, ddof=1) for chain in chains]
    W = np.mean(chain_vars)
    
    # Marginal posterior variance estimate
    var_plus = ((n - 1) / n) * W + (1 / n) * B
    
    # R-hat statistic
    if W > 0:
        rhat = np.sqrt(var_plus / W)
    else:
        rhat = 1.0
    
    return rhat

def hdi(samples: np.ndarray, credible_interval: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Highest Density Interval (HDI)
    計算最高密度區間
    
    Parameters:
    -----------
    samples : np.ndarray
        Posterior samples
    credible_interval : float
        Credible interval level (default 0.95)
        
    Returns:
    --------
    Tuple[float, float]
        Lower and upper bounds of HDI
    """
    samples = np.asarray(samples)
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    
    # Number of samples to include
    n_include = int(np.ceil(credible_interval * n))
    
    # Find the shortest interval
    min_width = np.inf
    best_lower = 0
    best_upper = n_include - 1
    
    for i in range(n - n_include + 1):
        width = sorted_samples[i + n_include - 1] - sorted_samples[i]
        if width < min_width:
            min_width = width
            best_lower = i
            best_upper = i + n_include - 1
    
    return sorted_samples[best_lower], sorted_samples[best_upper]

def log_sum_exp(x: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Numerically stable log-sum-exp calculation
    數值穩定的log-sum-exp計算
    
    Parameters:
    -----------
    x : np.ndarray
        Input array
    axis : int, optional
        Axis along which to compute
        
    Returns:
    --------
    Union[float, np.ndarray]
        log(sum(exp(x)))
    """
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    
    if axis is None:
        x_max = np.max(x)
        return x_max + np.log(np.sum(np.exp(x - x_max)))
    else:
        return x_max.squeeze(axis) + np.log(np.sum(np.exp(x - x_max), axis=axis))

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate Kullback-Leibler divergence
    計算KL散度
    
    Parameters:
    -----------
    p : np.ndarray
        Reference distribution
    q : np.ndarray
        Approximate distribution  
    epsilon : float
        Small value to avoid log(0)
        
    Returns:
    --------
    float
        KL divergence
    """
    p = np.asarray(p) + epsilon
    q = np.asarray(q) + epsilon
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sum(p * np.log(p / q))

def wasserstein_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate 1-Wasserstein distance between empirical distributions
    計算經驗分佈間的1-Wasserstein距離
    
    Parameters:
    -----------
    x, y : np.ndarray
        Sample arrays
        
    Returns:
    --------
    float
        Wasserstein distance
    """
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # Empirical CDFs
    n_x, n_y = len(x_sorted), len(y_sorted)
    
    # Create combined sorted array
    all_values = np.concatenate([x_sorted, y_sorted])
    all_values = np.unique(all_values)
    
    # Calculate CDFs at all points
    cdf_x = np.searchsorted(x_sorted, all_values, side='right') / n_x
    cdf_y = np.searchsorted(y_sorted, all_values, side='right') / n_y
    
    # Calculate Wasserstein distance as integral of |CDF_x - CDF_y|
    distances = np.abs(cdf_x - cdf_y)
    
    # Approximate integral using differences
    if len(all_values) > 1:
        dx = np.diff(all_values)
        distance = np.sum(distances[:-1] * dx)
    else:
        distance = 0.0
    
    return distance

def numerical_gradient(func, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
    """
    Calculate numerical gradient using central differences
    使用中心差分計算數值梯度
    
    Parameters:
    -----------
    func : callable
        Function to differentiate
    x : np.ndarray
        Point at which to calculate gradient
    h : float
        Step size
        
    Returns:
    --------
    np.ndarray
        Gradient vector
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
    
    return grad

def robust_covariance(X: np.ndarray, method: str = "mcd") -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate robust covariance matrix
    計算穩健協方差矩陣
    
    Parameters:
    -----------
    X : np.ndarray
        Data matrix (n_samples, n_features)
    method : str
        Method to use ('mcd' for Minimum Covariance Determinant)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Robust mean and covariance matrix
    """
    try:
        from sklearn.covariance import MinCovDet
        
        if method == "mcd":
            mcd = MinCovDet().fit(X)
            return mcd.location_, mcd.covariance_
        else:
            # Fallback to sample covariance
            return np.mean(X, axis=0), np.cov(X.T)
            
    except ImportError:
        warnings.warn("scikit-learn not available, using sample covariance")
        return np.mean(X, axis=0), np.cov(X.T)