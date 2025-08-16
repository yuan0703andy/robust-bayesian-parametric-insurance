"""
Model Factory for VI+MCMC Framework
模型工廠 - 建立不同類型的貝氏模型
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pymc as pm
import pytensor.tensor as pt
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

# Models are built directly with PyMC - no need to import specific classes


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    model_type: str
    epsilon: Optional[float] = None
    nu: Optional[float] = None
    prior_strength: float = 10.0
    contamination_scale: float = 5.0


class ModelFactory:
    """Factory for creating PyMC models"""
    
    @staticmethod
    def create_epsilon_model(config: ModelConfig, X: np.ndarray, y: np.ndarray) -> pm.Model:
        """Create epsilon-contamination model"""
        
        with pm.Model() as model:
            # Priors
            beta = pm.Normal('beta', mu=0, sigma=config.prior_strength, shape=X.shape[1])
            sigma = pm.HalfNormal('sigma', sigma=config.prior_strength)
            
            # Linear predictor
            mu = pm.math.dot(X, beta)
            
            # Epsilon contamination
            epsilon = config.epsilon or 0.1
            
            # Main component (clean)
            main_dist = pm.Normal.dist(mu=mu, sigma=sigma)
            
            # Contamination component (outliers)
            contam_dist = pm.StudentT.dist(
                nu=3, 
                mu=mu, 
                sigma=sigma * config.contamination_scale
            )
            
            # Mixture
            y_obs = pm.Mixture(
                'y_obs',
                w=[1 - epsilon, epsilon],
                comp_dists=[main_dist, contam_dist],
                observed=y
            )
            
        return model
    
    @staticmethod
    def create_student_t_model(config: ModelConfig, X: np.ndarray, y: np.ndarray) -> pm.Model:
        """Create Student-t robust model"""
        
        with pm.Model() as model:
            # Priors
            beta = pm.Normal('beta', mu=0, sigma=config.prior_strength, shape=X.shape[1])
            sigma = pm.HalfNormal('sigma', sigma=config.prior_strength)
            
            # Degrees of freedom
            nu = config.nu or 5
            
            # Linear predictor
            mu = pm.math.dot(X, beta)
            
            # Student-t likelihood
            y_obs = pm.StudentT('y_obs', nu=nu, mu=mu, sigma=sigma, observed=y)
            
        return model
    
    @staticmethod
    def create_standard_model(config: ModelConfig, X: np.ndarray, y: np.ndarray) -> pm.Model:
        """Create standard Bayesian regression model"""
        
        with pm.Model() as model:
            # Priors
            beta = pm.Normal('beta', mu=0, sigma=config.prior_strength, shape=X.shape[1])
            sigma = pm.HalfNormal('sigma', sigma=config.prior_strength)
            
            # Linear predictor
            mu = pm.math.dot(X, beta)
            
            # Normal likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
        return model
    
    @staticmethod
    def create_hierarchical_model(config: ModelConfig, X: np.ndarray, y: np.ndarray) -> pm.Model:
        """Create hierarchical Bayesian model"""
        
        with pm.Model() as model:
            # Hyperpriors
            mu_beta = pm.Normal('mu_beta', mu=0, sigma=config.prior_strength)
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=5)
            
            # Hierarchical parameters
            beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, shape=X.shape[1])
            sigma = pm.HalfNormal('sigma', sigma=config.prior_strength)
            
            # Linear predictor
            mu = pm.math.dot(X, beta)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
        return model
    
    @classmethod
    def create_model(cls, config: ModelConfig, X: np.ndarray, y: np.ndarray) -> pm.Model:
        """Create model based on configuration"""
        
        if config.model_type == 'epsilon':
            return cls.create_epsilon_model(config, X, y)
        elif config.model_type == 'student_t':
            return cls.create_student_t_model(config, X, y)
        elif config.model_type == 'hierarchical':
            return cls.create_hierarchical_model(config, X, y)
        else:  # standard
            return cls.create_standard_model(config, X, y)