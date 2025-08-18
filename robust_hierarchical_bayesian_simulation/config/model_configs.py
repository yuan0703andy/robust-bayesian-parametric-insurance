"""
Model Configuration System
模型配置系統

Centralized configuration management for the 8-stage modular framework.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class ModelComplexity(Enum):
    """Model complexity levels"""
    SIMPLE = "simple"
    STANDARD = "standard"
    ADVANCED = "advanced"
    HPC = "hpc"

class ComputeEnvironment(Enum):
    """Compute environment types"""
    LOCAL = "local"
    CLUSTER = "cluster"
    GPU = "gpu"
    CPU_OPTIMIZED = "cpu_optimized"

@dataclass
class IntegratedFrameworkConfig:
    """Integrated framework configuration"""
    
    # Model complexity
    complexity_level: ModelComplexity = ModelComplexity.STANDARD
    compute_environment: ComputeEnvironment = ComputeEnvironment.LOCAL
    
    # Stage 1: Data Processing
    nc_bounds: Tuple[float, float, float, float] = (-84.5, -75.5, 33.5, 37.0)  # (west, east, south, north)
    year_range: Tuple[int, int] = (1980, 2024)
    resolution: float = 0.1  # degrees
    
    # Stage 2: Robust Priors
    epsilon_empirical: float = 0.1
    epsilon_model: float = 0.05
    contamination_level: float = 0.15
    
    # Stage 3: Hierarchical Modeling
    n_hierarchy_levels: int = 3
    use_spatial_effects: bool = True
    spatial_correlation_range: float = 100.0  # km
    
    # Stage 4: VI Screening
    vi_n_iterations: int = 5000
    vi_learning_rate: float = 0.01
    vi_convergence_tolerance: float = 1e-6
    
    # Stage 5: CRPS Framework
    crps_ensemble_size: int = 1000
    crps_weight_sensitivity: bool = True
    basis_risk_threshold: float = 0.1
    
    # Stage 6: MCMC Validation
    mcmc_n_samples: int = 2000
    mcmc_n_chains: int = 4
    mcmc_n_warmup: int = 1000
    mcmc_target_accept: float = 0.8
    
    # Stage 7: Posterior Analysis
    credible_interval_level: float = 0.95
    n_posterior_predictive_samples: int = 1000
    use_robust_credible_intervals: bool = True
    
    # Stage 8: Parametric Insurance
    parametric_product_types: List[str] = None
    optimization_method: str = "bayesian_optimization"
    risk_tolerance: float = 0.05
    
    def __post_init__(self):
        if self.parametric_product_types is None:
            self.parametric_product_types = ["wind_speed", "pressure", "composite"]
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate configuration parameters"""
        warnings = []
        
        # Check basic bounds
        if self.epsilon_empirical < 0 or self.epsilon_empirical > 1:
            warnings.append("epsilon_empirical should be between 0 and 1")
        
        if self.epsilon_model < 0 or self.epsilon_model > 1:
            warnings.append("epsilon_model should be between 0 and 1")
        
        if self.mcmc_n_chains < 2:
            warnings.append("mcmc_n_chains should be at least 2 for convergence diagnostics")
        
        if self.credible_interval_level <= 0 or self.credible_interval_level >= 1:
            warnings.append("credible_interval_level should be between 0 and 1")
        
        # Check year range
        if self.year_range[1] <= self.year_range[0]:
            warnings.append("Invalid year range: end year should be after start year")
        
        return len(warnings) == 0, warnings
    
    def summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "complexity": self.complexity_level.value,
            "environment": self.compute_environment.value,
            "data_years": f"{self.year_range[0]}-{self.year_range[1]}",
            "spatial_resolution": f"{self.resolution}°",
            "epsilon_contamination": f"{self.epsilon_empirical:.3f}",
            "mcmc_setup": f"{self.mcmc_n_chains} chains × {self.mcmc_n_samples} samples",
            "vi_iterations": self.vi_n_iterations,
            "crps_ensemble": self.crps_ensemble_size,
        }

def create_standard_analysis_config() -> IntegratedFrameworkConfig:
    """Create standard analysis configuration"""
    return IntegratedFrameworkConfig(
        complexity_level=ModelComplexity.STANDARD,
        compute_environment=ComputeEnvironment.LOCAL,
        mcmc_n_samples=2000,
        mcmc_n_chains=4,
        vi_n_iterations=5000,
        crps_ensemble_size=1000
    )

def create_hpc_config() -> IntegratedFrameworkConfig:
    """Create HPC-optimized configuration"""
    return IntegratedFrameworkConfig(
        complexity_level=ModelComplexity.HPC,
        compute_environment=ComputeEnvironment.CLUSTER,
        mcmc_n_samples=10000,
        mcmc_n_chains=8,
        vi_n_iterations=20000,
        crps_ensemble_size=5000,
        n_posterior_predictive_samples=5000
    )

def create_gpu_config() -> IntegratedFrameworkConfig:
    """Create GPU-optimized configuration"""
    return IntegratedFrameworkConfig(
        complexity_level=ModelComplexity.ADVANCED,
        compute_environment=ComputeEnvironment.GPU,
        mcmc_n_samples=8000,
        mcmc_n_chains=6,
        vi_n_iterations=15000,
        crps_ensemble_size=3000
    )

def create_development_config() -> IntegratedFrameworkConfig:
    """Create lightweight development configuration"""
    return IntegratedFrameworkConfig(
        complexity_level=ModelComplexity.SIMPLE,
        compute_environment=ComputeEnvironment.LOCAL,
        mcmc_n_samples=500,
        mcmc_n_chains=2,
        vi_n_iterations=1000,
        crps_ensemble_size=200,
        n_posterior_predictive_samples=200
    )