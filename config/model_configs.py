#!/usr/bin/env python3
"""
Model Configurations Module
模型配置模組

統一的模型配置管理，支援8階段工作流程
遵循 05_complete_integrated_framework.py 的架構

核心功能:
- 統一的模型配置管理
- 8階段工作流程配置
- 預設配置和自定義配置支援

Author: Research Team  
Date: 2025-01-17
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import os

# ========================================
# 工作流程階段定義
# ========================================

class WorkflowStage(Enum):
    """8階段工作流程定義"""
    DATA_PROCESSING = "1_data_processing"
    ROBUST_PRIORS = "2_robust_priors"
    HIERARCHICAL_MODELING = "3_hierarchical_modeling"
    MODEL_SELECTION = "4_model_selection"
    HYPERPARAMETER_OPTIMIZATION = "5_hyperparameter_optimization"
    MCMC_VALIDATION = "6_mcmc_validation"
    POSTERIOR_ANALYSIS = "7_posterior_analysis"
    PARAMETRIC_INSURANCE = "8_parametric_insurance"

class ModelComplexity(Enum):
    """模型複雜度級別"""
    SIMPLE = "simple"           # 快速原型
    STANDARD = "standard"       # 標準分析
    COMPREHENSIVE = "comprehensive"  # 全面分析
    RESEARCH = "research"       # 研究級別

# ========================================
# 配置結構
# ========================================

@dataclass
class DataProcessingConfig:
    """數據處理配置 - 階段1"""
    use_climada_loader: bool = True
    data_validation: bool = True
    missing_data_strategy: str = "interpolation"  # interpolation, removal, imputation
    outlier_detection: bool = True
    outlier_threshold: float = 3.0  # Z-score threshold
    
@dataclass
class RobustPriorsConfig:
    """穩健先驗配置 - 階段2"""
    use_epsilon_contamination: bool = True
    epsilon_estimation_method: str = "empirical_frequency"
    contamination_class: str = "typhoon_specific"
    robustness_criterion: str = "worst_case"
    # ε-contamination特定參數
    epsilon_range: Tuple[float, float] = (0.01, 0.20)
    typhoon_frequency_per_year: float = 3.2

@dataclass
class HierarchicalModelingConfig:
    """階層建模配置 - 階段3"""
    likelihood_family: str = "lognormal"
    prior_scenario: str = "weak_informative"
    vulnerability_type: str = "emanuel"
    include_spatial_effects: bool = True
    include_region_effects: bool = True
    spatial_covariance_function: str = "exponential"
    
@dataclass
class VIScreeningConfig:
    """VI篩選配置 - 階段4"""
    use_vi_screening: bool = True
    vi_max_iterations: int = 5000
    vi_learning_rate: float = 0.01
    vi_convergence_tolerance: float = 1e-6
    basis_risk_aware: bool = True
    differentiable_crps: bool = True

@dataclass
class CRPSFrameworkConfig:
    """CRPS框架配置 - 階段5"""
    use_crps_optimization: bool = True
    weight_sensitivity_analysis: bool = True
    density_ratio_estimation: bool = True
    crps_ensemble_size: int = 500
    basis_risk_threshold: float = 0.1

@dataclass
class MCMCValidationConfig:
    """MCMC驗證配置 - 階段6"""
    n_samples: int = 2000
    n_warmup: int = 1000
    n_chains: int = 4
    target_accept: float = 0.8
    max_treedepth: int = 10
    use_nuts: bool = True
    # HPC特定配置
    use_hpc_optimization: bool = False
    cores_per_chain: int = 1

@dataclass
class PosteriorAnalysisConfig:
    """後驗分析配置 - 階段7"""
    compute_credible_intervals: bool = True
    robust_intervals: bool = True
    posterior_predictive_checks: bool = True
    mixture_approximation: bool = True
    convergence_diagnostics: bool = True
    # 輸出配置
    generate_plots: bool = True
    save_trace: bool = False

@dataclass
class ParametricInsuranceConfig:
    """參數保險配置 - 階段8"""
    basis_risk_minimization: bool = True
    product_optimization: bool = True
    technical_premium_calculation: bool = True
    risk_metrics_computation: bool = True
    # 產品特定參數
    coverage_ratio_range: Tuple[float, float] = (0.5, 1.0)
    deductible_range: Tuple[float, float] = (0.0, 0.3)

@dataclass
class ComputationConfig:
    """計算環境配置"""
    # 環境設置
    device: str = "cpu"  # cpu, gpu, tpu
    float_precision: str = "float64"
    compile_mode: str = "FAST_COMPILE"
    
    # 平行化配置
    use_multiprocessing: bool = True
    max_workers: int = 4
    threading_layer: str = "GNU"
    
    # HPC配置
    detect_hpc: bool = True
    slurm_integration: bool = True
    pbs_integration: bool = True
    
    # 記憶體管理
    memory_efficient: bool = True
    cache_intermediate_results: bool = True
    max_memory_gb: float = 8.0
    
    def __post_init__(self):
        """自動配置HPC環境"""
        if self.detect_hpc:
            self._configure_hpc_environment()
    
    def _configure_hpc_environment(self):
        """配置HPC環境變數"""
        # 設置基本環境變數
        os.environ['PYTENSOR_FLAGS'] = f'device={self.device},floatX={self.float_precision},mode={self.compile_mode},linker=py'
        os.environ['MKL_THREADING_LAYER'] = self.threading_layer
        
        # HPC系統檢測
        if 'SLURM_CPUS_PER_TASK' in os.environ and self.slurm_integration:
            self.max_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
            os.environ['OMP_NUM_THREADS'] = os.environ['SLURM_CPUS_PER_TASK']
            print(f"🖥️ 檢測到SLURM環境，使用 {self.max_workers} 個CPU核心")
            
        elif 'PBS_NCPUS' in os.environ and self.pbs_integration:
            self.max_workers = int(os.environ['PBS_NCPUS'])
            os.environ['OMP_NUM_THREADS'] = os.environ['PBS_NCPUS']
            print(f"🖥️ 檢測到PBS環境，使用 {self.max_workers} 個CPU核心")
            
        else:
            os.environ['OMP_NUM_THREADS'] = str(self.max_workers)
            print(f"🖥️ 使用本地環境，{self.max_workers} 個工作進程")

@dataclass
class IntegratedFrameworkConfig:
    """整合框架配置 - 完整的8階段工作流程"""
    # 基本設置
    complexity_level: ModelComplexity = ModelComplexity.STANDARD
    random_seed: int = 42
    verbose: bool = True
    
    # 各階段配置
    data_processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    robust_priors: RobustPriorsConfig = field(default_factory=RobustPriorsConfig)
    hierarchical_modeling: HierarchicalModelingConfig = field(default_factory=HierarchicalModelingConfig)
    vi_screening: VIScreeningConfig = field(default_factory=VIScreeningConfig)
    crps_framework: CRPSFrameworkConfig = field(default_factory=CRPSFrameworkConfig)
    mcmc_validation: MCMCValidationConfig = field(default_factory=MCMCValidationConfig)
    posterior_analysis: PosteriorAnalysisConfig = field(default_factory=PosteriorAnalysisConfig)
    parametric_insurance: ParametricInsuranceConfig = field(default_factory=ParametricInsuranceConfig)
    
    # 計算配置
    computation: ComputationConfig = field(default_factory=ComputationConfig)
    
    def __post_init__(self):
        """根據複雜度級別調整配置"""
        self._adjust_for_complexity()
        
        # 設置隨機種子
        np.random.seed(self.random_seed)
    
    def _adjust_for_complexity(self):
        """根據複雜度級別調整配置參數"""
        if self.complexity_level == ModelComplexity.SIMPLE:
            # 快速原型配置
            self.mcmc_validation.n_samples = 500
            self.mcmc_validation.n_warmup = 250
            self.mcmc_validation.n_chains = 2
            self.crps_framework.crps_ensemble_size = 200
            self.vi_screening.vi_max_iterations = 1000
            
        elif self.complexity_level == ModelComplexity.STANDARD:
            # 標準配置（已在dataclass中定義）
            pass
            
        elif self.complexity_level == ModelComplexity.COMPREHENSIVE:
            # 全面分析配置
            self.mcmc_validation.n_samples = 5000
            self.mcmc_validation.n_warmup = 2000
            self.mcmc_validation.n_chains = 6
            self.crps_framework.crps_ensemble_size = 1000
            self.vi_screening.vi_max_iterations = 10000
            self.posterior_analysis.save_trace = True
            
        elif self.complexity_level == ModelComplexity.RESEARCH:
            # 研究級別配置
            self.mcmc_validation.n_samples = 10000
            self.mcmc_validation.n_warmup = 5000
            self.mcmc_validation.n_chains = 8
            self.crps_framework.crps_ensemble_size = 2000
            self.vi_screening.vi_max_iterations = 20000
            self.posterior_analysis.save_trace = True
            self.computation.max_memory_gb = 16.0
    
    def get_stage_config(self, stage: WorkflowStage) -> Any:
        """獲取特定階段的配置"""
        stage_mapping = {
            WorkflowStage.DATA_PROCESSING: self.data_processing,
            WorkflowStage.ROBUST_PRIORS: self.robust_priors,
            WorkflowStage.HIERARCHICAL_MODELING: self.hierarchical_modeling,
            WorkflowStage.VI_SCREENING: self.vi_screening,
            WorkflowStage.CRPS_FRAMEWORK: self.crps_framework,
            WorkflowStage.MCMC_VALIDATION: self.mcmc_validation,
            WorkflowStage.POSTERIOR_ANALYSIS: self.posterior_analysis,
            WorkflowStage.PARAMETRIC_INSURANCE: self.parametric_insurance
        }
        return stage_mapping.get(stage)
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """驗證配置的一致性"""
        warnings = []
        
        # 檢查MCMC配置
        if self.mcmc_validation.n_samples < 100:
            warnings.append("MCMC樣本數過少，可能影響收斂性")
        
        if self.mcmc_validation.n_chains < 2:
            warnings.append("建議使用至少2條MCMC鏈進行診斷")
        
        # 檢查記憶體配置
        if self.computation.max_memory_gb < 2.0:
            warnings.append("記憶體配置可能不足以完成分析")
        
        # 檢查ε-contamination配置
        if (self.robust_priors.use_epsilon_contamination and 
            self.robust_priors.epsilon_range[1] > 0.5):
            warnings.append("ε污染程度過高可能導致不穩定")
        
        # 檢查VI配置
        if (self.vi_screening.use_vi_screening and 
            self.vi_screening.vi_max_iterations < 1000):
            warnings.append("VI迭代次數可能不足以收斂")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings
    
    def summary(self) -> Dict[str, Any]:
        """配置摘要"""
        return {
            "複雜度級別": self.complexity_level.value,
            "隨機種子": self.random_seed,
            "MCMC設置": f"{self.mcmc_validation.n_samples} samples, {self.mcmc_validation.n_chains} chains",
            "ε-contamination": self.robust_priors.use_epsilon_contamination,
            "空間效應": self.hierarchical_modeling.include_spatial_effects,
            "VI篩選": self.vi_screening.use_vi_screening,
            "CRPS優化": self.crps_framework.use_crps_optimization,
            "計算設備": self.computation.device,
            "工作進程": self.computation.max_workers
        }

# ========================================
# 預設配置生成器
# ========================================

def create_quick_prototype_config() -> IntegratedFrameworkConfig:
    """創建快速原型配置"""
    return IntegratedFrameworkConfig(
        complexity_level=ModelComplexity.SIMPLE,
        verbose=True
    )

def create_standard_analysis_config() -> IntegratedFrameworkConfig:
    """創建標準分析配置"""
    return IntegratedFrameworkConfig(
        complexity_level=ModelComplexity.STANDARD,
        verbose=True
    )

def create_comprehensive_research_config() -> IntegratedFrameworkConfig:
    """創建全面研究配置"""
    return IntegratedFrameworkConfig(
        complexity_level=ModelComplexity.COMPREHENSIVE,
        verbose=True
    )

def create_hpc_optimized_config() -> IntegratedFrameworkConfig:
    """創建HPC優化配置"""
    config = IntegratedFrameworkConfig(
        complexity_level=ModelComplexity.RESEARCH,
        verbose=False  # HPC環境中減少輸出
    )
    
    # HPC特定調整
    config.computation.detect_hpc = True
    config.computation.memory_efficient = True
    config.computation.cache_intermediate_results = False  # 避免磁碟IO
    config.mcmc_validation.use_hpc_optimization = True
    
    return config

def create_epsilon_contamination_focused_config() -> IntegratedFrameworkConfig:
    """創建專注於ε-contamination的配置"""
    config = IntegratedFrameworkConfig(complexity_level=ModelComplexity.STANDARD)
    
    # 強化ε-contamination相關設置
    config.robust_priors.use_epsilon_contamination = True
    config.robust_priors.epsilon_estimation_method = "bayesian_model_selection"
    config.robust_priors.contamination_class = "typhoon_specific"
    config.robust_priors.epsilon_range = (0.005, 0.15)  # 更精確的範圍
    
    # 增強後驗分析
    config.posterior_analysis.robust_intervals = True
    config.posterior_analysis.mixture_approximation = True
    
    return config

def test_model_configs():
    """測試模型配置功能"""
    print("🧪 測試模型配置模組...")
    
    # 測試標準配置
    print("✅ 測試標準配置:")
    config = create_standard_analysis_config()
    print(f"   複雜度: {config.complexity_level.value}")
    print(f"   MCMC鏈數: {config.mcmc_validation.n_chains}")
    
    # 測試配置驗證
    print("✅ 測試配置驗證:")
    is_valid, warnings = config.validate_configuration()
    print(f"   驗證結果: {is_valid}")
    if warnings:
        print(f"   警告數量: {len(warnings)}")
    
    # 測試配置摘要
    print("✅ 測試配置摘要:")
    summary = config.summary()
    print(f"   摘要項目: {len(summary)}")
    
    # 測試HPC配置
    print("✅ 測試HPC配置:")
    hpc_config = create_hpc_optimized_config()
    print(f"   HPC優化: {hpc_config.computation.detect_hpc}")
    
    print("✅ 模型配置測試完成")

if __name__ == "__main__":
    test_model_configs()