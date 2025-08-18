#!/usr/bin/env python3
"""
Prior Specifications Module
先驗規格定義模組

從 parametric_bayesian_hierarchy.py 拆分出的先驗相關定義
包含所有的 Enum 類別、配置結構和先驗規格

核心功能:
- 概似函數族定義 (LikelihoodFamily)
- 事前情境定義 (PriorScenario)
- 脆弱度函數類型定義 (VulnerabilityFunctionType)
- 污染分布定義 (ContaminationDistribution)
- 模型規格配置 (ModelSpec)

Author: Research Team  
Date: 2025-01-17
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# ========================================
# 基本枚舉定義
# ========================================

class LikelihoodFamily(Enum):
    """概似函數族"""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    STUDENT_T = "student_t"
    LAPLACE = "laplace"
    GAMMA = "gamma"
    BETA = "beta"
    WEIBULL = "weibull"
    
    # ε-contamination擴展
    EPSILON_CONTAMINATION_FIXED = "epsilon_contamination_fixed"
    EPSILON_CONTAMINATION_ESTIMATED = "epsilon_contamination_estimated"
    
    # Generalized Pareto Distribution for extreme value modeling
    GPD = "gpd"

class PriorScenario(Enum):
    """事前分佈情境"""
    NON_INFORMATIVE = "non_informative"      # 非信息事前
    WEAK_INFORMATIVE = "weak_informative"    # 弱信息事前  
    INFORMATIVE = "informative"              # 信息事前
    OPTIMISTIC = "optimistic"                # 樂觀事前（較低損失）
    PESSIMISTIC = "pessimistic"              # 悲觀事前（較高損失）
    CONSERVATIVE = "conservative"            # 保守事前
    
    # ε-contamination specific scenarios
    ROBUST_WEAK = "robust_weak"              # 穩健弱事前
    ROBUST_CONSERVATIVE = "robust_conservative"  # 穩健保守事前

class VulnerabilityFunctionType(Enum):
    """脆弱度函數類型"""
    EMANUEL = "emanuel"           # Emanuel USA颱風損失函數
    LINEAR = "linear"             # 線性脆弱度函數
    POLYNOMIAL = "polynomial"     # 多項式脆弱度函數
    EXPONENTIAL = "exponential"   # 指數脆弱度函數
    LOGISTIC = "logistic"         # 邏輯脆弱度函數
    PIECEWISE = "piecewise"       # 分段脆弱度函數

class ContaminationDistribution(Enum):
    """ε-contamination模型中的污染分布類型"""
    CAUCHY = "cauchy"             # Cauchy分布（重尾）
    STUDENT_T = "student_t"       # Student-t分布（可調尾部）
    LAPLACE = "laplace"           # Laplace分布（雙指數）
    UNIFORM = "uniform"           # 均勻分布
    EXPONENTIAL = "exponential"   # 指數分布
    GPD = "gpd"                   # Generalized Pareto Distribution

class CovarianceFunction(Enum):
    """空間協方差函數類型"""
    EXPONENTIAL = "exponential"   # 指數協方差
    MATERN_32 = "matern_32"      # Matérn 3/2
    MATERN_52 = "matern_52"      # Matérn 5/2
    GAUSSIAN = "gaussian"         # 高斯協方差
    LINEAR = "linear"            # 線性協方差
    SPHERICAL = "spherical"      # 球面協方差

# ========================================
# 數據結構定義
# ========================================

@dataclass
class VulnerabilityData:
    """脆弱度建模數據"""
    hazard_intensities: np.ndarray      # H_ij - 災害強度（如風速 m/s）
    exposure_values: np.ndarray         # E_i - 暴險值（如建築物價值 USD）
    observed_losses: np.ndarray         # L_ij - 觀測損失 (USD)
    event_ids: Optional[np.ndarray] = None      # 事件ID
    location_ids: Optional[np.ndarray] = None   # 地點ID
    
    # 空間信息
    hospital_coordinates: Optional[np.ndarray] = None    # 醫院座標 [(lat1, lon1), ...]
    hospital_names: Optional[List[str]] = None           # 醫院名稱
    region_assignments: Optional[np.ndarray] = None      # 區域分配 [0, 1, 2, ...]
    
    def __post_init__(self):
        """驗證數據一致性"""
        arrays = [self.hazard_intensities, self.exposure_values, self.observed_losses]
        lengths = [len(arr) for arr in arrays if arr is not None]
        
        if len(set(lengths)) > 1:
            raise ValueError(f"數據長度不一致: {lengths}")
        
        # 檢查是否有空間信息
        self._check_spatial_info()
    
    def _check_spatial_info(self):
        """檢查空間信息的完整性"""
        if self.hospital_coordinates is not None:
            if self.hospital_coordinates.shape[1] != 2:
                raise ValueError("醫院座標必須是 (n_hospitals, 2) 的格式")
        
        if self.region_assignments is not None:
            if len(self.region_assignments) != self.n_hospitals:
                raise ValueError("區域分配長度必須等於醫院數量")
    
    @property
    def n_observations(self) -> int:
        """觀測數量"""
        return len(self.hazard_intensities)
    
    @property 
    def n_hospitals(self) -> int:
        """醫院數量（如果有空間信息）"""
        if self.hospital_coordinates is not None:
            return len(self.hospital_coordinates)
        elif self.location_ids is not None:
            return len(np.unique(self.location_ids))
        else:
            return 1  # 假設只有一個位置
    
    @property
    def has_spatial_info(self) -> bool:
        """是否有空間信息"""
        return (self.hospital_coordinates is not None and 
                len(self.hospital_coordinates) > 1)
    
    @property
    def has_region_info(self) -> bool:
        """是否有區域信息"""
        return (self.region_assignments is not None and 
                len(np.unique(self.region_assignments)) > 1)

@dataclass 
class SpatialConfig:
    """空間效應配置"""
    covariance_function: CovarianceFunction = CovarianceFunction.EXPONENTIAL
    length_scale: float = 50.0           # 空間長度尺度 (km)
    variance: float = 1.0                # 空間變異數
    nugget: float = 0.1                  # nugget effect
    
    # 先驗配置
    length_scale_prior: Tuple[float, float] = (10.0, 100.0)
    variance_prior: Tuple[float, float] = (0.5, 2.0)
    
    def __post_init__(self):
        if isinstance(self.covariance_function, str):
            self.covariance_function = CovarianceFunction(self.covariance_function)

@dataclass
class ModelSpec:
    """模型規格配置"""
    # 基本模型配置
    likelihood_family: LikelihoodFamily = LikelihoodFamily.LOGNORMAL
    prior_scenario: PriorScenario = PriorScenario.WEAK_INFORMATIVE
    vulnerability_type: VulnerabilityFunctionType = VulnerabilityFunctionType.EMANUEL
    model_name: Optional[str] = None
    
    # 空間效應配置
    include_spatial_effects: bool = True           # 是否包含空間隨機效應 δ_i
    include_region_effects: bool = True            # 是否包含區域效應 α_r(i)
    spatial_covariance_function: str = "exponential"  # 空間協方差函數
    spatial_length_scale_prior: Tuple[float, float] = (10.0, 100.0)  # 長度尺度先驗
    spatial_variance_prior: Tuple[float, float] = (0.5, 2.0)         # 空間變異數先驗
    
    # ε-contamination 配置
    epsilon_contamination: Optional[float] = None    # 固定ε值 (如 3.2/365 ≈ 0.0088)
    epsilon_prior: Tuple[float, float] = (1.0, 30.0)  # Beta先驗參數 (α, β) for estimated ε
    contamination_distribution: ContaminationDistribution = ContaminationDistribution.CAUCHY  # 污染分布類型
    
    # GPD 特定參數
    gpd_threshold: Optional[float] = None           # GPD閾值 (自動計算如果為None)
    gpd_xi_prior: Tuple[float, float] = (0.0, 0.5)  # GPD形狀參數先驗 N(μ, σ)
    gpd_sigma_prior: float = 1.0                    # GPD尺度參數先驗
    
    def __post_init__(self):
        # 類型轉換支援
        if isinstance(self.likelihood_family, str):
            self.likelihood_family = LikelihoodFamily(self.likelihood_family)
        if isinstance(self.prior_scenario, str):
            self.prior_scenario = PriorScenario(self.prior_scenario)
        if isinstance(self.vulnerability_type, str):
            self.vulnerability_type = VulnerabilityFunctionType(self.vulnerability_type)
        if isinstance(self.contamination_distribution, str):
            self.contamination_distribution = ContaminationDistribution(self.contamination_distribution)
        
        if self.model_name is None:
            # 包含污染分布信息在模型名稱中（如果使用ε-contamination）
            if self.likelihood_family in [LikelihoodFamily.EPSILON_CONTAMINATION_FIXED, 
                                         LikelihoodFamily.EPSILON_CONTAMINATION_ESTIMATED]:
                self.model_name = f"{self.likelihood_family.value}_{self.prior_scenario.value}_{self.contamination_distribution.value}"
            else:
                self.model_name = f"{self.likelihood_family.value}_{self.prior_scenario.value}_{self.vulnerability_type.value}"

# ========================================
# 先驗規格生成函數
# ========================================

def get_prior_parameters(prior_scenario: PriorScenario, 
                        parameter_type: str) -> Dict[str, Any]:
    """
    獲取指定事前情境的參數配置
    
    Parameters:
    -----------
    prior_scenario : PriorScenario
        事前情境
    parameter_type : str
        參數類型 ('alpha', 'beta', 'sigma', etc.)
        
    Returns:
    --------
    Dict[str, Any]
        參數配置字典
    """
    
    if parameter_type == "alpha":
        if prior_scenario == PriorScenario.NON_INFORMATIVE:
            return {"mu": 0, "sigma": 10}
        elif prior_scenario == PriorScenario.WEAK_INFORMATIVE:
            return {"mu": 0, "sigma": 2}
        elif prior_scenario == PriorScenario.OPTIMISTIC:
            return {"mu": -1, "sigma": 1}
        elif prior_scenario == PriorScenario.PESSIMISTIC:
            return {"mu": 1, "sigma": 1}
        elif prior_scenario == PriorScenario.CONSERVATIVE:
            return {"mu": 0.5, "sigma": 0.5}
    
    elif parameter_type == "beta":
        if prior_scenario == PriorScenario.NON_INFORMATIVE:
            return {"sigma": 5}
        elif prior_scenario == PriorScenario.WEAK_INFORMATIVE:
            return {"sigma": 1}
        elif prior_scenario == PriorScenario.OPTIMISTIC:
            return {"sigma": 0.5}
        elif prior_scenario == PriorScenario.PESSIMISTIC:
            return {"sigma": 2}
        elif prior_scenario == PriorScenario.CONSERVATIVE:
            return {"sigma": 0.8}
    
    elif parameter_type == "sigma":
        if prior_scenario == PriorScenario.NON_INFORMATIVE:
            return {"sigma": 5}
        elif prior_scenario == PriorScenario.WEAK_INFORMATIVE:
            return {"sigma": 1}
        elif prior_scenario == PriorScenario.OPTIMISTIC:
            return {"sigma": 0.5}
        elif prior_scenario == PriorScenario.PESSIMISTIC:
            return {"sigma": 2}
        elif prior_scenario == PriorScenario.CONSERVATIVE:
            return {"sigma": 0.8}
    
    # 預設回退
    return {"mu": 0, "sigma": 1}

def get_contamination_parameters(contamination_dist: ContaminationDistribution,
                               location: float = 0.0,
                               scale: float = 1.0) -> Dict[str, Any]:
    """
    獲取污染分布的參數配置
    
    Parameters:
    -----------
    contamination_dist : ContaminationDistribution
        污染分布類型
    location : float
        位置參數
    scale : float
        尺度參數
        
    Returns:
    --------
    Dict[str, Any]
        污染分布參數
    """
    
    if contamination_dist == ContaminationDistribution.CAUCHY:
        return {
            "alpha": location,
            "beta": scale * 2  # Cauchy更寬的尺度
        }
    
    elif contamination_dist == ContaminationDistribution.STUDENT_T:
        return {
            "nu": 3,  # 重尾
            "mu": location,
            "sigma": scale * 1.5
        }
    
    elif contamination_dist == ContaminationDistribution.LAPLACE:
        return {
            "mu": location,
            "b": scale
        }
    
    elif contamination_dist == ContaminationDistribution.UNIFORM:
        return {
            "lower": location - 3 * scale,
            "upper": location + 3 * scale
        }
    
    elif contamination_dist == ContaminationDistribution.EXPONENTIAL:
        return {
            "lam": 1.0 / scale
        }
    
    elif contamination_dist == ContaminationDistribution.GPD:
        return {
            "mu": location,  # 閾值
            "sigma": scale,  # 尺度參數
            "xi": 0.1       # 形狀參數
        }
    
    # 預設回退到正態分布
    return {"mu": location, "sigma": scale}

def validate_model_spec(model_spec: ModelSpec) -> bool:
    """
    驗證模型規格的一致性
    
    Parameters:
    -----------
    model_spec : ModelSpec
        模型規格
        
    Returns:
    --------
    bool
        是否通過驗證
    """
    
    # 檢查ε-contamination配置
    if model_spec.likelihood_family in [LikelihoodFamily.EPSILON_CONTAMINATION_FIXED,
                                       LikelihoodFamily.EPSILON_CONTAMINATION_ESTIMATED]:
        if model_spec.contamination_distribution is None:
            print("⚠️ ε-contamination模型需要指定污染分布")
            return False
        
        if (model_spec.likelihood_family == LikelihoodFamily.EPSILON_CONTAMINATION_FIXED and 
            model_spec.epsilon_contamination is None):
            print("⚠️ 固定ε模型需要指定ε值")
            return False
    
    # 檢查空間效應配置
    if model_spec.include_spatial_effects:
        if model_spec.spatial_covariance_function not in [cf.value for cf in CovarianceFunction]:
            print(f"⚠️ 不支援的空間協方差函數: {model_spec.spatial_covariance_function}")
            return False
    
    # 檢查GPD配置
    if model_spec.likelihood_family == LikelihoodFamily.GPD:
        if model_spec.gpd_threshold is None:
            print("⚠️ GPD模型需要指定閾值（或設為None以自動計算）")
    
    return True

def test_prior_specifications():
    """測試先驗規格功能"""
    print("🧪 測試先驗規格定義...")
    
    # 測試基本枚舉
    print("✅ 測試概似函數族:")
    for family in LikelihoodFamily:
        print(f"   {family.value}")
    
    # 測試模型規格
    print("✅ 測試模型規格:")
    model_spec = ModelSpec(
        likelihood_family=LikelihoodFamily.LOGNORMAL,
        prior_scenario=PriorScenario.WEAK_INFORMATIVE,
        vulnerability_type=VulnerabilityFunctionType.EMANUEL
    )
    print(f"   模型名稱: {model_spec.model_name}")
    
    # 測試驗證
    print("✅ 測試規格驗證:")
    is_valid = validate_model_spec(model_spec)
    print(f"   驗證結果: {is_valid}")
    
    print("✅ 先驗規格測試完成")

if __name__ == "__main__":
    test_prior_specifications()