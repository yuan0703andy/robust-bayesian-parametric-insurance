"""
Unified Parametric Insurance Engine
統一參數型保險引擎

This module consolidates all parametric insurance functionality including:
- Cat-in-a-Circle index extraction
- Payout function generation and optimization
- Technical premium calculation
- Product performance evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


def calculate_correct_step_payouts(wind_speeds, thresholds, payout_ratios, max_payout):
    """計算正確的階梯式賠付（無 break，允許高閾值覆蓋）"""
    payouts = np.zeros(len(wind_speeds))
    
    for i, wind_speed in enumerate(wind_speeds):
        for j, threshold in enumerate(thresholds):
            if wind_speed >= threshold:
                payouts[i] = max_payout * payout_ratios[j]
                # 不要 break！繼續檢查更高的閾值
    
    return payouts


def calculate_crps_score(observation, forecast_ensemble):
    """計算 CRPS 分數"""
    forecast_ensemble = np.array(forecast_ensemble)
    n = len(forecast_ensemble)
    
    if n == 0:
        return abs(observation)
    
    sorted_forecasts = np.sort(forecast_ensemble)
    crps = 0.0
    
    # 主要項：觀測值與預測分布的差異
    for i, forecast in enumerate(sorted_forecasts):
        weight = (2 * i + 1) / n
        crps += weight * abs(observation - forecast)
    
    # 修正項：預測值之間的差異
    for i in range(n):
        for j in range(i + 1, n):
            crps -= abs(sorted_forecasts[i] - sorted_forecasts[j]) / (n * n)
    
    return crps

class ParametricIndexType(Enum):
    """參數指標類型"""
    CAT_IN_CIRCLE = "cat_in_circle"
    MAXIMUM_WIND = "maximum_wind"
    AVERAGE_WIND = "average_wind"
    DURATION_ABOVE_THRESHOLD = "duration_above_threshold"

class PayoutFunctionType(Enum):
    """賠付函數類型"""
    STEP = "step"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SIGMOID = "sigmoid"

@dataclass
class ParametricProduct:
    """參數型保險產品數據類"""
    product_id: str
    name: str
    description: str
    index_type: ParametricIndexType
    payout_function_type: PayoutFunctionType
    trigger_thresholds: List[float]
    payout_amounts: List[float]
    max_payout: float
    technical_premium: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProductPerformance:
    """產品績效數據類"""
    product_id: str
    rmse: float
    mae: float
    correlation: float
    hit_rate: float
    false_alarm_rate: float
    coverage_ratio: float
    basis_risk: float
    skill_scores: Dict[str, float] = field(default_factory=dict)
    technical_metrics: Dict[str, float] = field(default_factory=dict)

class ParametricIndexExtractor(ABC):
    """參數指標提取器抽象基類"""
    
    @abstractmethod
    def extract_indices(self, hazard_data: Any, exposure_locations: Any) -> np.ndarray:
        """提取參數指標"""
        pass

class CatInCircleExtractor(ParametricIndexExtractor):
    """Cat-in-a-Circle指標提取器"""
    
    def __init__(self, radius_km: float = 30, statistical_metric: str = 'max'):
        self.radius_km = radius_km
        self.statistical_metric = statistical_metric
        self._spatial_tree = None
        self._centroids_coords = None
    
    def extract_indices(self, tc_hazard: Any, exposure_locations: Any) -> Dict[int, np.ndarray]:
        """
        提取Cat-in-a-Circle參數指標
        
        Parameters:
        -----------
        tc_hazard : TropCyclone
            CLIMADA颱風災害物件
        exposure_locations : List[Tuple[float, float]]
            曝險點位置 (lat, lon)
            
        Returns:
        --------
        Dict[int, np.ndarray]
            事件參數指標字典
        """
        return self._extract_optimized(tc_hazard, exposure_locations)
    
    def _extract_optimized(self, tc_hazard, exposure_locations):
        """優化的提取邏輯"""
        # 這裡整合原本的優化邏輯
        from scipy.spatial import cKDTree
        
        # 建立空間索引
        if self._spatial_tree is None:
            self._build_spatial_index(tc_hazard.centroids.lat, tc_hazard.centroids.lon)
        
        parametric_indices = {}
        n_events = tc_hazard.size
        
        for event_idx in range(n_events):
            if event_idx >= tc_hazard.size:
                break
                
            # 獲取災害強度場
            intensity_field = tc_hazard.intensity[event_idx, :].toarray().flatten()
            parametric_values = []
            
            for exp_lat, exp_lon in exposure_locations:
                # 使用空間索引查詢
                distances, indices = self._spatial_tree.query([exp_lon, exp_lat], k=10)
                
                if len(indices) > 0:
                    # 精確距離計算
                    candidate_intensities = intensity_field[indices]
                    valid_intensities = candidate_intensities[candidate_intensities > 0]
                    
                    if len(valid_intensities) > 0:
                        if self.statistical_metric == 'max':
                            param_value = np.max(valid_intensities)
                        elif self.statistical_metric == 'mean':
                            param_value = np.mean(valid_intensities)
                        else:
                            param_value = np.max(valid_intensities)
                    else:
                        param_value = 0.0
                else:
                    param_value = 0.0
                    
                parametric_values.append(param_value)
            
            parametric_indices[event_idx] = np.array(parametric_values)
        
        return parametric_indices
    
    def _build_spatial_index(self, centroids_lat, centroids_lon):
        """建立空間索引"""
        from scipy.spatial import cKDTree
        
        self._centroids_coords = np.column_stack([centroids_lon, centroids_lat])
        self._spatial_tree = cKDTree(self._centroids_coords)

class PayoutFunction(ABC):
    """賠付函數抽象基類"""
    
    def __init__(self, trigger_thresholds: List[float], payout_amounts: List[float], max_payout: float):
        self.trigger_thresholds = trigger_thresholds
        self.payout_amounts = payout_amounts
        self.max_payout = max_payout
    
    @abstractmethod
    def calculate_payout(self, parametric_index: float) -> float:
        """計算單個參數指標的賠付"""
        pass
    
    def calculate_payouts_batch(self, parametric_indices: np.ndarray) -> np.ndarray:
        """批量計算賠付"""
        return np.array([self.calculate_payout(idx) for idx in parametric_indices])

class StepPayoutFunction(PayoutFunction):
    """階梯賠付函數"""
    
    def calculate_payout(self, parametric_index: float) -> float:
        """計算階梯賠付"""
        payout = 0.0
        
        for i, threshold in enumerate(self.trigger_thresholds):
            if parametric_index >= threshold:
                if i < len(self.payout_amounts):
                    payout = self.payout_amounts[i]
        
        return min(payout, self.max_payout)

class LinearPayoutFunction(PayoutFunction):
    """線性賠付函數"""
    
    def calculate_payout(self, parametric_index: float) -> float:
        """計算線性賠付"""
        if len(self.trigger_thresholds) < 2:
            return 0.0
        
        min_threshold = min(self.trigger_thresholds)
        max_threshold = max(self.trigger_thresholds)
        
        if parametric_index < min_threshold:
            return 0.0
        elif parametric_index >= max_threshold:
            return self.max_payout
        else:
            # 線性插值
            ratio = (parametric_index - min_threshold) / (max_threshold - min_threshold)
            return ratio * self.max_payout

class ParametricInsuranceEngine:
    """
    統一的參數型保險引擎
    
    整合了所有參數型保險相關功能，提供高級的統一介面。
    """
    
    def __init__(self):
        self.index_extractors = {
            ParametricIndexType.CAT_IN_CIRCLE: CatInCircleExtractor()
        }
        self.payout_function_classes = {
            PayoutFunctionType.STEP: StepPayoutFunction,
            PayoutFunctionType.LINEAR: LinearPayoutFunction
        }
        self.products = {}
        self.performance_cache = {}
    
    def create_parametric_product(self,
                                product_id: str,
                                name: str,
                                description: str,
                                index_type: ParametricIndexType,
                                payout_function_type: PayoutFunctionType,
                                trigger_thresholds: List[float],
                                payout_amounts: List[float],
                                max_payout: float,
                                **kwargs) -> ParametricProduct:
        """
        創建參數型保險產品
        
        Parameters:
        -----------
        product_id : str
            產品ID
        name : str
            產品名稱
        description : str
            產品描述
        index_type : ParametricIndexType
            參數指標類型
        payout_function_type : PayoutFunctionType
            賠付函數類型
        trigger_thresholds : List[float]
            觸發閾值列表
        payout_amounts : List[float]
            賠付金額列表
        max_payout : float
            最大賠付金額
        **kwargs : dict
            額外參數
            
        Returns:
        --------
        ParametricProduct
            創建的產品
        """
        
        product = ParametricProduct(
            product_id=product_id,
            name=name,
            description=description,
            index_type=index_type,
            payout_function_type=payout_function_type,
            trigger_thresholds=trigger_thresholds,
            payout_amounts=payout_amounts,
            max_payout=max_payout,
            metadata=kwargs
        )
        
        self.products[product_id] = product
        return product
    
    
    def evaluate_product_performance(self,
                                   product: ParametricProduct,
                                   parametric_indices: np.ndarray,
                                   observed_losses: np.ndarray) -> ProductPerformance:
        """
        評估產品績效
        
        Parameters:
        -----------
        product : ParametricProduct
            要評估的產品
        parametric_indices : np.ndarray
            參數指標
        observed_losses : np.ndarray
            觀測損失
            
        Returns:
        --------
        ProductPerformance
            產品績效結果
        """
        
        # 創建賠付函數
        payout_func_class = self.payout_function_classes[product.payout_function_type]
        payout_func = payout_func_class(
            product.trigger_thresholds,
            product.payout_amounts,
            product.max_payout
        )
        
        # 計算賠付
        predicted_payouts = payout_func.calculate_payouts_batch(parametric_indices)
        
        # 計算基本指標
        rmse = np.sqrt(np.mean((observed_losses - predicted_payouts)**2))
        mae = np.mean(np.abs(observed_losses - predicted_payouts))
        correlation = np.corrcoef(observed_losses, predicted_payouts)[0,1] if np.std(predicted_payouts) > 0 else 0
        hit_rate = np.mean((observed_losses > 0) & (predicted_payouts > 0))
        false_alarm_rate = np.mean((observed_losses == 0) & (predicted_payouts > 0))
        coverage_ratio = np.sum(predicted_payouts) / np.sum(observed_losses) if np.sum(observed_losses) > 0 else 0
        basis_risk = np.std(observed_losses - predicted_payouts)
        
        # 技術指標
        technical_metrics = {
            'payout_frequency': np.mean(predicted_payouts > 0),
            'average_payout': np.mean(predicted_payouts[predicted_payouts > 0]) if np.any(predicted_payouts > 0) else 0,
            'max_payout': np.max(predicted_payouts),
            'total_exposure': np.sum(predicted_payouts),
            'utilization_rate': np.sum(predicted_payouts) / (len(predicted_payouts) * product.max_payout)
        }
        
        performance = ProductPerformance(
            product_id=product.product_id,
            rmse=rmse,
            mae=mae,
            correlation=correlation,
            hit_rate=hit_rate,
            false_alarm_rate=false_alarm_rate,
            coverage_ratio=coverage_ratio,
            basis_risk=basis_risk,
            technical_metrics=technical_metrics
        )
        
        # 緩存結果
        self.performance_cache[product.product_id] = performance
        
        return performance
    
    def optimize_product_portfolio(self,
                                 products: List[ParametricProduct],
                                 parametric_indices: np.ndarray,
                                 observed_losses: np.ndarray,
                                 optimization_criteria: List[str] = None) -> pd.DataFrame:
        """
        優化產品組合
        
        Parameters:
        -----------
        products : List[ParametricProduct]
            候選產品列表
        parametric_indices : np.ndarray
            參數指標
        observed_losses : np.ndarray
            觀測損失
        optimization_criteria : List[str]
            優化標準
            
        Returns:
        --------
        pd.DataFrame
            優化結果
        """
        
        if optimization_criteria is None:
            optimization_criteria = ['rmse', 'correlation', 'coverage_ratio']
        
        # 評估所有產品
        performance_results = []
        for product in products:
            performance = self.evaluate_product_performance(product, parametric_indices, observed_losses)
            
            result_dict = {
                'product_id': product.product_id,
                'name': product.name,
                'description': product.description,
                'category': product.metadata.get('category', 'unknown'),
                'rmse': performance.rmse,
                'mae': performance.mae,
                'correlation': performance.correlation,
                'hit_rate': performance.hit_rate,
                'false_alarm_rate': performance.false_alarm_rate,
                'coverage_ratio': performance.coverage_ratio,
                'basis_risk': performance.basis_risk,
                **performance.technical_metrics
            }
            
            performance_results.append(result_dict)
        
        results_df = pd.DataFrame(performance_results)
        
        # 多目標優化評分
        normalized_scores = {}
        for criterion in optimization_criteria:
            if criterion in results_df.columns:
                values = results_df[criterion].values
                if criterion in ['rmse', 'mae', 'basis_risk', 'false_alarm_rate']:
                    # 越小越好
                    normalized_scores[criterion] = 1 - (values - np.min(values)) / (np.max(values) - np.min(values)) if np.max(values) > np.min(values) else np.ones_like(values)
                else:
                    # 越大越好
                    normalized_scores[criterion] = (values - np.min(values)) / (np.max(values) - np.min(values)) if np.max(values) > np.min(values) else np.ones_like(values)
        
        # 計算綜合評分
        composite_scores = np.mean([normalized_scores[criterion] for criterion in optimization_criteria if criterion in normalized_scores], axis=0)
        results_df['composite_score'] = composite_scores
        
        # 排序
        results_df = results_df.sort_values('composite_score', ascending=False)
        
        return results_df
    
    def extract_parametric_indices(self,
                                 index_type: ParametricIndexType,
                                 hazard_data: Any,
                                 exposure_locations: Any,
                                 **kwargs) -> np.ndarray:
        """
        提取參數指標
        
        Parameters:
        -----------
        index_type : ParametricIndexType
            指標類型
        hazard_data : Any
            災害數據
        exposure_locations : Any
            曝險位置
        **kwargs : dict
            額外參數
            
        Returns:
        --------
        np.ndarray
            參數指標
        """
        
        if index_type not in self.index_extractors:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        extractor = self.index_extractors[index_type]
        
        # 更新提取器參數
        for key, value in kwargs.items():
            if hasattr(extractor, key):
                setattr(extractor, key, value)
        
        return extractor.extract_indices(hazard_data, exposure_locations)
    
    def calculate_technical_premium(self,
                                  product: ParametricProduct,
                                  performance: ProductPerformance,
                                  risk_load_factor: float = 0.20,
                                  expense_load_factor: float = 0.15) -> Dict[str, float]:
        """
        計算技術保費
        
        Parameters:
        -----------
        product : ParametricProduct
            保險產品
        performance : ProductPerformance
            產品績效
        risk_load_factor : float
            風險附加費率
        expense_load_factor : float
            費用附加費率
            
        Returns:
        --------
        Dict[str, float]
            保費組成
        """
        
        # 年平均損失 (AAL)
        aal = performance.technical_metrics.get('average_payout', 0) * performance.technical_metrics.get('payout_frequency', 0)
        
        # 風險附加費
        risk_load = aal * risk_load_factor
        
        # 費用附加費
        expense_load = aal * expense_load_factor
        
        # 技術保費
        technical_premium = aal + risk_load + expense_load
        
        return {
            'aal': aal,
            'risk_load': risk_load,
            'expense_load': expense_load,
            'technical_premium': technical_premium,
            'risk_load_factor': risk_load_factor,
            'expense_load_factor': expense_load_factor
        }