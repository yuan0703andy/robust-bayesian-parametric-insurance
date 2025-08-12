#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hospital-Based Payout Configuration
醫院基礎賠付配置

This module configures maximum payouts based on hospital exposure values
本模組基於醫院曝險值配置最大賠付
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class HospitalPayoutConfig:
    """
    醫院基礎賠付配置
    Hospital-based payout configuration
    """
    
    # Hospital value parameters 醫院價值參數
    n_hospitals: int = 20  # 醫院數量
    base_hospital_value: float = 1e7  # 每家醫院基礎價值 ($10M USD)
    
    # Hospital type multipliers 醫院類型乘數
    hospital_type_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'general': 1.0,        # 一般醫院
        'emergency': 2.0,      # 急救中心
        'specialty': 1.5,      # 專科醫院
        'regional': 2.5,       # 區域醫療中心
        'university': 3.0,     # 大學醫院
        'community': 0.8       # 社區醫院
    })
    
    # Payout coverage ratios 賠付覆蓋比例
    coverage_ratios: Dict[str, float] = field(default_factory=lambda: {
        'single': 0.25,      # 單閾值: 25% 總曝險
        'double': 0.40,      # 雙閾值: 40% 總曝險
        'triple': 0.60,      # 三閾值: 60% 總曝險
        'quadruple': 0.80    # 四閾值: 80% 總曝險
    })
    
    # Radius-based adjustments 半徑基礎調整
    radius_multipliers: Dict[int, float] = field(default_factory=lambda: {
        15: 1.5,   # 15km: 局部影響，高賠付密度
        30: 1.2,   # 30km: 標準影響範圍
        50: 1.0,   # 50km: 基準範圍
        75: 0.9,   # 75km: 較大範圍，降低密度
        100: 0.8   # 100km: 區域影響，最低密度
    })
    
    # Dynamic adjustment parameters 動態調整參數
    enable_dynamic_adjustment: bool = True
    min_payout_ratio: float = 0.1  # 最小賠付比例
    max_payout_ratio: float = 1.0  # 最大賠付比例
    
    def calculate_total_exposure(self, 
                                  hospital_df=None,
                                  use_type_multipliers: bool = True) -> float:
        """
        計算總曝險值
        Calculate total exposure value
        
        Parameters:
        -----------
        hospital_df : DataFrame, optional
            包含醫院資訊的數據框
        use_type_multipliers : bool
            是否使用醫院類型乘數
            
        Returns:
        --------
        float : 總曝險值
        """
        if hospital_df is None:
            # 使用預設值
            return self.n_hospitals * self.base_hospital_value
        
        total_value = 0
        for _, hospital in hospital_df.iterrows():
            base_value = self.base_hospital_value
            
            if use_type_multipliers and 'hospital_type' in hospital:
                multiplier = self.hospital_type_multipliers.get(
                    hospital['hospital_type'], 1.0
                )
                base_value *= multiplier
            
            total_value += base_value
        
        return total_value
    
    def get_max_payout_amounts(self, 
                                total_exposure: Optional[float] = None,
                                radius_km: Optional[int] = None) -> Dict[str, float]:
        """
        獲取最大賠付金額配置
        Get maximum payout amounts configuration
        
        Parameters:
        -----------
        total_exposure : float, optional
            總曝險值，如果未提供則自動計算
        radius_km : int, optional
            分析半徑（公里）
            
        Returns:
        --------
        Dict[str, float] : 各產品類型的最大賠付金額
        """
        if total_exposure is None:
            total_exposure = self.calculate_total_exposure()
        
        # 基礎賠付金額
        max_payouts = {}
        for product_type, coverage_ratio in self.coverage_ratios.items():
            max_payouts[product_type] = total_exposure * coverage_ratio
        
        # 應用半徑調整
        if radius_km and radius_km in self.radius_multipliers:
            radius_mult = self.radius_multipliers[radius_km]
            for product_type in max_payouts:
                max_payouts[product_type] *= radius_mult
        
        return max_payouts
    
    def calculate_dynamic_payout(self,
                                  affected_hospitals: List[Dict],
                                  total_hospitals: List[Dict],
                                  base_payout: float) -> float:
        """
        基於受影響醫院計算動態賠付
        Calculate dynamic payout based on affected hospitals
        
        Parameters:
        -----------
        affected_hospitals : List[Dict]
            受影響的醫院列表
        total_hospitals : List[Dict]
            所有醫院列表
        base_payout : float
            基礎賠付金額
            
        Returns:
        --------
        float : 調整後的賠付金額
        """
        if not self.enable_dynamic_adjustment:
            return base_payout
        
        # 計算受影響比例
        impact_ratio = len(affected_hospitals) / len(total_hospitals)
        
        # 計算受影響醫院的加權價值
        affected_value = sum(
            self.base_hospital_value * self.hospital_type_multipliers.get(
                h.get('hospital_type', 'general'), 1.0
            )
            for h in affected_hospitals
        )
        
        total_value = sum(
            self.base_hospital_value * self.hospital_type_multipliers.get(
                h.get('hospital_type', 'general'), 1.0
            )
            for h in total_hospitals
        )
        
        value_ratio = affected_value / total_value if total_value > 0 else 0
        
        # 綜合調整因子
        adjustment_factor = (impact_ratio + value_ratio) / 2
        adjustment_factor = np.clip(
            adjustment_factor, 
            self.min_payout_ratio, 
            self.max_payout_ratio
        )
        
        return base_payout * adjustment_factor
    
    def get_steinmann_config_with_hospital_payouts(self, 
                                                    hospital_df=None,
                                                    radius_km: int = 50):
        """
        獲取整合醫院賠付的Steinmann配置
        Get Steinmann configuration with hospital-based payouts
        
        Parameters:
        -----------
        hospital_df : DataFrame, optional
            醫院數據
        radius_km : int
            分析半徑
            
        Returns:
        --------
        SteinmannProductConfig : 配置對象
        """
        from insurance_analysis_refactored.core.saffir_simpson_products import SteinmannProductConfig
        
        # 計算基於醫院的最大賠付
        total_exposure = self.calculate_total_exposure(hospital_df)
        max_payouts = self.get_max_payout_amounts(total_exposure, radius_km)
        
        # 創建Steinmann配置
        config = SteinmannProductConfig(
            max_payout_amounts=max_payouts,
            use_saffir_simpson_thresholds=True,
            payout_increments=[0.25, 0.50, 0.75, 1.00]
        )
        
        print(f"📊 醫院基礎賠付配置:")
        print(f"   • 醫院數量: {self.n_hospitals}")
        print(f"   • 每家醫院基礎價值: ${self.base_hospital_value:,.0f}")
        print(f"   • 總曝險值: ${total_exposure:,.0f}")
        print(f"   • 分析半徑: {radius_km}km")
        print(f"   • 最大賠付金額:")
        for ptype, amount in max_payouts.items():
            print(f"     - {ptype}: ${amount:,.0f}")
        
        return config


def create_hospital_based_config(hospital_data=None, 
                                  n_hospitals: int = 20,
                                  base_value_per_hospital: float = 1e7) -> HospitalPayoutConfig:
    """
    創建基於醫院的賠付配置
    Create hospital-based payout configuration
    
    Parameters:
    -----------
    hospital_data : DataFrame or dict, optional
        醫院數據
    n_hospitals : int
        醫院數量
    base_value_per_hospital : float
        每家醫院基礎價值
        
    Returns:
    --------
    HospitalPayoutConfig : 配置對象
    """
    config = HospitalPayoutConfig(
        n_hospitals=n_hospitals,
        base_hospital_value=base_value_per_hospital
    )
    
    return config


# Example usage 使用範例
if __name__ == "__main__":
    # 創建配置
    config = create_hospital_based_config(
        n_hospitals=20,
        base_value_per_hospital=1e7  # $10M per hospital
    )
    
    # 獲取最大賠付金額
    max_payouts = config.get_max_payout_amounts(radius_km=30)
    
    print("\n🏥 Hospital-Based Payout Configuration:")
    print(f"Total Exposure: ${config.calculate_total_exposure():,.0f}")
    print("\nMax Payout Amounts (30km radius):")
    for product_type, amount in max_payouts.items():
        print(f"  {product_type}: ${amount:,.0f}")