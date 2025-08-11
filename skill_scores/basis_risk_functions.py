"""
Basis Risk Functions for Parametric Insurance
參數型保險基差風險函數

提供各種基差風險的數學定義和計算方法，從 bayesian_decision_theory.py 遷移而來
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import warnings

class BasisRiskType(Enum):
    """基差風險類型"""
    ABSOLUTE = "absolute"                    # 絕對基差風險
    ASYMMETRIC = "asymmetric"               # 不對稱基差風險 (統一命名)
    ASYMMETRIC_UNDER = "asymmetric_under"   # 不對稱基差風險 (向後兼容)
    WEIGHTED_ASYMMETRIC = "weighted_asymmetric"  # 加權不對稱基差風險
    QUADRATIC = "quadratic"                # 二次基差風險
    RMSE = "rmse"                          # 傳統RMSE
    MAE = "mae"                            # 平均絕對誤差
    RELATIVE_ABSOLUTE = "relative_absolute"  # 相對絕對基差風險
    RELATIVE_WEIGHTED_ASYMMETRIC = "relative_weighted_asymmetric"  # 相對加權不對稱基差風險
    
@dataclass
class BasisRiskConfig:
    """基差風險計算配置"""
    risk_type: BasisRiskType = BasisRiskType.ABSOLUTE
    w_under: float = 2.0    # 賠不夠的懲罰權重 (undercompensation penalty)
    w_over: float = 0.5     # 賠多了的懲罰權重 (overcompensation penalty)
    normalize: bool = True   # 是否標準化
    min_loss_threshold: float = 1e6  # 相對基差風險的最小損失閾值

@dataclass 
class BasisRiskLossFunction:
    """基差風險損失函數"""
    risk_type: BasisRiskType
    w_under: float = 1.0     # 賠不夠的懲罰權重
    w_over: float = 0.3      # 賠多了的懲罰權重
    use_relative: bool = False  # 是否使用相對基差風險
    min_loss_threshold: float = 1e6  # 相對基差風險的最小損失閾值
    
    def calculate_loss(self, actual_loss: float, payout: float) -> float:
        """計算基差風險損失"""
        
        if self.risk_type == BasisRiskType.ABSOLUTE:
            # 絕對基差風險
            return abs(actual_loss - payout)
            
        elif self.risk_type in [BasisRiskType.ASYMMETRIC_UNDER, BasisRiskType.ASYMMETRIC]:
            # 不對稱基差風險 (只懲罰賠不夠)
            return max(0, actual_loss - payout)
            
        elif self.risk_type == BasisRiskType.WEIGHTED_ASYMMETRIC:
            # 加權不對稱基差風險
            under_coverage = max(0, actual_loss - payout)
            over_coverage = max(0, payout - actual_loss)
            return self.w_under * under_coverage + self.w_over * over_coverage
            
        elif self.risk_type == BasisRiskType.QUADRATIC:
            # 二次基差風險
            return (actual_loss - payout) ** 2
            
        elif self.risk_type == BasisRiskType.RELATIVE_ABSOLUTE:
            # 相對絕對基差風險: |actual - payout| / max(actual, threshold)
            denominator = max(actual_loss, self.min_loss_threshold)
            return abs(actual_loss - payout) / denominator
            
        elif self.risk_type == BasisRiskType.RELATIVE_WEIGHTED_ASYMMETRIC:
            # 相對加權不對稱基差風險
            if actual_loss < self.min_loss_threshold:
                # 對於小損失，使用絕對基差風險
                under_coverage = max(0, actual_loss - payout)
                over_coverage = max(0, payout - actual_loss)
                return self.w_under * under_coverage + self.w_over * over_coverage
            else:
                # 對於大損失，使用相對基差風險
                under_coverage = max(0, actual_loss - payout) / actual_loss
                over_coverage = max(0, payout - actual_loss) / actual_loss
                return self.w_under * under_coverage + self.w_over * over_coverage
            
        else:
            raise ValueError(f"Unsupported risk type: {self.risk_type}")

class BasisRiskCalculator:
    """
    基差風險計算器
    
    實現多種基差風險定義和計算方法，支持傳統和貝葉斯框架。
    整合來自 basis_risk_calculator.py 和原始 skill_scores/basis_risk_functions.py 的功能。
    """
    
    def __init__(self, config: BasisRiskConfig = None):
        """
        初始化基差風險計算器
        
        Parameters:
        -----------
        config : BasisRiskConfig
            基差風險計算配置
        """
        self.config = config or BasisRiskConfig()
        
        # 損失函數映射
        self.loss_functions = {
            BasisRiskType.ABSOLUTE: self._absolute_basis_risk,
            BasisRiskType.ASYMMETRIC: self._asymmetric_basis_risk,
            BasisRiskType.ASYMMETRIC_UNDER: self._asymmetric_basis_risk,  # 向後兼容
            BasisRiskType.WEIGHTED_ASYMMETRIC: self._weighted_asymmetric_basis_risk,
            BasisRiskType.RMSE: self._rmse_loss,
            BasisRiskType.MAE: self._mae_loss,
            BasisRiskType.RELATIVE_ABSOLUTE: self._relative_absolute_basis_risk,
            BasisRiskType.RELATIVE_WEIGHTED_ASYMMETRIC: self._relative_weighted_asymmetric_basis_risk
        }
    
    def calculate_basis_risk(self, 
                           actual_losses: np.ndarray, 
                           payouts: np.ndarray,
                           risk_type: BasisRiskType = None) -> Union[float, np.ndarray]:
        """
        計算基差風險
        
        Parameters:
        -----------
        actual_losses : np.ndarray
            實際損失
        payouts : np.ndarray  
            保險賠付
        risk_type : BasisRiskType, optional
            基差風險類型，如果None則使用配置中的類型
            
        Returns:
        --------
        Union[float, np.ndarray]
            基差風險值
        """
        if risk_type is None:
            risk_type = self.config.risk_type
            
        loss_function = self.loss_functions[risk_type]
        return loss_function(actual_losses, payouts)
    
    def _absolute_basis_risk(self, actual_losses: np.ndarray, payouts: np.ndarray) -> Union[float, np.ndarray]:
        """
        定義一：絕對基差風險 (Absolute Basis Risk)
        L(θ, a) = | Actual_Loss(θ) - Payout(a) |
        
        最簡單的定義，計算實際損失和保險賠付之間的絕對差距。
        """
        basis_risk = np.abs(actual_losses - payouts)
        
        if self.config.normalize and np.sum(actual_losses) > 0:
            # 標準化：相對於總實際損失的比例
            basis_risk = basis_risk / np.mean(actual_losses)
            
        return np.mean(basis_risk) if basis_risk.ndim > 0 else basis_risk
    
    def _asymmetric_basis_risk(self, actual_losses: np.ndarray, payouts: np.ndarray) -> Union[float, np.ndarray]:
        """
        定義二：不對稱基差風險 (Asymmetric Basis Risk)
        L(θ, a) = max(0, Actual_Loss(θ) - Payout(a))
        
        只計算「賠不夠」的情況，即未被覆蓋的損失 (uncovered loss)。
        這通常是保險設計中最關心的風險。
        """
        # 只考慮賠不夠的情況 (undercompensation)
        uncovered_loss = np.maximum(0, actual_losses - payouts)
        
        if self.config.normalize and np.sum(actual_losses) > 0:
            uncovered_loss = uncovered_loss / np.mean(actual_losses)
            
        return np.mean(uncovered_loss) if uncovered_loss.ndim > 0 else uncovered_loss
    
    def _weighted_asymmetric_basis_risk(self, actual_losses: np.ndarray, payouts: np.ndarray) -> Union[float, np.ndarray]:
        """
        定義三：加權不對稱基差風險 (Weighted Asymmetric Basis Risk)
        L(θ, a) = w_under * max(0, Actual_Loss(θ) - Payout(a)) + w_over * max(0, Payout(a) - Actual_Loss(θ))
        
        同時考慮「賠不夠」和「賠多了」兩種情況，並給予不同的懲罰權重。
        """
        # 賠不夠的損失 (undercompensation)
        uncovered_loss = np.maximum(0, actual_losses - payouts)
        
        # 賠多了的損失 (overcompensation)  
        excess_payout = np.maximum(0, payouts - actual_losses)
        
        # 加權組合
        weighted_loss = (self.config.w_under * uncovered_loss + 
                        self.config.w_over * excess_payout)
        
        if self.config.normalize and np.sum(actual_losses) > 0:
            weighted_loss = weighted_loss / np.mean(actual_losses)
            
        return np.mean(weighted_loss) if weighted_loss.ndim > 0 else weighted_loss
    
    def _rmse_loss(self, actual_losses: np.ndarray, payouts: np.ndarray) -> float:
        """傳統RMSE損失函數"""
        return np.sqrt(np.mean((actual_losses - payouts) ** 2))
    
    def _mae_loss(self, actual_losses: np.ndarray, payouts: np.ndarray) -> float:
        """平均絕對誤差損失函數"""
        return np.mean(np.abs(actual_losses - payouts))
    
    def _relative_absolute_basis_risk(self, actual_losses: np.ndarray, payouts: np.ndarray) -> Union[float, np.ndarray]:
        """相對絕對基差風險"""
        risks = []
        for actual, payout in zip(actual_losses, payouts):
            denominator = max(actual, self.config.min_loss_threshold)
            risk = abs(actual - payout) / denominator
            risks.append(risk)
        return np.mean(risks)
    
    def _relative_weighted_asymmetric_basis_risk(self, actual_losses: np.ndarray, payouts: np.ndarray) -> Union[float, np.ndarray]:
        """相對加權不對稱基差風險"""
        risks = []
        for actual, payout in zip(actual_losses, payouts):
            if actual < self.config.min_loss_threshold:
                # 小損失使用絕對基差風險
                under_coverage = max(0, actual - payout)
                over_coverage = max(0, payout - actual)
                risk = self.config.w_under * under_coverage + self.config.w_over * over_coverage
            else:
                # 大損失使用相對基差風險
                under_coverage_pct = max(0, actual - payout) / actual
                over_coverage_pct = max(0, payout - actual) / actual
                risk = self.config.w_under * under_coverage_pct + self.config.w_over * over_coverage_pct
            risks.append(risk)
        return np.mean(risks)
    
    def calculate_expected_basis_risk(self,
                                    posterior_samples: List[np.ndarray],
                                    payout_function: Callable,
                                    wind_indices: np.ndarray,
                                    risk_type: BasisRiskType = None) -> float:
        """
        計算期望基差風險 (Bayesian Decision Theory)
        
        對於給定的產品設計，使用後驗樣本計算期望基差風險。
        
        Parameters:
        -----------
        posterior_samples : List[np.ndarray]
            後驗損失分佈樣本列表
        payout_function : Callable
            賠付函數 (接受風速指標，返回賠付)
        wind_indices : np.ndarray
            風速指標
        risk_type : BasisRiskType, optional
            基差風險類型
            
        Returns:
        --------
        float
            期望基差風險
        """
        if risk_type is None:
            risk_type = self.config.risk_type
            
        total_risk = 0.0
        n_samples = len(posterior_samples)
        
        # 計算賠付
        payouts = np.array([payout_function(wind_speed) for wind_speed in wind_indices])
        
        # 對每個後驗樣本計算基差風險
        for posterior_sample in posterior_samples:
            # 確保長度匹配
            min_length = min(len(posterior_sample), len(payouts))
            sample_losses = posterior_sample[:min_length]
            sample_payouts = payouts[:min_length]
            
            # 計算此樣本的基差風險
            risk = self.calculate_basis_risk(sample_losses, sample_payouts, risk_type)
            total_risk += risk
        
        # 返回期望基差風險
        return total_risk / n_samples if n_samples > 0 else 0.0
    
    def find_optimal_product(self,
                            products: List[Dict],
                            posterior_samples: List[np.ndarray],
                            wind_indices: np.ndarray,
                            risk_type: BasisRiskType = None) -> Tuple[Dict, float, List[float]]:
        """
        找到期望基差風險最小的最佳產品 (Bayesian Optimization)
        
        Parameters:
        -----------
        products : List[Dict]
            候選產品列表
        posterior_samples : List[np.ndarray]
            後驗損失分佈樣本
        wind_indices : np.ndarray
            風速指標
        risk_type : BasisRiskType, optional
            基差風險類型
            
        Returns:
        --------
        Tuple[Dict, float, List[float]]
            (最佳產品, 最小期望風險, 所有產品的期望風險列表)
        """
        if risk_type is None:
            risk_type = self.config.risk_type
            
        print(f"🔍 貝葉斯產品優化 - 風險類型: {risk_type.value}")
        print(f"   候選產品數: {len(products)}")
        print(f"   後驗樣本數: {len(posterior_samples)}")
        
        expected_risks = []
        
        for i, product in enumerate(products):
            if (i + 1) % 10 == 0:
                print(f"   進度: {i+1}/{len(products)}")
            
            # 創建賠付函數
            def payout_function(wind_speed):
                return self._calculate_step_payout(
                    wind_speed,
                    product['trigger_thresholds'],
                    product['payout_ratios'],
                    product['max_payout']
                )
            
            # 計算期望基差風險
            expected_risk = self.calculate_expected_basis_risk(
                posterior_samples, payout_function, wind_indices, risk_type
            )
            
            expected_risks.append(expected_risk)
        
        # 找到最小期望風險
        min_risk_idx = np.argmin(expected_risks)
        optimal_product = products[min_risk_idx]
        min_expected_risk = expected_risks[min_risk_idx]
        
        print(f"✅ 最佳產品: {optimal_product['product_id']}")
        print(f"   最小期望{risk_type.value}風險: {min_expected_risk:.6f}")
        
        return optimal_product, min_expected_risk, expected_risks
    
    def _calculate_step_payout(self, wind_speed: float, thresholds: List[float], 
                              ratios: List[float], max_payout: float) -> float:
        """計算階梯式賠付"""
        payout = 0.0
        for i in range(len(thresholds) - 1, -1, -1):
            if wind_speed >= thresholds[i]:
                payout = ratios[i] * max_payout
                break
        return payout
    
    def generate_comprehensive_analysis(self,
                                      products: List[Dict],
                                      actual_losses: np.ndarray,
                                      wind_indices: np.ndarray,
                                      posterior_samples: Optional[List[np.ndarray]] = None) -> Dict:
        """
        生成全面的基差風險分析報告
        
        Parameters:
        -----------
        products : List[Dict]
            產品列表
        actual_losses : np.ndarray
            實際損失數據
        wind_indices : np.ndarray
            風速指標
        posterior_samples : List[np.ndarray], optional
            後驗樣本（用於貝葉斯分析）
            
        Returns:
        --------
        Dict
            綜合分析結果
        """
        print("📊 生成全面基差風險分析...")
        
        results = {
            'traditional_analysis': {},
            'bayesian_analysis': {},
            'comparison': {},
            'rankings': {}
        }
        
        # 傳統分析
        print("   🔄 傳統基差風險分析...")
        traditional_results = []
        
        for product in products:
            # 計算賠付
            payouts = np.array([
                self._calculate_step_payout(
                    wind_speed,
                    product['trigger_thresholds'],
                    product['payout_ratios'], 
                    product['max_payout']
                ) for wind_speed in wind_indices
            ])
            
            # 確保長度匹配
            min_length = min(len(actual_losses), len(payouts))
            losses = actual_losses[:min_length]
            pays = payouts[:min_length]
            
            # 計算不同類型的基差風險
            product_result = {
                'product_id': product['product_id'],
                'structure_type': product['structure_type'],
                'absolute_risk': self.calculate_basis_risk(losses, pays, BasisRiskType.ABSOLUTE),
                'asymmetric_risk': self.calculate_basis_risk(losses, pays, BasisRiskType.ASYMMETRIC),
                'weighted_risk': self.calculate_basis_risk(losses, pays, BasisRiskType.WEIGHTED_ASYMMETRIC),
                'rmse': self.calculate_basis_risk(losses, pays, BasisRiskType.RMSE),
                'mae': self.calculate_basis_risk(losses, pays, BasisRiskType.MAE),
                'correlation': np.corrcoef(losses, pays)[0,1] if np.std(pays) > 0 else 0,
                'trigger_rate': np.mean(pays > 0),
                'mean_payout': np.mean(pays),
                'coverage_ratio': np.sum(pays) / np.sum(losses) if np.sum(losses) > 0 else 0
            }
            
            traditional_results.append(product_result)
        
        results['traditional_analysis'] = traditional_results
        
        # 貝葉斯分析 (如果有後驗樣本)
        if posterior_samples is not None:
            print("   🧠 貝葉斯基差風險分析...")
            
            # 對不同風險類型進行優化
            bayesian_results = {}
            for risk_type in [BasisRiskType.ABSOLUTE, BasisRiskType.ASYMMETRIC, BasisRiskType.WEIGHTED_ASYMMETRIC]:
                optimal_product, min_risk, all_risks = self.find_optimal_product(
                    products, posterior_samples, wind_indices, risk_type
                )
                
                bayesian_results[risk_type.value] = {
                    'optimal_product': optimal_product,
                    'min_expected_risk': min_risk,
                    'all_expected_risks': all_risks
                }
            
            results['bayesian_analysis'] = bayesian_results
        
        # 排名分析
        print("   📈 生成排名分析...")
        rankings = {}
        
        # 傳統排名
        for metric in ['absolute_risk', 'asymmetric_risk', 'weighted_risk', 'rmse']:
            sorted_products = sorted(traditional_results, key=lambda x: x[metric])
            rankings[f'traditional_{metric}'] = [
                (p['product_id'], p[metric]) for p in sorted_products[:10]
            ]
        
        results['rankings'] = rankings
        
        print("✅ 全面基差風險分析完成")
        return results

    @staticmethod
    def calculate_absolute_basis_risk(actual_loss: float, payout: float) -> float:
        """絕對基差風險: |actual - payout|"""
        return abs(actual_loss - payout)
    
    @staticmethod 
    def calculate_asymmetric_basis_risk(actual_loss: float, payout: float) -> float:
        """不對稱基差風險: max(0, actual - payout)"""
        return max(0, actual_loss - payout)
    
    @staticmethod
    def calculate_weighted_asymmetric_basis_risk(
        actual_loss: float, 
        payout: float,
        w_under: float = 2.0,
        w_over: float = 0.5
    ) -> float:
        """加權不對稱基差風險"""
        under_coverage = max(0, actual_loss - payout)
        over_coverage = max(0, payout - actual_loss)
        return w_under * under_coverage + w_over * over_coverage
    
    @staticmethod
    def calculate_quadratic_basis_risk(actual_loss: float, payout: float) -> float:
        """二次基差風險: (actual - payout)²"""
        return (actual_loss - payout) ** 2
    
    @staticmethod
    def calculate_relative_absolute_basis_risk(
        actual_loss: float, 
        payout: float,
        min_loss_threshold: float = 1e6
    ) -> float:
        """
        相對絕對基差風險: |actual - payout| / max(actual, threshold)
        
        Parameters:
        -----------
        actual_loss : float
            真實損失
        payout : float
            賠付金額
        min_loss_threshold : float
            最小損失閾值，避免除以過小的數值
            
        Returns:
        --------
        float
            相對基差風險值 (0-1 區間內為正常範圍)
        """
        denominator = max(actual_loss, min_loss_threshold)
        return abs(actual_loss - payout) / denominator
    
    @staticmethod
    def calculate_relative_weighted_asymmetric_basis_risk(
        actual_loss: float, 
        payout: float,
        w_under: float = 2.0,
        w_over: float = 0.5,
        min_loss_threshold: float = 1e6
    ) -> float:
        """
        相對加權不對稱基差風險
        
        對於大損失: 使用相對基差風險 = (w_under * under% + w_over * over%)
        對於小損失: 使用絕對基差風險，避免相對值過大
        
        Parameters:
        -----------
        actual_loss : float
            真實損失
        payout : float  
            賠付金額
        w_under, w_over : float
            懲罰權重
        min_loss_threshold : float
            切換到相對計算的最小損失閾值
            
        Returns:
        --------
        float
            相對基差風險值
        """
        if actual_loss < min_loss_threshold:
            # 小損失使用絕對基差風險
            under_coverage = max(0, actual_loss - payout)
            over_coverage = max(0, payout - actual_loss)
            return w_under * under_coverage + w_over * over_coverage
        else:
            # 大損失使用相對基差風險
            under_coverage_pct = max(0, actual_loss - payout) / actual_loss
            over_coverage_pct = max(0, payout - actual_loss) / actual_loss
            return w_under * under_coverage_pct + w_over * over_coverage_pct
    
    def calculate_portfolio_basis_risk(self,
                                     actual_losses: np.ndarray,
                                     payouts: np.ndarray,
                                     risk_type: BasisRiskType = BasisRiskType.WEIGHTED_ASYMMETRIC,
                                     **risk_params) -> Dict[str, float]:
        """
        計算投資組合基差風險統計
        
        Parameters:
        -----------
        actual_losses : np.ndarray
            真實損失陣列
        payouts : np.ndarray
            賠付陣列
        risk_type : BasisRiskType
            基差風險類型
        **risk_params : dict
            風險參數 (如 w_under, w_over)
            
        Returns:
        --------
        Dict[str, float]
            基差風險統計結果
        """
        
        if len(actual_losses) != len(payouts):
            raise ValueError("actual_losses 和 payouts 長度必須相等")
        
        # 計算個別基差風險
        individual_risks = []
        
        for actual, payout in zip(actual_losses, payouts):
            if risk_type == BasisRiskType.ABSOLUTE:
                risk = self.calculate_absolute_basis_risk(actual, payout)
            elif risk_type == BasisRiskType.ASYMMETRIC_UNDER:
                risk = self.calculate_asymmetric_basis_risk(actual, payout)
            elif risk_type == BasisRiskType.WEIGHTED_ASYMMETRIC:
                risk = self.calculate_weighted_asymmetric_basis_risk(
                    actual, payout, 
                    w_under=risk_params.get('w_under', 2.0),
                    w_over=risk_params.get('w_over', 0.5)
                )
            elif risk_type == BasisRiskType.QUADRATIC:
                risk = self.calculate_quadratic_basis_risk(actual, payout)
            elif risk_type == BasisRiskType.RELATIVE_ABSOLUTE:
                risk = self.calculate_relative_absolute_basis_risk(
                    actual, payout,
                    min_loss_threshold=risk_params.get('min_loss_threshold', 1e6)
                )
            elif risk_type == BasisRiskType.RELATIVE_WEIGHTED_ASYMMETRIC:
                risk = self.calculate_relative_weighted_asymmetric_basis_risk(
                    actual, payout,
                    w_under=risk_params.get('w_under', 2.0),
                    w_over=risk_params.get('w_over', 0.5),
                    min_loss_threshold=risk_params.get('min_loss_threshold', 1e6)
                )
            else:
                raise ValueError(f"Unsupported risk type: {risk_type}")
            
            individual_risks.append(risk)
        
        individual_risks = np.array(individual_risks)
        
        # 計算統計指標
        stats = {
            'mean_basis_risk': float(np.mean(individual_risks)),
            'median_basis_risk': float(np.median(individual_risks)),
            'std_basis_risk': float(np.std(individual_risks)),
            'max_basis_risk': float(np.max(individual_risks)),
            'min_basis_risk': float(np.min(individual_risks)),
            'total_basis_risk': float(np.sum(individual_risks)),
            'basis_risk_95th_percentile': float(np.percentile(individual_risks, 95)),
            'basis_risk_5th_percentile': float(np.percentile(individual_risks, 5))
        }
        
        return stats
    
    def analyze_coverage_breakdown(self,
                                 actual_losses: np.ndarray,
                                 payouts: np.ndarray) -> Dict[str, Any]:
        """
        分析保險覆蓋情況分解
        
        Returns:
        --------
        Dict[str, Any]
            覆蓋情況分析結果
        """
        
        n_total = len(actual_losses)
        
        # 分類事件
        perfect_match = np.isclose(actual_losses, payouts, rtol=1e-3)
        under_coverage = actual_losses > payouts
        over_coverage = actual_losses < payouts
        no_loss_no_payout = (actual_losses == 0) & (payouts == 0)
        loss_no_payout = (actual_losses > 0) & (payouts == 0)
        no_loss_payout = (actual_losses == 0) & (payouts > 0)
        
        # 計算覆蓋統計
        under_coverage_amount = np.sum(np.maximum(0, actual_losses - payouts))
        over_coverage_amount = np.sum(np.maximum(0, payouts - actual_losses))
        
        breakdown = {
            'total_events': n_total,
            'perfect_match_count': int(np.sum(perfect_match)),
            'under_coverage_count': int(np.sum(under_coverage)),
            'over_coverage_count': int(np.sum(over_coverage)),
            'no_loss_no_payout_count': int(np.sum(no_loss_no_payout)),
            'loss_no_payout_count': int(np.sum(loss_no_payout)),
            'no_loss_payout_count': int(np.sum(no_loss_payout)),
            
            'perfect_match_rate': float(np.mean(perfect_match)),
            'under_coverage_rate': float(np.mean(under_coverage)),
            'over_coverage_rate': float(np.mean(over_coverage)),
            'trigger_rate': float(np.mean(payouts > 0)),
            'loss_rate': float(np.mean(actual_losses > 0)),
            
            'total_under_coverage_amount': float(under_coverage_amount),
            'total_over_coverage_amount': float(over_coverage_amount),
            'average_under_coverage': float(under_coverage_amount / max(1, np.sum(under_coverage))),
            'average_over_coverage': float(over_coverage_amount / max(1, np.sum(over_coverage))),
            
            'coverage_efficiency': float(1 - (under_coverage_amount + over_coverage_amount) / max(1, np.sum(actual_losses)))
        }
        
        return breakdown

def create_basis_risk_function(risk_type: str = "weighted_asymmetric",
                             w_under: float = 2.0,
                             w_over: float = 0.5,
                             use_relative: bool = False,
                             min_loss_threshold: float = 1e6) -> BasisRiskLossFunction:
    """
    創建基差風險損失函數的便利函數
    
    Parameters:
    -----------
    risk_type : str
        風險類型 ("absolute", "asymmetric_under", "weighted_asymmetric", "quadratic",
                 "relative_absolute", "relative_weighted_asymmetric")
    w_under : float
        賠不夠的懲罰權重
    w_over : float
        賠多了的懲罰權重
    use_relative : bool
        是否使用相對基差風險 (會自動映射到相對類型)
    min_loss_threshold : float
        相對基差風險的最小損失閾值
        
    Returns:
    --------
    BasisRiskLossFunction
        基差風險損失函數
    """
    
    # 如果使用相對模式，自動映射風險類型
    if use_relative:
        if risk_type == "absolute":
            risk_type = "relative_absolute"
        elif risk_type == "weighted_asymmetric":
            risk_type = "relative_weighted_asymmetric"
    
    risk_type_enum = BasisRiskType(risk_type)
    
    return BasisRiskLossFunction(
        risk_type=risk_type_enum,
        w_under=w_under,
        w_over=w_over,
        use_relative=use_relative,
        min_loss_threshold=min_loss_threshold
    )


def calculate_step_payouts_batch(wind_speeds: np.ndarray, 
                               thresholds: List[float],
                               ratios: List[float], 
                               max_payout: float) -> np.ndarray:
    """
    批量計算階梯式賠付
    
    Parameters:
    -----------
    wind_speeds : np.ndarray
        風速數組
    thresholds : List[float]
        觸發閾值列表
    ratios : List[float]
        賠付比例列表
    max_payout : float
        最大賠付金額
        
    Returns:
    --------
    np.ndarray
        賠付金額數組
    """
    payouts = np.zeros(len(wind_speeds))
    
    for i, wind_speed in enumerate(wind_speeds):
        for j in range(len(thresholds) - 1, -1, -1):
            if wind_speed >= thresholds[j]:
                payouts[i] = ratios[j] * max_payout
                break
    
    return payouts