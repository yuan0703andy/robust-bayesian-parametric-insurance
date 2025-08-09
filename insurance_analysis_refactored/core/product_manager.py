"""
Insurance Product Manager
保險產品管理器

This module provides high-level product management functionality,
including product lifecycle management, comparison, and reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle

from .parametric_engine import ParametricProduct, ProductPerformance

class ProductStatus(Enum):
    """產品狀態"""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"

@dataclass
class ProductPortfolio:
    """產品組合"""
    portfolio_id: str
    name: str
    description: str
    products: List[ParametricProduct]
    weights: List[float]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    creation_date: str = ""
    status: ProductStatus = ProductStatus.DRAFT

class InsuranceProductManager:
    """
    保險產品管理器
    
    提供產品生命週期管理、比較和報告功能。
    """
    
    def __init__(self):
        self.products = {}
        self.portfolios = {}
        self.performance_history = {}
        self.metadata_store = {}
    
    def register_product(self, product: ParametricProduct, status: ProductStatus = ProductStatus.DRAFT) -> None:
        """
        註冊產品
        
        Parameters:
        -----------
        product : ParametricProduct
            要註冊的產品
        status : ProductStatus
            產品狀態
        """
        self.products[product.product_id] = product
        self.metadata_store[product.product_id] = {
            'status': status,
            'creation_date': pd.Timestamp.now().isoformat(),
            'last_updated': pd.Timestamp.now().isoformat()
        }
    
    def update_product_performance(self, product_id: str, performance: ProductPerformance) -> None:
        """
        更新產品績效
        
        Parameters:
        -----------
        product_id : str
            產品ID
        performance : ProductPerformance
            績效數據
        """
        if product_id not in self.performance_history:
            self.performance_history[product_id] = []
        
        performance_record = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'performance': performance
        }
        
        self.performance_history[product_id].append(performance_record)
        
        # 更新最後修改時間
        if product_id in self.metadata_store:
            self.metadata_store[product_id]['last_updated'] = pd.Timestamp.now().isoformat()
    
    def create_portfolio(self, 
                        portfolio_id: str,
                        name: str,
                        description: str,
                        product_ids: List[str],
                        weights: Optional[List[float]] = None) -> ProductPortfolio:
        """
        創建產品組合
        
        Parameters:
        -----------
        portfolio_id : str
            組合ID
        name : str
            組合名稱
        description : str
            組合描述
        product_ids : List[str]
            產品ID列表
        weights : List[float], optional
            權重列表
            
        Returns:
        --------
        ProductPortfolio
            創建的產品組合
        """
        
        # 驗證產品存在
        products = []
        for product_id in product_ids:
            if product_id in self.products:
                products.append(self.products[product_id])
            else:
                raise ValueError(f"Product {product_id} not found")
        
        # 設置權重
        if weights is None:
            weights = [1.0 / len(products)] * len(products)
        elif len(weights) != len(products):
            raise ValueError("Weights length must match products length")
        
        # 正規化權重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        portfolio = ProductPortfolio(
            portfolio_id=portfolio_id,
            name=name,
            description=description,
            products=products,
            weights=weights,
            creation_date=pd.Timestamp.now().isoformat()
        )
        
        self.portfolios[portfolio_id] = portfolio
        return portfolio
    
    def compare_products(self, 
                        product_ids: List[str],
                        metrics: List[str] = None) -> pd.DataFrame:
        """
        比較產品
        
        Parameters:
        -----------
        product_ids : List[str]
            要比較的產品ID列表
        metrics : List[str], optional
            要比較的指標
            
        Returns:
        --------
        pd.DataFrame
            產品比較結果
        """
        
        if metrics is None:
            metrics = ['rmse', 'mae', 'correlation', 'hit_rate', 'coverage_ratio']
        
        comparison_data = []
        
        for product_id in product_ids:
            if product_id not in self.products:
                continue
            
            product = self.products[product_id]
            row = {
                'product_id': product_id,
                'name': product.name,
                'description': product.description,
                'index_type': product.index_type.value,
                'payout_function_type': product.payout_function_type.value,
                'n_thresholds': len(product.trigger_thresholds),
                'max_payout': product.max_payout
            }
            
            # 添加最新績效數據
            if product_id in self.performance_history and self.performance_history[product_id]:
                latest_performance = self.performance_history[product_id][-1]['performance']
                
                for metric in metrics:
                    if hasattr(latest_performance, metric):
                        row[metric] = getattr(latest_performance, metric)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_product_performance_history(self, product_id: str) -> pd.DataFrame:
        """
        獲取產品績效歷史
        
        Parameters:
        -----------
        product_id : str
            產品ID
            
        Returns:
        --------
        pd.DataFrame
            績效歷史數據
        """
        
        if product_id not in self.performance_history:
            return pd.DataFrame()
        
        history_data = []
        for record in self.performance_history[product_id]:
            performance = record['performance']
            row = {
                'timestamp': record['timestamp'],
                'rmse': performance.rmse,
                'mae': performance.mae,
                'correlation': performance.correlation,
                'hit_rate': performance.hit_rate,
                'false_alarm_rate': performance.false_alarm_rate,
                'coverage_ratio': performance.coverage_ratio,
                'basis_risk': performance.basis_risk
            }
            history_data.append(row)
        
        df = pd.DataFrame(history_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    
    def generate_product_report(self, product_id: str) -> Dict[str, Any]:
        """
        生成產品報告
        
        Parameters:
        -----------
        product_id : str
            產品ID
            
        Returns:
        --------
        Dict[str, Any]
            產品報告
        """
        
        if product_id not in self.products:
            raise ValueError(f"Product {product_id} not found")
        
        product = self.products[product_id]
        metadata = self.metadata_store.get(product_id, {})
        
        report = {
            'product_info': {
                'product_id': product.product_id,
                'name': product.name,
                'description': product.description,
                'index_type': product.index_type.value,
                'payout_function_type': product.payout_function_type.value,
                'trigger_thresholds': product.trigger_thresholds,
                'payout_amounts': product.payout_amounts,
                'max_payout': product.max_payout,
                'technical_premium': product.technical_premium
            },
            'metadata': metadata,
            'performance_summary': {},
            'recommendations': []
        }
        
        # 添加績效摘要
        if product_id in self.performance_history and self.performance_history[product_id]:
            history_df = self.get_product_performance_history(product_id)
            
            if not history_df.empty:
                latest_performance = history_df.iloc[-1]
                performance_trend = self._analyze_performance_trend(history_df)
                
                report['performance_summary'] = {
                    'latest_rmse': latest_performance['rmse'],
                    'latest_correlation': latest_performance['correlation'],
                    'latest_coverage_ratio': latest_performance['coverage_ratio'],
                    'performance_trend': performance_trend,
                    'n_evaluations': len(history_df)
                }
        
        # 生成建議
        recommendations = self._generate_product_recommendations(product, report['performance_summary'])
        report['recommendations'] = recommendations
        
        return report
    
    def find_similar_products(self, 
                            reference_product_id: str,
                            similarity_threshold: float = 0.8) -> List[tuple[str, float]]:
        """
        找尋相似產品
        
        Parameters:
        -----------
        reference_product_id : str
            參考產品ID
        similarity_threshold : float
            相似度閾值
            
        Returns:
        --------
        List[Tuple[str, float]]
            相似產品列表 (product_id, similarity_score)
        """
        
        if reference_product_id not in self.products:
            raise ValueError(f"Reference product {reference_product_id} not found")
        
        reference_product = self.products[reference_product_id]
        similar_products = []
        
        for product_id, product in self.products.items():
            if product_id == reference_product_id:
                continue
            
            similarity = self._calculate_product_similarity(reference_product, product)
            
            if similarity >= similarity_threshold:
                similar_products.append((product_id, similarity))
        
        # 按相似度排序
        similar_products.sort(key=lambda x: x[1], reverse=True)
        
        return similar_products
    
    def optimize_portfolio(self, 
                          portfolio_id: str,
                          optimization_objective: str = "risk_adjusted_return") -> Dict[str, Any]:
        """
        優化產品組合
        
        Parameters:
        -----------
        portfolio_id : str
            組合ID
        optimization_objective : str
            優化目標
            
        Returns:
        --------
        Dict[str, Any]
            優化結果
        """
        
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        portfolio = self.portfolios[portfolio_id]
        
        # 獲取所有產品的最新績效
        product_performances = []
        for product in portfolio.products:
            if product.product_id in self.performance_history and self.performance_history[product.product_id]:
                latest_perf = self.performance_history[product.product_id][-1]['performance']
                product_performances.append(latest_perf)
            else:
                # 如果沒有績效數據，跳過
                continue
        
        if not product_performances:
            return {'error': 'No performance data available for portfolio optimization'}
        
        # 簡化的優化邏輯
        if optimization_objective == "risk_adjusted_return":
            # 使用correlation/rmse作為風險調整回報
            scores = []
            for perf in product_performances:
                if perf.rmse > 0:
                    score = perf.correlation / (perf.rmse / 1e9)  # 正規化RMSE
                else:
                    score = perf.correlation
                scores.append(score)
            
            # 計算最優權重（簡化版本）
            scores = np.array(scores)
            if np.sum(scores) > 0:
                optimal_weights = scores / np.sum(scores)
            else:
                optimal_weights = np.ones(len(scores)) / len(scores)
            
            optimization_result = {
                'original_weights': portfolio.weights[:len(optimal_weights)],
                'optimal_weights': optimal_weights.tolist(),
                'improvement_score': np.dot(optimal_weights, scores) / np.dot(portfolio.weights[:len(optimal_weights)], scores) if np.dot(portfolio.weights[:len(optimal_weights)], scores) > 0 else 1,
                'objective': optimization_objective
            }
            
            return optimization_result
        
        return {'error': f'Unsupported optimization objective: {optimization_objective}'}
    
    def export_portfolio(self, portfolio_id: str, output_path: str, format: str = "json") -> None:
        """
        導出產品組合
        
        Parameters:
        -----------
        portfolio_id : str
            組合ID
        output_path : str
            輸出路徑
        format : str
            導出格式 ('json', 'pickle')
        """
        
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        portfolio = self.portfolios[portfolio_id]
        
        if format.lower() == "json":
            # 轉換為可序列化的格式
            export_data = {
                'portfolio_info': {
                    'portfolio_id': portfolio.portfolio_id,
                    'name': portfolio.name,
                    'description': portfolio.description,
                    'creation_date': portfolio.creation_date,
                    'status': portfolio.status.value
                },
                'products': [],
                'weights': portfolio.weights
            }
            
            for product in portfolio.products:
                product_data = {
                    'product_id': product.product_id,
                    'name': product.name,
                    'description': product.description,
                    'index_type': product.index_type.value,
                    'payout_function_type': product.payout_function_type.value,
                    'trigger_thresholds': product.trigger_thresholds,
                    'payout_amounts': product.payout_amounts,
                    'max_payout': product.max_payout,
                    'metadata': product.metadata
                }
                export_data['products'].append(product_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "pickle":
            with open(output_path, 'wb') as f:
                pickle.dump(portfolio, f)
        
        print(f"✅ Portfolio exported to: {output_path}")
    
    # ========== 私有方法 ==========
    
    def _calculate_product_similarity(self, product1: ParametricProduct, product2: ParametricProduct) -> float:
        """計算產品相似度"""
        similarity_score = 0.0
        
        # 指標類型相似度
        if product1.index_type == product2.index_type:
            similarity_score += 0.3
        
        # 賠付函數類型相似度
        if product1.payout_function_type == product2.payout_function_type:
            similarity_score += 0.2
        
        # 閾值數量相似度
        threshold_diff = abs(len(product1.trigger_thresholds) - len(product2.trigger_thresholds))
        threshold_similarity = max(0, 1 - threshold_diff / 5)  # 最多5個閾值的差異
        similarity_score += 0.2 * threshold_similarity
        
        # 最大賠付相似度
        if product1.max_payout > 0 and product2.max_payout > 0:
            payout_ratio = min(product1.max_payout, product2.max_payout) / max(product1.max_payout, product2.max_payout)
            similarity_score += 0.3 * payout_ratio
        
        return min(similarity_score, 1.0)
    
    def _analyze_performance_trend(self, history_df: pd.DataFrame) -> str:
        """分析績效趨勢"""
        if len(history_df) < 2:
            return "insufficient_data"
        
        # 分析RMSE趨勢
        recent_rmse = history_df['rmse'].tail(3).mean()
        early_rmse = history_df['rmse'].head(3).mean()
        
        if recent_rmse < early_rmse * 0.95:
            return "improving"
        elif recent_rmse > early_rmse * 1.05:
            return "deteriorating"
        else:
            return "stable"
    
    def _generate_product_recommendations(self, product: ParametricProduct, performance_summary: Dict[str, Any]) -> List[str]:
        """生成產品建議"""
        recommendations = []
        
        if performance_summary:
            # 基於績效的建議
            if 'latest_correlation' in performance_summary:
                correlation = performance_summary['latest_correlation']
                if correlation < 0.3:
                    recommendations.append("相關性偏低，建議調整觸發閾值")
                elif correlation > 0.7:
                    recommendations.append("相關性良好，產品設計合理")
            
            if 'latest_coverage_ratio' in performance_summary:
                coverage = performance_summary['latest_coverage_ratio']
                if coverage < 0.5:
                    recommendations.append("覆蓋率不足，考慮增加賠付金額")
                elif coverage > 1.5:
                    recommendations.append("覆蓋率過高，可能導致道德風險")
            
            if 'performance_trend' in performance_summary:
                trend = performance_summary['performance_trend']
                if trend == "deteriorating":
                    recommendations.append("績效下降，需要重新校準產品參數")
                elif trend == "improving":
                    recommendations.append("績效改善中，可考慮推廣此產品設計")
        
        # 基於產品結構的建議
        if len(product.trigger_thresholds) == 1:
            recommendations.append("單閾值產品，考慮添加多層觸發機制")
        elif len(product.trigger_thresholds) > 4:
            recommendations.append("觸發層級較多，可能增加操作複雜度")
        
        if not recommendations:
            recommendations.append("產品結構合理，繼續監控績效表現")
        
        return recommendations