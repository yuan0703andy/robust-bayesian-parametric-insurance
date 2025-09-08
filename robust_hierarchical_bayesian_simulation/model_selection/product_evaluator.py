#!/usr/bin/env python3
"""
Product Evaluator Module
產品評估器模組

正確實現兩步驟架構的 Step 2: 產品評估與排名

使用 Step 1 訓練好的損失預測器作為「完美尺子」，
評估 350 個固定 Steinmann 產品的基差風險

Author: Research Team  
Date: 2025-01-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .hierarchical_loss_predictor_vi import HierarchicalLossPredictorVI


class ProductEvaluator:
    """
    產品評估器 - 使用訓練好的損失預測器評估保險產品
    
    核心理念：用最準確的損失預測器作為評估標準，
    客觀地為 350 個固定產品排名
    """
    
    def __init__(self, 
                 trained_loss_predictor: HierarchicalLossPredictorVI,
                 steinmann_products: List[Dict]):
        """
        初始化產品評估器
        
        Args:
            trained_loss_predictor: Step 1 訓練好的損失預測器
            steinmann_products: 350 個固定的 Steinmann 產品
        """
        self.loss_predictor = trained_loss_predictor
        self.products = steinmann_products
        
        print(f"📊 產品評估器初始化")
        print(f"   損失預測器: {type(trained_loss_predictor).__name__}")
        print(f"   待評估產品數: {len(steinmann_products)}")
        
        # 基差風險計算器
        self.basis_risk_calculator = BasisRiskCalculator()
    
    def evaluate_all_products(self, 
                            hazard_data: np.ndarray,
                            trained_predictor_params: Dict,
                            evaluation_events: Optional[List[int]] = None) -> Dict:
        """
        評估所有產品的基差風險
        
        Args:
            hazard_data: 災害數據 [n_hospitals, n_events]
            trained_predictor_params: Step 1 的訓練結果
            evaluation_events: 評估事件索引（None表示全部事件）
            
        Returns:
            評估結果字典
        """
        print(f"📊 開始評估 {len(self.products)} 個產品...")
        print(f"   災害數據: {hazard_data.shape}")
        
        if evaluation_events is None:
            evaluation_events = list(range(hazard_data.shape[1]))
        
        print(f"   評估事件數: {len(evaluation_events)}")
        
        # 1. 使用訓練好的損失預測器預測損失分佈
        print("🧠 使用損失預測器生成損失分佈...")
        
        predicted_losses = self.loss_predictor.predict(
            hazard_data[:, evaluation_events], 
            trained_predictor_params
        )
        
        loss_mean = predicted_losses['loss_mean']  # [n_hospitals, n_eval_events]
        loss_samples = predicted_losses['loss_samples']  # [n_hospitals, n_eval_events, n_samples]
        
        print(f"✅ 損失預測完成")
        print(f"   預測損失範圍: ${loss_mean.min()/1e6:.1f}M - ${loss_mean.max()/1e6:.1f}M")
        
        # 2. 為每個產品計算基差風險
        print("🔧 計算產品基差風險...")
        
        product_scores = {}
        product_details = []
        
        for product_idx, product in enumerate(self.products):
            # 計算該產品在所有評估事件上的賠付
            total_basis_risk = 0.0
            event_basis_risks = []
            
            for eval_event_idx, original_event_idx in enumerate(evaluation_events):
                # 計算該事件的Cat-in-Circle指數和產品賠付
                event_hazard = hazard_data[:, original_event_idx]  # [n_hospitals]
                
                # Cat-in-Circle: 半徑內最大風速
                max_wind_in_radius = np.max(event_hazard)  # 簡化：使用所有醫院的最大風速
                
                # 計算產品賠付
                product_payout = self._calculate_product_payout(
                    product, max_wind_in_radius
                )
                
                # 該事件的真實損失分佈（來自損失預測器）
                event_loss_distribution = loss_samples[:, eval_event_idx, :]  # [n_hospitals, n_samples]
                event_total_loss_mean = np.sum(loss_mean[:, eval_event_idx])  # 該事件總損失
                
                # 計算基差風險
                event_basis_risk = self.basis_risk_calculator.compute_basis_risk(
                    predicted_loss=event_total_loss_mean,
                    product_payout=product_payout,
                    loss_distribution_samples=np.sum(event_loss_distribution, axis=0),  # 總損失樣本
                    basis_risk_type='weighted'
                )
                
                event_basis_risks.append(event_basis_risk)
                total_basis_risk += event_basis_risk
            
            # 平均基差風險
            avg_basis_risk = total_basis_risk / len(evaluation_events)
            
            # 記錄結果
            product_scores[product_idx] = avg_basis_risk
            
            product_details.append({
                'product_id': product_idx,
                'product_config': product,
                'avg_basis_risk': avg_basis_risk,
                'event_basis_risks': event_basis_risks,
                'trigger_thresholds': product['trigger_thresholds'],
                'payout_ratios': product['payout_ratios'],
                'max_payout': product['max_payout'],
                'radius_km': product['radius_km']
            })
            
            # 進度報告
            if (product_idx + 1) % 50 == 0:
                print(f"   已評估 {product_idx+1}/{len(self.products)} 個產品")
        
        print(f"✅ 產品評估完成!")
        
        return {
            'product_scores': product_scores,
            'product_details': product_details,
            'evaluation_summary': {
                'n_products': len(self.products),
                'n_evaluation_events': len(evaluation_events),
                'predictor_params': trained_predictor_params
            }
        }
    
    def _calculate_product_payout(self, product: Dict, cat_in_circle_index: float) -> float:
        """
        計算單個產品的賠付
        
        Args:
            product: 產品配置
            cat_in_circle_index: Cat-in-Circle 指數（最大風速）
            
        Returns:
            產品賠付金額
        """
        thresholds = product['trigger_thresholds']
        ratios = product['payout_ratios']
        max_payout = product['max_payout']
        
        # Steinmann 2023 標準：階梯式賠付
        total_payout = 0.0
        
        # 按閾值從高到低檢查
        for i in range(len(thresholds)-1, -1, -1):
            if cat_in_circle_index >= thresholds[i]:
                total_payout = max_payout * ratios[i]
                break
        
        return total_payout
    
    def rank_products(self, evaluation_results: Dict) -> pd.DataFrame:
        """
        根據基差風險對產品排名
        
        Args:
            evaluation_results: evaluate_all_products 的結果
            
        Returns:
            產品排名 DataFrame
        """
        product_details = evaluation_results['product_details']
        
        # 按基差風險排序（越小越好）
        sorted_products = sorted(product_details, key=lambda x: x['avg_basis_risk'])
        
        # 創建排名 DataFrame
        ranking_data = []
        for rank, product_detail in enumerate(sorted_products, 1):
            ranking_data.append({
                'rank': rank,
                'product_id': product_detail['product_id'],
                'avg_basis_risk': product_detail['avg_basis_risk'],
                'radius_km': product_detail['radius_km'],
                'n_thresholds': len([t for t in product_detail['trigger_thresholds'] if t < 999]),
                'min_threshold': min([t for t in product_detail['trigger_thresholds'] if t < 999]),
                'max_threshold': max([t for t in product_detail['trigger_thresholds'] if t < 999]),
                'max_payout': product_detail['max_payout'],
                'trigger_thresholds': product_detail['trigger_thresholds'],
                'payout_ratios': product_detail['payout_ratios']
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # 顯示前10名
        print(f"\n🏆 產品排名 (前10名):")
        print("=" * 80)
        print(f"{'排名':<4} {'產品ID':<8} {'基差風險(M)':<12} {'半徑(km)':<8} {'閾值數':<6} {'最大賠付(M)':<12}")
        print("=" * 80)
        
        for i in range(min(10, len(ranking_df))):
            row = ranking_df.iloc[i]
            print(f"{row['rank']:<4} {row['product_id']:<8} "
                  f"{row['avg_basis_risk']/1e6:<12.2f} {row['radius_km']:<8} "
                  f"{row['n_thresholds']:<6} {row['max_payout']/1e6:<12.2f}")
        
        return ranking_df
    
    def compare_with_traditional_analysis(self, 
                                        evaluation_results: Dict,
                                        traditional_results_path: str) -> Dict:
        """
        與04腳本的傳統分析結果比較
        
        Args:
            evaluation_results: 本評估器的結果
            traditional_results_path: 04腳本結果路徑
            
        Returns:
            比較結果
        """
        try:
            import pickle
            with open(traditional_results_path, 'rb') as f:
                traditional_results = pickle.load(f)
            
            print(f"📊 與傳統分析比較...")
            
            # 比較冠軍產品
            hierarchical_champion = min(evaluation_results['product_details'], 
                                      key=lambda x: x['avg_basis_risk'])
            
            comparison = {
                'hierarchical_champion': {
                    'product_id': hierarchical_champion['product_id'],
                    'basis_risk': hierarchical_champion['avg_basis_risk'],
                    'config': hierarchical_champion['product_config']
                },
                'traditional_champion': traditional_results.get('best_product', {}),
                'improvement': None
            }
            
            # 計算改進幅度
            if 'best_basis_risk' in traditional_results.get('best_product', {}):
                trad_risk = traditional_results['best_product']['best_basis_risk']
                hier_risk = hierarchical_champion['avg_basis_risk']
                improvement = (trad_risk - hier_risk) / trad_risk * 100
                comparison['improvement'] = improvement
                
                print(f"🏆 方法比較:")
                print(f"   傳統方法最佳基差風險: {trad_risk/1e6:.2f}M")
                print(f"   階層貝葉斯最佳基差風險: {hier_risk/1e6:.2f}M")
                print(f"   改進幅度: {improvement:.1f}%")
            
            return comparison
            
        except Exception as e:
            print(f"⚠️ 無法載入傳統分析結果: {e}")
            return {'error': str(e)}


class BasisRiskCalculator:
    """基差風險計算器"""
    
    def compute_basis_risk(self, 
                          predicted_loss: float,
                          product_payout: float, 
                          loss_distribution_samples: np.ndarray = None,
                          basis_risk_type: str = 'weighted') -> float:
        """
        計算基差風險
        
        Args:
            predicted_loss: 預測損失（可以是均值）
            product_payout: 產品賠付
            loss_distribution_samples: 損失分佈樣本（可選）
            basis_risk_type: 基差風險類型
            
        Returns:
            基差風險值
        """
        if loss_distribution_samples is not None:
            # 使用完整分佈計算更準確的基差風險
            payouts = np.full_like(loss_distribution_samples, product_payout)
            differences = loss_distribution_samples - payouts
        else:
            # 使用點估計
            differences = np.array([predicted_loss - product_payout])
        
        if basis_risk_type == 'absolute':
            return np.mean(np.abs(differences))
        elif basis_risk_type == 'asymmetric':
            # 只懲罰賠付不足
            return np.mean(np.maximum(0, differences))
        elif basis_risk_type == 'weighted':
            # 加權不對稱懲罰：賠付不足懲罰更重
            under_penalty = np.maximum(0, differences) * 2.0
            over_penalty = np.maximum(0, -differences) * 0.5
            return np.mean(under_penalty + over_penalty)
        else:
            return np.mean(np.abs(differences))


if __name__ == "__main__":
    print("📊 產品評估器 - 獨立測試")
    
    # 這裡可以添加獨立測試代碼
    pass