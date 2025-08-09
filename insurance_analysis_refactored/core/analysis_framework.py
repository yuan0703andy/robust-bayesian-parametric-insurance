"""
Unified Analysis Framework
統一分析框架

This is the highest-level interface that integrates all insurance analysis components
into a single, powerful, and easy-to-use framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

from .parametric_engine import (
    ParametricInsuranceEngine, ParametricProduct, ProductPerformance,
    ParametricIndexType, PayoutFunctionType
)
from .skill_evaluator import SkillScoreEvaluator, SkillScoreType, SkillScoreResult
from .product_manager import InsuranceProductManager
# Merged from unified_product_engine.py - Input Adapter functionality
from .input_adapters import InputAdapter, CLIMADAInputAdapter, BayesianInputAdapter
from .enhanced_spatial_analysis import EnhancedCatInCircleAnalyzer, create_standard_steinmann_config
from .saffir_simpson_products import (
    SaffirSimpsonProductGenerator, PayoutStructure, 
    generate_steinmann_2023_products, validate_steinmann_compatibility
)

# Merged from unified_product_engine.py - Evaluation and Product Design enums
class EvaluationMode(Enum):
    """評估模式"""
    TRADITIONAL = "traditional"      # 確定性評估 (RMSE, correlation)
    PROBABILISTIC = "probabilistic"  # 機率性評估 (CRPS, distribution-based)
    HYBRID = "hybrid"               # 混合評估

class ProductDesignType(Enum):
    """產品設計類型"""
    STEINMANN_70 = "steinmann_70"          # Steinmann標準70產品
    CUSTOM_PRODUCTS = "custom_products"    # 自定義產品
    OPTIMIZED_SELECTION = "optimized"      # 優化選擇

class AnalysisType(Enum):
    """分析類型"""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    STEINMANN = "steinmann"
    BAYESIAN = "bayesian"
    ROBUST_BAYESIAN = "robust_bayesian"
    COMPARATIVE = "comparative"

@dataclass
class AnalysisConfig:
    """分析配置"""
    analysis_type: AnalysisType
    skill_scores: List[SkillScoreType] = field(default_factory=lambda: [
        SkillScoreType.RMSE, SkillScoreType.MAE, SkillScoreType.CORRELATION,
        SkillScoreType.CRPS, SkillScoreType.EDI, SkillScoreType.TSS
    ])
    bootstrap_enabled: bool = True
    confidence_level: float = 0.95
    max_products: int = 70
    optimization_criteria: List[str] = field(default_factory=lambda: ['rmse', 'correlation', 'coverage_ratio'])
    technical_premium_config: Dict[str, float] = field(default_factory=lambda: {
        'risk_load_factor': 0.20,
        'expense_load_factor': 0.15
    })
    # 新增：快速測試模式設定
    fast_mode: bool = False
    skill_score_sample_size: Optional[int] = None  # 限制技能評分計算的產品數量
    
    # 新增：貝氏分析設定
    bayesian_config: Optional[Any] = None
    enable_robustness_analysis: bool = False
    enable_bayesian_visualization: bool = False
    
    # Merged from unified_product_engine.py - Product Design Configuration
    design_type: ProductDesignType = ProductDesignType.STEINMANN_70
    evaluation_mode: EvaluationMode = EvaluationMode.TRADITIONAL
    
    # Cat-in-a-Circle 配置
    spatial_radii_km: List[float] = field(default_factory=lambda: [15.0, 30.0, 50.0])
    spatial_statistics: List[str] = field(default_factory=lambda: ['max', 'mean', '95th'])
    
    # 評估配置
    enable_bootstrap_ci: bool = True
    
    # 輸出配置
    enable_visualization: bool = True
    save_intermediate_results: bool = True
    output_format: str = "comprehensive"  # "minimal", "standard", "comprehensive"

    @classmethod
    def create_fast_test_config(cls, analysis_type: AnalysisType = AnalysisType.BASIC) -> 'AnalysisConfig':
        """創建快速測試配置"""
        return cls(
            analysis_type=analysis_type,
            skill_scores=[SkillScoreType.RMSE, SkillScoreType.CORRELATION],  # 只計算基本評分
            bootstrap_enabled=False,  # 關閉bootstrap以加速
            max_products=20,  # 減少產品數量
            fast_mode=True,
            skill_score_sample_size=5  # 只為5個產品計算技能評分
        )
    
    @classmethod 
    def create_robust_bayesian_config(cls) -> 'AnalysisConfig':
        """創建穩健貝氏分析配置"""
        # Updated to use new bayesian module
        try:
            from bayesian import HierarchicalModelConfig as BayesianConfig
            # Define simplified enums for backward compatibility
            from enum import Enum
            class PriorScenario(Enum):
                NORMAL = "normal"
                GAMMA = "gamma"
            class LikelihoodType(Enum):
                NORMAL = "normal"
                GAMMA = "gamma"
        except ImportError:
            # Fallback definitions
            from enum import Enum
            class BayesianConfig:
                pass
            class PriorScenario(Enum):
                NORMAL = "normal"
            class LikelihoodType(Enum):
                NORMAL = "normal"
        
        bayesian_config = BayesianConfig(
            n_mcmc_samples=5000,
            n_burn_in=1000,
            prior_scenarios=[PriorScenario.NEUTRAL, PriorScenario.OPTIMISTIC, PriorScenario.PESSIMISTIC],
            likelihood_types=[LikelihoodType.NORMAL, LikelihoodType.T_DISTRIBUTION],
            ensemble_size=50,
            robust_optimization=True
        )
        
        return cls(
            analysis_type=AnalysisType.ROBUST_BAYESIAN,
            skill_scores=[SkillScoreType.CRPS, SkillScoreType.CRPSS, SkillScoreType.RMSE, SkillScoreType.CORRELATION],
            bootstrap_enabled=True,
            max_products=50,
            bayesian_config=bayesian_config,
            enable_robustness_analysis=True,
            enable_bayesian_visualization=True
        )

@dataclass
class AnalysisResults:
    """分析結果"""
    analysis_type: AnalysisType
    products: List[ParametricProduct]
    performance_results: pd.DataFrame
    skill_score_results: Dict[str, Dict[SkillScoreType, SkillScoreResult]]
    optimization_results: pd.DataFrame
    technical_premium_results: pd.DataFrame
    best_products: Dict[str, ParametricProduct]
    summary_statistics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 新增：貝氏分析結果
    bayesian_results: Optional[Dict[str, 'BayesianAnalysisResult']] = None
    robustness_analysis: Optional[Dict[str, Any]] = None
    loss_distributions: Optional[Dict[str, Dict[str, 'ProbabilisticLossDistribution']]] = None

class UnifiedAnalysisFramework:
    """
    統一分析框架
    
    這是最高級別的介面，整合所有保險分析組件，
    提供統一、強大且易於使用的分析框架。
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        初始化統一分析框架
        
        Parameters:
        -----------
        config : AnalysisConfig, optional
            分析配置
        """
        self.config = config or AnalysisConfig(AnalysisType.COMPREHENSIVE)
        
        # 初始化核心組件
        self.parametric_engine = ParametricInsuranceEngine()
        self.skill_evaluator = SkillScoreEvaluator(
            bootstrap_samples=1000 if self.config.bootstrap_enabled else 0,
            confidence_level=self.config.confidence_level
        )
        self.product_manager = InsuranceProductManager()
        
        # 初始化貝氏分析組件
        if self.config.bayesian_config is not None:
            # Updated to use new bayesian module
            from bayesian import RobustBayesianAnalyzer
            # Bayesian visualization moved to visualization/ module
            BayesianVisualization = None  # Placeholder
            
            self.bayesian_analyzer = RobustBayesianAnalyzer(self.config.bayesian_config)
            if self.config.enable_bayesian_visualization:
                self.bayesian_visualizer = BayesianVisualization()
            else:
                self.bayesian_visualizer = None
        else:
            self.bayesian_analyzer = None
            self.bayesian_visualizer = None
        
        # Merged from unified_product_engine.py - Additional components
        self.spatial_analyzer = EnhancedCatInCircleAnalyzer(
            create_standard_steinmann_config()
        )
        self.product_generator = SaffirSimpsonProductGenerator()
        
        # 運行統計
        self.run_statistics = {
            'total_analyses': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'average_computation_time': 0.0
        }
        
        # 分析結果緩存
        self.analysis_cache = {}
    
    def run_comprehensive_analysis(self,
                                 parametric_indices: np.ndarray,
                                 observed_losses: np.ndarray,
                                 analysis_name: str = "comprehensive_analysis") -> AnalysisResults:
        """
        執行全面分析
        
        Parameters:
        -----------
        parametric_indices : np.ndarray
            參數指標數據
        observed_losses : np.ndarray
            觀測損失數據
        analysis_name : str
            分析名稱
            
        Returns:
        --------
        AnalysisResults
            完整分析結果
        """
        
        print(f"🚀 執行全面分析: {analysis_name}")
        print("=" * 60)
        
        # 1. 生成產品組合
        print("📦 階段1: 生成參數型保險產品...")
        max_payout = np.max(observed_losses) * 2 if np.any(observed_losses > 0) else 1e9
        
        if self.config.analysis_type == AnalysisType.STEINMANN:
            products = self._generate_steinmann_products_helper(
                parametric_indices, max_payout, self.config.max_products, observed_losses
            )
        else:
            # 生成多樣化產品組合
            products = self._generate_diverse_products(parametric_indices, max_payout)
        
        print(f"   ✅ 生成了 {len(products)} 個產品")
        
        # 2. 評估產品績效
        print("📊 階段2: 評估產品績效...")
        performance_results = []
        
        for i, product in enumerate(products):
            if (i + 1) % 20 == 0:
                print(f"   進度: {i+1}/{len(products)}")
            
            performance = self.parametric_engine.evaluate_product_performance(
                product, parametric_indices, observed_losses
            )
            
            performance_dict = {
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
            
            performance_results.append(performance_dict)
        
        performance_df = pd.DataFrame(performance_results)
        print(f"   ✅ 完成績效評估")
        
        # 3. 計算技能評分
        print("🎯 階段3: 計算技能評分...")
        skill_score_results = {}
        
        # 決定要計算技能評分的產品數量
        if self.config.skill_score_sample_size is not None:
            skill_score_limit = min(self.config.skill_score_sample_size, len(products))
        elif self.config.fast_mode:
            skill_score_limit = len(products)  # 快速模式也評估全部產品
        else:
            skill_score_limit = len(products)  # 評估全部產品
            
        products_for_skill_scores = products[:skill_score_limit]
        
        print(f"   計算 {len(products_for_skill_scores)} 個產品的技能評分...")
        
        for i, product in enumerate(products_for_skill_scores):
            print(f"   進度: {i+1}/{len(products_for_skill_scores)} - {product.name[:30]}...")
            
            try:
                # 重新計算預測
                payout_func_class = self.parametric_engine.payout_function_classes[product.payout_function_type]
                payout_func = payout_func_class(
                    product.trigger_thresholds,
                    product.payout_amounts,
                    product.max_payout
                )
                predictions = payout_func.calculate_payouts_batch(parametric_indices)
                
                # 計算技能評分
                scores = self.skill_evaluator.evaluate_multiple_scores(
                    self.config.skill_scores,
                    observed_losses,
                    predictions
                )
                
                skill_score_results[product.product_id] = scores
                
            except Exception as e:
                warnings.warn(f"Failed to calculate skill scores for {product.product_id}: {e}")
                continue
        
        print(f"   ✅ 完成 {len(skill_score_results)} 個產品的技能評分")
        
        # 4. 產品優化
        print("🏆 階段4: 產品組合優化...")
        optimization_results = self.parametric_engine.optimize_product_portfolio(
            products, parametric_indices, observed_losses, self.config.optimization_criteria
        )
        print(f"   ✅ 完成產品優化")
        
        # 5. 技術保費計算
        print("💰 階段5: 技術保費計算...")
        premium_results = []
        
        for product in products:  # 為全部產品計算保費
            performance = self.parametric_engine.performance_cache.get(product.product_id)
            if performance:
                premium = self.parametric_engine.calculate_technical_premium(
                    product, performance, **self.config.technical_premium_config
                )
                
                premium_result = {
                    'product_id': product.product_id,
                    'name': product.name,
                    **premium
                }
                premium_results.append(premium_result)
        
        premium_df = pd.DataFrame(premium_results)
        print(f"   ✅ 完成技術保費計算")
        
        # 6. 識別最佳產品
        print("🎖️ 階段6: 識別最佳產品...")
        best_products = self._identify_best_products(optimization_results, products)
        
        # 7. 生成摘要統計
        summary_stats = self._generate_summary_statistics(
            products, performance_df, optimization_results, skill_score_results
        )
        
        # 創建分析結果
        results = AnalysisResults(
            analysis_type=self.config.analysis_type,
            products=products,
            performance_results=performance_df,
            skill_score_results=skill_score_results,
            optimization_results=optimization_results,
            technical_premium_results=premium_df,
            best_products=best_products,
            summary_statistics=summary_stats,
            metadata={
                'analysis_name': analysis_name,
                'n_events': len(parametric_indices),
                'n_products': len(products),
                'config': self.config
            }
        )
        
        # 緩存結果
        self.analysis_cache[analysis_name] = results
        
        print("✅ 全面分析完成!")
        self._print_summary_report(results)
        
        return results
    
    def run_steinmann_analysis(self,
                             parametric_indices: np.ndarray,
                             observed_losses: np.ndarray) -> AnalysisResults:
        """
        執行Steinmann et al. (2023)標準分析
        
        Parameters:
        -----------
        parametric_indices : np.ndarray
            參數指標數據
        observed_losses : np.ndarray
            觀測損失數據
            
        Returns:
        --------
        AnalysisResults
            Steinmann分析結果
        """
        
        # 設置Steinmann配置
        steinmann_config = AnalysisConfig(
            analysis_type=AnalysisType.STEINMANN,
            max_products=70,
            skill_scores=[SkillScoreType.RMSE, SkillScoreType.MAE, SkillScoreType.CORRELATION, 
                         SkillScoreType.CRPS, SkillScoreType.EDI, SkillScoreType.TSS]
        )
        
        original_config = self.config
        self.config = steinmann_config
        
        try:
            results = self.run_comprehensive_analysis(
                parametric_indices, observed_losses, "steinmann_analysis"
            )
            results.analysis_type = AnalysisType.STEINMANN
            return results
        finally:
            self.config = original_config
    
    def run_robust_bayesian_analysis(self,
                                   parametric_indices: np.ndarray,
                                   observed_losses: np.ndarray,
                                   hazard_intensities: np.ndarray,
                                   exposure_values: np.ndarray,
                                   vulnerability_params: Dict[str, float] = None,
                                   analysis_name: str = "robust_bayesian_analysis") -> AnalysisResults:
        """
        執行穩健貝氏分析
        
        實現從確定性思維到機率性思維的完整轉換，包括：
        1. 生成損失後驗預測分佈
        2. CRPS-based 產品優化
        3. 穩健性分析
        
        Parameters:
        -----------
        parametric_indices : np.ndarray
            參數指標數據
        observed_losses : np.ndarray
            觀測損失數據
        hazard_intensities : np.ndarray
            災害強度數據
        exposure_values : np.ndarray
            曝險價值數據
        vulnerability_params : Dict[str, float], optional
            脆弱度函數參數
        analysis_name : str
            分析名稱
            
        Returns:
        --------
        AnalysisResults
            完整的貝氏分析結果
        """
        
        if self.bayesian_analyzer is None:
            raise ValueError("貝氏分析器未初始化。請在配置中提供bayesian_config。")
        
        print(f"🎯 執行穩健貝氏分析: {analysis_name}")
        print("=" * 70)
        
        # 設置默認脆弱度參數
        if vulnerability_params is None:
            vulnerability_params = {
                'damage_threshold': 17.5,
                'beta': 3.0,
                'vulnerability_factor': 1.0
            }
        
        # 1. 首先執行傳統分析以獲得產品候選
        print("📦 階段1: 生成產品候選...")
        max_payout = np.max(observed_losses) * 2 if np.any(observed_losses > 0) else 1e9
        
        if self.config.analysis_type == AnalysisType.ROBUST_BAYESIAN:
            # 使用較少產品進行貝氏分析以節省計算時間
            products = self._generate_steinmann_products_helper(
                parametric_indices, max_payout, min(30, self.config.max_products)
            )
        else:
            products = self._generate_diverse_products(parametric_indices, max_payout)
        
        print(f"   ✅ 生成了 {len(products)} 個候選產品")
        
        # 2. 生成損失後驗預測分佈
        print("\n🎲 階段2: 生成損失後驗預測分佈...")
        event_ids = [f"event_{i}" for i in range(len(parametric_indices))]
        
        loss_distributions = self.bayesian_analyzer.generate_posterior_predictive_distributions(
            hazard_intensities=hazard_intensities,
            exposure_values=exposure_values,
            vulnerability_params=vulnerability_params,
            event_ids=event_ids
        )
        
        # 3. 使用CRPS優化產品
        print("\n🎯 階段3: CRPS-based 產品優化...")
        bayesian_optimization_results = self.bayesian_analyzer.optimize_products_with_crps(
            product_candidates=products,
            loss_distributions=loss_distributions,
            parametric_indices=parametric_indices
        )
        
        # 4. 執行穩健性分析
        robustness_results = None
        if self.config.enable_robustness_analysis:
            print("\n🛡️ 階段4: 穩健性分析...")
            robustness_results = self.bayesian_analyzer.perform_robustness_analysis(
                bayesian_optimization_results
            )
        
        # 5. 執行傳統績效評估用於比較
        print("\n📊 階段5: 傳統績效評估...")
        performance_results = []
        
        for product in products:
            performance = self.parametric_engine.evaluate_product_performance(
                product, parametric_indices, observed_losses
            )
            
            performance_dict = {
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
            
            performance_results.append(performance_dict)
        
        performance_df = pd.DataFrame(performance_results)
        
        # 6. 識別最佳產品（貝氏vs傳統）
        print("\n🏆 階段6: 識別最佳產品...")
        best_products = {}
        
        # 傳統最佳產品
        if not performance_df.empty:
            best_rmse_idx = performance_df['rmse'].idxmin()
            best_corr_idx = performance_df['correlation'].idxmax()
            best_products['traditional_best_rmse'] = products[best_rmse_idx]
            best_products['traditional_best_correlation'] = products[best_corr_idx]
        
        # 貝氏最佳產品
        if robustness_results and 'robust_products' in robustness_results:
            robust_product_ids = robustness_results['robust_products']
            if robust_product_ids:
                robust_product = next((p for p in products if p.product_id == robust_product_ids[0]), None)
                if robust_product:
                    best_products['bayesian_most_robust'] = robust_product
        
        # 7. 生成摘要統計
        summary_stats = self._generate_bayesian_summary_statistics(
            products, performance_df, bayesian_optimization_results, robustness_results
        )
        
        # 8. 創建分析結果
        results = AnalysisResults(
            analysis_type=AnalysisType.ROBUST_BAYESIAN,
            products=products,
            performance_results=performance_df,
            skill_score_results={},  # 在貝氏分析中由CRPS取代
            optimization_results=pd.DataFrame(),  # 由貝氏結果取代
            technical_premium_results=pd.DataFrame(),
            best_products=best_products,
            summary_statistics=summary_stats,
            bayesian_results=bayesian_optimization_results,
            robustness_analysis=robustness_results,
            loss_distributions=loss_distributions,
            metadata={
                'analysis_name': analysis_name,
                'n_events': len(parametric_indices),
                'n_products': len(products),
                'config': self.config,
                'bayesian_config': self.config.bayesian_config
            }
        )
        
        # 緩存結果
        self.analysis_cache[analysis_name] = results
        
        print("✅ 穩健貝氏分析完成!")
        self._print_bayesian_summary_report(results)
        
        return results
    
    def compare_methods(self,
                       parametric_indices: np.ndarray,
                       observed_losses: np.ndarray,
                       method_results: Dict[str, AnalysisResults]) -> Dict[str, Any]:
        """
        比較不同方法的分析結果
        
        Parameters:
        -----------
        parametric_indices : np.ndarray
            參數指標數據
        observed_losses : np.ndarray
            觀測損失數據
        method_results : Dict[str, AnalysisResults]
            方法結果字典
            
        Returns:
        --------
        Dict[str, Any]
            比較結果
        """
        
        print("🏆 執行方法比較分析")
        print("=" * 50)
        
        comparison_results = {
            'method_performance': {},
            'best_products_by_method': {},
            'skill_score_comparison': {},
            'statistical_tests': {},
            'recommendations': []
        }
        
        # 1. 方法表現比較
        for method_name, results in method_results.items():
            perf_df = results.performance_results
            
            comparison_results['method_performance'][method_name] = {
                'n_products': len(results.products),
                'best_rmse': perf_df['rmse'].min(),
                'best_correlation': perf_df['correlation'].max(),
                'mean_rmse': perf_df['rmse'].mean(),
                'mean_correlation': perf_df['correlation'].mean(),
                'triggered_products': len(perf_df[perf_df['payout_frequency'] > 0])
            }
            
            # 最佳產品
            best_idx = perf_df['rmse'].idxmin()
            comparison_results['best_products_by_method'][method_name] = perf_df.loc[best_idx].to_dict()
        
        # 2. 技能評分比較
        skill_comparison = {}
        for score_type in self.config.skill_scores:
            skill_comparison[score_type.value] = {}
            
            for method_name, results in method_results.items():
                scores = []
                for product_scores in results.skill_score_results.values():
                    if score_type in product_scores:
                        scores.append(product_scores[score_type].value)
                
                if scores:
                    skill_comparison[score_type.value][method_name] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores)
                    }
        
        comparison_results['skill_score_comparison'] = skill_comparison
        
        # 3. 生成建議
        recommendations = self._generate_method_recommendations(comparison_results)
        comparison_results['recommendations'] = recommendations
        
        print("✅ 方法比較分析完成!")
        self._print_comparison_report(comparison_results)
        
        return comparison_results
    
    def export_results(self,
                      results: AnalysisResults,
                      output_path: str,
                      format: str = "excel") -> None:
        """
        導出分析結果
        
        Parameters:
        -----------
        results : AnalysisResults
            分析結果
        output_path : str
            輸出路徑
        format : str
            輸出格式 ('excel', 'csv', 'json')
        """
        
        if format.lower() == "excel":
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                results.performance_results.to_excel(writer, sheet_name='Performance', index=False)
                results.optimization_results.to_excel(writer, sheet_name='Optimization', index=False)
                results.technical_premium_results.to_excel(writer, sheet_name='Premium', index=False)
                
                # 摘要統計
                summary_df = pd.DataFrame([results.summary_statistics])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        elif format.lower() == "csv":
            base_path = output_path.replace('.csv', '')
            results.performance_results.to_csv(f"{base_path}_performance.csv", index=False)
            results.optimization_results.to_csv(f"{base_path}_optimization.csv", index=False)
            results.technical_premium_results.to_csv(f"{base_path}_premium.csv", index=False)
        
        print(f"✅ 結果已導出至: {output_path}")
    
    # ========== 私有方法 ==========
    
    def _generate_diverse_products(self, parametric_indices: np.ndarray, max_payout: float) -> List[ParametricProduct]:
        """生成多樣化的產品組合"""
        products = []
        
        # 基本Steinmann產品
        steinmann_products = self._generate_steinmann_products_helper(
            parametric_indices, max_payout, 50
        )
        products.extend(steinmann_products)
        
        # 添加線性產品
        non_zero_indices = parametric_indices[parametric_indices > 0]
        if len(non_zero_indices) > 0:
            min_threshold = np.percentile(non_zero_indices, 10)
            max_threshold = np.percentile(non_zero_indices, 90)
            
            for i in range(10):
                linear_product = self.parametric_engine.create_parametric_product(
                    product_id=f"linear_{i}",
                    name=f"Linear Product {i}",
                    description=f"線性產品_{i}",
                    index_type=ParametricIndexType.CAT_IN_CIRCLE,
                    payout_function_type=PayoutFunctionType.LINEAR,
                    trigger_thresholds=[min_threshold, max_threshold],
                    payout_amounts=[0, max_payout],
                    max_payout=max_payout,
                    category='linear'
                )
                products.append(linear_product)
        
        return products
    
    def _identify_best_products(self, optimization_results: pd.DataFrame, 
                              products: List[ParametricProduct]) -> Dict[str, ParametricProduct]:
        """識別最佳產品"""
        best_products = {}
        
        if not optimization_results.empty:
            # 按不同標準找最佳產品
            criteria_mapping = {
                'best_overall': 'composite_score',
                'best_rmse': 'rmse',
                'best_correlation': 'correlation',
                'best_coverage': 'coverage_ratio'
            }
            
            products_dict = {p.product_id: p for p in products}
            
            for label, criterion in criteria_mapping.items():
                if criterion in optimization_results.columns:
                    if criterion == 'rmse':
                        best_idx = optimization_results[criterion].idxmin()
                    else:
                        best_idx = optimization_results[criterion].idxmax()
                    
                    best_product_id = optimization_results.loc[best_idx, 'product_id']
                    if best_product_id in products_dict:
                        best_products[label] = products_dict[best_product_id]
        
        return best_products
    
    def _generate_summary_statistics(self, products: List[ParametricProduct], 
                                   performance_df: pd.DataFrame,
                                   optimization_df: pd.DataFrame,
                                   skill_scores: Dict[str, Dict]) -> Dict[str, Any]:
        """生成摘要統計"""
        summary = {
            'total_products': len(products),
            'product_categories': {},
            'performance_summary': {},
            'optimization_summary': {},
            'skill_score_summary': {}
        }
        
        # 產品類別統計
        categories = {}
        for product in products:
            category = product.metadata.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        summary['product_categories'] = categories
        
        # 績效摘要
        if not performance_df.empty:
            summary['performance_summary'] = {
                'best_rmse': performance_df['rmse'].min(),
                'worst_rmse': performance_df['rmse'].max(),
                'mean_rmse': performance_df['rmse'].mean(),
                'best_correlation': performance_df['correlation'].max(),
                'mean_correlation': performance_df['correlation'].mean(),
                'triggered_products': len(performance_df[performance_df['payout_frequency'] > 0])
            }
        
        # 優化摘要
        if not optimization_df.empty:
            summary['optimization_summary'] = {
                'best_composite_score': optimization_df['composite_score'].max(),
                'mean_composite_score': optimization_df['composite_score'].mean(),
                'top_10_mean_score': optimization_df.head(10)['composite_score'].mean()
            }
        
        # 技能評分摘要
        if skill_scores:
            score_summaries = {}
            for score_type in self.config.skill_scores:
                values = []
                for product_scores in skill_scores.values():
                    if score_type in product_scores:
                        values.append(product_scores[score_type].value)
                
                if values:
                    score_summaries[score_type.value] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            summary['skill_score_summary'] = score_summaries
        
        return summary
    
    def _generate_method_recommendations(self, comparison_results: Dict[str, Any]) -> List[str]:
        """生成方法建議"""
        recommendations = []
        
        method_perf = comparison_results.get('method_performance', {})
        
        if method_perf:
            # 找出最佳RMSE方法
            best_rmse_method = min(method_perf.keys(), 
                                 key=lambda x: method_perf[x]['best_rmse'])
            recommendations.append(f"RMSE表現最佳: {best_rmse_method}")
            
            # 找出最佳相關性方法
            best_corr_method = max(method_perf.keys(), 
                                 key=lambda x: method_perf[x]['best_correlation'])
            recommendations.append(f"相關性表現最佳: {best_corr_method}")
            
            # 觸發率分析
            for method, stats in method_perf.items():
                trigger_rate = stats['triggered_products'] / stats['n_products'] * 100
                recommendations.append(f"{method}產品觸發率: {trigger_rate:.1f}%")
        
        return recommendations
    
    def _generate_bayesian_summary_statistics(self, products: List[ParametricProduct], 
                                            performance_df: pd.DataFrame,
                                            bayesian_results: Dict[str, 'BayesianAnalysisResult'],
                                            robustness_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成貝氏分析摘要統計"""
        summary = {
            'total_products': len(products),
            'product_categories': {},
            'bayesian_summary': {},
            'robustness_summary': {},
            'comparison_summary': {}
        }
        
        # 產品類別統計
        categories = {}
        for product in products:
            category = product.metadata.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        summary['product_categories'] = categories
        
        # 貝氏分析摘要
        if bayesian_results:
            all_crps_scores = []
            scenario_summaries = {}
            
            for scenario_key, result in bayesian_results.items():
                crps_values = list(result.crps_scores.values())
                scenario_summaries[scenario_key] = {
                    'best_crps': min(crps_values) if crps_values else 0,
                    'mean_crps': np.mean(crps_values) if crps_values else 0,
                    'worst_crps': max(crps_values) if crps_values else 0
                }
                all_crps_scores.extend(crps_values)
            
            summary['bayesian_summary'] = {
                'n_scenarios': len(bayesian_results),
                'overall_best_crps': min(all_crps_scores) if all_crps_scores else 0,
                'overall_mean_crps': np.mean(all_crps_scores) if all_crps_scores else 0,
                'scenario_details': scenario_summaries
            }
        
        # 穩健性分析摘要
        if robustness_results:
            robust_products = robustness_results.get('robust_products', [])
            robustness_metrics = robustness_results.get('robustness_metrics', {})
            
            summary['robustness_summary'] = {
                'n_robust_products': len(robust_products),
                'robust_products': robust_products,  # 全部健壯產品
                'avg_robustness_cv': np.mean([m.get('coefficient_of_variation', 0) 
                                            for m in robustness_metrics.values()]) if robustness_metrics else 0
            }
        
        # 傳統vs貝氏比較
        if not performance_df.empty and bayesian_results:
            traditional_best_rmse = performance_df['rmse'].min()
            traditional_best_corr = performance_df['correlation'].max()
            
            summary['comparison_summary'] = {
                'traditional_best_rmse': traditional_best_rmse,
                'traditional_best_correlation': traditional_best_corr,
                'bayesian_provides_robustness': len(robustness_results.get('robust_products', [])) > 0 if robustness_results else False
            }
        
        return summary
    
    def _print_bayesian_summary_report(self, results: AnalysisResults) -> None:
        """打印貝氏分析摘要報告"""
        print(f"\n📋 穩健貝氏分析摘要報告")
        print("=" * 60)
        
        summary = results.summary_statistics
        
        print(f"📊 基本統計:")
        print(f"   總產品數: {summary['total_products']}")
        print(f"   產品類別: {list(summary['product_categories'].keys())}")
        
        # 貝氏分析結果
        if 'bayesian_summary' in summary and summary['bayesian_summary']:
            bayes_summary = summary['bayesian_summary']
            print(f"\n🎲 貝氏分析結果:")
            print(f"   分析情境數: {bayes_summary['n_scenarios']}")
            print(f"   整體最佳CRPS: {bayes_summary['overall_best_crps']:.4f}")
            print(f"   整體平均CRPS: {bayes_summary['overall_mean_crps']:.4f}")
            
            print(f"\n   各情境表現:")
            for scenario, stats in bayes_summary.get('scenario_details', {}).items():
                print(f"     {scenario}: 最佳={stats['best_crps']:.4f}, 平均={stats['mean_crps']:.4f}")
        
        # 穩健性分析結果
        if 'robustness_summary' in summary and summary['robustness_summary']:
            robust_summary = summary['robustness_summary']
            print(f"\n🛡️ 穩健性分析結果:")
            print(f"   穩健產品數: {robust_summary['n_robust_products']}")
            print(f"   平均變異係數: {robust_summary['avg_robustness_cv']:.3f}")
            
            if robust_summary['robust_products']:
                print(f"   頂級穩健產品:")
                for i, product_id in enumerate(robust_summary['robust_products'], 1):
                    print(f"     {i}. {product_id}")
        
        # 方法比較
        if 'comparison_summary' in summary and summary['comparison_summary']:
            comp_summary = summary['comparison_summary']
            print(f"\n⚖️ 方法比較:")
            print(f"   傳統最佳RMSE: ${comp_summary['traditional_best_rmse']/1e9:.3f}B")
            print(f"   傳統最高相關性: {comp_summary['traditional_best_correlation']:.3f}")
            print(f"   貝氏方法提供穩健性: {'是' if comp_summary['bayesian_provides_robustness'] else '否'}")
        
        # 最佳產品
        if results.best_products:
            print(f"\n🏆 最佳產品推薦:")
            for label, product in results.best_products.items():
                print(f"   {label}: {product.name}")
    
    def _print_summary_report(self, results: AnalysisResults) -> None:
        """打印摘要報告"""
        print(f"\n📋 分析摘要報告")
        print("=" * 50)
        
        summary = results.summary_statistics
        
        print(f"📊 基本統計:")
        print(f"   總產品數: {summary['total_products']}")
        print(f"   產品類別: {list(summary['product_categories'].keys())}")
        
        if 'performance_summary' in summary:
            perf = summary['performance_summary']
            print(f"\n🎯 績效摘要:")
            print(f"   最佳RMSE: ${perf['best_rmse']/1e9:.3f}B")
            print(f"   最高相關性: {perf['best_correlation']:.3f}")
            print(f"   觸發產品數: {perf['triggered_products']}")
        
        if results.best_products:
            print(f"\n🏆 最佳產品:")
            for label, product in results.best_products.items():
                print(f"   {label}: {product.name}")
    
    def _print_comparison_report(self, comparison_results: Dict[str, Any]) -> None:
        """打印比較報告"""
        print(f"\n📊 方法比較報告")
        print("=" * 50)
        
        method_perf = comparison_results.get('method_performance', {})
        
        for method, stats in method_perf.items():
            print(f"\n{method}:")
            print(f"   產品數: {stats['n_products']}")
            print(f"   最佳RMSE: ${stats['best_rmse']/1e9:.3f}B")
            print(f"   最高相關性: {stats['best_correlation']:.3f}")
            print(f"   觸發產品: {stats['triggered_products']}")
        
        recommendations = comparison_results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 建議:")
            for rec in recommendations:
                print(f"   • {rec}")
    
    # Merged from unified_product_engine.py - Input Adapter Integration Methods
    def design_parametric_products_with_adapter(self,
                                               input_adapter: InputAdapter,
                                               config_override: Optional[AnalysisConfig] = None) -> AnalysisResults:
        """
        使用輸入適配器進行統一的產品設計流程
        Merged from unified_product_engine.py
        
        Parameters:
        -----------
        input_adapter : InputAdapter
            輸入適配器 (CLIMADA或Bayesian)
        config_override : AnalysisConfig, optional
            覆蓋配置
            
        Returns:
        --------
        AnalysisResults
            統一的產品設計結果
        """
        import time
        start_time = time.time()
        config = config_override or self.config
        
        print(f"\n🏭 開始統一產品設計流程")
        print(f"   輸入類型: {input_adapter.get_input_type()}")
        print(f"   評估模式: {config.evaluation_mode.value}")
        print("=" * 60)
        
        try:
            # Step 1: 提取參數指標
            print("\n📊 Step 1: 提取參數指標...")
            parametric_indices_dict = input_adapter.extract_parametric_indices()
            observed_losses = input_adapter.get_loss_data()
            event_metadata = input_adapter.get_event_metadata()
            
            # Convert dict to main array for compatibility
            main_indices = list(parametric_indices_dict.values())[0]
            
            print(f"   提取了 {len(parametric_indices_dict)} 種參數指標")
            print(f"   事件數量: {len(observed_losses)}")
            print(f"   總損失: ${np.sum(observed_losses)/1e9:.2f}B")
            
            # Step 2: 使用統一框架進行分析
            print(f"\n🎯 Step 2: 執行{config.evaluation_mode.value}評估...")
            if config.evaluation_mode == EvaluationMode.TRADITIONAL or config.evaluation_mode == EvaluationMode.HYBRID:
                results = self.run_steinmann_analysis(main_indices, observed_losses)
            elif config.evaluation_mode == EvaluationMode.PROBABILISTIC:
                results = self.run_comprehensive_analysis(main_indices, observed_losses)
            else:
                results = self.run_comprehensive_analysis(main_indices, observed_losses)
            
            # Update metadata with adapter information
            results.metadata.update({
                'input_adapter_type': input_adapter.get_input_type(),
                'parametric_indices_types': list(parametric_indices_dict.keys()),
                'computation_time': time.time() - start_time,
                'event_metadata': event_metadata.to_dict() if hasattr(event_metadata, 'to_dict') else {}
            })
            
            # 更新統計
            self.run_statistics['total_analyses'] += 1
            self.run_statistics['successful_runs'] += 1
            computation_time = time.time() - start_time
            self.run_statistics['average_computation_time'] = (
                (self.run_statistics['average_computation_time'] * (self.run_statistics['total_analyses'] - 1) + 
                 computation_time) / self.run_statistics['total_analyses']
            )
            
            print(f"\n✅ 統一產品設計完成!")
            print(f"   計算時間: {computation_time:.2f}秒")
            print(f"   生成產品: {len(results.products)}個")
            
            return results
            
        except Exception as e:
            self.run_statistics['total_analyses'] += 1
            self.run_statistics['failed_runs'] += 1
            
            print(f"\n❌ 產品設計失敗: {e}")
            import traceback
            traceback.print_exc()
            
            # Return minimal results with error info
            empty_results = AnalysisResults(
                analysis_type=config.analysis_type,
                products=[],
                performance_results=pd.DataFrame(),
                skill_score_results={},
                optimization_results=pd.DataFrame(),
                technical_premium_results=pd.DataFrame(),
                best_products={},
                summary_statistics={'error': str(e)},
                metadata={'computation_time': time.time() - start_time}
            )
            return empty_results
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """獲取引擎運行統計 - Merged from unified_product_engine.py"""
        return self.run_statistics.copy()
    
    def reset_statistics(self):
        """重置統計 - Merged from unified_product_engine.py"""
        self.run_statistics = {
            'total_analyses': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'average_computation_time': 0.0
        }
    
    def _generate_steinmann_products_helper(self, parametric_indices, max_payout, max_products=70, observed_losses=None):
        """Helper method to generate Steinmann products using specialized implementation"""
        from .saffir_simpson_products import generate_steinmann_2023_products
        
        # 使用基於損失的閾值來生成產品
        steinmann_structures, metadata = generate_steinmann_2023_products(
            observed_losses=observed_losses, 
            loss_based_thresholds=True
        )
        
        # Convert to ParametricProduct objects
        products = []
        for structure in steinmann_structures[:max_products]:  # Limit to max_products
            product = self.parametric_engine.create_parametric_product(
                product_id=structure.product_id,
                name=f"Steinmann Product {structure.product_id}",
                description=f"{structure.structure_type} threshold product",
                index_type=ParametricIndexType.CAT_IN_CIRCLE,
                payout_function_type=PayoutFunctionType.STEP,
                trigger_thresholds=structure.thresholds,
                payout_amounts=[structure.max_payout * p for p in structure.payouts],
                max_payout=structure.max_payout
            )
            products.append(product)
        
        return products