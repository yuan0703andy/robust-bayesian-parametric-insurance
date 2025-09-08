"""Configuration classes moved from toy_example_complete.py."""
from typing import Dict, List
from enum import Enum
import numpy as np

class PriorScenario(Enum):
    """先驗情境"""
    NON_INFORMATIVE = "non_informative"
    WEAK_INFORMATIVE = "weak_informative"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"

class LikelihoodFamily(Enum):
    """似然函數族"""
    NORMAL = "normal"
    LOGNORMAL = "lognormal" 
    STUDENT_T = "student_t"


class ModelConfiguration:
    """模型配置類 - 全面的Prior/Likelihood組合測試框架"""
    
    @staticmethod 
    def get_comprehensive_test_configs() -> List[Dict]:
        """
        獲取全面的Prior/Likelihood組合測試配置
        4種先驗情境 × 3種似然族 × 3種穩健程度 = 36種組合測試矩陣
        """
        configs = []
        
        # 定義基礎組合
        prior_scenarios = [
            (PriorScenario.NON_INFORMATIVE, "非信息"),
            (PriorScenario.WEAK_INFORMATIVE, "弱信息"),
            (PriorScenario.OPTIMISTIC, "樂觀"),
            (PriorScenario.PESSIMISTIC, "悲觀")
        ]
        
        likelihood_families = [
            (LikelihoodFamily.NORMAL, "正態"),
            (LikelihoodFamily.LOGNORMAL, "對數正態"),
            (LikelihoodFamily.STUDENT_T, "Student-t")
        ]
        
        # 3種穩健程度的配置
        robustness_levels = [
            {
                'level': '基線(無污染)',
                'epsilon_prior': 0.0,
                'epsilon_likelihood': 0.0,
                'category': 'baseline'
            },
            {
                'level': '中等穩健',
                'epsilon_prior': 0.08,
                'epsilon_likelihood': 0.10,
                'category': 'moderate'
            },
            {
                'level': '極高穩健',
                'epsilon_prior': 0.15,
                'epsilon_likelihood': 0.18,
                'category': 'extreme'
            }
        ]
        
        # 生成所有組合
        for prior_scenario, prior_name in prior_scenarios:
            for likelihood_family, likelihood_name in likelihood_families:
                for robustness in robustness_levels:
                    
                    # 創建配置名稱
                    config_name = f"{robustness['level']}-{prior_name}先驗+{likelihood_name}似然"
                    
                    # 生成描述
                    description = f"{prior_name}先驗 + {likelihood_name}似然"
                    if robustness['category'] != 'baseline':
                        description += f" + ε-contamination({robustness['epsilon_prior']:.2f}, {robustness['epsilon_likelihood']:.2f})"
                    
                    config = {
                        'name': config_name,
                        'prior_scenario': prior_scenario,
                        'likelihood_family': likelihood_family,
                        'epsilon_prior': robustness['epsilon_prior'],
                        'epsilon_likelihood': robustness['epsilon_likelihood'],
                        'robustness_category': robustness['category'],
                        'description': description
                    }
                    
                    configs.append(config)
        
        # 選擇代表性的配置進行演示（避免過多組合）
        representative_configs = [
            # 基線對照組
            configs[0],   # 非信息+正態+無污染
            configs[3],   # 非信息+對數正態+無污染
            
            # 不同先驗的中等穩健組合
            configs[12],  # 非信息+正態+中等穩健
            configs[21],  # 弱信息+對數正態+中等穩健
            configs[30],  # 樂觀+Student-t+中等穩健
            configs[33],  # 悲觀+正態+中等穩健
            
            # 極高穩健組合
            configs[14],  # 非信息+Student-t+極高穩健
            configs[35],  # 悲觀+對數正態+極高穩健
        ]
        
        return representative_configs
    
    @staticmethod 
    def get_test_configs() -> List[Dict]:
        """獲取簡化版測試配置 - 向後兼容"""
        return [
            {
                'name': '傳統貝葉斯模型 (無污染)',
                'epsilon_prior': 0.0,
                'epsilon_likelihood': 0.0,
                'prior_contamination': 'none',
                'likelihood_contamination': 'none',
                'description': '標準貝葉斯模型，作為基線對照組'
            },
            {
                'name': '僅Prior污染模型', 
                'epsilon_prior': 0.08,
                'epsilon_likelihood': 0.0,
                'prior_contamination': 'typhoon_specific',
                'likelihood_contamination': 'none',
                'description': '僅對先驗進行ε-contamination，測試先驗穩健性'
            },
            {
                'name': '雙重污染模型 (Prior+Likelihood)',
                'epsilon_prior': 0.08,
                'epsilon_likelihood': 0.12,
                'prior_contamination': 'typhoon_specific', 
                'likelihood_contamination': 'extreme_events',
                'description': '先驗+似然雙重污染，最大穩健性配置'
            }
        ]
    
    @staticmethod
    def get_steinmann_product_configs() -> List[Dict]:
        """獲取Steinmann保險產品配置（與資料量級對齊）"""
        return [
            {
                'name': 'Standard Multi-Level (scaled)',
                'thresholds': [0.2e6, 0.4e6, 0.6e6, 0.8e6],  # 20萬~80萬
                'ratios':     [0.25,  0.5,   0.75,  1.0],
                'max_payout': 2e6,                            # 上限 200 萬
                'steepness':  0.2                              # 較平滑，避免梯度硬切
            },
            {
                'name': 'Dual Threshold Product (scaled)', 
                'thresholds': [0.3e6, 0.6e6, 999e6, 999e6],
                'ratios': [0.5, 1.0, 0.0, 0.0],
                'max_payout': 2e6,
                'steepness': 0.2
            },
            {
                'name': 'Multi-Level Product (wide)',
                'thresholds': [0.1e6, 0.3e6, 0.5e6, 0.7e6],
                'ratios': [0.25, 0.5, 0.75, 1.0],
                'max_payout': 2e6,
                'steepness': 0.2
            }
        ]

    @staticmethod
    def build_param_target_config(hazards_ms: np.ndarray,
                                  distance_matrix_km: np.ndarray,
                                  per_site_limit: float,
                                  given_radius_km: float = None) -> dict:
        """
        依資料自動校準 cat-in-circle 硬條款：
          - 半徑：預設取醫院間距中位數（可覆寫）
          - trigger：圓內 site 最大風速的 60 分位
          - exhaustion：85 分位（至少高於 trigger 5 m/s）
          - site_limits：每院上限 = per_site_limit（建議 = product_config['max_payout']）
          - payout_cap：事件層上限 = H * per_site_limit
        """
        import torch
        
        hazards = torch.as_tensor(hazards_ms, dtype=torch.float32)   # [H,E]
        H, E = hazards.shape
        D = torch.as_tensor(distance_matrix_km, dtype=torch.float32) # [H,H]

        R_km = float(np.median(D[D.numpy()>0])) if given_radius_km is None else float(given_radius_km)
        mask = (D <= R_km).float()

        big_neg = torch.tensor(-1e9, dtype=hazards.dtype)
        masked  = hazards.unsqueeze(0).expand(H,-1,-1).clone() + (mask.unsqueeze(-1)-1.0) * (-big_neg)
        I_site  = masked.max(dim=1).values   # [H,E] 每 site 的圓內硬 max
        I_evt   = I_site.max(dim=0).values   # [E] 事件層代表強度（最保守）

        q60 = torch.quantile(I_evt, 0.60).item()
        q85 = torch.quantile(I_evt, 0.85).item()
        trigger    = float(q60)
        exhaustion = float(max(q85, trigger + 5.0))

        site_limits = np.full(H, float(per_site_limit), dtype=np.float32)
        payout_cap  = float(H * per_site_limit)

        print(f"🎯 參數型條款自動配置:")
        print(f"   半徑: {R_km:.1f} km | 觸發: {trigger:.1f} m/s | 耗盡: {exhaustion:.1f} m/s")
        print(f"   單院限額: ${per_site_limit/1e6:.1f}M | 事件上限: ${payout_cap/1e6:.1f}M")

        return dict(
            radius_km=R_km,
            trigger_ms=trigger,
            exhaustion_ms=exhaustion,
            site_limits=site_limits,
            payout_cap=payout_cap,
            smooth_tau=0.3,  # param target 用於 y_target（可微時用），不影響這裡的硬條款
        )

print("✅ 配置類定義完成")

__all__ = [
    'ModelConfiguration'
]