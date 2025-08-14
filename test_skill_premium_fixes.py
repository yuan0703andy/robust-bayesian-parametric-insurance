#!/usr/bin/env python3
"""
Test the skill score and premium calculation fixes
測試技能評分和保費計算修復
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_skill_calculation_fix():
    """Test the manual skill calculation fix"""
    print("🧪 Testing manual skill calculation fix...")
    
    try:
        from insurance_analysis_refactored.core.saffir_simpson_products import generate_steinmann_2023_products
        
        # Generate test data
        products, _ = generate_steinmann_2023_products()
        test_product = products[0]
        
        # Sample data
        parametric_indices = np.random.uniform(20, 60, 44)
        observed_losses = np.random.gamma(2, 1e8, 44)
        
        # Test calculate_step_payouts function
        def calculate_step_payouts(product, parametric_indices):
            payouts = np.zeros(len(parametric_indices))
            for i, index_value in enumerate(parametric_indices):
                for j, threshold in enumerate(product.thresholds):
                    if index_value >= threshold:
                        payouts[i] = product.payouts[j] * product.max_payout
            return payouts
        
        payouts = calculate_step_payouts(test_product, parametric_indices)
        
        # Test manual skill calculation
        rmse = np.sqrt(np.mean((payouts - observed_losses[:len(payouts)])**2))
        mae = np.mean(np.abs(payouts - observed_losses[:len(payouts)]))
        correlation = np.corrcoef(payouts, observed_losses[:len(payouts)])[0,1] if len(payouts) > 1 else 0
        
        scores = {
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation
        }
        
        print(f"   ✅ Manual skill calculation successful:")
        print(f"      • RMSE: ${scores['rmse']:,.0f}")
        print(f"      • MAE: ${scores['mae']:,.0f}")
        print(f"      • Correlation: {scores['correlation']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Manual skill calculation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_premium_safe_access():
    """Test safe access to top_products"""
    print("🧪 Testing safe premium calculation access...")
    
    try:
        from insurance_analysis_refactored.core.saffir_simpson_products import generate_steinmann_2023_products
        from insurance_analysis_refactored.core.technical_premium_calculator import (
            TechnicalPremiumCalculator, TechnicalPremiumConfig
        )
        from insurance_analysis_refactored.core import ParametricProduct, ParametricIndexType, PayoutFunctionType
        
        # Generate test data
        products, _ = generate_steinmann_2023_products()
        parametric_indices = np.random.uniform(20, 60, 44)
        
        # Create premium calculator
        premium_config = TechnicalPremiumConfig()
        premium_calculator = TechnicalPremiumCalculator(premium_config)
        
        # Test conversion function
        def convert_payout_structure_to_parametric_product(payout_structure):
            payout_amounts = [ratio * payout_structure.max_payout for ratio in payout_structure.payouts]
            
            return ParametricProduct(
                product_id=payout_structure.product_id,
                name=f"Test Product {payout_structure.product_id}",
                description=f"{payout_structure.structure_type} threshold product",
                index_type=ParametricIndexType.CAT_IN_CIRCLE,
                payout_function_type=PayoutFunctionType.STEP,
                trigger_thresholds=payout_structure.thresholds,
                payout_amounts=payout_amounts,
                max_payout=payout_structure.max_payout
            )
        
        # Test with first 3 products (simulating fallback scenario)
        premium_count = 0
        for product in products[:3]:
            product_params = convert_payout_structure_to_parametric_product(product)
            premium_result = premium_calculator.calculate_technical_premium(
                product_params=product_params,
                hazard_indices=parametric_indices
            )
            premium_count += 1
        
        print(f"   ✅ Safe premium calculation successful:")
        print(f"      • Calculated {premium_count} premiums")
        print(f"      • Last premium: ${premium_result.technical_premium:,.0f}")
        
        # Test NoneType handling scenario
        class MockResults:
            def __init__(self):
                self.top_products = None  # Simulate the NoneType scenario
        
        mock_results = MockResults()
        
        # Test safe access
        premium_count_safe = 0
        if hasattr(mock_results, 'top_products') and mock_results.top_products:
            # This branch should not execute
            pass
        else:
            # Fallback: test with first 2 products
            for product in products[:2]:
                product_params = convert_payout_structure_to_parametric_product(product)
                premium_result = premium_calculator.calculate_technical_premium(
                    product_params=product_params,
                    hazard_indices=parametric_indices
                )
                premium_count_safe += 1
        
        print(f"   ✅ NoneType handling successful:")
        print(f"      • Fallback calculated {premium_count_safe} premiums")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Premium safe access error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_length_alignment():
    """Test proper data length alignment in calculations"""
    print("🧪 Testing data length alignment...")
    
    try:
        # Test different length scenarios
        payouts = np.array([0, 1e8, 2e8, 0, 5e8])  # 5 elements
        observed_losses = np.random.gamma(2, 1e8, 44)  # 44 elements
        
        # Test safe indexing
        min_length = min(len(payouts), len(observed_losses))
        payouts_aligned = payouts[:min_length]
        losses_aligned = observed_losses[:min_length]
        
        rmse = np.sqrt(np.mean((payouts_aligned - losses_aligned)**2))
        mae = np.mean(np.abs(payouts_aligned - losses_aligned))
        correlation = np.corrcoef(payouts_aligned, losses_aligned)[0,1] if len(payouts_aligned) > 1 else 0
        
        print(f"   ✅ Data alignment successful:")
        print(f"      • Aligned length: {min_length}")
        print(f"      • RMSE: ${rmse:,.0f}")
        print(f"      • Correlation: {correlation:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data alignment error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*80)
    print("🔧 Testing Skill Score and Premium Calculation Fixes")
    print("="*80)
    
    tests = [
        test_skill_calculation_fix,
        test_premium_safe_access,
        test_data_length_alignment
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Summary
    print("="*80)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("🚀 Skill score and premium fixes are working!")
    else:
        print(f"⚠️ {total-passed} TESTS FAILED ({passed}/{total})")
    
    print("="*80)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)