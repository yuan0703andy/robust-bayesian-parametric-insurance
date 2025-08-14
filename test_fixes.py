#!/usr/bin/env python3
"""
Test Fixes for 05_test.py and 05_robust_bayesian_unified.py
測試修復後的功能
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_technical_premium_fix():
    """Test TechnicalPremiumCalculator fix"""
    print("🧪 Testing TechnicalPremiumCalculator fix...")
    
    try:
        from insurance_analysis_refactored.core.technical_premium_calculator import (
            TechnicalPremiumCalculator, TechnicalPremiumConfig
        )
        
        # Create config and calculator
        config = TechnicalPremiumConfig()
        calculator = TechnicalPremiumCalculator(config)
        
        print("   ✅ TechnicalPremiumCalculator initialization successful")
        return True
        
    except Exception as e:
        print(f"   ❌ TechnicalPremiumCalculator error: {e}")
        return False

def test_payout_calculation_fix():
    """Test payout calculation fix"""
    print("🧪 Testing payout calculation fix...")
    
    try:
        from insurance_analysis_refactored.core.saffir_simpson_products import generate_steinmann_2023_products
        
        # Generate products
        products, summary = generate_steinmann_2023_products()
        print(f"   ✅ Generated {len(products)} products")
        
        # Test calculate_step_payouts function
        def calculate_step_payouts(product, parametric_indices):
            payouts = np.zeros(len(parametric_indices))
            for i, index_value in enumerate(parametric_indices):
                for j, threshold in enumerate(product.thresholds):
                    if index_value >= threshold:
                        payouts[i] = product.payouts[j] * product.max_payout
            return payouts
        
        # Test with sample data
        test_product = products[0]
        test_indices = np.array([20, 35, 45, 60])  # Sample wind speeds
        payouts = calculate_step_payouts(test_product, test_indices)
        
        print(f"   ✅ Payout calculation successful: payouts = {payouts}")
        return True
        
    except Exception as e:
        print(f"   ❌ Payout calculation error: {e}")
        return False

def test_product_conversion_fix():
    """Test product conversion fix"""
    print("🧪 Testing product conversion fix...")
    
    try:
        from insurance_analysis_refactored.core.saffir_simpson_products import generate_steinmann_2023_products
        from insurance_analysis_refactored.core import ParametricProduct, ParametricIndexType, PayoutFunctionType
        
        # Generate test product
        products, _ = generate_steinmann_2023_products()
        test_payout_structure = products[0]
        
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
        
        converted_product = convert_payout_structure_to_parametric_product(test_payout_structure)
        
        print(f"   ✅ Product conversion successful: {converted_product.product_id}")
        return True
        
    except Exception as e:
        print(f"   ❌ Product conversion error: {e}")
        return False

def test_integration():
    """Test integration of all fixes"""
    print("🧪 Testing integration of all fixes...")
    
    try:
        from insurance_analysis_refactored.core.saffir_simpson_products import generate_steinmann_2023_products
        from insurance_analysis_refactored.core.technical_premium_calculator import (
            TechnicalPremiumCalculator, TechnicalPremiumConfig
        )
        from insurance_analysis_refactored.core import ParametricProduct, ParametricIndexType, PayoutFunctionType
        
        # 1. Generate products
        products, _ = generate_steinmann_2023_products()
        test_product = products[0]
        
        # 2. Calculate payouts
        def calculate_step_payouts(product, parametric_indices):
            payouts = np.zeros(len(parametric_indices))
            for i, index_value in enumerate(parametric_indices):
                for j, threshold in enumerate(product.thresholds):
                    if index_value >= threshold:
                        payouts[i] = product.payouts[j] * product.max_payout
            return payouts
        
        test_indices = np.random.uniform(20, 60, 44)  # Sample data
        payouts = calculate_step_payouts(test_product, test_indices)
        
        # 3. Convert product
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
        
        parametric_product = convert_payout_structure_to_parametric_product(test_product)
        
        # 4. Calculate premium
        config = TechnicalPremiumConfig()
        calculator = TechnicalPremiumCalculator(config)
        
        premium_result = calculator.calculate_technical_premium(
            product_params=parametric_product,
            hazard_indices=test_indices
        )
        
        print(f"   ✅ Integration test successful:")
        print(f"      • Product: {parametric_product.product_id}")
        print(f"      • Payouts calculated: {len(payouts)} values")
        print(f"      • Premium: ${premium_result.technical_premium:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*80)
    print("🔧 Testing Fixes for 05_test.py and 05_robust_bayesian_unified.py")
    print("="*80)
    
    tests = [
        test_technical_premium_fix,
        test_payout_calculation_fix,
        test_product_conversion_fix,
        test_integration
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
        print("🚀 Ready to run 05_test.py and 05_robust_bayesian_unified.py!")
    else:
        print(f"⚠️ {total-passed} TESTS FAILED ({passed}/{total})")
    
    print("="*80)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)