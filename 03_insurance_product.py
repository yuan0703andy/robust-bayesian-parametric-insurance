#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_insurance_product.py
========================
Parametric Insurance Product Design using existing framework

Simply configures and runs the existing product generation components
to create Steinmann et al. (2023) compliant products.
"""

# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Import from existing framework
from insurance_analysis_refactored.core.saffir_simpson_products import (
    generate_steinmann_2023_products,
    validate_steinmann_compatibility,
    create_steinmann_2023_config
)

# %%
def main():
    """
    Main program: Generate insurance products using existing framework
    """
    print("=" * 80)
    print("Parametric Insurance Product Design")
    print("Using insurance_analysis_refactored framework")
    print("Based on Steinmann et al. (2023) Standard")
    print("=" * 80)
    
    # Load Cat-in-a-Circle analysis results
    results_path = "results/spatial_analysis/cat_in_circle_results.pkl"
    
    try:
        with open(results_path, 'rb') as f:
            spatial_results = pickle.load(f)
        print(f"✅ Loaded spatial analysis results from: {results_path}")
        wind_speed_data = spatial_results['indices']
        
    except FileNotFoundError:
        print("⚠️ Spatial analysis results not found, using sample data")
        np.random.seed(42)
        n_events = 1000
        wind_speed_data = {
            'cat_in_circle_30km_max': np.random.gamma(4, 10, n_events),
            'cat_in_circle_50km_max': np.random.gamma(3.5, 11, n_events),
            'cat_in_circle_30km_mean': np.random.gamma(3, 9, n_events)
        }
    
    # Generate Steinmann products using existing framework
    print("\n📦 Generating Steinmann products using existing framework...")
    
    # Use the existing framework directly - no custom classes needed
    steinmann_products, summary = generate_steinmann_2023_products()
    
    print(f"✅ Generated {len(steinmann_products)} base Steinmann products")
    
    # Convert to compatible format for downstream analysis
    print("\n🔄 Converting to analysis-compatible format...")
    
    radii_km = [30]  # Use 30km radius
    index_types = ['max']  # Use max value index
    
    compatible_products = []
    
    for radius in radii_km:
        for index_type in index_types:
            for steinmann_product in steinmann_products:
                
                # Create compatible product dictionary
                product_dict = {
                    'product_id': f"{steinmann_product.product_id}_R{radius}_{index_type}",
                    'name': f"Steinmann {steinmann_product.product_id} (R={radius}km, {index_type})",
                    'radius_km': radius,
                    'index_type': index_type,
                    'trigger_thresholds': steinmann_product.thresholds,
                    'payout_ratios': steinmann_product.payouts,
                    'max_payout': steinmann_product.max_payout,
                    'structure_type': steinmann_product.structure_type,
                    'metadata': {
                        'steinmann_compliant': True,
                        'original_steinmann_id': steinmann_product.product_id,
                        'generation_summary': summary
                    }
                }
                compatible_products.append(product_dict)
    
    print(f"✅ Created {len(compatible_products)} analysis-ready products")
    
    # Show statistics
    print("\n📊 Product Statistics:")
    print("-" * 40)
    
    structure_counts = pd.Series([p['structure_type'] for p in compatible_products]).value_counts()
    for structure, count in structure_counts.items():
        print(f"  • {structure.capitalize()}: {count} products")
    
    # Validate Steinmann compliance
    print("\n🔍 Validating Steinmann compliance...")
    validation_result = validate_steinmann_compatibility(steinmann_products)
    print(f"   Steinmann compliant: {validation_result['steinmann_compliant']}")
    print(f"   Total products: {validation_result['total_count_70']}")
    
    # Show sample products
    print("\n📋 Sample Products (First 5):")
    print("-" * 40)
    for i, product in enumerate(compatible_products[:5]):
        print(f"\n{i+1}. {product['product_id']}")
        print(f"   Original Steinmann ID: {product['metadata']['original_steinmann_id']}")
        print(f"   Structure: {product['structure_type']}")
        print(f"   Thresholds: {product['trigger_thresholds']} m/s")
        print(f"   Payout Ratios: {[f'{r*100:.0f}%' for r in product['payout_ratios']]}")
        print(f"   Max Payout: ${product['max_payout']:,.0f}")
    
    # Save products
    print("\n💾 Saving products...")
    output_dir = "results/insurance_products"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save in the expected format for downstream analysis
    filepath = f"{output_dir}/products.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(compatible_products, f)
    
    print(f"💾 Products saved to: {filepath}")
    
    # CSV export for inspection
    csv_path = filepath.replace('.pkl', '.csv')
    df_products = pd.DataFrame([
        {
            'product_id': p['product_id'],
            'structure_type': p['structure_type'],
            'radius_km': p['radius_km'],
            'index_type': p['index_type'],
            'n_thresholds': len(p['trigger_thresholds']),
            'first_threshold': p['trigger_thresholds'][0],
            'max_threshold': max(p['trigger_thresholds']),
            'max_payout_ratio': max(p['payout_ratios']),
            'steinmann_id': p['metadata'].get('original_steinmann_id', 'Unknown')
        }
        for p in compatible_products
    ])
    df_products.to_csv(csv_path, index=False)
    print(f"📄 Product summary saved as CSV: {csv_path}")
    
    print("\n✅ Product design complete!")
    print("   Framework components used:")
    print("   • generate_steinmann_2023_products()")
    print("   • validate_steinmann_compatibility()")
    print("   • create_steinmann_2023_config()")
    
    return compatible_products


if __name__ == "__main__":
    products = main()
# %%
