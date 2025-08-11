#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檢查產品的max_payout值
"""

import pickle

# 載入產品
with open("results/insurance_products/products.pkl", 'rb') as f:
    products = pickle.load(f)

print("檢查產品的max_payout值：")
print("=" * 60)

# 按結構類型分組
from collections import defaultdict
by_type = defaultdict(list)

for p in products:
    by_type[p['structure_type']].append(p['max_payout'])

# 顯示每種類型的統計
for struct_type in ['single', 'double', 'triple', 'quadruple']:
    if struct_type in by_type:
        payouts = by_type[struct_type]
        unique_payouts = set(payouts)
        print(f"\n{struct_type.upper()}類型產品:")
        print(f"  產品數量: {len(payouts)}")
        print(f"  唯一賠付金額: {unique_payouts}")
        print(f"  最小值: {min(payouts):,.0f}")
        print(f"  最大值: {max(payouts):,.0f}")

# 顯示前10個產品的詳細資訊
print("\n前10個產品的詳細資訊:")
print("-" * 60)
for i in range(min(10, len(products))):
    p = products[i]
    print(f"{p['product_id']}: {p['structure_type']}, max_payout=${p['max_payout']:,.0f}")