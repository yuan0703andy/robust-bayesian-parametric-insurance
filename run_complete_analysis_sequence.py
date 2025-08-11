#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_complete_analysis_sequence.py
==================================
完整分析序列執行器 - 按順序執行 05 → 06 → 07
Complete Analysis Sequence Runner - Execute 05 → 06 → 07 in sequence

自動按順序執行三個分析腳本，並確保數據流暢接軌
Automatically executes three analysis scripts in sequence with smooth data flow

Author: Research Team
Date: 2025-01-11
"""

import subprocess
import sys
import os
from pathlib import Path
import time


def run_script_with_logging(script_path, stage_name):
    """執行腳本並記錄輸出"""
    
    print(f"\n{'='*80}")
    print(f"🚀 執行 {stage_name}")
    print(f"   腳本: {script_path}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # 執行腳本
        result = subprocess.run(
            [sys.executable, script_path], 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(script_path),
            timeout=1800  # 30分鐘超時
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {stage_name} 執行成功！")
            print(f"⏱️ 執行時間: {duration:.2f} 秒")
            
            # 顯示部分輸出
            if result.stdout:
                lines = result.stdout.split('\n')
                print(f"\n📝 輸出摘要 (最後10行):")
                for line in lines[-10:]:
                    if line.strip():
                        print(f"   {line}")
            
            return True, result.stdout, result.stderr
            
        else:
            print(f"❌ {stage_name} 執行失敗！")
            print(f"   返回碼: {result.returncode}")
            print(f"   錯誤訊息: {result.stderr}")
            
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {stage_name} 執行超時 (30分鐘)")
        return False, "", "Execution timeout"
        
    except Exception as e:
        print(f"💥 {stage_name} 執行異常: {e}")
        return False, "", str(e)


def check_data_dependencies():
    """檢查數據依賴性"""
    
    print("🔍 檢查數據依賴性...")
    
    required_files = [
        "results/insurance_products/products.pkl",
        "results/spatial_analysis/cat_in_circle_results.pkl"
    ]
    
    optional_files = [
        "climada_complete_data.pkl"
    ]
    
    missing_required = []
    missing_optional = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_required.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    for file_path in optional_files:
        if not Path(file_path).exists():
            missing_optional.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    if missing_required:
        print(f"\n❌ 缺少必要檔案:")
        for file_path in missing_required:
            print(f"   - {file_path}")
        print("\n請先執行前置腳本:")
        print("   python 01_run_climada.py")
        print("   python 02_spatial_analysis.py") 
        print("   python 04_traditional_parm_insurance.py")
        return False
    
    if missing_optional:
        print(f"\n⚠️ 缺少可選檔案 (將使用模擬數據):")
        for file_path in missing_optional:
            print(f"   - {file_path}")
    
    return True


def create_execution_summary(results):
    """創建執行摘要報告"""
    
    summary_dir = Path("results/execution_summary") 
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = summary_dir / f"analysis_sequence_summary_{int(time.time())}.md"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# 完整分析序列執行摘要\n")
        f.write("# Complete Analysis Sequence Execution Summary\n\n")
        
        f.write(f"執行時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 執行結果 Execution Results\n\n")
        
        stage_names = [
            "Stage 1: Robust Bayesian Analysis 強健貝氏分析",
            "Stage 2: Weight Sensitivity Analysis 權重敏感度分析", 
            "Stage 3: Technical Premium Analysis 技術保費分析"
        ]
        
        for i, (stage_name, (success, stdout, stderr)) in enumerate(zip(stage_names, results)):
            status = "✅ 成功" if success else "❌ 失敗"
            f.write(f"### {stage_name}\n")
            f.write(f"**狀態**: {status}\n\n")
            
            if success:
                f.write("**輸出摘要**:\n```\n")
                if stdout:
                    # 取最後20行作為摘要
                    lines = stdout.split('\n')[-20:]
                    f.write('\n'.join(lines))
                f.write("\n```\n\n")
            else:
                f.write("**錯誤訊息**:\n```\n")
                f.write(stderr)
                f.write("\n```\n\n")
        
        # 整體結果
        all_success = all(result[0] for result in results)
        f.write("## 整體結果 Overall Result\n\n")
        f.write(f"**狀態**: {'✅ 全部成功' if all_success else '❌ 部分失敗'}\n")
        f.write(f"**成功率**: {sum(result[0] for result in results)}/{len(results)} ({100*sum(result[0] for result in results)/len(results):.1f}%)\n\n")
        
        if all_success:
            f.write("## 後續步驟 Next Steps\n\n")
            f.write("- 檢查 `results/` 目錄中的所有輸出檔案\n")
            f.write("- 回顧各階段的分析結果\n")
            f.write("- 根據分析結果制定保險產品策略\n")
            f.write("- 考慮實施建議並進行進一步測試\n")
    
    print(f"📋 執行摘要已保存: {summary_file}")
    return summary_file


def main():
    """主執行函數"""
    
    print("🚀 完整參數保險分析序列執行器")
    print("   Complete Parametric Insurance Analysis Sequence Runner")
    print("=" * 80)
    
    # 檢查數據依賴性
    if not check_data_dependencies():
        print("\n❌ 數據依賴性檢查失敗，無法繼續執行")
        return False
    
    print("\n✅ 數據依賴性檢查通過，開始執行分析序列...")
    
    # 定義執行序列
    scripts_to_run = [
        ("05_robust_bayesian_parm_insurance.py", "Stage 1: Robust Bayesian Analysis 強健貝氏分析"),
        ("06_sensitivity_analysis.py", "Stage 2: Weight Sensitivity Analysis 權重敏感度分析"),
        ("07_technical_premium_analysis.py", "Stage 3: Technical Premium Analysis 技術保費分析")
    ]
    
    results = []
    overall_start_time = time.time()
    
    # 按順序執行腳本
    for script_name, stage_name in scripts_to_run:
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            print(f"❌ 腳本不存在: {script_path}")
            results.append((False, "", f"Script not found: {script_path}"))
            continue
        
        success, stdout, stderr = run_script_with_logging(script_path, stage_name)
        results.append((success, stdout, stderr))
        
        # 如果某個階段失敗，詢問是否繼續
        if not success:
            response = input(f"\n{stage_name} 執行失敗，是否繼續執行下一階段？(y/N): ").lower()
            if response != 'y':
                print("❌ 用戶選擇停止執行")
                break
        
        # 階段間短暫停頓
        print(f"\n⏸️ 階段完成，等待 3 秒...")
        time.sleep(3)
    
    overall_duration = time.time() - overall_start_time
    
    # 顯示最終結果
    print("\n" + "=" * 80)
    print("📋 完整分析序列執行完成")
    print("=" * 80)
    
    successful_stages = sum(result[0] for result in results)
    total_stages = len(results)
    
    print(f"⏱️ 總執行時間: {overall_duration:.2f} 秒")
    print(f"✅ 成功階段: {successful_stages}/{total_stages}")
    print(f"📊 成功率: {100*successful_stages/total_stages:.1f}%")
    
    # 創建執行摘要
    summary_file = create_execution_summary(results)
    
    # 顯示各階段結果
    stage_names = ["強健貝氏分析", "權重敏感度分析", "技術保費分析"]
    for i, (success, _, _) in enumerate(results):
        status = "✅" if success else "❌"
        print(f"   {status} Stage {i+1}: {stage_names[i] if i < len(stage_names) else 'Unknown'}")
    
    if successful_stages == total_stages:
        print(f"\n🎉 所有分析階段執行成功！")
        print(f"📁 檢查 results/ 目錄查看完整結果")
        
        # 列出主要輸出目錄
        output_dirs = [
            "results/robust_hierarchical_bayesian_analysis/",
            "results/sensitivity_analysis_modular/", 
            "results/technical_premium_modular/"
        ]
        
        print(f"\n📂 主要輸出目錄:")
        for output_dir in output_dirs:
            if Path(output_dir).exists():
                print(f"   ✅ {output_dir}")
            else:
                print(f"   ❓ {output_dir} (可能在執行過程中未創建)")
        
        return True
    else:
        print(f"\n⚠️ 部分階段執行失敗，請檢查錯誤訊息並重新執行")
        print(f"📋 詳細執行摘要: {summary_file}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)