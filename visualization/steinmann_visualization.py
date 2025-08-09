"""
Steinmann Analysis Visualization
===============================

創建Steinmann分析結果的可視化圖表，展示：
1. RMSE vs MAE基差風險散點圖
2. 風速閾值特徵分析
3. 觸發機率和賠付效率分析
4. 財務特徵比較
5. 環境因素（Saffir-Simpson等級）分析
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體和圖表樣式
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class SteinmannVisualization:
    """
    Steinmann Analysis Visualizer
    Steinmann分析視覺化器
    """
    
    def __init__(self, style: str = "whitegrid", figsize: tuple = (12, 8)):
        """
        初始化視覺化器
        
        Parameters:
        -----------
        style : str
            圖表樣式
        figsize : tuple
            預設圖表大小
        """
        self.style = style
        self.default_figsize = figsize
        
        # 設置樣式
        plt.style.use('default')
        sns.set_style(style)
    
    def plot_basis_risk_analysis(self, results_df, save_path=None):
        """繪製基差風險分析圖"""
        # 基本實現
        fig, ax = plt.subplots(figsize=self.default_figsize)
        ax.text(0.5, 0.5, 'Steinmann基差風險分析\n(需要實際數據)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Steinmann基差風險分析')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def load_results():
    """載入分析結果"""
    try:
        df = pd.read_excel("steinmann_comprehensive_results.xlsx", sheet_name="All_Products")
        return df
    except FileNotFoundError:
        print("❌ 結果文件不存在，請先運行 steinmann_comprehensive_analysis.py")
        return None

def create_rmse_mae_analysis(df):
    """創建RMSE vs MAE基差風險分析圖"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. RMSE vs MAE散點圖
    scatter = ax1.scatter(df['rmse']/1e9, df['mae']/1e9, 
                         c=df['trigger_frequency']*100, 
                         cmap='viridis', alpha=0.7, s=50)
    ax1.set_xlabel('RMSE Basis Risk (Billion $)')
    ax1.set_ylabel('MAE Basis Risk (Billion $)')
    ax1.set_title('RMSE vs MAE Basis Risk\n(Color: Trigger Frequency %)')
    
    # 添加最優產品標註
    best_rmse_idx = df['rmse'].idxmin()
    best_mae_idx = df['mae'].idxmin()
    
    ax1.scatter(df.loc[best_rmse_idx, 'rmse']/1e9, df.loc[best_rmse_idx, 'mae']/1e9, 
               color='red', s=100, marker='*', label='Best RMSE')
    ax1.scatter(df.loc[best_mae_idx, 'rmse']/1e9, df.loc[best_mae_idx, 'mae']/1e9, 
               color='orange', s=100, marker='*', label='Best MAE')
    ax1.legend()
    
    # 顏色條
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Trigger Frequency (%)')
    
    # 2. 產品類別的基差風險比較
    category_data = df.groupby('category').agg({
        'rmse': 'mean',
        'mae': 'mean'
    }) / 1e9
    
    x = range(len(category_data))
    width = 0.35
    
    ax2.bar([i - width/2 for i in x], category_data['rmse'], width, 
           label='RMSE', alpha=0.8, color='skyblue')
    ax2.bar([i + width/2 for i in x], category_data['mae'], width, 
           label='MAE', alpha=0.8, color='lightcoral')
    
    ax2.set_xlabel('Product Category')
    ax2.set_ylabel('Basis Risk (Billion $)')
    ax2.set_title('Average Basis Risk by Product Category')
    ax2.set_xticks(x)
    ax2.set_xticklabels([cat.replace('_', '\n') for cat in category_data.index], rotation=0)
    ax2.legend()
    
    # 3. 觸發頻率 vs 覆蓋效率
    scatter2 = ax3.scatter(df['trigger_frequency']*100, df['coverage_efficiency']*100,
                          c=df['rmse']/1e9, cmap='Reds', alpha=0.7, s=50)
    ax3.set_xlabel('Trigger Frequency (%)')
    ax3.set_ylabel('Coverage Efficiency (%)')
    ax3.set_title('Trigger Frequency vs Coverage Efficiency\n(Color: RMSE Basis Risk)')
    
    cbar2 = plt.colorbar(scatter2, ax=ax3)
    cbar2.set_label('RMSE (Billion $)')
    
    # 4. 風速閾值分佈
    # 提取最小和最大閾值
    min_thresholds = []
    max_thresholds = []
    
    for _, row in df.iterrows():
        thresholds = eval(row['wind_thresholds']) if isinstance(row['wind_thresholds'], str) else row['wind_thresholds']
        min_thresholds.append(min(thresholds))
        max_thresholds.append(max(thresholds))
    
    ax4.hist(min_thresholds, bins=20, alpha=0.5, label='Min Threshold', color='lightblue')
    ax4.hist(max_thresholds, bins=20, alpha=0.5, label='Max Threshold', color='lightgreen')
    ax4.set_xlabel('Wind Speed (m/s)')
    ax4.set_ylabel('Number of Products')
    ax4.set_title('Distribution of Wind Speed Thresholds')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('steinmann_rmse_mae_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_financial_analysis(df):
    """創建財務特徵分析圖"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 最大賠付 vs 基差風險
    scatter1 = ax1.scatter(df['max_possible_payout']/1e9, df['rmse']/1e9,
                          c=df['loss_ratio'], cmap='plasma', alpha=0.7, s=50)
    ax1.set_xlabel('Maximum Possible Payout (Billion $)')
    ax1.set_ylabel('RMSE Basis Risk (Billion $)')
    ax1.set_title('Maximum Payout vs Basis Risk\n(Color: Loss Ratio)')
    
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Loss Ratio')
    
    # 2. 商業保費 vs 觸發頻率
    ax2.scatter(df['commercial_premium']/1e6, df['trigger_frequency']*100,
               alpha=0.7, s=50, color='green')
    ax2.set_xlabel('Commercial Premium (Million $)')
    ax2.set_ylabel('Trigger Frequency (%)')
    ax2.set_title('Premium vs Trigger Frequency')
    
    # 3. 賠付效率分佈
    ax3.hist(df['payout_efficiency'], bins=30, alpha=0.7, color='orange')
    ax3.set_xlabel('Payout Efficiency')
    ax3.set_ylabel('Number of Products')
    ax3.set_title('Distribution of Payout Efficiency')
    ax3.axvline(df['payout_efficiency'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["payout_efficiency"].mean():.3f}')
    ax3.legend()
    
    # 4. 損失比率 vs 基差風險
    ax4.scatter(df['loss_ratio'], df['rmse']/1e9, alpha=0.7, s=50, color='purple')
    ax4.set_xlabel('Loss Ratio')
    ax4.set_ylabel('RMSE Basis Risk (Billion $)')
    ax4.set_title('Loss Ratio vs Basis Risk')
    
    # 添加理想區域（損失比率接近1）
    ax4.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Loss Ratio')
    ax4.fill_betweenx(ax4.get_ylim(), 0.8, 1.2, alpha=0.2, color='green', 
                     label='Good Range (0.8-1.2)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('steinmann_financial_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_saffir_simpson_analysis(df):
    """創建Saffir-Simpson等級分析圖"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Saffir-Simpson等級覆蓋統計
    saffir_coverage = {
        'TD': sum(df['lowest_category'] == 'TD'),
        'TS': sum(df['lowest_category'] == 'TS'),
        'H1': sum(df['lowest_category'] == 'H1'),
        'H2': sum(df['lowest_category'] == 'H2'),
        'H3': sum(df['lowest_category'] == 'H3'),
        'H4': sum(df['lowest_category'] == 'H4'),
        'H5': sum(df['lowest_category'] == 'H5')
    }
    
    categories = list(saffir_coverage.keys())
    counts = list(saffir_coverage.values())
    colors = ['lightblue', 'blue', 'green', 'yellow', 'orange', 'red', 'darkred']
    
    ax1.bar(categories, counts, color=colors, alpha=0.7)
    ax1.set_xlabel('Lowest Saffir-Simpson Category')
    ax1.set_ylabel('Number of Products')
    ax1.set_title('Product Distribution by Lowest Category')
    
    # 2. 主要颶風覆蓋能力 vs 基差風險
    major_hurricane_products = df[df['covers_major_hurricane'] == True]
    non_major_products = df[df['covers_major_hurricane'] == False]
    
    ax2.scatter(major_hurricane_products['rmse']/1e9, major_hurricane_products['mae']/1e9,
               alpha=0.7, label='Covers Major Hurricane (H3+)', color='red', s=50)
    ax2.scatter(non_major_products['rmse']/1e9, non_major_products['mae']/1e9,
               alpha=0.7, label='No Major Hurricane Coverage', color='blue', s=50)
    ax2.set_xlabel('RMSE Basis Risk (Billion $)')
    ax2.set_ylabel('MAE Basis Risk (Billion $)')
    ax2.set_title('Major Hurricane Coverage vs Basis Risk')
    ax2.legend()
    
    # 3. 氣候韌性評分分佈
    ax3.hist(df['climate_resilience_score'], bins=30, alpha=0.7, color='green')
    ax3.set_xlabel('Climate Resilience Score')
    ax3.set_ylabel('Number of Products')
    ax3.set_title('Distribution of Climate Resilience Scores')
    ax3.axvline(df['climate_resilience_score'].mean(), color='red', linestyle='--',
               label=f'Mean: {df["climate_resilience_score"].mean():.3f}')
    ax3.legend()
    
    # 4. 風速閾值範圍 vs 氣候韌性
    threshold_spans = []
    for _, row in df.iterrows():
        thresholds = eval(row['wind_thresholds']) if isinstance(row['wind_thresholds'], str) else row['wind_thresholds']
        threshold_spans.append(max(thresholds) - min(thresholds))
    
    ax4.scatter(threshold_spans, df['climate_resilience_score'], alpha=0.7, s=50)
    ax4.set_xlabel('Threshold Span (max - min wind speed)')
    ax4.set_ylabel('Climate Resilience Score')
    ax4.set_title('Threshold Span vs Climate Resilience')
    
    plt.tight_layout()
    plt.savefig('steinmann_saffir_simpson_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_top_products_summary(df):
    """創建最優產品摘要表"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # 選擇前10個最優產品（按RMSE排序）
    top_products = df.nsmallest(10, 'rmse')
    
    # 準備表格數據
    table_data = []
    for i, (_, row) in enumerate(top_products.iterrows()):
        thresholds = eval(row['wind_thresholds']) if isinstance(row['wind_thresholds'], str) else row['wind_thresholds']
        
        table_data.append([
            f"{i+1}",
            row['category'].replace('_', ' ').title(),
            f"{thresholds}",
            f"${row['rmse']/1e9:.3f}B",
            f"${row['mae']/1e9:.3f}B",
            f"{row['trigger_frequency']*100:.1f}%",
            f"{row['coverage_efficiency']*100:.1f}%",
            f"${row['max_possible_payout']/1e9:.1f}B",
            f"{row['lowest_category']}-{row['highest_category']}",
            f"{row['loss_ratio']:.2f}"
        ])
    
    columns = [
        'Rank', 'Category', 'Wind Thresholds\n(m/s)', 'RMSE\nBasis Risk', 'MAE\nBasis Risk',
        'Trigger\nFreq.', 'Coverage\nEfficiency', 'Max\nPayout', 'Saffir-Simpson\nRange', 'Loss\nRatio'
    ]
    
    # 創建表格
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # 設置表格樣式
    # 標題行
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 數據行交替顏色
    for i in range(1, len(table_data) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)
    
    # 突出顯示最優值
    # 最佳RMSE (第1行)
    for j in range(len(columns)):
        table[(1, j)].set_facecolor('#ffeb3b')
        table[(1, j)].set_text_props(weight='bold')
    
    plt.title('Top 10 Parametric Insurance Products (Ranked by RMSE Basis Risk)\n' +
              'Steinmann et al. (2023) Methodology Analysis', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('steinmann_top_products_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_dashboard(df):
    """創建綜合儀表板"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 創建網格布局
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. 主要散點圖 (佔據2x2空間)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    scatter = ax1.scatter(df['rmse']/1e9, df['mae']/1e9, 
                         c=df['trigger_frequency']*100, 
                         cmap='viridis', alpha=0.7, s=60)
    ax1.set_xlabel('RMSE Basis Risk (Billion $)', fontsize=12)
    ax1.set_ylabel('MAE Basis Risk (Billion $)', fontsize=12)
    ax1.set_title('RMSE vs MAE Basis Risk Analysis', fontsize=14, fontweight='bold')
    
    # 標註最優產品
    best_rmse_idx = df['rmse'].idxmin()
    best_mae_idx = df['mae'].idxmin()
    ax1.scatter(df.loc[best_rmse_idx, 'rmse']/1e9, df.loc[best_rmse_idx, 'mae']/1e9, 
               color='red', s=150, marker='*', label='Best RMSE', edgecolor='black')
    ax1.scatter(df.loc[best_mae_idx, 'rmse']/1e9, df.loc[best_mae_idx, 'mae']/1e9, 
               color='orange', s=150, marker='*', label='Best MAE', edgecolor='black')
    ax1.legend()
    
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label('Trigger Frequency (%)', fontsize=10)
    
    # 2. 產品類別統計 (右上)
    ax2 = fig.add_subplot(gs[0, 2:])
    category_counts = df['category'].value_counts()
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    ax2.pie(category_counts.values, labels=[cat.replace('_', '\n') for cat in category_counts.index], 
           autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Product Category Distribution', fontsize=12, fontweight='bold')
    
    # 3. 風速閾值分析 (右中)
    ax3 = fig.add_subplot(gs[1, 2:])
    min_thresholds = []
    for _, row in df.iterrows():
        thresholds = eval(row['wind_thresholds']) if isinstance(row['wind_thresholds'], str) else row['wind_thresholds']
        min_thresholds.append(min(thresholds))
    
    ax3.hist(min_thresholds, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Minimum Wind Threshold (m/s)', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('Distribution of Minimum Wind Thresholds', fontsize=12, fontweight='bold')
    
    # 4. 觸發頻率 vs 覆蓋效率 (左下)
    ax4 = fig.add_subplot(gs[2, 0:2])
    scatter2 = ax4.scatter(df['trigger_frequency']*100, df['coverage_efficiency']*100,
                          c=df['rmse']/1e9, cmap='Reds', alpha=0.7, s=50)
    ax4.set_xlabel('Trigger Frequency (%)', fontsize=12)
    ax4.set_ylabel('Coverage Efficiency (%)', fontsize=12)
    ax4.set_title('Trigger Frequency vs Coverage Efficiency', fontsize=12, fontweight='bold')
    
    cbar2 = plt.colorbar(scatter2, ax=ax4, shrink=0.8)
    cbar2.set_label('RMSE (Billion $)', fontsize=10)
    
    # 5. 主要颶風覆蓋統計 (右下)
    ax5 = fig.add_subplot(gs[2, 2:])
    major_coverage = [
        sum(df['covers_major_hurricane'] == True),
        sum(df['covers_major_hurricane'] == False)
    ]
    labels = ['Covers Major\nHurricanes (H3+)', 'No Major\nHurricane Coverage']
    colors = ['#FF6B6B', '#4ECDC4']
    
    ax5.pie(major_coverage, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax5.set_title('Major Hurricane Coverage', fontsize=12, fontweight='bold')
    
    # 6. 關鍵統計摘要 (底部)
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # 計算關鍵統計
    best_rmse = df['rmse'].min()
    best_mae = df['mae'].min()
    avg_trigger = df['trigger_frequency'].mean()
    max_coverage = df['coverage_efficiency'].max()
    avg_premium = df['commercial_premium'].mean()
    
    stats_text = f"""
    Key Statistics Summary (Steinmann et al. 2023 Methodology):
    
    • Total Products Analyzed: {len(df):,}
    • Best RMSE Basis Risk: ${best_rmse/1e9:.3f}B
    • Best MAE Basis Risk: ${best_mae/1e9:.3f}B
    • Average Trigger Frequency: {avg_trigger*100:.1f}%
    • Maximum Coverage Efficiency: {max_coverage*100:.1f}%
    • Average Commercial Premium: ${avg_premium/1e6:.1f}M
    • Products Covering Major Hurricanes: {sum(df['covers_major_hurricane'])}/{len(df)}
    """
    
    ax6.text(0.05, 0.5, stats_text, fontsize=14, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            verticalalignment='center')
    
    plt.suptitle('Steinmann Parametric Insurance Analysis - Comprehensive Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('steinmann_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主執行函數"""
    
    print("🎨 創建Steinmann分析可視化圖表")
    print("=" * 50)
    
    # 載入結果
    df = load_results()
    if df is None:
        return
    
    print(f"載入 {len(df)} 個產品的分析結果")
    
    # 創建各種分析圖表
    print("\n1. 創建RMSE vs MAE基差風險分析圖...")
    create_rmse_mae_analysis(df)
    
    print("\n2. 創建財務特徵分析圖...")
    create_financial_analysis(df)
    
    print("\n3. 創建Saffir-Simpson等級分析圖...")
    create_saffir_simpson_analysis(df)
    
    print("\n4. 創建最優產品摘要表...")
    create_top_products_summary(df)
    
    print("\n5. 創建綜合儀表板...")
    create_comprehensive_dashboard(df)
    
    print("\n✅ 所有可視化圖表創建完成！")
    print("\n📁 生成的圖表文件:")
    print("  • steinmann_rmse_mae_analysis.png")
    print("  • steinmann_financial_analysis.png") 
    print("  • steinmann_saffir_simpson_analysis.png")
    print("  • steinmann_top_products_summary.png")
    print("  • steinmann_comprehensive_dashboard.png")

if __name__ == "__main__":
    main()