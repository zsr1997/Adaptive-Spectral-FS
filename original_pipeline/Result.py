import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# 1. 配置文件与解析函数
# ==============================================================
INPUT_EXCEL = "grapevine_sensitivity_results.xlsx"
OUTPUT_FIG = "Sensitivity_Analysis_Trends_2Rows.pdf"

def parse_mean_std(val_str):
    """将 '0.8437 ± 0.0215' 解析为 float 的 mean 和 std"""
    if pd.isna(val_str):
        return np.nan, np.nan
    val_str = str(val_str).strip()
    if '±' in val_str:
        parts = val_str.split('±')
        return float(parts[0].strip()), float(parts[1].strip())
    else:
        return float(val_str), 0.0

# 5 个我们要画图的超参数
params = ['N', 'T', 'CR', 'MR', 'Temp']
param_labels = ['Population Size (N)', 'Generations (G)', 
                'Crossover Rate (CR)', 'Mutation Rate (MR)', 'Temperature (Temp)']

# ==============================================================
# 2. 绘图设置 (GridSpec 高级布局：上3下2居中)
# ==============================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(18, 10)) # 调整为更适合两排的画布尺寸 (宽18，高10)

# 把画布分成 2 行 6 列的网格
gs = fig.add_gridspec(2, 6)

# 第一排：3 张图，分别占据 (0~2), (2~4), (4~6) 列
ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2:4])
ax3 = fig.add_subplot(gs[0, 4:6])

# 第二排：2 张图居中，分别占据 (1~3), (3~5) 列
ax4 = fig.add_subplot(gs[1, 1:3])
ax5 = fig.add_subplot(gs[1, 3:5])

axes = [ax1, ax2, ax3, ax4, ax5]

fig.suptitle('Sensitivity Analysis of Hyperparameters (Mean $\\pm$ Std over 30 Runs)', 
             fontsize=20, fontweight='bold')

colors = {'Random_Split': '#d62728', 'Temporal_Transfer': '#1f77b4'}
labels = {'Random_Split': 'Random Split (70/30)', 'Temporal_Transfer': 'Temporal Transfer (LOYO)'}

# ==============================================================
# 3. 循环读取数据并画图
# ==============================================================
for i, param in enumerate(params):
    ax = axes[i]
    sheet_name = f"Trend_{param}"
    
    try:
        # 读取多级表头
        df = pd.read_excel(INPUT_EXCEL, sheet_name=sheet_name, header=[0, 1], index_col=0)
    except Exception as e:
        print(f"Warning: 无法读取 {sheet_name}，跳过... ({e})")
        continue

    # 提取 Accuracy 列
    if 'Accuracy' not in df.columns.levels[0]:
        continue
    df_acc = df['Accuracy']
    
    # 提取 X 轴刻度
    x_values = [str(x) for x in df_acc.index]
    x_pos = np.arange(len(x_values))
    
    # 画两条线：Random Split 和 Temporal Transfer
    for exp_mode in ['Random_Split', 'Temporal_Transfer']:
        if exp_mode not in df_acc.columns: continue
            
        means, stds = [], []
        for val in df_acc[exp_mode]:
            m, s = parse_mean_std(val)
            means.append(m)
            stds.append(s)
            
        means = np.array(means)
        stds = np.array(stds)
        
        # 绘制主均值折线
        ax.plot(x_pos, means, marker='o', markersize=7, linewidth=2.5, 
                color=colors[exp_mode], label=labels[exp_mode])
        # 绘制误差阴影
        ax.fill_between(x_pos, means - stds, means + stds, 
                        color=colors[exp_mode], alpha=0.15)
        
    ax.set_title(f'Impact of {param_labels[i]}', fontsize=15, pad=10)
    ax.set_xlabel(param_labels[i], fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_values, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # 只给每行的第一张图 (索引 0 和 3) 添加 Y 轴标签
    if i in [0, 3]:
        ax.set_ylabel('Classification Accuracy', fontsize=14)
    
    # 只给第一张图添加图例
    if i == 0:
        ax.legend(fontsize=12, loc='lower right')

# 调整子图间距并保存 (rect 留出顶部空间给主标题)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_FIG, bbox_inches='tight', dpi=300)
print(f"\n[+] 敏感度分析图 (2排居中布局) 已成功生成：{OUTPUT_FIG}")
plt.show()