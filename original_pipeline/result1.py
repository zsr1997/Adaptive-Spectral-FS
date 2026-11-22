import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# 1. 配置文件与解析函数
# ==============================================================
INPUT_EXCEL = "sugar_sensitivity_results.xlsx"
OUTPUT_FIG = "Sugar_Sensitivity_Analysis_Trends.pdf"

def parse_mean_std(val_str):
    """解析 '0.9863 ± 0.0170' 格式为 float 的 mean 和 std"""
    if pd.isna(val_str):
        return np.nan, np.nan
    val_str = str(val_str).strip()
    if '±' in val_str:
        parts = val_str.split('±')
        return float(parts[0].strip()), float(parts[1].strip())
    else:
        return float(val_str), 0.0

params = ['N', 'T', 'CR', 'MR', 'Temp']
param_labels = ['Population Size (N)', 'Generations (G)', 
                'Crossover Rate (CR)', 'Mutation Rate (MR)', 'Temperature (Temp)']

# ==============================================================
# 2. 绘图设置 (GridSpec 高级布局：上3下2居中)
# ==============================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(19, 10))

gs = fig.add_gridspec(2, 6)
ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2:4])
ax3 = fig.add_subplot(gs[0, 4:6])
ax4 = fig.add_subplot(gs[1, 1:3])
ax5 = fig.add_subplot(gs[1, 3:5])

axes = [ax1, ax2, ax3, ax4, ax5]

fig.suptitle('Sensitivity Analysis of Hyperparameters on the Dual-Task Sugar Dataset', 
             fontsize=20, fontweight='bold')

# 双任务专属颜色：红色代表分类，蓝色代表回归
color_acc = '#d62728' 
color_r2 = '#1f77b4'  

# ==============================================================
# 3. 循环读取数据并画双 Y 轴图
# ==============================================================
for i, param in enumerate(params):
    ax = axes[i]
    sheet_name = f"Trend_{param}"
    
    try:
        df = pd.read_excel(INPUT_EXCEL, sheet_name=sheet_name, header=[0, 1], index_col=0)
    except Exception as e:
        print(f"Warning: 无法读取 {sheet_name}，跳过... ({e})")
        continue

    # 提取分类和回归列
    df_acc = df[('Class_Accuracy', 'GA+DNN (Adaptive)')]
    df_r2 = df[('Reg_R2', 'GA+DNN (Adaptive)')]
    
    x_values = [str(x) for x in df_acc.index]
    x_pos = np.arange(len(x_values))
    
    ax_r2 = ax.twinx() # 创建共享 X 轴的第二个 Y 轴
    ax_r2.grid(False)  # 关闭副轴网格，防止重叠变乱
    
    # 提取均值和方差
    means_acc, stds_acc = zip(*[parse_mean_std(val) for val in df_acc])
    means_r2, stds_r2 = zip(*[parse_mean_std(val) for val in df_r2])
    
    means_acc, stds_acc = np.array(means_acc), np.array(stds_acc)
    means_r2, stds_r2 = np.array(means_r2), np.array(stds_r2)
    
    # 画分类 (左 Y 轴)
    l1 = ax.plot(x_pos, means_acc, marker='o', markersize=7, linewidth=2.5, 
                 color=color_acc, label='Classification Accuracy')
    ax.fill_between(x_pos, means_acc - stds_acc, means_acc + stds_acc, color=color_acc, alpha=0.15)
                    
    # 画回归 (右 Y 轴)
    l2 = ax_r2.plot(x_pos, means_r2, marker='s', markersize=7, linewidth=2.5, 
                    color=color_r2, label='Regression R2 Score')
    ax_r2.fill_between(x_pos, means_r2 - stds_r2, means_r2 + stds_r2, color=color_r2, alpha=0.15)
        
    # 标签与美化
    ax.set_title(f'Impact of {param_labels[i]}', fontsize=15, pad=10)
    ax.set_xlabel(param_labels[i], fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_values, fontsize=12)
    
    ax.tick_params(axis='y', labelcolor=color_acc, labelsize=12)
    ax_r2.tick_params(axis='y', labelcolor=color_r2, labelsize=12)
    
    if i in [0, 3]:
        ax.set_ylabel('Classification Accuracy', fontsize=14, color=color_acc)
    if i in [2, 4]:
        ax_r2.set_ylabel('Regression R2 Score', fontsize=14, color=color_r2)
    
    if i == 0:
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=11, loc='best')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_FIG, bbox_inches='tight', dpi=300)
print(f"\n[+] 双任务敏感度分析图已成功生成：{OUTPUT_FIG}")
plt.show()