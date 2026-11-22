import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# 1. 配置文件与解析函数
# ==============================================================
INPUT_EXCEL = "ip_sensitivity_results.xlsx"
OUTPUT_FIG = "IP_Sensitivity_Analysis_Trends.pdf"

def parse_mean_std(val_str):
    """解析 '0.5628 ± 0.0570' 格式为 float 的 mean 和 std"""
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

fig.suptitle('Sensitivity Analysis of Hyperparameters on the Hierarchical Indian Pines Dataset', 
             fontsize=20, fontweight='bold')

# 双任务专属颜色：红色代表细粒度分类(Task 1)，蓝色代表粗粒度分类(Task 2)
color_acc1 = '#d62728' 
color_acc2 = '#1f77b4'  

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

    # 提取细粒度与粗粒度列 (基于你真实存在的列名)
    df_task1 = df[('Acc_Task1_Fine', 'GA+DNN (Adaptive - Ours)')]
    df_task2 = df[('Acc_Task2_Coarse', 'GA+DNN (Adaptive - Ours)')]
    
    x_values = [str(x) for x in df_task1.index]
    x_pos = np.arange(len(x_values))
    
    ax_t2 = ax.twinx() # 创建共享 X 轴的第二个 Y 轴
    ax_t2.grid(False)  # 关闭副轴网格防止重叠
    
    # 提取均值和方差
    means_t1, stds_t1 = zip(*[parse_mean_std(val) for val in df_task1])
    means_t2, stds_t2 = zip(*[parse_mean_std(val) for val in df_task2])
    
    means_t1, stds_t1 = np.array(means_t1), np.array(stds_t1)
    means_t2, stds_t2 = np.array(means_t2), np.array(stds_t2)
    
    # 画细粒度任务 (左 Y 轴)
    l1 = ax.plot(x_pos, means_t1, marker='o', markersize=7, linewidth=2.5, 
                 color=color_acc1, label='Task 1 (Fine-grained 16-class)')
    ax.fill_between(x_pos, means_t1 - stds_t1, means_t1 + stds_t1, color=color_acc1, alpha=0.15)
                    
    # 画粗粒度任务 (右 Y 轴)
    l2 = ax_t2.plot(x_pos, means_t2, marker='s', markersize=7, linewidth=2.5, 
                    color=color_acc2, label='Task 2 (Coarse-grained)')
    ax_t2.fill_between(x_pos, means_t2 - stds_t2, means_t2 + stds_t2, color=color_acc2, alpha=0.15)
        
    # 标签与美化
    ax.set_title(f'Impact of {param_labels[i]}', fontsize=15, pad=10)
    ax.set_xlabel(param_labels[i], fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_values, fontsize=12)
    
    ax.tick_params(axis='y', labelcolor=color_acc1, labelsize=12)
    ax_t2.tick_params(axis='y', labelcolor=color_acc2, labelsize=12)
    
    if i in [0, 3]:
        ax.set_ylabel('Task 1 Accuracy', fontsize=14, color=color_acc1)
    if i in [2, 4]:
        ax_t2.set_ylabel('Task 2 Accuracy', fontsize=14, color=color_acc2)
    
    if i == 0:
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=11, loc='best')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_FIG, bbox_inches='tight', dpi=300)
print(f"\n[+] 印度松林敏感度分析图已成功生成：{OUTPUT_FIG}")
plt.show()