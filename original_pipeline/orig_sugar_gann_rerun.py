import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root (core/, utils/)
# orig_sugar_gann_rerun.py
# Final GA-NN Sugar cell of GA-NN_Github_Sugar.ipynb (the run that produced Final_Sugar_*_Ours.csv),
# re-run unchanged except for the classification head of the dual proxy, which is now a
# three-class cross-entropy head (patched core/networks.py). Same seeds (42 + run_idx),
# same splits and evaluation as the original MOEA and baseline runs.
# Outputs: Final_Sugar_*_Ours_CEfix.csv and Sugar_Fig*_CEfix.pdf (originals are not overwritten).
import matplotlib
matplotlib.use('Agg')   # no interactive windows when run in the background
import inspect as _inspect
from core import networks as _networks
if "CrossEntropyLoss()(c_pred" not in _inspect.getsource(_networks.train_softmask_dnn_dual_proxy):
    raise SystemExit("core/networks.py still uses the single-output BCE head: copy the patched networks.py into core/ first.")
# ==============================================================
# 专属验证实验：Sugar Dataset (Dual Task: Classification & Regression)
# 包含：GA-NN (Adaptive) 30次运行 + 自动绘制收敛曲线与权重分布箱线图
# 运行环境：Jupyter Notebook 兼容版
# ==============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
import time
import torch
import matplotlib.pyplot as plt

# --- Sklearn ---
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- Ours ---
# Ensure these imports match your project structure
from utils.data_loader import load_sugar_data
from core.ga_utils import jfs, keep_elite
from core.fitness_fast import Obj_Max_MI_Fast, Obj2_Min_Redundancy_Safe, Obj_Min_Regression_Safe
from core.networks import train_softmask_dnn_dual_proxy

# ==============================================================
# 0. Configuration & Utilities
# ==============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def calc_mean_std(x): 
    return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}" if len(x) > 1 else f"{x.mean():.4f} ± 0.0000"

def get_top_k(scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return idx if len(idx) > 0 else np.array([0])

# ==============================================================
# 1. Core Evaluation Routine (OURS ONLY)
# ==============================================================
def run_ours_only_sugar(feat, y_class, y_reg, run_idx, args):
    current_seed = 42 + run_idx
    set_seed(current_seed)
    
    # --- 数据划分 ---
    xt_tr, xt_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        feat, y_class, y_reg, test_size=0.3, stratify=y_class, random_state=current_seed
    )
    
    # 内部划分为 GA Obj3 (Regression Proxy) 提供 Train/Val 验证集
    xt_in, xv_in, yc_in, yc_val, yr_in, yr_val = train_test_split(
        xt_tr, yc_tr, yr_tr, test_size=0.25, stratify=yc_tr, random_state=current_seed
    )
    
    print(f"\n  [Run {run_idx}/{args.n_run}] Pre-computing Mutual Information...")
    mi_class = mutual_info_classif(xt_in, yc_in, random_state=current_seed)
    
    opts = {
        'xt_in': xt_in, 
        'yt_in': yr_in, 
        'xv_in': xv_in, 
        'yv_in': yr_val, 
        'mi_class': mi_class, 
        'N': 40, 
        'T': 40, 
        'CR': 0.8, 
        'MR': 0.01
    }
    
    results = []
    clf_lda = LinearDiscriminantAnalysis()
    reg_rf = RandomForestRegressor(n_estimators=50, random_state=42)
    
    print(f"    -> Optimizing with GA-NN (Adaptive)...")
    
    start_time = time.time()
    
    # 1. 独立单目标 GA 搜索 (记录收敛曲线)
    res1 = jfs(opts, lambda x, o: Obj_Max_MI_Fast(x, o, 'mi_class'))
    res2 = jfs(opts, Obj2_Min_Redundancy_Safe)
    res3 = jfs(opts, Obj_Min_Regression_Safe)
    
    curve1 = res1.get('curve', [])
    curve2 = res2.get('curve', [])
    curve3 = res3.get('curve', [])
    
    # 2. 提取精英解
    pop1 = keep_elite(**res1)
    pop2 = keep_elite(**res2)
    pop3 = keep_elite(**res3)
    
    # 3. 双任务深度特征融合 (记录最终的动态权重 lam, w_hist 和 Loss 历史 l_hist)
    sc_ad, lam, w_hist, l_hist = train_softmask_dnn_dual_proxy(
        pop1, pop2, pop3, xt_tr, yc_tr, yr_tr, xt_tr.shape[1], True, temp_scale=10.0
    )
    
    elapsed_time = time.time() - start_time
    
    # 验证分类精度与回归 R2 (K=1..15)
    for k in range(1, 16):
        idx = get_top_k(sc_ad, k)
        
        # Classification (LDA)
        clf_lda.fit(xt_tr[:, idx], yc_tr)
        acc = clf_lda.score(xt_te[:, idx], yc_te)
        results.append({
            "Run": run_idx, "Method": "GA-NN (Adaptive)", "K": k, 
            "Task": "Classification", "Model": "LDA", "Score": acc,
            "Time_Seconds": elapsed_time
        })
        
        # Regression (RF)
        reg_rf.fit(xt_tr[:, idx], yr_tr)
        r2 = r2_score(yr_te, reg_rf.predict(xt_te[:, idx]))
        results.append({
            "Run": run_idx, "Method": "GA-NN (Adaptive)", "K": k, 
            "Task": "Regression", "Model": "RF", "Score": r2,
            "Time_Seconds": elapsed_time
        })
            
    # 返回评估结果与绘图所需的历史记录
    return results, (curve1, curve2, curve3, l_hist, lam, w_hist)

# ==============================================================
# 2. Orchestrator & Execution
# ==============================================================
def run_benchmark(features, class_labels, regression_target, args):
    y_class = np.asarray(class_labels)
    y_reg = np.asarray(regression_target)
    
    feat = normalize(np.asarray(features), norm='l1')

    all_results = []
    
    # 绘图数据存储器
    all_c1, all_c2, all_c3 = [], [], []
    all_loss, all_weights, all_w_hist = [], [], []

    print("="*80)
    print(f"Starting EXCLUSIVE BENCHMARK FOR OUR METHOD (Sugar): {args.n_run} Runs")
    print("="*80)
    
    global_start_time = time.time()
    
    for run_idx in range(1, args.n_run + 1):
        res, histories = run_ours_only_sugar(feat, y_class, y_reg, run_idx, args)
        all_results.extend(res)
        
        # 收集绘图数据
        c1, c2, c3, l_hist, lam, w_hist = histories
        all_c1.append(c1)
        all_c2.append(c2)
        all_c3.append(c3)
        all_loss.append(l_hist)
        all_weights.append(lam)
        all_w_hist.append(w_hist)

    # ==============================================================
    # 3. Output Generation
    # ==============================================================
    df = pd.DataFrame(all_results)

    print("\n\n" + "="*80)
    print("             FINAL OURS-ONLY OUTPUT TABLES (Mean ± Std)")
    print("="*80)
    
    # 1. Classification (LDA) Table
    df_lda = df[(df["Task"] == "Classification") & (df["Model"] == "LDA")]
    if not df_lda.empty:
        tb_lda = df_lda.pivot_table(index="K", columns="Method", values="Score", aggfunc=calc_mean_std)
        print("\n[TABLE 1] Classification Accuracy (LDA)")
        print(tb_lda)
        tb_lda.to_csv("Final_Sugar_Class_LDA_Ours_CEfix.csv")
    
    # 2. Regression (RF) Table
    df_rf = df[(df["Task"] == "Regression") & (df["Model"] == "RF")]
    if not df_rf.empty:
        tb_rf = df_rf.pivot_table(index="K", columns="Method", values="Score", aggfunc=calc_mean_std)
        print("\n[TABLE 2] Regression R2 Score (RF)")
        print(tb_rf)
        tb_rf.to_csv("Final_Sugar_Reg_RF_Ours_CEfix.csv")

    # 3. Execution Time Table
    tb_time = df.pivot_table(index="Method", values="Time_Seconds", aggfunc=calc_mean_std)
    print("\n[TABLE 3] Execution Time (Seconds)")
    print(tb_time)
    tb_time.to_csv("Final_Sugar_Time_Ours_CEfix.csv")

    total_minutes = (time.time() - global_start_time) / 60
    print(f"\n-> GA-NN (Adaptive) 30 次独立运行测试完毕！共耗时 {total_minutes:.2f} 分钟。开始绘制审稿人要求的图表...\n")

    # ==============================================================
    # 4. Plotting for Reviewers (收敛曲线、最终权重与权重变化轨迹)
    # ==============================================================
    try:
        plot_reviewer_figures(all_c1, all_c2, all_c3, all_loss, all_weights, all_w_hist)
    except Exception as _e:
        print(f'[WARN] figures not drawn ({_e}); the CSV results are already saved.')


def plot_reviewer_figures(c1_list, c2_list, c3_list, loss_list, weights_list, w_hist_list):
    """专门为审稿人生成论文级别的图表 (Sugar Dataset 专属)"""
    
    c1_arr = np.array(c1_list)
    c2_arr = np.array(c2_list)
    c3_arr = np.array(c3_list)
    loss_arr = np.array(loss_list)
    weights_arr = np.array(weights_list)         # shape: (30, 3)
    w_hist_arr = np.array(w_hist_list)           # shape: (30, epochs, 3)

    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ---------------------------------------------------------
    # 图 1: 收敛曲线 (GA 3个目标 + DNN Loss)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle('Convergence Curves during Training (Mean $\\pm$ Std over 30 Runs) - Sugar Dataset', fontsize=16, fontweight='bold', y=1.05)

    titles = ['Obj1: Class MI (Negative)', 'Obj2: Redundancy', 'Obj3: Regression Error', 'DNN Dual-Proxy Training Loss']
    data_arrays = [c1_arr, c2_arr, c3_arr, loss_arr]
    x_labels = ['Generations'] * 3 + ['Epochs']

    for i in range(4):
        ax = axes[i]
        mean_val = np.mean(data_arrays[i], axis=0)
        std_val = np.std(data_arrays[i], axis=0)
        x_axis = np.arange(1, len(mean_val) + 1)
        
        ax.plot(x_axis, mean_val, color='b', linewidth=2, label='Mean')
        ax.fill_between(x_axis, mean_val - std_val, mean_val + std_val, color='b', alpha=0.2, label='$\\pm$ Std')
        
        ax.set_title(titles[i], fontsize=14)
        ax.set_xlabel(x_labels[i], fontsize=12)
        if i == 0: ax.set_ylabel('Fitness / Loss Value', fontsize=12)
        ax.legend()

    plt.tight_layout()
    plt.savefig("Sugar_Fig1_Convergence_CEfix.pdf", bbox_inches='tight', dpi=300)
    plt.show()

    # ---------------------------------------------------------
    # 图 2: 动态权重变化轨迹 (3个子图，展示 w1, w2, w3 的 Epoch 变化)
    # ---------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Dynamic Weight Trajectories during DNN Training (Mean $\\pm$ Std)', fontsize=16, fontweight='bold', y=1.05)

    weight_names = ['W_Classification', 'W_Redundancy', 'W_Regression']
    line_colors = ['#d62728', '#1f77b4', '#2ca02c'] # 红, 蓝, 绿

    for i in range(3):
        ax = axes2[i]
        mean_w = np.mean(w_hist_arr[:, :, i], axis=0)
        std_w = np.std(w_hist_arr[:, :, i], axis=0)
        epochs_x = np.arange(1, len(mean_w) + 1)

        ax.plot(epochs_x, mean_w, color=line_colors[i], linewidth=2.5, label=f'Mean {weight_names[i]}')
        ax.fill_between(epochs_x, mean_w - std_w, mean_w + std_w, color=line_colors[i], alpha=0.2)

        ax.set_title(f'Trajectory of {weight_names[i]}', fontsize=14)
        ax.set_xlabel('Epochs', fontsize=12)
        if i == 0: ax.set_ylabel('Weight Value (Softmax Output)', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.legend()

    plt.tight_layout()
    plt.savefig("Sugar_Fig2_Weight_Trajectories_CEfix.pdf", bbox_inches='tight', dpi=300)
    plt.show()

    # ---------------------------------------------------------
    # 图 3: 最终自适应权重分布 (Boxplot)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    # 使用箱线图展示 30 次运行的权重分布
    box = plt.boxplot([weights_arr[:, 0], weights_arr[:, 1], weights_arr[:, 2]], 
                      labels=['W_Classification', 'W_Redundancy', 'W_Regression'],
                      patch_artist=True)
    
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    plt.title('Final Adaptive Weight Distributions across 30 Independent Runs', fontsize=15, fontweight='bold')
    plt.ylabel('Final Weight Value', fontsize=13)
    plt.ylim(0, 1.0)
    
    # 标注出均值
    means = np.mean(weights_arr, axis=0)
    for i, mean_val in enumerate(means):
        plt.text(i+1, mean_val + 0.05, f'Mean: {mean_val:.3f}', horizontalalignment='center', fontsize=11, color='darkred')

    plt.tight_layout()
    plt.savefig("Sugar_Fig3_Weight_Boxplot_CEfix.pdf", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    class Args:
        n_run = int(__import__('os').environ.get('N_RUN', 30))  # 30 independent runs
        
    args = Args()

    try:
        print("Loading Data...")
        X, yc, yr = load_sugar_data()
        run_benchmark(X, yc, yr, args)
        
    except Exception as e:
        raise SystemExit(f"Data loading or run failed: {e}")
