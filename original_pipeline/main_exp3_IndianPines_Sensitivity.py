import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root (core/, utils/)
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import argparse
import time 
import json 

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils.data_loader import load_indian_pines
from core.ga_utils import jfs, keep_elite
from core.fitness_fast import Obj_Max_MI_Fast, Obj2_Min_Redundancy_Safe
from core.networks import train_softmask_dnn_proxy_multi
# --- revision guard: the GA must use the CR/MR passed in opts (patched core/ga_utils.py) ---
import inspect as _inspect
from core import ga_utils as _ga_utils
if "opts.get('CR'" not in _inspect.getsource(_ga_utils.jfs):
    raise SystemExit("core/ga_utils.py still ignores CR/MR: copy the patched ga_utils.py into core/ first.")


def run_sensitivity_fs(xt_tr, xt_te, yt1_tr, yt1_te, yt2_tr, yt2_te, num_classes_t1, run_idx, param_name, param_val, current_params, args):
    """
    Streamlined feature selection runner specifically for sensitivity analysis on the Indian Pines Dataset.
    Only evaluates the proposed GA+DNN (Adaptive - Ours) method to save computation time.
    """
    opts = {
        'xt_in': xt_tr, 
        'mi_task1': mutual_info_classif(xt_tr, yt1_tr), 
        'mi_task2': mutual_info_classif(xt_tr, yt2_tr), 
        'N': current_params['N'], 
        'T': current_params['T'],
        'CR': current_params['CR'],
        'MR': current_params['MR']
    }

    start_time = time.time()
    
    # Run Multi-Objective GA
    res1 = jfs(opts, lambda x, o: Obj_Max_MI_Fast(x, o, 'mi_task1'))
    res2 = jfs(opts, Obj2_Min_Redundancy_Safe)
    res3 = jfs(opts, lambda x, o: Obj_Max_MI_Fast(x, o, 'mi_task2'))

    # 提取精英文档与收敛曲线 (响应审稿人：the convergence curves during training)
    pop1, curve1 = keep_elite(**res1), res1.get('curve', [])
    pop2, curve2 = keep_elite(**res2), res2.get('curve', [])
    pop3, curve3 = keep_elite(**res3), res3.get('curve', [])

    # Train only the Adaptive DNN Proxy for the proposed method
    # 提取权重历史与 Loss 历史 (响应审稿人：showing changes in weights)
    sc_ad, lam, w_hist, l_hist = train_softmask_dnn_proxy_multi(
        pop1, pop2, pop3, xt_tr, yt1_tr, xt_tr.shape[1], num_classes_t1, True, temp_scale=current_params['Temp']
    )
    
    total_fs_time = time.time() - start_time 

    # Evaluation using Linear Discriminant Analysis for both tasks
    clf = LinearDiscriminantAnalysis()
    
    def get_top_k(scores, k): 
        idx = np.argsort(scores)[::-1][:k]
        return idx if len(idx) > 0 else [0]

    # Evaluate at a fixed K to observe the trend clearly
    idx = get_top_k(sc_ad, args.K)
    
    acc_fine = clf.fit(xt_tr[:, idx], yt1_tr).score(xt_te[:, idx], yt1_te)
    acc_coarse = clf.fit(xt_tr[:, idx], yt2_tr).score(xt_te[:, idx], yt2_te)

    result = {
        "Parameter_Tested": param_name,
        "Parameter_Value": param_val,
        "Run": run_idx, 
        "Method": "GA+DNN (Adaptive - Ours)", 
        "K": args.K, 
        "Acc_Task1_Fine": acc_fine,
        "Acc_Task2_Coarse": acc_coarse,
        "Time_Seconds": total_fs_time, 
        "W_Task1_Fine": lam[0] if lam is not None else None,
        "W_Redundancy": lam[1] if lam is not None else None,
        "W_Task2_Crop": lam[2] if lam is not None else None
    }
    
    history = {
        "Experiment": "Indian_Pines_Hierarchical",
        "Parameter_Tested": param_name,
        "Parameter_Value": param_val,
        "Run": run_idx,
        "GA_Curve_Obj1": curve1,
        "GA_Curve_Obj2": curve2,
        "GA_Curve_Obj3": curve3,
        "DNN_Weight_History": w_hist,
        "DNN_Loss_History": l_hist
    }
    
    return result, history

def run_sensitivity_experiments(features, task1_labels, task2_labels, args):
    le_t1 = LabelEncoder()
    t1_enc = le_t1.fit_transform(task1_labels)
    num_classes_t1 = len(le_t1.classes_)

    # Baseline Parameters derived from Table 8 
    baseline_params = {
        'N': 40,      # Population Size
        'T': 40,      # Generations (Max Iterations)
        'CR': 0.8,    # Crossover Rate
        'MR': 0.01,   # Mutation Rate 
        'Temp': 5.0   # Softmax 的基准温度系数
    }
    
    # Grids for One-at-a-Time (OAT) Sensitivity Analysis
    sensitivity_grids = {
        'N': [10, 20, 40, 60, 80],
        'T': [10, 20, 40, 60, 80], 
        'CR': [0.4, 0.6, 0.8, 0.9, 1.0],
        'MR': [0.001, 0.01, 0.05, 0.1, 0.2],
        'Temp': [1.0, 2.0, 5.0, 10.0, 20.0] 
    }
    sensitivity_grids = {k: v for k, v in sensitivity_grids.items() if k in args.params}
    
    all_results = []
    all_histories = []
    
    print("\n" + "="*50)
    print("Starting Indian Pines Parameter Sensitivity Analysis (OAT)")
    print("Baseline Parameters:", baseline_params)
    print("="*50)

    for param_name, param_values in sensitivity_grids.items():
        print(f"\n>>> Testing sensitivity for Parameter: {param_name}")
        
        for val in param_values:
            current_params = baseline_params.copy()
            current_params[param_name] = val
            
            print(f"  -> {param_name} = {val} | Running {args.n_run} iterations...")
            
            for run_idx in range(1, args.n_run + 1):
                # Split data per run to ensure statistical robustness (using stratify on the fine-grained labels)
                xt_tr, xt_te, yt1_tr, yt1_te, yt2_tr, yt2_te = train_test_split(
                    features, t1_enc, task2_labels, test_size=0.3, stratify=t1_enc, random_state=run_idx
                )
                
                res, hist = run_sensitivity_fs(
                    xt_tr, xt_te, yt1_tr, yt1_te, yt2_tr, yt2_te, num_classes_t1,
                    run_idx, param_name, val, current_params, args
                )
                all_results.append(res)
                all_histories.append(hist)

    df = pd.DataFrame(all_results)
    
    # Summarize Results: Calculate Mean ± Std directly (响应审稿人：means and standard deviations after 30 independent runs)
    def ms(x): return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}" if len(x)>1 else f"{x.mean():.4f}"
    
    print("\n" + "="*50)
    print(f"SENSITIVITY ANALYSIS RESULTS (Top K={args.K})")
    print("="*50)
    
    try:
        with pd.ExcelWriter(args.output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="Raw_Sensitivity_Data", index=False)
            
            for param in sensitivity_grids.keys():
                df_param = df[df["Parameter_Tested"] == param]
                
                # Pivot table to show both Task Accuracies, Weights, and Time
                tb = df_param.pivot_table(
                    index="Parameter_Value", 
                    columns="Method", 
                    values=["Acc_Task1_Fine", "Acc_Task2_Coarse", "W_Task1_Fine", "W_Redundancy", "W_Task2_Crop", "Time_Seconds"], 
                    aggfunc=ms
                )
                
                print(f"\n[TREND] Impact of {param}:")
                print(tb)
                
                tb.to_excel(writer, sheet_name=f"Trend_{param}")
                
        print(f"\n[INFO] Sensitivity results successfully saved to Excel file: {args.output_file}")
        
        # 导出包含了所有动态变化数据的 JSON (响应审稿人：convergence curves and changes in weights)
        history_file = args.output_file.replace('.xlsx', '_history.json')
        with open(history_file, 'w') as f:
            json.dump(all_histories, f)
        print(f"[INFO] Training histories (curves, weights) saved to: {history_file}")
        
    except Exception as excel_err:
        print(f"\n[ERROR] Failed to save output files: {excel_err}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian Pines Parameter Sensitivity Analysis")
    
    # 强制修改：按审稿人要求默认设为 30 次独立运行
    parser.add_argument("--n_run", type=int, default=30, help="Number of runs per param value (default: 30 to satisfy reviewer)")
    parser.add_argument("--K", type=int, default=15, help="Selected features to evaluate (default: 15)")
    parser.add_argument("--output_file", type=str, default="ip_sensitivity_results.xlsx", help="Output Excel filename")
    parser.add_argument("--params", nargs="+", default=["N", "T", "CR", "MR", "Temp"], help="parameters to test (default: all)")
    
    args = parser.parse_args()

    try:
        X, y_fine, y_crop = load_indian_pines()
        run_sensitivity_experiments(X, y_fine, y_crop, args)
    except Exception as e: 
        print(f"Error: {e}")