import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root (core/, utils/)
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import random
import torch
import argparse
import time 
import json 

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Ensure these imports match your project structure
from utils.data_loader import load_grapevine_data
from core.ga_utils import jfs, keep_elite
from core.fitness_fast import Obj_Max_MI_Fast, Obj2_Min_Redundancy_Safe, Obj3_Min_Year_MI_Fast
from core.networks import train_softmask_dnn_proxy
# --- revision guard: the GA must use the CR/MR passed in opts (patched core/ga_utils.py) ---
import inspect as _inspect
from core import ga_utils as _ga_utils
if "opts.get('CR'" not in _inspect.getsource(_ga_utils.jfs):
    raise SystemExit("core/ga_utils.py still ignores CR/MR: copy the patched ga_utils.py into core/ first.")


def run_sensitivity_fs(xtrain, xtest, ytrain, ytest, yyear_tr, run_idx, param_name, param_val, current_params, exp_name, args):
    """
    Streamlined feature selection runner specifically for sensitivity analysis.
    Only evaluates the proposed GA+DNN (Adaptive) method to save computation time.
    """
    opts = {
        'xt_in': xtrain, 
        'mi_class': mutual_info_classif(xtrain, ytrain), 
        'mi_year': mutual_info_classif(xtrain, yyear_tr), 
        'N': current_params['N'],
        'T': current_params['T'],
        'CR': current_params['CR'],
        'MR': current_params['MR']
    }
    
    start_time = time.time() 
    
    # Run Multi-Objective GA
    res1 = jfs(opts, lambda x, o: Obj_Max_MI_Fast(x, o, 'mi_class'))
    res2 = jfs(opts, Obj2_Min_Redundancy_Safe)
    res3 = jfs(opts, Obj3_Min_Year_MI_Fast)
    
    # 提取精英文档与收敛曲线 (响应审稿人：the convergence curves during training)
    pop1, curve1 = keep_elite(**res1), res1.get('curve', [])
    pop2, curve2 = keep_elite(**res2), res2.get('curve', [])
    pop3, curve3 = keep_elite(**res3), res3.get('curve', [])
    
    # Train only the Adaptive DNN Proxy for the proposed method
    # 提取权重历史与 Loss 历史 (响应审稿人：showing changes in weights)
    sc_ad, lam, w_hist, l_hist = train_softmask_dnn_proxy(
        pop1, pop2, pop3, xtrain, ytrain, xtrain.shape[1], True, temp_scale=current_params['Temp']
    )
    
    total_fs_time = time.time() - start_time
    
    clf = LinearDiscriminantAnalysis()
    
    def get_top_k(scores, k): 
        idx = np.argsort(scores)[::-1][:k]
        return idx if len(idx)>0 else [0]
    
    # We evaluate at a fixed K (e.g., K=15) to observe the trend clearly
    idx = get_top_k(sc_ad, args.K)
    accuracy = clf.fit(xtrain[:, idx], ytrain).score(xtest[:, idx], ytest)

    result = {
        "Experiment": exp_name,
        "Parameter_Tested": param_name,
        "Parameter_Value": param_val,
        "Run": run_idx, 
        "Method": "GA+DNN (Adaptive)", 
        "K": args.K, 
        "Accuracy": accuracy,
        "Time_Seconds": total_fs_time, 
        "W_Class": lam[0] if lam is not None else None,
        "W_Redundancy": lam[1] if lam is not None else None,
        "W_Year": lam[2] if lam is not None else None
    }
    
    history = {
        "Experiment": exp_name,
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

def run_sensitivity_experiments(features, class_labels, year_raw, args):
    le_year = LabelEncoder()
    y_year_encoded = le_year.fit_transform(year_raw)
    
    target_year_val = args.target_year
    target_year_idx = le_year.transform([target_year_val])[0] if target_year_val in year_raw else np.max(y_year_encoded)
    
    # Baseline Parameters
    baseline_params = {
        'N': 40,      
        'T': 40,      
        'CR': 0.8,    
        'MR': 0.01,   
        'Temp': 5.0   
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
    print("Starting Parameter Sensitivity Analysis (OAT) - Dual Split Mode")
    print("Baseline:", baseline_params)
    print("Target Year for LOYO:", target_year_val)
    print("="*50)

    for param_name, param_values in sensitivity_grids.items():
        print(f"\n>>> Testing sensitivity for Parameter: {param_name}")
        
        for val in param_values:
            current_params = baseline_params.copy()
            current_params[param_name] = val
            
            print(f"  -> {param_name} = {val} | Running {args.n_run} iterations...")
            
            for run_idx in range(1, args.n_run + 1):
                # ----------------------------------------------------
                # Data Split 1: Random Split 
                # ----------------------------------------------------
                xt_tr1, xt_te1, yt_tr1, yt_te1, yr_tr1, _ = train_test_split(
                    features, class_labels, y_year_encoded, 
                    test_size=0.3, stratify=class_labels, random_state=run_idx
                )
                res1, hist1 = run_sensitivity_fs(
                    xt_tr1, xt_te1, yt_tr1, yt_te1, yr_tr1, 
                    run_idx, param_name, val, current_params, "Random_Split", args
                )
                all_results.append(res1)
                all_histories.append(hist1)
                
                # ----------------------------------------------------
                # Data Split 2: Temporal Transfer (LOYO)
                # ----------------------------------------------------
                test_mask = (y_year_encoded == target_year_idx)
                train_mask = ~test_mask
                
                res2, hist2 = run_sensitivity_fs(
                    features[train_mask], features[test_mask], 
                    class_labels[train_mask], class_labels[test_mask], 
                    y_year_encoded[train_mask], 
                    run_idx, param_name, val, current_params, "Temporal_Transfer", args
                )
                all_results.append(res2)
                all_histories.append(hist2)

    df = pd.DataFrame(all_results)
    
    # Summarize Results: Calculate Mean ± Std directly (响应审稿人：means and standard deviations after 30 independent runs)
    def ms(x): return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}" if len(x)>1 else f"{x.mean():.4f}"
    
    print("\n" + "="*50)
    print("SENSITIVITY ANALYSIS RESULTS (Top K={})".format(args.K))
    print("="*50)
    
    try:
        with pd.ExcelWriter(args.output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="Raw_Sensitivity_Data", index=False)
            
            for param in sensitivity_grids.keys():
                df_param = df[df["Parameter_Tested"] == param]
                
                tb = df_param.pivot_table(
                    index="Parameter_Value", 
                    columns="Experiment", 
                    values=["Accuracy", "W_Class", "W_Redundancy", "W_Year", "Time_Seconds"], 
                    aggfunc=ms
                )
                
                print(f"\n[TREND] Impact of {param}:")
                print(tb)
                
                tb.to_excel(writer, sheet_name=f"Trend_{param}")
                
        print(f"\n[INFO] Sensitivity results successfully saved to Excel file: {args.output_file}")
        
        # 导出包含了所有动态变化数据的 JSON (响应审稿人：the convergence curves during training, showing changes in weights)
        history_file = args.output_file.replace('.xlsx', '_history.json')
        with open(history_file, 'w') as f:
            json.dump(all_histories, f)
        print(f"[INFO] Training histories (curves, weights) saved to: {history_file}")
        
    except Exception as excel_err:
        print(f"\n[ERROR] Failed to save output files: {excel_err}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Sensitivity Analysis Runner (Dual Validation)")
    
    # 强制修改：按审稿人要求默认设为 30 次独立运行
    parser.add_argument("--n_run", type=int, default=30, help="Number of independent runs per parameter value (default: 30)")
    parser.add_argument("--K", type=int, default=15, help="Number of selected features to evaluate (default: 15)")
    parser.add_argument("--target_year", type=int, default=4, help="Target year value for Temporal Transfer validation (default: 4)")
    parser.add_argument("--output_file", type=str, default="grapevine_sensitivity_results.xlsx", help="Output Excel filename")
    parser.add_argument("--params", nargs="+", default=["N", "T", "CR", "MR", "Temp"], help="parameters to test (default: all)")
    parser.add_argument("--grape_path", type=str, default="datasets/Spectral_DataSet.xlsx", help="Grapevine data file")
    
    args = parser.parse_args()

    try:
        np.random.seed(12345)  # fixed balanced subset
        X, y_cls, y_year_raw = load_grapevine_data(args.grape_path)
        run_sensitivity_experiments(X, y_cls, y_year_raw, args) 
    except Exception as e: 
        print(f"Error: {e}")