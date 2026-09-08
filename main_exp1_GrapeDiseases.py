# main_exp1_GrapeDiseases.py
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

from utils.data_loader import load_grapevine_data
from utils.metrics_tracker import METHOD_ORDER, perform_paired_ttest
from core.ga_utils import jfs, keep_elite
from core.fitness_fast import Obj_Max_MI_Fast, Obj2_Min_Redundancy_Safe, Obj3_Min_Year_MI_Fast
from core.networks import train_softmask_dnn_proxy

def run_fs_and_evaluate(xtrain, xtest, ytrain, ytest, yyear_tr, run_idx, exp_name, args):
    opts = {
        'xt_in': xtrain, 
        'mi_class': mutual_info_classif(xtrain, ytrain), 
        'mi_year': mutual_info_classif(xtrain, yyear_tr), 
        'N': args.N,
        'T': args.T,
        'CR': args.CR,
        'MR': args.MR
    }
    
    start_time = time.time() 
    

    res1 = jfs(opts, lambda x, o: Obj_Max_MI_Fast(x, o, 'mi_class'))
    res2 = jfs(opts, Obj2_Min_Redundancy_Safe)
    res3 = jfs(opts, Obj3_Min_Year_MI_Fast)
    
    pop1, curve1 = keep_elite(**res1), res1.get('curve', [])
    pop2, curve2 = keep_elite(**res2), res2.get('curve', [])
    pop3, curve3 = keep_elite(**res3), res3.get('curve', [])
    
    sc_c, sc_y, sc_mean = pop1.mean(0), pop3.mean(0), (pop1.mean(0)+pop2.mean(0)+pop3.mean(0))/3.0
    

    sc_eq, _, _, _ = train_softmask_dnn_proxy(pop1, pop2, pop3, xtrain, ytrain, xtrain.shape[1], False, [1/3]*3)
    sc_p1, _, _, _ = train_softmask_dnn_proxy(pop1, pop2, pop3, xtrain, ytrain, xtrain.shape[1], False, [0.5, 0.2, 0.3])
    sc_p2, _, _, _ = train_softmask_dnn_proxy(pop1, pop2, pop3, xtrain, ytrain, xtrain.shape[1], False, [0.2, 0.5, 0.3])
    
    sc_ad, lam, w_hist, l_hist = train_softmask_dnn_proxy(pop1, pop2, pop3, xtrain, ytrain, xtrain.shape[1], True)
    

    total_fs_time = time.time() - start_time
    
    clf = LinearDiscriminantAnalysis()
    def get_top_k(scores, k): 
        idx = np.argsort(scores)[::-1][:k]
        return idx if len(idx)>0 else [0]
    
    results = []
    
    for k in range(1, args.K + 1):
        methods = {
            "Single-Obj (Class)": get_top_k(sc_c, k), 
            "Single-Obj (Year)": get_top_k(sc_y, k), 
            "Mean-Agg (No NN)": get_top_k(sc_mean, k), 
            "GA+DNN (Fixed W: Equal)": get_top_k(sc_eq, k), 
            "GA+DNN (Fixed W: 0.5,0.2,0.3)": get_top_k(sc_p1, k), 
            "GA+DNN (Fixed W: 0.2,0.5,0.3)": get_top_k(sc_p2, k), 
            "GA+DNN (Adaptive)": get_top_k(sc_ad, k)
        }
        for m_name, idx in methods.items():
            results.append({
                "Experiment": exp_name, 
                "Run": run_idx, 
                "Method": m_name, 
                "K": k, 
                "Accuracy": clf.fit(xtrain[:, idx], ytrain).score(xtest[:, idx], ytest),
                "Time_Seconds": total_fs_time if m_name == "GA+DNN (Adaptive)" else 0, 
                "W_Class": lam[0] if lam is not None else None,
                "W_Redundancy": lam[1] if lam is not None else None,
                "W_Year": lam[2] if lam is not None else None
            })
            

    training_history = {
        "Run": run_idx,
        "Experiment": exp_name,
        "GA_Curve_Obj1": curve1,
        "GA_Curve_Obj2": curve2,
        "GA_Curve_Obj3": curve3,
        "DNN_Weight_History": w_hist,
        "DNN_Loss_History": l_hist
    }
            
    return results, training_history

def run_dual_experiment_benchmark(features, class_labels, year_raw, args):
    le_year = LabelEncoder()
    y_year_encoded = le_year.fit_transform(year_raw)
    
    target_year_val = args.target_year
    target_year_idx = le_year.transform([target_year_val])[0] if target_year_val in year_raw else np.max(y_year_encoded)
    
    all_results = []
    all_histories = [] 
    
    for run_idx in range(1, args.n_run + 1):
        print(f"\n>>> Starting Run {run_idx}/{args.n_run}...")
        
        xt_tr1, xt_te1, yt_tr1, yt_te1, yr_tr1, _ = train_test_split(
            features, class_labels, y_year_encoded, 
            test_size=0.3, stratify=class_labels
        )
        res1, hist1 = run_fs_and_evaluate(xt_tr1, xt_te1, yt_tr1, yt_te1, yr_tr1, run_idx, "Exp1_Random_Split", args)
        
        test_mask = (y_year_encoded == target_year_idx)
        train_mask = ~test_mask
        res2, hist2 = run_fs_and_evaluate(
            features[train_mask], features[test_mask], 
            class_labels[train_mask], class_labels[test_mask], 
            y_year_encoded[train_mask], run_idx, "Exp2_Temporal_Transfer", args
        )
        
        all_results.extend(res1 + res2)
        all_histories.extend([hist1, hist2])

    df = pd.DataFrame(all_results)
    def ms(x): return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}" if len(x)>1 else f"{x.mean():.4f}"
    
    tb1 = df[df["Experiment"] == "Exp1_Random_Split"].pivot_table(index="K", columns="Method", values="Accuracy", aggfunc=ms)
    tb1_ordered = tb1[[m for m in METHOD_ORDER if m in tb1.columns]]
    print("\n[TABLE 1] Exp 1: Random Split\n", tb1_ordered)
    
    tb2 = df[df["Experiment"] == "Exp2_Temporal_Transfer"].pivot_table(index="K", columns="Method", values="Accuracy", aggfunc=ms)
    tb2_ordered = tb2[[m for m in METHOD_ORDER if m in tb2.columns]]
    print("\n[TABLE 2] Exp 2: Temporal Transfer\n", tb2_ordered)

    try:
        with pd.ExcelWriter(args.output_file, engine='openpyxl') as writer:
            tb1_ordered.to_excel(writer, sheet_name="Random Split")
            tb2_ordered.to_excel(writer, sheet_name="Temporal Transfer")
            

            df.to_excel(writer, sheet_name="Raw_Data_Time_Weights", index=False)
            
        print(f"\n[INFO] Results successfully saved to Excel file: {args.output_file}")
        
        # 将曲线与历史文件作为 JSON 单独输出，后续可轻松通过 matplotlib 绘制
        history_file = args.output_file.replace('.xlsx', '_history.json')
        with open(history_file, 'w') as f:
            json.dump(all_histories, f)
        print(f"[INFO] Training histories (curves, weights) saved to: {history_file}")
        
    except Exception as excel_err:
        print(f"\n[ERROR] Failed to save output files: {excel_err}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Task-Guided Feature Selection Experiment Runner")
    
    parser.add_argument("--n_run", type=int, default=30, help="Number of independent runs (default: 30)")
    parser.add_argument("--K", type=int, default=15, help="Maximum number of selected features to evaluate (default: 15)")
    parser.add_argument("--N", type=int, default=40, help="GA Population size (default: 40)")
    parser.add_argument("--T", type=int, default=40, help="GA Max generations/iterations (default: 40)")
    parser.add_argument("--CR", type=float, default=0.8, help="GA Crossover Rate (default: 0.8)")
    parser.add_argument("--MR", type=float, default=0.2, help="GA Mutation Rate (default: 0.2)")
    parser.add_argument("--G", type=int, default=60, help="SoftMaskNet training epochs (default: 60)")
    parser.add_argument("--target_year", type=int, default=4, help="Target year value for LOYO validation (default: 4)")
    parser.add_argument("--output_file", type=str, default="grapevine_ablation_results.xlsx", help="Output Excel filename")
    
    args = parser.parse_args()

    print("=" * 50)
    print("Running Experiments with the following parameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 50)

    try:
        X, y_cls, y_year_raw = load_grapevine_data()
        run_dual_experiment_benchmark(X, y_cls, y_year_raw, args) 
    except Exception as e: 
        print(f"Error: {e}")