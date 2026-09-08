# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import argparse

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from utils.data_loader import load_sugar_data
from core.ga_utils import jfs, keep_elite
from core.fitness_fast import Obj_Max_MI_Fast, Obj2_Min_Redundancy_Safe, Obj_Min_Regression_Safe
from core.networks import train_softmask_dnn_dual_proxy

def run_sugar_ablation(features, yc_all, yr_all, args):
    all_results, lambdas_tracked = [], []

    for run_idx in range(1, args.n_run + 1):
        print(f"\n>>> Run {run_idx}/{args.n_run}...")
        
        xt_tr, xt_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
            features, yc_all, yr_all, test_size=0.3, stratify=yc_all
        )
        xt_in, xv_in, yc_in, yc_val, yr_in, yr_val = train_test_split(
            xt_tr, yc_tr, yr_tr, test_size=0.25, stratify=yc_tr
        )
        
        opts = {
            'xt_in': xt_in, 
            'yt_in': yr_in, 
            'xv_in': xv_in, 
            'yv_in': yr_val, 
            'mi_class': mutual_info_classif(xt_in, yc_in), 
            'N': args.N, 
            'T': args.T,
            'CR': args.CR,
            'MR': args.MR
        }

        pop1 = keep_elite(**jfs(opts, lambda x, o: Obj_Max_MI_Fast(x, o, 'mi_class')))
        pop2 = keep_elite(**jfs(opts, Obj2_Min_Redundancy_Safe))
        pop3 = keep_elite(**jfs(opts, Obj_Min_Regression_Safe))

        sc_t1, sc_t2, sc_mean = pop1.mean(0), pop3.mean(0), (pop1.mean(0)+pop2.mean(0)+pop3.mean(0))/3.0
        
        sc_eq, _ = train_softmask_dnn_dual_proxy(pop1, pop2, pop3, xt_tr, yc_tr, yr_tr, xt_tr.shape[1], False, [1/3]*3)
        sc_p1, _ = train_softmask_dnn_dual_proxy(pop1, pop2, pop3, xt_tr, yc_tr, yr_tr, xt_tr.shape[1], False, [0.5, 0.2, 0.3])
        sc_p2, _ = train_softmask_dnn_dual_proxy(pop1, pop2, pop3, xt_tr, yc_tr, yr_tr, xt_tr.shape[1], False, [0.2, 0.5, 0.3])
        sc_ad, lam = train_softmask_dnn_dual_proxy(pop1, pop2, pop3, xt_tr, yc_tr, yr_tr, xt_tr.shape[1], True)
        
        if lam is not None: 
            lambdas_tracked.append(lam)

        clf = LinearDiscriminantAnalysis()
        reg = RandomForestRegressor(n_estimators=50)
        
        def get_top_k(scores, k): 
            idx = np.argsort(scores)[::-1][:k]
            return idx if len(idx)>0 else [0]

        for k in range(1, args.K + 1):
            methods = {
                "Single-Obj (Class)": get_top_k(sc_t1, k), 
                "Single-Obj (Reg)": get_top_k(sc_t2, k), 
                "Mean-Agg (No NN)": get_top_k(sc_mean, k), 
                "GA+DNN (Fixed W: Equal)": get_top_k(sc_eq, k), 
                "GA+DNN (Fixed W: 0.5,0.2,0.3)": get_top_k(sc_p1, k), 
                "GA+DNN (Fixed W: 0.2,0.5,0.3)": get_top_k(sc_p2, k), 
                "GA+DNN (Adaptive)": get_top_k(sc_ad, k)
            }
            for m_name, idx in methods.items():
                all_results.append({
                    "Run": run_idx, 
                    "Method": m_name, 
                    "K": k, 
                    "Task": "Classification", 
                    "Value": clf.fit(xt_tr[:, idx], yc_tr).score(xt_te[:, idx], yc_te)
                })
                all_results.append({
                    "Run": run_idx, 
                    "Method": m_name, 
                    "K": k, 
                    "Task": "Regression", 
                    "Value": r2_score(yr_te, reg.fit(xt_tr[:, idx], yr_tr).predict(xt_te[:, idx]))
                })

    df = pd.DataFrame(all_results)
    
    def ms(x): 
        return f"{x.mean():.4f} \u00B1 {x.std(ddof=1):.4f}" if len(x)>1 else f"{x.mean():.4f}"
    
    SUGAR_ORDER = [
        "Single-Obj (Class)", "Single-Obj (Reg)", "Mean-Agg (No NN)", 
        "GA+DNN (Fixed W: Equal)", "GA+DNN (Fixed W: 0.5,0.2,0.3)", 
        "GA+DNN (Fixed W: 0.2,0.5,0.3)", "GA+DNN (Adaptive - Ours)"
    ]
    
    tb1 = df[df["Task"] == "Classification"].pivot_table(index="K", columns="Method", values="Value", aggfunc=ms)
    tb1_ordered = tb1[[m for m in SUGAR_ORDER if m in tb1.columns]]
    print("\n[TABLE 1] Classification Accuracy\n", tb1_ordered)
    
    tb2 = df[df["Task"] == "Regression"].pivot_table(index="K", columns="Method", values="Value", aggfunc=ms)
    tb2_ordered = tb2[[m for m in SUGAR_ORDER if m in tb2.columns]]
    print("\n[TABLE 2] Regression R2 Score\n", tb2_ordered)

    try:
        with pd.ExcelWriter(args.output_file, engine='openpyxl') as writer:
            tb1_ordered.to_excel(writer, sheet_name="Classification Accuracy")
            tb2_ordered.to_excel(writer, sheet_name="Regression R2 Score")
        print(f"\n[INFO] Results successfully saved to Excel file: {args.output_file}")
    except Exception as excel_err:
        print(f"\n[ERROR] Failed to save Excel file: {excel_err}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sugar Dataset Dual-Task Feature Selection Runner")
    
    parser.add_argument("--n_run", type=int, default=10, help="Number of independent runs (default: 10)")
    parser.add_argument("--K", type=int, default=15, help="Maximum number of selected features (default: 15)")
    parser.add_argument("--N", type=int, default=40, help="GA Population size (default: 40)")
    parser.add_argument("--T", type=int, default=40, help="GA Max generations/iterations (default: 40)")
    parser.add_argument("--CR", type=float, default=0.8, help="GA Crossover Rate (default: 0.8)")
    parser.add_argument("--MR", type=float, default=0.2, help="GA Mutation Rate (default: 0.2)")
    parser.add_argument("--G", type=int, default=60, help="SoftMaskNet training epochs (default: 60)")
    parser.add_argument("--output_file", type=str, default="sugar_ablation_results.xlsx", help="Output Excel filename")
    
    args = parser.parse_args()

    print("=" * 50)
    print("Running Sugar Experiments with the following parameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 50)

    try:
        X, yc, yr = load_sugar_data()
        run_sugar_ablation(X, yc, yr, args) 
    except Exception as e: 
        print(f"Error: {e}")