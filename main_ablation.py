# main_ablation.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
import torch

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from core.fitness_fast import Obj1_Max_Class_MI_Fast, Obj2_Min_Redundancy_Safe, Obj3_Min_Year_MI_Fast
from core.ga_utils import jfs, keep_elite
from core.networks import train_softmask_dnn_proxy
from utils.metrics_tracker import METHOD_ORDER, perform_paired_ttest
from utils.data_loader import load_grapevine_data

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_fs_and_evaluate(xtrain, xtest, ytrain, ytest, yyear_tr, run_idx, exp_name):
    n_features = xtrain.shape[1]
    print(f"  [{exp_name} - Run {run_idx}] Pre-computing Mutual Information...")
    mi_class = mutual_info_classif(xtrain, ytrain, random_state=42)
    mi_year = mutual_info_classif(xtrain, yyear_tr, random_state=42)
    
    opts = {'xt_in': xtrain, 'mi_class': mi_class, 'mi_year': mi_year, 'N': 40, 'T': 40}
    
    print(f"  [{exp_name} - Run {run_idx}] Evolving Independent GA Populations...")
    res1 = jfs(opts, Obj1_Max_Class_MI_Fast); pop1 = keep_elite(res1['X'], res1['fit'])
    res2 = jfs(opts, Obj2_Min_Redundancy_Safe); pop2 = keep_elite(res2['X'], res2['fit'])
    res3 = jfs(opts, Obj3_Min_Year_MI_Fast); pop3 = keep_elite(res3['X'], res3['fit'])
    
    scores_single_class = pop1.mean(0)
    scores_single_year = pop3.mean(0)
    scores_mean_agg = (pop1.mean(0) + pop2.mean(0) + pop3.mean(0)) / 3.0
    
    scores_dnn_fixed_eq, _ = train_softmask_dnn_proxy(pop1, pop2, pop3, xtrain, ytrain, n_features, adaptive=False, fixed_weights=[1/3, 1/3, 1/3])
    scores_dnn_fixed_p1, _ = train_softmask_dnn_proxy(pop1, pop2, pop3, xtrain, ytrain, n_features, adaptive=False, fixed_weights=[0.5, 0.2, 0.3])
    scores_dnn_fixed_p2, _ = train_softmask_dnn_proxy(pop1, pop2, pop3, xtrain, ytrain, n_features, adaptive=False, fixed_weights=[0.2, 0.5, 0.3])
    scores_dnn_adapt, learned_lambda = train_softmask_dnn_proxy(pop1, pop2, pop3, xtrain, ytrain, n_features, adaptive=True)
    
    def get_top_k(scores, k):
        idx = np.argsort(scores)[::-1][:k]
        return idx if len(idx) > 0 else [0]
        
    clf = LinearDiscriminantAnalysis()
    results = []
    for k in range(1, 16):
        methods = {
            "Single-Obj (Class)": get_top_k(scores_single_class, k),
            "Single-Obj (Year)": get_top_k(scores_single_year, k),
            "Mean-Agg (No NN)": get_top_k(scores_mean_agg, k),
            "GA+DNN (Fixed W: Equal)": get_top_k(scores_dnn_fixed_eq, k),
            "GA+DNN (Fixed W: 0.5,0.2,0.3)": get_top_k(scores_dnn_fixed_p1, k),
            "GA+DNN (Fixed W: 0.2,0.5,0.3)": get_top_k(scores_dnn_fixed_p2, k),
            "GA+DNN (Adaptive - Ours)": get_top_k(scores_dnn_adapt, k)
        }
        for m_name, idx in methods.items():
            clf.fit(xtrain[:, idx], ytrain)
            acc_disease = clf.score(xtest[:, idx], ytest)
            results.append({"Experiment": exp_name, "Run": run_idx, "Method": m_name, "K": k, "Accuracy": acc_disease})
            
    return results, learned_lambda

def run_dual_experiment_benchmark(features, class_labels, year_raw, target_year_val=4, n_runs=10):
    le_year = LabelEncoder()
    y_year_encoded = le_year.fit_transform(year_raw)
    target_year_idx = le_year.transform([target_year_val])[0] if target_year_val in year_raw else np.max(y_year_encoded)

    all_results, lambdas_exp1, lambdas_exp2 = [], [], []
    
    for run_idx in range(1, n_runs + 1):
        set_seed(42 + run_idx)
        print(f"\n>>> Starting Run {run_idx}/{n_runs}...")
        
        # Exp 1: Random Split
        xt_tr1, xt_te1, yt_tr1, yt_te1, yr_tr1, _ = train_test_split(features, class_labels, y_year_encoded, test_size=0.3, stratify=class_labels, random_state=42+run_idx)
        res1, lam1 = run_fs_and_evaluate(xt_tr1, xt_te1, yt_tr1, yt_te1, yr_tr1, run_idx, "Exp1_Random_Split")
        all_results.extend(res1); lambdas_exp1.append(lam1)
        
        # Exp 2: Temporal Transfer
        test_mask = (y_year_encoded == target_year_idx)
        train_mask = ~test_mask
        res2, lam2 = run_fs_and_evaluate(features[train_mask], features[test_mask], class_labels[train_mask], class_labels[test_mask], y_year_encoded[train_mask], run_idx, "Exp2_Temporal_Transfer")
        all_results.extend(res2); lambdas_exp2.append(lam2)

    df_results = pd.DataFrame(all_results)
    

    def calc_mean_std(x): return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}" if len(x) > 1 else f"{x.mean():.4f} ± 0.0000"
    
    print("\n[TABLE 1] Experiment 1: Random Split")
    table_exp1 = df_results[df_results["Experiment"] == "Exp1_Random_Split"].pivot_table(index="K", columns="Method", values="Accuracy", aggfunc=calc_mean_std)[[m for m in METHOD_ORDER]]
    print(table_exp1)
    
    print("\n[TABLE 2] Experiment 2: Temporal Transfer")
    table_exp2 = df_results[df_results["Experiment"] == "Exp2_Temporal_Transfer"].pivot_table(index="K", columns="Method", values="Accuracy", aggfunc=calc_mean_std)[[m for m in METHOD_ORDER]]
    print(table_exp2)
    
    print("\n[T-TEST] Experiment 2: Temporal Transfer")
    print(perform_paired_ttest(df_results, "Exp2_Temporal_Transfer"))

if __name__ == "__main__":
    try:
        X, y_cls, y_year_raw = load_grapevine_data("datasets/Spectral_DataSet(1).xlsx")
        run_dual_experiment_benchmark(X, y_cls, y_year_raw, target_year_val=4, n_runs=10) 
    except Exception as e:
        print(f"Error loading real data: {e}. Running safe mock benchmark...")
        from sklearn.preprocessing import normalize
        X_mock = normalize(np.random.rand(200, 50) + 1e-8, norm='l1', axis=1)
        run_dual_experiment_benchmark(X_mock, np.random.randint(0, 2, 200), np.random.choice([2020, 2021, 2022], 200), target_year_val=2022, n_runs=2)