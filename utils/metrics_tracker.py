# utils/metrics_tracker.py
import pandas as pd
from scipy import stats

METHOD_ORDER = [
    "Single-Obj (Class)", 
    "Single-Obj (Year)", 
    "Mean-Agg (No NN)", 
    "GA+DNN (Fixed W: Equal)",       
    "GA+DNN (Fixed W: 0.5,0.2,0.3)", 
    "GA+DNN (Fixed W: 0.2,0.5,0.3)", 
    "GA+DNN (Adaptive - Ours)"
]

def get_significance_stars(p_val):
    if pd.isna(p_val): return ""
    if p_val < 0.001: return "***"
    if p_val < 0.01: return "**"
    if p_val < 0.05: return "*"
    return ""

def perform_paired_ttest(df, experiment_name, target_method="GA+DNN (Adaptive - Ours)"):
    df_exp = df[df["Experiment"] == experiment_name]
    baselines = [m for m in METHOD_ORDER if m != target_method]
    k_values = sorted(df_exp["K"].unique())
    
    ttest_results = []
    for k in k_values:
        df_k = df_exp[df_exp["K"] == k]
        target_accs = df_k[df_k["Method"] == target_method].sort_values("Run")["Accuracy"].values
        row_result = {"K": k}
        
        for baseline in baselines:
            base_accs = df_k[df_k["Method"] == baseline].sort_values("Run")["Accuracy"].values
            if len(target_accs) > 1 and len(base_accs) > 1:
                t_stat, p_val = stats.ttest_rel(target_accs, base_accs)
                stars = get_significance_stars(p_val)
                row_result[baseline] = f"p={p_val:.4f}{stars}"
            else:
                row_result[baseline] = "N/A"
        ttest_results.append(row_result)
        
    return pd.DataFrame(ttest_results).set_index("K")