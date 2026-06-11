import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import normalize

# ---------------------------------------------------------
# Exp 1: Indian Pines
# ---------------------------------------------------------
def _load_mat_first_var(path):
    mat = sio.loadmat(path)
    for k in mat.keys():
        if not k.startswith('__'): return mat[k]
    raise ValueError(f"No valid variable found in {path}")

def load_indian_pines(data_path="datasets/Indian_pines_corrected.mat", gt_path="datasets/Indian_pines_gt.mat"):
    print("Loading Indian Pines Dataset...")
    img = _load_mat_first_var(data_path)
    lab = _load_mat_first_var(gt_path)
    if lab.ndim == 3 and lab.shape[-1] == 1: lab = lab[:, :, 0]
    lab = lab.astype(int)
    
    mask = lab > 0
    X = img[mask].astype(float)
    y_fine = lab[mask].astype(int)
    
    crop_classes = [1, 2, 3, 4, 9, 10, 11, 12, 14]
    y_crop = np.isin(y_fine, crop_classes).astype(int)
    
    np.random.seed(42)
    idx_selected = []
    for c in np.unique(y_fine):
        c_idx = np.where(y_fine == c)[0]
        selected = np.random.choice(c_idx, 20, replace=False) if len(c_idx) >= 20 else c_idx
        idx_selected.extend(selected)
    idx_selected = np.array(idx_selected)
    np.random.shuffle(idx_selected)
    
    X_sub, y_fine_sub, y_crop_sub = X[idx_selected], y_fine[idx_selected], y_crop[idx_selected]
    X_sub = normalize(X_sub + 1e-8, norm="l1", axis=1)
    return X_sub, y_fine_sub, y_crop_sub

# ---------------------------------------------------------
# Exp 2: Sugar Dual-Task
# ---------------------------------------------------------
def load_sugar_data(filepath="datasets/DATASET.xlsx"):
    print("Loading Sugar Dataset...")
    df = pd.read_excel(filepath)
    label = df.iloc[:, 1].to_numpy()
    sugar = df.iloc[:, 2].to_numpy()
    features = df.drop(df.columns[[0, 1, 2]], axis=1).values
    features = normalize(features + 1e-8, norm="l1", axis=1)
    return features, label, sugar

# ---------------------------------------------------------
# Exp 3: Grapevine Transfer
# ---------------------------------------------------------
def load_grapevine_data(filepath="datasets/Spectral_DataSet.xlsx"):
    print("Loading Grapevine Dataset...")
    df = pd.read_excel(filepath)
    class_col_idx, year_col_idx = -4, -1
    y_class_raw = df.iloc[:, class_col_idx].values
    df_filtered = df[np.isin(y_class_raw, [2, 3])].copy()
    
    balanced_indices = []
    curr_years, curr_classes = df_filtered.iloc[:, year_col_idx].values, df_filtered.iloc[:, class_col_idx].values
    for yr in np.unique(curr_years):
        for cls in [2, 3]:
            match_mask = (curr_years == yr) & (curr_classes == cls)
            match_indices = np.where(match_mask)[0]
            selected = np.random.choice(match_indices, 100, replace=False) if len(match_indices) >= 100 else match_indices
            balanced_indices.extend(selected)
            
    df_balanced = df_filtered.iloc[balanced_indices].copy().sample(frac=1, random_state=42).reset_index(drop=True)
    X = normalize(df_balanced.iloc[:, :600].values.astype(float) + 1e-8, norm='l1', axis=1)
    y_cls = np.where(df_balanced.iloc[:, class_col_idx].values == 2, 0, 1)
    y_year_raw = df_balanced.iloc[:, year_col_idx].values
    return X, y_cls, y_year_raw
