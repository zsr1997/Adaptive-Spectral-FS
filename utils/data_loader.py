# utils/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

def load_grapevine_data(filepath="datasets/Spectral_DataSet(1).xlsx"):
    print("Loading Grapevine Dataset...")
    df = pd.read_excel(filepath)
    
    class_col_idx = -4
    year_col_idx = -1
    y_class_raw = df.iloc[:, class_col_idx].values
    mask_cls = np.isin(y_class_raw, [2, 3])
    df_filtered = df[mask_cls].copy()
    
    balanced_indices = []
    curr_years = df_filtered.iloc[:, year_col_idx].values
    curr_classes = df_filtered.iloc[:, class_col_idx].values
    unique_years = np.unique(curr_years)
    
    for yr in unique_years:
        for cls in [2, 3]:
            match_mask = (curr_years == yr) & (curr_classes == cls)
            match_indices = np.where(match_mask)[0]
            if len(match_indices) >= 100:
                selected = np.random.choice(match_indices, 100, replace=False)
            else: 
                selected = match_indices
            balanced_indices.extend(selected)
            
    df_balanced = df_filtered.iloc[balanced_indices].copy()
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df_balanced.iloc[:, :600].values.astype(float)
    X = normalize(X + 1e-8, norm='l1', axis=1)
    y_cls = np.where(df_balanced.iloc[:, class_col_idx].values == 2, 0, 1)
    y_year_raw = df_balanced.iloc[:, year_col_idx].values
    
    return X, y_cls, y_year_raw