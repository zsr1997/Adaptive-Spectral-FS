# -*- coding: utf-8 -*-
# core/fitness_fast.py
import numpy as np

def _size_penalty(num_selected, total_features, lam=0.01):
    if num_selected == 0: return 1.0
    return lam * (num_selected / max(1, total_features))

def Obj1_Max_Class_MI_Fast(x, opts):
    sel = np.where(x == 1)[0]
    if len(sel) == 0: return 1e6
    mi_scores = opts['mi_class']
    return -float(np.mean(mi_scores[sel])) + _size_penalty(len(sel), len(x))

def Obj2_Min_Redundancy_Safe(x, opts):
    sel = np.where(x == 1)[0]
    xt_in = opts['xt_in']
    if len(sel) <= 1: return 0.0 + _size_penalty(len(sel), xt_in.shape[1])
    Xs = xt_in[:, sel]
    Xs = (Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + 1e-8)
    C = np.abs(np.dot(Xs.T, Xs) / max(1, Xs.shape[0] - 1))
    np.fill_diagonal(C, 0)
    n = C.shape[0]
    return float(C.sum() / (n * (n - 1))) + _size_penalty(n, xt_in.shape[1])

def Obj3_Min_Year_MI_Fast(x, opts):
    sel = np.where(x == 1)[0]
    if len(sel) == 0: return 1e6
    mi_scores = opts['mi_year']
    return float(np.mean(mi_scores[sel])) + _size_penalty(len(sel), len(x))