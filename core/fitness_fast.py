import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def _size_penalty(num_selected, total_features, lam=0.01):
    if num_selected == 0: return 1.0
    return lam * (num_selected / max(1, total_features))

def Obj2_Min_Redundancy_Safe(x, opts):
    sel = np.where(x == 1)[0]
    xt_in = opts['xt_in']
    if len(sel) <= 1: return 0.0 + _size_penalty(len(sel), xt_in.shape[1])
    Xs = xt_in[:, sel]
    Xs = (Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + 1e-8)
    C = np.abs(np.dot(Xs.T, Xs) / max(1, Xs.shape[0] - 1))
    np.fill_diagonal(C, 0)
    n = C.shape[0]
    return float(C.sum() / max(1, n * (n - 1))) + _size_penalty(n, xt_in.shape[1])

def Obj_Max_MI_Fast(x, opts, key='mi_class'):
    sel = np.where(x == 1)[0]
    if len(sel) == 0: return 1e6
    return -float(np.mean(opts[key][sel])) + _size_penalty(len(sel), len(x))

def Obj3_Min_Year_MI_Fast(x, opts):
    sel = np.where(x == 1)[0]
    if len(sel) == 0: return 1e6
    return float(np.mean(opts['mi_year'][sel])) + _size_penalty(len(sel), len(x))


def Obj_Min_Regression_Safe(x, opts):
    sel = np.where(x == 1)[0]
    if len(sel) == 0: return 1e6
    xt_in, yt_in, xv_in, yv_in = opts['xt_in'], opts['yt_in'], opts['xv_in'], opts['yv_in']
    model = LinearRegression().fit(xt_in[:, sel], yt_in)
    r2 = r2_score(yv_in, model.predict(xv_in[:, sel]))
    return -float(r2) + _size_penalty(len(sel), xt_in.shape[1])