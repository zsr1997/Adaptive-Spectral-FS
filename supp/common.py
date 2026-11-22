# supp/common.py
"""
Shared utilities for the supplementary experiments requested in review round 2.

Run every script from the repository root, e.g.
    python supp/step2_extended_ablation.py --case sugar

Design rules (all of them answer a specific reviewer concern):
  * Every run r uses seed = BASE_SEED + r for the split, MI estimation, GA and network,
    so that all methods (step 2) and all MOEAs (step 5) are evaluated on IDENTICAL splits
    -> paired statistical tests are valid.
  * Top-K selection uses a STABLE, deterministic tie-break (relevance to the primary target,
    then band index). The original code used np.argsort on float32/float64 arrays, which
    breaks ties differently and produced spurious differences between Mean-Agg and Fixed-Eq.
  * The GA honours CR / MR (the original jfs() hard-coded 0.8 / 0.01).
"""
import os
import sys
import time
import json
import random
import itertools

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, normalize

BASE_SEED = 2025
CASES = ["grape_random", "grape_loyo", "sugar", "ip"]

# Order of the three populations is the SAME as in the original code:
#   pop1 = primary task, pop2 = redundancy, pop3 = secondary task / confounder
TASK_NAMES = {
    "grape_random": ["Class", "Redundancy", "Year"],
    "grape_loyo": ["Class", "Redundancy", "Year"],
    "sugar": ["Class", "Redundancy", "Regression"],
    "ip": ["Fine", "Redundancy", "Coarse"],
}
METRICS = {
    "grape_random": ["Acc"],
    "grape_loyo": ["Acc"],
    "sugar": ["Acc", "R2"],
    "ip": ["AccFine", "AccCoarse"],
}
# Indian Pines: the paper says 10 cultivated crop classes (200 crop vs 120 non-crop).
# The released loader used [1,2,3,4,9,10,11,12,14] (includes Woods, misses Hay/Wheat -> 180 vs 140).
IP_CROP_PAPER = [1, 2, 3, 4, 8, 9, 10, 11, 12, 13]
IP_CROP_CODE = [1, 2, 3, 4, 9, 10, 11, 12, 14]


# ----------------------------------------------------------------------------- seeds / io
def run_seed(run):
    return BASE_SEED + int(run)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def append_csv(df, path):
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


# ----------------------------------------------------------------------------- data
def _load_mat_first_var(path):
    import scipy.io as sio
    mat = sio.loadmat(path)
    for k in mat.keys():
        if not k.startswith("__"):
            return mat[k]
    raise ValueError(path)


def load_ip(crop_classes, data_path="datasets/Indian_pines_corrected.mat",
            gt_path="datasets/Indian_pines_gt.mat"):
    """Same procedure as utils.data_loader.load_indian_pines, but the crop list is a parameter."""
    img = _load_mat_first_var(os.path.join(ROOT, data_path))
    lab = _load_mat_first_var(os.path.join(ROOT, gt_path)).astype(int)
    if lab.ndim == 3:
        lab = lab[:, :, 0]
    mask = lab > 0
    X, y_fine = img[mask].astype(float), lab[mask].astype(int)
    y_crop = np.isin(y_fine, crop_classes).astype(int)
    rs = np.random.RandomState(42)
    idx = []
    for c in np.unique(y_fine):
        ci = np.where(y_fine == c)[0]
        idx.extend(rs.choice(ci, 20, replace=False) if len(ci) >= 20 else ci)
    idx = np.array(idx)
    rs.shuffle(idx)
    X = normalize(X[idx] + 1e-8, norm="l1", axis=1)
    return X, y_fine[idx], y_crop[idx]


def load_case_data(case, grape_path=None, ip_crop="paper"):
    if case.startswith("grape"):
        from utils.data_loader import load_grapevine_data
        np.random.seed(12345)  # the released loader sub-samples WITHOUT a seed -> fix it here
        if grape_path:
            X, y, year_raw = load_grapevine_data(grape_path)
        else:
            X, y, year_raw = load_grapevine_data()
        return dict(X=np.asarray(X, float), y1=np.asarray(y).astype(int),
                    aux=LabelEncoder().fit_transform(year_raw), year_raw=np.asarray(year_raw))
    if case == "sugar":
        from utils.data_loader import load_sugar_data
        X, yc, yr = load_sugar_data(os.path.join(ROOT, "datasets/DATASET.xlsx"))
        return dict(X=np.asarray(X, float), y1=LabelEncoder().fit_transform(yc),
                    yreg=np.asarray(yr, float))
    if case == "ip":
        crop = IP_CROP_PAPER if ip_crop == "paper" else IP_CROP_CODE
        X, yf, yc = load_ip(crop)
        return dict(X=X, y1=LabelEncoder().fit_transform(yf), y2=yc.astype(int))
    raise ValueError(case)


def make_split(case, data, seed):
    """Identical split for every method of a given run (needed for paired tests)."""
    idx = np.arange(len(data["X"]))
    if case == "grape_loyo":
        test = data["aux"] == data["aux"].max()   # last acquisition year (2024)
        return idx[~test], idx[test]
    tr, te = train_test_split(idx, test_size=0.3, stratify=data["y1"], random_state=seed)
    return np.sort(tr), np.sort(te)


# ----------------------------------------------------------------------------- objectives
def _penalty(n, d, lam):
    return np.where(n == 0, 1.0, lam * n / d)


def make_obj_mi(mi, sign, lam=0.01):
    """sign=-1: maximise mean MI (task relevance); sign=+1: minimise mean MI (confounder)."""
    mi = np.asarray(mi, float)

    def f(X):
        X = np.asarray(X, float)
        n = X.sum(1)
        d = X.shape[1]
        val = sign * (X @ mi) / np.maximum(n, 1)
        out = val + lam * n / d
        out[n == 0] = 1e6
        return out
    return f


def abs_corr_matrix(Xobj):
    """Identical to Obj2_Min_Redundancy_Safe: |Xs^T Xs|/(n-1) with ddof=0 standardisation."""
    Xs = (Xobj - Xobj.mean(0)) / (Xobj.std(0) + 1e-8)
    C = np.abs(Xs.T @ Xs) / max(1, Xs.shape[0] - 1)
    np.fill_diagonal(C, 0.0)
    return C


def make_obj_red(C, lam=0.01):
    def f(X):
        X = np.asarray(X, float)
        n = X.sum(1)
        d = X.shape[1]
        num = np.einsum("ij,ij->i", X @ C, X)
        val = np.where(n > 1, num / np.maximum(n * (n - 1), 1), 0.0)
        return val + _penalty(n, d, lam)
    return f


def make_obj_reg(Xin, yin, Xval, yval, lam=0.01):
    """Identical to Obj_Min_Regression_Safe (OLS fitted on xt_in, R2 on xv_in), with a cache."""
    cache = {}

    def f(X):
        X = np.asarray(X)
        out = np.empty(len(X))
        for i, x in enumerate(X):
            key = np.packbits(x.astype(bool)).tobytes()
            if key not in cache:
                sel = np.flatnonzero(x)
                if len(sel) == 0:
                    cache[key] = 1e6
                else:
                    m = LinearRegression().fit(Xin[:, sel], yin)
                    cache[key] = -r2_score(yval, m.predict(Xval[:, sel])) + lam * len(sel) / x.size
            out[i] = cache[key]
        return out
    return f


# ----------------------------------------------------------------------------- GA
def run_ga(fitness, d, N=40, G=40, CR=0.8, MR=0.01, rng=None, init_p=0.5):
    """Same operators as core/ga_utils.jfs (fitness-proportional roulette on 1/(1+shifted fit),
    single-point crossover, bit-flip mutation, (mu+lambda) truncation) but CR/MR are honoured
    and the RNG is explicit."""
    rng = rng if rng is not None else np.random.default_rng()
    X = (rng.random((N, d)) < init_p).astype(np.int8)
    fit = fitness(X)
    n_evals = N
    curve = [float(fit.min())]
    for _ in range(1, G):
        inv = 1.0 / (1.0 + (fit - fit.min()) + 1e-8)
        prob = inv / inv.sum()
        Nc = max(1, int((rng.random(N) < CR).sum()))
        P1 = X[rng.choice(N, Nc, p=prob)]
        P2 = X[rng.choice(N, Nc, p=prob)]
        cut = rng.integers(1, d - 1, size=Nc) if d > 2 else np.ones(Nc, int)
        left = np.arange(d)[None, :] < cut[:, None]
        kids = np.vstack([np.where(left, P1, P2), np.where(left, P2, P1)])
        flip = rng.random(kids.shape) < MR
        kids = np.where(flip, 1 - kids, kids).astype(np.int8)
        fk = fitness(kids)
        n_evals += len(kids)
        XX, FF = np.vstack([X, kids]), np.concatenate([fit, fk])
        keep = np.argsort(FF, kind="stable")[:N]
        X, fit = XX[keep], FF[keep]
        curve.append(float(fit.min()))
    return dict(X=X, fit=fit, curve=curve, n_evals=n_evals)


def keep_elite(X, fit, ratio=0.2, min_n=5):
    ne = max(min_n, int(np.ceil(len(X) * ratio)))
    return X[np.argsort(fit, kind="stable")[:ne]]


def zscore_objectives(objs, d, rng, n_ref=200):
    """Normalisation used by Joint-GA: z-score each objective on random masks (same init
    distribution as the GA), so that a weighted sum is meaningful."""
    R = (rng.random((n_ref, d)) < 0.5).astype(np.int8)
    stats = []
    for f in objs:
        v = f(R)
        v = v[v < 1e5]
        stats.append((float(v.mean()), float(v.std() + 1e-12)))
    return stats


def make_joint_fitness(objs, stats, w):
    def f(X):
        tot = np.zeros(len(X))
        for wi, fi, (mu, sd) in zip(w, objs, stats):
            v = fi(X)
            tot += wi * np.where(v >= 1e5, 1e3, (v - mu) / sd)
        return tot
    return f


# ----------------------------------------------------------------------------- Top-K
def topk(scores, k, tiebreak):
    """Deterministic Top-K: sort by score (rounded to remove float32/float64 noise), then by
    the tie-break vector (higher first), then by band index."""
    s = np.round(np.asarray(scores, dtype=np.float64), 9)
    order = np.lexsort((np.arange(len(s)), -np.asarray(tiebreak, float), -s))
    return np.sort(order[:k])


def simplex_grid(T=3, step=0.1):
    m = int(round(1.0 / step))
    return [np.array(c, float) / m for c in itertools.product(range(m + 1), repeat=T) if sum(c) == m]


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / max(1, len(a | b))


# ----------------------------------------------------------------------------- run context
class Ctx:
    """Everything one (case, run) needs: split, objectives, evaluation, inner CV."""

    def __init__(self, case, data, run, lam=0.01, cv_folds=5, cv_rf_trees=50, rf_trees=50):
        self.case, self.run, self.seed = case, run, run_seed(run)
        self.data = data
        self.tr, self.te = make_split(case, data, self.seed)
        X = data["X"]
        self.Xtr, self.Xte = X[self.tr], X[self.te]
        self.d = X.shape[1]
        self.y1tr, self.y1te = data["y1"][self.tr], data["y1"][self.te]
        self.cv_folds, self.cv_rf_trees, self.rf_trees = cv_folds, cv_rf_trees, rf_trees
        self.names = TASK_NAMES[case]
        self.metrics = METRICS[case]
        self._cv_cache = {}
        s = self.seed

        if case.startswith("grape"):
            self.auxtr = data["aux"][self.tr]
            Xobj = self.Xtr
            self.mi1 = mutual_info_classif(Xobj, self.y1tr, random_state=s)
            self.mi_aux = mutual_info_classif(Xobj, self.auxtr, random_state=s)
            self.C = abs_corr_matrix(Xobj)
            self.objs = [make_obj_mi(self.mi1, -1, lam), make_obj_red(self.C, lam),
                         make_obj_mi(self.mi_aux, +1, lam)]
            self.proxy = dict(kind="binary", y=self.y1tr, logit_lr=0.01, legacy_sampling="per_sample")
        elif case == "sugar":
            self.yrtr, self.yrte = data["yreg"][self.tr], data["yreg"][self.te]
            # inner split used by the regression objective (as in main_exp2)
            itr, iva = train_test_split(np.arange(len(self.tr)), test_size=0.25,
                                        stratify=self.y1tr, random_state=s)
            Xin, Xva = self.Xtr[itr], self.Xtr[iva]
            self.mi1 = mutual_info_classif(Xin, self.y1tr[itr], random_state=s)
            self.C = abs_corr_matrix(Xin)
            xc = (Xin - Xin.mean(0)) / (Xin.std(0) + 1e-8)
            yc = (self.yrtr[itr] - self.yrtr[itr].mean()) / (self.yrtr[itr].std() + 1e-8)
            self.mi_aux = np.abs(xc.T @ yc) / len(yc)      # |Pearson r| with sugar (descriptive only)
            self.objs = [make_obj_mi(self.mi1, -1, lam), make_obj_red(self.C, lam),
                         make_obj_reg(Xin, self.yrtr[itr], Xva, self.yrtr[iva], lam)]
            self.proxy = dict(kind="dual", y=self.y1tr, yreg=self.yrtr, n_classes=len(np.unique(data["y1"])),
                              logit_lr=0.05, legacy_sampling="per_sample")
        elif case == "ip":
            self.y2tr, self.y2te = data["y2"][self.tr], data["y2"][self.te]
            self.mi1 = mutual_info_classif(self.Xtr, self.y1tr, random_state=s)
            self.mi_aux = mutual_info_classif(self.Xtr, self.y2tr, random_state=s)
            self.C = abs_corr_matrix(self.Xtr)
            self.objs = [make_obj_mi(self.mi1, -1, lam), make_obj_red(self.C, lam),
                         make_obj_mi(self.mi_aux, -1, lam)]
            self.proxy = dict(kind="multiclass", y=self.y1tr, n_classes=len(np.unique(data["y1"])),
                              logit_lr=0.02, legacy_sampling="per_batch")
        self.tiebreak = self.mi1  # relevance to the primary target, training data only

    # -- downstream evaluation on the hold-out test set (same models as the paper)
    def evaluate(self, idx):
        idx = np.asarray(idx, int)
        out = {}
        lda = LinearDiscriminantAnalysis()
        if self.case.startswith("grape"):
            out["Acc"] = lda.fit(self.Xtr[:, idx], self.y1tr).score(self.Xte[:, idx], self.y1te)
        elif self.case == "sugar":
            out["Acc"] = lda.fit(self.Xtr[:, idx], self.y1tr).score(self.Xte[:, idx], self.y1te)
            rf = RandomForestRegressor(n_estimators=self.rf_trees, random_state=self.seed)
            out["R2"] = r2_score(self.yrte, rf.fit(self.Xtr[:, idx], self.yrtr).predict(self.Xte[:, idx]))
        elif self.case == "ip":
            out["AccFine"] = lda.fit(self.Xtr[:, idx], self.y1tr).score(self.Xte[:, idx], self.y1te)
            out["AccCoarse"] = LinearDiscriminantAnalysis().fit(self.Xtr[:, idx], self.y2tr).score(
                self.Xte[:, idx], self.y2te)
        return out

    # -- inner cross-validation on the TRAINING set only (used to choose weights / native solutions)
    def _folds(self):
        if self.case == "grape_loyo":
            return list(LeaveOneGroupOut().split(self.Xtr, self.y1tr, groups=self.auxtr))
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        return list(skf.split(self.Xtr, self.y1tr))

    def cv(self, idx):
        key = tuple(sorted(int(i) for i in idx))
        if key in self._cv_cache:
            return self._cv_cache[key]
        idx = np.asarray(key, int)
        scores = []
        for a, b in self._folds():
            Xa, Xb = self.Xtr[a][:, idx], self.Xtr[b][:, idx]
            acc = LinearDiscriminantAnalysis().fit(Xa, self.y1tr[a]).score(Xb, self.y1tr[b])
            if self.case.startswith("grape"):
                scores.append(acc)
            elif self.case == "sugar":
                rf = RandomForestRegressor(n_estimators=self.cv_rf_trees, random_state=self.seed)
                r2 = r2_score(self.yrtr[b], rf.fit(Xa, self.yrtr[a]).predict(Xb))
                scores.append(0.5 * acc + 0.5 * r2)
            elif self.case == "ip":
                acc2 = LinearDiscriminantAnalysis().fit(Xa, self.y2tr[a]).score(Xb, self.y2tr[b])
                scores.append(0.5 * acc + 0.5 * acc2)
        self._cv_cache[key] = float(np.mean(scores))
        return self._cv_cache[key]


# ----------------------------------------------------------------------------- proxy network
def train_proxy(pops, ctx, seed, adaptive=True, fixed_w=None, epochs=800, temp=5.0,
                init="zero", sampling="legacy", sugar_cls_loss="ce", init_logits=None, device=None):
    """Generic SoftMaskNet (T populations). Hyper-parameters reproduce core/networks.py:
    hidden 64, ReLU, dropout 0.2, Adam lr 1e-3 for heads, separate lr for the logits,
    batch 32, final score = sum_t w_t * mean(pop_t).
    init='zero'  -> logits = 0 (what the paper states);
    init='legacy'-> logits = [1.1, 1.0, 0.9] (what the released code does)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ctx.proxy
    T, d = len(pops), ctx.d
    mode = cfg["legacy_sampling"] if sampling == "legacy" else sampling

    def head(out_dim):
        return nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, out_dim))

    X = torch.tensor(ctx.Xtr, dtype=torch.float32)
    tensors = [X]
    if cfg["kind"] == "binary":
        heads = nn.ModuleList([head(1)])
        tensors.append(torch.tensor(cfg["y"], dtype=torch.float32).unsqueeze(1))
    elif cfg["kind"] == "multiclass":
        heads = nn.ModuleList([head(cfg["n_classes"])])
        tensors.append(torch.tensor(cfg["y"], dtype=torch.long))
    else:  # dual
        n_out = 1 if sugar_cls_loss == "legacy_bce" else cfg["n_classes"]
        heads = nn.ModuleList([head(n_out), head(1)])
        yc = torch.tensor(cfg["y"], dtype=torch.float32 if sugar_cls_loss == "legacy_bce" else torch.long)
        tensors.append(yc.unsqueeze(1) if sugar_cls_loss == "legacy_bce" else yc)
        yr = (cfg["yreg"] - cfg["yreg"].mean()) / (cfg["yreg"].std() + 1e-8)
        tensors.append(torch.tensor(yr, dtype=torch.float32).unsqueeze(1))
    heads = heads.to(device)

    if adaptive:
        if init_logits is not None:
            init_vec = torch.tensor(init_logits, dtype=torch.float32)
        elif init == "legacy":
            base = [1.1, 1.0, 0.9] if T == 3 else list(np.linspace(1.1, 0.9, T))
            init_vec = torch.tensor(base, dtype=torch.float32)
        else:
            init_vec = torch.zeros(T)
        logits = nn.Parameter(init_vec.to(device))
        opt = torch.optim.Adam([{"params": heads.parameters()},
                                {"params": [logits], "lr": cfg["logit_lr"]}], lr=1e-3)
    else:
        w_fixed = torch.tensor(fixed_w, dtype=torch.float32, device=device)
        opt = torch.optim.Adam(heads.parameters(), lr=1e-3)

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(TensorDataset(*tensors), batch_size=32, shuffle=True, generator=g)
    pops_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in pops]

    def weights():
        return F.softmax(logits / temp, dim=0) if adaptive else w_fixed

    w_hist, loss_hist = [], []
    t0 = time.time()
    heads.train()
    for _ in range(epochs):
        ep = 0.0
        for batch in loader:
            batch = [b.to(device) for b in batch]
            bx = batch[0]
            B = bx.shape[0]
            if mode == "per_sample":
                ms = [p[torch.as_tensor(rng.integers(0, len(p), B), device=device)] for p in pops_t]
            else:
                ms = [p[int(rng.integers(0, len(p)))].unsqueeze(0).expand(B, -1) for p in pops_t]
            w = weights()
            mask = sum(w[t] * ms[t] for t in range(T))
            xm = bx * mask
            if cfg["kind"] == "binary":
                loss = nn.BCEWithLogitsLoss()(heads[0](xm), batch[1])
            elif cfg["kind"] == "multiclass":
                loss = nn.CrossEntropyLoss()(heads[0](xm), batch[1])
            else:
                if sugar_cls_loss == "legacy_bce":
                    lc = nn.BCEWithLogitsLoss()(heads[0](xm), batch[1])
                else:
                    lc = nn.CrossEntropyLoss()(heads[0](xm), batch[1])
                loss = lc + nn.MSELoss()(heads[1](xm), batch[2])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep += loss.item()
        loss_hist.append(ep / len(loader))
        with torch.no_grad():
            w_hist.append(weights().cpu().numpy().tolist())
    with torch.no_grad():
        w = weights().cpu().numpy().astype(np.float64)
    score = sum(w[t] * pops[t].mean(0) for t in range(T))   # Eq. (11): theta is NOT used here
    return dict(w=w, score=score, w_hist=w_hist, loss_hist=loss_hist, seconds=time.time() - t0)
