# supp/step4_mechanism.py
"""
STEP 4 - Mechanism analyses that EXPLAIN the ablation results (reviewer comment 4).
Reads the outputs of step 2; does not retrain anything unless --probe_runs > 0.

(a) Lexicographic check
    With centroids taking values in multiples of 1/n_elite, if w1/n_elite > w2 + w3 the
    adaptive ranking is exactly "task-1 centroid first; ties broken by w2*c2 + w3*c3".
    We report how often this holds and the Jaccard overlap between the adaptive Top-K and
    the explicit lexicographic Top-K.
(b) Properties of the selected subsets (training data)
    mean |corr| among the K selected bands (redundancy), mean relevance to the primary target,
    mean dependence on the auxiliary variable (year MI / coarse MI / |r| with sugar).
    + Spearman correlation, across (run, K), between the redundancy gap and the performance gap
    of FixedP2 vs Adaptive -> tests "redundancy only pays off after Top-K truncation".
(c) Regret of the learned weights w*
    For each run and K: best inner-CV score over the simplex grid minus the inner-CV score of
    the Adaptive subset, the percentile of Adaptive within the grid, and the CV-optimal weights.
    -> tests "the proxy-loss optimum is not the Top-K optimum".
(d) Optional proxy-loss probe (--probe_runs 3): retrain the proxy with FIXED weights on a few
    grid points, record the final training loss, and correlate it with the Top-K inner-CV score.

Usage (from repo root):
    python supp/step4_mechanism.py --case grape_loyo --grape_path datasets/Spectral_DataSet.xlsx
    python supp/step4_mechanism.py --case sugar --probe_runs 3 --probe_epochs 300
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supp.common import load_case_data, Ctx, topk, jaccard, train_proxy, simplex_grid, BASE_SEED  # noqa: E402


def lexicographic_check(case, art_dir, Kmax):
    rows = []
    for f in sorted(glob.glob(os.path.join(art_dir, "run_*.npz"))):
        a = np.load(f)
        pops = [a["pop1"], a["pop2"], a["pop3"]]
        w = a["w_adaptive"]
        c = [p.mean(0) for p in pops]
        n_el = len(pops[0])
        cond = bool(w[0] / n_el > w[1] + w[2])
        s_ad = sum(wi * ci for wi, ci in zip(w, c))
        secondary = w[1] * c[1] + w[2] * c[2]
        tb = a["tiebreak"]
        lex_order = np.lexsort((np.arange(len(tb)), -tb, -np.round(secondary, 9), -np.round(c[0], 9)))
        for K in range(1, Kmax + 1):
            rows.append(dict(run=int(a["seed"]) - BASE_SEED, K=K, w1=w[0], w2=w[1], w3=w[2], strict_lexicographic=cond,
                             jaccard_adaptive_vs_lex=jaccard(topk(s_ad, K, tb), lex_order[:K]),
                             n_bands_task1_centroid_pos=int((c[0] > 0).sum())))
    df = pd.DataFrame(rows)
    summ = df.groupby("K").agg(frac_strict_lex=("strict_lexicographic", "mean"),
                               jaccard_mean=("jaccard_adaptive_vs_lex", "mean"),
                               bands_with_c1_pos=("n_bands_task1_centroid_pos", "mean")).reset_index()
    return df, summ


def subset_properties(case, data, outdir, args):
    sel = pd.read_csv(os.path.join(outdir, "selections.csv"))
    rows = []
    for run, g in sel.groupby("run"):
        ctx = Ctx(case, data, run, cv_folds=args.cv_folds, cv_rf_trees=args.cv_rf_trees)
        C = ctx.C
        for _, r in g.iterrows():
            idx = np.array(list(map(int, str(r["indices"]).split())))
            red = C[np.ix_(idx, idx)].sum() / max(1, len(idx) * (len(idx) - 1)) if len(idx) > 1 else 0.0
            rows.append(dict(run=run, method=r["method"], K=r["K"], redundancy=red,
                             relevance=ctx.mi1[idx].mean(), aux_dependence=ctx.mi_aux[idx].mean()))
    props = pd.DataFrame(rows)
    summ = props.groupby(["method", "K"])[["redundancy", "relevance", "aux_dependence"]].mean().reset_index()

    # does lower redundancy of FixedP2 explain its performance advantage over Adaptive?
    raw = pd.read_csv(os.path.join(outdir, "raw_results.csv"))
    links = []
    for metric in raw.metric.unique():
        perf = raw[raw.metric == metric].pivot_table(index=["run", "K"], columns="method", values="value")
        pr = props.pivot_table(index=["run", "K"], columns="method", values="redundancy")
        j = pd.DataFrame({"d_perf": perf["FixedP2"] - perf["Adaptive"],
                          "d_red": pr["FixedP2"] - pr["Adaptive"]}).dropna()
        j = j[(j.d_red != 0) | (j.d_perf != 0)]
        if len(j) > 5:
            rho, p = stats.spearmanr(j.d_red, j.d_perf)
            links.append(dict(metric=metric, n_pairs=len(j), spearman_rho=rho, p=p,
                              mean_d_red=j.d_red.mean(), mean_d_perf=j.d_perf.mean()))
    return props, summ, pd.DataFrame(links)


def regret_of_wstar(outdir):
    grid = pd.read_csv(os.path.join(outdir, "grid_cv.csv"))
    cvs = pd.read_csv(os.path.join(outdir, "cv_scores.csv"))
    wcols = [c for c in grid.columns if c.startswith("w_")]
    ad = cvs[cvs.method == "Adaptive"].set_index(["run", "K"])["cv"]
    rows = []
    for (run, K), g in grid.groupby(["run", "K"]):
        a = ad.get((run, K), np.nan)
        best = g.loc[g.cv.idxmax()]
        rows.append(dict(run=run, K=K, cv_adaptive=a, cv_grid_best=best.cv, regret=best.cv - a,
                         percentile_adaptive=100 * (g.cv <= a).mean(),
                         **{f"opt_{c}": best[c] for c in wcols}))
    df = pd.DataFrame(rows)
    summ = df.groupby("K")[["regret", "percentile_adaptive"] + [f"opt_{c}" for c in wcols]].mean().reset_index()
    return df, summ


def proxy_loss_probe(case, data, art_dir, args):
    """Train the proxy with FIXED weights and compare its final loss with the Top-K CV score."""
    rows = []
    grid = simplex_grid(3, 0.5) + [np.array([1 / 3] * 3)]       # vertices, edge mid-points, centre
    for f in sorted(glob.glob(os.path.join(art_dir, "run_*.npz")))[:args.probe_runs]:
        a = np.load(f)
        run = int(a["seed"]) - BASE_SEED
        ctx = Ctx(case, data, run, cv_folds=args.cv_folds, cv_rf_trees=args.cv_rf_trees)
        pops = [a["pop1"], a["pop2"], a["pop3"]]
        cents = [p.mean(0) for p in pops]
        for w in grid + [a["w_adaptive"]]:
            r = train_proxy(pops, ctx, ctx.seed, adaptive=False, fixed_w=list(w), epochs=args.probe_epochs,
                            sugar_cls_loss=args.sugar_cls_loss)
            final_loss = float(np.mean(r["loss_hist"][-max(1, args.probe_epochs // 10):]))
            s = sum(wi * c for wi, c in zip(w, cents))
            for K in args.probe_K:
                rows.append(dict(run=run, w=str(np.round(w, 3)), K=K, final_train_loss=final_loss,
                                 topk_cv=ctx.cv(topk(s, K, ctx.tiebreak))))
    df = pd.DataFrame(rows)
    corr = []
    for K, g in df.groupby("K"):
        rho, p = stats.spearmanr(-g.final_train_loss, g.topk_cv)
        corr.append(dict(K=K, spearman_lowloss_vs_cv=rho, p=p, n=len(g)))
    return df, pd.DataFrame(corr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["grape_random", "grape_loyo", "sugar", "ip"])
    ap.add_argument("--out", default="results_supp")
    ap.add_argument("--Kmax", type=int, default=15)
    ap.add_argument("--grape_path", default=None)
    ap.add_argument("--ip_crop", choices=["paper", "code"], default="paper")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--cv_rf_trees", type=int, default=50)
    ap.add_argument("--probe_runs", type=int, default=0)
    ap.add_argument("--probe_epochs", type=int, default=300)
    ap.add_argument("--probe_K", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--sugar_cls_loss", choices=["ce", "legacy_bce"], default="ce")
    args = ap.parse_args()

    outdir = os.path.join(args.out, args.case)
    art_dir = os.path.join(outdir, "artifacts")
    data = load_case_data(args.case, args.grape_path, args.ip_crop)

    lex, lex_s = lexicographic_check(args.case, art_dir, args.Kmax)
    props, props_s, links = subset_properties(args.case, data, outdir, args)
    reg, reg_s = regret_of_wstar(outdir)
    sheets = {"lexicographic": lex_s, "lexicographic_raw": lex, "subset_props": props_s,
              "FixedP2_vs_Adaptive_link": links, "regret_wstar": reg_s, "regret_raw": reg}
    if args.probe_runs > 0:
        probe, probe_c = proxy_loss_probe(args.case, data, art_dir, args)
        sheets.update({"proxy_probe": probe, "proxy_probe_corr": probe_c})

    path = os.path.join(outdir, "step4_mechanism.xlsx")
    with pd.ExcelWriter(path) as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name[:31], index=False)

    pd.set_option("display.width", 200)
    print("\n(a) Lexicographic check (mean over runs):")
    print(lex_s.round(3).to_string(index=False))
    print("\n(b) Subset properties, K in {3,5,10,15}:")
    print(props_s[props_s.K.isin([3, 5, 10, 15])].pivot(index="method", columns="K",
                                                           values="redundancy").round(3).to_string())
    print("\n    FixedP2 - Adaptive: redundancy gap vs performance gap")
    print(links.round(3).to_string(index=False))
    print("\n(c) Regret of w* (grid-best inner-CV minus Adaptive inner-CV) and CV-optimal weights:")
    print(reg_s.round(3).to_string(index=False))
    if args.probe_runs > 0:
        print("\n(d) Proxy-loss probe: Spearman(low training loss, Top-K CV score)")
        print(probe_c.round(3).to_string(index=False))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
