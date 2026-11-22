# supp/step2_extended_ablation.py
"""
STEP 2 - Extended ablation (reviewer comment 4).

For every run, ALL variants share the same split, the same MI estimates and the same elite
populations, so the results are paired run-by-run (needed for step 3).

Variants (pop order = [task1, redundancy, task2]):
  SO-Task1, SO-Task2      : Top-K of a single task's elite centroid           (single-objective)
  MeanAgg                 : equal-weight centroid (== Fixed-Eq, see step 1)   (fixed weights)
  FixedP1 / FixedP2       : [0.5,0.2,0.3] / [0.2,0.5,0.3]                     (fixed weights)
  FixedCV-perK            : weights chosen on a simplex grid by inner CV on the TRAINING set,
                            separately for each K                             (tuned fixed weights)
  FixedCV-global          : one weight vector for all K, chosen by inner CV   (tuned fixed weights)
  JointGA-Eq              : ONE population, fitness = equal-weight sum of z-scored objectives,
                            evaluation budget matched to the T independent GAs (joint vs decoupled)
  JointGA-wstar           : same, but with the weights w* learned by Adaptive in the same run
                            (isolates decoupled search with the preference vector held fixed)
  MeanAgg-noRed           : equal-weight centroid of task1 & task2 only       (no redundancy objective)
  Adaptive-noRed          : SoftMaskNet on task1 & task2 populations only     (no redundancy objective)
  Adaptive                : full method

Outputs in <out>/<case>/:
  raw_results.csv   case, run, method, K, metric, value       (test-set scores, one row per item)
  cv_scores.csv     case, run, method, K, cv                   (inner-CV score of every selected subset)
  grid_cv.csv       case, run, w1..wT, K, cv                   (inner-CV score of every grid weight)
  selections.csv    case, run, method, K, indices              (selected band indices)
  timing.csv        seconds spent by the GA / Adaptive / Fixed-CV / Joint-GA
  artifacts/run_XX.npz, histories/run_XX.json

Usage (from repo root):
    python supp/step2_extended_ablation.py --case sugar --n_runs 30
    python supp/step2_extended_ablation.py --case grape_loyo --grape_path datasets/Spectral_DataSet.xlsx
Runs that already have an artifact file are skipped, so the script can be resumed.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supp.common import (load_case_data, Ctx, run_ga, keep_elite, train_proxy, topk, simplex_grid,  # noqa: E402
                         zscore_objectives, make_joint_fitness, set_seed, ensure_dir, append_csv)

FIXED = {"FixedP1": [0.5, 0.2, 0.3], "FixedP2": [0.2, 0.5, 0.3]}


def fixed_cv(ctx, cents, Kmax, step):
    """Grid search of fixed fusion weights using ONLY the training set (inner CV)."""
    t0 = time.time()
    grid = simplex_grid(len(cents), step)
    rows = []
    for w in grid:
        s = sum(wi * c for wi, c in zip(w, cents))
        for K in range(1, Kmax + 1):
            idx = topk(s, K, ctx.tiebreak)
            rows.append(dict(w=w, K=K, cv=ctx.cv(idx), idx=idx))
    per_k, glob = {}, {}
    for K in range(1, Kmax + 1):
        rk = [r for r in rows if r["K"] == K]
        per_k[K] = max(rk, key=lambda r: r["cv"])            # first max in grid order
    mean_cv = {}
    for r in rows:
        mean_cv.setdefault(tuple(r["w"]), []).append(r["cv"])
    w_best = max(mean_cv, key=lambda w: np.mean(mean_cv[w]))
    for r in rows:
        if tuple(r["w"]) == w_best:
            glob[r["K"]] = r
    return per_k, glob, rows, time.time() - t0


def one_run(case, data, run, args, outdir):
    ctx = Ctx(case, data, run, cv_folds=args.cv_folds, cv_rf_trees=args.cv_rf_trees, rf_trees=args.rf_trees)
    set_seed(ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    timing = {}

    # ---- 1) independent task-specific GAs (decoupled search)
    t0 = time.time()
    ga = [run_ga(f, ctx.d, args.N, args.G, args.CR, args.MR, rng, init_p=args.init_p) for f in ctx.objs]
    pops = [keep_elite(r["X"], r["fit"], ratio=args.elite_ratio) for r in ga]
    timing["GA_independent"] = time.time() - t0
    evals_indep = sum(r["n_evals"] for r in ga)
    cents = [p.mean(0) for p in pops]

    scores = {"SO-Task1": cents[0], "SO-Task2": cents[2],
              "MeanAgg": (cents[0] + cents[1] + cents[2]) / 3.0,
              "MeanAgg-noRed": (cents[0] + cents[2]) / 2.0}
    for name, w in FIXED.items():
        scores[name] = sum(wi * c for wi, c in zip(w, cents))   # fixed weights: theta is irrelevant

    # ---- 2) adaptive SoftMaskNet (full) and without the redundancy population
    kw = dict(epochs=args.epochs, temp=args.temp, init=args.init, sampling=args.sampling,
              sugar_cls_loss=args.sugar_cls_loss)
    ad = train_proxy(pops, ctx, ctx.seed, **kw)
    scores["Adaptive"] = ad["score"]
    timing["Adaptive_train"] = ad["seconds"]
    init2 = [1.1, 0.9] if args.init == "legacy" else None
    ad2 = train_proxy([pops[0], pops[2]], ctx, ctx.seed, init_logits=init2, **kw)
    scores["Adaptive-noRed"] = ad2["score"]

    # ---- 3) Joint-GA (single population, budget matched: T*N individuals, same G)
    stats = zscore_objectives(ctx.objs, ctx.d, rng)
    t0 = time.time()
    for name, w in [("JointGA-Eq", [1 / 3] * 3), ("JointGA-wstar", list(ad["w"]))]:
        fj = make_joint_fitness(ctx.objs, stats, w)
        rj = run_ga(fj, ctx.d, len(ctx.objs) * args.N, args.G, args.CR, args.MR, rng, init_p=args.init_p)
        scores[name] = keep_elite(rj["X"], rj["fit"], ratio=args.elite_ratio).mean(0)
        timing[f"{name}_evals"] = rj["n_evals"]
    timing["JointGA"] = time.time() - t0
    timing["Independent_evals"] = evals_indep

    # ---- 4) Fixed-CV (weights tuned on the training set only)
    per_k, glob, grid_rows, timing["FixedCV"] = fixed_cv(ctx, cents, args.Kmax, args.grid_step)

    # ---- 5) evaluate every variant on the hold-out test set
    res, cvs, sels = [], [], []
    for K in range(1, args.Kmax + 1):
        chosen = {m: topk(s, K, ctx.tiebreak) for m, s in scores.items()}
        chosen["FixedCV-perK"] = per_k[K]["idx"]
        chosen["FixedCV-global"] = glob[K]["idx"]
        for m, idx in chosen.items():
            for metric, v in ctx.evaluate(idx).items():
                res.append(dict(case=case, run=run, method=m, K=K, metric=metric, value=v))
            cvs.append(dict(case=case, run=run, method=m, K=K, cv=ctx.cv(idx)))
            sels.append(dict(case=case, run=run, method=m, K=K, indices=" ".join(map(str, idx))))

    wcols = [f"w_{n}" for n in ctx.names]
    grid_df = pd.DataFrame([dict(case=case, run=run, K=r["K"], cv=r["cv"], **dict(zip(wcols, r["w"])))
                            for r in grid_rows])
    chosen_w = pd.DataFrame([dict(case=case, run=run, K=K, scheme=sch, **dict(zip(wcols, d_[K]["w"])))
                             for sch, d_ in [("perK", per_k), ("global", glob)] for K in d_])

    append_csv(pd.DataFrame(res), os.path.join(outdir, "raw_results.csv"))
    append_csv(pd.DataFrame(cvs), os.path.join(outdir, "cv_scores.csv"))
    append_csv(pd.DataFrame(sels), os.path.join(outdir, "selections.csv"))
    append_csv(grid_df, os.path.join(outdir, "grid_cv.csv"))
    append_csv(chosen_w, os.path.join(outdir, "fixedcv_chosen_weights.csv"))
    append_csv(pd.DataFrame([dict(case=case, run=run, **timing)]), os.path.join(outdir, "timing.csv"))
    np.savez_compressed(os.path.join(outdir, "artifacts", f"run_{run:02d}.npz"),
                        pop1=pops[0], pop2=pops[1], pop3=pops[2], tr=ctx.tr, te=ctx.te,
                        w_adaptive=ad["w"], w_adaptive_noRed=ad2["w"], tiebreak=ctx.tiebreak,
                        mi1=ctx.mi1, mi_aux=ctx.mi_aux, seed=ctx.seed)
    with open(os.path.join(outdir, "histories", f"run_{run:02d}.json"), "w") as fh:
        json.dump(dict(ga_curves=[r["curve"] for r in ga], w_hist=ad["w_hist"], loss_hist=ad["loss_hist"],
                       w_hist_noRed=ad2["w_hist"]), fh)
    return ad["w"], timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["grape_random", "grape_loyo", "sugar", "ip"])
    ap.add_argument("--n_runs", type=int, default=30)
    ap.add_argument("--start_run", type=int, default=1)
    ap.add_argument("--Kmax", type=int, default=15)
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--G", type=int, default=40, help="GA generations")
    ap.add_argument("--CR", type=float, default=0.8)
    ap.add_argument("--MR", type=float, default=0.01)
    ap.add_argument("--elite_ratio", type=float, default=0.2,
                    help="fraction of the final population kept as elite set S_t (0.2 = 8 of 40, as in the "
                         "released code; 1.0 = whole final population -> finer selection frequencies)")
    ap.add_argument("--init_p", type=float, default=0.5, help="probability of a band being ON at initialisation")
    ap.add_argument("--epochs", type=int, default=800, help="SoftMaskNet epochs (released code: 800)")
    ap.add_argument("--temp", type=float, default=5.0)
    ap.add_argument("--init", choices=["zero", "legacy"], default="zero",
                    help="zero = as stated in the paper; legacy = [1.1,1.0,0.9] as in released code")
    ap.add_argument("--sampling", choices=["legacy", "per_sample", "per_batch"], default="legacy")
    ap.add_argument("--sugar_cls_loss", choices=["ce", "legacy_bce"], default="ce",
                    help="Sugar has 3 varieties: CE is correct; legacy_bce reproduces the released code")
    ap.add_argument("--ip_crop", choices=["paper", "code"], default="paper")
    ap.add_argument("--grid_step", type=float, default=0.1)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--cv_rf_trees", type=int, default=50)
    ap.add_argument("--rf_trees", type=int, default=50)
    ap.add_argument("--grape_path", default=None)
    ap.add_argument("--out", default="results_supp")
    args = ap.parse_args()

    outdir = ensure_dir(os.path.join(args.out, args.case))
    ensure_dir(os.path.join(outdir, "artifacts"))
    ensure_dir(os.path.join(outdir, "histories"))
    with open(os.path.join(outdir, "config_step2.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    data = load_case_data(args.case, args.grape_path, args.ip_crop)
    for run in range(args.start_run, args.start_run + args.n_runs):
        if os.path.exists(os.path.join(outdir, "artifacts", f"run_{run:02d}.npz")):
            print(f"[skip] run {run} already done")
            continue
        t0 = time.time()
        w, timing = one_run(args.case, data, run, args, outdir)
        print(f"[{args.case}] run {run} done in {time.time() - t0:.1f}s | w* = {np.round(w, 3)} | "
              f"Adaptive {timing['Adaptive_train']:.1f}s, Fixed-CV {timing['FixedCV']:.1f}s, "
              f"evals indep/joint = {timing['Independent_evals']}/{timing['JointGA-Eq_evals']}")

    df = pd.read_csv(os.path.join(outdir, "raw_results.csv"))
    tab = df.groupby(["metric", "method", "K"])["value"].agg(["mean", "std"]).reset_index()
    for metric in tab["metric"].unique():
        t = tab[tab.metric == metric].pivot(index="K", columns="method", values="mean")
        print(f"\n=== {args.case} | {metric} (mean over runs) ===")
        print(t.round(4).to_string())


if __name__ == "__main__":
    main()
