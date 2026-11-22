# supp/step5_moea_native.py
"""
STEP 5 - Does the frequency-ranking + Top-K post-processing treat the MOEAs unfairly?
(reviewer comment 3)

Every (run, algorithm) first does the UNCONSTRAINED run of the paper (same three objectives,
mild L0 penalty) and records:
  * the size distribution of the native non-dominated (ND) solutions  -> diagnostic
  * <ALG>|Freq : marginal frequency over the final ND set -> stable Top-K   (paper protocol)

Then, depending on --mode:
  paper  : nothing else (diagnostic only; native solutions of size 1..Kmax normally do not exist)
  fixedK : for each K, a SEPARATE run constrained to exactly K bands (exact-K sampling + repair).
           Its ND set contains only size-K solutions, i.e. the MOEA's own native answer at K.
             <ALG>|NativeCV : ND solution chosen by inner CV on the TRAINING set   (fair)
             <ALG>|Oracle   : best TEST score among the ND solutions           (upper bound, uses test)
  card   : one run with |c|/d as a 4th objective + sparse initialisation + external archive
           (standard MOFS formulation). Native solutions are taken from the archive ND set.
           With mean-based objectives the front tends to collapse to very small subsets; the
           coverage table shows how many K values are actually reachable.
Also saved: Jaccard(Freq subset, NativeCV subset) and the inner-CV score of both.

Usage (from repo root):
    python supp/step5_moea_native.py --case sugar --mode fixedK --n_runs 30
    python supp/step5_moea_native.py --case grape_loyo --mode fixedK --algos NSGA2 NSGA3 SMSEMOA \
           --K_list 1 2 3 4 5 8 10 15 --grape_path datasets/Spectral_DataSet.xlsx
Then (paired with GA-NN from step 2, same splits):
    python supp/step3_stats.py --inputs results_supp/sugar/raw_results.csv \
           results_supp/moea_fixedK/sugar/raw_results.csv --ref Adaptive --tag sugar_vs_moea \
           --exclude_substr Oracle
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

if not hasattr(np, "row_stack"):   # pymoo 0.6.x + NumPy >= 2.x compatibility
    np.row_stack = np.vstack

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supp.common import load_case_data, Ctx, topk, jaccard, set_seed, ensure_dir, append_csv  # noqa: E402

from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

ALGOS = ["NSGA2", "NSGA3", "SPEA2", "RVEA", "MOEAD", "AGEMOEA", "SMSEMOA"]


class FSProblem(Problem):
    """Binary feature-selection problem; archives every evaluated solution."""

    def __init__(self, objs, d, card):
        super().__init__(n_var=d, n_obj=len(objs) + int(card), xl=0, xu=1, vtype=bool)
        self.objs, self.card = objs, card
        self.arch_X, self.arch_F = [], []

    def _evaluate(self, X, out, *args, **kwargs):
        Xi = np.asarray(X).astype(np.int8)
        F = [f(Xi) for f in self.objs]
        if self.card:
            F.append(Xi.sum(1) / self.n_var)
        F = np.column_stack(F)
        out["F"] = F
        self.arch_X.append(Xi.astype(bool))
        self.arch_F.append(F)


class SparseBinarySampling(Sampling):
    def __init__(self, kmax):
        super().__init__()
        self.kmax = kmax

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=bool)
        for i in range(n_samples):
            s = np.random.randint(1, 2 * self.kmax + 1)
            X[i, np.random.choice(problem.n_var, s, replace=False)] = True
        return X


class ExactKSampling(Sampling):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=bool)
        for i in range(n_samples):
            X[i, np.random.choice(problem.n_var, self.k, replace=False)] = True
        return X


class ExactKRepair(Repair):
    """After crossover/mutation, randomly switch bands on/off until exactly K are selected."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X).astype(bool).copy()
        for i in range(len(X)):
            on = np.flatnonzero(X[i])
            if len(on) > self.k:
                X[i, np.random.choice(on, len(on) - self.k, replace=False)] = False
            elif len(on) < self.k:
                off = np.flatnonzero(~X[i])
                X[i, np.random.choice(off, self.k - len(on), replace=False)] = True
        return X


def make_algo(name, n_obj, N, CR, MR, sampling, repair=None):
    """Imports are lazy: AGEMOEA needs `pip install numba`."""
    # Das-Dennis directions with the count closest to N; every algorithm uses that population size
    best = None
    for p in range(1, 30):
        rd = get_reference_directions("das-dennis", n_obj, n_partitions=p)
        if best is None or abs(len(rd) - N) < abs(len(best) - N):
            best = rd
        if len(rd) > N:
            break
    ref, n_pop = best, len(best)
    ops = dict(sampling=sampling, crossover=SinglePointCrossover(prob=CR),
               mutation=BitflipMutation(prob=1.0, prob_var=MR))
    if repair is not None:
        ops["repair"] = repair
    if name == "NSGA2":
        from pymoo.algorithms.moo.nsga2 import NSGA2
        return NSGA2(pop_size=n_pop, eliminate_duplicates=True, **ops), n_pop
    if name == "NSGA3":
        from pymoo.algorithms.moo.nsga3 import NSGA3
        return NSGA3(ref_dirs=ref, pop_size=n_pop, eliminate_duplicates=True, **ops), n_pop
    if name == "SPEA2":
        from pymoo.algorithms.moo.spea2 import SPEA2
        return SPEA2(pop_size=n_pop, eliminate_duplicates=True, **ops), n_pop
    if name == "RVEA":
        from pymoo.algorithms.moo.rvea import RVEA
        return RVEA(ref_dirs=ref, pop_size=n_pop, eliminate_duplicates=True, **ops), n_pop
    if name == "MOEAD":
        from pymoo.algorithms.moo.moead import MOEAD
        return MOEAD(ref_dirs=ref, n_neighbors=min(15, n_pop - 1), prob_neighbor_mating=0.7, **ops), n_pop
    if name == "AGEMOEA":
        from pymoo.algorithms.moo.age import AGEMOEA
        return AGEMOEA(pop_size=n_pop, eliminate_duplicates=True, **ops), n_pop
    if name == "SMSEMOA":
        from pymoo.algorithms.moo.sms import SMSEMOA
        return SMSEMOA(pop_size=n_pop, eliminate_duplicates=True, **ops), n_pop
    raise ValueError(name)


def nondominated(X, F):
    X, keep = np.unique(X, axis=0, return_index=True)
    F = F[keep]
    ok = F.max(1) < 1e5                       # drop empty subsets (objective = 1e6)
    X, F = X[ok], F[ok]
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return X[nd], F[nd]


def run_moea(ctx, alg, objs, args, card=False, k_exact=None):
    set_seed(ctx.seed if k_exact is None else ctx.seed * 100 + k_exact)
    prob = FSProblem(objs, ctx.d, card=card)
    if k_exact is not None:
        sampling, repair = ExactKSampling(k_exact), ExactKRepair(k_exact)
    elif card:
        sampling, repair = SparseBinarySampling(max(args.K_list)), None
    else:
        sampling, repair = BinaryRandomSampling(), None
    algo, n_pop = make_algo(alg, prob.n_obj, args.N, args.CR, args.MR, sampling, repair)
    t0 = time.time()
    seed = ctx.seed if k_exact is None else ctx.seed * 100 + k_exact
    res = minimize(prob, algo, ("n_gen", args.G), seed=seed, verbose=False)
    Xf, Ff = nondominated(np.asarray(res.pop.get("X")).astype(bool), np.asarray(res.pop.get("F")))
    Xa, _ = nondominated(np.vstack(prob.arch_X), np.vstack(prob.arch_F))
    return dict(Xf=Xf, Xa=Xa, n_pop=n_pop, seconds=time.time() - t0)


def add_native(rows, info, ctx, alg, K, cands, idx_f):
    """Choose one native solution by inner CV (training data only) + record the oracle."""
    info["n_native_candidates"] = len(cands)
    if not cands:
        return
    cvs = [ctx.cv(c) for c in cands]
    best = cands[int(np.argmax(cvs))]
    ev_best = ctx.evaluate(best)
    ev_all = [ctx.evaluate(c) for c in cands]
    for metric in ctx.metrics:
        rows.append(dict(case=ctx.case, run=ctx.run, method=f"{alg}|NativeCV", K=K, metric=metric,
                         value=ev_best[metric]))
        rows.append(dict(case=ctx.case, run=ctx.run, method=f"{alg}|Oracle", K=K, metric=metric,
                         value=max(e[metric] for e in ev_all)))
    info["jaccard_freq_vs_nativecv"] = jaccard(idx_f, best)
    info["cv_freq"], info["cv_nativecv"] = ctx.cv(idx_f), max(cvs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["grape_random", "grape_loyo", "sugar", "ip"])
    ap.add_argument("--mode", choices=["paper", "fixedK", "card"], default="fixedK")
    ap.add_argument("--algos", nargs="+", default=ALGOS)
    ap.add_argument("--n_runs", type=int, default=30)
    ap.add_argument("--start_run", type=int, default=1)
    ap.add_argument("--K_list", type=int, nargs="+", default=list(range(1, 16)))
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--G", type=int, default=40)
    ap.add_argument("--CR", type=float, default=0.8)
    ap.add_argument("--MR", type=float, default=0.01)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--cv_rf_trees", type=int, default=50)
    ap.add_argument("--rf_trees", type=int, default=50)
    ap.add_argument("--ip_crop", choices=["paper", "code"], default="paper")
    ap.add_argument("--grape_path", default=None)
    ap.add_argument("--out", default="results_supp")
    args = ap.parse_args()

    outdir = ensure_dir(os.path.join(args.out, f"moea_{args.mode}", args.case))
    with open(os.path.join(outdir, "config_step5.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    data = load_case_data(args.case, args.grape_path, args.ip_crop)
    done_path = os.path.join(outdir, "done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()

    for run in range(args.start_run, args.start_run + args.n_runs):
        ctx = Ctx(args.case, data, run, cv_folds=args.cv_folds, cv_rf_trees=args.cv_rf_trees,
                  rf_trees=args.rf_trees)
        for alg in args.algos:
            tag = f"{alg}:{run}"
            if tag in done:
                continue
            # ---- unconstrained run of the paper: Freq protocol + size diagnostics
            base = run_moea(ctx, alg, ctx.objs, args)
            sizes = base["Xf"].sum(1)
            freq = base["Xf"].mean(0)
            rows, infos = [], []
            card = run_moea(ctx, alg, ctx.objs, args, card=True) if args.mode == "card" else None
            secs_native = 0.0
            for K in args.K_list:
                idx_f = topk(freq, K, ctx.tiebreak)
                for metric, v in ctx.evaluate(idx_f).items():
                    rows.append(dict(case=args.case, run=run, method=f"{alg}|Freq", K=K, metric=metric, value=v))
                info = dict(case=args.case, run=run, algo=alg, K=K)
                if args.mode == "paper":
                    cands = [np.flatnonzero(x) for x in base["Xf"][sizes == K]]
                elif args.mode == "card":
                    Xa = card["Xa"]
                    cands = [np.flatnonzero(x) for x in Xa[Xa.sum(1) == K]]
                else:
                    rk = run_moea(ctx, alg, ctx.objs, args, k_exact=K)
                    secs_native += rk["seconds"]
                    cands = [np.flatnonzero(x) for x in rk["Xf"]]
                add_native(rows, info, ctx, alg, K, cands, idx_f)
                infos.append(info)
            append_csv(pd.DataFrame(rows), os.path.join(outdir, "raw_results.csv"))
            append_csv(pd.DataFrame(infos), os.path.join(outdir, "native_info.csv"))
            append_csv(pd.DataFrame([dict(
                case=args.case, run=run, algo=alg, pop_size=base["n_pop"], seconds_unconstrained=base["seconds"],
                seconds_native_total=secs_native, n_final_nd=len(sizes), size_min=int(sizes.min()),
                size_median=float(np.median(sizes)), size_max=int(sizes.max()),
                frac_final_in_K_range=float(np.mean((sizes >= 1) & (sizes <= max(args.K_list)))))]),
                os.path.join(outdir, "size_diagnostics.csv"))
            with open(done_path, "a") as fh:
                fh.write(tag + "\n")
            n_cov = sum(1 for i in infos if i.get("n_native_candidates", 0) > 0)
            print(f"[{args.mode}|{args.case}] run {run} {alg}: native ND sizes of the paper run "
                  f"{sizes.min()}-{sizes.max()} (median {np.median(sizes):.0f}); "
                  f"K values with a native solution: {n_cov}/{len(args.K_list)}")

    # ---------------- summary
    raw = pd.read_csv(os.path.join(outdir, "raw_results.csv"))
    info = pd.read_csv(os.path.join(outdir, "native_info.csv"))
    sz = pd.read_csv(os.path.join(outdir, "size_diagnostics.csv"))
    pd.set_option("display.width", 250)
    print("\n=== Native ND solutions of the paper's (unconstrained) MOEA runs: subset sizes ===")
    print(sz.groupby("algo")[["size_min", "size_median", "size_max", "frac_final_in_K_range"]]
          .mean().round(2).to_string())
    print("\n=== Fraction of (run, K) with at least one native solution of size K ===")
    print(info.assign(has=info.n_native_candidates > 0).pivot_table(index="K", columns="algo", values="has")
          .round(2).to_string())
    for metric in raw.metric.unique():
        t = raw[raw.metric == metric].groupby(["K", "method"])["value"].mean().unstack()
        for alg in sorted({m.split("|")[0] for m in t.columns}):
            cols = [c for c in t.columns if c.startswith(alg + "|")]
            print(f"\n=== {args.case} | {metric} | {alg}: Freq vs NativeCV vs Oracle (mean over runs) ===")
            print(t[cols].round(4).to_string())
    if "jaccard_freq_vs_nativecv" in info:
        print("\n=== Jaccard(Freq Top-K, NativeCV solution), mean over runs ===")
        print(info.pivot_table(index="K", columns="algo", values="jaccard_freq_vs_nativecv").round(2).to_string())


if __name__ == "__main__":
    main()
