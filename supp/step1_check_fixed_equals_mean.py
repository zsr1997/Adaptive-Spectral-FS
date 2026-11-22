# supp/step1_check_fixed_equals_mean.py
"""
STEP 1 - Sanity check with the ORIGINAL released code (core/ga_utils.py, core/networks.py).

What it proves (needed for reviewer comments 1 and 4):
  (a) With fixed weights, the SoftMaskNet output is sum_t w_t * mean(pop_t); the trained
      network theta is never used -> Fixed-Eq and Mean-Agg give the same scores.
  (b) Training the network twice with different seeds gives the SAME output (theta irrelevant).
  (c) The differences between the "Mean-Agg" and "Fixed Eq" columns in Tables 5/8/11 are
      produced only by how np.argsort breaks ties (float64 vs float32); with a stable,
      deterministic tie-break they vanish.

Usage (from repo root):
    python supp/step1_check_fixed_equals_mean.py --case ip --n_runs 3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supp.common import load_case_data, Ctx, topk, set_seed, ensure_dir  # noqa: E402
from core.ga_utils import jfs, keep_elite                                  # original code
from core import networks                                                    # original code


def original_pops(ctx, N, T):
    """Run the ORIGINAL jfs() with objectives identical to the released scripts."""
    opts = {"xt_in": ctx.Xtr, "N": N, "T": T}
    pops = []
    for f in ctx.objs:   # vectorised objectives are numerically identical to fitness_fast.py
        res = jfs(opts, lambda x, o, f=f: float(f(x[None, :])[0]))
        pops.append(keep_elite(**res))
    return pops


def original_fixed_eq(ctx, pops, torch_seed):
    import torch
    torch.manual_seed(torch_seed)
    p1, p2, p3 = pops
    kind = ctx.proxy["kind"]
    if kind == "binary":
        out = networks.train_softmask_dnn_proxy(p1, p2, p3, ctx.Xtr, ctx.y1tr, ctx.d, False, [1 / 3] * 3)
    elif kind == "multiclass":
        out = networks.train_softmask_dnn_proxy_multi(p1, p2, p3, ctx.Xtr, ctx.y1tr, ctx.d,
                                                      ctx.proxy["n_classes"], False, [1 / 3] * 3)
    else:
        out = networks.train_softmask_dnn_dual_proxy(p1, p2, p3, ctx.Xtr, ctx.y1tr, ctx.yrtr, ctx.d,
                                                     False, [1 / 3] * 3)
    return np.asarray(out[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["grape_random", "grape_loyo", "sugar", "ip"])
    ap.add_argument("--n_runs", type=int, default=3)
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--T", type=int, default=40, help="GA generations (named T in the released code)")
    ap.add_argument("--Kmax", type=int, default=15)
    ap.add_argument("--grape_path", default=None)
    ap.add_argument("--out", default="results_supp")
    args = ap.parse_args()

    data = load_case_data(args.case, args.grape_path)
    lines = []
    n_diff_orig = n_diff_stable = n_total = n_cut_tie = 0
    tie_stats = {5: [], 10: [], 15: []}
    for run in range(1, args.n_runs + 1):
        ctx = Ctx(args.case, data, run)
        set_seed(ctx.seed)
        pops = original_pops(ctx, args.N, args.T)
        sc_mean = (pops[0].mean(0) + pops[1].mean(0) + pops[2].mean(0)) / 3.0   # as in main_exp*.py
        sc_eq_a = original_fixed_eq(ctx, pops, torch_seed=1)
        sc_eq_b = original_fixed_eq(ctx, pops, torch_seed=999)                  # different network
        levels = [len(np.unique(p.mean(0))) for p in pops]
        n_ties = len(sc_mean) - len(np.unique(np.round(sc_mean, 9)))
        lines.append(f"[run {run}] elites per pop = {[len(p) for p in pops]}, distinct centroid levels = {levels}, "
                     f"tied scores in Mean-Agg = {n_ties}/{ctx.d}")
        lines.append(f"          max|FixedEq - MeanAgg| = {np.max(np.abs(sc_eq_a - sc_mean)):.2e}   "
                     f"max|FixedEq(seed1) - FixedEq(seed999)| = {np.max(np.abs(sc_eq_a - sc_eq_b)):.2e}")
        for k in range(1, args.Kmax + 1):
            a = set(np.argsort(sc_mean)[::-1][:k])        # original get_top_k
            b = set(np.argsort(sc_eq_a)[::-1][:k])
            # the network output is float32 (noise ~1e-8), so compare at 1e-6 resolution
            c = set(topk(np.round(sc_mean, 6), k, ctx.tiebreak))
            e = set(topk(np.round(sc_eq_a, 6), k, ctx.tiebreak))
            srt = np.sort(np.round(sc_mean, 9))[::-1]
            n_total += 1
            n_diff_orig += a != b
            n_diff_stable += c != e
            n_cut_tie += bool(srt[k - 1] == srt[k])   # K-th and (K+1)-th scores tied
            if k in tie_stats:
                v = np.round(sc_mean, 9)
                cut = srt[k - 1]
                n_above = int((v > cut).sum())
                tie_stats[k].append((k - n_above, int((v == cut).sum())))
    lines.append("")
    lines.append(f"Top-K sets differing between Mean-Agg and Fixed-Eq, ORIGINAL argsort : "
                 f"{n_diff_orig}/{n_total} ({100 * n_diff_orig / n_total:.1f}%)")
    lines.append(f"Top-K sets differing between Mean-Agg and Fixed-Eq, STABLE tie-break: "
                 f"{n_diff_stable}/{n_total} ({100 * n_diff_stable / n_total:.1f}%)")
    lines.append(f"(run, K) cases where the K cut-off falls INSIDE a group of tied scores: "
                 f"{n_cut_tie}/{n_total} ({100 * n_cut_tie / n_total:.1f}%)  -> selection among ties "
                 f"was arbitrary in the original code for ALL methods")
    for k, st in tie_stats.items():
        if st:
            lines.append(f"K={k:2d}: on average {np.mean([a for a, _ in st]):.1f} of the {k} selected bands are decided "
                         f"by the tie-break, drawn from a tied group of {np.mean([b for _, b in st]):.1f} bands")
    txt = "\n".join(lines)
    print(txt)
    out = os.path.join(ensure_dir(args.out), f"step1_{args.case}.txt")
    with open(out, "w") as fh:
        fh.write(txt + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
