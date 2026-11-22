# supp/diag_ties.py
"""
DIAGNOSTIC - Do the fusion weights change the selected bands at all?

Reads the artifacts of step 2 (works while step 2 is still running: it uses the runs finished so far).
For every run it reports
  * mean number of bands per elite mask in each population
  * n_all_ones: bands selected by EVERY elite of ALL three populations (centroids = 1, 1, 1).
    Their fused score is 1 under ANY weights -> for K <= n_all_ones the weights cannot matter.
  * size of the top tie group of each fused score (MeanAgg, FixedP1, FixedP2, Adaptive)
  * fraction of K values where MeanAgg, FixedP1, FixedP2 and Adaptive select the SAME subset
  * the original-code choice (np.argsort(...)[::-1][:K]) for MeanAgg vs Adaptive, run 1
and, if raw_results.csv exists, how often Adaptive and MeanAgg obtain exactly the same test score.

Usage (from repo root):
    python supp/diag_ties.py --case grape_loyo --grape_path datasets/Spectral_DataSet.xlsx
    python supp/diag_ties.py --case ip
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supp.common import topk, BASE_SEED  # noqa: E402

FUSED = {"MeanAgg": [1 / 3, 1 / 3, 1 / 3], "FixedP1": [0.5, 0.2, 0.3], "FixedP2": [0.2, 0.5, 0.3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["grape_random", "grape_loyo", "sugar", "ip"])
    ap.add_argument("--out", default="results_supp")
    ap.add_argument("--Kmax", type=int, default=15)
    ap.add_argument("--grape_path", default=None, help="only used to print wavelength labels")
    args = ap.parse_args()

    outdir = os.path.join(args.out, args.case)
    files = sorted(glob.glob(os.path.join(outdir, "artifacts", "run_*.npz")))
    if not files:
        sys.exit(f"No artifacts in {outdir}/artifacts yet - wait until step 2 finishes at least one run.")

    labels = None
    if args.case.startswith("grape") and args.grape_path and os.path.exists(args.grape_path):
        labels = [str(c) for c in pd.read_excel(args.grape_path, nrows=0).columns[:600]]

    rows, first = [], None
    for f in files:
        a = np.load(f)
        run = int(a["seed"]) - BASE_SEED
        pops = [a["pop1"], a["pop2"], a["pop3"]]
        c = [p.mean(0) for p in pops]
        tb = a["tiebreak"]
        scores = {m: sum(wi * ci for wi, ci in zip(w, c)) for m, w in FUSED.items()}
        scores["Adaptive"] = sum(wi * ci for wi, ci in zip(a["w_adaptive"], c))
        all_ones = np.flatnonzero((c[0] == 1) & (c[1] == 1) & (c[2] == 1))
        r = dict(run=run, mask_size_pop1=pops[0].sum(1).mean(), mask_size_pop2=pops[1].sum(1).mean(),
                 mask_size_pop3=pops[2].sum(1).mean(), n_all_ones=len(all_ones),
                 w_adaptive=np.round(a["w_adaptive"], 3).tolist())
        for m, s in scores.items():
            v = np.round(s, 9)
            r[f"top_group_{m}"] = int((v == v.max()).sum())
        same = []
        for K in range(1, args.Kmax + 1):
            sets = [tuple(topk(scores[m], K, tb)) for m in ["MeanAgg", "FixedP1", "FixedP2", "Adaptive"]]
            same.append(len(set(sets)) == 1)
        r["frac_K_identical_all_fused"] = float(np.mean(same))
        r["frac_K<=5_identical_all_fused"] = float(np.mean(same[:5]))
        rows.append(r)
        if first is None:
            first = (run, all_ones, scores)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"=== {args.case}: {len(df)} finished run(s) ===")
    print(df.drop(columns=["w_adaptive"]).round(2).to_string(index=False))
    print("\nMean over runs:")
    print(df.drop(columns=["run", "w_adaptive"]).mean().round(2).to_string())

    run, all_ones, scores = first
    print(f"\nRun {run}: bands with centroid = 1 in all three populations ({len(all_ones)}):")
    print("  indices:", all_ones.tolist())
    if labels is not None:
        print("  column labels (wavelengths):", [labels[i] for i in all_ones])
    for m in ["MeanAgg", "Adaptive"]:
        print(f"  original-code Top-5 for {m:8s}: {np.argsort(scores[m])[::-1][:5].tolist()}")

    raw_path = os.path.join(outdir, "raw_results.csv")
    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path)
        p = raw[raw.method.isin(["MeanAgg", "FixedP1", "FixedP2", "Adaptive"])].pivot_table(
            index=["run", "K", "metric"], columns="method", values="value")
        eq = (p["Adaptive"] == p["MeanAgg"]) & (p["FixedP1"] == p["MeanAgg"]) & (p["FixedP2"] == p["MeanAgg"])
        k = p.reset_index()["K"].values
        print(f"\nTest scores identical for MeanAgg = FixedP1 = FixedP2 = Adaptive: "
              f"{eq.mean():.1%} of (run, K, metric); K<=5: {eq[k <= 5].mean():.1%}; K>=6: {eq[k >= 6].mean():.1%}")


if __name__ == "__main__":
    main()
