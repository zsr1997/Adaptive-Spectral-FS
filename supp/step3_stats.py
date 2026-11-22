# supp/step3_stats.py
"""
STEP 3 - Statistics for the ablation (step 2) and the MOEA comparison (step 5).

  * Paired Wilcoxon signed-rank test, reference method vs every other method, for every
    (case, metric, K); runs are paired because all methods share the split and the populations.
  * Holm correction across the K values of each (case, metric, method) family.
  * Effect size: matched-pairs rank-biserial correlation r (positive = reference better).
  * Win / Tie / Loss counts (significant after Holm), overall and by regime (K<=5, K>=6).
  * Average ranks per setting and overall, Friedman test, Nemenyi critical difference,
    CD diagram (PNG).

Usage (from repo root):
    python supp/step3_stats.py --inputs results_supp/*/raw_results.csv --ref Adaptive
    python supp/step3_stats.py --inputs results_supp/*/raw_results.csv results_supp/moea_*/raw_results.csv \
        --ref Adaptive --tag vs_moea --exclude_substr Oracle
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for i, j in enumerate(order):
        running = max(running, (m - i) * p[j])
        adj[j] = min(1.0, running)
    return adj


def rank_biserial(diff):
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paired_tests(df, ref, alpha):
    out = []
    for (case, metric), g in df.groupby(["case", "metric"]):
        for m in sorted(set(g.method) - {ref}):
            rows = []
            for K, gk in g.groupby("K"):
                a = gk[gk.method == ref].set_index("run")["value"]
                b = gk[gk.method == m].set_index("run")["value"]
                j = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
                if len(j) < 5:
                    continue
                diff = (j.a - j.b).values
                if np.allclose(diff, 0):
                    p = 1.0
                else:
                    p = stats.wilcoxon(j.a, j.b, zero_method="zsplit").pvalue
                rows.append(dict(case=case, metric=metric, method=m, K=K, n=len(j),
                                 mean_ref=j.a.mean(), mean_other=j.b.mean(), diff=diff.mean(),
                                 p=p, r_rb=rank_biserial(diff)))
            if not rows:
                continue
            p_adj = holm([r["p"] for r in rows])
            for r, pa in zip(rows, p_adj):
                r["p_holm"] = pa
                r["result"] = ("win" if r["diff"] > 0 else "loss") if pa < alpha else "tie"
                out.append(r)
    return pd.DataFrame(out)


def wtl_table(t):
    t = t.copy()
    t["regime"] = np.where(t.K <= 5, "K<=5", "K>=6")
    parts = []
    for by in [["method"], ["method", "regime"], ["case", "metric", "method"]]:
        c = t.groupby(by)["result"].value_counts().unstack(fill_value=0)
        for col in ["win", "tie", "loss"]:
            if col not in c:
                c[col] = 0
        c = c[["win", "tie", "loss"]].reset_index()
        c.insert(0, "grouping", "+".join(by))
        parts.append(c)
    return pd.concat(parts, ignore_index=True)


def nemenyi_cd(k, n_blocks, alpha=0.05):
    try:
        q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    except Exception:  # very old SciPy: Demsar (2006) table, alpha = 0.05
        table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
                 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}
        q = table[min(k, 15)]
    return q * np.sqrt(k * (k + 1) / (6.0 * n_blocks))


def rank_analysis(df):
    means = df.groupby(["case", "metric", "K", "method"])["value"].mean().unstack("method")
    means = means.loc[:, means.isna().mean() <= 0.5]   # drop methods missing in most blocks
    means = means.dropna(axis=0, how="any")             # keep blocks where every method exists
    ranks = means.rank(axis=1, ascending=False)     # 1 = best
    ranks = ranks.reset_index()
    ranks["setting"] = ranks["case"] + ":" + ranks["metric"]
    methods = [c for c in means.columns]
    by_setting = ranks.groupby("setting")[methods].mean().T
    by_setting["K<=5"] = ranks[ranks.K <= 5][methods].mean()
    by_setting["K>=6"] = ranks[ranks.K >= 6][methods].mean()
    by_setting["Overall"] = ranks[methods].mean()
    by_setting = by_setting.sort_values("Overall")
    fr = stats.friedmanchisquare(*[means[m].values for m in methods])
    cd = nemenyi_cd(len(methods), len(means))
    return by_setting, fr, cd, len(means)


def cd_diagram(avg_ranks, cd, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(avg_ranks.index)
    r = avg_ranks.values
    k = len(names)
    fig, ax = plt.subplots(figsize=(9, 0.45 * k + 1.6))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.8 * k - 0.8, 2)
    ax.axhline(0, color="k", lw=1)
    for i in range(1, k + 1):
        ax.plot([i, i], [0, 0.15], color="k", lw=1)
        ax.text(i, 0.3, str(i), ha="center", fontsize=9)
    ax.plot([1, 1 + cd], [1.2, 1.2], color="k", lw=2)
    ax.text(1 + cd / 2, 1.4, f"CD = {cd:.2f}", ha="center", fontsize=9)
    for i, (n, v) in enumerate(zip(names, r)):
        y = -(i + 1) * 0.8
        ax.plot([v, v], [0, y], color="gray", lw=0.8)
        ax.text(v + 0.05, y, f"{n} ({v:.2f})", va="center", fontsize=9)
    # bars joining methods whose ranks differ by less than CD
    order = np.argsort(r)
    rs = r[order]
    bars, start = [], 0
    for i in range(k):
        j = i
        while j + 1 < k and rs[j + 1] - rs[i] < cd:
            j += 1
        if j > i and not any(b[0] <= i and b[1] >= j for b in bars):
            bars.append((i, j))
    for b, (i, j) in enumerate(bars):
        ax.plot([rs[i] - 0.03, rs[j] + 0.03], [-0.3 - 0.18 * b] * 2, color="red", lw=3)
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--ref", default="Adaptive")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--exclude_substr", nargs="*", default=[],
                    help="drop methods whose name contains any of these strings (e.g. Oracle)")
    ap.add_argument("--out", default="results_supp")
    ap.add_argument("--tag", default="ablation")
    args = ap.parse_args()

    files = sorted({f for pat in args.inputs for f in glob.glob(pat)})
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for s in args.exclude_substr:
        df = df[~df.method.str.contains(s, regex=False)]
    print(f"Loaded {len(df)} rows from {len(files)} files; methods: {sorted(df.method.unique())}")

    summary = (df.groupby(["case", "metric", "K", "method"])["value"]
                 .agg(lambda x: f"{x.mean():.4f} ± {x.std(ddof=1):.4f}").unstack("method"))
    tests = paired_tests(df, args.ref, args.alpha)
    wtl = wtl_table(tests)
    ranks, fr, cd, nb = rank_analysis(df)

    out_xlsx = os.path.join(args.out, f"stats_{args.tag}.xlsx")
    os.makedirs(args.out, exist_ok=True)
    with pd.ExcelWriter(out_xlsx) as xw:
        summary.to_excel(xw, sheet_name="mean_std")
        tests.to_excel(xw, sheet_name="wilcoxon_holm", index=False)
        wtl.to_excel(xw, sheet_name="win_tie_loss", index=False)
        ranks.to_excel(xw, sheet_name="avg_ranks")
        pd.DataFrame([dict(friedman_chi2=fr.statistic, friedman_p=fr.pvalue, nemenyi_CD=cd,
                           n_blocks=nb, n_methods=len(ranks))]).to_excel(xw, sheet_name="friedman", index=False)
    png = os.path.join(args.out, f"cd_diagram_{args.tag}.png")
    cd_diagram(ranks["Overall"], cd, png, f"Average rank over {nb} (case, metric, K) blocks; "
                                          f"Friedman p = {fr.pvalue:.2g}")

    print("\nAverage ranks (1 = best):")
    print(ranks.round(2).to_string())
    print(f"\nFriedman chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.3g}; Nemenyi CD = {cd:.2f} "
          f"({nb} blocks, {len(ranks)} methods)")
    print(f"\nWin/Tie/Loss of '{args.ref}' vs each method (Holm-corrected Wilcoxon, alpha={args.alpha}):")
    for grp in ["method", "method+regime"]:
        print(wtl[wtl.grouping == grp].dropna(axis=1, how="all").drop(columns="grouping").to_string(index=False))
        print()
    print(f"\nSaved: {out_xlsx}\n       {png}")


if __name__ == "__main__":
    main()
