# supp/step6_make_tables.py
"""
STEP 6 - Turn the experiment outputs into what the revised manuscript \\input{}s.

Writes into <out>/tables/ (copy this folder next to the .tex file):
  ablation_grapevine.tex, ablation_sugar.tex, ablation_ip.tex   (reviewer comment 4)
  moea_freq_grapevine.tex, moea_freq_sugar.tex, moea_freq_ip.tex (main MOEA comparison, paper protocol)
  moea_native.tex, moea_native_sizes.tex                         (reviewer comment 3)
  gann_column_<case>.txt     GA-NN (Adaptive) column to paste into the learning-based tables
  summary_for_text.txt       every number the blue \\TODO{} markers in the manuscript ask for
and into <out>/figures/: <Case>_Convergence.pdf, <Case>_Weights.pdf (axis labels use omega).

Significance markers (paired Wilcoxon signed-rank vs GA-NN (Adaptive), Holm across K, alpha=0.05):
  + significantly better than Adaptive,  - significantly worse than Adaptive.
A table is only written when its inputs exist, so the script can be run at any time.

Usage (from repo root):
    python supp/step6_make_tables.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supp.common import TASK_NAMES, BASE_SEED  # noqa: E402
from supp.step3_stats import paired_tests       # noqa: E402

ALGO_LABEL = {"NSGA2": "NSGA-II", "NSGA3": "NSGA-III", "RVEA": "RVEA", "SPEA2": "SPEA2", "MOEAD": "MOEA/D",
              "AGEMOEA": "AGE-MOEA", "SMSEMOA": "SMS-EMOA"}
METRIC_LABEL = {"Acc": "classification accuracy", "R2": "regression $R^2$",
                "AccFine": "fine-grained accuracy", "AccCoarse": "coarse-grained accuracy"}
ALGO_ORDER = ["NSGA2", "NSGA3", "RVEA", "SPEA2", "MOEAD", "AGEMOEA", "SMSEMOA"]
SHORT = {"grape_random": ("Class", "Year"), "grape_loyo": ("Class", "Year"),
         "sugar": ("Class", "Reg"), "ip": ("Fine", "Crop")}
BLOCKS = {
    "grapevine": [("grape_random", "Acc", "Experiment 1: Random Split (70/30) -- Classification Accuracy (LDA)"),
                  ("grape_loyo", "Acc", "Experiment 2: Temporal Transfer (Train 2020--2023, Test 2024) -- "
                                        "Classification Accuracy (LDA)")],
    "sugar": [("sugar", "Acc", "Task 1: Classification Accuracy (LDA)"),
              ("sugar", "R2", "Task 2: Regression $R^2$ Score (Random Forest)")],
    "ip": [("ip", "AccFine", "Task 1: Fine-grained (16-Class) Accuracy (LDA)"),
           ("ip", "AccCoarse", "Task 2: Coarse-grained (Crop) Accuracy (LDA)")],
}


def ablation_methods(case):
    t1, t2 = SHORT[case]
    return [("SO-Task1", f"Single-Obj\\\\({t1})"), ("SO-Task2", f"Single-Obj\\\\({t2})"),
            ("MeanAgg", "Mean-Agg\\\\(= Fixed Eq)"), ("FixedP1", "Fixed P1"), ("FixedP2", "Fixed P2"),
            ("JointGA-Eq", "Joint-GA\\\\(Eq)"), ("JointGA-wstar", "Joint-GA\\\\($\\omega^*$)"),
            ("Adaptive-noRed", "Adaptive\\\\w/o Red."), ("Adaptive", "GA-NN\\\\(Adaptive)")]


def fmt(m, s):
    return f"{m:.4f} $\\pm$ {s:.4f}"


def block_rows(df, case, metric, methods, ref="Adaptive", with_std=True, Ks=None):
    """LaTeX rows for one (case, metric): mean +- std, best in bold, +/- significance vs ref."""
    d = df[(df.case == case) & (df.metric == metric)]
    present = [m for m, _ in methods if m in set(d.method)]
    tests = paired_tests(d[d.method.isin(present)], ref, 0.05) if ref in present else pd.DataFrame()
    sig = {}
    for _, r in tests.iterrows():
        sig[(r["method"], int(r["K"]))] = {"win": "-", "loss": "+", "tie": ""}[r["result"]]
    Ks = Ks or sorted(d.K.unique())
    rows, wtl = [], {m: [0, 0, 0] for m, _ in methods}
    for K in Ks:
        dk = d[d.K == K]
        means = {m: dk[dk.method == m].value.mean() for m, _ in methods if m in present}
        stds = {m: dk[dk.method == m].value.std(ddof=1) for m, _ in methods if m in present}
        best = max(means.values()) if means else None
        cells = []
        for m, _ in methods:
            if m not in means or np.isnan(means[m]):
                cells.append("--")
                continue
            txt = fmt(means[m], stds[m]) if with_std else f"{means[m]:.4f}"
            if best is not None and np.isclose(means[m], best, atol=5e-5):
                txt = f"\\textbf{{{txt}}}"
            mk = sig.get((m, K), "")
            if mk:
                txt += f"$^{{{mk}}}$"
                wtl[m][0 if mk == "-" else 2] += 1
            elif m != ref and (m, K) in sig:
                wtl[m][1] += 1
            cells.append(txt)
        rows.append(f"{K} & " + " & ".join(cells) + " \\\\")
    footer = "W/T/L & " + " & ".join(
        ("--" if m == ref or m not in present else f"{wtl[m][0]}/{wtl[m][1]}/{wtl[m][2]}") for m, _ in methods)
    return rows, footer + " \\\\"


def write_ablation(df, name, outdir):
    blocks = BLOCKS[name]
    if not all(((df.case == c) & (df.metric == mt)).any() for c, mt, _ in blocks):
        return None
    methods = ablation_methods(blocks[0][0])
    n = len(methods)
    head = " & ".join(f"\\textbf{{\\makecell{{{lab}}}}}" for _, lab in methods)
    lines = ["\\resizebox{\\textwidth}{!}{", f"\\begin{{tabular}}{{@{{}}c {'c' * n}@{{}}}}", "\\toprule"]
    for i, (case, metric, title) in enumerate(blocks):
        if i:
            lines.append("\\midrule")
        lines += [f"\\multicolumn{{{n + 1}}}{{c}}{{\\textbf{{{title}}}}} \\\\", "\\midrule",
                  f"\\textbf{{K}} & {head} \\\\", "\\midrule"]
        rows, footer = block_rows(df, case, metric, methods)
        lines += rows + ["\\midrule", footer]
    lines += ["\\bottomrule", "\\end{tabular}", "}"]
    path = os.path.join(outdir, f"ablation_{name}.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def write_moea_freq(df_abl, df_moea, name, outdir):
    blocks = BLOCKS[name]
    if df_moea is None:
        return None
    df = pd.concat([df_abl[df_abl.method == "Adaptive"], df_moea[df_moea.method.str.endswith("|Freq")]])
    df = df.assign(method=df.method.str.replace("|Freq", "", regex=False))
    if not all(((df.case == c) & (df.metric == mt) & (df.method != "Adaptive")).any() for c, mt, _ in blocks):
        return None
    methods = [(a, ALGO_LABEL[a]) for a in ALGO_ORDER] + [("Adaptive", "GA-NN (Adaptive)")]
    n = len(methods)
    head = " & ".join(f"\\textbf{{{lab}}}" for _, lab in methods)
    lines = ["\\resizebox{\\textwidth}{!}{", f"\\begin{{tabular}}{{@{{}}c {'c' * n}@{{}}}}", "\\toprule"]
    for i, (case, metric, title) in enumerate(blocks):
        if i:
            lines.append("\\midrule")
        lines += [f"\\multicolumn{{{n + 1}}}{{c}}{{\\textbf{{{title}}}}} \\\\", "\\midrule",
                  f"\\textbf{{K}} & {head} \\\\", "\\midrule"]
        rows, footer = block_rows(df, case, metric, methods)
        lines += rows + ["\\midrule", footer]
    lines += ["\\bottomrule", "\\end{tabular}", "}"]
    path = os.path.join(outdir, f"moea_freq_{name}.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def write_native(df_abl, df_nat, outdir, summary):
    """Native-CV mean per algorithm; arrow = significant change vs the same algorithm's Freq protocol."""
    if df_nat is None:
        return None
    case_titles = {"grape_random": "Grapevine, random split", "grape_loyo": "Grapevine, temporal transfer (LOYO)",
                   "sugar": "Sugar", "ip": "Indian Pines"}
    algos = [a for a in ALGO_ORDER if (df_nat.method == f"{a}|NativeCV").any()]
    has_cv = (df_abl.method == "FixedCV-perK").any()
    n = len(algos) + 2 + int(has_cv)
    head = " & ".join(f"\\textbf{{{ALGO_LABEL[a]}}}" for a in algos) + \
        " & \\textbf{\\makecell{GA-NN\\\\(Adaptive)}}" + \
        (" & \\textbf{\\makecell{GA-NN\\\\(CV weights)}}" if has_cv else "") + \
        " & \\textbf{\\makecell{Oracle\\\\(best MOEA)}}"
    lines = ["\\resizebox{\\textwidth}{!}{", f"\\begin{{tabular}}{{@{{}}c {'c' * n}@{{}}}}", "\\toprule"]
    first = True
    for case in ["grape_random", "grape_loyo", "sugar", "ip"]:
        dn = df_nat[df_nat.case == case]
        if dn.empty:
            continue
        for metric in dn.metric.unique():
            if not first:
                lines.append("\\midrule")
            first = False
            lines += [f"\\multicolumn{{{n + 1}}}{{c}}{{\\textbf{{{case_titles[case]} -- {METRIC_LABEL.get(metric, metric)}}}}} \\\\",
                      "\\midrule", f"\\textbf{{K}} & {head} \\\\", "\\midrule"]
            d = dn[dn.metric == metric]
            arrows = {}
            for a in algos:
                pair = d[d.method.isin([f"{a}|NativeCV", f"{a}|Freq"])].copy()
                pair["method"] = pair.method.map({f"{a}|NativeCV": "N", f"{a}|Freq": "F"})
                t = paired_tests(pair, "N", 0.05)
                for _, r in t.iterrows():
                    arrows[(a, int(r["K"]))] = {"win": "$\\uparrow$", "loss": "$\\downarrow$", "tie": ""}[r["result"]]
            ad = df_abl[(df_abl.case == case) & (df_abl.metric == metric) & (df_abl.method == "Adaptive")]
            cvw = df_abl[(df_abl.case == case) & (df_abl.metric == metric) & (df_abl.method == "FixedCV-perK")]
            for K in sorted(d.K.unique()):
                cells = []
                for a in algos:
                    v = d[(d.method == f"{a}|NativeCV") & (d.K == K)].value
                    cells.append(f"{v.mean():.4f}{arrows.get((a, K), '')}" if len(v) else "--")
                av = ad[ad.K == K].value
                cells.append(f"{av.mean():.4f}" if len(av) else "--")
                if has_cv:
                    cv_ = cvw[cvw.K == K].value
                    cells.append(f"{cv_.mean():.4f}" if len(cv_) else "--")
                orc = d[d.method.str.endswith("|Oracle") & (d.K == K)].groupby("method").value.mean()
                cells.append(f"{orc.max():.4f}" if len(orc) else "--")
                lines.append(f"{K} & " + " & ".join(cells) + " \\\\")
            # summary numbers for the text
            fr = d[d.method.str.endswith("|Freq")].groupby("K").value.mean()
            nc = d[d.method.str.endswith("|NativeCV")].groupby("K").value.mean()
            summary.append(f"[native] {case}:{metric} mean over algorithms, Freq vs NativeCV by K: " +
                           ", ".join(f"K={k}: {fr.get(k, np.nan):.4f}/{nc.get(k, np.nan):.4f}" for k in fr.index))
            bestnat = d[d.method.str.endswith("|NativeCV")].groupby(["K", "method"]).value.mean().groupby("K").max()
            adm = ad.groupby("K").value.mean()
            cvm = cvw.groupby("K").value.mean() if has_cv else pd.Series(dtype=float)
            summary.append(f"[native vs GA-NN] {case}:{metric} K: Adaptive / CV-weights / mean NativeCV / best NativeCV: " +
                           "; ".join(f"K={k}: {adm.get(k, np.nan):.4f}/{cvm.get(k, np.nan):.4f}/"
                                     f"{nc.get(k, np.nan):.4f}/{bestnat.get(k, np.nan):.4f}" for k in nc.index))
            n_up = sum(1 for k, v in arrows.items() if v == "$\\uparrow$")
            n_dn = sum(1 for k, v in arrows.items() if v == "$\\downarrow$")
            summary.append(f"[native] {case}:{metric} significant changes NativeCV vs Freq: up {n_up}, down {n_dn}, "
                           f"of {len(arrows)} (algorithm, K) cases")
    lines += ["\\bottomrule", "\\end{tabular}", "}"]
    path = os.path.join(outdir, "moea_native.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def write_native_sizes(base, outdir, summary):
    rows = []
    for f in glob.glob(os.path.join(base, "moea_*", "*", "size_diagnostics.csv")):
        rows.append(pd.read_csv(f))
    if not rows:
        return None
    sz = pd.concat(rows).drop_duplicates(subset=["case", "run", "algo"])
    cases = [c for c in ["grape_random", "grape_loyo", "sugar", "ip"] if c in set(sz.case)]
    dims = {"grape_random": 600, "grape_loyo": 600, "sugar": 204, "ip": 200}
    lab = {"grape_random": "Grapevine (random)", "grape_loyo": "Grapevine (LOYO)", "sugar": "Sugar", "ip": "Indian Pines"}
    lines = [f"\\begin{{tabular}}{{@{{}}l {'c' * len(cases)}@{{}}}}", "\\toprule",
             "\\textbf{Algorithm} & " + " & ".join(f"\\textbf{{{lab[c]}}} ($d={dims[c]}$)" for c in cases) + " \\\\",
             "\\midrule"]
    for a in ALGO_ORDER:
        cells = []
        for c in cases:
            s = sz[(sz.algo == a) & (sz.case == c)]
            cells.append(f"{s.size_median.median():.0f} [{s.size_min.min():.0f}, {s.size_max.max():.0f}]"
                         if len(s) else "--")
        lines.append(f"{ALGO_LABEL[a]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path = os.path.join(outdir, "moea_native_sizes.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    summary.append(f"[native sizes] overall min / median / max size of native ND solutions: "
                   f"{sz.size_min.min():.0f} / {sz.size_median.median():.0f} / {sz.size_max.max():.0f}; "
                   f"fraction of native ND solutions with 1<=size<=15: {sz.frac_final_in_K_range.mean():.3f}")
    return path


def figures(base, case, figdir, summary):
    hist = sorted(glob.glob(os.path.join(base, case, "histories", "run_*.json")))
    if not hist:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = [json.load(open(f)) for f in hist]
    names = TASK_NAMES[case]
    W = np.array([h["w_hist"] for h in H])              # runs x epochs x T
    L = np.array([h["loss_hist"] for h in H])
    ep = np.arange(1, W.shape[1] + 1)
    title = {"grape_random": "Grapevine (random split)", "grape_loyo": "Grapevine (LOYO)",
             "sugar": "Sugar", "ip": "Indian Pines"}[case]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for t, nm in enumerate(names):
        med = np.median(W[:, :, t], 0)
        q1, q3 = np.percentile(W[:, :, t], [25, 75], 0)
        ax[0].plot(ep, med, label=f"$\\omega_{{\\mathrm{{{nm}}}}}$")
        ax[0].fill_between(ep, q1, q3, alpha=0.2)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Weight (median, IQR over runs)")
    ax[0].legend()
    ax[0].set_title(f"{title}: weight trajectories")
    ax[1].boxplot([W[:, -1, t] for t in range(len(names))])
    ax[1].set_xticks(range(1, len(names) + 1), [f"$\\omega_{{\\mathrm{{{nm}}}}}$" for nm in names])
    ax[1].set_ylabel("Final weight")
    ax[1].set_title(f"Final weights over {len(H)} runs")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"{case}_Weights.pdf"))
    plt.close(fig)

    G = [np.array([h["ga_curves"][t] for h in H]) for t in range(len(names))]
    fig, ax = plt.subplots(1, len(names) + 1, figsize=(14, 3.2))
    for t, nm in enumerate(names):
        g = np.arange(1, G[t].shape[1] + 1)
        ax[t].plot(g, np.median(G[t], 0))
        ax[t].fill_between(g, *np.percentile(G[t], [25, 75], 0), alpha=0.25)
        ax[t].set_title(f"$\\mathcal{{F}}$: {nm}")
        ax[t].set_xlabel("Generation")
    ax[-1].plot(ep, np.median(L, 0))
    ax[-1].fill_between(ep, *np.percentile(L, [25, 75], 0), alpha=0.25)
    ax[-1].set_title("Proxy training loss")
    ax[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"{case}_Convergence.pdf"))
    plt.close(fig)

    fw = W[:, -1, :]
    summary.append(f"[weights] {case}: final weights mean {np.round(fw.mean(0), 3).tolist()}, "
                   f"median {np.round(np.median(fw, 0), 3).tolist()}, IQR "
                   f"{[f'{np.percentile(fw[:, t], 25):.3f}-{np.percentile(fw[:, t], 75):.3f}' for t in range(fw.shape[1])]}"
                   f" ({len(H)} runs; epochs = {W.shape[1]})")
    dom = np.argmax(fw, 1)
    summary.append(f"[weights] {case}: runs in which each task gets the largest weight: "
                   f"{ {names[t]: int((dom == t).sum()) for t in range(len(names))} }; "
                   f"runs with max weight > 0.9: {int((fw.max(1) > 0.9).sum())}")


def text_numbers(base, case, df, summary):
    """Numbers quoted in the manuscript text (lexicographic ordering, tie structure, subset redundancy)."""
    arts = sorted(glob.glob(os.path.join(base, case, "artifacts", "run_*.npz")))
    if not arts:
        return
    strict, all_ones, sizes = [], [], []
    for f in arts:
        a = np.load(f)
        c = [a["pop1"].mean(0), a["pop2"].mean(0), a["pop3"].mean(0)]
        w = a["w_adaptive"]
        strict.append(w[0] / len(a["pop1"]) > w[1] + w[2])
        all_ones.append(int(((c[0] == 1) & (c[1] == 1) & (c[2] == 1)).sum()))
        sizes.append([a[p].sum(1).mean() for p in ("pop1", "pop2", "pop3")])
    sizes = np.array(sizes)
    summary.append(f"[masks] {case}: elite mask size per population, mean over runs "
                   f"{np.round(sizes.mean(0), 1).tolist()}, range of run means {sizes.min():.0f}-{sizes.max():.0f}")
    summary.append(f"[lexicographic] {case}: runs with w1/|S| > w2 + w3 (strictly lexicographic ranking): "
                   f"{sum(strict)}/{len(strict)}")
    summary.append(f"[ties] {case}: bands selected by every elite of all three populations: "
                   f"mean {np.mean(all_ones):.1f}, range {min(all_ones)}-{max(all_ones)}")
    sel_path = os.path.join(base, case, "selections.csv")
    if os.path.exists(sel_path):
        sel = pd.read_csv(sel_path)
        fused = sel[sel.method.isin(["MeanAgg", "FixedP1", "FixedP2", "Adaptive"])]
        same = fused.groupby(["run", "K"])["indices"].nunique() == 1
        k = same.reset_index().K.values
        summary.append(f"[ties] {case}: Mean-Agg, Fixed P1, Fixed P2 and Adaptive select the same subset in "
                       f"{same.mean():.1%} of (run, K); K<=5: {same[k <= 5].mean():.1%}; K>=6: {same[k >= 6].mean():.1%}")
    d = df[df.case == case]
    if (d.method == "Adaptive").any():
        t = paired_tests(d[~d.method.str.startswith("FixedCV")], "Adaptive", 0.05)
        if t.empty:
            summary.append(f"[W/T/L] {case}: not enough paired runs yet (need >= 5)")
            return
        c = t.groupby("method").result.value_counts().unstack(fill_value=0)
        for col in ["win", "tie", "loss"]:
            if col not in c:
                c[col] = 0
        summary.append(f"[W/T/L of Adaptive vs ...] {case} (all metrics, K=1..15): " +
                       "; ".join(f"{m}: {r.win}/{r.tie}/{r.loss}" for m, r in c.iterrows()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_supp")
    args = ap.parse_args()
    tabdir = os.path.join(args.out, "tables")
    figdir = os.path.join(args.out, "figures")
    os.makedirs(tabdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    raw = [pd.read_csv(f) for f in glob.glob(os.path.join(args.out, "*", "raw_results.csv"))
           if "moea_" not in f]
    df = pd.concat(raw, ignore_index=True) if raw else pd.DataFrame(columns=["case", "run", "method", "K", "metric", "value"])
    mp = [pd.read_csv(f) for f in glob.glob(os.path.join(args.out, "moea_paper", "*", "raw_results.csv"))]
    nk = [pd.read_csv(f) for f in glob.glob(os.path.join(args.out, "moea_fixedK", "*", "raw_results.csv"))]
    df_nat = pd.concat(nk, ignore_index=True) if nk else None
    # Frequency protocol (main MOEA tables): paper-mode results, completed by the identical
    # unconstrained runs recorded in fixedK mode (same seeds -> same values) where paper mode is missing
    freq_parts = mp + ([df_nat[df_nat.method.str.endswith("|Freq")]] if df_nat is not None else [])
    df_moea = (pd.concat(freq_parts, ignore_index=True)
               .drop_duplicates(subset=["case", "run", "method", "K", "metric"]) if freq_parts else None)

    summary, written = [], []
    for name in BLOCKS:
        for p in [write_ablation(df, name, tabdir), write_moea_freq(df, df_moea, name, tabdir)]:
            if p:
                written.append(p)
    for p in [write_native(df, df_nat, tabdir, summary), write_native_sizes(args.out, tabdir, summary)]:
        if p:
            written.append(p)
    for case in ["grape_random", "grape_loyo", "sugar", "ip"]:
        if (df.case == case).any():
            ad = df[(df.case == case) & (df.method == "Adaptive")]
            col = ad.groupby(["metric", "K"]).value.agg(["mean", "std"]).reset_index()
            with open(os.path.join(tabdir, f"gann_column_{case}.txt"), "w") as fh:
                for _, r in col.iterrows():
                    fh.write(f"{r.metric}\tK={int(r.K)}\t{r['mean']:.4f} $\\pm$ {r['std']:.4f}\n")
            text_numbers(args.out, case, df, summary)
            figures(args.out, case, figdir, summary)
    with open(os.path.join(tabdir, "summary_for_text.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    print("Written:\n  " + "\n  ".join(written + [os.path.join(tabdir, "summary_for_text.txt"), figdir]))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
