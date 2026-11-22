# Supplementary experiments for review round 2

Copy this `supp/` folder into the repository root (next to `core/`, `utils/`, `datasets/`)
and run every command **from the repository root**.

## Install

```bash
pip install "pymoo>=0.6.0,<0.7" numba torch scikit-learn scipy pandas openpyxl matplotlib
```
`numba` is only needed for AGE-MOEA. Place `Spectral_DataSet.xlsx` (Grapevine) wherever you like
and pass it with `--grape_path`.

## Defaults (aligned with the paper text)

| Setting | Default | Reproduce released code |
|---|---|---|
| Logit initialisation | zero | `--init legacy` ([1.1, 1.0, 0.9]) |
| Sugar classification loss | cross-entropy, 3 outputs | `--sugar_cls_loss legacy_bce` |
| Indian Pines crop classes | paper: [1,2,3,4,8,9,10,11,12,13] (200 vs 120) | `--ip_crop code` |
| Top-K tie-break | relevance to primary target, then band index | — |
| GA CR / MR | honoured (0.8 / 0.01) | — |
| SoftMaskNet epochs | 800 | — |

Run `r` always uses seed `2025 + r`, so all methods (step 2) and all MOEAs (step 5) share the
same split and are paired.

## Run order

```bash
# Step 0 - check packages, folder layout and the three datasets             (seconds)
python supp/step0_check_setup.py --grape_path datasets/Spectral_DataSet.xlsx

# Step 1 - Fixed-Eq == Mean-Agg with the ORIGINAL code; tie statistics       (minutes)
python supp/step1_check_fixed_equals_mean.py --case ip --n_runs 3
python supp/step1_check_fixed_equals_mean.py --case grape_loyo --grape_path PATH --n_runs 3

# Step 2 - extended ablation (Joint-GA, no-redundancy, Fixed-CV)             (~1.5 h / case, 1 core)
python supp/step2_extended_ablation.py --case grape_random --grape_path PATH
python supp/step2_extended_ablation.py --case grape_loyo   --grape_path PATH
python supp/step2_extended_ablation.py --case sugar
python supp/step2_extended_ablation.py --case ip

# Step 3 - statistics (Wilcoxon + Holm, W/T/L, ranks, Friedman, CD diagram)
python supp/step3_stats.py --inputs "results_supp/*/raw_results.csv" --ref Adaptive

# Step 4 - mechanism analyses
python supp/step4_mechanism.py --case grape_loyo --grape_path PATH
python supp/step4_mechanism.py --case sugar --probe_runs 3

# Step 5 - MOEA native solutions (slowest; can run in parallel with step 2)
python supp/step5_moea_native.py --case sugar --mode fixedK
python supp/step5_moea_native.py --case grape_loyo --mode fixedK --grape_path PATH \
       --K_list 1 2 3 4 5 8 10 15
python supp/step3_stats.py --inputs results_supp/sugar/raw_results.csv \
       results_supp/moea_fixedK/sugar/raw_results.csv --ref Adaptive --tag sugar_vs_moea \
       --exclude_substr Oracle
```

Step 2 and step 5 append to CSV files and skip finished runs, so they can be interrupted and
resumed. For a quick smoke test add e.g. `--n_runs 2 --epochs 20 --Kmax 5 --grid_step 0.25`
(step 2) or `--n_runs 1 --K_list 1 3 5` (step 5).

## What each step verifies

| Step | Reviewer comment | Question answered |
|---|---|---|
| 0 | — | Is everything installed and are the three datasets readable? |
| 1 | 1, 4 | Is the network used at inference? Are Fixed-Eq and Mean-Agg the same method? |
| 2 | 4 | Contribution of decoupled search (Joint-GA), of the redundancy objective (noRed), and of learned vs tuned weights (Fixed-CV) |
| 3 | 4 | Which differences are statistically significant (paired, Holm-corrected) |
| 4 | 4 | Why Adaptive wins / loses: lexicographic ordering, subset redundancy, regret of w*, proxy-loss vs Top-K mismatch |
| 5 | 3 | Do frequency ranking + Top-K disadvantage the MOEAs compared with their native solutions? |

## Outputs (`results_supp/`)

- `<case>/raw_results.csv` — test scores, one row per (run, method, K, metric)
- `<case>/cv_scores.csv`, `grid_cv.csv`, `fixedcv_chosen_weights.csv` — inner-CV scores (training data only)
- `<case>/selections.csv` — selected band indices
- `<case>/timing.csv` — GA / Adaptive / Fixed-CV / Joint-GA time and evaluation counts
- `<case>/artifacts/run_XX.npz`, `histories/run_XX.json` — elite populations, learned weights, curves
- `<case>/step4_mechanism.xlsx`
- `moea_<mode>/<case>/` — `raw_results.csv`, `native_info.csv`, `size_diagnostics.csv`
- `stats_<tag>.xlsx`, `cd_diagram_<tag>.png`
