# Original experimental pipeline

Run every script from the repository root, e.g. `python original_pipeline/orig_sugar_gann_rerun.py`.
Data files are read from `datasets/` (Grapevine: `--grape_path`, default `datasets/Spectral_DataSet.xlsx`).

| File | Purpose |
|---|---|
| `notebooks/GA-NN_Github-*_SOTA.ipynb` | the seven MOEA baselines of Tables 2, 3, 6 and 9 (one notebook per dataset) |
| `notebooks/GA-NN_Github-Jaunisse-Final.ipynb`, `GA-NN_Github_Sugar.ipynb`, `GA-NN_Github-IndianPina.ipynb` | GA-NN and the learning-based baselines of Tables 4, 7 and 10 |
| `orig_sugar_gann_rerun.py` | GA-NN on the Sugar dataset with the three-class proxy head (same seeds and splits as the baselines) |
| `main_exp1_GrapeDiseases_Sensitivity.py`, `main_exp2_GrapeMaturity_Sensitivity.py`, `main_exp3_IndianPines_Sensitivity.py` | sensitivity analyses (all parameters by default, or a subset with `--params N T CR MR Temp`) |
| `merge_sensitivity.py` | optional: merges workbooks of parameters run separately with `--params` |
| `Result.py`, `result1.py`, `result2.py` | sensitivity figures of Case Studies 1, 2 and 3 (read the workbooks written by the scripts above) |

The notebooks are archived as run; their absolute data paths must be adapted before re-running them.
