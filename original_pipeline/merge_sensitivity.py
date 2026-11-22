# merge_sensitivity.py
"""
Replace, in an original sensitivity workbook, the parameters that were re-run.

The original CR/MR sensitivity runs used a GA that ignored CR and MR. After re-running only
those parameters (``--params CR MR``) with the patched core/ga_utils.py, this script puts the new
rows into the original workbook (sheet Raw_Sensitivity_Data and sheets Trend_<param>), so that the
original plotting scripts (Result.py, result1.py, result2.py) can be used unchanged.

The original workbook is first copied to <name>_original_backup.xlsx (only once).

Usage:
    python merge_sensitivity.py --original grapevine_sensitivity_results.xlsx \
        --new grapevine_sens_CR.xlsx grapevine_sens_MR.xlsx
"""
import argparse
import os
import shutil

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="default: overwrite --original (after a backup)")
    args = ap.parse_args()

    out = args.out or args.original
    backup = args.original.replace(".xlsx", "_original_backup.xlsx")
    if out == args.original and not os.path.exists(backup):
        shutil.copy2(args.original, backup)
        print(f"backup written: {backup}")
    source = backup if os.path.exists(backup) else args.original

    sheets = pd.read_excel(source, sheet_name=None, header=None)      # keep the layout of every sheet
    raw = pd.read_excel(source, sheet_name="Raw_Sensitivity_Data")
    replaced = []
    for f in args.new:
        new_raw = pd.read_excel(f, sheet_name="Raw_Sensitivity_Data")
        params = sorted(new_raw["Parameter_Tested"].unique())
        raw = pd.concat([raw[~raw["Parameter_Tested"].isin(params)], new_raw], ignore_index=True)
        new_sheets = pd.read_excel(f, sheet_name=None, header=None)
        for p in params:
            sheets[f"Trend_{p}"] = new_sheets[f"Trend_{p}"]
            replaced.append(p)
        print(f"{f}: replaced parameters {params} ({len(new_raw)} rows)")

    with pd.ExcelWriter(out, engine="openpyxl") as w:
        raw.to_excel(w, sheet_name="Raw_Sensitivity_Data", index=False)
        for name, df in sheets.items():
            if name != "Raw_Sensitivity_Data":
                df.to_excel(w, sheet_name=name, header=False, index=False)
    counts = raw.groupby("Parameter_Tested").size().to_dict()
    print(f"written: {out} | rows per parameter: {counts} | re-run parameters: {sorted(set(replaced))}")


if __name__ == "__main__":
    main()
