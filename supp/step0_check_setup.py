# supp/step0_check_setup.py
"""
STEP 0 - Check the environment, the folder layout and the three datasets.

Usage (from the repository root, i.e. the folder that contains core/, utils/, datasets/, supp/):
    python supp/step0_check_setup.py --grape_path datasets/Spectral_DataSet.xlsx
"""
import argparse
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ok = True


def check(cond, msg_ok, msg_fail):
    global ok
    print(("  [OK]   " + msg_ok) if cond else ("  [FAIL] " + msg_fail))
    ok = ok and cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grape_path", default="datasets/Spectral_DataSet.xlsx")
    args = ap.parse_args()

    print("1) Working directory")
    check(os.path.abspath(os.getcwd()) == ROOT, f"running from repo root ({ROOT})",
          f"please `cd {ROOT}` first (you are in {os.getcwd()})")
    for d in ["core", "utils", "datasets", "supp"]:
        check(os.path.isdir(os.path.join(ROOT, d)), f"folder {d}/ found", f"folder {d}/ missing")

    print("2) Python packages")
    print(f"  Python {sys.version.split()[0]}")
    for pkg in ["numpy", "scipy", "pandas", "sklearn", "torch", "pymoo", "openpyxl", "matplotlib"]:
        try:
            m = importlib.import_module(pkg)
            check(True, f"{pkg} {getattr(m, '__version__', '')}", "")
        except ImportError:
            check(False, "", f"{pkg} not installed -> pip install {'scikit-learn' if pkg == 'sklearn' else pkg}")
    try:
        importlib.import_module("numba")
        print("  [OK]   numba (needed only for AGE-MOEA)")
    except ImportError:
        print("  [WARN] numba not installed: AGE-MOEA in step 5 will fail (pip install numba)")
    try:
        import torch
        print(f"  [INFO] torch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except ImportError:
        pass

    print("3) Datasets")
    sys.path.insert(0, ROOT)
    import numpy as np
    try:
        from supp.common import load_case_data
        d = load_case_data("sugar")
        check(d["X"].shape == (274, 204) and len(np.unique(d["y1"])) == 3,
              f"Sugar: X {d['X'].shape}, 3 varieties", f"Sugar: unexpected shape {d['X'].shape}")
    except Exception as e:
        check(False, "", f"Sugar failed to load: {e}")
    try:
        d = load_case_data("ip")
        check(d["X"].shape == (320, 200), f"Indian Pines: X {d['X'].shape}, crop/non-crop = "
              f"{int(d['y2'].sum())}/{int((1 - d['y2']).sum())}", f"Indian Pines: unexpected shape {d['X'].shape}")
    except Exception as e:
        check(False, "", f"Indian Pines failed to load: {e}")
    if not os.path.exists(args.grape_path):
        check(False, "", f"Grapevine file not found at {args.grape_path}: copy Spectral_DataSet.xlsx "
                         f"into datasets/ or pass --grape_path")
    else:
        try:
            d = load_case_data("grape_random", args.grape_path)
            years = [y.item() if hasattr(y, "item") else y for y in np.unique(d["year_raw"])]
            check(d["X"].shape[1] == 600 and set(np.unique(d["y1"])) == {0, 1},
                  f"Grapevine: X {d['X'].shape}, years {years}, test year for LOYO = {years[-1]}",
                  f"Grapevine: unexpected content X {d['X'].shape}")
        except Exception as e:
            check(False, "", f"Grapevine failed to load: {e}")

    print("\nAll checks passed - continue with step 1." if ok else "\nFix the [FAIL] items above first.")


if __name__ == "__main__":
    main()
