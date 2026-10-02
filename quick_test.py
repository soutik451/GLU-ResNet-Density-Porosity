# quick_test.py
# runs the three notebooks on the synthetic example data (QUICK_TEST mode, few trials / epochs)
# without jupyter. takes ~1-3 min on a normal laptop CPU.
#
#   python quick_test.py
#
# you should see "QUICK TEST PASSED" at the end. output files go to outputs/

import os, sys, json, time
import matplotlib
matplotlib.use("Agg")

NOTEBOOKS = [
    "01_feature_relevance_SHAP.ipynb",
    "02_density_prediction_GRN.ipynb",
    "03_porosity_prediction_GRN.ipynb",
]

EXPECTED = [
    "outputs/shap/shap_ranking_Density.csv",
    "outputs/density/density_grn_metrics.csv",
    "outputs/density/density_grn_weights.pt",
    "outputs/density/density_grn_section.npy",
    "outputs/porosity/porosity_grn_metrics.csv",
    "outputs/porosity/porosity_grn_weights.pt",
    "outputs/porosity/porosity_grn_section.npy",
]


def run_notebook(path):
    nb = json.load(open(path, encoding="utf-8"))
    code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "__main__"}
    for i, src in enumerate(code):
        try:
            exec(compile(src, f"{path} [cell {i}]", "exec"), ns)
        except Exception:
            print(f"\n!! error in {path}, code cell {i}:\n{src[:300]}")
            raise
    return ns


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())

    if not os.path.exists("example_data/synthetic_well_trace.csv"):
        exec(open("make_synthetic_data.py").read())

    t0 = time.time()
    for nb in NOTEBOOKS:
        print("\n" + "=" * 70 + "\n running", nb, "\n" + "=" * 70)
        ns = run_notebook(nb)
        if "res" in ns:
            print(ns["res"].round(4))

    missing = [f for f in EXPECTED if not os.path.exists(f)]
    if missing:
        print("\nQUICK TEST FAILED, missing outputs:", missing)
        sys.exit(1)
    print("\nQUICK TEST PASSED  (%.0f s)" % (time.time() - t0))
