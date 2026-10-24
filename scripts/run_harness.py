"""Step 12f: score saved checkpoints with the shared evaluation harness.

Run from the LEMBAS folder (GPU optional; ~10-30 s per checkpoint):
    python scripts/run_harness.py                          # the default baseline set below
    python scripts/run_harness.py --runs C4=outputs/composition_variants/C4/checkpoint.pt
    python scripts/run_harness.py --sealed                 # also write sealed holdout files
Writes outputs/harness/12f_<variant>.json (+ sealed/12f_<variant>_holdout.json); then:
    python tests/test_step12f_harness.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.eval_harness import evaluate_run, load_model, write_result

DEFAULT_RUNS = {
    "S5": "outputs/objective_variants/S5/checkpoint.pt",      # adopted step-12e baseline (baseline_s5)
    "C0": "outputs/composition_variants/C0/checkpoint.pt",    # all GEO
    "C4": "outputs/composition_variants/C4/checkpoint.pt",    # no GEO
    "stageA": "outputs/stage_a_variants/V6/checkpoint.pt",    # stage A V6 (drug only, route neighbourhood)
}
SHOW = ["r_tf_drug", "r_tf_drug_resid", "r_tf_drug_signal", "r_tf_drug_resid_signal", "baseline_drug_mean",
        "ceiling_drug_rmax_signal", "r_tf_geo", "r_tf_geo_resid", "baseline_geo_mean", "r_tf_val",
        "rho_absA", "rho_signedA", "route_dSting1", "gain_drug", "gain_geo", "delta_tf_median",
        "solver_residual_max"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="NAME=path/to/checkpoint.pt")
    ap.add_argument("--step", default="12f")
    ap.add_argument("--device")
    ap.add_argument("--sealed", action="store_true", help="also write sealed holdout metrics")
    a = ap.parse_args()
    cfg = load_config()
    runs = dict(r.split("=", 1) for r in a.runs) if a.runs else DEFAULT_RUNS
    rows = []
    for name, ck in runs.items():
        ck = Path(ck) if Path(ck).is_absolute() else ROOT / ck
        if not ck.exists():
            print(f"[skip] {name}: {ck} not found")
            continue
        import torch
        dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        m, meta = load_model(cfg, ck, dev)
        res = evaluate_run(cfg, ck, step=a.step, variant=name, device=dev, model=m, meta=meta)
        p = write_result(cfg, res)
        rows.append({"variant": name, **{k: res[k] for k in SHOW}})
        print(f"[{name}] -> {p.relative_to(ROOT)}")
        if a.sealed:
            from lembas_par2.eval_sealed import evaluate_holdouts
            _, sp = evaluate_holdouts(cfg, ck, step=a.step, variant=name, device=dev, model=m)
            print(f"        sealed -> {sp.relative_to(ROOT)}")
    t = pd.DataFrame(rows).set_index("variant").T
    print("\n" + t.to_string(float_format=lambda x: f"{x:+.3f}" if abs(x) >= 1e-3 or x == 0 else f"{x:.1e}"))
    print("\nselection metric: r_tf_drug_resid_signal (drugs with step-12f0 r_max >= 0.3); "
          "holdouts are not shown here.\nnext: python tests/test_step12f_harness.py")


if __name__ == "__main__":
    main()
