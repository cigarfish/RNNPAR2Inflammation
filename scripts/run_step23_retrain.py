"""Step 23: retrain with the Ca2+ arm in place and measure what fitting does to dSting1.

Option (b) of 2026-10-19. The topology changed in step 22 (SE_CA2 -> Vdac1 added and promoted to
T0; Cgas -> Sting1 demoted to T1), so every checkpoint is stale. This retrains, re-profiles the
route, and prints a before/after comparison.

**The question this answers, and the one it does not.** At initialisation dSting1 is +0.024, down
from +0.50 but still the wrong sign. This asks whether FITTING moves it, before we spend more
prior edges on the Ca2+ arm. It does NOT test the PAR2 -> STING claim: PAR2_29's pSTING has
already been read (it is DOWN), so nothing measured here can validate a revised model against it.
Whatever comes out is exploratory.

**Nothing is selected on dSting1.** Step 18's rule picks the readout variant on residualised drug
r, exactly as before, and the holdout set is unchanged. dSting1 is reported only.

    python scripts/run_step23_retrain.py                 # ~16 min on a GPU (10 runs)
    python scripts/run_step23_retrain.py --seeds 0 1 2    # ~10 min
    python scripts/run_step23_retrain.py --report-only    # re-print without retraining
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config

PRE = "step21_route_pre_step22"      # route profile from before the Ca2+ arm was added
CA2_CHAIN = ["Gnaq", "Plcb1", "SE_IP3", "SE_CA2", "Vdac1"]
KEY = ["Mapk1", "Mcl1", "Bcl2l11", "Bax", "Bak1", "SE_CYTO_MTDNA", "Cgas", "Sting1", "Irf3"]


def run(cmd, label):
    print(f"\n{'=' * 78}\n[{label}] {' '.join(cmd[1:])}\n{'=' * 78}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"\n[{label}] FAILED (exit {r.returncode}). Stopping so the failure is not hidden.")
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=["0", "1", "2", "3", "4"])
    ap.add_argument("--variants", nargs="*", default=["R4", "R0"])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    py = sys.executable

    if not a.report_only:
        run([py, "scripts/run_source_variants.py", "--seeds", *a.seeds,
             "--variants", *a.variants], "1/3 retrain")
        run([py, "scripts/run_route_profile.py", "--variant", a.variants[0],
             "--seeds", *a.seeds], "2/3 route profile")
        run([py, "tests/check_readout_param.py"], "3/3 readout-variant rule")

    print(f"\n{'=' * 78}\n[summary] what fitting did to the route\n{'=' * 78}")
    post = cfg.paths.output_dir / "step21_route" / "route.csv"
    pre = cfg.paths.output_dir / PRE / "route.csv"
    if not post.exists():
        sys.exit(f"missing {post}; drop --report-only")
    P = pd.read_csv(post).set_index("node")
    rows = []
    if pre.exists():
        Q = pd.read_csv(pre).set_index("node")
        for n in KEY + [x for x in CA2_CHAIN if x not in KEY]:
            if n in P.index:
                rows.append(dict(node=n,
                                 before=float(Q.at[n, "delta_lps"]) if n in Q.index else np.nan,
                                 after=float(P.at[n, "delta_lps"]),
                                 rank_after=int(P.at[n, "rank_lps"])))
    else:
        print(f"[warn] no {PRE}/route.csv, showing the new profile only")
        for n in KEY + [x for x in CA2_CHAIN if x not in KEY]:
            if n in P.index:
                rows.append(dict(node=n, before=np.nan, after=float(P.at[n, "delta_lps"]),
                                 rank_after=int(P.at[n, "rank_lps"])))
    T = pd.DataFrame(rows)
    T["change"] = T.after - T.before
    print("\nPAR2 KO under LPS, median over seeds (before = same model, pre-Ca2+ topology):")
    print(T.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    s = T[T.node == "Sting1"]
    if len(s):
        b, af = float(s.before.iloc[0]), float(s.after.iloc[0])
        print(f"\n  dSting1  {b:+.4f} -> {af:+.4f}")
        print("  VERDICT: " + ("still POSITIVE - fitting did not flip it; the Ca2+ arm needs the "
                               "rest of its chain in T0, or the claim is simply wrong"
                               if af > 0 else
                               "NEGATIVE - fitting flipped the sign. Report as exploratory: "
                               "PAR2_29's pSTING was already read, so it cannot validate this."))
    print("\n  Reminder: the holdout set is unchanged and still contains no Sting1 observation,\n"
          "  so the 8/10 headline neither supports nor contradicts whatever is printed above.")
    print("\nnext: python scripts/run_step13.py --out step13_ca2 --perm 200 --variants R4 R0")


if __name__ == "__main__":
    main()
