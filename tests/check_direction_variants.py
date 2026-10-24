"""Compare direction-term variants (outputs/direction_variants/summary.csv).

Run from the LEMBAS folder after scripts/run_direction_variants.py:
    python tests/check_direction_variants.py [--dir DIR]

Selection rule, fixed in advance (holdouts are printed but never used):
  1. not dominated by a few constraints:   top-3 share <= 25%
  2. ERK stays responsive:                 pre-activation z(Mapk1) in [0, 1]
  3. constraints are still being used:     fraction satisfied not below the D0 baseline
  4. among those: best validation median r(TF), then lowest validation loss
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lembas_par2 import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--dir")
a = ap.parse_args()
cfg = load_config()
d = Path(a.dir) if a.dir else cfg.paths.output_dir / "direction_variants"
s = pd.read_csv(d / "summary.csv").sort_values("variant")
pd.set_option("display.width", 250)
trained = "val_r_tf" in s.columns and s.val_r_tf.notna().any()

missing = [v for v in cfg.direction_variants if v not in set(s.variant)]
print(f"variants: {list(s.variant)}" + (f"   (missing: {missing})" if missing else ""))

print("\n== how the term behaves at the stage A starting point")
cols = ["variant", "spec", "lambda_direction", "top3_share", "gini", "frac_satisfied", "erk_grad_ratio"]
print(s[cols].round(3).to_string(index=False))
print("[INFO] top3_share = share of the term from its 3 largest constraints (D0 was 68%);"
      "\n       erk_grad_ratio = |direction gradient on ERK bias| / |measured-data gradient| (D0 was ~50x)")

if trained:
    done = s[s.val_r_tf.notna()].copy()
    skipped = sorted(set(s.variant) - set(done.variant))
    print("\n== after the short stage B run" + (f"   (static only, not trained: {skipped})" if skipped else ""))
    cols = ["variant", "val_r_tf_before", "val_r_tf", "val_loss_before", "val_loss", "frac_satisfied_after",
            "route_dSting1", "route_dMapk1", "z_Mapk1", "z_Sting1", "rho"]
    print(done[[c for c in cols if c in done.columns]].round(3).to_string(index=False))
    print("[INFO] frac_satisfied_after = share of the 206 constraints with the right sign, on the training"
          "\n       conditions where they live (the validation fold holds only drug and GEO conditions)")
    s = done
    base = s[s.variant == "D0"]
    base_sat = float(base.frac_satisfied_after.iloc[0]) if len(base) else 0.0
    s["not_dominated"] = s.top3_share <= 0.25
    s["erk_responsive"] = s.z_Mapk1.between(0.0, 1.0)
    s["constraints_used"] = s.frac_satisfied_after >= base_sat - 1e-9
    s["eligible"] = s.not_dominated & s.erk_responsive & s.constraints_used
    print("\n== selection (validation fit among eligible variants; holdouts NOT used)")
    print(s[["variant", "not_dominated", "erk_responsive", "constraints_used", "eligible",
             "val_r_tf", "val_loss"]].round(3).to_string(index=False))
    rank = s[s.eligible].sort_values(["val_r_tf", "val_loss"], ascending=[False, True])
    if len(rank):
        b = rank.iloc[0]
        print(f"\nbest by the rule: {b.variant}  (val r_tf {b.val_r_tf:+.3f}, val loss {b.val_loss:.3f}, "
              f"top-3 share {b.top3_share:.1%}, z(ERK) {b.z_Mapk1:+.2f})")
    else:
        print("\nno variant meets all three criteria - report which criterion each fails and reconsider")

    hc = [c for c in s.columns if c.startswith("holdout_")]
    if hc:
        print("\n== holdout Sting1 delta (information only, not used for selection)")
        print(s[["variant"] + hc].round(3).to_string(index=False))
else:
    print("\n[INFO] static run only (--steps 0); re-run without --steps 0 for the training comparison")
