"""Compare stage A variants (outputs/stage_a_variants/summary.csv).

Run from the LEMBAS folder after scripts/run_stage_a_variants.py:
    python tests/check_stage_a_variants.py [--dir DIR]

Selection rule (holdouts excluded on purpose):
  1. responsive: route dSting1 under simulated MEK inhibition >= 0.02 and the STING
     arm (mtDNA, Cgas, Sting1, Tbk1) inside the responsive range (pre-activation 0..1)
  2. among responsive variants: best validation median r(TF), then validation loss
Holdout predictions are printed separately for information only.
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
d = Path(a.dir) if a.dir else cfg.paths.output_dir / "stage_a_variants"
s = pd.read_csv(d / "summary.csv")
pd.set_option("display.width", 250)
arm = ["SE_CYTO_MTDNA", "Cgas", "Sting1", "Tbk1"]

missing = [v for v in cfg.stage_a_variants if v not in set(s.variant)]
print(f"variants: {list(s.variant)}" + (f"   (missing: {missing})" if missing else ""))
for v in s.variant:
    ok = all((d / v / f).exists() for f in ["checkpoint.pt", "log.jsonl", "report.json"])
    if not ok:
        print(f"[WARN] {v}: outputs incomplete")

print("\n== fit (validation = drug fold 0)")
print(s[["variant", "spec", "steps", "train_loss", "val_loss", "train_r_tf", "val_r_tf", "jak_r_tf", "rho"]]
      .round(3).to_string(index=False))
print("\n== mechanism and drift")
cols = ["variant", "jak_stat_w"] + [f"dbias_{n}" for n in arm] + [f"z_{n}" for n in arm] + ["leak_frac", "sat_frac"]
print(s[cols].round(3).to_string(index=False))
print("\n== route under simulated MEK inhibition (pooled reference)")
print(s[["variant", "route_dMapk1", "route_dBax", "route_dMtDNA", "route_dCgas", "route_dSting1"]].round(3).to_string(index=False))

z_ok = s[[f"z_{n}" for n in arm]].apply(lambda r: ((r >= 0) & (r <= 1.0)).all(), axis=1)
s["responsive"] = (s.route_dSting1 >= 0.02) & z_ok
rank = s[s.responsive].sort_values(["val_r_tf", "val_loss"], ascending=[False, True])
print("\n== selection (validation fit among responsive variants; holdouts NOT used)")
print(s[["variant", "responsive", "val_r_tf", "val_loss"]].round(3).to_string(index=False))
if len(rank):
    print(f"\nbest by the rule: {rank.iloc[0].variant}  (val r_tf {rank.iloc[0].val_r_tf:+.3f}, val loss {rank.iloc[0].val_loss:.3f})")
else:
    print("\nno variant meets the responsiveness criterion")

hcols = [c for c in s.columns if c.startswith("holdout_")]
print("\n== holdout Sting1 delta (information only, not used for selection)")
print(s[["variant"] + hcols].round(3).to_string(index=False))
