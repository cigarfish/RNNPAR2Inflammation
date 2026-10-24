"""Did removing the cell-composition signal recover the drug fit?
(outputs/composition_variants/summary.csv)

Run from the LEMBAS folder after scripts/run_composition_variants.py:
    python tests/check_composition_variants.py [--dir DIR]

Reference points, both measured earlier:
    drug-only training (stage A)        val r(drug) = 0.224   <- the ceiling
    all GEO contexts (stage B, step 12e) val r(drug) <= 0.033  <- the problem

Question fixed in advance: does excluding or down-weighting the contaminated bulk GEO
contexts move val r(drug) materially towards 0.224? The validation set is identical in
every variant, so the comparison is direct.
  verdict "composition"  if a variant with GEO data reaches val r(drug) >= 0.10
  verdict "cell type"    if only C4 (no GEO at all) recovers it
  verdict "neither"      if nothing recovers it - the cause is then the asymmetry between
                         a stimulus contrast and an inhibitor contrast, which neither
                         filtering nor macrophage extraction would fix
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
d = Path(a.dir) if a.dir else cfg.paths.output_dir / "composition_variants"
s = pd.read_csv(d / "summary.csv").sort_values("variant")
sc = pd.read_csv(d / "scores.csv")
pd.set_option("display.width", 250)
CEILING, BASE = 0.224, 0.033

print(f"contamination score = mean log2FC of the {cfg.composition['score_panel']} panel")
print(sc.groupby("kind")[["score", "overall_abs_median"]].agg(["size", "median"]).round(2).to_string())
print(f"\nmost contaminated GEO contexts:")
print(sc.head(8)[["condition_id", "cls", "kind", "score", "macrophage", "overall_abs_median"]].round(2).to_string(index=False))

if "val_r_tf_drug" not in s.columns or s.val_r_tf_drug.isna().all():
    print("\n[INFO] scores only (--steps 0); re-run without --steps 0 for the training comparison")
    sys.exit(0)

s = s[s.val_r_tf_drug.notna()].copy()
print("\n== training comparison (validation set identical in every variant)")
cols = ["variant", "spec", "n_train", "n_train_geo", "n_dropped_geo", "val_r_tf_drug_init",
        "val_r_tf_drug", "val_r_tf_geo", "val_r_tf", "tf_gain", "route_dSting1", "rho"]
print(s[[c for c in cols if c in s.columns]].round(3).to_string(index=False))
print(f"[INFO] reference: drug-only stage A reached {CEILING:+.3f}; all-GEO stage B gave <= {BASE:+.3f}")

with_geo = s[(s.n_train_geo > 0)]
best_geo = with_geo.loc[with_geo.val_r_tf_drug.idxmax()] if len(with_geo) else None
c4 = s[s.variant == "C4"]
print("\n== verdict")
if best_geo is not None and best_geo.val_r_tf_drug >= 0.10:
    print(f"COMPOSITION: {best_geo.variant} keeps {int(best_geo.n_train_geo)} GEO contexts and reaches "
          f"val r(drug) {best_geo.val_r_tf_drug:+.3f} (>= 0.10).\n"
          "  The contamination was the problem. Filtering or down-weighting is enough, and\n"
          "  macrophage extraction would be a cleaner version of the same fix - worth doing.")
elif len(c4) and float(c4.val_r_tf_drug.iloc[0]) >= 0.10:
    print(f"CELL TYPE / CONTRAST: only C4 (no GEO at all) recovers the drug fit "
          f"({float(c4.val_r_tf_drug.iloc[0]):+.3f}); filtering contaminated sets does not.\n"
          "  Removing infiltration genes is not sufficient, so the conflict is with the GEO\n"
          "  contrasts themselves. Macrophage extraction may still help (it changes the\n"
          "  contrast, not just the gene set), but it is no longer the obvious fix - compare\n"
          "  against simply training the perturbation model separately.")
else:
    print("NEITHER: no variant recovers val r(drug) to 0.10, not even dropping GEO entirely.\n"
          "  Then the contamination is not the cause and neither is cell type: what differs is\n"
          "  the size of the perturbation (a stimulus switches inputs on and off, a drug nudges\n"
          "  one node deep inside the network). Macrophage extraction would not fix that, and\n"
          "  the honest route is to report the claim on curated evidence plus holdout directions.")
    if len(c4):
        print(f"  (C4, no GEO, reached {float(c4.val_r_tf_drug.iloc[0]):+.3f} against the stage A "
              f"reference {CEILING:+.3f} - if C4 is far below it, the difference is the perturbation\n"
              "   papers and their sign constraints, which stage A did not train on.)")

hc = [c for c in s.columns if c.startswith("holdout_")]
if hc:
    print("\n== holdout Sting1 delta (information only, never used to choose)")
    print(s[["variant"] + hc].round(3).to_string(index=False))
