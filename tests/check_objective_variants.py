"""Compare data-objective variants (outputs/objective_variants/summary.csv).

Run from the LEMBAS folder after
    python scripts/run_transmission_variants.py --objective
    python tests/check_objective_variants.py [--dir DIR]

The question: does making the TF term scale-free let training improve the pattern it is
judged on? Under 'absolute' the loss sits at the target variance (~1.86) because
predicting ~0 is cheapest, and the leftover gradient produced the step-11b bias drift
and step-12c T1 shrinkage instead.

Selection rule, fixed in advance (holdouts printed, never used):
  1. the loss must move the metric: val r(TF) after > val r(TF) at initialisation
  2. the network must stay usable: rho <= 0.97, route dSting1 >= 0.02, every watched
     node's pre-activation in [0, 1]
  3. a fitted gain must be non-negative (amendment, step 12e - see below)
  4. among those: best val r(TF), then lowest val loss
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
d = Path(a.dir) if a.dir else cfg.paths.output_dir / "objective_variants"
s = pd.read_csv(d / "summary.csv").sort_values("variant")
pd.set_option("display.width", 250)
watch = cfg.objective_test["watch"]
zc = [f"z_{n}" for n in watch if f"z_{n}" in s.columns]

print(f"variants: {list(s.variant)}")
print("\n== initialisation (identical network across variants; only the objective differs)")
print(s[["variant", "tf_mode", "median_w_T1", "tf_delta_median_init", "val_r_tf_init", "val_loss_init"]]
      .round(4).to_string(index=False))

if "val_r_tf" not in s.columns or s.val_r_tf.isna().all():
    print("\n[INFO] static run only; re-run without --steps 0")
    sys.exit(0)

s = s[s.val_r_tf.notna()].copy()
print("\n== after training")
cols = ["variant", "tf_mode", "val_r_tf_init", "val_r_tf", "val_r_tf_drug", "val_r_tf_geo", "val_loss",
        "tf_delta_median", "median_w_T1_after", "tf_gain", "route_dSting1", "rho"] + zc
print(s[[c for c in cols if c in s.columns]].round(4).to_string(index=False))
print("[INFO] val_loss is not comparable across tf_mode (different objectives);"
      "\n       val r(TF) is the common, scale-invariant metric")

s["metric_improved"] = s.val_r_tf > s.val_r_tf_init
s["usable"] = (s.rho <= 0.97) & (s.route_dSting1 >= 0.02) & \
              s[zc].apply(lambda r: r.between(0.0, 1.0).all(), axis=1)
# Amendment after step 12e: a fitted gain must not be negative. A negative gain fits a
# condition by predicting the opposite pattern and flipping it, so the sign of the
# response is unconstrained - measured at 24/119 conditions for gain_condition. The
# original rule did not exclude it; `loss.tf_gain_nonneg` is now the default and this
# criterion makes the exclusion explicit rather than an argument after the fact.
neg = s.get("gain_frac_negative")
s["gain_ok"] = ((s.tf_gain.isna() | (s.tf_gain >= 0)) &
                (True if neg is None else (neg.isna() | (neg <= 0))))
s["eligible"] = s.metric_improved & s.usable & s.gain_ok
print("\n== selection (holdouts NOT used)")
print(s[["variant", "tf_mode", "metric_improved", "usable", "gain_ok", "eligible", "val_r_tf",
         "val_r_tf_drug", "val_loss"]].round(4).to_string(index=False))
rank = s[s.eligible].sort_values(["val_r_tf", "val_loss"], ascending=[False, True])
if len(rank):
    b = rank.iloc[0]
    print(f"\nbest by the rule: {b.variant} ({b.tf_mode})  val r(TF) {b.val_r_tf_init:+.3f} -> {b.val_r_tf:+.3f}, "
          f"route dSting1 {b.route_dSting1:+.3f}, rho {b.rho:.3f}")
    base = s[s.variant == "S0"]
    if len(base):
        print(f"baseline S0 (absolute): val r(TF) {float(base.val_r_tf_init.iloc[0]):+.3f} -> "
              f"{float(base.val_r_tf.iloc[0]):+.3f}")
else:
    print("\nno variant both improved val r(TF) and stayed usable - the limit is then the model, "
          "not the objective; report that rather than tuning further")

hc = [c for c in s.columns if c.startswith("holdout_")]
if hc:
    print("\n== holdout Sting1 delta (information only)")
    print(s[["variant"] + hc].round(3).to_string(index=False))
