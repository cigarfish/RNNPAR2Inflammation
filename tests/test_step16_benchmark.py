"""Step 16 validation: the comparison can detect signal when it is there, no method sees the test
targets, and the metric behaves as specified.

    python scripts/run_benchmark_16.py
    python tests/test_step16_benchmark.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from lembas_par2 import load_config
from lembas_par2.benchmark import METHODS, drug_table, score

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
bc = cfg.benchmark
out = cfg.paths.output_dir / "step16"
if not check("step-16 outputs present", (out / "report.json").exists() and (out / "summary.csv").exists()):
    sys.exit(1)
rep = json.loads((out / "report.json").read_text())
summ = pd.read_csv(out / "summary.csv")

X, Y, folds, ids, rmax = drug_table(cfg)
vf = bc["primary_fold"]
test = [i for i in range(len(ids)) if folds[i] == vf and rmax[i] >= bc["signal_rmax"]]
train = [i for i in range(len(ids)) if folds[i] != vf]
rng = np.random.default_rng(0)

print("== setup")
check("every method gets the same inputs and conditions", rep["n_drugs"] == len(ids) and rep["n_signal_drugs"] == len(test),
      f"{len(ids)} drugs, {len(test)} signal drugs in fold {vf}")
check("LEMBAS is scored from the frozen harness results, 5 seeds",
      any(m.startswith("LEMBAS") for m in summ.method) and "seeds\": 5" in "".join(summ["info"].astype(str)))

print("\n== metric")
P = np.repeat(np.nanmean(Y.values[:, train], axis=1)[:, None], Y.shape[1], axis=1)
check("a constant profile scores ~0 residualised and high raw",
      abs(np.nanmedian(score(P, Y, train, test))) < 0.05 and np.nanmedian(score(P, Y, train, test, resid=False)) > 0.4,
      f"resid {np.nanmedian(score(P, Y, train, test)):+.3f}, raw {np.nanmedian(score(P, Y, train, test, resid=False)):+.3f}")
Pr, _ = METHODS["ridge"](cfg, X, Y, train, test, 0)
shift = rng.normal(size=Y.shape[0]) * np.nanstd(Y.values) * 3
check("residualised r is invariant to adding any constant profile",
      abs(np.nanmedian(score(Pr, Y, train, test)) - np.nanmedian(score(Pr + shift[:, None], Y, train, test))) < 1e-9)

print("\n== detection power (planted linear response, noised to the measured ceiling)")
# Not every method has to pass: power is a REPORTED quantity, because a method scoring ~0 on the
# real data AND low here is underpowered at this sample size rather than evidence of no signal.
# What must hold is that the comparison is informative at all, i.e. that some method has power.
sens = {r.method: r.sensitivity for r in summ.itertuples() if np.isfinite(r.sensitivity)}
check("power reported for every non-LEMBAS method",
      set(sens) == {m for m in summ.method if not str(m).startswith("LEMBAS")}, sorted(sens))
powered = [m for m, v in sens.items() if v >= bc["beats_chance"]["min_resid_r"]]
check("at least two methods can detect planted signal (else the comparison says nothing)",
      len(powered) >= 2, f"powered: {powered}")
check("the constant baseline has no power, as it must", sens.get("constant_mean_profile", 1.0) < 0.05,
      f"{sens.get('constant_mean_profile'):+.3f}")
check("the report records which methods are powered", set(rep["powered_methods"]) == set(powered),
      rep["powered_methods"])
for m, v in sorted(sens.items(), key=lambda kv: -kv[1]):
    print(f"[INFO] power {m:24s} {v:+.3f}" + ("" if v >= bc["beats_chance"]["min_resid_r"] else "   (underpowered)"))

print("\n== leakage: replacing the test-fold targets must not change any prediction")
for name in ["ridge", "pls", "gradient_boosting", "random_forest", "knn_occupancy_cosine", "diffusion_curated_graph"]:
    Y2 = Y.copy()
    Y2.iloc[:, test] = rng.normal(size=(Y.shape[0], len(test))) * 10
    A, _ = METHODS[name](cfg, X, Y, train, test, 0)
    B, _ = METHODS[name](cfg, X, Y2, train, test, 0)
    check(f"{name} never reads the test targets", np.allclose(np.nan_to_num(A[:, test]), np.nan_to_num(B[:, test]), atol=1e-8))

print("\n== verdict")
learners = ["ridge", "elastic_net", "pls", "gradient_boosting", "random_forest"]
beat = [r.method for r in summ.itertuples() if not str(r.method).startswith("LEMBAS")
        and r.resid_r >= bc["beats_chance"]["min_resid_r"] and r.perm_p < bc["beats_chance"]["max_perm_p"]]
check("recorded verdict follows the fixed rule", rep["verdict"].startswith("A METHOD BEATS CHANCE" if beat else "DATA LIMIT"),
      rep["verdict"][:60])
check("GNN decision follows the fixed rule",
      rep["gnn_decision"].startswith("build" if any(m in beat for m in learners) else "do NOT build"))
print("[INFO] " + "; ".join(f"{r.method} {r.resid_r:+.3f}" for r in summ.itertuples()))
print(f"[INFO] noise ceiling {rep['ceiling']:.3f}; any constant profile scores 0 residualised by construction")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
