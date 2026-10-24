"""Step 13 validation: mechanics (prior-only reproduces step 9b, binomial, ablation identity,
attribution gradient, sign-free loading), completeness, and the verdict vs the fixed rules.

    python scripts/run_step13.py
    python tests/test_step13.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.model import build_model
from lembas_par2.step13 import binom_tail, holdout_observations, ko_attribution, ko_delta, score_holdouts

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
st = cfg.step13
o = cfg.paths.output_dir
out = o / "step13"
if not check("step-13 outputs present", (out / "report.json").exists() and (out / "runs.csv").exists()):
    sys.exit(1)
rep = json.loads((out / "report.json").read_text())
runs = pd.read_csv(out / "runs.csv")

print("== mechanics")
check("binomial tail: P(X >= 9 | 10) = 11/1024", abs(binom_tail(9, 10) - 11 / 1024) < 1e-12)
obs = holdout_observations(cfg)
check("10 unique scored holdout observations (Ifnb1 duplicate removed)", len(obs) == 10, len(obs))
m = build_model(cfg, device="cpu", seed=0)
m.initialize()
s, tab = score_holdouts(m, obs, st["call_threshold"], st["route_subset"])
base = pd.read_csv(o / "step09b_holdout_baseline.csv").dropna(subset=["observed"])
base = base.drop_duplicates(["condition", "node", "layer"]).set_index(["condition", "node"]).predicted
mine = tab.set_index(["condition_id", "node"]).predicted
diff = float((mine - base.reindex(mine.index)).abs().max())
check("prior-only predictions reproduce the step-9b holdout baseline", diff < 1e-4, f"max |diff| {diff:.1e}")
print(f"[INFO] prior-only: {s['k']}/{s['n']} with the call threshold {st['call_threshold']}, "
      f"{s['k_sign_only']}/{s['n']} by sign alone (step 9b reported 8/11 by sign, with the duplicate)")
w = m.weights()
d0 = ko_delta(m)[m.idx["Sting1"]].item()
d1 = ko_delta(m, weights=w.clone())[m.idx["Sting1"]].item()
check("ablation with an empty mask leaves PAR2-KO dSting1 unchanged", abs(d0 - d1) < 1e-9)
d, contrib = ko_attribution(m)
e = m.edges
k = int(np.flatnonzero((e.source == "Cgas").values & (e.target == "Sting1").values)[0])
eps = 1e-3 * abs(w[k].item())
wp = w.clone()
wp[k] = wp[k] + eps
fd = (ko_delta(m, weights=wp)[m.idx["Sting1"]].item() - d0) / eps
check("attribution gradient matches finite differences (Cgas->Sting1)", abs(contrib[k] / w[k].item() - fd) < 2e-2 * max(1, abs(fd)),
      f"adjoint {contrib[k] / w[k].item():.4f} vs FD {fd:.4f}")
check("attribution reproduces the simulated dSting1", abs(d - d0) < 1e-6, f"{d:+.5f} vs {d0:+.5f}")
check("step 13 never calls model selection", "select(" not in (ROOT / "scripts" / "run_step13.py").read_text())

print("\n== completeness")
need = len(st["seeds"])
seeds = runs.groupby("variant").size().to_dict()
for v in [st["primary"], st["primary"] + "F", st["secondary"], st["secondary"] + "F",
          st["primary"] + "U", st["secondary"] + "U"]:
    check(f"{v}: {need} seeds", seeds.get(v, 0) >= need, seeds.get(v, 0))
f = runs[runs.variant.str.endswith("F")]
if len(f):
    check("sign-free runs really had free route signs", (f.n_unlocked == 135).all(), f.n_unlocked.unique().tolist())

print("\n== verdict")
agg = rep["variants"]
p, fp = st["primary"], st["primary"] + "F"
if agg.get(p, {}).get("n_seeds", 0) >= need and agg.get(fp, {}).get("n_seeds", 0) >= need:
    exp = ("CONTRADICTED" if agg[fp]["ko_pooled_dSting1_median"] < -st["reversal_threshold"] else
           "CONSISTENT" if agg[p]["holdout_p_at_median"] < st["holdout_alpha"] else "WEAKENED")
else:
    exp = "PENDING"
check("recorded verdict follows the fixed rules", rep["verdict"].startswith(exp), rep["verdict"][:70])
check("route-sign null computed with the configured number of permutations",
      rep["sign_null"]["n"] >= st["sign_null_permutations"], f"{rep['sign_null']['n']} / {st['sign_null_permutations']}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
