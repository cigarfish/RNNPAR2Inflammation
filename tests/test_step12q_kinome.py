"""Step 12q validation: EV1 loader sanity, a positive control for the profile-similarity
predictor at the measured noise level, and the recorded verdict vs the fixed rule.

    python tests/test_step12q_kinome.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.kinome import load_ev1, selected_vs_annotated
from lembas_par2.oracle import resid_scores

spec = importlib.util.spec_from_file_location("k12q", ROOT / "scripts" / "run_kinome_12q.py")
k12q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k12q)

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
k = load_ev1(cfg)
print("== EV1 loader")
check("all 58 EV1 drugs map to our drug names", k.drug_ours.notna().all() and k.drug_ours.nunique() == 58,
      f"{k.drug_ours.nunique()} mapped")
check("occupancy in (0, 1)", k.occ.dropna().between(0, 1, inclusive="neither").all())
# the selected kinase is dosed at its own EC50 -> occupancy 0.5 (single-target selections)
single = k[~k["Selected for"].astype(str).str.contains(r"[;,]")]
hit = single[single.Target.str.upper() == single["Selected for"].astype(str).str.upper().str.strip()]
check("selected kinase sits at occupancy 0.5 (dose = its EC50)", np.allclose(hit.occ, 0.5, atol=1e-6),
      f"{len(hit)} drug-kinase rows checked")
s = selected_vs_annotated(cfg)
print(f"[INFO] dosed-for kinase in our annotation: {int(s.selected_in_annotation.sum())}/58; "
      f"any annotated target at occupancy >= 0.5: {int(s.annotated_occ_ge_half.sum())}/58")

print("\n== positive control: responses planted as a linear function of the union profiles")
occ = pd.read_csv(cfg.paths.output_dir / "step12q" / "occupancy.csv", index_col=0)
X = occ.values
rng = np.random.default_rng(0)
n_tf = 577
W = rng.normal(size=(X.shape[1], n_tf))
S = (X @ W).T
r_target = 0.55
k_snr = r_target ** 2 / (1 - r_target ** 2)
Ysyn = S + rng.normal(size=S.shape) * S.std() / np.sqrt(k_snr)
cond = pd.read_csv(cfg.paths.output_dir / "step06_conditions.csv").set_index("condition_id")
folds = cond.loc[occ.index, "fold"].astype(int).values
tr = [j for j in range(len(occ)) if folds[j] != 0]
ev = [j for j in range(len(occ)) if folds[j] == 0]
P = k12q.similarity_predict(Ysyn, X, tr, ev)
have = [i for i, j in enumerate(ev) if np.isfinite(P[:, j]).any()]
v = float(np.nanmedian(resid_scores(np.nan_to_num(P), Ysyn, tr, ev)[have]))
check("similarity predictor recovers planted profile-driven signal (>= 0.15)", v >= 0.15, f"{v:+.3f} on {len(have)} drugs")

print("\n== verdict")
rep = json.loads((cfg.paths.output_dir / "step12q" / "report.json").read_text())
u = rep["variants"]["union"]
pos = any(u[c]["val"] >= 0.15 and u[c]["perm_p"] < 0.05 for c in ["T", "O"])
neg = max(u["T"]["val"], u["O"]["val"]) < 0.05
expect = "MEASURED" if pos else "POLYPHARMACOLOGY" if neg else "AMBIGUOUS"
check("recorded verdict matches the fixed rule", rep["verdict"].startswith(expect), rep["verdict"][:50])
print(f"[INFO] union T {u['T']['val']:+.3f} (p {u['T']['perm_p']:.3f}, oof {u['T']['oof']:+.3f}); "
      f"O {u['O']['val']:+.3f} (p {u['O']['perm_p']:.3f}, oof {u['O']['oof']:+.3f})")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
