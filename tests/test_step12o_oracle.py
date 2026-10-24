"""Step 12o validation: the oracle and target-identity checks can detect signal when it is
there (positive controls at the measured noise ceiling), never read validation drugs when
choosing settings, and the recorded verdict follows the fixed rules.

    python tests/test_step12o_oracle.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.eval_harness import SIGNAL_RMAX, load_model
from lembas_par2.oracle import adjacency, corr, hops, predict, resid_scores, target_identity

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
o = cfg.paths.output_dir
rep_p = o / "step12o" / "report.json"
if not check("12o report present", rep_p.exists()):
    sys.exit(1)
rep = json.loads(rep_p.read_text())

m, _ = load_model(cfg, o / "source_variants" / "J3" / "checkpoint.pt", "cpu")
cond = m.design.cond
rmax = pd.read_csv(o / "step12f0" / "ceiling.csv").set_index("condition_id").r_max
drugs = [i for i in range(len(cond)) if cond.source.iloc[i] == "drug"]
ids = list(cond.condition_id)
vf = cfg.stage_b["val_fold"]
tr = [j for j, i in enumerate(drugs) if cond.fold.iloc[i] != vf]
ev = [j for j, i in enumerate(drugs) if cond.fold.iloc[i] == vf and float(rmax.get(ids[i], np.nan)) >= SIGNAL_RMAX]
n = len(m.nodes)
delta = np.zeros((n, len(drugs)))
for j, i in enumerate(drugs):
    for t in cond.trt_inhibit.iloc[i]:
        delta[m.idx[t], j] = -1.0
tf_idx = m.tf_idx.cpu().numpy()
rng = np.random.default_rng(0)


def add_noise(S, r_target=0.55):
    """Noise so that r(truth, measured) ~ r_target (the measured ceiling) after residualising."""
    with np.errstate(all="ignore"):
        Sr = S - np.nanmean(S[:, tr], axis=1, keepdims=True)
    sd = np.nanstd(Sr)
    k = r_target ** 2 / (1 - r_target ** 2)          # signal/noise variance ratio
    return S + rng.normal(size=S.shape) * sd / np.sqrt(k)


# ---------- 1. positive control: propagation signal ----------
print("== positive control: targets planted from curated 3-hop propagation, noise at the ceiling")
H = hops(adjacency(m, "curated"), delta)
scale = [np.sqrt(np.mean(h[tf_idx][:, tr] ** 2)) or 1.0 for h in H]
S = predict(H, tf_idx, 1.0, 3, scale)
Ysyn = add_noise(S)
best = -np.inf
for K in range(1, 7):
    for a in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]:
        P = predict(H, tf_idx, a, K, scale)
        s_tr = np.nanmedian(resid_scores(P, Ysyn, tr, [j for j in tr]))
        if s_tr > best:
            best, Pbest = s_tr, P
val = float(np.nanmedian(resid_scores(Pbest, Ysyn, tr, ev)))
check("oracle recovers planted propagation signal (val resid r >= 0.30)", val >= 0.30, f"{val:+.3f}")

# ---------- 2. positive control: target-identity signal ----------
print("\n== positive control: each target contributes a fixed random TF profile, noise at the ceiling")
tg = sorted({t for i in drugs for t in cond.trt_inhibit.iloc[i]})
prof = {t: rng.normal(size=len(tf_idx)) for t in tg}
S2 = np.stack([sum(prof[t] for t in cond.trt_inhibit.iloc[i]) for i in drugs], axis=1)
Y2 = add_noise(S2)
targets = [set(cond.trt_inhibit.iloc[i]) for i in drugs]
classes = [cond.cls.iloc[i] for i in drugs]
Pt = target_identity(Y2, targets, classes, tr, ev, "target")
have = [k for k, j in enumerate(ev) if np.isfinite(Pt[:, j]).any()]
sc = resid_scores(np.nan_to_num(Pt, nan=0.0), Y2, tr, ev)
v2 = float(np.nanmedian(sc[have]))
check("target-identity baseline recovers planted target signal (>= 0.15)", v2 >= 0.15, f"{v2:+.3f} on {len(have)} drugs")

# ---------- 3. no leakage ----------
print("\n== leakage")
Yreal = np.zeros((len(tf_idx), len(drugs)))
Y3 = Ysyn.copy()
Y3[:, ev] = rng.normal(size=(len(tf_idx), len(ev))) * 50
P1 = target_identity(Ysyn, targets, classes, tr, ev, "target")
P3 = target_identity(Y3, targets, classes, tr, ev, "target")
check("target-identity predictions do not read validation targets",
      np.allclose(np.nan_to_num(P1[:, ev]), np.nan_to_num(P3[:, ev])))
gt = pd.read_csv(o / "step12o" / "oracle_grid.csv")
g0 = gt[gt.fold == vf]
sel = g0.sort_values("train_score", ascending=False).iloc[0]
check("oracle setting chosen by the training score, not the validation score",
      (sel.graph, int(sel.K), float(sel.alpha)) == (rep["oracle"]["selected"]["graph"], rep["oracle"]["selected"]["K"],
                                                    rep["oracle"]["selected"]["alpha"]),
      f"{sel.graph} K={int(sel.K)} a={sel.alpha}; best val-score setting would be "
      f"{g0.sort_values('val_score', ascending=False).iloc[0][['graph', 'K', 'alpha', 'val_score']].to_dict()}")

# ---------- 4. verdict follows the rules ----------
print("\n== verdict")
O, T = rep["effective"]["O"], rep["effective"]["T"]
expect = ("TRANSMISSION" if O >= 0.15 else "GRAPH/KNOWLEDGE" if (O < 0.05 and T >= 0.15)
          else "IDENTIFIABILITY" if (O < 0.05 and T < 0.05) else "AMBIGUOUS")
check("recorded verdict matches the fixed rules", rep["verdict"].startswith(expect), rep["verdict"][:60])
print(f"[INFO] oracle {rep['oracle']['val']:+.3f} (p {rep['oracle']['perm_p']:.3f}, oof {rep['oracle']['oof']:+.3f}); "
      f"target {rep['target_identity']['target']['val']:+.3f}; class {rep['target_identity']['class']['val']:+.3f}; "
      f"best in-sample oracle {g0.train_score.max():+.3f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
