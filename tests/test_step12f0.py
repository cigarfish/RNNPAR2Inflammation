"""Step 12f0 validation: drug noise ceiling + trivial baselines (lembas_par2/noise_ceiling.py).

Run from the LEMBAS folder after `python scripts/run_noise_ceiling.py`:
    python tests/test_step12f0.py
CPU only, ~1 min.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.noise_ceiling import (ULM, baseline_predictions, ceiling_from_r12, corr, evaluate_baselines)

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
o = cfg.paths.output_dir
out = o / "step12f0"
files = ["ceiling.csv", "baselines.csv", "baselines_per_condition.csv", "shared_reference.csv",
         "approx_full_lfc.csv.gz", "report.json"]
if not check("run outputs present", all((out / f).exists() for f in files),
             {f: (out / f).exists() for f in files}):
    sys.exit(1)
rep = json.loads((out / "report.json").read_text())
ceil = pd.read_csv(out / "ceiling.csv")
clip = cfg.loss["tf_target_clip"]
tfs = list(pd.read_csv(o / "step05_tf_outputs.csv").tf)
reg = pd.read_csv(o / "step05_regulons.csv.gz")
cond = pd.read_csv(o / "step06_conditions.csv")
tfa = pd.read_csv(o / "step06_tf_activity.csv.gz", index_col=0).reindex(tfs).clip(-clip, clip)

# ---------- 1. the ceiling formula ----------
print("\n== ceiling formula")
ok = True
for k in [0.05, 0.3, 1.0, 4.0, 20.0]:
    r12 = k / np.sqrt((k + 4 / 3) * (k + 5 / 6))
    k2, R, rmax = ceiling_from_r12(r12)
    ok &= abs(k2 - k) < 1e-8 * max(1, k) and abs(R - k / (k + 0.5)) < 1e-10
check("ceiling_from_r12 inverts r12(k) exactly", ok)
check("non-positive r12 gives a zero ceiling", ceiling_from_r12(-0.2)[2] == 0.0 and ceiling_from_r12(np.nan)[2] == 0.0)

# Monte Carlo: correlated TF-like signal, replicate noise; r_max must match r(full, truth)
rng = np.random.default_rng(0)
emp, pred = [], []
for k in [0.2, 1.0, 3.0]:
    zs, rf = [], []
    for _ in range(400):
        s = rng.normal(size=500) * np.sqrt(k)
        d = rng.normal(size=(3, 500)); c = rng.normal(size=(6, 500))
        h1 = s + d[0] - c[:3].mean(0)
        h2 = s + d[1:].mean(0) - c[3:].mean(0)
        f = s + d.mean(0) - c.mean(0)
        zs.append(np.arctanh(corr(h1, h2))); rf.append(corr(f, s))
    emp.append(np.mean(rf)); pred.append(ceiling_from_r12(np.tanh(np.mean(zs)))[2])
check("simulated r_max matches r(full target, truth) within 0.02", np.max(np.abs(np.subtract(emp, pred))) < 0.02,
      " ".join(f"{e:.3f}/{p:.3f}" for e, p in zip(emp, pred)))

# ---------- 2. approximation vs the real targets ----------
print("\n== approximate log2FC vs the step-6 targets")
full = pd.read_csv(out / "approx_full_lfc.csv.gz", index_col=0)
deg = cfg.root / cfg.conditions["drug_deg_dir"]
r_lfc, r_tf = [], []
ulm = ULM(full.index, reg, tfs, cfg.conditions["min_tf_targets_condition"], clip)
for drug in full.columns:
    p = deg / f"{drug}_DEG.csv"
    cid = f"GSE219229_{drug}"
    if not p.exists() or cid not in tfa.columns:
        continue
    d = pd.read_csv(p).drop_duplicates("SYMBOL").set_index("SYMBOL")
    d = d[d.baseMean >= 20]
    g = full.index.intersection(d.index)
    r_lfc.append(corr(full.loc[g, drug].values, d.loc[g, "log2FoldChange"].values))
    r_tf.append(corr(ulm(full[drug]).values, tfa[cid].values))
check("gene log2FC agrees with pydeseq2 (median r >= 0.90, baseMean >= 20)", np.median(r_lfc) >= 0.90,
      f"median {np.median(r_lfc):.3f}, min {np.min(r_lfc):.3f}, n {len(r_lfc)}")
check("TF profile agrees with the step-6 target (median r >= 0.90)", np.median(r_tf) >= 0.90,
      f"median {np.median(r_tf):.3f}, min {np.min(r_tf):.3f}")

# ---------- 3. ceiling table ----------
print("\n== ceiling table")
check("ceiling for all 56 GSE219229 drugs, 60 splits each", len(ceil) >= 56 and (ceil.n_splits == 60).all(),
      f"{len(ceil)} drugs, splits {sorted(ceil.n_splits.unique())}")
check("r_max in [0, 1]", ceil.r_max.between(0, 1).all())
check("replicate batch removed: median split-half r12 > 0 and <= 20 % negative",
      ceil.r12.median() > 0 and (ceil.r12 < 0).mean() <= 0.20,
      f"median {ceil.r12.median():.3f}, negative {(ceil.r12 < 0).sum()}/{len(ceil)}")
check("ceiling rises with signal (Spearman r_max vs n_sig > 0.3)",
      ceil[["r_max", "n_sig"]].dropna().corr(method="spearman").iloc[0, 1] > 0.3,
      f"{ceil[['r_max', 'n_sig']].dropna().corr(method='spearman').iloc[0, 1]:.2f}")

# ---------- 4. baselines: no leakage ----------
print("\n== baselines")
drugs = cond[cond.source == "drug"]
val = list(drugs.condition_id[drugs.fold == 0])
train = list(drugs.condition_id[drugs.fold != 0])
P1 = baseline_predictions(tfa, cond, val, train)
t2 = tfa.copy()
t2[val] = rng.normal(size=t2[val].shape)
P2 = baseline_predictions(t2, cond, val, train)
check("baseline predictions do not read validation targets",
      all(np.allclose(P1[k].values, P2[k].values, equal_nan=True) for k in P1))
s, _ = evaluate_baselines(tfa, cond, "drug", 0)
check("baselines reproducible from the saved table",
      abs(s.loc["B_mean", "val_fold_median"] - rep["baselines"]["drug:B_mean"]["val"]) < 1e-9)

# ---------- 5. verdict (rules fixed in noise_ceiling.py before the run) ----------
print("\n== verdict")
v, ref = rep["verdict"], rep["reference"]
print(f"[INFO] noise ceiling, val drugs: median r_max {rep['r_max_median_val']:.3f} "
      f"(signal-bearing {rep['r_max_median_val_signal']:.3f}); all drugs {rep['r_max_median_all']:.3f}")
print(f"[INFO] drug val r: B_mean {rep['baselines']['drug:B_mean']['val']:+.3f}, "
      f"B_class {rep['baselines']['drug:B_class']['val']:+.3f}, B_target {rep['baselines']['drug:B_target']['val']:+.3f} "
      f"| model C4 {ref.get('model_C4_drug', float('nan')):+.3f}, stage A {ref.get('model_stageA_drug', float('nan')):+.3f}, "
      f"C0 {ref.get('model_C0_drug', float('nan')):+.3f}")
print(f"[INFO] mean-of-other-drugs r: shared DMSO {rep['shared_reference']['r_shared_median']:+.3f}, "
      f"disjoint DMSO {rep['shared_reference']['r_disjoint_median']:+.3f}  (difference = shared-reference artefact)")
print(f"[INFO] GEO val r: B_mean {rep['baselines']['geo:B_mean']['val']:+.3f} vs model C0 {ref.get('model_C0_geo', float('nan')):+.3f}")
print(f"[INFO] rule 1 constant profile >= C4: {v['constant_beats_or_ties_C4']}; "
      f"common response (disjoint) >= C4: {v['common_response_beats_C4']}")
print(f"[INFO] rule 2 ceiling < 0.40: {v['ceiling_low']}; C4 / ceiling = {v['C4_fraction_of_ceiling']:.2f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
