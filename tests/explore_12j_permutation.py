"""Step 12j follow-up: is the residualised drug r (~0.04-0.07) drug-specific?

For each validation signal drug i, correlate the model's residualised prediction for drug i
with the residualised target of every OTHER validation drug j. If the matched r (i = j) is
no higher than this mismatched null, the "skill" is a component shared by all drugs (e.g.
the validation fold's own common response), not knowledge of which drug was given.

    python tests/explore_12j_permutation.py [J0 J3 J0_s1 ...]
Writes outputs/source_variants/permutation_null.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.eval_harness import SIGNAL_RMAX, _corr, load_model, split
from lembas_par2.loss import Targets

cfg = load_config()
names = sys.argv[1:] or ["J0", "J0_s1", "J0_s2", "J3", "J3_s1", "J3_s2"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
rmax = pd.read_csv(cfg.paths.output_dir / "step12f0" / "ceiling.csv").set_index("condition_id").r_max
rows = []
for n in names:
    ck = cfg.paths.output_dir / "source_variants" / n / "checkpoint.pt"
    if not ck.exists():
        print(f"[skip] {n}")
        continue
    m, _ = load_model(cfg, ck, dev)
    cond = m.design.cond
    ids, src = list(cond.condition_id), cond.source.values
    val, train, by = split(cfg, cond)
    vd = [i for i in val if src[i] == "drug" and float(rmax.get(ids[i], np.nan)) >= SIGNAL_RMAX]
    T = Targets(cfg, m)
    pos = sorted(set(vd) | set(by["drug"]))
    with torch.no_grad():
        P = m(conditions=pos)["tf_pred"].double().cpu().numpy()
    col = {p: k for k, p in enumerate(pos)}
    Y = T.tf.double().cpu().numpy()
    M = T.tf_mask.double().cpu().numpy() > 0
    Yn = np.where(M, Y, np.nan)
    Pn = {i: np.where(M[:, i], P[:, col[i]], np.nan) for i in pos}
    with np.errstate(all="ignore"):
        mu_y = np.nanmean(Yn[:, by["drug"]], axis=1)
        mu_p = np.nanmean(np.stack([Pn[i] for i in by["drug"]], 1), axis=1)
    # prediction residual: use drug i's prediction on the TFs measured for target j
    pr = {i: P[:, col[i]] - mu_p for i in vd}
    tr = {j: Yn[:, j] - mu_y for j in vd}
    R = np.array([[_corr(pr[i], tr[j]) for j in vd] for i in vd])
    matched = np.diag(R)
    off = R[~np.eye(len(vd), dtype=bool)]
    # permutation p: fraction of random drug relabellings whose median matched r >= observed
    rng = np.random.default_rng(0)
    obs = np.median(matched)
    perm = [np.median(R[np.arange(len(vd)), rng.permutation(len(vd))]) for _ in range(5000)]
    p = float(np.mean(np.array(perm) >= obs))
    rows.append(dict(run=n, n_drugs=len(vd), matched_median=float(obs), mismatched_median=float(np.median(off)),
                     matched_minus_mismatched=float(obs - np.median(off)), perm_p=p,
                     n_drugs_matched_gt_row_median=int((matched > np.median(R, axis=1)).sum())))
    print(f"{n:6s} matched {obs:+.3f}  mismatched {np.median(off):+.3f}  diff {obs - np.median(off):+.3f}  "
          f"perm p {p:.3f}  ({rows[-1]['n_drugs_matched_gt_row_median']}/{len(vd)} drugs above their own row median)")
d = pd.DataFrame(rows)
out = cfg.paths.output_dir / "source_variants" / "permutation_null.csv"
d.to_csv(out, index=False)
print(f"wrote {out}")
