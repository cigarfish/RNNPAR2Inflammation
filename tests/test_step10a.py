"""Step 10a validation: data loss terms (TF activity, gene readout, perturbation directions).

Run from the LEMBAS folder:
    python tests/test_step10a.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.loss import Targets, data_loss
from lembas_par2.model import build_model

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = build_model(cfg, device=dev)
m.initialize()
T = Targets(cfg, m)
# this step's exact values are defined for the raw-MSE TF term; the configured default
# is whatever step 12e selected, so pin the mode here and report the default separately
configured_tf_mode = T.tf_mode
T.configure_tf("absolute")
cond = m.design.cond
C, n = m.design.C, len(m.nodes)
allpos = list(range(C))

# ---------- 1. alignment ----------
print("== alignment")
check("TF targets: output TFs x conditions", T.tf.shape == (len(m.tf_outputs), C), tuple(T.tf.shape))
check("gene targets: readout genes x conditions", T.gene.shape == (len(m.genes), C), tuple(T.gene.shape))
src = cond.source.values
hm = T.has_matrix.cpu().numpy()
check("measured matrices exist exactly for drug and GEO conditions", (hm == np.isin(src, ["drug", "geo"])).all(),
      f"{int(hm.sum())} conditions")
check("holdout conditions have weight 0", bool((T.weight.cpu().numpy()[src == "holdout"] == 0).all()))
check("TF targets clipped", T.tf.abs().max().item() <= cfg.loss["tf_target_clip"] + 1e-6, f"max |t| {T.tf.abs().max().item():.1f}")
print(f"[INFO] direction constraints used: {len(T.d_sign)} (signal {int((~T.d_mrna).sum())}, mrna {int(T.d_mrna.sum())}); "
      f"dropped {len(T.dropped_directions)}")
print(f"[INFO] configured TF mode is '{configured_tf_mode}' (step 12e); this test pins 'absolute' "
      f"for its hand-computed values")
check("no holdout constraint among direction targets", not set(T.d_cond.cpu().numpy()) & set(np.where(src == "holdout")[0]))

# ---------- 2. hand-computed values ----------
print("\n== hand-computed values")
def fake(tf_shift=0.0, gene_shift=0.0, delta=None, pos=allpos):
    p = torch.tensor(pos, device=dev)
    return dict(conditions=pos, tf_pred=T.tf[:, p] + tf_shift, gene_pred=T.gene[:, p] + gene_shift,
                delta=torch.zeros(n, len(pos), dtype=m.dtype, device=dev) if delta is None else delta)
L = data_loss(fake(), T)
check("perfect TF and gene predictions give 0", L["tf"].item() < 1e-10 and L["gene"].item() < 1e-10)
L = data_loss(fake(1.0, 1.0), T)
check("predictions off by 1 everywhere give TF and gene loss exactly 1", abs(L["tf"].item() - 1) < 1e-5 and abs(L["gene"].item() - 1) < 1e-5,
      f"{L['tf'].item():.6f}, {L['gene'].item():.6f}")
signed = T.d_sign != 0
exp = (T.d_weight[signed].sum() * T.margin ** 2 / T.d_weight.sum()).item()
gp0 = T.gene.clone()
gp0[T.d_index[T.d_mrna], T.d_cond[T.d_mrna]] = 0.0          # mrna constraints read 0 too
L0 = data_loss(dict(conditions=allpos, tf_pred=T.tf, gene_pred=gp0,
                    delta=torch.zeros(n, C, dtype=m.dtype, device=dev)), T)
check("zero deltas: direction loss = weighted share of signed constraints x margin^2 (signal + mrna at 0)",
      abs(L0["direction"].item() - exp) < 1e-7, f"{L0['direction'].item():.3e} vs {exp:.3e}")
# satisfy every signal constraint with a comfortable margin, set mrna predictions too
delta = torch.zeros(n, C, dtype=m.dtype, device=dev)
gp = T.gene.clone()
for c_, i_, mr, s_ in zip(T.d_cond.tolist(), T.d_index.tolist(), T.d_mrna.tolist(), T.d_sign.tolist()):
    if mr:
        gp[i_, c_] = s_ * (T.margin + 0.1)
    else:
        delta[i_, c_] = s_ * (T.margin + 0.1)
out = dict(conditions=allpos, tf_pred=T.tf, gene_pred=gp, delta=delta)
L1 = data_loss(out, T)
uc = (T.d_sign == 0)
check("all constraints satisfied (unchanged ones at 0) give direction loss 0", L1["direction"].item() < 1e-10,
      f"{L1['direction'].item():.3e} ({int(uc.sum())} 'unchanged' constraints)")
delta_wrong = -delta
L2 = data_loss(dict(conditions=allpos, tf_pred=T.tf, gene_pred=-gp, delta=delta_wrong), T)
check("violated constraints are penalised", L2["direction"].item() > L1["direction"].item() + 1e-3,
      f"{L2['direction'].item():.3e} vs {L1['direction'].item():.3e}")
hpos = list(np.where(src == "holdout")[0])
Lh = data_loss(fake(5.0, 5.0, pos=hpos), T)
check("a holdout-only batch contributes nothing", Lh["total"].item() == 0.0)
fold0 = [i for i in allpos if cond.fold.iloc[i] == 0]
Lf = data_loss(fake(1.0, 0.0, pos=fold0), T)
check("subset (CV fold 0) loss computed on its own conditions", abs(Lf["tf"].item() - 1) < 1e-5, f"{len(fold0)} conditions")

# ---------- 3. real forward ----------
print("\n== initialised model")
m.zero_grad()
o = m()
L = data_loss(o, T)
check("all terms finite and positive", all(torch.isfinite(v).item() and v.item() > 0 for v in L.values()),
      ", ".join(f"{k} {v.item():.4f}" for k, v in L.items()))
L["total"].backward()
g = {k: (p.grad.abs().mean().item() if p.grad is not None else 0.0) for k, p in m.named_parameters()}
check("every parameter group receives gradient from the data loss", all(v > 0 for v in g.values()),
      ", ".join(f"{k} {v:.1e}" for k, v in g.items()))
with torch.no_grad():
    v_sig = o["delta"][T.d_index.clamp(max=n - 1), T.d_cond]
    v_mr = o["gene_pred"][T.d_index.clamp(max=len(m.genes) - 1), T.d_cond]
    v = torch.where(T.d_mrna, v_mr, v_sig)
    sat = ((T.d_sign != 0) & (torch.sign(v) == T.d_sign)).sum().item() / (T.d_sign != 0).sum().item()
print(f"[INFO] at initialisation {sat:.0%} of up/down perturbation constraints already have the right sign")

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 10a:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
