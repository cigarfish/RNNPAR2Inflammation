"""Step 12j unit test: the per-source readout nuisance in lembas_par2/loss.py.

No training; CPU is fine (~30 s):
    python tests/test_step12j.py
"""
import copy
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
dev = "cpu"
torch.manual_seed(0)
m = build_model(cfg, device=dev, seed=0)
m.initialize()
cond = m.design.cond
src = cond.source.values
vf = cfg.stage_b["val_fold"]
val = [i for i in range(len(cond)) if src[i] in ("drug", "geo") and cond.fold.iloc[i] == vf]
train = [i for i in range(len(cond)) if src[i] in ("drug", "geo", "perturb") and i not in set(val)]
drug_tr = [i for i in train if src[i] == "drug"]
with torch.no_grad():
    # give the untrained network a visible response so the gains are not degenerate
    m.alpha_raw.add_(5.0)


def T_new(**kw):
    T = Targets(cfg, m)
    T.lambdas = dict(tf=1.0, gene=1.0, direction=0.0)
    if kw:
        T.configure_nuisance(**kw)
    return T


def L(T, pos, part="total"):
    with torch.no_grad():
        return float(data_loss(m(conditions=pos), T)[part])


# ---------- 1. guards ----------
print("== guards")
T = T_new(intercept=("drug",))
try:
    L(T, train)
    ok = False
except RuntimeError:
    ok = True
check("data_loss refuses an unfitted nuisance", ok)
try:
    T_new(intercept=("perturb",))
    ok = False
except ValueError:
    ok = True
check("intercept only for drug / geo", ok)
sv = cfg.source_variants
check("config: J0-J4 defined, GEO guard 0.15, gene intercept on",
      set(sv) >= {"J0", "J1", "J2", "J3", "J4"} and cfg.source_test["geo_resid_min"] == 0.15
      and cfg.source_test["gene_intercept"] is True)

# ---------- 2. reduces to the S5 readout ----------
print("\n== no intercept, global gain vs the step-12e gain_global on the same conditions")
# step-12e's gain is the UNWEIGHTED least-squares fit over the batch; the loss weights each
# condition by w_c / n_tf_c. The nuisance fit uses the loss's own weights, so it is the
# loss minimiser: never worse, and close.
batch = train[:40]
T0 = T_new().configure_tf("gain_global", True)
T1 = T_new(intercept=(), gain="global")
T1.refit_nuisance(m, batch)
a, b = L(T0, batch, "tf"), L(T1, batch, "tf")
check("weighted fit never worse than gain_global, within 1 %", b <= a + 1e-7 and (a - b) < 0.01 * a,
      f"{a:.6f} vs {b:.6f}; gain {T0.last_tf_gain:.4f} (unweighted) vs {T1.last_tf_gain:.4f} (weighted)")

# ---------- 3. the fit is the least-squares optimum ----------
print("\n== closed form is optimal (gradient of the loss in a and mu ~ 0)")
for spec in [dict(intercept=("drug",)), dict(intercept=("drug", "geo")), dict(intercept=("drug", "geo"), gain="source")]:
    T = T_new(**spec)
    T.refit_nuisance(m, train)
    a_, mu_, mg_ = (T.a_src.clone().requires_grad_(True), T.mu_tf.clone().requires_grad_(True),
                    T.mu_gene.clone().requires_grad_(True))
    T.a_src, T.mu_tf, T.mu_gene = a_, mu_, mg_
    with torch.no_grad():
        out = m(conditions=train)
    D = data_loss(out, T)
    ga, gm, gg = torch.autograd.grad(D["tf"], [a_, mu_]) + torch.autograd.grad(D["gene"], [mg_])
    used = [k for k, s_ in enumerate(T.SOURCES) if s_ in spec["intercept"]]
    # a global gain is one parameter shared by both entries: its gradient is their sum
    g_gain = ga[:2].abs().max() if spec.get("gain") == "source" else ga[:2].sum().abs()
    rel = max(float(g_gain * T.a_src[:2].detach().abs().max()) / max(float(D["tf"]), 1e-12),
              float(gm[used].abs().max()) / max(float(D["tf"]), 1e-12),
              float(gg[used].abs().max()) / max(float(D["gene"]), 1e-12))
    check(f"{spec}: |dL/d(a, mu_tf, mu_gene)| (scale-free) < 1e-4", rel < 1e-4, f"{rel:.1e}")

# ---------- 4. the point of 12j: a constant per-source profile is invisible ----------
print("\n== a constant profile added to every drug target changes nothing (with mu_drug)")
gen = torch.Generator().manual_seed(1)
c_tf = torch.randn(T.tf.shape[0], generator=gen, dtype=T.tf.dtype) * 2.0
c_gene = torch.randn(T.gene.shape[0], generator=gen, dtype=T.gene.dtype) * 0.5


def loss_and_grad(T, pos):
    m.zero_grad(set_to_none=True)
    D = data_loss(m(conditions=pos), T)
    (D["tf"] + D["gene"]).backward()
    g = torch.cat([p.grad.flatten() for p in [m.theta_locked, m.theta_free, m.bias] if p.grad is not None])
    return float(D["tf"] + D["gene"]), g.clone()


for intercept, expect_same in [(("drug",), True), ((), False)]:
    Ta = T_new(intercept=intercept) if intercept else T_new().configure_tf("gain_global", True)
    Tb = T_new(intercept=intercept) if intercept else T_new().configure_tf("gain_global", True)
    idx = torch.tensor(drug_tr)
    Tb.tf[:, idx] = Tb.tf[:, idx] + c_tf[:, None] * Tb.tf_mask[:, idx]
    Tb.gene[:, idx] = Tb.gene[:, idx] + c_gene[:, None] * Tb.gene_mask[:, idx]
    if intercept:
        Ta.refit_nuisance(m, drug_tr)
        Tb.refit_nuisance(m, drug_tr)
    la, ga = loss_and_grad(Ta, drug_tr)
    lb, gb = loss_and_grad(Tb, drug_tr)
    gdiff = float((ga - gb).norm() / ga.norm().clamp(min=1e-30))
    if expect_same:
        check("with mu_drug: loss and network gradient unchanged", abs(la - lb) < 1e-5 * la and gdiff < 1e-4,
              f"loss {la:.5f} vs {lb:.5f}, gradient rel. diff {gdiff:.1e}")
    else:
        check("without intercept: the same shift changes loss and gradient (control)", abs(la - lb) > 1e-3 * la and gdiff > 1e-2,
              f"loss {la:.5f} vs {lb:.5f}, gradient rel. diff {gdiff:.1e}")

# ---------- 5. no leakage, gain constraint ----------
print("\n== validation never enters the fit; gain >= 0")
Ta, Tb = T_new(intercept=("drug", "geo"), gain="source"), T_new(intercept=("drug", "geo"), gain="source")
Tb.tf[:, torch.tensor(val)] = torch.randn(Tb.tf.shape[0], len(val), generator=gen, dtype=Tb.tf.dtype) * 5
Tb.gene[:, torch.tensor(val)] = 0.0
Ta.refit_nuisance(m, train)
Tb.refit_nuisance(m, train)
check("mu and a unchanged when validation targets are replaced",
      torch.equal(Ta.mu_tf, Tb.mu_tf) and torch.equal(Ta.a_src, Tb.a_src) and torch.equal(Ta.mu_gene, Tb.mu_gene))
Tf = T_new(intercept=("drug",), gain="source")
pos_a = Ta.a_src[0].item()
Tf.tf = -Tf.tf
Tf.refit_nuisance(m, train)
w = (Tf.weight[drug_tr] * Tf.has_matrix[drug_tr])
mm = Tf.tf_mask[:, drug_tr]
v = mm * (w / mm.sum(0).clamp(min=1))
ybar = (v * Tf.tf[:, drug_tr]).sum(1) / v.sum(1).clamp(min=1e-20)
check("anti-correlated targets: gain clamped to 0 and mu = weighted target mean",
      pos_a > 0 and Tf.a_src[0].item() == 0.0 and torch.allclose(Tf.mu_tf[0], ybar, atol=1e-5),
      f"a_drug {pos_a:.3f} -> {Tf.a_src[0].item():.3f}")

# ---------- 6. source balancing ----------
print("\n== per-source balancing")
Tbal = T_new(intercept=("drug", "geo"))
Tbal.refit_nuisance(m, train)
W0 = float(Tbal.weight[train].sum())
fac = Tbal.balance_sources(m, train)
Tbal.refit_nuisance(m, train)
contrib = {}
for k, s_ in enumerate(Tbal.SOURCES):
    cols = [i for i in train if src[i] == s_]
    contrib[s_] = float(Tbal.weight[cols].sum()) * L(Tbal, cols, "total")
check("drug and GEO contribute equally after balancing (before the post-balance refit moves mu)",
      abs(fac["drug"]["factor"] * fac["drug"]["weight"] * fac["drug"]["loss"]
          - fac["geo"]["factor"] * fac["geo"]["weight"] * fac["geo"]["loss"]) < 1e-6 * contrib["drug"],
      {k: round(v["factor"], 3) for k, v in fac.items()})
check("total training weight preserved", abs(float(Tbal.weight[train].sum()) - W0) < 1e-4 * W0)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
