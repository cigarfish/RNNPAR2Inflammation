"""Step 18 unit test: the three ways of treating unsigned (free-sign) readout edges.

CPU, no training (~1 min):
    python tests/test_step18_readout_param.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.loss import Priors
from lembas_par2.model import build_model

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


def model(unsigned_init=None, seed=0):
    cfg = load_config()
    cfg["model"]["readout_unsigned_init"] = unsigned_init
    m = build_model(cfg, device="cpu", seed=seed)
    return cfg, m


print("== the problem this step addresses")
cfg, m = model()
W = m.readout_weights().detach()
n_free, n_tot = len(m.r_free_idx), len(m.readout)
check("unsigned edges are a small minority but the only freely-moving ones",
      n_free / n_tot < 0.05 and m.W_free.requires_grad,
      f"{n_free}/{n_tot} = {n_free / n_tot:.1%} unsigned")
ro = m.readout
for g in ["Cgas", "Sting1"]:
    sub = ro[ro.gene == g]
    check(f"{g} readout is dominated by unsigned edges (the claim's own readout)",
          (sub.sign == 0).mean() >= 0.5, f"{int((sub.sign == 0).sum())}/{len(sub)} unsigned")

print("\n== R1: zero initialisation")
_, m0 = model(0.0)
W0 = m0.readout_weights().detach()
check("unsigned edges start at exactly zero", float(W0[m0.r_free_idx].abs().max()) == 0.0)
check("signed edges are untouched by that option",
      torch.allclose(W0[m0.r_locked_idx], W[m.r_locked_idx]))
check("unsigned edges can still move away from zero (not frozen)",
      m0.W_free.requires_grad and m0.W_free.numel() == n_free)
m0.zero_grad()
m0.readout_weights()[m0.r_free_idx].sum().backward()
check("gradient reaches the zero-initialised unsigned edges",
      m0.W_free.grad is not None and float(m0.W_free.grad.abs().min()) > 0)

print("\n== R2: learning-rate multiplier (uses the existing Trainer mechanism)")
from lembas_par2.loss import Targets
from lembas_par2.train import Trainer, all_trainable
cfg2, m2 = model()
m2.initialize()
T, P = Targets(cfg2, m2), Priors(cfg2, m2)
tr = Trainer(m2, T, P, mask=all_trainable(m2), lr=0.01, lr_mult={"W_free": 0.11})
groups = {n: g for n, g in zip([k for k, _ in m2.named_parameters()], tr.opt.param_groups)}
check("only W_free gets the reduced multiplier",
      abs(groups["W_free"]["mult"] - 0.11) < 1e-12
      and all(abs(g["mult"] - 1.0) < 1e-12 for n, g in groups.items() if n != "W_free"))

print("\n== R3: extra L1 on unsigned edges only")
cfg3, m3 = model()
m3.initialize()
Pa, Pb = Priors(cfg3, m3), Priors(cfg3, m3)
Pb.configure(readout_free_mult=10.0)
ta, tb = Pa.terms(m3), Pb.terms(m3)
check("mult = 1 reproduces the baseline prior exactly",
      "readout_l1_free" not in ta and abs(float(Pa.weighted(ta)) - float(Priors(cfg3, m3).weighted(ta))) < 1e-12)
check("mult > 1 adds a term and raises the penalty", "readout_l1_free" in tb
      and float(Pb.weighted(tb)) > float(Pa.weighted(ta)),
      f"{float(Pa.weighted(ta)):.6f} -> {float(Pb.weighted(tb)):.6f}")
g = torch.autograd.grad(Pb.weighted(Pb.terms(m3)), [m3.W_free, m3.W_locked], allow_unused=True)
ga, gl = (float(x.abs().mean()) if x is not None else 0.0 for x in g)
check("the extra penalty pushes on unsigned edges harder than on signed ones", ga > gl,
      f"mean |grad| free {ga:.3e} vs locked {gl:.3e}")

print("\n== config")
rt = cfg.readout_param_test
check("R0-R4 defined and R0 is the untouched baseline",
      set(rt["variants"]) == {"R0", "R1", "R2", "R3", "R4"} and cfg.source_variants["R0"] == {})
check("the target metric must at least halve and the fit guard is set",
      rt["target_reduction"] == 0.5 and set(rt["noninferiority"]) == {"r_tf_drug_resid_signal", "r_tf_geo_resid"})
check("the holdout readouts are excluded from selection in writing",
      "NOT selection criteria" in rt["selection"] and "Holdouts are NOT" in rt["selection"])

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
