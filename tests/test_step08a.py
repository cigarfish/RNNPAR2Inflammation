"""Step 8a validation: design encoding (inputs U, inhibition S, learnable drug kappa).

Run from the LEMBAS folder:
    python tests/test_step08a.py
Uses outputs/step05_signalling_edges.csv.gz and outputs/step06_conditions.csv.
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
from lembas_par2.core import Structure, simulate
from lembas_par2.perturb import Design

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
pc = cfg.perturbation
DT = torch.float64

# ---------- 1. toy design ----------
print("== toy design")
nodes = ["A", "B", "C", "D"]
toy = pd.DataFrame([
    dict(condition_id="drugX", source="drug", name="X", ref_inputs=["A"], trt_inputs=["A"], trt_inhibit=["B"]),
    dict(condition_id="toyKO", source="perturb", name="ko", ref_inputs=["A"], trt_inputs=["A"], trt_inhibit=["C"]),
    dict(condition_id="geoD", source="geo", name="stim", ref_inputs=[], trt_inputs=["D"], trt_inhibit=[]),
])
d = Design(cfg, toy, nodes, dtype=DT)
U = d.U
check("U: input_level at inputs of each arm",
      U[0, 0] == pc["input_level"] and U[0, 1] == pc["input_level"] and U[3, 5] == pc["input_level"]
      and U.sum() == 5 * pc["input_level"], U.numpy().tolist())
S = d.S()
check("S: drug arm 1 - kappa_init, KO arm 1 - 0.95, everything else 1",
      abs(S[1, 1].item() - (1 - pc["drug_kappa_init"])) < 1e-12 and abs(S[2, 3].item() - 0.05) < 1e-12
      and (S == 1).sum().item() == S.numel() - 2)
check("reference arms never inhibited", bool((S[:, d.ref_cols()] == 1).all()))
check("kappa in (0, 1) and learnable", bool(((d.kappa() > 0) & (d.kappa() < 1)).all()) and d.theta.requires_grad)
X = torch.rand(4, 6, dtype=DT)
check("delta = treated - reference", torch.allclose(d.delta(X), X[:, 1::2] - X[:, 0::2]))
check("subset returns both arms of selected conditions", d.subset(["toyKO"]).tolist() == [2, 3])
try:
    Design(cfg, toy, ["A", "B", "C"])
    check("missing node is rejected", False)
except ValueError:
    check("missing node is rejected", True)

# gradient through S -> simulate -> loss w.r.t. drug theta (finite differences)
st = Structure.build([0, 1, 2, 0], [1, 2, 3, 2], 4)
w = torch.tensor([0.6, 0.7, 0.5, -0.4], dtype=DT)
b = torch.zeros(4, dtype=DT)
def loss_of(theta):
    d.theta = theta
    X = simulate(st, w, b, d.U, d.S(), max_iter=5000, tol=1e-13)
    return d.delta(X)[3, 0]          # effect of drug X (inhibits B) on D
th = torch.full((1,), d.theta.item(), dtype=DT, requires_grad=True)
L = loss_of(th); L.backward()
h = 1e-6
fd = (loss_of(torch.full((1,), th.item() + h, dtype=DT)).item() - loss_of(torch.full((1,), th.item() - h, dtype=DT)).item()) / (2 * h)
check("d(effect)/d(theta) matches finite differences", abs(th.grad.item() - fd) < 1e-7, f"{th.grad.item():.6e} vs {fd:.6e}")
check("inhibiting an activator lowers the downstream node", L.item() < 0, f"delta D = {L.item():+.4f}")

# ---------- 2. real design ----------
print("\n== real design")
out = cfg.paths.output_dir
E = pd.read_csv(out / "step05_signalling_edges.csv.gz", usecols=["source", "target"], low_memory=False)
real_nodes = sorted(set(E.source) | set(E.target))
cond = pd.read_csv(out / "step06_conditions.csv")
D = Design(cfg, cond, real_nodes)
check("two arms per condition", D.U.shape == (len(real_nodes), 2 * len(cond)), tuple(D.U.shape))
inh = D.inhibition
print(f"[INFO] inhibitions by kind: {inh.kind.value_counts().to_dict()}")
check("one learnable kappa per drug", len(D.drugs) == (cond.source == "drug").sum(), f"{len(D.drugs)} drugs")
c = D.cond
lps = {D.idx[x] for x in cfg.conditions["lps_inputs"]}
drug_pos = np.where(c.source == "drug")[0]
U = D.U
check("drug arms: LPS inputs in both arms, nothing else",
      all(set(torch.nonzero(U[:, 2 * i]).flatten().tolist()) == lps and set(torch.nonzero(U[:, 2 * i + 1]).flatten().tolist()) == lps
          for i in drug_pos))
S = D.S()
pim = int(np.where(c.name == "Pimasertib")[0][0])
mek = [D.idx["Map2k1"], D.idx["Map2k2"]]
check("Pimasertib treated arm inhibits Map2k1/2 with learnable kappa",
      torch.allclose(S[mek, 2 * pim + 1], torch.full((2,), 1 - pc["drug_kappa_init"])), S[mek, 2 * pim + 1].tolist())
h = c[c.source == "holdout"]
f2 = D.idx["F2rl1"]
hk = {r.condition_id: S[f2, 2 * i + 1].item() for i, r in h.iterrows()}
check("holdout PAR2 KO inhibits F2rl1 (PAR2_29 only since 2026-10-27; CRISPR KO 0.95)",
      set(hk) == {"PAR2_29_C1"} and abs(hk["PAR2_29_C1"] - 0.05) < 1e-6, hk)
check("geo chemical inhibitors use the geo_chemical kappa",
      set(inh[inh.kind == "geo_chemical"].node) == {"Lrp5", "Lrp6", "Pparg", "Egfr"})
check("U and S finite, S within [0, 1]", bool(torch.isfinite(U).all() and ((S >= 0) & (S <= 1)).all()))
inh.to_csv(out / "step08a_inhibitions.csv", index=False)

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 8a:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
