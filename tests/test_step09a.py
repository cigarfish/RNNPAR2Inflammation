"""Step 9a validation: model assembly (parameters, sign locks, forward outputs, gradients).

Run from the LEMBAS folder:
    python tests/test_step09a.py
Uses step 5 / step 6 outputs. Weights here are only made small enough to converge;
proper initialisation is step 9b.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.model import build_model, softplus_inv

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make(dev):
    m = build_model(cfg, device=dev)
    with torch.no_grad():                                   # small weights so the test converges
        t1 = torch.tensor(m.tier[m.locked_idx.cpu().numpy()] == "T1", device=dev)
        m.theta_locked[t1] = softplus_inv(torch.tensor(0.005)).to(dev, m.dtype)
        m.theta_free.mul_(0.05)
    return m


m = make(devices[-1])
dev = devices[-1]
print(f"device {dev}; {len(m.nodes)} nodes, {len(m.edges)} edges, {m.design.C} conditions")

# ---------- 1. parameters ----------
print("\n== parameters")
ps = m.parameter_summary()
print(ps.to_string(index=False))
want = m.edges.tier.astype(str).value_counts().to_dict()
check("signalling parameters per tier match the edge table",
      all(int(ps.set_index("group").loc["signalling " + t, "n"]) == n for t, n in want.items()), want)
check("one kappa per drug, one scale per TF output", m.kappa_theta.numel() == 57 and m.alpha_raw.numel() == len(m.tf_outputs))
n_params = sum(p.numel() for p in m.parameters())
print(f"[INFO] trainable parameters: {n_params:,}")

# ---------- 2. sign locks ----------
print("\n== sign locks")
locked = m.locked_idx.cpu().numpy()
prior = np.sign(m.edges.sign.values[locked])
with torch.no_grad():
    saved = m.theta_locked.clone()
    m.theta_locked.add_(torch.randn_like(m.theta_locked) * 5)
    ok = np.array_equal(np.sign(m.weights()[m.locked_idx].cpu().numpy()), prior)
    m.theta_locked.copy_(saved)
check("T0/T1 weights keep their curated sign under any theta", ok)
with torch.no_grad():
    saved = m.theta_free.clone()
    m.theta_free.copy_(-saved)
    flipped = (np.sign(m.weights()[m.free_idx].cpu().numpy()) == -np.sign(saved.cpu().numpy())).all()
    m.theta_free.copy_(saved)
check("T2/T3 weights can change sign", bool(flipped))
rl = m.r_locked_idx.cpu().numpy()
check("signed readout edges keep the regulon sign",
      np.array_equal(np.sign(m.readout_weights()[m.r_locked_idx].detach().cpu().numpy()), np.sign(m.readout.sign.values[rl])))
put = m.readout[m.readout.dbs == "putative_isg"].index
check("putative ISG readout edges are free (unsigned)", set(put) <= set(m.r_free_idx.cpu().numpy()))

# ---------- 3. forward ----------
print("\n== forward")
info = {}
o = m(info=info)
C, n, G = m.design.C, len(m.nodes), len(m.genes)
check("output shapes", o["X"].shape == (n, 2 * C) and o["delta"].shape == (n, C)
      and o["tf_pred"].shape == (len(m.tf_outputs), C) and o["gene_pred"].shape == (G, C),
      f"X {tuple(o['X'].shape)}, tf {tuple(o['tf_pred'].shape)}, genes {tuple(o['gene_pred'].shape)}")
check("forward converged, outputs finite", info["last_forward"]["converged"] and all(torch.isfinite(o[k]).all().item()
      for k in ["X", "delta", "tf_pred", "gene_pred"]), info["last_forward"])
c = int(np.where(m.design.cond.name == "Ruxolitinib")[0][0])
i_tf = m.tf_outputs.index("Stat1")
manual = (m.alpha()[i_tf] * o["delta"][m.idx["Stat1"], c]).item()
check("TF prediction = alpha * delta at the TF node (Stat1, Ruxolitinib)", abs(o["tf_pred"][i_tf, c].item() - manual) < 1e-6)
gi = m.genes.index("Isg15")
rows = np.where(m.readout.gene.values == "Isg15")[0]
W = m.readout_weights()
manual = sum(W[r].item() * o["delta"][m.idx[m.readout.tf.values[r]], c].item() for r in rows)
check("gene prediction = sum of W * delta over its TFs (Isg15)", abs(o["gene_pred"][gi, c].item() - manual) < 1e-6,
      f"{len(rows)} regulators")
sub = [3, 50, 120]
o2 = m(conditions=sub)
check("forward on a subset equals those columns of the full forward",
      torch.allclose(o2["tf_pred"], o["tf_pred"][:, sub], atol=1e-5) and torch.allclose(o2["gene_pred"], o["gene_pred"][:, sub], atol=1e-5))

# ---------- 4. gradients ----------
print("\n== gradients")
m.zero_grad()
o = m()
(o["tf_pred"].pow(2).mean() + o["gene_pred"].pow(2).mean()).backward()
grads = {name: p.grad.abs().sum().item() if p.grad is not None else 0.0 for name, p in m.named_parameters()}
check("every parameter group receives gradient", all(v > 0 for v in grads.values()),
      ", ".join(f"{k} {v:.1e}" for k, v in grads.items()))

# ---------- 5. devices ----------
if len(devices) == 2:
    mc = make("cpu")
    oc = mc()
    og = m()
    check("CPU and GPU predictions agree", torch.allclose(oc["tf_pred"], og["tf_pred"].cpu(), atol=1e-4)
          and torch.allclose(oc["gene_pred"], og["gene_pred"].cpu(), atol=1e-4),
          f"max diff {(oc['tf_pred'] - og['tf_pred'].cpu()).abs().max().item():.1e}")

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 9a:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
