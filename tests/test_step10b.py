"""Step 10b validation: priors, regularisation, route term, modes, full loss, short training.

Run from the LEMBAS folder:
    python tests/test_step10b.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.core import perron_radius
from lembas_par2.loss import Priors, Targets, balance_data_terms, data_loss, full_loss
from lembas_par2.model import build_model

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def fresh(mode=None):
    m = build_model(cfg, device=dev)
    m.initialize()
    T = Targets(cfg, m)
    P = Priors(cfg, m, mode=mode)
    return m, T, P


m, T, P = fresh()

# ---------- 1. balance ----------
print("== data-term balance")
lam = balance_data_terms(m, T)
with torch.no_grad():
    D = data_loss(m(), T)
check("each weighted data term (lambda x term) = 1 at initialisation after balancing",
      all(abs(lam[k] * D[k].item() - 1) < 1e-4 for k in ["tf", "gene", "direction"]) and abs(D["total"].item() - 3) < 1e-3,
      f"lambdas tf {lam['tf']:.3f}, gene {lam['gene']:.3f}, direction {lam['direction']:.1f}")

# ---------- 2. prior terms at initialisation ----------
print("\n== prior terms at initialisation")
with torch.no_grad():
    R = P.terms(m)
check("anchors are 0 at the starting weights", R["anchor_T0"].item() < 1e-10 and R["anchor_T1"].item() < 1e-10)
check("relative L1 terms are 1 at the start", abs(R["l1_T2"].item() - 1) < 1e-4 and abs(R["l1_T3"].item() - 1) < 1e-4
      and abs(R["readout_l1"].item() - 1) < 1e-4)
check("bias term 0, spectral term 0 (rho 0.9 < rho_max)", R["bias"].item() < 1e-12 and R["spectral"].item() == 0,
      f"rho {R['rho'].item():.3f}")
check("route constraint satisfied at initialisation (MEKi raises Bax/Bak1/mtDNA, lowers ERK)", R["route"].item() < 1e-10,
      f"{R['route'].item():.2e}")
print("[INFO] " + ", ".join(f"{k} {v.item():.4f}" for k, v in R.items()))


# ---------- 3. gradient routing ----------
print("\n== gradient routing (which parameters each term moves)")
def grads_of(term_fn):
    m.zero_grad()
    term_fn().backward()
    return {k: (p.grad.abs() > 0) if p.grad is not None else torch.zeros_like(p, dtype=torch.bool)
            for k, p in m.named_parameters()}


with torch.no_grad():
    m.theta_locked.add_(0.05 * torch.randn_like(m.theta_locked))      # move off the anchor
g = grads_of(lambda: P.terms(m)["anchor_T0"])
t0_locked = torch.tensor(m.tier[m.locked_idx.cpu().numpy()] == "T0", device=dev)
check("T0 anchor moves only T0 weights", bool(g["theta_locked"][t0_locked].all() and not g["theta_locked"][~t0_locked].any())
      and not g["theta_free"].any() and not g["bias"].any())
g = grads_of(lambda: P.terms(m)["l1_T3"])
t3_free = torch.tensor(m.tier[m.free_idx.cpu().numpy()] == "T3", device=dev)
check("T3 L1 moves only T3 weights", bool(g["theta_free"][t3_free].all() and not g["theta_free"][~t3_free].any())
      and not g["theta_locked"].any())
g = grads_of(lambda: perron_radius(m.struct, m.weights(), max_iter=3000, tol=1e-6))
check("stability gradient only on edges inside feedback loops",
      bool(g["theta_free"].any()) and not g["bias"].any(), "route edges (acyclic) excluded")
g = grads_of(lambda: P.route_delta(m)[P.r_up].sum())
check("route term reaches edge weights and biases", bool(g["theta_locked"].any() and g["bias"].any()))
m, T, P = fresh()
balance_data_terms(m, T)

# ---------- 4. modes ----------
print("\n== anchored vs unanchored")
mu, Tu, Pu = fresh("unanchored")
check("unanchored: T0 anchored like T1 and no route term",
      Pu.lam_anchor["T0"] == Pu.lam_anchor["T1"] and Pu.lam["route"] == 0 and P.lam["route"] > 0
      and P.lam_anchor["T0"] > P.lam_anchor["T1"], f"anchored {P.lam_anchor}, route {P.lam['route']}")
del mu, Tu, Pu

# ---------- 5. full loss + short training ----------
print("\n== short training on a small batch")
cond = m.design.cond
pick = [int(np.where(cond.name == "Pimasertib")[0][0]), int(np.where(cond.name == "Ruxolitinib")[0][0]),
        int(np.where(cond.source == "geo")[0][0]), int(np.where(cond.source == "perturb")[0][0])]
signs0 = torch.sign(m.weights()[m.locked_idx]).detach().clone()
opt = torch.optim.Adam(m.parameters(), lr=0.01)
hist, info, t0 = [], {}, time.time()
for step in range(80):
    opt.zero_grad()
    total, terms, _ = full_loss(m, T, P, conditions=pick, info=info)
    total.backward()
    opt.step()
    hist.append({k: float(v) for k, v in terms.items()} | {"total": total.item()})
el = time.time() - t0
h0, h1 = hist[0], hist[-1]
print(f"[INFO] {len(hist)} steps in {el:.1f} s; total {h0['total']:.3f} -> {h1['total']:.3f}; data {h0['data']:.3f} -> {h1['data']:.3f}; "
      f"TF {h0['data_tf']:.3f} -> {h1['data_tf']:.3f}, gene {h0['data_gene']:.3f} -> {h1['data_gene']:.3f}, "
      f"direction {h0['data_direction']:.3f} -> {h1['data_direction']:.3f}; prior {h0['prior']:.4f} -> {h1['prior']:.4f}")
check("total loss decreases", h1["total"] < 0.8 * h0["total"], f"{h0['total']:.3f} -> {h1['total']:.3f}")
check("data loss decreases on the batch", h1["data"] < 0.8 * h0["data"])
check("curated signs never flip during training", torch.equal(torch.sign(m.weights()[m.locked_idx]).detach(), signs0))
with torch.no_grad():
    rho = perron_radius(m.struct, m.weights(), max_iter=3000, tol=1e-6).item()
check("stability kept (rho(|A|) <= rho_max + 0.02)", rho <= cfg.loss["rho_max"] + 0.02, f"rho {rho:.3f}")
check("forward still converging", info["last_forward"]["converged"] and info["last_backward"]["converged"],
      f"forward {info['last_forward']['iterations']} it, backward {info['last_backward']['iterations']} it")
print(f"[INFO] drug kappa after training: Pimasertib {m.design.kappa()[m.design.drugs.index('Pimasertib')].item():.2f}, "
      f"Ruxolitinib {m.design.kappa()[m.design.drugs.index('Ruxolitinib')].item():.2f}")

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 10b:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
