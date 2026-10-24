"""Exploration (not a pass/fail test): remedies for signal decay along the route.

For each setting, simulate under a reference input (LPS + PAR2 agonist):
    PAR2 KO : F2rl1 inhibited (kappa 0.95)
    MEKi    : Map2k1/2 inhibited (kappa 0.7)
and report the change at ERK (Mapk1) and at Sting1, their ratio (transmission),
whether all 15 route nodes move in the predicted direction, and how many nodes sit
in the leak (z < 0) or saturated (z > 2) region at baseline.

Settings vary
    network  : route only (77 T0 edges) | full signalling network (254k edges)
    weights  : T0 route weight g0; bulk (T1/T2/T3) rescaled so the stability
               criterion reaches 0.9 (only the bulk is rescaled)
    criterion: rho(|A|) (guaranteed) | signed rho(A) (original LEMBAS)
    bias     : flat 0.25 | centred at the reference input | centred at no input
    leak     : 0.01 | 0.05

Run from the LEMBAS folder:
    python tests/explore_08b_remedies.py
Writes outputs/explore_08b_remedies.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.core import Structure, center_biases, perron_radius, signed_radius, steady_state

cfg = load_config()
DT = torch.float64
E = pd.read_csv(cfg.paths.output_dir / "step05_signalling_edges.csv.gz",
                usecols=["source", "target", "sign", "tier"], low_memory=False)
EXPECT = {"Mapk1": -1, "Mapk3": -1, "Rps6ka1": -1, "Gsk3b": 1, "Bcl2l11": 1, "Bad": 1, "Mcl1": -1,
          "Bax": 1, "Bak1": 1, "SE_CYTO_MTDNA": 1, "Cgas": 1, "Sting1": 1, "Tbk1": 1, "Irf3": 1, "Ifnb1": 1}
REF_INPUTS = ["Tlr4", "Cd14", "F2rl1"]
rng = np.random.default_rng(0)


def setup(sub):
    nodes = sorted(set(sub.source) | set(sub.target))
    idx = {n: i for i, n in enumerate(nodes)}
    st = Structure.build(sub.source.map(idx).values, sub.target.map(idx).values, len(nodes))
    return nodes, idx, st


def weights(sub, g0, criterion, st):
    t0 = (sub.tier == "T0").values
    bulk = np.where(sub.tier == "T1", sub.sign * 0.5,
                    np.where(sub.tier == "T2", rng.choice([-1, 1], len(sub)) * 0.1,
                             rng.choice([-1, 1], len(sub)) * 0.01)).astype(float)
    route = np.where(t0, sub.sign * g0, 0.0)
    bulk = np.where(t0, 0.0, bulk)
    rho = (lambda w: perron_radius(st, w, max_iter=5000, tol=1e-8).item()) if criterion == "abs" else \
          (lambda w: signed_radius(st, w))
    W = lambda s: torch.tensor(route + s * bulk, dtype=DT)
    if rho(W(1.0)) <= 0.9:
        return W(1.0), 1.0, rho(W(1.0))
    lo, hi = 0.0, 1.0                         # bisection on the bulk scale
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rho(W(mid)) <= 0.9 else (lo, mid)
    return W(lo), lo, rho(W(lo))


def experiment(label, sub, g0=1.0, criterion="abs", bias="flat", leak=0.01):
    t = time.time()
    nodes, idx, st = setup(sub)
    w, s, r = weights(sub, g0, criterion, st)
    n = len(nodes)
    U = torch.zeros(n, 3, dtype=DT)
    for x in REF_INPUTS:
        if x in idx:
            U[idx[x], :] = 1.0
    S = torch.ones(n, 3, dtype=DT)
    S[idx["F2rl1"], 1] = 0.05
    S[idx["Map2k1"], 2] = S[idx["Map2k2"], 2] = 0.3
    if bias == "flat":
        b = torch.full((n,), 0.25, dtype=DT)
    else:
        Uc = U[:, :1] if bias == "centre_ref" else torch.zeros(n, 1, dtype=DT)
        b, _ = center_biases(st, w, Uc, leak=leak)
    sol = steady_state(st, w, b, U, S, leak=leak, max_iter=20000, tol=1e-12)
    Y = S * sol.X
    d = {"PAR2_KO": Y[:, 1] - Y[:, 0], "MEKi": Y[:, 2] - Y[:, 0]}
    z = sol.Z[:, 0]
    row = dict(setting=label, nodes=n, g0=g0, criterion=criterion, bias=bias, leak=leak, bulk_scale=round(s, 4),
               rho=round(r, 3), converged=sol.converged, leak_nodes=int((z < 0).sum()), sat_nodes=int((z > 2).sum()))
    for k, v in d.items():
        sgn_ok = all(np.sign(v[idx[g]].item()) == e for g, e in EXPECT.items())
        row.update({f"{k}_dMapk1": v[idx["Mapk1"]].item(), f"{k}_dSting1": v[idx["Sting1"]].item(),
                    f"{k}_ratio": abs(v[idx["Sting1"]].item() / v[idx["Mapk1"]].item()), f"{k}_signs_ok": sgn_ok})
    row["seconds"] = round(time.time() - t, 1)
    return row


T0 = E[E.tier == "T0"].reset_index(drop=True)
FULL = E.reset_index(drop=True)
rows = []
plan = [
    ("route | baseline (w0.5, flat bias)", T0, dict(g0=0.5)),
    ("route | g0=1", T0, dict(g0=1.0)),
    ("route | g0=1 + centre(ref)", T0, dict(g0=1.0, bias="centre_ref")),
    ("route | g0=1 + centre(basal)", T0, dict(g0=1.0, bias="centre_basal")),
    ("route | g0=1 + leak 0.05", T0, dict(g0=1.0, leak=0.05)),
    ("route | g0=1 + centre(ref) + leak 0.05", T0, dict(g0=1.0, bias="centre_ref", leak=0.05)),
    ("full  | g0=1, |A|, flat", FULL, dict(g0=1.0)),
    ("full  | g0=1, |A|, centre(ref)", FULL, dict(g0=1.0, bias="centre_ref")),
    ("full  | g0=1, |A|, centre(basal)", FULL, dict(g0=1.0, bias="centre_basal")),
    ("full  | g0=1, |A|, centre(ref), leak 0.05", FULL, dict(g0=1.0, bias="centre_ref", leak=0.05)),
    ("full  | g0=1, signed, centre(ref)", FULL, dict(g0=1.0, criterion="signed", bias="centre_ref")),
    ("full  | g0=1, signed, centre(ref), leak 0.05", FULL, dict(g0=1.0, criterion="signed", bias="centre_ref", leak=0.05)),
]
for label, sub, kw in plan:
    r = experiment(label, sub, **kw)
    rows.append(r)
    print(f"{label:46s} conv={r['converged']!s:5s} rho={r['rho']:.2f} bulk x{r['bulk_scale']:.3f} "
          f"leak/sat {r['leak_nodes']:4d}/{r['sat_nodes']:3d} | "
          f"KO: dMapk1 {r['PAR2_KO_dMapk1']:+.1e} dSting1 {r['PAR2_KO_dSting1']:+.1e} ratio {r['PAR2_KO_ratio']:.1e} signs {r['PAR2_KO_signs_ok']!s:5s} | "
          f"MEKi: ratio {r['MEKi_ratio']:.1e} signs {r['MEKi_signs_ok']!s:5s} ({r['seconds']} s)", flush=True)
pd.DataFrame(rows).to_csv(cfg.paths.output_dir / "explore_08b_remedies.csv", index=False)
