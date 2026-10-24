"""Step 8b validation: two-arm simulation effects (delta = treated - reference).

Run from the LEMBAS folder:
    python tests/test_step08b.py
Uses outputs/step05_signalling_edges.csv.gz for the route (T0) subgraph check.
"""
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.core import Structure, perron_radius, simulate
from lembas_par2.perturb import Design

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
DT = torch.float64
TIGHT = dict(max_iter=5000, tol=1e-12)


def run(nodes, edges, conds, bias=0.25, kappa=None):
    """edges: list of (src, tgt, w); conds: list of (id, source, ref_inputs, trt_inputs, trt_inhibit)."""
    idx = {n: i for i, n in enumerate(nodes)}
    st = Structure.build([idx[s] for s, _, _ in edges], [idx[t] for _, t, _ in edges], len(nodes))
    w = torch.tensor([x for _, _, x in edges], dtype=DT)
    cond = pd.DataFrame([dict(condition_id=c, source=s, name=c, ref_inputs=r, trt_inputs=t, trt_inhibit=i)
                         for c, s, r, t, i in conds])
    d = Design(cfg, cond, nodes, dtype=DT)
    if kappa is not None:
        d.theta = torch.logit(torch.full_like(d.theta, kappa))
    S = d.S()
    X = simulate(st, w, torch.full((len(nodes),), bias, dtype=DT), d.U, S, **TIGHT)
    return pd.DataFrame(d.delta(X, S).numpy(), index=nodes, columns=cond.condition_id)


sgn = lambda v, eps=1e-9: 0 if abs(v) < eps else (1 if v > 0 else -1)

# ---------- 1. toy networks ----------
print("== toy networks")
D = run(["A", "B", "C"], [("A", "B", 0.6), ("B", "C", 0.6)],
        [("koB", "perturb", ["A"], ["A"], ["B"]), ("same", "perturb", ["A"], ["A"], [])])
check("chain A->B->C, KO B: B (effective) down, C down, A unchanged",
      (sgn(D.at["B", "koB"]), sgn(D.at["C", "koB"]), sgn(D.at["A", "koB"])) == (-1, -1, 0), D["koB"].round(4).tolist())
check("identical arms give delta = 0 exactly", bool((D["same"] == 0).all()))

D = run(["A", "B", "C"], [("A", "B", 0.6), ("B", "C", -0.6)], [("koB", "perturb", ["A"], ["A"], ["B"])])
check("A->B-|C, KO B: C up (inhibitor removed)", sgn(D.at["C", "koB"]) == 1, f"{D.at['C', 'koB']:+.4f}")

D = run(["A", "B", "C", "D"], [("A", "B", 0.6), ("B", "D", 0.6), ("A", "C", 0.6), ("C", "D", -0.6)],
        [("koB", "perturb", ["A"], ["A"], ["B"]), ("koC", "perturb", ["A"], ["A"], ["C"])])
check("parallel paths: KO of activating branch lowers D, KO of inhibiting branch raises D",
      sgn(D.at["D", "koB"]) == -1 and sgn(D.at["D", "koC"]) == 1, f"{D.at['D', 'koB']:+.4f}, {D.at['D', 'koC']:+.4f}")

D = run(["A", "B"], [("A", "B", 0.6), ("B", "A", -0.6)], [("koB", "perturb", ["A"], ["A"], ["B"])])
check("negative feedback A->B-|A, KO B: A up (feedback removed)", sgn(D.at["A", "koB"]) == 1, f"{D.at['A', 'koB']:+.4f}")

D = run(["A", "B", "C"], [("A", "B", 0.6), ("B", "C", 0.6)], [("stim", "geo", [], ["A"], [])])
check("input (GEO-style stimulus vs none) raises downstream nodes",
      all(sgn(D.at[x, "stim"]) == 1 for x in "ABC"), D["stim"].round(4).tolist())

eff = [run(["A", "B", "C"], [("A", "B", 0.6), ("B", "C", 0.6)], [("drug", "drug", ["A"], ["A"], ["B"])], kappa=k).at["C", "drug"]
       for k in (0.2, 0.5, 0.8, 0.95)]
check("stronger drug kappa -> larger effect (monotonic)", all(a > b for a, b in zip(eff, eff[1:])),
      ", ".join(f"{x:+.4f}" for x in eff))

# ---------- 2. route (T0) subgraph: PAR2 KO and MEK inhibition ----------
print("\n== route subgraph (77 T0 edges, prior signs, weight 0.5, bias 0.25)")
E = pd.read_csv(cfg.paths.output_dir / "step05_signalling_edges.csv.gz", usecols=["source", "target", "sign", "tier"],
                low_memory=False)
T0 = E[E.tier == "T0"]
nodes = sorted(set(T0.source) | set(T0.target))
idx = {n: i for i, n in enumerate(nodes)}
st = Structure.build(T0.source.map(idx), T0.target.map(idx), len(nodes))
w = torch.tensor(0.5 * T0.sign.values, dtype=DT)
rho = perron_radius(st, w, max_iter=20000, tol=1e-13).item()
scale = min(1.0, 0.9 / rho) if rho > 0 else 1.0
print(f"[INFO] {len(nodes)} nodes, rho(|A|) = {rho:.3f} -> weights x {scale:.3f}")
edges = [(s, t, 0.5 * sg * scale) for s, t, sg in zip(T0.source, T0.target, T0.sign)]
D = run(nodes, edges, [
    ("PAR2_KO", "perturb", ["F2rl1"], ["F2rl1"], ["F2rl1"]),
    ("MEKi", "drug", ["F2rl1"], ["F2rl1"], ["Map2k1", "Map2k2"]),
], kappa=0.7)
expect = {"Mapk1": -1, "Mapk3": -1, "Rps6ka1": -1, "Gsk3b": 1, "Bcl2l11": 1, "Bad": 1, "Mcl1": -1,
          "Bax": 1, "Bak1": 1, "SE_CYTO_MTDNA": 1, "Cgas": 1, "Sting1": 1, "Tbk1": 1, "Irf3": 1, "Ifnb1": 1}
for cid in ["PAR2_KO", "MEKi"]:
    got = {k: sgn(D.at[k, cid], 1e-12) for k in expect}
    wrong = {k: f"{D.at[k, cid]:+.2e}" for k in expect if got[k] != expect[k]}
    check(f"{cid}: every route node moves in the predicted direction", not wrong, wrong if wrong else
          ", ".join(f"{k} {D.at[k, cid]:+.1e}" for k in ["Mapk1", "Bax", "SE_CYTO_MTDNA", "Sting1"]))
D.to_csv(cfg.paths.output_dir / "step08b_route_deltas.csv")

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 8b:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
