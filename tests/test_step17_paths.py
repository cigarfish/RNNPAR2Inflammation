"""Step 17 validation: the path enumerator is correct on graphs whose answer is known by hand, and
the recorded run reproduces the figure already in REPORT.md.

    python scripts/run_paths_17.py
    python tests/test_step17_paths.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.paths import (enumerate_paths, mrna_paths, path_edge_matrix, sign_null,
                               signed_adjacency, summarise)

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


def toy(edges):
    adj = defaultdict(list)
    for i, (s, t, sg) in enumerate(edges):
        adj[s].append((t, sg, i))
    return adj


print("== enumerator on hand-checked graphs")
# one path, two negatives -> net +1 -> predicts DOWN
d, _, _ = enumerate_paths(toy([("F2rl1", "B", -1), ("B", "C", -1)]), {"C"}, 5)
check("two negative edges give net +1 (predicts DOWN)", len(d) == 1 and d.net_sign.iloc[0] == 1)
# one negative -> net -1 -> predicts UP
d, _, _ = enumerate_paths(toy([("F2rl1", "B", 1), ("B", "C", -1)]), {"C"}, 5)
check("one negative edge gives net -1 (predicts UP)", len(d) == 1 and d.net_sign.iloc[0] == -1)
# two routes of opposite net sign are both found
d, _, _ = enumerate_paths(toy([("F2rl1", "B", 1), ("B", "C", -1), ("F2rl1", "D", -1), ("D", "C", -1)]), {"C"}, 5)
check("both routes found with their own net signs", len(d) == 2 and set(d.net_sign) == {-1, 1})
# a cycle must not produce infinite or repeated-node paths
d, _, _ = enumerate_paths(toy([("F2rl1", "B", 1), ("B", "C", 1), ("C", "B", 1)]), {"C"}, 10)
check("cycles give simple paths only", len(d) == 1 and d.path.iloc[0] == "F2rl1->B->C")
# depth cap
chain = [("F2rl1", "A", 1), ("A", "B", 1), ("B", "C", 1), ("C", "T", 1)]
check("depth cap excludes longer paths", len(enumerate_paths(toy(chain), {"T"}, 3)[0]) == 0
      and len(enumerate_paths(toy(chain), {"T"}, 4)[0]) == 1)
# intermediate target is reported as well as the final one
d, _, _ = enumerate_paths(toy([("F2rl1", "M", -1), ("M", "T", 1)]), {"M", "T"}, 5)
check("a target that is also an intermediate node is reported", set(d.target) == {"M", "T"}
      and d[d.target == "M"].net_sign.iloc[0] == -1 and d[d.target == "T"].net_sign.iloc[0] == -1)
# summarise counts
s = summarise(pd.DataFrame(dict(target=["x"] * 3, depth=[1, 2, 3], net_sign=[-1, -1, 1], path=[""] * 3)))
check("summarise counts net-negative paths as 'predicts UP'", s["n_predict_up"] == 2 and s["n_paths"] == 3)
# mrna extension multiplies by the regulator's readout sign and adds a hop
regs = pd.DataFrame(dict(tf=["M"], curated_sign=[-1], sign_used_all=[-1]))
base = pd.DataFrame(dict(target=["M"], depth=[2], net_sign=[-1], path=["F2rl1->x->M"]))
mp = mrna_paths(base, regs, "curated_sign")
check("mRNA extension multiplies by the readout sign and adds a hop",
      mp.net_sign.iloc[0] == 1 and mp.depth.iloc[0] == 3)
# the sign null sits at 50 % on a graph with many paths
cfg = load_config()
adj, n_edges = signed_adjacency(cfg, ["T0"])
_d, _e, _c, _el = enumerate_paths(adj, {"Sting1"}, 20, return_edges=True)
null = sign_null(path_edge_matrix(_el, n_edges), n_perm=200, seed=0)
check("random signs give ~50 % 'up' paths", abs(np.nanmedian(null) - 0.5) < 0.1, f"{np.nanmedian(null):.3f}")
# the fast null must agree with re-signing the graph by brute force
import numpy as _np
rng = _np.random.default_rng(1)
Mt = path_edge_matrix(_el, n_edges)
sgn = rng.choice([-1, 1], n_edges)
brute = []
for es in _el:
    v = 1
    for e in es:
        v *= sgn[e]
    brute.append(v < 0)
fast = ((Mt @ (sgn < 0).astype(_np.int8)) % 2 == 1)
check("fast parity null equals an explicit sign product per path", bool((_np.array(brute) == fast).all()))

print("\n== reproduces the record")
df, _, _ = enumerate_paths(adj, {"Sting1"}, 20)
check("all simple T0 paths F2rl1 -> Sting1 = 2640, every one predicting UP (REPORT.md figure)",
      len(df) == 2640 and bool((df.net_sign < 0).all()), f"{len(df)} paths, {(df.net_sign < 0).mean():.1%} up")
# the enumeration itself must be weight-free; only learned_readout_signs reads the fitted model,
# and only to supply signs for the 3 Ifnb1 regulators that have no curated sign (the 'all' set)
import inspect

import lembas_par2.paths as _P
core = "".join(inspect.getsource(f) for f in [_P.signed_adjacency, _P.enumerate_paths, _P.summarise,
                                              _P.sign_null, _P.path_edge_matrix, _P.mrna_paths])
check("no fitted weight enters the enumeration itself",
      not any(k in core for k in ["readout_weights", "checkpoint", "torch", "load_model"]),
      [k for k in ["readout_weights", "checkpoint", "torch", "load_model"] if k in core])
check("the fitted model is read only for the unsigned regulators' signs",
      all(k in inspect.getsource(_P.learned_readout_signs) for k in ["readout_weights", "checkpoint"]))

print("\n== recorded run")
out = cfg.paths.output_dir / "step17"
if check("step-17 outputs present", (out / "report.json").exists() and (out / "summary.csv").exists()):
    rep = json.loads((out / "report.json").read_text())
    s = pd.read_csv(out / "summary.csv")
    t0 = s[(s.layer == "T0") & (s.label == "Sting1_signal")].iloc[0]
    check("recorded T0 Sting1 enumeration matches", int(t0.n_paths) == 2640 and t0.frac_up == 1.0)
    check("both regulator sets were run", {"Ifnb1_mrna_curated", "Ifnb1_mrna_all"} <= set(s.label))
    check("the null is reported on BOTH sides (a one-sided p was misread once)",
          {"p_up", "p_down", "direction"} <= set(s.columns))
    check("T0 and T0+T1 are recorded as favouring OPPOSITE directions for Ifnb1",
          s[(s.layer == "T0") & (s.label == "Ifnb1_signal")].direction.iloc[0] == "up"
          and s[(s.layer == "T0_T1") & (s.label == "Ifnb1_signal")].direction.iloc[0] == "down")
    check("no enumeration hit its explore cap", not bool(s.capped.any()), s.capped.tolist())
    print("\n[INFO] fraction of paths predicting the target UP on PAR2 loss:")
    for r in s.itertuples():
        print(f"  {r.layer:6s} {r.label:22s} {r.n_paths:7d} paths  up {r.frac_up:.1%}"
              + (f"  null {r.null_frac_up_median:.1%}  favours {r.direction.upper()}"
                 f" (p up {r.p_up:.3f} / down {r.p_down:.3f})" if "p_up" in s else ""))
    for lay, b in rep.get("bypass", {}).items():
        print(f"[INFO] {lay}: {b['n_up_bypassing_STING']} of {b['n_up']} Ifnb1-up paths avoid Sting1"
              + (f"; shortest {b['shortest_bypass']}" if b["shortest_bypass"] else ""))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
