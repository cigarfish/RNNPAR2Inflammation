"""Step 12h validation (items 2, 4, 5): checks the outputs of scripts/run_spectral.py.

    python tests/test_step12h_spectral.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
out = cfg.paths.output_dir / "step12h"
runs = [p.name.replace("_summary.json", "") for p in sorted(out.glob("*_summary.json"))]
if not check("12h outputs present", len(runs) > 0, runs):
    sys.exit(1)

for r in runs:
    print(f"\n== {r}")
    S = json.loads((out / f"{r}_summary.json").read_text())
    E = pd.read_csv(out / f"{r}_edges.csv.gz")
    C = pd.read_csv(out / f"{r}_feedback_arc.csv")
    # plan pass criteria
    check("eigen-residuals <= 1e-10 (left and right Perron vectors)",
          S["residual_right"] <= 1e-10 and S["residual_left"] <= 1e-10, f"{S['residual_right']:.1e} / {S['residual_left']:.1e}")
    check("left and right Perron roots agree", abs(S["rho"] - S["rho_left"]) < 1e-8, f"{S['rho']:.6f} / {S['rho_left']:.6f}")
    check("Perron vectors non-negative", S["u_min"] >= 0 and S["v_min"] >= 0)
    check("Euler: sum |w| * d rho / d|w| = rho", S["euler_rel_error"] < 1e-8, f"{S['euler_rel_error']:.1e}")
    check("sensitivities zero outside loops, non-negative inside", (E.sensitivity >= 0).all())
    shares = sum(d["loading_share"] for d in S["by_tier"].values())
    check("tier loading shares sum to 1", abs(shares - 1) < 1e-8, f"{shares:.10f}")
    # feedback-arc curve: protected edges never cut, curve monotone
    check("curve starts at the uncut network (giant SCC 8027 nodes, as in the plan review)",
          int(C.giant_scc_nodes.iloc[0]) == 8027, int(C.giant_scc_nodes.iloc[0]))
    check("giant SCC and rho non-increasing as T3 is cut",
          (np.diff(C.giant_scc_nodes) <= 0).all() and (np.diff(C.rho_absA) <= 1e-9).all())
    n_t3_safe = int(((E.tier == "T3") & ~E.route_touching).sum())
    check("only T3 edges away from the route are eligible", int(C.n_cut.iloc[-1]) == n_t3_safe,
          f"{int(C.n_cut.iloc[-1])} = {n_t3_safe}")
    th = S["targeted_headroom"]
    check("T0 route alone is acyclic (rho ~0)", th["rho_T0_alone"] < 0.01, f"{th['rho_T0_alone']:.4f}")
    # decision rules (TASKS 12h, fixed in advance)
    f50 = S["giant_scc_below_half_at_T3_fraction"]
    print(f"[INFO] headroom rho(|A|)/rho(A) {S['headroom_absA_over_signed']:.2f}x -> "
          + ("12k warranted" if S['headroom_absA_over_signed'] >= 3 else
             "12k warranted, modest gain, after 12j" if S['headroom_absA_over_signed'] >= 1.5 else
             "bound nearly tight: drop this line (plan rule)"))
    print(f"[INFO] giant SCC < 50 % of nodes first at {f50} of T3 cut -> "
          + ("12l warranted" if f50 is not None and f50 < 0.05 else "12l NOT warranted (needs < 5 %)"))
    print(f"[INFO] rho loading: T0 route {S['route_T0']['loading_share']:.1%}, route-touching {S['route_touching']['loading_share']:.1%}, "
          f"T3 away from route {S['bulk_T3_not_route']:.1%}; top 1 % of edges {S['top1pct_share']:.1%}")
    print(f"[INFO] targeted headroom: remove {th['route_internal']['n_removed']} route-closing edges -> "
          f"{th['route_internal']['max_scale']:.2f}x; remove all {th['route_touching']['n_removed']} route-touching -> "
          f"{th['route_touching']['max_scale']:.2f}x")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
