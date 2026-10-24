"""Step 12h items 2, 4, 5: Perron sensitivities, feedback-arc curve, loop-gain headroom.

Read-only; no training. GPU optional (~1-3 min per checkpoint on CPU).
    python scripts/run_spectral.py                      # S5 (baseline_s5) and J3 (12j winner)
    python scripts/run_spectral.py --runs X=path/to/checkpoint.pt
Writes outputs/step12h/{<run>_summary.json, <run>_edges.csv.gz, <run>_feedback_arc.csv}; then:
    python tests/test_step12h_spectral.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.core import signed_radius
from lembas_par2.eval_harness import load_model
from lembas_par2.spectral import feedback_arc_curve, loop_gain, sensitivities, targeted_headroom

RUNS = {"S5": "outputs/objective_variants/S5/checkpoint.pt", "J3": "outputs/source_variants/J3/checkpoint.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = load_config()
    out = cfg.paths.output_dir / "step12h"
    out.mkdir(parents=True, exist_ok=True)
    runs = dict(r.split("=", 1) for r in a.runs) if a.runs else RUNS
    se = pd.read_csv(cfg.paths.output_dir / "step05_signalling_edges.csv.gz", usecols=["source", "target", "string_score"])
    se = se.drop_duplicates(["source", "target"]).set_index(["source", "target"]).string_score
    for name, ck in runs.items():
        ck = ROOT / ck
        if not ck.exists():
            print(f"[skip] {name}: {ck}")
            continue
        m, _ = load_model(cfg, ck, a.device)
        print(f"\n=== {name}  ({ck.relative_to(ROOT)})", flush=True)
        S, edges = sensitivities(m)
        S["rho_signedA"] = signed_radius(m.struct, m.weights())
        S["headroom_absA_over_signed"] = S["rho"] / S["rho_signedA"]
        S["loop_gain"] = loop_gain(S["rho"])
        score = se.reindex(list(zip(m.edges.source, m.edges.target))).values.astype(float)
        curve, ref = feedback_arc_curve(m, score)
        S["feedback_arc_reference"] = ref
        below = curve[curve.giant_scc_frac < 0.5]
        S["giant_scc_below_half_at_T3_fraction"] = float(below.fraction_T3_cut.iloc[0]) if len(below) else None
        S["targeted_headroom"] = targeted_headroom(m)
        (out / f"{name}_summary.json").write_text(json.dumps(S, indent=2, default=float))
        edges.to_csv(out / f"{name}_edges.csv.gz", index=False)
        curve.to_csv(out / f"{name}_feedback_arc.csv", index=False)

        print(f"  rho(|A|) {S['rho']:.4f} (left {S['rho_left']:.4f}; residuals {S['residual_right']:.1e} / "
              f"{S['residual_left']:.1e}); signed {S['rho_signedA']:.4f}; headroom {S['headroom_absA_over_signed']:.2f}x")
        print(f"  Euler check: sum |w| s = {S['euler_sum']:.6f} (rel. error {S['euler_rel_error']:.1e})")
        print("  rho loading by tier: " + ", ".join(
            f"{t} {d['loading_share']:.1%} ({d['n_loop']}/{d['n_edges']} in loops)" for t, d in S["by_tier"].items()))
        print(f"  route-touching edges: {S['route_touching']['n_edges']} carry {S['route_touching']['loading_share']:.1%}; "
              f"T3 bulk away from the route {S['bulk_T3_not_route']:.1%}; top 1 % of edges ({S['top1pct_n']}) carry "
              f"{S['top1pct_share']:.1%}")
        r0 = S["route_T0"]
        print(f"  T0 route: {r0['loading_share']:.2%} of rho; {r0['n_T0_outside_loops']} of 77 outside every loop; "
              f"median sensitivity {r0['median_sensitivity']:.2e} (max {r0['max_sensitivity']:.2e})")
        print("  feedback-arc curve (T3 cut in ascending STRING score, route edges protected):")
        print(curve.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"  reference, T0+T1 only: giant SCC {ref['T0_T1_only_giant_scc_frac']:.1%} of nodes, rho {ref['T0_T1_only_rho']:.3f}")
        th = S["targeted_headroom"]
        print(f"  rho of the T0 route alone {th['rho_T0_alone']:.4f} (acyclic); headroom for the non-T0 edges at rho<=0.95: "
              + ", ".join(f"{k} removed ({v['n_removed']}) -> rho {v['rho_after']:.3f}, scale {v['max_scale']:.2f}x"
                          for k, v in th.items() if isinstance(v, dict)))
        print(f"  loop gain 1/(1-rho): " + ", ".join(f"rho {k}: {v:.1f}" for k, v in S["loop_gain"].items()))
    print(f"\nwrote {out}\nnext: python tests/test_step12h_spectral.py")


if __name__ == "__main__":
    main()
