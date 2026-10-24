"""Step 50: do the dynasore route nodes rank high in the WHOLE network, as the PAR2-KO ones did?

Panel D makes this argument for the PAR2 knockout: 19 of the 25 most-changed nodes of 8 393 are
on the route, and the median route node ranks 17th. This is the same measurement for dynasore,
over both arms.

WHAT IS AND IS NOT EVIDENCE HERE -- the same caveat panel D carries, and it is sharper for
dynasore. The cut edges are T0, and T0 edges are ~5x the weight of T1 and ~300x T3, so nodes
immediately downstream of a severed T0 edge are expected to move most. A high rank for TRIF or
TBK1 is therefore partly built in and must not be sold as a discovery.

WHAT IS GENUINELY INFORMATIVE is the CONTRAST BETWEEN THE TWO BRANCHES OF THE SAME RECEPTOR.
MyD88 and TRIF both sit one step from TLR4, both reached by curated edges of comparable tier --
but dynasore cuts only the TRIF side. If the TRIF branch ranks high while the MyD88 branch ranks
low, that is a statement about the perturbation rather than about edge weights, because the two
branches share the receptor, the context and the tier structure. That contrast is the built-in
control, and this script reports it explicitly.

Writes outputs/step50_dynasore_ranks/{node_ranks.csv, top25.csv}.

    python scripts/run_step50_dynasore_ranks.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.eval_harness import load_model
from lembas_par2.core import simulate

torch.set_grad_enabled(False)
ENDO_ERK = [(a, t) for a in ("Arrb1", "Arrb2")
            for t in ("Raf1", "Map2k1", "Map2k2", "Mapk1", "Mapk3")]
TRIF = [("Tlr4", "Ticam1"), ("Tlr4", "Ticam2")]


@torch.no_grad()
def main():
    cfg = load_config()
    # this step annotates the ranks with the drawn route, so step 46 must have run first
    route_csv = ROOT / "outputs" / "step46_dynasore_route" / "nodes.csv"
    if not route_csv.exists():
        sys.exit(f"missing {route_csv}. Produce it first:"
                 "\n  python scripts/run_step46_dynasore_route.py")
    route = pd.read_csv(route_csv)
    meta = route.drop_duplicates("node").set_index("node")[["label", "segment", "row", "cut"]]

    frames = []
    for sd in [0, 1, 2, 3, 4]:
        run = "R4" if sd == 0 else f"R4_s{sd}"
        ck = cfg.paths.output_dir / "source_variants" / run / "checkpoint.pt"
        if not ck.exists():
            continue
        try:
            m, _ = load_model(cfg, ck, "cpu")
        except (RuntimeError, ValueError) as exc:
            sys.exit(f"{run}: checkpoint does not match the network ({exc}).\nRetrain first:\n"
                     f"  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
        n = len(m.nodes)
        u = torch.zeros(n, 1, dtype=m.dtype)
        for x in cfg.step13["contexts"]["lps"]:
            u[m.idx[x], 0] = cfg.perturbation["input_level"]
        S = torch.ones(n, 1, dtype=m.dtype)
        key = {(p, q): i for i, (p, q) in enumerate(zip(m.edges.source, m.edges.target))}
        X0 = simulate(m.struct, m.weights(), m.bias, u, S, max_iter=3000, tol=1e-8)
        w = m.weights().clone()
        w[[key[p] for p in ENDO_ERK + TRIF if p in key]] = 0.0
        X1 = simulate(m.struct, w, m.bias, u, S, max_iter=3000, tol=1e-8)
        d = (X1 - X0)[:, 0].numpy()
        r = pd.Series(np.abs(d)).rank(ascending=False, method="min").values
        frames.append(pd.DataFrame(dict(seed=sd, node=m.nodes, delta=d, absd=np.abs(d), rank=r)))
        print(f"  {run}: {n} nodes ranked", flush=True)

    if not frames:
        sys.exit("no checkpoints found. Train first:" 
                 "\n  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
    A = pd.concat(frames)
    n_nodes = int(A.groupby("seed").node.nunique().median())
    med = A.groupby("node").agg(delta=("delta", "median"), absd=("absd", "median"),
                                rank=("rank", "median"))
    med = med.join(meta, how="left")
    out = cfg.paths.output_dir / "step50_dynasore_ranks"
    out.mkdir(parents=True, exist_ok=True)
    med.sort_values("rank").to_csv(out / "node_ranks.csv")

    top = med.sort_values("rank").head(25).copy()
    top["on_route"] = top.label.notna()
    top.to_csv(out / "top25.csv")
    print(f"\n=== top 25 of {n_nodes} nodes by |delta| under dynasore (median of "
          f"{A.seed.nunique()} seeds) ===")
    print(top[["delta", "rank", "label", "segment", "on_route"]].round(4).to_string())
    print(f"\n   on a drawn route: {int(top.on_route.sum())} of 25")

    R = med[med.label.notna()].copy()
    R["arm"] = np.where(R.row == 0, "PAR2/ERK", "TLR4")
    R["branch"] = np.where(R.segment == "MyD88 branch", "MyD88 (NOT cut)",
                           np.where(R.row == 0, "PAR2/ERK (cut)", "TRIF (cut)"))
    print(f"\n=== rank of every drawn route node, of {n_nodes} ===")
    for br, g in R.groupby("branch"):
        g = g.sort_values("rank")
        print(f"\n  -- {br}: median rank {int(g['rank'].median())}, "
              f"{int((g['rank'] <= 25).sum())}/{len(g)} inside the top 25")
        print(g[["label", "delta", "rank"]].to_string(index=False,
                                                     float_format=lambda v: f"{v:+.4f}"))

    print("\n=== THE CONTROL: two branches of the SAME receptor, only one cut ===")
    t = R[R.branch == "TRIF (cut)"]
    mm = R[R.branch == "MyD88 (NOT cut)"]
    print(f"   TRIF branch  median rank {int(t['rank'].median()):6d}   "
          f"median |delta| {t.absd.median():.4f}")
    print(f"   MyD88 branch median rank {int(mm['rank'].median()):6d}   "
          f"median |delta| {mm.absd.median():.4f}")
    print(f"   ratio of median |delta|: {t.absd.median() / max(mm.absd.median(), 1e-12):.0f}x")
    print("\n   Both branches hang off TLR4 one step apart; dynasore cuts only TRIF. The gap is"
          "\n   therefore attributable to the perturbation, not to tier or distance -- which is"
          "\n   the one part of this analysis that is NOT built in by T0 weighting.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
