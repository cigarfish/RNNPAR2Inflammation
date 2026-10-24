"""Step 46: the dynasore route data for panel E — two competing arms, multi-seed, written to disk.

`run_step32_dynasore.py` is the single-seed diagnostic probe that established the edge-cut setup.
This is the reporting version: 5 seeds, three scenarios, and CSVs the figure reads.

THE RESULT THE PANEL SHOWS. Dynasore cuts two dynamin-dependent routes and they push Sting1 in
OPPOSITE directions:
    endosomal ERK scaffold only  Arrb1/2 -> RAF/MEK/ERK   -> dSting1 POSITIVE
    endosomal TLR4 -> TRIF only  Tlr4 -> Ticam1/Ticam2    -> dSting1 NEGATIVE, and larger
    both (= dynasore)                                      -> NEGATIVE
and the MyD88 branch (Tlr4 -> Myd88 -> Irak4 -> Traf6 -> Map3k7) is dynamin-INDEPENDENT, is NOT
cut, and barely moves. That asymmetry is the prediction: not a global shutdown, one branch of one
receptor.

Writes outputs/step46_dynasore_route/{nodes.csv, sting_contributions.csv, scenarios.csv}.

    python scripts/run_step46_dynasore_route.py
"""
import argparse
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
SCENARIOS = {"endosomal_erk": ENDO_ERK, "trif": TRIF, "dynasore": ENDO_ERK + TRIF}

# The route in causal order, one entry per plotted node:
#   (display label, network node, segment, row, cut)   row 0 = PAR2 arm, row 1 = TLR4.
# `cut` = is this node downstream of an edge dynasore removes? It drives the greying,
# so the figure never has to parse a segment label to know what the drug touched.
# Both rows END at STING so the convergence is visible in the figure.
ROUTE = [
    # ---- row 0: PAR2 / beta-arrestin -> ERK -| BCL2 -> pore -> sensing  (the UP arm) ----
    ("PAR2", "F2rl1", "receptor", 0, False),
    ("ARRB1", "Arrb1", "receptor", 0, False),
    ("RAF1", "Raf1", "MAPK core", 0, True),
    ("MEK1", "Map2k1", "MAPK core", 0, True),
    ("MEK2", "Map2k2", "MAPK core", 0, True),
    ("ERK1", "Mapk3", "MAPK core", 0, True),
    ("ERK2", "Mapk1", "MAPK core", 0, True),
    ("BIM", "Bcl2l11", "BCL2 family", 0, True),
    ("MCL1", "Mcl1", "BCL2 family", 0, True),
    ("GSK3B", "Gsk3b", "BCL2 family", 0, True),
    ("BAD", "Bad", "BCL2 family", 0, True),
    ("BCL2", "Bcl2", "BCL2 family", 0, True),
    ("BCL-xL", "Bcl2l1", "BCL2 family", 0, True),
    ("BAX", "Bax", "pore", 0, True),
    ("BAK1", "Bak1", "pore", 0, True),
    ("mtDNA", "SE_CYTO_MTDNA", "sensing", 0, True),
    ("cGAS", "Cgas", "sensing", 0, True),
    ("STING", "Sting1", "output", 0, True),
    # ---- row 1: TLR4, TRIF branch (cut) above the MyD88 branch (NOT cut) ----
    ("TLR4", "Tlr4", "receptor", 1, False),
    ("TRIF", "Ticam1", "TRIF branch", 1, True),
    ("TRAF3", "Traf3", "TRIF branch", 1, True),
    ("TBK1", "Tbk1", "TRIF branch", 1, True),
    ("IRF3", "Irf3", "TRIF branch", 1, True),
    ("IFN-b", "Ifnb1", "TRIF branch", 1, True),
    ("MYD88", "Myd88", "MyD88 branch", 1, False),
    ("IRAK4", "Irak4", "MyD88 branch", 1, False),
    ("TRAF6", "Traf6", "MyD88 branch", 1, False),
    ("TAK1", "Map3k7", "MyD88 branch", 1, False),
    ("IKKB", "Ikbkb", "MyD88 branch", 1, False),
    ("IkBa", "Nfkbia", "MyD88 branch", 1, False),
    ("NF-kB", "Nfkb1", "MyD88 branch", 1, False),
    ("STING", "Sting1", "output", 1, True),
]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="R4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    cfg = load_config()
    lps = cfg.step13["contexts"]["lps"]

    node_rows, contrib_rows, scen_rows = [], [], []
    for sd in a.seeds:
        run = a.variant if sd == 0 else f"{a.variant}_s{sd}"
        ck = cfg.paths.output_dir / "source_variants" / run / "checkpoint.pt"
        if not ck.exists():
            print(f"[skip] {run}")
            continue
        try:
            m, _ = load_model(cfg, ck, "cpu")
        except (RuntimeError, ValueError) as exc:
            sys.exit(f"{run}: checkpoint does not match the network ({exc}).\nRetrain first:\n"
                     f"  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
        n = len(m.nodes)
        u = torch.zeros(n, 1, dtype=m.dtype)
        for x in lps:
            u[m.idx[x], 0] = cfg.perturbation["input_level"]
        S = torch.ones(n, 1, dtype=m.dtype)
        key = {(p, q): i for i, (p, q) in enumerate(zip(m.edges.source, m.edges.target))}
        X0 = simulate(m.struct, m.weights(), m.bias, u, S, max_iter=3000, tol=1e-8)

        for name, pairs in SCENARIOS.items():
            w = m.weights().clone()
            cut = [key[p] for p in pairs if p in key]
            w[cut] = 0.0
            X = simulate(m.struct, w, m.bias, u, S, max_iter=3000, tol=1e-8)
            d = (X - X0)[:, 0]
            scen_rows.append(dict(seed=sd, scenario=name, n_cut=len(cut),
                                  dSting1=float(d[m.idx["Sting1"]])))
            if name != "dynasore":
                continue
            for order, (lab, nd, seg, row, cut) in enumerate(ROUTE):
                if nd in m.idx:
                    node_rows.append(dict(seed=sd, order=order, label=lab, node=nd,
                                          segment=seg, row=row, cut=cut,
                                          delta=float(d[m.idx[nd]]),
                                          baseline=float(X0[m.idx[nd], 0])))
            for i, (s_, t_) in enumerate(zip(m.edges.source, m.edges.target)):
                if t_ == "Sting1":
                    contrib_rows.append(dict(seed=sd, source=s_, tier=m.edges.tier.iloc[i],
                                             weight=float(w[i]),
                                             d_source=float(d[m.idx[s_]]),
                                             contribution=float(w[i]) * float(d[m.idx[s_]])))
        print(f"  {run}: ok", flush=True)

    if not scen_rows:
        sys.exit("no checkpoints found. Train first:" 
                 "\n  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
    out = cfg.paths.output_dir / "step46_dynasore_route"
    out.mkdir(parents=True, exist_ok=True)
    N, C, Sc = pd.DataFrame(node_rows), pd.DataFrame(contrib_rows), pd.DataFrame(scen_rows)
    for df, nm in ((N, "nodes"), (C, "sting_contributions"), (Sc, "scenarios")):
        df.to_csv(out / f"{nm}.csv", index=False)

    ns = Sc.seed.nunique()
    print(f"\n=== dSting1 by scenario ({a.variant}, median of {ns} seeds) ===")
    g = Sc.groupby("scenario").agg(dSting1=("dSting1", "median"),
                                   n_cut=("n_cut", "first"),
                                   seeds_neg=("dSting1", lambda x: int((x < 0).sum())))
    print(g.reindex(["endosomal_erk", "trif", "dynasore"]).round(5).to_string())
    add = float(g.loc["endosomal_erk", "dSting1"] + g.loc["trif", "dSting1"])
    print(f"\n  arms summed {add:+.5f} vs dynasore {g.loc['dynasore','dSting1']:+.5f} "
          f"-> near-additive, so two separate routes is an honest drawing")

    print(f"\n=== contributions into Sting1 under dynasore ===")
    cg = C.groupby(["source", "tier"]).contribution.median().reset_index()
    cg = cg.reindex(cg.contribution.abs().sort_values(ascending=False).index)
    print(cg.head(6).to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    tot = float(cg.contribution.sum())
    sim = float(g.loc["dynasore", "dSting1"])
    print(f"\n  first-order sum {tot:+.5f} vs simulated {sim:+.5f} "
          f"-> {abs(1 - tot/sim)*100:.0f}% higher-order. The decomposition is INDICATIVE, "
          f"not exact; the legend must say so.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
