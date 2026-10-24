"""Step 21: the predicted PAR2-KO change at EVERY node, so the route can be read in context.

Panel D showed ten hand-picked route nodes. Two questions that raises, and that this answers
with a measurement rather than an assertion:

  1. What happens UPSTREAM of ERK - at the beta-arrestins, RAF and MEK - which the panel skipped?
  2. Are the route nodes actually the most-changed nodes in the network, or merely the ones we
     chose to plot? Only a ranking over all 8393 nodes can say, and the answer decides whether
     "the route is where the action is" is a finding or a selection effect.

Writes outputs/step21_route/{node_deltas.csv.gz, route.csv, report.json}: the full per-node
delta for every seed and context, the ordered route table panel D plots, and the ranking.

NOTE ON WHICH MODEL. This uses the CURRENT variant (default R4). The frozen J0 that carries the
8/10 holdout headline CANNOT be loaded any more - its W_locked shape changed when curation moved
edges between signed and unsigned - so its route numbers exist only as the recorded columns in
outputs/step13/runs.csv. The report compares the two on the nodes they share, which is the only
honest way to carry the panel over.

    python scripts/run_route_profile.py                      # R4, 5 seeds, ~1 min on CPU
    python scripts/run_route_profile.py --variant R0 --seeds 0 1 2
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
from lembas_par2.eval_harness import load_model
from lembas_par2.step13 import ko_delta

# The prior route in causal order, as config.yaml: route_edges lays it out. Both arms from PAR2
# to ERK are included: the membrane arm (Gq -> PLCb -> DAG -> PKC/RasGRP -> Ras -> RAF -> MEK)
# and the endosomal arm (beta-arrestin scaffolding RAF/MEK/ERK directly).
ROUTE = [
    ("F2rl1", "PAR2", "receptor"),
    ("Gnaq", "Gq", "membrane arm"),
    ("Plcb1", "PLCB1", "membrane arm"),
    ("Plcb3", "PLCB3", "membrane arm"),
    ("SE_DAG", "DAG", "membrane arm"),
    ("Prkca", "PKCa", "membrane arm"),
    ("Rasgrp1", "RasGRP1", "membrane arm"),
    ("Rasgrp3", "RasGRP3", "membrane arm"),
    ("Kras", "KRAS", "membrane arm"),
    ("Hras", "HRAS", "membrane arm"),
    ("Arrb1", "ARRB1", "endosomal arm"),
    ("Arrb2", "ARRB2", "endosomal arm"),
    # step 22: the Ca2+ arm, which competes with Bax/Bak1 at the mitochondrion instead of
    # running through ERK. Added so the profile shows whether it carries any signal at all.
    ("SE_IP3", "IP3", "Ca2+ arm"),
    ("SE_CA2", "Ca2+", "Ca2+ arm"),
    ("Vdac1", "VDAC1", "Ca2+ arm"),
    ("Raf1", "RAF1", "shared MAPK core"),
    ("Braf", "BRAF", "shared MAPK core"),
    ("Map2k1", "MEK1", "shared MAPK core"),
    ("Map2k2", "MEK2", "shared MAPK core"),
    ("Mapk1", "ERK2", "shared MAPK core"),
    ("Mapk3", "ERK1", "shared MAPK core"),
    ("Rps6ka1", "RSK1", "BCL2 family"),
    ("Rps6ka3", "RSK3", "BCL2 family"),
    ("Mcl1", "MCL1", "BCL2 family"),
    ("Gsk3b", "GSK3B", "BCL2 family"),
    ("Bad", "BAD", "BCL2 family"),
    ("Bcl2l11", "BIM", "BCL2 family"),
    ("Bcl2", "BCL2", "BCL2 family"),
    ("Bcl2l1", "BCL-xL", "BCL2 family"),
    ("Bax", "BAX", "pore"),
    ("Bak1", "BAK1", "pore"),
    ("SE_CYTO_MTDNA", "cytosolic mtDNA", "sensing"),
    ("Cgas", "cGAS", "sensing"),
    ("Sting1", "STING", "sensing"),
    ("Tbk1", "TBK1", "output"),
    ("Irf3", "IRF3", "output"),
]
MEASURED = {"SE_CYTO_MTDNA", "Bcl2l1"}   # the only route nodes with a sealed holdout observation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="R4")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--contexts", nargs="*", default=["lps", "pooled"])
    a = ap.parse_args()

    cfg = load_config()
    st = cfg.step13
    out = cfg.paths.output_dir / "step21_route"
    out.mkdir(parents=True, exist_ok=True)

    frames, missing = [], []
    for s in a.seeds:
        run = a.variant if s == 0 else f"{a.variant}_s{s}"
        ck = cfg.paths.output_dir / "source_variants" / run / "checkpoint.pt"
        if not ck.exists():
            missing.append(run)
            continue
        try:
            m, _ = load_model(cfg, ck, a.device)
        except Exception as exc:
            # the usual cause is a curation change that moved edges between tiers or layers, so
            # the saved parameter shapes no longer match. Say so instead of raising a bare
            # ValueError from deep in the loader.
            sys.exit(
                f"{run}: checkpoint does not match the current network ({exc}). "
                f"The topology changed since it was trained, so retrain first: "
                f"python scripts/run_source_variants.py "
                f"--seeds {chr(32).join(str(x) for x in a.seeds)} --variants {a.variant}")
        for ctx in a.contexts:
            d = ko_delta(m, st["contexts"][ctx], st["ko_kappa"]).cpu().numpy()
            frames.append(pd.DataFrame(dict(node=m.nodes, seed=s, context=ctx, delta=d)))
        print(f"  {run}: {len(a.contexts)} contexts", flush=True)
    if missing:
        print(f"[warn] no checkpoint for {missing}")
    if not frames:
        sys.exit(f"no {a.variant} checkpoints; run scripts/run_source_variants.py --variants {a.variant}")
    D = pd.concat(frames, ignore_index=True)
    D.to_csv(out / "node_deltas.csv.gz", index=False)

    med = D.groupby(["context", "node"]).delta.median().unstack(0)
    # rank every node by |delta|, so a route node's position is measured against the whole network
    rank = {c: med[c].abs().rank(ascending=False, method="min") for c in med.columns}

    rows = []
    for i, (node, lab, seg) in enumerate(ROUTE):
        if node not in med.index:
            print(f"[warn] {node} is not a network node")
            continue
        r = dict(order=i, node=node, label=lab, segment=seg, measured=node in MEASURED)
        for c in med.columns:
            r[f"delta_{c}"] = float(med.at[node, c])
            r[f"rank_{c}"] = int(rank[c][node])
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(out / "route.csv", index=False)

    n_nodes = int(med.shape[0])
    print(f"\n== route, PAR2 KO (median of {D.seed.nunique()} seeds), {n_nodes} nodes ranked")
    cols = ["label", "segment"] + [c for c in R.columns if c.startswith(("delta_", "rank_"))]
    print(R[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    rep = dict(variant=a.variant, seeds=[int(s) for s in sorted(D.seed.unique())], n_nodes=n_nodes,
               missing=missing, route=R.to_dict("records"))
    for c in med.columns:
        top = med[c].abs().sort_values(ascending=False).head(25)
        on_route = [n for n in top.index if n in {x[0] for x in ROUTE}]
        rep[f"top25_{c}"] = [dict(node=n, delta=float(med.at[n, c]),
                                  on_route=bool(n in {x[0] for x in ROUTE})) for n in top.index]
        best = int(R[f"rank_{c}"].min()) if len(R) else None
        print(f"\n[{c}] of the 25 most-changed nodes, {len(on_route)} are on the route: {on_route}")
        print(f"[{c}] best route rank {best} of {n_nodes}; "
              f"median route rank {int(R[f'rank_{c}'].median())}")
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
