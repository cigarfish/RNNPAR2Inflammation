"""Step 27: who actually moves a node on PAR2 knockout — every in-edge, weighted, ranked.

For a target node this lists each incoming edge as

    contribution = weight x delta(source)

so the UP list and the DOWN list can be compared directly and the question "what would have to
change for DOWN to win?" becomes arithmetic instead of argument.

**This is a first-order attribution.** The steady state is nonlinear (the activation saturates and
the network has feedback), so the contributions do not sum exactly to the node's own delta. The
printed residual says how far off the linearisation is; treat the ranking as reliable and the
absolute numbers as indicative.

    python scripts/run_node_contributions.py                       # Sting1, R4, 5 seeds
    python scripts/run_node_contributions.py --node SE_CYTO_MTDNA
    python scripts/run_node_contributions.py --node Sting1 --context pooled --top 40
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
from lembas_par2.step13 import ko_delta


@torch.no_grad()
def contributions(cfg, ck, node, context, kappa, device="cpu"):
    m, _ = load_model(cfg, ck, device)
    d = ko_delta(m, cfg.step13["contexts"][context], kappa).cpu().numpy()
    w = m.weights().detach().cpu().numpy()
    e = m.edges
    k = np.flatnonzero((e.target == node).values)
    rows = []
    for i in k:
        src = e.source.iloc[i]
        rows.append(dict(source=src, tier=e.tier.iloc[i], sign=int(e.sign.iloc[i]),
                         weight=float(w[i]), d_source=float(d[m.idx[src]]),
                         contribution=float(w[i]) * float(d[m.idx[src]])))
    return pd.DataFrame(rows), float(d[m.idx[node]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default="Sting1")
    ap.add_argument("--variant", default="R4")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--context", default="lps")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    cfg = load_config()
    out = cfg.paths.output_dir / "step27_contributions"
    out.mkdir(parents=True, exist_ok=True)

    frames, deltas = [], []
    for s in a.seeds:
        run = a.variant if s == 0 else f"{a.variant}_s{s}"
        ck = cfg.paths.output_dir / "source_variants" / run / "checkpoint.pt"
        if not ck.exists():
            continue
        try:
            df, dn = contributions(cfg, ck, a.node, a.context, cfg.step13["ko_kappa"], a.device)
        except Exception as exc:
            sys.exit(f"{run}: {exc}. Retrain first: python scripts/run_step23_retrain.py")
        df["seed"] = s
        frames.append(df)
        deltas.append(dn)
    if not frames:
        sys.exit(f"no {a.variant} checkpoints; run python scripts/run_step23_retrain.py")

    D = pd.concat(frames, ignore_index=True)
    D.to_csv(out / f"{a.node}_{a.context}_per_seed.csv", index=False)
    g = (D.groupby(["source", "tier", "sign"])
           .agg(weight=("weight", "median"), d_source=("d_source", "median"),
                contribution=("contribution", "median"))
           .reset_index()
           .sort_values("contribution"))
    g.to_csv(out / f"{a.node}_{a.context}.csv", index=False)

    node_delta = float(np.median(deltas))
    up = g[g.contribution > 0]
    dn = g[g.contribution < 0]
    print(f"\n== who moves {a.node} on PAR2 KO ({a.context}, {a.variant}, "
          f"median of {len(frames)} seeds)\n")
    print(f"   {a.node} itself: {node_delta:+.4f}   |   {len(g)} in-edges\n")

    def block(title, frame, n):
        print(f"  -- {title} ({len(frame)} edges, total {frame.contribution.sum():+.5f})")
        if not len(frame):
            print("     none")
            return
        print(f"     {'source':16s} {'tier':5s} {'sign':>4s} {'weight':>9s} "
              f"{'d(source)':>10s} {'contribution':>13s}")
        for r in frame.head(n).itertuples():
            print(f"     {r.source:16s} {r.tier:5s} {r.sign:+4d} {r.weight:+9.4f} "
                  f"{r.d_source:+10.4f} {r.contribution:+13.5f}")

    block("PUSHING DOWN", dn.sort_values("contribution"), a.top)
    print()
    block("PUSHING UP", up.sort_values("contribution", ascending=False), a.top)

    tot_up, tot_dn = float(up.contribution.sum()), float(dn.contribution.sum())
    net = tot_up + tot_dn
    print(f"\n  UP {tot_up:+.5f}   DOWN {tot_dn:+.5f}   NET {net:+.5f}")
    print(f"  (first-order; {a.node}'s actual delta is {node_delta:+.4f}, "
          f"residual {node_delta - net:+.4f} from nonlinearity and feedback)")

    if net > 0 and len(up):
        print(f"\n  To flip {a.node} negative, the DOWN side must gain {net:+.5f}. Options, with"
              f"\n  the factor each would need:")
        for r in dn.sort_values("contribution").head(5).itertuples():
            need = (abs(r.contribution) + net) / abs(r.contribution) if r.contribution else np.inf
            print(f"     {r.source:16s} carries {r.contribution:+.5f}; it would have to be "
                  f"{need:5.1f}x larger alone")
        big = up.sort_values("contribution", ascending=False).iloc[0]
        print(f"     or the largest UP edge ({big.source}, {big.contribution:+.5f}) would have to"
              f" fall by {min(1.0, net / big.contribution):.0%}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
