"""Step 40: contribution tables under setup B, across ALL seeds.

contribution = weight x delta(source), the step-27 decomposition, but computed per seed and
reported as a median with the seed count agreeing on sign. Single-seed contribution tables have
misled this project twice (step 37's Nfkb1 table was seed 0, which turned out to be the one seed
that went the other way), so the per-seed agreement column is not optional.

Setup B = PAR2_29: LPS context, F2rl1 KO at the configured kappa, Tlr4 knocked down at 0.60.

    python scripts/run_step40_contributions.py
    python scripts/run_step40_contributions.py --nodes Sting1 Nfkb1 --kappa-tlr4 0.77
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="R4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--nodes", nargs="+",
                    default=["Sting1", "Nfkb1", "Nfkbia", "Ikbkb", "Cgas", "Tbk1"])
    ap.add_argument("--kappa-tlr4", type=float, default=0.60)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    cfg = load_config()
    lps = cfg.step13["contexts"]["lps"]
    frames, deltas = [], []
    for s in a.seeds:
        run = a.variant if s == 0 else f"{a.variant}_s{s}"
        ck = cfg.paths.output_dir / "source_variants" / run / "checkpoint.pt"
        if not ck.exists():
            continue
        m, _ = load_model(cfg, ck, "cpu")
        n = len(m.nodes)
        u = torch.zeros(n, 1, dtype=m.dtype)
        for x in lps:
            u[m.idx[x], 0] = cfg.perturbation["input_level"]
        U = u.repeat(1, 2)
        S = torch.ones(n, 2, dtype=m.dtype)
        S[m.idx["F2rl1"], 1] = 1 - cfg.step13["ko_kappa"]
        S[m.idx["Tlr4"], 1] = 1 - a.kappa_tlr4
        X = simulate(m.struct, m.weights(), m.bias, U, S, max_iter=3000, tol=1e-8)
        d = (S * X)[:, 1] - (S * X)[:, 0]
        W, E = m.weights(), m.edges
        deltas.append({nd: float(d[m.idx[nd]]) for nd in m.nodes if nd in m.idx} if False
                      else {nd: float(d[m.idx[nd]]) for nd in a.nodes})
        for tgt in a.nodes:
            for i in [i for i, t in enumerate(E.target) if t == tgt]:
                src = E.source.iloc[i]
                frames.append(dict(seed=s, target=tgt, source=src, tier=E.tier.iloc[i],
                                   weight=float(W[i]), d_source=float(d[m.idx[src]]),
                                   contribution=float(W[i]) * float(d[m.idx[src]])))
        print(f"  {run}: ok", flush=True)

    if not frames:
        sys.exit("no checkpoints; retrain first")
    C = pd.DataFrame(frames)
    out = cfg.paths.output_dir / "step40_contributions"
    out.mkdir(parents=True, exist_ok=True)
    C.to_csv(out / f"contributions_k{a.kappa_tlr4}.csv", index=False)
    ns = C.seed.nunique()
    DD = pd.DataFrame(deltas)

    for tgt in a.nodes:
        sub = C[C.target == tgt]
        g = sub.groupby(["source", "tier"]).agg(
            weight=("weight", "median"), d_source=("d_source", "median"),
            contribution=("contribution", "median"),
            agree=("contribution", lambda x: int(max((x > 0).sum(), (x < 0).sum())))).reset_index()
        g = g.reindex(g.contribution.abs().sort_values(ascending=False).index)
        tot = sub.groupby("seed").contribution.sum().median()
        up = g[g.contribution > 0].contribution.sum()
        dn = g[g.contribution < 0].contribution.sum()
        print(f"\n=== {tgt}: delta {DD[tgt].median():+.5f} "
              f"(seeds negative {int((DD[tgt] < 0).sum())}/{ns}), "
              f"{sub.source.nunique()} in-edges ===")
        print(f"    UP {up:+.5f} | DOWN {dn:+.5f} | NET {up+dn:+.5f}  (sum over all edges {tot:+.5f})")
        print(f"    'agree' = how many of {ns} seeds share that contribution's sign")
        print(g.head(a.top).to_string(index=False,
              formatters={"weight": "{:+.4f}".format, "d_source": "{:+.5f}".format,
                          "contribution": "{:+.5f}".format}))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
