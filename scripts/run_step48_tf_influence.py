"""Step 48: is Pparg over-influential in the readout layer, or is it just a big regulon?

PROMPTED BY: Pparg is the single largest contributor to BOTH scored misses -- Ifnb1 in PAR2_30
(-0.0054 of a -0.016 net) and Nfkbib under dynasore (+0.0029 of a +0.0018 net). Two genes is not
a pattern, but if one TF dominates the readout layer generally then those are two symptoms of one
defect rather than two unrelated misses, and the SI panels should say so.

WHAT IS MEASURED, per TF, over every measured gene it regulates:
  n_genes     how many TF->gene readout edges it has (regulon size)
  med_w       median |readout weight|
  delta       its own activity change under the chosen perturbation
  influence   sum over its genes of |weight x delta|  -- total swing it imposes on the readout
  n_top       for how many genes it is the LARGEST single |contribution|
  frac_top    n_top / n_genes

THE CONTROL THAT MATTERS. A TF with a big regulon is top-contributor for many genes simply because
it is present in many, so n_top alone proves nothing. `frac_top` and the influence-per-gene
columns are what separate "widely connected" from "over-weighted". Both are reported, and the
summary states Pparg's rank on each.

    python scripts/run_step48_tf_influence.py                    # PAR2 KO, no TLR4 input
    python scripts/run_step48_tf_influence.py --setup dynasore
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
FOCUS = "Pparg"


@torch.no_grad()
def delta_for(m, cfg, setup):
    n = len(m.nodes)
    if setup == "dynasore":
        u = torch.zeros(n, 1, dtype=m.dtype)
        for x in cfg.step13["contexts"]["lps"]:
            u[m.idx[x], 0] = cfg.perturbation["input_level"]
        S = torch.ones(n, 1, dtype=m.dtype)
        key = {(p, q): i for i, (p, q) in enumerate(zip(m.edges.source, m.edges.target))}
        X0 = simulate(m.struct, m.weights(), m.bias, u, S, max_iter=3000, tol=1e-8)
        w = m.weights().clone()
        w[[key[p] for p in ENDO_ERK + TRIF if p in key]] = 0.0
        X1 = simulate(m.struct, w, m.bias, u, S, max_iter=3000, tol=1e-8)
        return (X1 - X0)[:, 0]
    U = torch.zeros(n, 2, dtype=m.dtype)          # PAR2 KO, no TLR4 input (PAR2_26 / PAR2_30)
    S = torch.ones(n, 2, dtype=m.dtype)
    S[m.idx["F2rl1"], 1] = 1 - cfg.step13["ko_kappa"]
    X = simulate(m.struct, m.weights(), m.bias, U, S, max_iter=3000, tol=1e-8)
    return (S * X)[:, 1] - (S * X)[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="R4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--setup", default="par2_ko", choices=["par2_ko", "dynasore"])
    a = ap.parse_args()
    cfg = load_config()

    frames = []
    for sd in a.seeds:
        run = a.variant if sd == 0 else f"{a.variant}_s{sd}"
        ck = cfg.paths.output_dir / "source_variants" / run / "checkpoint.pt"
        if not ck.exists():
            continue
        try:
            m, _ = load_model(cfg, ck, "cpu")
        except (RuntimeError, ValueError) as exc:
            sys.exit(f"{run}: checkpoint does not match the network ({exc}).\nRetrain first:\n"
                     f"  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
        d = delta_for(m, cfg, a.setup).numpy()
        rtf, rg = m.r_tf.numpy(), m.r_gene.numpy()
        rw = m.readout_weights().numpy()
        c = rw * d[rtf]
        E = pd.DataFrame(dict(tf=np.asarray(m.nodes)[rtf], gene=np.asarray(m.genes)[rg],
                              w=rw, dtf=d[rtf], c=c))
        # which TF is the largest single contribution for each gene
        E["absc"] = E.c.abs()
        top = E.loc[E.groupby("gene").absc.idxmax(), ["gene", "tf"]].rename(
            columns={"tf": "top_tf"})
        E = E.merge(top, on="gene")
        g = E.groupby("tf").agg(n_genes=("gene", "nunique"),
                                med_w=("w", lambda x: float(np.median(np.abs(x)))),
                                delta=("dtf", "first"),
                                influence=("absc", "sum"),
                                n_top=("top_tf", lambda s: int((s == s.name).sum())
                                       if False else 0)).reset_index()
        nt = E[E.tf == E.top_tf].groupby("tf").gene.nunique().rename("n_top")
        g = g.drop(columns="n_top").merge(nt, on="tf", how="left").fillna({"n_top": 0})
        g["seed"] = sd
        frames.append(g)
        print(f"  {run}: ok", flush=True)

    if not frames:
        sys.exit("no checkpoints found. Train first:" 
                 "\n  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
    A = pd.concat(frames).groupby("tf").median(numeric_only=True).drop(columns="seed")
    A["frac_top"] = A.n_top / A.n_genes
    A["infl_per_gene"] = A.influence / A.n_genes
    out = cfg.paths.output_dir / "step48_tf_influence"
    out.mkdir(parents=True, exist_ok=True)
    A.sort_values("influence", ascending=False).to_csv(out / f"tf_influence_{a.setup}.csv")

    n = len(A)
    print(f"\n=== readout-layer TF influence, setup={a.setup}, {a.variant}, "
          f"{len(frames)} seeds, {n} TFs ===\n")
    print("TOP 12 by total influence (sum |weight x delta| over the TF's genes):")
    print(A.sort_values("influence", ascending=False).head(12).round(4).to_string())

    print(f"\n=== where does {FOCUS} rank? (1 = most extreme) ===")
    for col, hi in [("n_genes", True), ("med_w", True), ("delta", True),
                    ("influence", True), ("n_top", True), ("frac_top", True),
                    ("infl_per_gene", True)]:
        v = A[col].abs() if col == "delta" else A[col]
        r = int((v > v[FOCUS]).sum()) + 1
        print(f"   {col:14s} {A.loc[FOCUS, col]:+10.4f}   rank {r:4d} / {n}"
              f"   ({'top' if r <= n * 0.01 else 'not top'} 1%)")

    big = A[A.n_genes >= 20]
    print(f"\n=== the control: among the {len(big)} TFs with >= 20 genes ===")
    print(f"   frac_top      {FOCUS} {A.loc[FOCUS,'frac_top']:.3f}  vs median "
          f"{big.frac_top.median():.3f}  (rank {int((big.frac_top > A.loc[FOCUS,'frac_top']).sum())+1}/{len(big)})")
    print(f"   infl_per_gene {FOCUS} {A.loc[FOCUS,'infl_per_gene']:.5f}  vs median "
          f"{big.infl_per_gene.median():.5f}  "
          f"(rank {int((big.infl_per_gene > A.loc[FOCUS,'infl_per_gene']).sum())+1}/{len(big)})")
    print(f"   |delta|       {FOCUS} {abs(A.loc[FOCUS,'delta']):.5f}  vs median "
          f"{big.delta.abs().median():.5f}  "
          f"(rank {int((big.delta.abs() > abs(A.loc[FOCUS,'delta'])).sum())+1}/{len(big)})")
    print(f"\nwrote {out / f'tf_influence_{a.setup}.csv'}")


if __name__ == "__main__":
    main()
