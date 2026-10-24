"""Step 49: data for the two SI panels on the scored misses.

SI-1 (the chain)    why Pparg swings: the T0 membrane arm drives PKCa down, ONE UNSIGNED T2 edge
                    (Prkca -> Pparg, PSP phosphorylation with no site function, sign LEARNED at
                    -0.147) converts that into Pparg UP, and Pparg then reaches 388 regulon genes.
SI-2 (the symptoms) the two misses that chain produces -- Ifnb1 (PAR2_30) and Nfkbib (dynasore).

Writes outputs/step49_si_misses/
    chain_nodes.csv        the F2rl1 -> ... -> Pparg walk, per seed
    pparg_in_edges.csv     Pparg's in-edge contributions, per seed
    ifnb1_layers.csv       Ifnb1 signal node vs mRNA readout, per seed
    ifnb1_regulon.csv      Ifnb1's 44 TF -> gene readout contributions, per seed
    nfkbib_regulon.csv     Nfkbib's entire 3 TF regulon, per seed
    regulon_sizes.csv      TF-edge count for every measured gene (for the sparse-tail histogram)

The TF influence table SI-1 also needs comes from scripts/run_step48_tf_influence.py.

    python scripts/run_step48_tf_influence.py
    python scripts/run_step49_si_misses.py
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

# the walk from the receptor to Pparg. Everything up to Prkca is T0; the Prkca -> Pparg step is
# the unsigned T2 edge whose sign was fitted, and it is where the sign flips.
CHAIN = [("PAR2", "F2rl1"), ("Gq", "Gnaq"), ("PLCB1", "Plcb1"), ("DAG", "SE_DAG"),
         ("PKCa", "Prkca"), ("PPARg", "Pparg")]
ENDO_ERK = [(a, t) for a in ("Arrb1", "Arrb2")
            for t in ("Raf1", "Map2k1", "Map2k2", "Mapk1", "Mapk3")]
TRIF = [("Tlr4", "Ticam1"), ("Tlr4", "Ticam2")]


@torch.no_grad()
def main():
    cfg = load_config()
    kap = cfg.step13["ko_kappa"]
    out = cfg.paths.output_dir / "step49_si_misses"
    out.mkdir(parents=True, exist_ok=True)
    chain, pin, lay, ifr, nfr, sizes = [], [], [], [], [], []

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
        rtf, rg = m.r_tf.numpy(), m.r_gene.numpy()
        rw = m.readout_weights().numpy()
        gpos = {g: i for i, g in enumerate(m.genes)}

        # --- setup A: pure PAR2 KO, no TLR4 input (PAR2_26 / PAR2_30) ---
        U = torch.zeros(n, 2, dtype=m.dtype)
        S = torch.ones(n, 2, dtype=m.dtype)
        S[m.idx["F2rl1"], 1] = 1 - kap
        X = simulate(m.struct, m.weights(), m.bias, U, S, max_iter=3000, tol=1e-8)
        dA = ((S * X)[:, 1] - (S * X)[:, 0]).numpy()
        gA = np.zeros(len(m.genes))
        np.add.at(gA, rg, rw * dA[rtf])

        for order, (lab, nd) in enumerate(CHAIN):
            tier = m.edges[(m.edges.target == nd)].tier
            chain.append(dict(seed=sd, order=order, label=lab, node=nd, delta=float(dA[m.idx[nd]])))
        W = m.weights().numpy()
        for i, (s_, t_) in enumerate(zip(m.edges.source, m.edges.target)):
            if t_ == "Pparg":
                pin.append(dict(seed=sd, source=s_, tier=m.edges.tier.iloc[i], w=float(W[i]),
                                d_source=float(dA[m.idx[s_]]),
                                c=float(W[i]) * float(dA[m.idx[s_]])))
        lay.append(dict(seed=sd, layer="signal node", value=float(dA[m.idx["Ifnb1"]])))
        lay.append(dict(seed=sd, layer="mRNA readout", value=float(gA[gpos["Ifnb1"]])))
        for e in np.where(rg == gpos["Ifnb1"])[0]:
            ifr.append(dict(seed=sd, tf=m.nodes[rtf[e]], w=float(rw[e]),
                            d_tf=float(dA[rtf[e]]), c=float(rw[e] * dA[rtf[e]])))

        # --- dynasore, for Nfkbib ---
        u = torch.zeros(n, 1, dtype=m.dtype)
        for x in cfg.step13["contexts"]["lps"]:
            u[m.idx[x], 0] = cfg.perturbation["input_level"]
        S1 = torch.ones(n, 1, dtype=m.dtype)
        key = {(p, q): i for i, (p, q) in enumerate(zip(m.edges.source, m.edges.target))}
        X0 = simulate(m.struct, m.weights(), m.bias, u, S1, max_iter=3000, tol=1e-8)
        w2 = m.weights().clone()
        w2[[key[p] for p in ENDO_ERK + TRIF if p in key]] = 0.0
        X1 = simulate(m.struct, w2, m.bias, u, S1, max_iter=3000, tol=1e-8)
        dD = (X1 - X0)[:, 0].numpy()
        for e in np.where(rg == gpos["Nfkbib"])[0]:
            nfr.append(dict(seed=sd, tf=m.nodes[rtf[e]], w=float(rw[e]),
                            d_tf=float(dD[rtf[e]]), c=float(rw[e] * dD[rtf[e]])))
        if sd == 0:
            cnt = pd.Series(rg).value_counts()
            sizes = [dict(gene=m.genes[i], n_tf=int(c)) for i, c in cnt.items()]
        print(f"  {run}: ok", flush=True)

    if not chain:
        sys.exit("no checkpoints found. Train first:" 
                 "\n  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
    for df, nm in ((pd.DataFrame(chain), "chain_nodes"),
                   (pd.DataFrame(pin), "pparg_in_edges"),
                   (pd.DataFrame(lay), "ifnb1_layers"),
                   (pd.DataFrame(ifr), "ifnb1_regulon"),
                   (pd.DataFrame(nfr), "nfkbib_regulon"),
                   (pd.DataFrame(sizes), "regulon_sizes")):
        df.to_csv(out / f"{nm}.csv", index=False)
        print(f"    {nm}.csv  {len(df)} rows")

    C = pd.DataFrame(chain).groupby(["order", "label"]).delta.median()
    print("\nchain (median of seeds):")
    print("   " + "  ->  ".join(f"{l} {v:+.3f}" for (_, l), v in C.items()))
    P = pd.DataFrame(pin).groupby("source").c.median().sort_values(key=abs, ascending=False)
    print(f"\nPparg top driver: {P.index[0]} {P.iloc[0]:+.5f} "
          f"= {P.iloc[0] / P.sum() * 100:.0f}% of the net {P.sum():+.5f}")
    L = pd.DataFrame(lay).groupby("layer").value.median()
    print(f"\nIfnb1  signal {L['signal node']:+.5f} (observed UP)  |  "
          f"mRNA {L['mRNA readout']:+.5f}")
    N = pd.DataFrame(nfr).groupby("tf").c.median().sort_values(key=abs, ascending=False)
    print(f"Nfkbib regulon = {list(N.index)}  net {N.sum():+.5f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
