"""Step 52 (2026-10-27): per-seed figure data for the ERK -| BCL2 route, and the Nfkbib miss.

All under the step-51 setup: PAR2 KO (F2rl1, kappa = step13.ko_kappa) in the LPS context, on the
models retrained after the 2026-10-27 T0 rewrite. One column per seed, so figures can show all 5.

Writes outputs/step52_figure_data/
    route_<variant>.csv           route nodes (RAF1 ... IRF3), signalling activity, seeds 0-4,
                                  plus whole-network rank by |delta| per seed
    pparg_chain_<variant>.csv     PAR2 -> Gq -> PLCB1 -> DAG -> PKCa -> PPARg, seeds 0-4
    pparg_in_edges_<variant>.csv  every in-edge of Pparg: w x delta(source), seeds 0-4
    nfkbib_regulon_<variant>.csv  every TF -> Nfkbib readout edge: w x delta(TF), seeds 0-4

Contributions are first-order (edge weight x source delta), the step-40/49 convention: they share
out the INPUT to a node, not its output after the activation function, so they are shares, not
exact decompositions of the node's delta.

    python scripts/run_step52_figure_data.py
    python scripts/run_step52_figure_data.py --variants R4
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

ROUTE = [("RAF1", "Raf1"), ("MEK1", "Map2k1"), ("MEK2", "Map2k2"), ("ERK1", "Mapk3"),
         ("ERK2", "Mapk1"), ("BIM", "Bcl2l11"), ("MCL1", "Mcl1"), ("GSK3B", "Gsk3b"),
         ("BAD", "Bad"), ("BCL2", "Bcl2"), ("BCL-xL", "Bcl2l1"), ("BAX", "Bax"), ("BAK1", "Bak1"),
         ("mtDNA", "SE_CYTO_MTDNA"), ("cGAS", "Cgas"), ("STING", "Sting1"), ("TBK1", "Tbk1"),
         ("IRF3", "Irf3")]
CHAIN = [("PAR2", "F2rl1"), ("Gq", "Gnaq"), ("PLCB1", "Plcb1"), ("DAG", "SE_DAG"),
         ("PKCa", "Prkca"), ("PPARg", "Pparg")]


def wide(rows, keys):
    D = pd.DataFrame(rows)
    W = D.pivot_table(index=keys, columns="seed", values="value", sort=False)
    W.columns = [f"seed{c}" for c in W.columns]
    W["median"] = W.median(axis=1)
    return W.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["R4", "R0"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    cfg = load_config()
    kap = cfg.step13["ko_kappa"]
    out = cfg.paths.output_dir / "step52_figure_data"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    for v in a.variants:
        route, rank, chain, pin, nfr = [], [], [], [], []
        for sd in a.seeds:
            run = v if sd == 0 else f"{v}_s{sd}"
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
            for x in cfg.step13["contexts"]["lps"]:
                u[m.idx[x], 0] = cfg.perturbation["input_level"]
            U = u.repeat(1, 2)
            S = torch.ones(n, 2, dtype=m.dtype)
            S[m.idx["F2rl1"], 1] = 1 - kap
            X = simulate(m.struct, m.weights(), m.bias, U, S, max_iter=3000, tol=1e-8)
            d = ((S * X)[:, 1] - (S * X)[:, 0]).numpy()
            rk = np.empty(n, dtype=int)
            rk[np.argsort(-np.abs(d))] = np.arange(1, n + 1)
            for lab, nd in ROUTE:
                route.append(dict(label=lab, node=nd, seed=sd, value=float(d[m.idx[nd]])))
                rank.append(dict(label=lab, node=nd, seed=sd, value=int(rk[m.idx[nd]])))
            for lab, nd in CHAIN:
                chain.append(dict(label=lab, node=nd, seed=sd, value=float(d[m.idx[nd]])))
            W = m.weights().numpy()
            E = m.edges
            for i in np.where(E.target.values == "Pparg")[0]:
                s_ = E.source.iloc[i]
                pin.append(dict(source=s_, tier=E.tier.iloc[i], seed=sd,
                                value=float(W[i]) * float(d[m.idx[s_]])))
            rtf, rg = m.r_tf.numpy(), m.r_gene.numpy()
            rw = m.readout_weights().numpy()
            g = m.genes.index("Nfkbib")
            for e in np.where(rg == g)[0]:
                nfr.append(dict(tf=m.nodes[rtf[e]], seed=sd, value=float(rw[e] * d[rtf[e]])))
            nfr.append(dict(tf="NET (mRNA prediction)", seed=sd,
                            value=float(sum(rw[e] * d[rtf[e]] for e in np.where(rg == g)[0]))))
            print(f"  {run}: ok", flush=True)
        if not route:
            continue

        R = wide(route, ["label", "node"])
        K = wide(rank, ["label", "node"])
        K.columns = [c if c in ("label", "node") else f"rank_{c}" for c in K.columns]
        R = R.merge(K, on=["label", "node"])
        R.to_csv(out / f"route_{v}.csv", index=False)
        C = wide(chain, ["label", "node"])
        C.to_csv(out / f"pparg_chain_{v}.csv", index=False)
        P = wide(pin, ["source", "tier"])
        net = P["median"].sum()
        P["share_of_net_%"] = 100 * P["median"] / net
        P = P.reindex(P["median"].abs().sort_values(ascending=False).index)
        P.to_csv(out / f"pparg_in_edges_{v}.csv", index=False)
        N = wide(nfr, ["tf"])
        N.to_csv(out / f"nfkbib_regulon_{v}.csv", index=False)

        f = lambda x: f"{x:+.5f}"
        print(f"\n################ {v} ################")
        print("ROUTE (activity delta per seed; rank = position by |delta| among all nodes)")
        print(R.to_string(index=False, float_format=f))
        print("\nPPARg CHAIN")
        print(C.to_string(index=False, float_format=f))
        print(f"\nPPARg IN-EDGE CONTRIBUTIONS (w x delta source), top 12 of {len(P)}; "
              f"net {net:+.5f}")
        print(P.head(12).to_string(index=False, float_format=f))
        print("\nNfkbib REGULON (readout w x delta TF)")
        print(N.to_string(index=False, float_format=f))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
