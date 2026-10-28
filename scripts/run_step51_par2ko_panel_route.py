"""Step 51 (2026-10-27): PAR2 KO on the ERK -| BCL2 route -- the 17-gene panel and the route ranks.

Run on the models retrained after the 2026-10-27 T0 rewrite (ERK -| Bcl2/Bcl2l1, TLR4 out of T0).

PART A - the 17-gene panel (mRNA readout). There is NO PAR2-KO RNA-seq for these genes in this
project. The only measurement is the collaborators' DY+LPS vs LPS RNA-seq, where all of them have
negative log2FC (curation/dynasore_panel_prereg_2026-10-20.md). Two perturbations are simulated,
both under LPS (the collaborators' background):
  * PAR2 KO      F2rl1 node KO (kappa = step13.ko_kappa), what the user asked for
  * dynasore     the step-43 edge cut (Arrb1/2 -> RAF/MEK/ERK, Tlr4 -> Ticam1/2), the matched one
The test is step 43's: a sign count is not enough when most of the transcriptome moves the same
way, so the panel's median percentile is compared with a permutation null over the transcriptome.

Not independent of the model revisions, and so reported apart from the scored set:
  Sting1        the 2026-10-27 T0 rewrite was made with its direction in view
  Il1b, Il6     PAR2_29 scored holdouts

PART B - the route. Signalling-node (activity) change under PAR2 KO, and its rank by |delta| among
all signalling nodes (rank 1 = largest change). Ranks are partly BUILT IN: T0 edges are exempt from
the stability projection, so T0 nodes carry more signal by construction.

    python scripts/run_step51_par2ko_panel_route.py                 # R4 and R0, 5 seeds each
    python scripts/run_step51_par2ko_panel_route.py --variants R4
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
from lembas_par2.core import simulate

torch.set_grad_enabled(False)

PANEL = ["Ikbke", "Ccl5", "Cxcl10", "Irf7", "Samhd1", "Mefv", "Ccl4", "Aim2", "Ifi204", "Il33",
         "Nlrp3", "Zbp1", "Sting1", "Nfkbia", "Il1b", "Il6", "Nfkbib"]
NOT_INDEPENDENT = {"Sting1": "T0 rewrite made with its direction in view",
                   "Il1b": "PAR2_29 scored holdout", "Il6": "PAR2_29 scored holdout"}
ROUTE = [("Raf1", "RAF1"), ("Map2k1", "MEK1"), ("Map2k2", "MEK2"), ("Mapk3", "ERK1"),
         ("Mapk1", "ERK2"), ("Bcl2l11", "BIM"), ("Mcl1", "MCL1"), ("Gsk3b", "GSK3B"),
         ("Bad", "BAD"), ("Bcl2", "BCL2"), ("Bcl2l1", "BCL-xL"), ("Bax", "BAX"), ("Bak1", "BAK1"),
         ("SE_CYTO_MTDNA", "mtDNA"), ("Cgas", "cGAS"), ("Sting1", "STING"), ("Tbk1", "TBK1"),
         ("Irf3", "IRF3")]
ENDO_ERK = [(a, t) for a in ("Arrb1", "Arrb2") for t in ("Raf1", "Map2k1", "Map2k2", "Mapk1", "Mapk3")]
TRIF = [("Tlr4", "Ticam1"), ("Tlr4", "Ticam2")]


def run_one(m, cfg, inputs, kos=None, cut=None):
    n = len(m.nodes)
    u = torch.zeros(n, 1, dtype=m.dtype)
    for x in inputs:
        u[m.idx[x], 0] = cfg.perturbation["input_level"]
    w0 = m.weights()
    S0 = torch.ones(n, 1, dtype=m.dtype)
    X0 = simulate(m.struct, w0, m.bias, u, S0, max_iter=3000, tol=1e-8)
    S1, w1 = S0.clone(), w0
    for node, kap in (kos or {}).items():
        S1[m.idx[node], 0] = 1.0 - kap
    if cut:
        key = {(p, q): i for i, (p, q) in enumerate(zip(m.edges.source, m.edges.target))}
        w1 = w0.clone()
        w1[[key[p] for p in cut if p in key]] = 0.0
    X1 = simulate(m.struct, w1, m.bias, u, S1, max_iter=3000, tol=1e-8)
    d = (S1 * X1 - S0 * X0)[:, 0]
    g = torch.zeros(len(m.genes), dtype=d.dtype).index_add(
        0, m.r_gene, m.readout_weights() * d[m.r_tf])
    return d.numpy(), g.numpy()


def pctile(x):
    p = np.empty_like(x)
    p[np.argsort(x)] = np.linspace(0, 100, len(x))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["R4", "R0"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--perm-seed", type=int, default=20261027)
    a = ap.parse_args()

    cfg = load_config()
    lps = cfg.step13["contexts"]["lps"]
    kap = cfg.step13["ko_kappa"]
    setups = {"PAR2 KO": dict(kos={"F2rl1": kap}), "dynasore": dict(cut=ENDO_ERK + TRIF)}
    panel_rows, route_rows, bg = [], [], []
    for v in a.variants:
        for s in a.seeds:
            run = v if s == 0 else f"{v}_s{s}"
            ck = cfg.paths.output_dir / "source_variants" / run / "checkpoint.pt"
            if not ck.exists():
                print(f"[skip] {run}")
                continue
            try:
                m, _ = load_model(cfg, ck, "cpu")
            except (RuntimeError, ValueError) as exc:
                sys.exit(f"{run}: checkpoint does not match the network ({exc}).\nRetrain first:\n"
                         f"  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
            gidx = {g: i for i, g in enumerate(m.genes)}
            for lab, kw in setups.items():
                d, g = run_one(m, cfg, lps, **kw)
                gp = pctile(g)
                bg.append(dict(variant=v, seed=s, setup=lab, frac_neg=float((g < 0).mean()), pct=gp))
                for gene in PANEL:
                    if gene in gidx:
                        panel_rows.append(dict(variant=v, seed=s, setup=lab, gene=gene,
                                               mrna=float(g[gidx[gene]]), pctile=float(gp[gidx[gene]])))
                if lab == "PAR2 KO":
                    rank = np.empty(len(d), dtype=int)
                    rank[np.argsort(-np.abs(d))] = np.arange(1, len(d) + 1)
                    for node, label in ROUTE:
                        i = m.idx[node]
                        route_rows.append(dict(variant=v, seed=s, node=node, label=label,
                                               delta=float(d[i]), rank=int(rank[i]), n_nodes=len(d)))
            print(f"  {run}: ok", flush=True)
    if not panel_rows:
        sys.exit("no checkpoints found. Train first:" 
                 "
  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")

    P = pd.DataFrame(panel_rows)
    R = pd.DataFrame(route_rows)
    out = cfg.paths.output_dir / "step51_par2ko"
    out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "panel_per_seed.csv", index=False)
    R.to_csv(out / "route_per_seed.csv", index=False)
    pd.set_option("display.width", 250)
    rep = {}

    print("\n" + "=" * 100 + "\nPART A - 17-gene panel, mRNA, LPS context. Observed (DY+LPS vs LPS): all DOWN\n")
    for v in a.variants:
        for lab in setups:
            sub = P[(P.variant == v) & (P.setup == lab)]
            agg = sub.groupby("gene").agg(mrna=("mrna", "median"), pctile=("pctile", "median"),
                                          seeds_neg=("mrna", lambda x: int((x < 0).sum())),
                                          seeds=("mrna", "size")).reindex(PANEL).dropna(how="all")
            agg["scored"] = [g not in NOT_INDEPENDENT for g in agg.index]
            agg["note"] = [NOT_INDEPENDENT.get(g, "") for g in agg.index]
            sc = agg[agg.scored]
            b = [x for x in bg if x["variant"] == v and x["setup"] == lab]
            bgneg = float(np.median([x["frac_neg"] for x in b]))
            rng = np.random.default_rng(a.perm_seed)
            obs = float(sc.pctile.median())
            null = np.array([np.median(rng.choice(b[0]["pct"], size=len(sc), replace=False))
                             for _ in range(a.n_perm)])
            p = float((null <= obs).mean())
            nneg = int((sc.mrna < 0).sum())
            print(f"--- {v} | {lab} ---")
            print(agg.sort_values("mrna").to_string(float_format=lambda x: f"{x:+.5f}"))
            print(f"  scored {len(sc)} genes: {nneg}/{len(sc)} negative (background {bgneg*100:.1f}% "
                  f"of the transcriptome negative); median percentile {obs:.1f} (50 = background), "
                  f"permutation p {p:.4f}\n")
            agg.to_csv(out / f"panel_{v}_{lab.replace(' ', '_')}.csv")
            rep[f"{v}|{lab}"] = dict(n_scored=int(len(sc)), n_neg=nneg, background_frac_neg=bgneg,
                                     median_pctile=obs, p_perm=p)

    print("=" * 100 + "\nPART B - route under PAR2 KO (LPS), signalling activity, rank by |delta| "
          "among all signalling nodes (1 = largest)\n")
    for v in a.variants:
        sub = R[R.variant == v]
        agg = sub.groupby(["node", "label"], sort=False).agg(
            delta=("delta", "median"), rank=("rank", "median"),
            rank_min=("rank", "min"), rank_max=("rank", "max"),
            seeds_same_sign=("delta", lambda x: int((np.sign(x) == np.sign(np.median(x))).sum()))
        ).reset_index()
        agg["direction"] = np.where(agg.delta > 0, "UP", "DOWN")
        agg["rank_by_|delta|_within_route"] = agg.delta.abs().rank(ascending=False).astype(int)
        print(f"--- {v} (of {int(sub.n_nodes.iloc[0])} signalling nodes) ---")
        print(agg[["label", "node", "direction", "delta", "seeds_same_sign", "rank", "rank_min",
                   "rank_max", "rank_by_|delta|_within_route"]]
              .to_string(index=False, float_format=lambda x: f"{x:+.4f}" if abs(x) < 1 else f"{x:.0f}"))
        agg.to_csv(out / f"route_{v}.csv", index=False)
        print()
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
