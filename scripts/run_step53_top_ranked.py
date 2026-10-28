"""Step 53 (2026-10-27): top-ranked nodes and genes under PAR2 KO (LPS context), per seed.

Same setup as steps 51/52 (F2rl1 KO, kappa = step13.ko_kappa, LPS inputs). Ranks by |delta|:
signalling activity over all signalling nodes, and mRNA readout over all measured genes.
Order = median |delta| over the 5 seeds.

Writes outputs/step53_top_ranked/{signal,mrna}_<variant>.csv (all nodes/genes, seeds 0-4 + median)

    python scripts/run_step53_top_ranked.py
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


def table(cols, names, top):
    D = pd.DataFrame(cols, index=names)
    D.columns = [f"seed{c}" for c in D.columns]
    D["median"] = D.median(axis=1)
    D["seeds_same_sign"] = (np.sign(D.filter(like="seed")).eq(np.sign(D["median"]), axis=0)).sum(axis=1)
    R = D.filter(like="seed").drop(columns="seeds_same_sign").abs().rank(ascending=False).astype(int)
    D["rank_median_|delta|"] = D["median"].abs().rank(ascending=False, method="min").astype(int)
    D["rank_best_seed"], D["rank_worst_seed"] = R.min(axis=1), R.max(axis=1)
    return D.sort_values("rank_median_|delta|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["R4", "R0"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    cfg = load_config()
    kap = cfg.step13["ko_kappa"]
    out = cfg.paths.output_dir / "step53_top_ranked"
    out.mkdir(parents=True, exist_ok=True)
    for v in a.variants:
        sig, mr = {}, {}
        for sd in a.seeds:
            run = v if sd == 0 else f"{v}_s{sd}"
            ck = cfg.paths.output_dir / "source_variants" / run / "checkpoint.pt"
            if not ck.exists():
                print(f"[skip] {run}")
                continue
            m, _ = load_model(cfg, ck, "cpu")
            n = len(m.nodes)
            u = torch.zeros(n, 1, dtype=m.dtype)
            for x in cfg.step13["contexts"]["lps"]:
                u[m.idx[x], 0] = cfg.perturbation["input_level"]
            S = torch.ones(n, 2, dtype=m.dtype)
            S[m.idx["F2rl1"], 1] = 1 - kap
            X = simulate(m.struct, m.weights(), m.bias, u.repeat(1, 2), S, max_iter=3000, tol=1e-8)
            d = (S * X)[:, 1] - (S * X)[:, 0]
            g = torch.zeros(len(m.genes), dtype=d.dtype).index_add(0, m.r_gene, m.readout_weights() * d[m.r_tf])
            sig[sd], mr[sd] = d.numpy(), g.numpy()
            nodes, genes = list(m.nodes), list(m.genes)
        for lab, cols, names in (("signal", sig, nodes), ("mrna", mr, genes)):
            T = table(cols, names, a.top)
            T.to_csv(out / f"{lab}_{v}.csv", index_label="node" if lab == "signal" else "gene")
            print(f"\n### {v} | {lab} (top {a.top} of {len(T)})")
            print(T.head(a.top).to_string(float_format=lambda x: f"{x:+.4f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
