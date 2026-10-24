"""Step 43: the dynasore 19-gene panel test.

PRE-REGISTERED IN FULL BEFORE RUNNING: curation/dynasore_panel_prereg_2026-10-20.md.
Read it first. The decision rule below is copied from it and must not be edited to fit the output.

Collaborators' contrast: DY+LPS vs LPS, all 19 genes negative log2FC.

Only the 13 PRIMARY genes are scored -- those independent of every revision made in this project.
Sting1 (T0 set rebuilt, step 38), Nfkbia (promoted to T0, step 33), Il1b and Il6 (PAR2_29 scored
holdouts) are REPORTED BUT NOT SCORED; Mlkl and Ripk3 have no measured expression.

WHY A SIGN TEST IS NOT ENOUGH: ~80% of the measured transcriptome moves negative under this cut,
so "all 13 negative" has probability ~0.8^13 = 5.5% under the background and would be true of
almost any gene set. The test is POSITIONAL: median percentile of the 13 against a permutation
null drawn from the measured transcriptome.

    python scripts/run_step43_dynasore_panel.py
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

PRIMARY = ["Cxcl10", "Ccl4", "Mefv", "Nlrp3", "Ifi204", "Zbp1", "Ikbke",
           "Irf7", "Aim2", "Samhd1", "Nfkbib", "Il33", "Ccl5"]
REPORTED_NOT_SCORED = {"Sting1": "TUNED (step 38 rebuilt its T0 set)",
                       "Nfkbia": "TOUCHED (step 33 promoted it to T0)",
                       "Il1b": "PAR2_29 scored holdout",
                       "Il6": "PAR2_29 scored holdout"}
ENDO_ERK = [(a, t) for a in ("Arrb1", "Arrb2")
            for t in ("Raf1", "Map2k1", "Map2k2", "Mapk1", "Mapk3")]
TRIF = [("Tlr4", "Ticam1"), ("Tlr4", "Ticam2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="R4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--perm-seed", type=int, default=20261020)
    a = ap.parse_args()

    cfg = load_config()
    lps = cfg.step13["contexts"]["lps"]
    per_seed, frac_neg = [], []
    for s in a.seeds:
        run = a.variant if s == 0 else f"{a.variant}_s{s}"
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
        S1 = torch.ones(n, 1, dtype=m.dtype)
        key = {(p, q): i for i, (p, q) in enumerate(zip(m.edges.source, m.edges.target))}
        X0 = simulate(m.struct, m.weights(), m.bias, u, S1, max_iter=3000, tol=1e-8)
        w = m.weights().clone()
        cut = [key[p] for p in ENDO_ERK + TRIF if p in key]
        w[cut] = 0.0
        X1 = simulate(m.struct, w, m.bias, u, S1, max_iter=3000, tol=1e-8)
        d = (X1 - X0)[:, 0]
        gp = torch.zeros(len(m.genes), dtype=d.dtype).index_add(
            0, m.r_gene, m.readout_weights() * d[m.r_tf]).numpy()
        gidx = {g: i for i, g in enumerate(m.genes)}
        order = np.argsort(gp)
        pct = np.empty_like(gp)
        pct[order] = np.linspace(0, 100, len(gp))
        frac_neg.append(float((gp < 0).mean()))
        per_seed.append(dict(seed=s, gp=gp, gidx=gidx, pct=pct, n_genes=len(gp)))
        print(f"  {run}: {len(cut)} edges cut, {frac_neg[-1]*100:.1f}% of transcriptome negative",
              flush=True)

    if not per_seed:
        sys.exit("no checkpoints found. Train first:" 
                 "\n  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
    ns = len(per_seed)
    bg = float(np.median(frac_neg))

    rows = []
    for g in PRIMARY + list(REPORTED_NOT_SCORED):
        vals = [(r["gp"][r["gidx"][g]], r["pct"][r["gidx"][g]]) for r in per_seed
                if g in r["gidx"]]
        if not vals:
            continue
        mr = float(np.median([v[0] for v in vals]))
        pc = float(np.median([v[1] for v in vals]))
        rows.append(dict(gene=g, scored=g in PRIMARY, mrna=mr, pctile=pc,
                         seeds_neg=int(sum(v[0] < 0 for v in vals)), seeds=len(vals),
                         note="" if g in PRIMARY else REPORTED_NOT_SCORED[g]))
    D = pd.DataFrame(rows)
    out = cfg.paths.output_dir / "step43_dynasore_panel"
    out.mkdir(parents=True, exist_ok=True)
    D.to_csv(out / "panel.csv", index=False)

    print(f"\n=== background: {bg*100:.1f}% of the measured transcriptome is negative "
          f"({ns} seeds) ===")
    print(f"    so 'all 13 negative' alone has p ~ {bg**13:.3f} -- NOT the test\n")
    P = D[D.scored]
    print("=== PRIMARY (scored): 13 genes independent of every revision ===")
    print(P[["gene", "mrna", "pctile", "seeds_neg", "seeds"]].to_string(index=False))
    print("\n=== REPORTED, NOT SCORED ===")
    print(D[~D.scored][["gene", "mrna", "pctile", "seeds_neg", "note"]].to_string(index=False))

    # permutation null on the median percentile
    rng = np.random.default_rng(a.perm_seed)
    obs = float(P.pctile.median())
    ref = per_seed[0]
    null = np.array([np.median(rng.choice(ref["pct"], size=len(P), replace=False))
                     for _ in range(a.n_perm)])
    p = float((null <= obs).mean())
    nneg = int((P.mrna < 0).sum())

    print(f"\n=== THE TEST (pre-registered) ===")
    print(f"  median percentile of the 13 : {obs:.2f}   (50 = indistinguishable from background)")
    print(f"  permutation p ({a.n_perm} draws, seed {a.perm_seed}) : {p:.4f}")
    print(f"  secondary: {nneg}/{len(P)} negative, background rate {bg*100:.1f}%")
    if obs < 50 and p < 0.05:
        v = ("PASS -- the panel sits in the negative tail beyond background. First out-of-sample "
             "support for the dynasore mechanism.")
    elif obs < 50:
        v = ("WEAK/NULL -- directionally consistent but not distinguishable from background. "
             "Does NOT support the mechanism.")
    else:
        v = "FAIL -- the model does not reproduce the panel. That is the result."
    print(f"\n  VERDICT: {v}")
    print("\n  THE MODEL IS FROZEN. No tier change, edge drop, sign change or retrain may be made "
          "in response to this result (prereg section 6). A PASS does NOT validate the "
          "PAR2->STING direction, the TLR4 promotion or the Sting1 T0 set.")
    (out / "report.json").write_text(json.dumps(dict(
        median_pctile=obs, p_perm=p, n_neg=nneg, n_scored=int(len(P)),
        background_frac_neg=bg, verdict=v, seeds=[int(r["seed"]) for r in per_seed]), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
