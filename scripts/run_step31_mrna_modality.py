"""Step 31: the endosomal-KO comparison, in the modality and the perturbation actually used.

WHY THIS EXISTS. The collaborators' endosomal result is RNA-seq (Sting1 and Bak1 both
log2FC ~ -1) from a block of Arrb1/2 on the endosome. Until 2026-10-19 the record compared it
against the `Bak1` / `Sting1` SIGNAL nodes (which are activity, not transcript) under an F2rl1 KO
(not the perturbation performed), and concluded from the mismatch that the ERK -| BCL2 arm had a
sign error. Both halves of that comparison were wrong. This script runs the matched perturbation
in the matched modality so the claim is testable.

WHAT IT PRINTS, and why each column is needed:
  * mRNA delta    - gene readout = readout_weights x delta(TF activity), summed over TF -> gene
                    edges. The same construction `tests/check_readout_param.py` uses.
  * pctile        - the gene's rank among ALL measured genes, 0 = most negative. REQUIRED, because
                    most of the transcriptome moves down under these KOs (69 % under F2rl1, 79 %
                    under Arrb1), so a negative sign on its own is background, not a prediction.
  * signal        - the activity node, for the contrast that is the whole point of the step.
  * seeds_neg     - how many seeds agree on the sign. A median that hides a sign flip is not a
                    result; the F2rl1-KO Sting1 number flips in seed 2 and must not be quoted.

DOES NOT VALIDATE THE CENTRAL CLAIM. PAR2_29 measured pSTING protein under a PAR2 edit, which maps
to the SIGNAL node under F2rl1 KO. That stays positive and stays contradicted. Transcript-level
agreement here is a different modality and a different perturbation.

    python scripts/run_step31_mrna_modality.py
    python scripts/run_step31_mrna_modality.py --variant R0 --seeds 0 1 2
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

# the mitochondrial/STING arm under dispute, plus the PAR2_29 outputs the model gets right
FOCUS = ["Sting1", "Bak1", "Cgas", "Bax", "Bcl2l11", "Mcl1", "Bcl2", "Bcl2l1",
         "Mapk1", "Mapk3", "Src", "Vdac1", "Ifnb1", "Tlr4", "Il1b", "Il6", "Nlrp3"]
KO_NODES = ["F2rl1", "Arrb1", "Arrb2"]


@torch.no_grad()
def gene_delta(m, d):
    """Transcript-level delta: TF activity change pushed through the readout layer."""
    contrib = m.readout_weights() * d[m.r_tf]
    return torch.zeros(len(m.genes), dtype=contrib.dtype).index_add(0, m.r_gene, contrib)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="R4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--context", default="lps")
    a = ap.parse_args()

    cfg = load_config()
    out = cfg.paths.output_dir / "step31_mrna"
    out.mkdir(parents=True, exist_ok=True)
    ctx = cfg.step13["contexts"][a.context]
    kappa = cfg.step13["ko_kappa"]

    rows, bg, missing = [], [], []
    for s in a.seeds:
        run = a.variant if s == 0 else f"{a.variant}_s{s}"
        ck = cfg.paths.output_dir / "source_variants" / run / "checkpoint.pt"
        if not ck.exists():
            missing.append(run)
            continue
        try:
            m, _ = load_model(cfg, ck, "cpu")
        except (RuntimeError, ValueError) as exc:
            # a curation change moves edges between tiers and the saved shapes stop matching;
            # say which command fixes it rather than raising from inside the loader.
            sys.exit(f"{run}: checkpoint does not match the current network ({exc}). "
                     f"Retrain first: python scripts/run_step23_retrain.py")
        gidx = {g: i for i, g in enumerate(m.genes)}
        nidx = {n: i for i, n in enumerate(m.nodes)}
        for ko in [n for n in KO_NODES if n in m.idx]:
            d = ko_delta(m, ctx, kappa, node=ko)
            gp = gene_delta(m, d).numpy()
            bg.append(dict(seed=s, ko=ko, frac_neg=float((gp < 0).mean()),
                           median_all=float(np.median(gp)), n_genes=len(gp)))
            order = np.argsort(gp)
            pct = np.empty_like(gp)
            pct[order] = np.linspace(0, 100, len(gp))
            for g in FOCUS:
                if g not in gidx:
                    continue
                rows.append(dict(seed=s, ko=ko, gene=g, mrna=float(gp[gidx[g]]),
                                 pctile=float(pct[gidx[g]]),
                                 signal=float(d[nidx[g]]) if g in nidx else np.nan))
        print(f"  {run}: {len([n for n in KO_NODES if n in m.idx])} KOs", flush=True)

    if missing:
        print(f"[warn] no checkpoint for {missing}")
    if not rows:
        sys.exit(f"no {a.variant} checkpoints; run scripts/run_step23_retrain.py")

    D = pd.DataFrame(rows)
    B = pd.DataFrame(bg)
    D.to_csv(out / "focus_deltas.csv", index=False)
    B.to_csv(out / "background.csv", index=False)

    nseed = D.seed.nunique()
    print(f"\n=== whole-transcriptome background ({a.variant}, {a.context}, median of "
          f"{nseed} seeds) ===")
    print("a negative sign alone is NOT a prediction -- read the percentiles below against this")
    print(B.groupby("ko").median(numeric_only=True).drop(columns="seed").round(5).to_string())

    summary = {}
    for ko in D.ko.unique():
        sub = D[D.ko == ko]
        agg = sub.groupby("gene").agg(
            mrna=("mrna", "median"), pctile=("pctile", "median"),
            signal=("signal", "median"),
            seeds_neg=("mrna", lambda x: int((x < 0).sum())))
        agg = agg.reindex([g for g in FOCUS if g in agg.index])
        agg["seeds"] = nseed
        print(f"\n=== KO {ko} ({a.context}) ===")
        print(agg.round(5).to_string())
        summary[ko] = {g: dict(mrna=float(r.mrna), pctile=float(r.pctile),
                               signal=float(r.signal), seeds_neg=int(r.seeds_neg))
                       for g, r in agg.iterrows()}

    rep = dict(variant=a.variant, context=a.context, seeds=sorted(int(s) for s in D.seed.unique()),
               background=B.groupby("ko").median(numeric_only=True).drop(
                   columns="seed").to_dict("index"),
               focus=summary,
               note="PAR2_29 measured pSTING (activity) under a PAR2 edit -> compare the SIGNAL "
                    "node under F2rl1 KO, which is positive and contradicted. The transcript "
                    "agreement under Arrb1 KO is a different modality and perturbation.")
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
