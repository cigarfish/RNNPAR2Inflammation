"""Is the route visible in the drug data? Per route gene and per STING-activity readout:
expression level, how many drugs change it (padj < 0.05), effect size, and whether the drug
effects on it are reproducible between disjoint replicate halves (correlation ACROSS DRUGS of
the half-1 vs half-2 log2FC, replicate batch removed as in step 12f0).
Then the route's own prediction in model-free form: MEK/ERK inhibition (the closest drug analogue
of losing PAR2 -> ERK signalling) should raise STING activity -> ISG / Irf3-target genes up.

    python tests/explore_route_visibility.py
Writes outputs/step12v/{genes.csv, programs.csv, mek_erk.csv}.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.noise_ceiling import batch_corrected, load_counts, log2fc, sample_groups, size_factors

ROUTE = ["F2rl1", "Gnaq", "Plcb1", "Plcb3", "Prkca", "Rasgrp1", "Rasgrp3", "Hras", "Kras", "Raf1", "Braf",
         "Map2k1", "Map2k2", "Mapk1", "Mapk3", "Rps6ka1", "Rps6ka3", "Gsk3b",
         "Mcl1", "Bad", "Bcl2l11", "Bcl2", "Bcl2l1", "Bax", "Bak1", "Cgas", "Sting1", "Tbk1", "Irf3"]
STING_OUT = ["Ifnb1", "Cxcl10", "Isg15", "Ifit1", "Ifit3", "Rsad2", "Mx1", "Oasl1", "Irf7", "Usp18", "Ccl5"]
MEK_ERK = ["Pimasertib", "Selumetinib", "Refametinib", "GDC0994", "Ulixertinib"]


def main():
    cfg = load_config()
    o = cfg.paths.output_dir
    out = o / "step12v"
    out.mkdir(parents=True, exist_ok=True)
    X = load_counts(cfg)
    N = X / size_factors(X)
    G = sample_groups(X.columns)
    Nc = batch_corrected(N, G)
    drugs = [g for g in G if g != "DMSO" and len(G[g]) == 3]
    ref = sorted(G["DMSO"])
    splits = [(list(a), [c for c in ref if c not in a]) for a in itertools.combinations(ref, 3) if ref[0] in a]
    genes = [g for g in ROUTE + STING_OUT if g in Nc.index]
    # reproducibility across drugs: within EACH disjoint split, correlate (across drugs) the
    # half-1 and half-2 log2FC of a gene; then average the per-split r (Fisher z). Averaging the
    # LFCs over splits first would make both halves ~ the full estimate and inflate r to ~1.
    combos = list(itertools.product(range(3), splits))
    zs = {g: [] for g in genes}
    for i, (ra, rb) in combos:
        a = pd.DataFrame({d: log2fc(Nc.loc[genes], [sorted(G[d])[i]], ra) for d in drugs})
        b = pd.DataFrame({d: log2fc(Nc.loc[genes], [r for j, r in enumerate(sorted(G[d])) if j != i], rb) for d in drugs})
        for g in genes:
            zs[g].append(np.arctanh(np.clip(np.corrcoef(a.loc[g], b.loc[g])[0, 1], -0.999999, 0.999999)))
    rel_of = {g: float(np.tanh(np.nanmean(v))) for g, v in zs.items()}
    deg_dir = cfg.root / cfg.conditions["drug_deg_dir"]
    lfc, padj, base = {}, {}, {}
    for d in drugs:
        t = pd.read_csv(deg_dir / f"{d}_DEG.csv").drop_duplicates("SYMBOL").set_index("SYMBOL")
        lfc[d], padj[d], base[d] = t.log2FoldChange, t.padj, t.baseMean
    L, P, B = pd.DataFrame(lfc), pd.DataFrame(padj), pd.DataFrame(base)
    rows = []
    for g in genes:
        rel = rel_of[g]
        rows.append(dict(gene=g, group="route" if g in ROUTE else "STING output",
                         baseMean=float(B.loc[g].median()) if g in B.index else np.nan,
                         n_drugs_padj05=int((P.loc[g] < 0.05).sum()) if g in P.index else 0,
                         median_abs_lfc=float(L.loc[g].abs().median()) if g in L.index else np.nan,
                         max_abs_lfc=float(L.loc[g].abs().max()) if g in L.index else np.nan,
                         reproducibility_across_drugs=float(rel)))
    gt = pd.DataFrame(rows)
    gt.to_csv(out / "genes.csv", index=False)

    # programs: ISG (STING / IFN output) and MPAS (ERK output) as gene-set means of z-scored LFC
    progs = dict(cfg.conditions["programs"])
    prow = []
    for name, gs in progs.items():
        gs = [g for g in gs if g in Nc.index]
        z = []
        for i, (ra, rb) in combos:
            a = [log2fc(Nc.loc[gs], [sorted(G[d])[i]], ra).mean() for d in drugs]
            b = [log2fc(Nc.loc[gs], [r for j, r in enumerate(sorted(G[d])) if j != i], rb).mean() for d in drugs]
            z.append(np.arctanh(np.clip(np.corrcoef(a, b)[0, 1], -0.999999, 0.999999)))
        prow.append(dict(program=name, n_genes=len(gs), reproducibility_across_drugs=float(np.tanh(np.mean(z)))))
    pt = pd.DataFrame(prow)
    pt.to_csv(out / "programs.csv", index=False)

    # MEK/ERK inhibitors: route prediction says STING activity (ISG) up, Sting1-arm genes ?
    tfa = pd.read_csv(o / "step06_tf_activity.csv.gz", index_col=0)
    prog = pd.read_csv(o / "step06_programs.csv", index_col=0)
    mrow = []
    for d in MEK_ERK + ["Trametinib"]:
        cid = f"GSE219229_{d}" if d != "Trametinib" else "GSE134486_Trametinib"
        r = dict(drug=d, MPAS=float(prog.at[cid, "MPAS"]) if cid in prog.index else np.nan,
                 ISG=float(prog.at[cid, "ISG"]) if cid in prog.index else np.nan,
                 Irf3_activity=float(tfa.at["Irf3", cid]) if cid in tfa.columns and "Irf3" in tfa.index else np.nan)
        for g in ["Sting1", "Cgas", "Bax", "Bak1", "Bcl2l11", "Mcl1", "Ifnb1", "Cxcl10"]:
            r[f"lfc_{g}"] = float(L.at[g, d]) if (d in L.columns and g in L.index) else np.nan
            r[f"padj_{g}"] = float(P.at[g, d]) if (d in P.columns and g in P.index) else np.nan
        mrow.append(r)
    mt = pd.DataFrame(mrow)
    mt.to_csv(out / "mek_erk.csv", index=False)
    # null for ISG: all other drugs
    others = [f"GSE219229_{d}" for d in drugs if d not in MEK_ERK]
    isg_null = prog.loc[[c for c in others if c in prog.index], "ISG"]

    pd.set_option("display.width", 200)
    print("== per gene (GSE219229, 56 drugs)")
    print(gt.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n== programs: reproducibility of drug effects across drugs")
    print(pt.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n== MEK/ERK inhibitors (route predicts: ERK output down, STING activity / ISG up)")
    cols = ["drug", "MPAS", "ISG", "Irf3_activity"] + [c for c in mt.columns if c.startswith("lfc_")]
    print(mt[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print(f"\nISG score across the other drugs: median {isg_null.median():+.2f}, IQR "
          f"[{isg_null.quantile(.25):+.2f}, {isg_null.quantile(.75):+.2f}]")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
