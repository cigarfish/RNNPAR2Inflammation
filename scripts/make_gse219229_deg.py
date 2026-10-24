"""DESeq2 (pydeseq2) for every GSE219229 drug: drug + LPS vs DMSO + LPS.

Input : curation/GSE219229.csv  (Symbol + integer counts; 6 DMSO + 3 replicates per drug)
Output: data_external/GSE219229_DEG/<Drug>_DEG.csv   SYMBOL,log2FoldChange,lfcSE,stat,pvalue,padj,baseMean
        data_external/GSE219229_DEG/summary.csv      per-drug n genes, n padj<0.05, Dusp6 log2FC

Design: one model over all 180 samples (~cond, all six DMSO pooled as reference
'DMSO'), then one Wald contrast per drug. Pooling dispersion estimates across all
samples is what reproduces the four published contrasts (log2FC r = 1.000 with
all six DMSO as control; a per-drug model gives the same log2FC but less power).
Duplicate symbols are summed; genes with < 10 total counts are dropped.

Run from the LEMBAS folder (needs pydeseq2: pip install pydeseq2):
    python scripts/make_gse219229_deg.py [n_cpus]
"""
import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "curation" / "GSE219229.csv"
OUT = ROOT / "data_external" / "GSE219229_DEG"
CONTROL = {"DMSOs1", "DMSOs2"}
REF = "DMSO"
MIN_COUNTS = 10


def group_of(col):
    g = re.sub(r"_2hLPS_R\d+$", "", col)
    return REF if g in CONTROL else g


def main(n_cpus):
    d = pd.read_csv(SRC)
    d["Symbol"] = d.Symbol.astype(str).str.strip()
    counts = d.groupby("Symbol").sum()
    X = counts.T
    X = X.loc[:, X.sum() >= MIN_COUNTS]
    meta = pd.DataFrame({"cond": [group_of(c) for c in X.index]}, index=X.index)
    drugs = sorted(set(meta.cond) - {REF})
    print(f"{X.shape[0]} samples, {X.shape[1]} genes, {len(drugs)} drugs; fitting one model ...", flush=True)

    dds = DeseqDataSet(counts=X, metadata=meta, design="~cond", ref_level=["cond", REF],
                       n_cpus=n_cpus, quiet=True)
    dds.deseq2()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, drug in enumerate(drugs, 1):
        st = DeseqStats(dds, contrast=["cond", drug, REF], n_cpus=n_cpus, quiet=True)
        st.summary()
        r = st.results_df.rename_axis("SYMBOL").reset_index()
        r = r[["SYMBOL", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj", "baseMean"]]
        r.to_csv(OUT / f"{drug}_DEG.csv", index=False)
        dusp6 = r.loc[r.SYMBOL == "Dusp6", "log2FoldChange"]
        rows.append(dict(drug=drug, n_drug=int((meta.cond == drug).sum()), n_ctrl=int((meta.cond == REF).sum()),
                         n_genes=len(r), n_padj05=int((r.padj < 0.05).sum()),
                         dusp6_lfc=float(dusp6.iloc[0]) if len(dusp6) else np.nan))
        print(f"[{i}/{len(drugs)}] {drug}: {rows[-1]['n_padj05']} genes padj<0.05, "
              f"Dusp6 {rows[-1]['dusp6_lfc']:+.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(f"wrote {len(drugs)} DEG tables to {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
