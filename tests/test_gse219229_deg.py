"""Validate data_external/GSE219229_DEG against the four published contrasts.

Run from the LEMBAS folder after scripts/make_gse219229_deg.py:
    python tests/test_gse219229_deg.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEG = ROOT / "data_external" / "GSE219229_DEG"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


counts = pd.read_csv(ROOT / "curation" / "GSE219229.csv", nrows=1)
drugs = sorted({re.sub(r"_2hLPS_R\d+$", "", c) for c in counts.columns[1:]} - {"DMSOs1", "DMSOs2"})
missing = [d for d in drugs if not (DEG / f"{d}_DEG.csv").exists()]
check(f"DEG table for every drug ({len(drugs)})", not missing, missing)

targets = set(pd.read_csv(ROOT / "curation" / "drug_targets.csv").drug)
check("every drug has a row in curation/drug_targets.csv", set(drugs) <= targets, sorted(set(drugs) - targets))

print("\n== reproduce published contrasts")
for drug, pub_name in [("Pimasertib", "Pimasertib"), ("Selumetinib", "Selumetinib"),
                       ("GDC0994", "GDC-0994"), ("Ulixertinib", "Ulixertinib")]:
    ours = pd.read_csv(DEG / f"{drug}_DEG.csv").set_index("SYMBOL")
    pub = pd.read_csv(ROOT / "data_external" / f"GSE219229_{pub_name}.tsv", sep="\t")
    pub = pub.drop_duplicates("gene_symbol").set_index("gene_symbol")
    j = ours.join(pub[["effect", "padj"]], rsuffix="_pub", how="inner").dropna(subset=["log2FoldChange", "effect"])
    r = np.corrcoef(j.log2FoldChange, j.effect)[0, 1]
    n_o, n_p = int((ours.padj < 0.05).sum()), int((pub.padj < 0.05).sum())
    agree = ((j.padj < 0.05) == (j.padj_pub < 0.05)).mean()
    check(f"{drug}: log2FC matches published (r >= 0.99)", r >= 0.99,
          f"r={r:.3f}, n genes {len(j)}, padj<0.05 ours {n_o} vs published {n_p}, call agreement {agree:.3f}")

print("\n== on-target sanity (Dusp6, ERK pathway target)")
summ = pd.read_csv(DEG / "summary.csv").set_index("drug")
for d in ["Pimasertib", "Selumetinib", "Refametinib", "GDC0994", "Ulixertinib", "Encorafenib"]:
    print(f"[INFO] {d:12s} Dusp6 {summ.loc[d, 'dusp6_lfc']:+.2f}, genes padj<0.05: {summ.loc[d, 'n_padj05']}")
print("\n[INFO] drugs by number of genes padj<0.05:")
print(summ.n_padj05.sort_values(ascending=False).to_string())

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
sys.exit(1 if n_fail else 0)
