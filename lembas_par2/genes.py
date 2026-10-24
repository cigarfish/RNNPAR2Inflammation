"""Step 1: gene universe and DEG symbol harmonization.

Universe = genes in the STRING backbone (== Reactome pathway genes == feature matrix),
after merging human-style duplicates (Tp53 -> Trp53, ...) into their mouse symbol.
DEG tables are standardized to columns [gene, log2fc, pvalue, padj], old symbols are
renamed to current ones, and rows are restricted to the universe.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DEG_SUFFIX = "_DEG.csv"


# ---------- universe ----------
def string_genes(cfg):
    s = pd.read_csv(cfg.paths.string_ppi, sep="\t", usecols=["gene1", "gene2"])
    return set(s.gene1) | set(s.gene2)


def reactome_genes(cfg, take_only=False):
    r = pd.read_csv(cfg.paths.reactome_pathway_gene)
    if take_only:
        r = r[r.TAKE == 1]
    return {g.strip() for x in r.Signaling_Genes.dropna() for g in x.split(",") if g.strip()}


def feature_genes(cfg):
    return set(pd.read_csv(cfg.paths.gene_features, usecols=["gene"]).gene)


def symbol_map(cfg):
    """Combined old-symbol and human-style -> current mouse symbol map."""
    return {**cfg.symbol_aliases, **cfg.universe_merge}


def canonical(symbols, cfg):
    m = symbol_map(cfg)
    return {m.get(g, g) for g in symbols}


def load_universe(cfg):
    """Sorted gene list; the STRING backbone defines it, duplicates merged."""
    return sorted(canonical(string_genes(cfg), cfg))


# ---------- DEG ----------
def geo_ids(cfg):
    return list(pd.read_csv(cfg.paths.geo_metadata).geo_id)


def read_deg_raw(cfg, geo_id):
    df = pd.read_csv(Path(cfg.paths.deg_dir) / f"{geo_id}{DEG_SUFFIX}")
    df.columns = df.columns.str.strip().str.strip('"')
    return df


def harmonize_symbols(symbols, aliases):
    """Map old symbols to current ones; unknown symbols pass through unchanged."""
    s = pd.Series(symbols, dtype=str).str.strip()
    return s.map(lambda x: aliases.get(x, x))


def load_deg(cfg, geo_id, universe=None):
    """Standardized DEG table for one context, one row per gene.

    If an old and a current symbol collapse onto the same gene, the row with the
    smallest padj is kept (they never co-occur in the current data).
    """
    raw = read_deg_raw(cfg, geo_id)
    df = pd.DataFrame({
        "raw_symbol": raw["SYMBOL"].astype(str).str.strip(),
        "log2fc": pd.to_numeric(raw["log2FoldChange"], errors="coerce"),
        "pvalue": pd.to_numeric(raw["pvalue"], errors="coerce"),
        "padj": pd.to_numeric(raw["padj"], errors="coerce"),
    })
    df["gene"] = harmonize_symbols(df.raw_symbol, symbol_map(cfg)).values
    df = df[np.isfinite(df.log2fc)]
    if universe is not None:
        df = df[df.gene.isin(set(universe))]
    # duplicate symbols: keep the most significant row; ties (identical padj, seen in
    # GSE248539o / GSE276111) broken by pvalue, then |log2FC|, with a stable sort so that a
    # rebuild picks the same row every time (found 2026-10-17)
    df["_abs"] = -df.log2fc.abs()
    df = (df.sort_values(["padj", "pvalue", "_abs"], na_position="last", kind="stable")
            .drop(columns="_abs")
            .drop_duplicates("gene", keep="first")
            .sort_values("gene")
            .reset_index(drop=True))
    return df[["gene", "raw_symbol", "log2fc", "pvalue", "padj"]]


def deg_matrices(cfg, universe=None):
    """log2fc and padj matrices (genes x contexts); NaN = not measured."""
    universe = universe or load_universe(cfg)
    lfc, padj = {}, {}
    for gid in geo_ids(cfg):
        d = load_deg(cfg, gid, universe).set_index("gene")
        lfc[gid], padj[gid] = d.log2fc, d.padj
    lfc = pd.DataFrame(lfc).reindex(universe)
    padj = pd.DataFrame(padj).reindex(universe)
    lfc.index.name = padj.index.name = "gene"
    return lfc, padj
