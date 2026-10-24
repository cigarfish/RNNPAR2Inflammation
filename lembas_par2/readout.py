"""Step 5: split the network into a signalling layer and a transcriptional readout layer.

Signalling layer  all non-transcriptional edges (T0-T3); node value = activity.
Readout layer     TF -> target mRNA from CollecTRI + TRRUST regulons:
                  - TF-activity outputs: TFs with enough signed, measured targets
                  - gene readout: measured genes regulated by >= 1 network TF
Regulon TFs outside the gene universe (Stat1, Elk1, ...) join the network when a
curated signalling edge from a universe gene reaches them.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .edges import curated_edges
from .genes import deg_matrices, load_universe, symbol_map
from .network import build_network


# ---------- regulons ----------
def load_collectri(cfg):
    p = Path(cfg.paths.collectri)
    if not p.exists():
        return pd.DataFrame(columns=["tf", "target", "sign", "db"])
    c = pd.read_csv(p, dtype={"source": str, "target": str})
    return pd.DataFrame({"tf": c.source, "target": c.target,
                         "sign": np.sign(c.weight.astype(float)).astype(int), "db": "collectri"})


def load_trrust_regulon(cfg):
    t = pd.read_csv(cfg.paths.trrust, sep="\t", header=None, names=["tf", "target", "mode", "pmid"], dtype=str)
    sign = t["mode"].map({"Activation": 1, "Repression": -1}).fillna(0).astype(int)
    return pd.DataFrame({"tf": t.tf, "target": t.target, "sign": sign, "db": "trrust"})


def load_regulons(cfg):
    """One row per (tf, target): consensus sign over both sources (conflict -> 0)."""
    r = pd.concat([load_collectri(cfg), load_trrust_regulon(cfg)], ignore_index=True)
    m = symbol_map(cfg)
    r["tf"] = r.tf.map(lambda x: m.get(x, x))
    r["target"] = r.target.map(lambda x: m.get(x, x))
    r = r[r.tf != r.target]

    def cons(s):
        nz = set(s[s != 0])
        return nz.pop() if len(nz) == 1 else 0

    g = r.groupby(["tf", "target"], sort=False)
    out = g.agg(sign=("sign", cons), dbs=("db", lambda s: ";".join(sorted(set(s))))).reset_index()
    out["conflict"] = g.sign.apply(lambda s: s.gt(0).any() and s.lt(0).any()).values
    return apply_readout_exclusions(cfg, apply_readout_sign_decisions(cfg, out))


def apply_readout_sign_decisions(cfg, reg):
    """Explicit, evidence-recorded sign decisions for TF -> gene pairs whose sources conflict
    (curation/readout_sign_decisions.csv; added 2026-10-17 after the step-13 Ifnb1 trace).
    The sign is set, `conflict` is kept as the record of the disagreement, `decision` marks it."""
    p = cfg.paths.get("readout_sign_decisions") if hasattr(cfg.paths, "get") else getattr(cfg.paths, "readout_sign_decisions", None)
    reg = reg.copy()
    reg["decision"] = ""
    if not p or not Path(p).exists():
        return reg
    d = pd.read_csv(p)
    key = {(a, b): int(s) for a, b, s in zip(d.tf, d.target, d.sign)}
    hit = [(a, b) in key for a, b in zip(reg.tf, reg.target)]
    reg.loc[hit, "sign"] = [key[(a, b)] for a, b, h in zip(reg.tf, reg.target, hit) if h]
    reg.loc[hit, "decision"] = "readout_sign_decision"
    missing = set(key) - set(zip(reg.tf, reg.target))
    if missing:
        raise ValueError(f"readout sign decisions for pairs not in any regulon: {sorted(missing)}")
    return reg


# ---------- extension TFs ----------
def apply_readout_exclusions(cfg, reg):
    """Drop TF -> gene pairs that encode FEEDBACK rather than direct transcriptional regulation
    (curation/readout_exclusions.csv; added 2026-10-18).

    A regulon pair says the TF's activity tracks the target, which for a DOWNSTREAM effector of
    that target is a consequence, not a cause: IFN-beta -> IFNAR -> JAK -> STAT1/STAT2/IRF9, so
    STAT1 activity at 2 h is largely produced BY Ifnb1. Leaving such a pair in the readout lets
    the model explain a gene by its own output. Removing the pair also drops that target from the
    TF's regulon, which is consistent: if the TF does not regulate the gene, the gene should not
    inform the TF's activity either."""
    p = getattr(cfg.paths, 'readout_exclusions', None)
    if not p or not Path(p).exists():
        return reg
    d = pd.read_csv(p)
    drop = set(zip(d.tf, d.target))
    missing = drop - set(zip(reg.tf, reg.target))
    if missing:
        raise ValueError(f'readout exclusions for pairs not in any regulon: {sorted(missing)}')
    keep = [(a, b) not in drop for a, b in zip(reg.tf, reg.target)]
    return reg[keep].reset_index(drop=True)


def extension_tfs(cfg, reg, universe):
    """Regulon TFs outside the universe reached by a curated edge from a universe gene."""
    U = set(universe)
    cand = sorted(set(reg.tf) - U)
    agg = curated_edges(cfg, sorted(U | set(cand)))["agg"]
    hit = agg[agg.target.isin(cand) & agg.source.isin(U)]
    return sorted(set(hit.target)), hit


# ---------- layers ----------
def _is_transcription(mech, tx):
    toks = {t for t in str(mech).split(";") if t}
    return bool(toks) and toks <= tx


def build_layers(cfg):
    rc = cfg.readout
    U = load_universe(cfg)
    reg = load_regulons(cfg)
    ext, ext_inputs = extension_tfs(cfg, reg, U)
    net_genes = sorted(set(U) | set(ext))
    N = build_network(cfg, net_genes)
    E = N["edges"]

    tx = set(rc["transcription_mechanisms"])
    is_tx = E.mechanism.map(lambda m: _is_transcription(m, tx)) & (E.tier != "T0")
    signalling = E[~is_tx].reset_index(drop=True)
    moved = E[is_tx].reset_index(drop=True)

    nodes = set(signalling.source) | set(signalling.target)
    reg_net = reg[reg.tf.isin(nodes)].copy()

    # putative (data-driven) regulators: unsigned, only where no regulon pair exists
    have = set(zip(reg_net.tf, reg_net.target))
    put = pd.DataFrame([dict(tf=tf, target=g, sign=0, dbs="putative_isg", conflict=False)
                        for g, tfs in rc.get("putative_regulators", {}).items()
                        for tf in tfs if tf in nodes and (tf, g) not in have])
    if len(put):
        reg_net = pd.concat([reg_net, put], ignore_index=True)

    # measured log2FC for every regulon target and network gene (incl. outside universe)
    genes = sorted(set(reg_net.target) | set(net_genes))
    lfc, padj = deg_matrices(cfg, genes)
    n_ctx = lfc.notna().sum(axis=1)

    # TF-activity outputs: enough signed, measured targets
    measured = set(n_ctx[n_ctx >= rc["min_gene_contexts"]].index)
    signed = reg_net[(reg_net.sign != 0) & reg_net.target.isin(measured)]
    n_t = signed.groupby("tf").size()
    tf_outputs = sorted(n_t[n_t >= rc["min_tf_targets"]].index)

    # gene readout: measured genes with >= 1 network TF regulator
    readout = reg_net[reg_net.target.isin(measured)].rename(columns={"target": "gene"})
    readout_genes = sorted(set(readout.gene))

    return dict(regulons=reg, ext_tfs=ext, ext_inputs=ext_inputs, network=N, signalling=signalling,
                moved_transcription=moved, tf_outputs=tf_outputs, tf_n_targets=n_t,
                readout_edges=readout.reset_index(drop=True), readout_genes=readout_genes,
                lfc=lfc, padj=padj, n_contexts=n_ctx)
