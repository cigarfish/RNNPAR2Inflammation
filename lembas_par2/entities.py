"""Step 3: non-gene signal entities (mtDNA, Ca2+, cAMP, PIP3, ...).

Nodes come from the gene-package table, edges from the seed edge table. Edge signs
follow biochemical_role (config entity_role_signs) with explicit overrides
(config entity_sign_overrides). The same entity->entity link listed both as
'upstream_signal' and 'downstream_signal' is merged into one edge.
Output edges use the step-2 per-db column layout with db = 'signal_entity'.
"""
import pandas as pd

from .genes import load_universe, symbol_map

DB = "signal_entity"


def load_entity_nodes(cfg):
    p = pd.read_csv(cfg.paths.signal_entity_packages, encoding="utf-8-sig", dtype=str)
    nodes = pd.DataFrame({
        "node": p.signal_symbol, "display_name": p.display_name, "kind": "signal_entity",
        "activation_mode": p.activation_mode, "status": p.status,
        "input_only": p.activation_mode.eq("input_only"), "directness": p.directness,
    })
    return nodes.reset_index(drop=True)


def load_entity_edges(cfg, universe=None):
    universe = set(universe or load_universe(cfg))
    e = pd.read_csv(cfg.paths.signal_entity_edges, encoding="utf-8-sig", dtype=str)
    smap = symbol_map(cfg)
    fix = lambda typ, x: smap.get(x, x) if typ == "gene" else x
    e["source"] = [fix(t, x) for t, x in zip(e.src_type, e.src_id)]
    e["target"] = [fix(t, x) for t, x in zip(e.dst_type, e.dst_id)]

    roles = cfg.entity_role_signs
    unknown = sorted(set(e.biochemical_role) - set(roles))
    if unknown:
        raise ValueError(f"unmapped biochemical_role values: {unknown}")
    e["sign"] = e.biochemical_role.map(roles).astype(int)
    over = {(s, t): sg for s, t, sg in cfg.get("entity_sign_overrides", [])}
    e["sign"] = [over.get((s, t), sg) for s, t, sg in zip(e.source, e.target, e.sign)]
    e["overridden"] = [(s, t) in over for s, t in zip(e.source, e.target)]

    genes = pd.concat([e[e.src_type == "gene"].source, e[e.dst_type == "gene"].target])
    missing = sorted(set(genes) - universe)
    if missing:
        raise ValueError(f"signal-entity genes outside universe: {missing}")

    g = e.groupby(["source", "target"], sort=False)
    agg = g.agg(
        sign=("sign", lambda s: int(s.iloc[0]) if s.nunique() == 1 else 0),
        n_records=("sign", "size"),
        mechanism=("biochemical_role", lambda s: ";".join(sorted(set(s)))),
        directness=("directness", lambda s: ";".join(sorted(set(s)))),
        overridden=("overridden", "any"),
        detail=("notes", lambda s: " | ".join(x for x in s.dropna())),
        src_type=("src_type", "first"), dst_type=("dst_type", "first"),
    ).reset_index()
    agg["db"] = DB
    agg["directed"] = True
    agg["n_pos"] = (agg.sign > 0).astype(int) * agg.n_records
    agg["n_neg"] = (agg.sign < 0).astype(int) * agg.n_records
    agg["phospho_kind"] = ""
    agg["mixed"] = False
    return agg
