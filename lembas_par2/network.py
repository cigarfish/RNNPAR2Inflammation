"""Step 4: merge curated + entity edges, assign evidence tiers, add STRING backbone,
and restrict convergence-node inputs.

Tiers
  T0  prior route edges (config route_edges); curated sign must agree
  T1  curated, consensus sign across databases (or strict majority)
  T2  curated but unsigned / tied conflict -> sign learned
  T3  STRING-only pair (score >= string_min_score), both directions, sign learned
Convergence nodes (config) keep only T0/T1 inputs.
"""
import numpy as np
import pandas as pd

from .edges import curated_edges, load_decisions
from .entities import load_entity_edges, load_entity_nodes
from .genes import load_universe, symbol_map

TIERS = ["T0", "T1", "T2", "T3"]


def _fmt(db, s):
    return f"{db}:{int(s):+d}"


def merge_curated(per_db, cfg):
    """per_db: step-2 agg + step-3 entity edges (one row per source,target,db).
    Returns one row per directed pair with consensus sign and provenance."""
    rows = []
    for (s, t), g in per_db.groupby(["source", "target"], sort=False):
        nz = g[g.sign != 0]
        n_pos, n_neg = int((nz.sign > 0).sum()), int((nz.sign < 0).sum())
        if n_pos and not n_neg:
            sign, rule = 1, "consensus"
        elif n_neg and not n_pos:
            sign, rule = -1, "consensus"
        elif not n_pos and not n_neg:
            sign, rule = 0, "unsigned"
        elif cfg.get("conflict_rule") == "majority" and n_pos != n_neg:
            sign, rule = (1 if n_pos > n_neg else -1), "majority"
        else:
            sign, rule = 0, "tied_conflict"
        rows.append(dict(source=s, target=t, sign=sign, sign_rule=rule,
                         dbs=";".join(g.db), db_signs=";".join(_fmt(d, x) for d, x in zip(g.db, g.sign)),
                         n_dbs=len(g), mechanism=";".join(sorted({m for x in g.mechanism.astype(str) for m in x.split(";")})),
                         detail=" || ".join(f"{d}: {x}" for d, x in zip(g.db, g.detail.astype(str)))))
    return pd.DataFrame(rows)


def load_string(cfg):
    s = pd.read_csv(cfg.paths.string_ppi, sep="\t")
    m = symbol_map(cfg)
    s["gene1"] = s.gene1.map(lambda x: m.get(x, x))
    s["gene2"] = s.gene2.map(lambda x: m.get(x, x))
    s = s[s.gene1 != s.gene2]
    s["a"] = s[["gene1", "gene2"]].min(axis=1)
    s["b"] = s[["gene1", "gene2"]].max(axis=1)
    return s.groupby(["a", "b"], as_index=False).combined_score.max()


def build_network(cfg, universe=None, cur=None):
    """Returns dict(edges, nodes, removed_convergence, route_check, string_stats)."""
    universe = universe or load_universe(cfg)
    if cur is None:
        cur = curated_edges(cfg, universe)["agg"]
    ent = load_entity_edges(cfg, universe)
    cols = ["source", "target", "db", "sign", "mechanism", "detail"]
    per_db = pd.concat([cur[cols], ent[cols]], ignore_index=True)
    m = merge_curated(per_db, cfg)

    # tiers for curated pairs
    m["tier"] = np.where(m.sign != 0, "T1", "T2")
    m["sign_source"] = np.where(m.sign != 0, "data", "learned")

    # the prior route. `route_tier` is normally T0; set it to T1 to run with NO manual
    # prior at all - every route edge stays in the network, with its curated sign, but
    # competes on the same footing as the rest of the curated layer (T1 init, and
    # subject to the stability projection that T0 is exempt from).
    rt = cfg.get("route_tier", "T0") if hasattr(cfg, "get") else getattr(cfg, "route_tier", "T0")
    route_check = []
    key = {(s, t): i for i, (s, t) in enumerate(zip(m.source, m.target))}
    added = []
    for item in cfg.route_edges:
        s, t, sg = item[0], item[1], int(item[2])
        prior = len(item) > 3 and item[3] == "prior"
        why = item[4] if len(item) > 4 else ""
        i = key.get((s, t))
        if i is None and prior:
            status = "added_from_prior"
            added.append(dict(source=s, target=t, sign=sg, sign_rule="prior", dbs="prior",
                              db_signs=f"prior:{sg:+d}", n_dbs=0, mechanism="prior",
                              detail=why, tier=rt, sign_source="prior"))
        elif i is None:
            status = "missing_in_data"
        elif m.at[i, "sign"] == sg:
            status = "data_agrees"
        elif m.at[i, "sign"] == 0 and prior:
            status = "sign_from_prior"
            m.at[i, "sign"], m.at[i, "sign_source"] = sg, "prior"
        else:
            status = f"data_sign_{m.at[i, 'sign']:+d}"
        if i is not None and status in ("data_agrees", "sign_from_prior"):
            m.at[i, "tier"] = rt
        route_check.append(dict(source=s, target=t, route_sign=sg, prior=prior, status=status,
                                db_signs=m.at[i, "db_signs"] if i is not None else "", prior_source=why))
    route_check = pd.DataFrame(route_check)
    if added:
        m = pd.concat([m, pd.DataFrame(added)], ignore_index=True)

    # STRING annotation + T3 backbone
    st = load_string(cfg)
    st_score = {(a, b): sc for a, b, sc in zip(st.a, st.b, st.combined_score)}
    m["string_score"] = [st_score.get((min(s, t), max(s, t)), np.nan) for s, t in zip(m.source, m.target)]
    covered ={(min(s, t), max(s, t)) for s, t in zip(m.source, m.target)}
    # pairs whose only curated record the user dropped stay out of the STRING tier too
    dec = load_decisions(cfg)
    dec = dec[dec.action == "drop"]
    covered |= {(min(s, t), max(s, t)) for s, t in zip(dec.source, dec.target)}
    thr = cfg.get("string_min_score", 700)
    only = st[[(a, b) not in covered for a, b in zip(st.a, st.b)]]
    string_stats = {f">={x}": int((only.combined_score >= x).sum()) for x in (500, 700, 900)}
    only = only[only.combined_score >= thr]
    t3 = pd.concat([
        pd.DataFrame({"source": only.a, "target": only.b}),
        pd.DataFrame({"source": only.b, "target": only.a}),
    ], ignore_index=True)
    t3["string_score"] = np.concatenate([only.combined_score.values] * 2)
    t3 = t3.assign(sign=0, sign_rule="string_only", dbs="string", db_signs="", n_dbs=0,
                   mechanism="ppi", detail="", tier="T3", sign_source="learned")
    edges = pd.concat([m, t3[m.columns]], ignore_index=True)
    ctx = {(s, t) for s, t in cfg.get("context_dependent_edges", [])}
    edges["context_dependent"] = [(s, t) in ctx for s, t in zip(edges.source, edges.target)]

    # convergence restriction
    conv = set(cfg.convergence_nodes)
    bad = edges.target.isin(conv) & edges.tier.isin(["T2", "T3"])
    removed = edges[bad].reset_index(drop=True)
    edges = edges[~bad].reset_index(drop=True)

    # nodes
    ent_nodes = load_entity_nodes(cfg)
    genes = pd.DataFrame({"node": universe, "kind": "gene"})
    nodes = pd.concat([genes, ent_nodes[["node", "kind", "input_only"]]], ignore_index=True)
    nodes["input_only"] = nodes.input_only.eq(True)
    used = set(edges.source) | set(edges.target)
    nodes["in_network"] = nodes.node.isin(used)
    edges["tier"] = pd.Categorical(edges.tier, TIERS)
    return dict(edges=edges, nodes=nodes, removed_convergence=removed,
                route_check=route_check, string_stats=dict(only_pairs=string_stats, threshold=thr))
