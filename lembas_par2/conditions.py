"""Step 6: conditions (experiments) and training / validation targets.

A condition is a contrast "treated vs reference". The model will simulate both arms:
    reference arm: ref_inputs active
    treated arm  : trt_inputs active, trt_inhibit nodes inhibited
and compare treated - reference with the measured contrast.

Sources
  drug     GSE219229 drug + LPS vs DMSO + LPS (56 drugs) and GSE134486 trametinib
  geo      48 lung inflammation sets, stimulus vs control
  perturb  literature KO / knockdown / antagonist contrasts (direction-only)
  holdout  PAR2_26 / 29 / 30 (validation only, never trained)

Targets
  tf_activity  ULM t-value of each output TF's signed regulon on z-scored log2FC
  gene_lfc     log2FC of readout genes (clipped), with padj
  programs     MPAS (ERK activity) and ISG mean z-scores
  directions   perturbation observations as sign constraints on nodes / readout genes
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .genes import geo_ids, load_deg, symbol_map

FOLD_TRAIN_ONLY = -1


# ---------- measured contrasts ----------
def _harmonize(df, cfg, sym_col, lfc_col, padj_col):
    m = symbol_map(cfg)
    out = pd.DataFrame({
        "gene": df[sym_col].astype(str).str.strip().map(lambda x: m.get(x, x)),
        "log2fc": pd.to_numeric(df[lfc_col], errors="coerce"),
        "padj": pd.to_numeric(df[padj_col], errors="coerce"),
    })
    out = out[np.isfinite(out.log2fc)]
    return out.sort_values("padj", na_position="last", kind="stable").drop_duplicates("gene").set_index("gene")


def drug_contrasts(cfg):
    """{condition_id: DataFrame[log2fc, padj]} and per-drug metadata."""
    c = cfg.conditions
    tgt = pd.read_csv(cfg.root / c["drug_targets"])
    tgt = tgt[~tgt.drug.isin(c["drug_skip"])]
    deg_dir = cfg.root / c["drug_deg_dir"]
    data, meta = {}, []
    for r in tgt.itertuples():
        df = pd.read_csv(deg_dir / f"{r.drug}_DEG.csv")
        cid = f"GSE219229_{r.drug}"
        data[cid] = _harmonize(df, cfg, "SYMBOL", "log2FoldChange", "padj")
        meta.append(dict(condition_id=cid, dataset="GSE219229", name=r.drug,
                         targets=r.targets.split(";"), cls=r.target_class, relevance=r.relevance))
    for cid, spec in c.get("extra_drug_contrasts", {}).items():
        df = pd.read_csv(cfg.root / spec["file"], sep="\t")
        df = df[df.effect_available.astype(str).str.lower() == "true"]
        data[cid] = _harmonize(df, cfg, "gene_symbol", "effect", "padj")
        meta.append(dict(condition_id=cid, dataset=cid.split("_")[0], name=cid.split("_", 1)[1],
                         targets=list(spec["targets"]), cls="MEK", relevance="high", weight=spec["weight"]))
    return data, meta


def geo_contrasts(cfg):
    data = {}
    for gid in geo_ids(cfg):
        d = load_deg(cfg, gid).set_index("gene")[["log2fc", "padj"]]
        data[gid] = d
    return data


# ---------- condition table ----------
def _drug_weight(n_sig, w):
    if n_sig < w["null_below"]:
        return w["null_weight"]
    return float(min(w["max"], w["base"] + w["per_gene"] * n_sig))


def _bg_inputs(text, rules):
    t = str(text).lower()
    for key, nodes in rules.items():
        if key.lower() in t:
            return list(nodes), key
    return [], ""


def geo_folds(meta, k, seed):
    """Grouped (by GSE series), class-stratified, LPS-spread fold assignment."""
    rng = np.random.default_rng(seed)
    meta = meta.copy()
    meta["series"] = meta.geo_id.str.extract(r"^(GSE\d+)")[0]
    groups = meta.groupby("series").agg(n=("geo_id", "size"), cls=("stimulus", "first"),
                                        lps=("stimulus_subtype", lambda s: int((s == "LPS").sum())))
    groups = groups.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    groups = groups.sort_values(["lps", "n"], ascending=False, kind="stable")
    load = {f: {"total": 0, "lps": 0} for f in range(k)}
    cls_load = {}
    fold_of = {}
    for s, g in groups.iterrows():
        cl = cls_load.setdefault(g.cls, {f: 0 for f in range(k)})
        f = min(range(k), key=lambda f: (load[f]["lps"] if g.lps else 0, cl[f], load[f]["total"], f))
        fold_of[s] = f
        cl[f] += g.n
        load[f]["total"] += g.n
        load[f]["lps"] += g.lps
    return meta.series.map(fold_of)


def build_condition_table(cfg, drug_meta, drug_data, net_nodes):
    c = cfg.conditions
    rows = []
    # drugs: folds round-robin over drugs sorted by target class (classes spread)
    k = c["cv_folds"]
    for m in sorted(drug_meta, key=lambda m: (m["cls"], m["name"])):
        n_sig = int((drug_data[m["condition_id"]].padj < 0.05).sum())
        w = m.get("weight", _drug_weight(n_sig, c["drug_weight"]))
        tg = [t for t in m["targets"] if t in net_nodes]
        rows.append(dict(condition_id=m["condition_id"], source="drug", dataset=m["dataset"], name=m["name"],
                         cls=m["cls"], ref_inputs=c["lps_inputs"], trt_inputs=c["lps_inputs"], trt_inhibit=tg,
                         weight=w, n_sig=n_sig, fold=None, notes=f"relevance={m['relevance']}"))
    for i, r in enumerate(rows):
        r["fold"] = i % k

    # GEO
    meta = pd.read_csv(cfg.paths.geo_metadata)
    folds = geo_folds(meta, k, c["seed"])
    for r, f in zip(meta.itertuples(), folds):
        sub = r.stimulus_subtype
        w = c["geo_weight"] * (c["geo_single_cell_factor"] if r.geo_id in c["geo_single_cell"] else 1.0)
        rows.append(dict(condition_id=r.geo_id, source="geo", dataset=r.geo_id, name=sub, cls=r.stimulus,
                         ref_inputs=[], trt_inputs=list(c["stimulus_inputs"].get(sub, [])),
                         trt_inhibit=list(c["stimulus_inhibit"].get(sub, [])),
                         weight=w, n_sig=None, fold=int(f), notes=""))

    # literature perturbations (train) and holdouts
    P = cfg.paths
    st = pd.read_csv(P.perturbation_studies, encoding="utf-8-sig")
    co = pd.read_csv(P.perturbation_contrasts, encoding="utf-8-sig")
    iv = pd.read_csv(P.perturbation_interventions, encoding="utf-8-sig")
    hold = set(cfg.holdout_studies)
    co = co.merge(st[["study_id", "is_holdout"]], on="study_id")
    for r in co.itertuples():
        is_hold = r.study_id in hold
        if not is_hold and int(r.use_for_training) != 1:
            continue
        ivs = iv[iv.contrast_id == r.contrast_id]
        bg, rule = _bg_inputs(r.background_stimulus, c["background_inputs"])
        trt_in, trt_inh = list(bg), []
        for v in ivs.itertuples():
            targets = [t for t in str(v.graph_target_id).split(";") if t in net_nodes]
            if v.perturbation_direction == "gain_of_function":
                trt_in += targets
            else:
                trt_inh += targets
        w = 0.0 if is_hold else c["perturb_weight"] * c["perturb_species_factor"].get(r.species, 0.5)
        rows.append(dict(condition_id=r.contrast_id, source="holdout" if is_hold else "perturb",
                         dataset=r.study_id, name=str(r.treatment_arm), cls=str(r.stimulus_class),
                         ref_inputs=bg, trt_inputs=sorted(set(trt_in)), trt_inhibit=sorted(set(trt_inh)),
                         weight=w, n_sig=None, fold=FOLD_TRAIN_ONLY, notes=f"species={r.species}; background='{r.background_stimulus}' -> {rule or 'none'}"))
    t = pd.DataFrame(rows)
    t.loc[t.source == "holdout", "fold"] = None
    return t


# ---------- targets ----------
def zscore(s):
    med = s.median()
    mad = 1.4826 * (s - med).abs().median()
    return (s - med) / (mad if mad > 0 else s.std())


def tf_activity(contrasts, regulon, tfs, min_targets):
    """ULM: per condition, regress z-scored log2FC on each TF's signed regulon; t-value."""
    reg = regulon[(regulon.sign != 0) & regulon.tf.isin(tfs)]
    tf_index = {t: i for i, t in enumerate(tfs)}
    out = {}
    for cid, d in contrasts.items():
        z = zscore(d.log2fc)
        genes = z.index
        gi = {g: i for i, g in enumerate(genes)}
        r = reg[reg.target.isin(gi)]
        X = sp.csr_matrix((r.sign.astype(float).values, ([gi[g] for g in r.target], [tf_index[t] for t in r.tf])),
                          shape=(len(genes), len(tfs)))
        n = len(genes)
        zv = z.values
        sx = np.asarray(X.sum(axis=0)).ravel()
        sxx = np.asarray(X.multiply(X).sum(axis=0)).ravel() - sx ** 2 / n
        sxy = X.T @ zv - sx * zv.mean()
        syy = ((zv - zv.mean()) ** 2).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            b = sxy / sxx
            ssr = np.maximum(syy - b * sxy, 1e-12)
            t = b / np.sqrt(ssr / (n - 2) / sxx)
        n_t = np.asarray((X != 0).sum(axis=0)).ravel()
        t[n_t < min_targets] = np.nan
        out[cid] = t
    return pd.DataFrame(out, index=tfs)


def program_scores(contrasts, programs, min_genes=3):
    rows = {}
    for cid, d in contrasts.items():
        z = zscore(d.log2fc)
        rows[cid] = {p: (z.reindex(g).dropna().mean() if z.reindex(g).notna().sum() >= min_genes else np.nan)
                     for p, g in programs.items()}
    return pd.DataFrame(rows).T


def gene_targets(contrasts, genes, clip):
    lfc = pd.DataFrame({cid: d.log2fc.reindex(genes) for cid, d in contrasts.items()}).clip(-clip, clip)
    padj = pd.DataFrame({cid: d.padj.reindex(genes) for cid, d in contrasts.items()})
    return lfc, padj


def direction_targets(cfg, cond, signal_nodes, readout_genes):
    """Observation rows -> (condition, node, layer, sign, weight) with drop reasons."""
    c = cfg.conditions
    ob = pd.read_csv(cfg.paths.perturbation_observations, encoding="utf-8-sig")
    cw = dict(zip(cond.condition_id, cond.weight))
    src = dict(zip(cond.condition_id, cond.source))
    sign_of = {"up": 1, "down": -1, "unchanged": 0}
    kept, dropped = [], []
    for r in ob.itertuples():
        base = dict(observation_id=r.observation_id, condition_id=r.contrast_id, target_raw=r.target_raw,
                    readout_type=r.readout_type, effect_direction=r.effect_direction)
        if r.contrast_id not in cw:
            dropped.append(dict(base, reason="contrast_not_used"))
            continue
        layer = c["readout_layers"].get(r.readout_type)
        if layer is None:
            dropped.append(dict(base, reason=f"readout_type:{r.readout_type}"))
            continue
        if r.effect_direction not in sign_of:
            dropped.append(dict(base, reason=f"direction:{r.effect_direction}"))
            continue
        if pd.isna(r.graph_target_id):
            dropped.append(dict(base, reason="unmapped"))
            continue
        for node in str(r.graph_target_id).split(";"):
            pool = signal_nodes if layer == "signal" else readout_genes
            if node not in pool:
                dropped.append(dict(base, node=node, reason=f"not_in_{layer}_layer"))
                continue
            w = cw[r.contrast_id] * (c["secretion_factor"] if "secretion" in str(r.readout_type) else 1.0)
            kept.append(dict(base, node=node, layer=layer, sign=sign_of[r.effect_direction],
                             significant=int(r.significant), weight=w, source=src[r.contrast_id]))
    return pd.DataFrame(kept), pd.DataFrame(dropped)


def marker_weights(cfg, cond, tfa):
    """Re-weight drug conditions by on-target marker-TF evidence.

    Adds columns w_genes, marker_set, marker_score, markers_used, w_marker; drug
    weight becomes max(w_genes, w_marker) unless the drug has an explicit weight.
    """
    c = cfg.conditions
    mw = c["drug_marker_weight"]
    cls_set = c["marker_set_of_class"]
    explicit = set(c.get("extra_drug_contrasts", {}))
    cond = cond.copy()
    for col in ["marker_set", "markers_used"]:
        cond[col] = ""
    cond["w_genes"] = cond.weight
    cond["marker_score"] = np.nan
    cond["w_marker"] = 0.0
    for i, r in cond[cond.source == "drug"].iterrows():
        mset = cls_set.get(r.cls)
        if not mset:
            continue
        vals, used = [], []
        for tf, sg in c["drug_markers"][mset]:
            if tf in tfa.index and np.isfinite(tfa.at[tf, r.condition_id]):
                v = sg * tfa.at[tf, r.condition_id]
                vals.append(v)
                used.append(f"{tf}:{v:+.2f}")
        if not vals:
            continue
        s = float(np.mean(vals))
        w = mw["max"] * float(np.clip((s - mw["lo"]) / (mw["hi"] - mw["lo"]), 0, 1))
        cond.loc[i, ["marker_set", "markers_used"]] = [mset, ";".join(used)]
        cond.loc[i, ["marker_score", "w_marker"]] = [s, w]
        if r.condition_id not in explicit:
            cond.loc[i, "weight"] = max(r.weight, w)
    return cond


# ---------- main ----------
def build_conditions(cfg, L):
    """L = readout.build_layers(cfg) output."""
    c = cfg.conditions
    S = L["signalling"]
    signal_nodes = set(S.source) | set(S.target)
    drug_data, drug_meta = drug_contrasts(cfg)
    geo_data = geo_contrasts(cfg)
    contrasts = {**drug_data, **geo_data}
    cond = build_condition_table(cfg, drug_meta, drug_data, signal_nodes)

    reg = L["regulons"]
    tfs = list(L["tf_outputs"])
    tfa = tf_activity(contrasts, reg, tfs, c["min_tf_targets_condition"])
    cond = marker_weights(cfg, cond, tfa)
    prog = program_scores(contrasts, c["programs"])
    lfc, padj = gene_targets(contrasts, L["readout_genes"], c["lfc_clip"])
    dirs, dropped = direction_targets(cfg, cond, signal_nodes, set(L["readout_genes"]))
    return dict(conditions=cond, contrasts=contrasts, tf_activity=tfa, programs=prog,
                gene_lfc=lfc, gene_padj=padj, directions=dirs, dropped_observations=dropped)


def to_jsonable(cond):
    t = cond.copy()
    for col in ["ref_inputs", "trt_inputs", "trt_inhibit"]:
        t[col] = t[col].map(json.dumps)
    return t
