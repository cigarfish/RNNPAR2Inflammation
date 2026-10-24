"""Step 2: curated edge loaders and sign-conflict reporting.

Every source is parsed into one record table (one row per raw evidence record):
    source, target, sign (+1/-1/0), mechanism, db, directed, phospho_kind, detail
then aggregated to one row per (source, target, db).

Rules
- Symbols go through the step-1 symbol map; both ends must be in the universe.
- PSP kinase/phosphatase -> substrate signs come from the site function
  (downstream_interactions on_function); phosphatases flip the site sign.
- cellsignal phospho-type rows (phosphorylate / inactivate / dephosphorylate) are
  kept only when PSP has no record of that pair. If PSP has it, PSP wins: the
  cellsignal row is 'superseded' (consistent) or 'contradicted' (kind or sign
  differs) and removed. A PSP pair without a sign takes the cellsignal sign.
- cellsignal_location ('carry' relocation edges) is loaded but not used as edges.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .genes import load_universe, symbol_map

COLS = ["source", "target", "sign", "mechanism", "db", "directed", "phospho_kind", "detail"]
DBS = ["omnipath", "signor", "trrust", "cellsignal", "psp", "manual"]


def _records(df, db):
    df = df.copy()
    df["db"] = db
    for c in COLS:
        if c not in df:
            df[c] = None
    return df[COLS]


# ---------- OmniPath ----------
def load_omnipath(cfg):
    o = pd.read_csv(cfg.paths.omnipath, sep="\t", dtype=str)
    stim = o.is_stimulation.eq("TRUE")
    inh = o.is_inhibition.eq("TRUE")
    sign = np.select([stim & ~inh, inh & ~stim], [1, -1], 0)
    mech = np.select([stim & inh, stim, inh], ["mixed", "stimulation", "inhibition"], "unsigned")
    df = pd.DataFrame({
        "source": o.source_mouse_genesymbol, "target": o.target_mouse_genesymbol,
        "sign": sign, "mechanism": mech, "directed": o.is_directed.eq("TRUE"),
        "detail": o.source_genesymbol + ">" + o.target_genesymbol,
    })
    return _records(df, "omnipath")


# ---------- SIGNOR ----------
def _signor_sign(effect):
    e = str(effect)
    if e.startswith("up-regulates"):
        return 1
    if e.startswith("down-regulates"):
        return -1
    return 0


def load_signor(cfg):
    files = sorted(glob.glob(str(Path(cfg.paths.signor_dir) / "SIGNOR-*.tsv")))
    s = pd.concat([pd.read_csv(f, sep="\t", dtype=str) for f in files], ignore_index=True)
    s = s.dropna(subset=["ENTITYA_MOUSE", "ENTITYB_MOUSE"]).drop_duplicates("SIGNOR_ID")
    fam = cfg.get("signor_families", {})
    expand = lambda x: fam.get(x, [x])
    rows = []
    for r in s.itertuples(index=False):
        sign = _signor_sign(r.EFFECT)
        complex_ = r.EFFECT == "form complex"
        mech = r.MECHANISM if isinstance(r.MECHANISM, str) else str(r.EFFECT)
        detail = f"{r.SIGNOR_ID};{r.EFFECT};{r.RESIDUE if isinstance(r.RESIDUE, str) else ''};pmid {r.PMID}"
        for a in expand(r.ENTITYA_MOUSE):
            for b in expand(r.ENTITYB_MOUSE):
                rows.append((a, b, sign, mech, not complex_, detail))
    df = pd.DataFrame(rows, columns=["source", "target", "sign", "mechanism", "directed", "detail"])
    return _records(df, "signor")


# ---------- TRRUST ----------
def load_trrust(cfg):
    t = pd.read_csv(cfg.paths.trrust, sep="\t", header=None, names=["tf", "target", "mode", "pmid"], dtype=str)
    sign = t["mode"].map({"Activation": 1, "Repression": -1}).fillna(0).astype(int)
    df = pd.DataFrame({"source": t.tf, "target": t.target, "sign": sign,
                       "mechanism": "transcription", "directed": True,
                       "detail": t["mode"] + ";pmid " + t.pmid.astype(str)})
    return _records(df, "trrust")


# ---------- cellsignal ----------
def load_cellsignal(cfg):
    files = sorted(glob.glob(str(Path(cfg.paths.cellsignal_dir) / "*.tsv")))
    c = pd.concat([pd.read_csv(f, sep="\t", dtype=str) for f in files], ignore_index=True)
    eff = cfg.cellsignal_effects
    unknown = sorted(set(c.EFFECT) - set(eff))
    if unknown:
        raise ValueError(f"unmapped cellsignal EFFECT values: {unknown}")
    df = pd.DataFrame({
        "source": c.ENTITYA_MOUSE, "target": c.ENTITYB_MOUSE,
        "sign": c.EFFECT.map(lambda e: eff[e][0]),
        "mechanism": c.EFFECT.map(lambda e: eff[e][1]),
        "phospho_kind": c.EFFECT.map(lambda e: eff[e][2]),
        "directed": True,
        "detail": c.PATHWAY_NAME + ";" + c.EFFECT,
    })
    # SCAFFOLD column: scaffold -> product of the scaffolded reaction (unsigned;
    # route priors give the sign where needed)
    sc = c[c.SCAFFOLD.fillna("na").ne("na")]
    scaffold = pd.DataFrame([
        dict(source=s.strip(), target=r.ENTITYB_MOUSE, sign=0, mechanism="scaffold",
             phospho_kind=None, directed=True,
             detail=f"{r.PATHWAY_NAME};scaffold of {r.ENTITYA_MOUSE}>{r.ENTITYB_MOUSE}@{r.SCAFFOLD_loc}")
        for r in sc.itertuples() for s in r.SCAFFOLD.split(",")
    ], columns=["source", "target", "sign", "mechanism", "phospho_kind", "directed", "detail"])
    return _records(pd.concat([df, scaffold], ignore_index=True), "cellsignal")


# ---------- manual curation ----------
def load_manual(cfg):
    """curation/manual_edges.csv: literature edges missing from the databases.
    Columns: source,target,sign,mechanism,site,reference,context_dependent,notes"""
    p = Path(cfg.paths.manual_edges)
    if not p.exists():
        return _records(pd.DataFrame(columns=COLS), "manual")
    m = pd.read_csv(p, dtype=str).fillna("")
    df = pd.DataFrame({
        "source": m.source, "target": m.target, "sign": m.sign.astype(int),
        "mechanism": m.mechanism, "directed": True,
        "detail": m.site + ";" + m.reference + (";" + m.notes).where(m.notes != "", ""),
    })
    return _records(df, "manual")


def load_cellsignal_location(cfg):
    """Relocation ('carry') edges; kept for later location modelling, not used as edges."""
    files = sorted(glob.glob(str(Path(cfg.paths.cellsignal_location_dir) / "*.tsv")))
    return pd.concat([pd.read_csv(f, sep="\t", dtype=str) for f in files], ignore_index=True)


# ---------- PSP ----------
def _term_sign(terms, pos, neg):
    p, n = bool(terms & pos), bool(terms & neg)
    return 1 if p and not n else -1 if n and not p else 0


def psp_site_signs(cfg):
    """(substrate, site) -> sign from the site's on_function terms (0 = none/mixed).
    Activity terms take priority; abundance terms are used only without them."""
    act = (set(cfg.psp_activity_positive), set(cfg.psp_activity_negative))
    abd = (set(cfg.psp_abundance_positive), set(cfg.psp_abundance_negative))
    out = {}
    for f in glob.glob(str(Path(cfg.paths.psp_dir) / "*_downstream_interactions.csv")):
        if os.path.getsize(f) < 200:
            continue
        d = pd.read_csv(f, dtype=str, usecols=["gene_mouse", "mod_rsd", "on_function"])
        d = d[d.mod_rsd.str.endswith("-p", na=False)]
        for (gene, rsd), grp in d.groupby(["gene_mouse", "mod_rsd"]):
            terms = {t.strip() for x in grp.on_function.dropna() for t in x.split(";")}
            if terms & (act[0] | act[1]):
                out[(gene, rsd[:-2])] = _term_sign(terms, *act)
            else:
                out[(gene, rsd[:-2])] = _term_sign(terms, *abd)
    return out


def load_psp(cfg):
    site_sign = psp_site_signs(cfg)
    psp = Path(cfg.paths.psp_dir)
    rows = []
    for f in glob.glob(str(psp / "*_upstream_kinases.csv")):
        if os.path.getsize(f) < 200:
            continue
        d = pd.read_csv(f, dtype=str, usecols=["SUBSTRATE_GENE_MOUSE", "KINASE_GENE_MOUSE", "MOD_RSD"])
        for r in d.dropna().itertuples(index=False):
            s = site_sign.get((r.SUBSTRATE_GENE_MOUSE, r.MOD_RSD), 0)
            rows.append((r.KINASE_GENE_MOUSE, r.SUBSTRATE_GENE_MOUSE, s, "phosphorylation", "kinase", r.MOD_RSD))
    for f in glob.glob(str(psp / "*_upstream_phosphatases.csv")):
        d = pd.read_csv(f, dtype=str, usecols=["SUBSTRATE_GENE_MOUSE", "PHOSPHATASE_MOUSE", "MOD_RSD"])
        for r in d.dropna().itertuples(index=False):
            s = -site_sign.get((r.SUBSTRATE_GENE_MOUSE, r.MOD_RSD), 0)
            rows.append((r.PHOSPHATASE_MOUSE, r.SUBSTRATE_GENE_MOUSE, s, "dephosphorylation", "phosphatase", r.MOD_RSD))
    df = pd.DataFrame(rows, columns=["source", "target", "sign", "mechanism", "phospho_kind", "detail"])
    df["directed"] = True
    return _records(df, "psp")


# ---------- harmonize + aggregate ----------
def harmonize(rec, cfg, universe):
    """Apply symbol map, drop self-loops and edges leaving the universe.
    Returns (kept, dropped) where dropped carries a 'reason' column."""
    m = symbol_map(cfg)
    rec = rec.copy()
    rec["source"] = rec.source.map(lambda x: m.get(x, x))
    rec["target"] = rec.target.map(lambda x: m.get(x, x))
    U = set(universe)
    out_u = ~(rec.source.isin(U) & rec.target.isin(U))
    self_ = rec.source == rec.target
    dropped = rec[out_u | self_].assign(reason=np.where(self_[out_u | self_], "self_loop", "outside_universe"))
    return rec[~(out_u | self_)].reset_index(drop=True), dropped.reset_index(drop=True)


def _agg_sign(signs):
    """Unanimous nonzero sign, else 0."""
    nz = {s for s in signs if s != 0}
    return nz.pop() if len(nz) == 1 else 0


def _majority_sign(signs):
    """Majority of nonzero signs (ties -> 0); used for PSP multi-site pairs."""
    return int(np.sign(sum(int(s) for s in signs)))


def aggregate(rec):
    """One row per (source, target, db). 'mixed' = both signs seen within the db.
    PSP pairs use the site majority, all other dbs require unanimity."""
    g = rec.groupby(["source", "target", "db"], sort=False)
    agg = g.agg(
        sign=("sign", _agg_sign),
        n_pos=("sign", lambda s: int((s > 0).sum())),
        n_neg=("sign", lambda s: int((s < 0).sum())),
        n_records=("sign", "size"),
        directed=("directed", "any"),
        mechanism=("mechanism", lambda s: ";".join(sorted({str(x) for x in s}))),
        phospho_kind=("phospho_kind", lambda s: ";".join(sorted({str(x) for x in s if x}))),
        detail=("detail", lambda s: " | ".join(list(dict.fromkeys(map(str, s)))[:8])),
    ).reset_index()
    psp = agg.db == "psp"
    agg.loc[psp, "sign"] = np.sign(agg.loc[psp, "n_pos"] - agg.loc[psp, "n_neg"]).astype(int)
    agg["mixed"] = (agg.n_pos > 0) & (agg.n_neg > 0)
    return agg


# ---------- cellsignal vs PSP ----------
def resolve_cellsignal_psp(cst_rec, psp_rec, restore=frozenset()):
    """Apply 'PSP wins' to cellsignal phospho-type records.

    restore: {(source, target)} cellsignal pairs the user chose to keep anyway.
    Returns (cst_kept_records, psp_sign_fill, resolution_report).
    psp_sign_fill: {(kinase, substrate): sign} for PSP pairs lacking a sign.
    """
    psp_pairs = {}
    for (a, b), grp in psp_rec.groupby(["source", "target"]):
        psp_pairs[(a, b)] = (set(grp.phospho_kind), _majority_sign(grp.sign), bool((grp.sign != 0).any()))
    keep, report, fill = [], [], {}
    for r in cst_rec.itertuples(index=False):
        if not r.phospho_kind:
            keep.append(True)
            continue
        hit = psp_pairs.get((r.source, r.target))
        if hit is None:
            status = "kept_no_psp"
        else:
            kinds, psp_sign, psp_signed = hit
            if r.phospho_kind not in kinds:
                status = "contradicted_kind"
            elif psp_sign != 0 and r.sign != 0 and psp_sign != r.sign:
                status = "contradicted_sign"
            else:
                status = "superseded_by_psp"
                # fill only when PSP has no signed site at all (not when its sites disagree)
                if not psp_signed and r.sign != 0:
                    fill[(r.source, r.target)] = r.sign
                    status = "superseded_psp_takes_cst_sign"
        if (r.source, r.target) in restore and status != "kept_no_psp":
            status += "+restored_by_user"
        keep.append(status == "kept_no_psp" or status.endswith("restored_by_user"))
        report.append(dict(source=r.source, target=r.target, cst_effect=r.detail, cst_sign=r.sign,
                           cst_kind=r.phospho_kind,
                           psp_kinds=";".join(sorted(hit[0])) if hit else "",
                           psp_sign=hit[1] if hit else None, status=status))
    return cst_rec[np.array(keep)].reset_index(drop=True), fill, pd.DataFrame(report)


# ---------- conflicts ----------
def sign_conflicts(agg):
    """Pairs where >= 2 dbs give nonzero, different signs (one row per db record)."""
    nz = agg[agg.sign != 0]
    per_pair = nz.groupby(["source", "target"]).sign.nunique()
    bad = per_pair[per_pair > 1].index
    out = nz.set_index(["source", "target"]).loc[bad].reset_index() if len(bad) else nz.iloc[0:0]
    out = out[["source", "target", "db", "sign", "n_records", "mechanism", "detail"]]
    out = out.sort_values(["source", "target", "db"]).reset_index(drop=True)
    out["action"] = ""  # user fills: keep | drop
    return out


def load_decisions(cfg):
    """User decisions, columns source,target,db,action with action in
    {drop, restore, keep, ''}. 'drop' removes that db's edge for the pair;
    'restore' keeps a cellsignal phospho row that PSP would override."""
    p = cfg.paths.sign_decisions
    if not Path(p).exists():
        return pd.DataFrame(columns=["source", "target", "db", "action"])
    d = pd.read_csv(p, dtype=str).fillna("")
    d["action"] = d.action.str.strip().str.lower()
    bad = sorted(set(d.action) - {"drop", "restore", "keep", ""})
    if bad:
        raise ValueError(f"unknown actions in {p}: {bad}")
    return d


def apply_decisions(agg, cfg):
    d = load_decisions(cfg)
    d = d[d.action == "drop"]
    if d.empty:
        return agg, 0
    key = agg.source + "|" + agg.target + "|" + agg.db
    drop = set(d.source + "|" + d.target + "|" + d.db)
    return agg[~key.isin(drop)].reset_index(drop=True), int(key.isin(drop).sum())


# ---------- main entry ----------
def curated_edges(cfg, universe=None):
    """Load all sources and return a dict of tables:
    records (kept raw records), dropped, agg (per pair x db, after PSP/CST rule and
    user decisions), cst_psp (resolution report), conflicts, mixed (within-db)."""
    universe = universe or load_universe(cfg)
    raw = {"omnipath": load_omnipath(cfg), "signor": load_signor(cfg), "trrust": load_trrust(cfg),
           "cellsignal": load_cellsignal(cfg), "psp": load_psp(cfg), "manual": load_manual(cfg)}
    kept, dropped = {}, []
    for db, r in raw.items():
        k, d = harmonize(r, cfg, universe)
        kept[db] = k
        dropped.append(d)
    dec = load_decisions(cfg)
    restore = set(zip(*[dec[(dec.action == "restore") & (dec.db == "cellsignal")][c] for c in ("source", "target")]))
    cst, fill, cst_psp = resolve_cellsignal_psp(kept["cellsignal"], kept["psp"], restore)
    kept["cellsignal"] = cst
    if fill:
        p = kept["psp"]
        pair = list(zip(p.source, p.target))
        p["sign"] = [fill.get(k, s) if s == 0 else s for k, s in zip(pair, p.sign)]
        p.loc[[k in fill for k in pair], "mechanism"] = "phosphorylation(sign_from_cellsignal)"
    records = pd.concat(kept.values(), ignore_index=True)
    records["sign"] = records.sign.astype(int)
    agg = aggregate(records)
    conflicts = sign_conflicts(agg)
    agg, n_dropped_by_user = apply_decisions(agg, cfg)
    return dict(raw_counts={db: len(r) for db, r in raw.items()}, records=records,
                dropped=pd.concat(dropped, ignore_index=True), agg=agg, cst_psp=cst_psp,
                conflicts=conflicts, mixed=agg[agg.mixed].reset_index(drop=True),
                n_dropped_by_user=n_dropped_by_user)
