"""Measured kinase-inhibitor target profiles for the GSE219229 drugs (step 12q).

Source: Gualdrini et al. 2024, Mol Syst Biol, Dataset EV1 (data_external/44320_2024_40_MOESM2_ESM.xlsx),
sheet "Kinase-to-Inhibitor annotation": per drug, every kinobeads target from Klaeger et al. 2017
with its EC50 (nM, human-lysate competition binding), the mouse gene id, the kinase the drug was
"Selected for", and the concentration actually used ("nM used" = the EC50 of the selected kinase).

Occupancy at the dose used, single-site binding:  occ = c / (c + EC50).
The selected kinase therefore sits at occ = 0.5 by construction.

Why this matters (measured 2026-10-16): the drugs were dosed for a chosen kinase, and for most of
them that kinase is not the textbook target in curation/drug_targets.csv (e.g. Afatinib dosed at
2.5 uM for MAPK9, Midostaurin for TBK1, Omipalisib for IRAK3), so the model's perturbation inputs
were mis-specified for most drugs.
"""
import re

import numpy as np
import pandas as pd

EV1 = "data_external/44320_2024_40_MOESM2_ESM.xlsx"
SHEET = "Kinase-to-Inhibitor annotation"


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def load_ev1(cfg):
    k = pd.read_excel(cfg.root / EV1, SHEET, header=1)
    k = k[["Drug", "Target", "Mmu_id", "EC50", "Selected for", "nM used"]].dropna(subset=["Drug", "Target"])
    k["mouse"] = k.Mmu_id.astype(str).str.replace(r"_\d+$", "", regex=True)
    k["EC50"] = pd.to_numeric(k.EC50, errors="coerce")
    k["nM used"] = pd.to_numeric(k["nM used"], errors="coerce")
    k["occ"] = k["nM used"] / (k["nM used"] + k.EC50)
    ours = pd.read_csv(cfg.root / "curation" / "drug_targets.csv")
    name = {norm(d): d for d in ours.drug}
    k["drug_ours"] = k.Drug.map(lambda d: name.get(norm(d)))
    return k


def occupancy_matrix(cfg, symbol_map=None):
    """DataFrame: rows GSE219229 condition ids, columns mouse genes, values occupancy (0 = not a
    kinobeads target). Duplicate rows (same drug, same gene) keep the highest occupancy."""
    k = load_ev1(cfg)
    if symbol_map:
        k["mouse"] = k.mouse.map(lambda g: symbol_map.get(g, g))
    k = k.dropna(subset=["drug_ours", "occ"])
    k["cid"] = "GSE219229_" + k.drug_ours
    return k.pivot_table(index="cid", columns="mouse", values="occ", aggfunc="max").fillna(0.0)


def selected_vs_annotated(cfg):
    """Per drug: the kinase(s) it was dosed for vs our annotated targets."""
    k = load_ev1(cfg)
    ours = pd.read_csv(cfg.root / "curation" / "drug_targets.csv").set_index("drug")
    rows = []
    for d, g in k.groupby("drug_ours"):
        sel = {s.strip().upper() for s in re.split(r"[;,]", str(g["Selected for"].iloc[0])) if s.strip()}
        ann = {t.upper() for t in ours.at[d, "targets"].split(";")}
        hi = set(g.loc[g.occ >= 0.5, "mouse"].str.upper())
        rows.append(dict(drug=d, selected_for=";".join(sorted(sel)), nM_used=float(g["nM used"].iloc[0]),
                         annotated=ours.at[d, "targets"], n_kinobeads_targets=len(g),
                         n_occ_ge_half=len(hi), selected_in_annotation=bool(sel & ann),
                         annotated_occ_ge_half=bool(ann & hi)))
    return pd.DataFrame(rows)
