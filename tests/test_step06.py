"""Step 6 validation: conditions, TF-activity / gene / program targets, directions, folds.

Run from the LEMBAS folder (needs data_external/GSE219229_DEG and data/collectri_mouse.csv):
    python tests/test_step06.py

Writes to outputs/:
    step06_conditions.csv          one row per condition (inputs/inhibit as JSON lists)
    step06_tf_activity.csv.gz      output TF x condition ULM t-values
    step06_gene_lfc.csv.gz         readout gene x condition log2FC (clipped)
    step06_gene_padj.csv.gz        readout gene x condition padj
    step06_programs.csv            MPAS / ISG scores per condition
    step06_directions.csv          perturbation sign constraints (train + holdout rows)
    step06_dropped_observations.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.conditions import build_conditions, to_jsonable
from lembas_par2.readout import build_layers

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
c = cfg.conditions
out = cfg.paths.output_dir
L = build_layers(cfg)
D = build_conditions(cfg, L)
cond, tfa, prog, dirs = D["conditions"], D["tf_activity"], D["programs"], D["directions"]
signal_nodes = set(L["signalling"].source) | set(L["signalling"].target)

# ---------- 1. conditions ----------
print("== conditions")
n = cond.source.value_counts().to_dict()
print(f"[INFO] conditions by source: {n}")
check("57 drug conditions (56 GSE219229 + GSE134486)", n.get("drug") == 57, n.get("drug"))
check("48 GEO conditions", n.get("geo") == 48, n.get("geo"))
check("perturbation training contrasts loaded", n.get("perturb", 0) > 0, n.get("perturb"))
check("3 holdout contrasts", n.get("holdout") == 3, sorted(cond[cond.source == "holdout"].condition_id))
check("skipped drugs absent", not cond.name.isin(c["drug_skip"]).any())
used = {x for col in ["ref_inputs", "trt_inputs", "trt_inhibit"] for v in cond[col] for x in v}
check("every input / inhibited node is a signalling node", used <= signal_nodes, sorted(used - signal_nodes))
geo = cond[cond.source == "geo"]
unmapped = geo[(geo.trt_inputs.map(len) == 0) & (geo.trt_inhibit.map(len) == 0)]
check("every GEO stimulus mapped to inputs or inhibition", unmapped.empty, unmapped.name.tolist())
drug = cond[cond.source == "drug"]
check("every drug inhibits >= 1 network node", (drug.trt_inhibit.map(len) > 0).all(),
      drug[drug.trt_inhibit.map(len) == 0].name.tolist())
pert = cond[cond.source == "perturb"]
print(f"[INFO] perturbation backgrounds: "
      f"{pert.notes.str.extract(r'-> (.*)$')[0].value_counts().to_dict()}")
check("finite non-negative weights", np.isfinite(cond.weight.astype(float)).all() and (cond.weight >= 0).all())

w = drug.set_index("name")
nulls = w[w.n_sig < c["drug_weight"]["null_below"]]
check("gene-count weight of null drugs is the null weight", (nulls.w_genes == c["drug_weight"]["null_weight"]).all(),
      f"{len(nulls)} null drugs: {sorted(nulls.index)}")

# marker-TF re-weighting
print("\n== drug marker-TF weighting")
markers = {tf for m in c["drug_markers"].values() for tf, _ in m}
miss = sorted(markers - set(tfa.index))
print(f"[INFO] marker TFs not among TF outputs (ignored): {miss}" if miss else "[INFO] all marker TFs are TF outputs")
check("weight = max(gene-count weight, marker weight)",
      np.allclose(drug[~drug.name.eq("Trametinib")].weight,
                  np.maximum(drug[~drug.name.eq("Trametinib")].w_genes, drug[~drug.name.eq("Trametinib")].w_marker)))
nw = c["drug_weight"]["null_weight"]
check("Ruxolitinib up-weighted by JAK markers despite 0 significant genes", w.loc["Ruxolitinib", "weight"] > nw,
      f"{w.loc['Ruxolitinib', 'weight']:.2f} ({w.loc['Ruxolitinib', 'markers_used']})")
for d in ["Encorafenib", "GDC0994", "Ulixertinib"]:
    check(f"{d} (weak/mixed ERK evidence: MPAS up, markers ~flat) stays at the null weight", w.loc[d, "weight"] == nw,
          f"marker score {w.loc[d, 'marker_score']:+.2f} ({w.loc[d, 'markers_used']}), MPAS {prog.loc['GSE219229_' + d, 'MPAS']:+.2f}")
print("[INFO] markers_used values are (expected sign x TF activity): positive = moved the expected way")
tab = (drug[drug.marker_set != ""].sort_values(["marker_set", "marker_score"], ascending=[True, False])
       [["name", "marker_set", "n_sig", "w_genes", "marker_score", "w_marker", "weight", "markers_used"]])
print(tab.round(2).to_string(index=False))
changed = drug[drug.weight > drug.w_genes + 1e-9]
print(f"[INFO] drugs up-weighted by markers: {len(changed)} -> {sorted(changed.name)}")
print(f"[INFO] drug weights: Pimasertib {w.loc['Pimasertib', 'weight']:.2f}, Refametinib {w.loc['Refametinib', 'weight']:.2f}, "
      f"BGT226 {w.loc['BGT226', 'weight']:.2f}, Trametinib {w.loc['Trametinib', 'weight']:.2f}")

# ---------- 2. holdouts never trained ----------
print("\n== holdouts")
hold = cond[cond.source == "holdout"]
check("holdout conditions have weight 0 and no fold", (hold.weight == 0).all() and hold.fold.isna().all())
hs = set(cfg.holdout_studies)
check("no training row from a holdout study",
      not cond[(cond.source != "holdout") & cond.dataset.isin(hs)].shape[0]
      and not dirs[(dirs.source != "holdout") & dirs.condition_id.str.startswith(tuple(hs))].shape[0])
hd = dirs[dirs.source == "holdout"]
print(f"[INFO] holdout sign targets: {len(hd)} -> "
      + ", ".join(f"{r.condition_id}:{r.node}({'+' if r.sign > 0 else '-' if r.sign < 0 else '0'})" for r in hd.itertuples()))
dropped = D["dropped_observations"]
hdrop = dropped[dropped.condition_id.str.startswith(tuple(hs))]
print(f"[INFO] holdout observations not scorable: "
      + ", ".join(f"{r.condition_id}:{r.target_raw}[{r.reason}]" for r in hdrop.itertuples()))

# ---------- 3. targets ----------
print("\n== targets")
check("TF activity matrix: output TFs x conditions", tfa.shape == (len(L["tf_outputs"]), len(D["contrasts"])), tfa.shape)
print(f"[INFO] TF activity finite fraction {np.isfinite(tfa.values).mean():.3f}; "
      f"gene targets {D['gene_lfc'].shape}, measured fraction {D['gene_lfc'].notna().values.mean():.3f}")

mpas = prog.MPAS
for cid in ["GSE219229_Pimasertib", "GSE134486_Trametinib"]:
    check(f"MPAS clearly negative in {cid}", mpas[cid] < -0.5, f"{mpas[cid]:+.2f}")
print("[INFO] MPAS other MEK/ERK/BRAF drugs: " + ", ".join(
    f"{d}:{mpas['GSE219229_' + d]:+.2f}" for d in ["Selumetinib", "Refametinib", "GDC0994", "Ulixertinib", "Encorafenib"]))
mek = ["GSE219229_Pimasertib", "GSE134486_Trametinib"]
v = tfa.loc["Elk1", mek]
check("Elk1 activity down under MEK inhibition", (v < 0).all(), ", ".join(f"{x:+.2f}" for x in v))
print(f"[INFO] Etv4 activity under MEK inhibition (Pimasertib, Trametinib): "
      + ", ".join(f"{x:+.2f}" for x in tfa.loc["Etv4", mek]))
v = tfa.loc["Stat1", "GSE219229_Filgotinib"]
check("Stat1 activity down under the JAK1 inhibitor Filgotinib", v < 0, f"{v:+.2f}")
print("[INFO] Stat1 activity under other JAK inhibitors: " + ", ".join(
    f"{d}:{tfa.loc['Stat1', 'GSE219229_' + d]:+.2f}"
    for d in ["BMS911543", "Tofacitinib", "Momelotinib", "Pacritinib", "Ruxolitinib"]))

isg_viral = prog.loc[geo[geo.cls == "Viral"].condition_id, "ISG"]
check("ISG score positive in most viral GEO sets", (isg_viral > 0).mean() >= 0.75,
      f"{int((isg_viral > 0).sum())}/{len(isg_viral)}, mean {isg_viral.mean():+.2f}")
lps = geo[geo.name == "LPS"].condition_id
rela = tfa.loc["Rela", lps]
check("NF-kB (Rela) activity positive in most LPS GEO sets", (rela > 0).mean() >= 0.75,
      f"{int((rela > 0).sum())}/{len(rela)}")

# ---------- 4. directions ----------
print("\n== perturbation directions")
tr = dirs[dirs.source == "perturb"]
check("training sign constraints built", len(tr) > 0,
      f"{len(tr)} ({tr.layer.value_counts().to_dict()}), signs {tr.sign.value_counts().to_dict()}")
print(f"[INFO] dropped observations by reason: {dropped.reason.value_counts().to_dict()}")

# ---------- 5. folds ----------
print("\n== folds")
gf = geo.assign(series=geo.condition_id.str.extract(r"^(GSE\d+)")[0])
check("GEO series sub-contexts share a fold", (gf.groupby("series").fold.nunique() == 1).all())
tab = pd.crosstab(gf.fold, gf.cls)
print("[INFO] GEO folds x class:\n" + tab.to_string())
check("every GEO fold has all three stimulus classes", (tab > 0).all().all())
lps_per = gf[gf.name == "LPS"].groupby("fold").size().reindex(range(c["cv_folds"]), fill_value=0)
check("LPS sets spread (<= 2 per fold)", lps_per.max() <= 2, lps_per.tolist())
dtab = drug.groupby("fold").size()
check("drug folds balanced", dtab.max() - dtab.min() <= 1, dtab.tolist())

# ---------- outputs ----------
to_jsonable(cond).to_csv(out / "step06_conditions.csv", index=False)
tfa.to_csv(out / "step06_tf_activity.csv.gz")
D["gene_lfc"].to_csv(out / "step06_gene_lfc.csv.gz")
D["gene_padj"].to_csv(out / "step06_gene_padj.csv.gz")
prog.to_csv(out / "step06_programs.csv")
dirs.to_csv(out / "step06_directions.csv", index=False)
dropped.to_csv(out / "step06_dropped_observations.csv", index=False)

n_fail = sum(not ok for _, ok, _ in results)
(out / "step06_report.json").write_text(json.dumps(dict(
    conditions=n, checks=[dict(name=nm, ok=ok, detail=d) for nm, ok, d in results]), indent=2))
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 6:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
