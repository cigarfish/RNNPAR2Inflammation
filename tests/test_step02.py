"""Step 2 validation: curated edge loaders, PSP-over-cellsignal rule, sign-conflict report.

Run from the LEMBAS folder:
    python tests/test_step02.py

Writes to outputs/:
    step02_edges_by_db.csv     one row per (source, target, db) after rules/decisions
    step02_cst_vs_psp.csv      how each cellsignal phospho row was resolved against PSP
    step02_within_db_mixed.csv pairs with both signs inside one database
    step02_review.csv          everything needing a decision (route-related first)

To record decisions: copy step02_review.csv to curation/sign_decisions.csv, fill the
'action' column (drop / restore / keep), and re-run. Already-made decisions are
carried into the next review file.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2 import edges as E
from lembas_par2.genes import load_universe

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_rows", 200)

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
out = cfg.paths.output_dir
U = set(load_universe(cfg))
R = E.curated_edges(cfg, sorted(U))
agg, rec = R["agg"], R["records"]

# ---------- 1. loading ----------
print("== loading")
for db in E.DBS:
    check(f"{db}: records loaded", R["raw_counts"][db] > 0, R["raw_counts"][db])
d = R["dropped"].groupby(["db", "reason"]).size().to_dict()
print(f"[INFO] dropped records: {d}")
check("SIGNOR summary file excluded (no empty entities)", rec[rec.db == "signor"].source.notna().all())
fam_left = rec[(rec.db == "signor") & (rec.source.str.contains("/") | rec.target.str.contains("/"))]
check("SIGNOR family names expanded (no '/' left)", fam_left.empty, len(fam_left))
erk = rec[(rec.db == "signor") & rec.source.isin(["Mapk1", "Mapk3"])]
check("SIGNOR Erk1/2 rows expanded to Mapk1/Mapk3", len(erk) > 0, f"{len(erk)} records")
check("all endpoints in universe", rec.source.isin(U).all() and rec.target.isin(U).all())
check("no self-loops", (rec.source != rec.target).all())
check("signs in {-1, 0, 1}", set(rec.sign.unique()) <= {-1, 0, 1}, sorted(rec.sign.unique()))
loc = E.load_cellsignal_location(cfg)
check("cellsignal_location loaded (not used as edges)", len(loc) > 0 and set(loc.EFFECT) == {"carry"},
      f"{len(loc)} rows, effects {sorted(set(loc.EFFECT))}")

summary = agg.groupby("db").agg(pairs=("sign", "size"), pos=("sign", lambda s: int((s > 0).sum())),
                                neg=("sign", lambda s: int((s < 0).sum())),
                                unsigned=("sign", lambda s: int((s == 0).sum())))
print("[INFO] pairs per db after rules:\n" + summary.to_string())

# ---------- 2. PSP site signs ----------
print("\n== PSP site signs")
site = E.psp_site_signs(cfg)
check("PSP Bcl2l11 S69 site sign = -1 (degradation)", site.get(("Bcl2l11", "S69")) == -1, site.get(("Bcl2l11", "S69")))
check("PSP Bax S163 site sign = +1", site.get(("Bax", "S163")) == 1, site.get(("Bax", "S163")))

# ---------- 3. cellsignal vs PSP ----------
print("\n== cellsignal vs PSP")
cp = R["cst_psp"]
print("[INFO] " + str(cp.status.value_counts().to_dict()))
psp_pairs = set(zip(rec[rec.db == "psp"].source, rec[rec.db == "psp"].target))
cst_phos = rec[(rec.db == "cellsignal") & rec.phospho_kind.fillna("").ne("")]
restored = set(zip(cp[cp.status.str.endswith("restored_by_user")].source,
                   cp[cp.status.str.endswith("restored_by_user")].target))
leak = [p for p in zip(cst_phos.source, cst_phos.target) if p in psp_pairs and p not in restored]
check("no cellsignal phospho row kept where PSP has the pair (unless restored)", not leak, leak[:10])
contra = cp[cp.status.str.startswith("contradicted")]
print("[INFO] cellsignal rows contradicted by PSP (removed unless restored):")
print(contra.to_string(index=False) if len(contra) else "  none")

# ---------- 4. known route edges ----------
print("\n== known edges (db: expected sign)")
expected = [
    ("F2rl1", "Gnaq", "cellsignal", 1), ("Arrb1", "Mapk1", "omnipath", 1),
    ("Map2k1", "Mapk1", "signor", 1), ("Map2k1", "Mapk3", "psp", 1),
    ("Map2k1", "Mapk1", "psp", 1), ("Mapk1", "Bcl2l11", "psp", -1),
    ("Mapk1", "Bcl2l11", "omnipath", -1), ("Mapk1", "Mcl1", "omnipath", 1),
    ("Mapk1", "Rps6ka1", "psp", 1), ("Rps6ka1", "Bad", "psp", -1),
    ("Gsk3b", "Bax", "psp", 1), ("Gsk3b", "Mcl1", "omnipath", -1),
    ("Mcl1", "Bak1", "signor", -1), ("Mcl1", "Bax", "signor", -1),
    ("Bcl2l11", "Bax", "signor", 1), ("Bcl2", "Bax", "signor", -1),
    ("Cgas", "Sting1", "cellsignal", 1), ("Sting1", "Tbk1", "cellsignal", 1),
    ("Tbk1", "Irf3", "psp", 1), ("Trp53", "Bax", "trrust", 1),
    ("Src", "Cgas", "manual", -1),
]
lookup = {(r.source, r.target, r.db): r.sign for r in agg.itertuples()}
for s, t, db, sg in expected:
    got = lookup.get((s, t, db))
    check(f"{s} -> {t} [{db}] = {sg:+d}", got == sg, f"got {got}")

# cellsignal SCAFFOLD column -> unsigned scaffold edges
scaf = agg[(agg.db == "cellsignal") & agg.mechanism.str.contains("scaffold")]
arrb = {(s, t) for s, t in zip(scaf.source, scaf.target) if s in ("Arrb1", "Arrb2")}
want = {(a, b) for a in ("Arrb1", "Arrb2") for b in ("Map2k1", "Map2k2", "Mapk1", "Mapk3")}
check("Arrb1/2 scaffold edges to Map2k1/2 and Mapk1/3 loaded", want <= arrb, sorted(want - arrb))
print(f"[INFO] cellsignal scaffold pairs: {len(scaf)} "
      f"(scaffolds: {sorted(set(scaf.source))})")

# user drops of collapsed indirect edges
for s in ("Gnai1", "Gnas"):
    check(f"{s} -> Src [omnipath] dropped by decision", (s, "Src", "omnipath") not in lookup)

# ---------- 5. review report ----------
print("\n== sign conflicts")
route = set(cfg.route_genes)
conf = R["conflicts"].assign(issue="cross_db_sign_conflict")
cst_rows = contra.rename(columns={"cst_sign": "sign", "cst_effect": "detail"}).assign(
    db="cellsignal", issue="cellsignal_contradicted_by_psp",
    mechanism=lambda x: "psp " + x.psp_kinds + " sign " + x.psp_sign.astype(str), n_records=1, action="")
review = pd.concat([conf, cst_rows[conf.columns]], ignore_index=True)
review["route_related"] = review.source.isin(route) & review.target.isin(route)

prev = E.load_decisions(cfg)
if len(prev):
    prev_map = {(r.source, r.target, r.db): r.action for r in prev.itertuples()}
    review["action"] = [prev_map.get(k, "") for k in zip(review.source, review.target, review.db)]
# keep decisions that are not conflicts (e.g. manual_curation drops) so copying the
# review file back over the decisions file never loses them
if len(prev):
    in_review = set(zip(review.source, review.target, review.db))
    not_in = pd.Series([k not in in_review for k in zip(prev.source, prev.target, prev.db)], index=prev.index)
    extra = prev[not_in & (prev.action != "")]
    review = pd.concat([review, extra.reindex(columns=review.columns)], ignore_index=True)
    review["route_related"] = review.source.isin(route) & review.target.isin(route)
review = review.sort_values(["route_related", "issue", "source", "target", "db"],
                            ascending=[False, True, True, True, True])
cols = ["issue", "route_related", "source", "target", "db", "sign", "n_records", "mechanism", "detail", "action"]
review = review[cols]

n_pairs = review.groupby(["source", "target"]).ngroups
check("review report built", len(review) > 0, f"{n_pairs} pairs, {len(review)} rows")
rr = review[review.route_related]
print(f"[INFO] route-related rows to review ({rr.groupby(['source', 'target']).ngroups} pairs):")
print(rr.drop(columns=["route_related"]).to_string(index=False) if len(rr) else "  none")
print(f"[INFO] decisions file: {cfg.paths.sign_decisions} "
      f"({'found, ' + str(len(prev)) + ' rows' if len(prev) else 'not found'}); "
      f"edges dropped by user decisions: {R['n_dropped_by_user']}")
mixed = R["mixed"]
print(f"[INFO] within-db pairs with both signs (PSP: site majority, others: sign 0): "
      f"{mixed.db.value_counts().to_dict()}")

# ---------- outputs ----------
agg.to_csv(out / "step02_edges_by_db.csv", index=False)
cp.to_csv(out / "step02_cst_vs_psp.csv", index=False)
mixed.to_csv(out / "step02_within_db_mixed.csv", index=False)
review.to_csv(out / "step02_review.csv", index=False)

n_fail = sum(not ok for _, ok, _ in results)
(out / "step02_report.json").write_text(json.dumps(dict(
    raw_counts=R["raw_counts"], pairs_per_db=summary.to_dict(), review_pairs=n_pairs,
    checks=[dict(name=n, ok=ok, detail=dt) for n, ok, dt in results]), indent=2))
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 2:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
