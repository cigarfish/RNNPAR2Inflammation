"""Step 1 validation: gene universe and DEG symbol harmonization.

Run from the LEMBAS folder:
    python tests/test_step01.py
Writes outputs/step01_universe.txt, step01_log2fc.csv.gz, step01_padj.csv.gz,
step01_coverage.csv and step01_report.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2 import genes as G

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
out = cfg.paths.output_dir
out.mkdir(exist_ok=True)
A = cfg.symbol_aliases

# ---------- 1. universe ----------
print("== universe")
raw_string = G.string_genes(cfg)
n_raw = cfg.expected_counts["raw_string_genes"]
check(f"raw STRING genes = {n_raw}", len(raw_string) == n_raw, len(raw_string))
for name, other in [("Reactome genes", G.reactome_genes(cfg)),
                    ("Reactome TAKE=1 genes", G.reactome_genes(cfg, take_only=True)),
                    ("feature-matrix genes", G.feature_genes(cfg))]:
    check(f"raw STRING genes == {name}", other == raw_string,
          f"only STRING {len(raw_string - other)}, only {name} {len(other - raw_string)}")

M = cfg.universe_merge
check("merge sources are in raw STRING", all(h in raw_string for h in M), [h for h in M if h not in raw_string])
merged_into_existing = [h for h, m in M.items() if m in raw_string]
check("all merges but Znf598 collapse onto an existing mouse node",
      sorted(set(M) - set(merged_into_existing)) == ["Znf598"], merged_into_existing)

U = G.load_universe(cfg)
Us = set(U)
n_exp = cfg.expected_counts["universe_genes"]
check(f"universe size after merge = {n_exp}", len(U) == n_exp, len(U))
check("no merge source left in universe", not (set(M) & Us), sorted(set(M) & Us))

bind = G.canonical(pd.read_csv(cfg.paths.gene_binding_profiles, usecols=["gene"],
                               encoding_errors="replace").gene, cfg)
check("binding-profile genes subset of universe", bind <= Us,
      f"{len(bind)} genes; {len(Us - bind)} universe genes lack a profile "
      f"(route genes lacking: {[g for g in cfg.route_genes if g not in bind]})")

missing_route = [g for g in cfg.route_genes if g not in Us]
check("all route genes in universe", not missing_route, missing_route)

# ---------- 2. alias table ----------
print("\n== alias table")
raw_symbols = {gid: set(G.read_deg_raw(cfg, gid).SYMBOL.astype(str).str.strip())
               for gid in G.geo_ids(cfg)}
check("alias sources not in universe", all(a not in Us for a in A),
      [a for a in A if a in Us])
check("alias targets in universe", all(t in Us for t in A.values()),
      [t for t in A.values() if t not in Us])
co = {f"{a}->{t}": [gid for gid, s in raw_symbols.items() if a in s and t in s]
      for a, t in A.items()}
co = {k: v for k, v in co.items() if v}
check("old and current symbol never co-occur in one DEG file", not co, co)
used = {a: sum(a in s for s in raw_symbols.values()) for a in A}
check("every alias is used by at least one DEG file", all(used.values()),
      [a for a, n in used.items() if n == 0])

# ---------- 3. harmonized DEG ----------
print("\n== harmonized DEG")
lfc, padj = G.deg_matrices(cfg, U)
check("matrix shape = universe x 48", lfc.shape == (n_exp, cfg.expected_counts["geo_contexts"]),
      lfc.shape)
check("matrix index == universe", list(lfc.index) == U)
check("log2fc finite where measured", np.isfinite(lfc.values[~lfc.isna().values]).all())
p = padj.values[~padj.isna().values]
check("padj within [0, 1]", ((p >= 0) & (p <= 1)).all(), f"{p.size} values")

measured = lfc.notna()
cov = measured.mean().sort_values()
check("per-context coverage >= 0.70", (cov >= 0.70).all(),
      f"min {cov.index[0]} {cov.iloc[0]:.3f}, median {cov.median():.3f}")

n_sting = int(measured.loc["Sting1"].sum())
no_sting = sorted(measured.columns[~measured.loc["Sting1"]])
check("Sting1 measured in 47/48 contexts (all but GSE158267)",
      n_sting == 47 and no_sting == ["GSE158267"], f"{n_sting}, missing {no_sting}")
check("Cgas measured in 46/48 contexts", int(measured.loc["Cgas"].sum()) == 46,
      f"{int(measured.loc['Cgas'].sum())}, missing {sorted(measured.columns[~measured.loc['Cgas']])}")
route_cov = {g: int(measured.loc[g].sum()) for g in cfg.route_genes}
check("route genes measured in >= 40 contexts", min(route_cov.values()) >= 40, route_cov)

# value preserved through rename: Tmem173 raw value == harmonized Sting1 value
old = sorted(gid for gid, s in raw_symbols.items() if "Tmem173" in s)
ok = True
for gid in old:
    raw = G.read_deg_raw(cfg, gid)
    v_raw = float(raw.loc[raw.SYMBOL.astype(str).str.strip() == "Tmem173", "log2FoldChange"].iloc[0])
    ok &= np.isclose(v_raw, lfc.at["Sting1", gid])
check(f"Tmem173 values carried to Sting1 in all {len(old)} old-annotation files", ok)

# one row per gene in every context
dups = {gid: int(G.load_deg(cfg, gid, U).gene.duplicated().sum()) for gid in G.geo_ids(cfg)}
check("no duplicate genes per context", not any(dups.values()),
      {k: v for k, v in dups.items() if v})

check("Trp53 measured in all 48 contexts", int(measured.loc["Trp53"].sum()) == 48,
      int(measured.loc["Trp53"].sum()))

# universe genes never measured anywhere would hint at more naming mismatches
never = sorted(lfc.index[measured.sum(axis=1) == 0])
check("every universe gene measured in >= 1 context", not never, never)
rare = sorted(lfc.index[(measured.sum(axis=1) > 0) & (measured.sum(axis=1) <= 10)])
print(f"[INFO] universe genes measured in 1-10 contexts: {len(rare)}")

# ---------- outputs ----------
(out / "step01_universe.txt").write_text("\n".join(U) + "\n")
lfc.to_csv(out / "step01_log2fc.csv.gz")
padj.to_csv(out / "step01_padj.csv.gz")
cov.rename("coverage").to_csv(out / "step01_coverage.csv")

n_fail = sum(not ok for _, ok, _ in results)
(out / "step01_report.json").write_text(json.dumps(dict(
    universe=len(U), never_measured=never, route_coverage=route_cov,
    checks=[dict(name=n, ok=ok, detail=d) for n, ok, d in results]), indent=2))
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 1:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
