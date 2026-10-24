"""Step 0 validation: environment, config and data paths.

Run from the LEMBAS folder:
    python tests/test_step00.py
Prints one line per check and a final PASS/FAIL summary; exit code 1 on failure.
"""
import importlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def header(path, sep):
    # errors="replace": gene_binding_profiles.csv has broken "→" bytes in its notes column
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return [c.strip().strip('"') for c in f.readline().rstrip("\r\n").split(sep)]


# ---------- 1. environment ----------
print("== environment")
print(f"python {platform.python_version()} at {sys.executable}")
check("python >= 3.9", sys.version_info >= (3, 9), platform.python_version())

versions = {}
for pkg in ["numpy", "scipy", "pandas", "torch", "yaml", "networkx"]:
    try:
        mod = importlib.import_module(pkg)
        versions[pkg] = getattr(mod, "__version__", "?")
        check(f"import {pkg}", True, versions[pkg])
    except Exception as e:
        check(f"import {pkg}", False, repr(e))

if "torch" in versions and "scipy" in versions:
    import numpy as np
    import scipy.sparse as sp
    import torch

    # float64 sparse steady-state style iteration, as LEMBAS does
    A = sp.random(200, 200, density=0.02, format="csr", random_state=0) * 0.1
    x = np.zeros((200, 3))
    b = np.random.default_rng(0).normal(size=(200, 3))
    for _ in range(100):
        x = np.tanh(A.dot(x) + b)
    check("scipy sparse float64 iteration", np.isfinite(x).all())

    w = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
    (w @ torch.ones(5, 1, dtype=torch.float64)).sum().backward()
    check("torch float64 autograd", w.grad is not None and torch.isfinite(w.grad).all(),
          f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")

# ---------- 2. config ----------
print("\n== config")
cfg = None
try:
    from lembas_par2 import load_config
    cfg = load_config()
    check("load config.yaml", True, str(ROOT / "config.yaml"))
except Exception as e:
    check("load config.yaml", False, repr(e))

if cfg is not None:
    P = cfg.paths
    root = ROOT.resolve()

    # every path inside the project
    outside = [k for k, p in P.items() if not str(p).startswith(str(root))]
    check("all paths inside project root", not outside, ", ".join(outside))

    # every input path exists
    for k, p in P.items():
        if k in ("output_dir", "sign_decisions", "manual_edges", "collectri"):  # created later / optional
            continue
        check(f"exists: {k}", p.exists(), str(p.relative_to(root)))

    # output dir writable
    try:
        P.output_dir.mkdir(exist_ok=True)
        t = P.output_dir / ".write_test"
        t.write_text("ok")
        t.unlink()
        check("output_dir writable", True, str(P.output_dir.relative_to(root)))
    except Exception as e:
        check("output_dir writable", False, repr(e))

    # ---------- 3. data shape checks ----------
    print("\n== data files")
    n = cfg.expected_counts["geo_contexts"]
    degs = sorted(P.deg_dir.glob("*_DEG.csv"))
    enr = sorted(P.enrichment_dir.glob("*_Reactome_enrichment_full.csv"))
    check(f"DEG files = {n}", len(degs) == n, str(len(degs)))
    check(f"enrichment files = {n}", len(enr) == n, str(len(enr)))

    import pandas as pd
    meta = pd.read_csv(P.geo_metadata)
    ids_meta = set(meta.geo_id)
    ids_deg = {f.name[: -len("_DEG.csv")] for f in degs}
    ids_enr = {f.name[: -len("_Reactome_enrichment_full.csv")] for f in enr}
    check("metadata IDs == DEG IDs == enrichment IDs", ids_meta == ids_deg == ids_enr,
          f"meta-deg={sorted(ids_meta - ids_deg)} deg-meta={sorted(ids_deg - ids_meta)} "
          f"deg-enr={sorted(ids_deg ^ ids_enr)}")

    expected_cols = {
        "string_ppi": ("\t", ["gene1", "gene2", "combined_score"]),
        "reactome_pathway_gene": (",", ["Pathway_ID", "Signaling_Genes", "TAKE"]),
        "gene_binding_profiles": (",", ["gene", "binding_mode"]),
        "gene_features": (",", ["gene", "loc_cytoplasm", "loc_nuclear", "length"]),
        "omnipath": ("\t", ["source_mouse_genesymbol", "target_mouse_genesymbol",
                            "is_stimulation", "is_inhibition"]),
        "signal_entity_edges": (",", ["src_type", "src_id", "biochemical_role", "dst_type", "dst_id"]),
        "signal_entity_packages": (",", ["signal_symbol", "positive_source_or_release_genes"]),
        "perturbation_studies": (",", ["study_id", "training_role", "is_holdout"]),
        "perturbation_contrasts": (",", ["contrast_id", "study_id"]),
        "perturbation_interventions": (",", ["intervention_id", "contrast_id", "graph_target_id"]),
        "perturbation_observations": (",", ["observation_id", "contrast_id", "graph_target_id",
                                            "effect_direction"]),
        "external_metadata": ("\t", ["geo_id", "inhibitor target", "drug"]),
    }
    for k, (sep, cols) in expected_cols.items():
        h = header(P[k], sep)
        missing = [c for c in cols if c not in h]
        check(f"columns: {k}", not missing, f"missing {missing}" if missing else "")

    # encoding: every top-level text file must be UTF-8, except known damage confined
    # to free-text columns (gene_binding_profiles.csv 'notes')
    text_keys = [k for k, p in P.items() if p.is_file()]
    for k in text_keys:
        raw = P[k].read_bytes()
        try:
            raw.decode("utf-8")
            continue
        except UnicodeDecodeError:
            pass
        df = pd.read_csv(P[k], sep="\t" if P[k].suffix in (".tsv", ".txt") else ",",
                         dtype=str, encoding="utf-8", encoding_errors="replace")
        bad_cols = [c for c in df.columns if df[c].str.contains("�", na=False).any()]
        check(f"encoding: {k} damage only in free-text columns", bad_cols == ["notes"],
              f"non-UTF-8 bytes in columns {bad_cols}; read with encoding_errors='replace'")

    # headerless files: check column count on first line
    check("reactome_relation has 2 columns", len(header(P.reactome_relation, "\t")) == 2)
    check("trrust has 4 columns", len(header(P.trrust, "\t")) == 4)

    for k, pattern in [("signor_dir", "*.tsv"), ("psp_dir", "*_upstream_kinases.csv"),
                       ("cellsignal_dir", "*.tsv"), ("cellsignal_location_dir", "*.tsv")]:
        files = list(P[k].glob(pattern))
        check(f"{k} non-empty", len(files) > 0, f"{len(files)} {pattern}")
    check("cellsignal-STING.tsv present", (P.cellsignal_dir / "cellsignal-STING.tsv").exists())

    # external MEK/ERK contrasts
    from lembas_par2.config import external_file
    ext_cols = ["gene_symbol", "effect", "effect_available", "padj", "contrast_id"]
    for stem in cfg.external_contrasts:
        f = external_file(cfg, stem)
        if check(f"external exists: {stem}", f.exists()):
            h = header(f, "\t")
            missing = [c for c in ext_cols if c not in h]
            check(f"external columns: {stem}", not missing, f"missing {missing}" if missing else "")

    # holdouts are real study IDs and marked as holdout
    studies = pd.read_csv(P.perturbation_studies, encoding="utf-8-sig")
    hs = studies[studies.study_id.isin(cfg.holdout_studies)]
    check("holdout studies found", len(hs) == len(cfg.holdout_studies),
          f"{sorted(hs.study_id)}")
    check("holdout studies flagged is_holdout=1", (hs.is_holdout.astype(str) == "1").all(),
          hs[["study_id", "is_holdout"]].to_dict("records"))

# ---------- summary ----------
n_fail = sum(not ok for _, ok in results)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
if cfg is not None:
    report = dict(python=platform.python_version(), packages=versions,
                  checks=[dict(name=n, ok=ok) for n, ok in results])
    (cfg.paths.output_dir / "step00_report.json").write_text(json.dumps(report, indent=2))
print("STEP 0:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
