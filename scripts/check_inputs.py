"""Check that every input the model needs is present, before running the pipeline.

The step scripts fail with a plain FileNotFoundError when an input is missing, which names the
path but not what the file is or where to get it. Run this first:

    python scripts/check_inputs.py

It reports each configured input as present or missing, groups the missing ones by the section of
data/README.md that explains how to obtain them, and exits non-zero if anything required is absent.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config

# (config key or literal path, human description, required?, where it comes from)
INPUTS = [
    ("string_ppi", "STRING mouse backbone, Reactome-filtered", True, "network"),
    ("reactome_pathway_gene", "Reactome pathway -> gene membership", True, "network"),
    ("reactome_relation", "Reactome pathway hierarchy", True, "network"),
    ("omnipath", "OmniPath signed mouse interactions", True, "network"),
    ("trrust", "TRRUST v2 mouse", True, "network"),
    ("signor_dir", "SIGNOR mouse export (directory)", True, "network"),
    ("psp_dir", "PhosphoSitePlus per-gene CSVs (directory)", True, "psp"),
    ("collectri", "CollecTRI mouse regulon", True, "network"),
    ("cellsignal_dir", "CST-curated edges (included)", True, "included"),
    ("cellsignal_location_dir", "CST subcellular location (included)", True, "included"),
    ("gene_binding_profiles", "derived binding annotation", False, "network"),
    ("gene_features", "derived feature matrix", False, "network"),
    ("geo_metadata", "48 GEO context accessions (included)", True, "included"),
    ("deg_dir", "per-context differential expression (directory)", True, "expression"),
    ("enrichment_dir", "pathway program scores (optional)", False, "optional"),
    ("sign_decisions", "sign-conflict decisions (included)", True, "included"),
    ("readout_sign_decisions", "readout sign decisions (included)", True, "included"),
    ("readout_exclusions", "readout exclusions (included)", True, "included"),
    ("manual_edges", "manually curated edges (included)", True, "included"),
]

WHERE = {
    "network": "data/README.md -> 'Required - the network'",
    "psp": "data/README.md -> the PhosphoSitePlus warning. You MUST obtain this yourself; "
           "it is not redistributable.",
    "expression": "data/README.md -> 'Required - expression data'",
    "included": "shipped with this repository - if missing, your checkout is incomplete",
    "optional": "data/README.md -> 'Optional'",
}


def main():
    cfg = load_config()
    rows, missing = [], []
    for key, desc, required, group in INPUTS:
        p = getattr(cfg.paths, key, None)
        if p is None:
            rows.append(("?", key, desc, "no such config key"))
            continue
        p = Path(p)
        if p.exists():
            if p.is_dir():
                n = sum(1 for _ in p.iterdir())
                rows.append(("OK", key, desc, f"{n} entries"))
                if n == 0 and required:
                    missing.append((key, desc, group, "directory is empty"))
            else:
                rows.append(("OK", key, desc, f"{p.stat().st_size / 1e6:.1f} MB"))
        else:
            rows.append(("--" if not required else "MISSING", key, desc, str(p)))
            if required:
                missing.append((key, desc, group, str(p)))

    w = max(len(r[1]) for r in rows + [('', 'drug_targets', '', '')])
    print(f"{'':<8}{'config key':<{w}}  description")
    # two inputs live under cfg.conditions rather than cfg.paths; append BEFORE printing
    dt = ROOT / cfg.conditions["drug_targets"]
    if dt.exists():
        rows.append(("OK", "drug_targets", "drug -> target mapping (included)",
                     f"{dt.stat().st_size / 1e3:.1f} kB"))
    else:
        rows.append(("MISSING", "drug_targets", "drug -> target mapping (included)", str(dt)))
        missing.append(("drug_targets", "drug -> target mapping", "included", str(dt)))

    ext = ROOT / cfg.conditions["drug_deg_dir"]
    if ext.exists() and any(ext.iterdir()):
        rows.append(("OK", "drug_deg_dir", "GSE219229 drug DEG tables",
                     f"{sum(1 for _ in ext.iterdir())} files"))
    else:
        rows.append(("MISSING", "drug_deg_dir", "GSE219229 drug DEG tables", str(ext)))
        missing.append(("drug_deg_dir", "GSE219229 drug DEG tables", "expression", str(ext)))

    print("-" * (10 + w + 44))
    for status, key, desc, extra in rows:
        print(f"[{status:^6}] {key:<{w}}  {desc}")
        if status in ("MISSING", "?"):
            print(f"{'':<9}{'':<{w}}  -> {extra}")

    if not missing:
        print("\nAll required inputs present. Next:\n  python tests/test_step00.py")
        return 0

    print(f"\n{len(missing)} required input(s) missing:\n")
    for group in dict.fromkeys(g for _, _, g, _ in missing):
        items = [m for m in missing if m[2] == group]
        print(f"  {WHERE[group]}")
        for key, desc, _, where in items:
            print(f"    - {key}: {desc}")
            print(f"        expected at {where}")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
