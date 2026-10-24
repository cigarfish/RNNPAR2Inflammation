"""Freeze the presented LEMBAS model (J0) so later work cannot change it silently.

Copies the J0 / J0U / J0F checkpoints and the step-5/6 inputs they were built from into
outputs/frozen/<tag>/, with a manifest of sha256 hashes, package versions and the selection
provenance. Re-running it verifies an existing freeze instead of overwriting it.

    python scripts/freeze_model.py                 # create (or verify) the freeze
    python scripts/freeze_model.py --verify        # verify only, non-zero exit on mismatch
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config

TAG = "J0_2026-10-18"
VARIANTS = ["J0", "J0U", "J0F"]
SEEDS = [0, 1, 2, 3, 4]
INPUTS = ["step05_signalling_edges.csv.gz", "step05_readout_edges.csv.gz", "step05_tf_outputs.csv",
          "step05_regulons.csv.gz", "step06_conditions.csv", "step06_tf_activity.csv.gz",
          "step06_gene_lfc.csv.gz", "step06_gene_padj.csv.gz", "step06_directions.csv"]
CURATION = ["sign_decisions.csv", "readout_sign_decisions.csv", "manual_edges.csv", "drug_targets.csv"]

PROVENANCE = """J0 is the presented model. Its configuration was fixed by rules written before the
validation data were read:
 * step 12e selected tf_mode gain_global with a non-negative fitted gain, fan_in_normalise sqrt and
   the targeted stability projection, by a rule on validation fit, holdouts sealed (variant S5).
 * step 12j tested four readout alternatives (J1-J4) and, by its pre-registered rule, adopted none,
   so the S5 readout (= J0) remained the baseline. Selection metric: residualised drug r.
 * the three held-out PAR2-KO papers were read ONCE, in step 13, after the configuration was fixed.
 * everything after step 13 is a labelled sensitivity analysis: the Irf3/Rela -> Ifnb1 readout sign
   correction (step 14 curation) and the locked-weight reparameterisation test (step 14, NOT
   adopted). Neither changed the step-13 verdict (WEAKENED: 8/10 holdout observations, p 0.055).
J0 was not chosen for its PAR2-KO prediction. From here on, any model change is a NEW variant with
a new name; these files are never edited."""

KEY_RESULTS = {
    "step13_holdout_agreement_J0": "8/10 (seeds 7-9), one-sided binomial p 0.055",
    "step13_PAR2_KO_dSting1_J0_pooled": "+0.094 (5 seeds: +0.054 to +0.124)",
    "step13_PAR2_KO_dSting1_J0U_unanchored": "+0.003 (route priors off)",
    "step13_sign_free_route_edges_flipped": "0 of 77 T0 edges in 10 runs",
    "step12f_drug_residualised_r_J0": "0.056 (noise ceiling 0.545)",
    "step12f0_constant_profile_baseline": "raw r 0.570 drug / 0.714 GEO",
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(cfg):
    """{relative destination: source path} for everything the freeze pins."""
    o = cfg.paths.output_dir
    files = {}
    for v in VARIANTS:
        for s in SEEDS:
            run = v if s == 0 else f"{v}_s{s}"
            for name in ["checkpoint.pt", "report.json"]:
                p = o / "source_variants" / run / name
                if p.exists():
                    files[f"checkpoints/{run}/{name}"] = p
            nz = o / "source_variants" / run / "nuisance.pt"
            if nz.exists():
                files[f"checkpoints/{run}/nuisance.pt"] = nz
    for name in INPUTS:
        if (o / name).exists():
            files[f"inputs/{name}"] = o / name
    for name in CURATION:
        p = ROOT / "curation" / name
        if p.exists():
            files[f"curation/{name}"] = p
    files["config.yaml"] = ROOT / "config.yaml"
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tag", default=TAG)
    a = ap.parse_args()
    cfg = load_config()
    dest = cfg.paths.output_dir / "frozen" / a.tag
    man_p = dest / "manifest.json"
    files = collect(cfg)
    missing = [k for k, v in files.items() if not v.exists()]
    if missing:
        sys.exit(f"missing source files: {missing}")

    if man_p.exists():
        man = json.loads(man_p.read_text())
        bad = []
        for rel, h in man["files"].items():
            p = dest / rel
            if not p.exists():
                bad.append(f"{rel}: gone from the freeze")
            elif sha256(p) != h:
                bad.append(f"{rel}: hash changed inside the freeze")
            elif rel in files and sha256(files[rel]) != h:
                bad.append(f"{rel}: the live copy differs from the frozen one")
        print(f"[freeze] verifying {len(man['files'])} files in {dest.relative_to(ROOT)}")
        for b in bad:
            print(f"  MISMATCH {b}")
        if bad:
            print("\nA live-copy mismatch is expected after any rebuild or retraining: the frozen "
                  "model is the one in this folder, and the live tree has moved on.")
        else:
            print("  all files match the manifest and the live tree")
        sys.exit(1 if bad and a.verify else 0)
    if a.verify:
        sys.exit(f"no freeze at {dest}")

    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for rel, src in sorted(files.items()):
        d = dest / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d)
        hashes[rel] = sha256(d)
    import numpy
    import pandas
    import torch
    man = dict(
        tag=a.tag, created=str(date.today()), model="J0 (5 seeds), with J0U / J0F for the controls",
        provenance=PROVENANCE.strip(), key_results=KEY_RESULTS,
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      numpy=numpy.__version__, pandas=pandas.__version__, platform=platform.platform()),
        archive_of_pre_correction_state="outputs/archive_pre_readout_signs_2026-10-17/",
        files=hashes)
    man_p.write_text(json.dumps(man, indent=2))
    print(f"[freeze] wrote {len(hashes)} files + manifest to {dest.relative_to(ROOT)}")
    print(f"  checkpoints: {sum(1 for k in hashes if k.startswith('checkpoints/') and k.endswith('checkpoint.pt'))}")
    print(f"  verify later with: python scripts/freeze_model.py --verify")


if __name__ == "__main__":
    main()
