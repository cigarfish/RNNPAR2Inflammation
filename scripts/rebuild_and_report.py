"""Rebuild steps 5, 6 and 9 after a curation change, then report exactly what moved.

Archives the current outputs first, re-runs the step tests that write them, and diffs the result
against the archive so the scope of a curation change is visible rather than assumed.

    python scripts/rebuild_and_report.py --tag step19_erkfos_stat_exclusion
    python scripts/rebuild_and_report.py --tag <name> --no-archive     # rebuild only
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config

STEPS = ["test_step05", "test_step06", "test_step09a", "test_step09b"]
EXTRA = ["test_readout_sign_decisions", "test_step19_route_and_exclusions"]
FILES = ["step05_signalling_edges.csv.gz", "step05_readout_edges.csv.gz", "step05_tf_outputs.csv",
         "step06_conditions.csv", "step06_tf_activity.csv.gz", "step06_gene_lfc.csv.gz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="archive folder name, e.g. step19_...")
    ap.add_argument("--no-archive", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    o = cfg.paths.output_dir
    arch = o / f"archive_pre_{a.tag}"

    if not a.no_archive:
        arch.mkdir(parents=True, exist_ok=True)
        n = 0
        for pat in ["step05_*", "step06_*", "step09b_*"]:
            for p in o.glob(pat):
                if p.is_file():
                    shutil.copy2(p, arch / p.name)
                    n += 1
        print(f"[archive] {n} files -> {arch.relative_to(ROOT)}")

    for t in STEPS + EXTRA:
        f = ROOT / "tests" / f"{t}.py"
        if not f.exists():
            continue
        r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, cwd=ROOT)
        tail = [l for l in r.stdout.splitlines() if "SUMMARY" in l or "checks passed" in l]
        bad = [l for l in r.stdout.splitlines() if l.startswith("[FAIL]")]
        print(f"[{t}] {tail[-1].strip() if tail else 'no summary'}" + ("  <-- FAILURES" if bad else ""))
        for l in bad:
            print("   ", l)
        if r.returncode != 0 and not bad:
            print("   ", (r.stderr.strip().splitlines() or ["(no stderr)"])[-1])

    if a.no_archive or not arch.exists():
        return
    print("\n== what changed vs the archive")
    for f in FILES:
        pa, pb = arch / f, o / f
        if not (pa.exists() and pb.exists()):
            continue
        A, B = pd.read_csv(pa), pd.read_csv(pb)
        if f == "step05_signalling_edges.csv.gz":
            ka, kb = A.set_index(["source", "target"]), B.set_index(["source", "target"])
            common = ka.index.intersection(kb.index)
            tier = (ka.loc[common, "tier"] != kb.loc[common, "tier"])
            print(f"  {f}: {len(B) - len(A):+d} edges; tier changed for {int(tier.sum())}: "
                  + ", ".join(f"{s}->{t} {ka.at[(s, t), 'tier']}->{kb.at[(s, t), 'tier']}"
                              for s, t in common[tier][:6]))
        elif f == "step05_readout_edges.csv.gz":
            sa, sb = set(zip(A.tf, A.gene)), set(zip(B.tf, B.gene))
            print(f"  {f}: {len(B) - len(A):+d} edges; removed "
                  + (", ".join(f"{a_}->{b_}" for a_, b_ in sorted(sa - sb)[:6]) or "none")
                  + (f"; added {len(sb - sa)}" if sb - sa else ""))
        elif f.endswith(".csv.gz") and "tf_activity" in f:
            A, B = pd.read_csv(pa, index_col=0), pd.read_csv(pb, index_col=0)
            d = (A - B.reindex_like(A)).abs()
            ch = d.index[(d > 1e-6).any(axis=1)]
            print(f"  {f}: {len(ch)} TF rows changed" + (f": {list(ch)[:8]}" if len(ch) else ""))
        else:
            same = A.equals(B)
            print(f"  {f}: {'unchanged' if same else 'CHANGED'}")
    print("\nNOTE: checkpoints trained before this rebuild are no longer loadable if an edge moved "
          "between signed and unsigned, or between the signalling and readout layers. Retrain:\n"
          "  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")


if __name__ == "__main__":
    main()
