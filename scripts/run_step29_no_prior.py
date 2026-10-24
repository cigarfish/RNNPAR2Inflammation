"""Step 29: run with NO manual prior — every route edge demoted to T1 — and see where Sting1 goes.

The T0 route was built to describe one mechanism, and `SE_CYTO_MTDNA -> Cgas` sits in it at
weight 1.0 carrying 94 % of cGAS's response (step 27). That is a designed-in answer. This flips
`route_tier` to T1 so the 76 route edges keep their curated signs but compete on the same footing
as the other ~16 000 curated edges, with no exemption from the stability projection.

**HOW TO READ THE RESULT, fixed here before it is run.** Removing T0 subjects every route edge to
the stability projection, which shrinks T1 weights to ~8 % of their configured value. So EVERY
delta will shrink by roughly an order of magnitude and "Sting1 is near zero" is the expected
outcome whatever the truth is. The informative quantities are therefore:

  1. the SIGN of dSting1, and
  2. the scale-free ratio dSting1 / |dMapk1| - how much of the receptor signal arrives at STING,
     compared with the same ratio under the prior.

A near-zero dSting1 with no consistent sign across seeds means **the data alone do not determine
the direction** - which is a result about the evidence, not about STING. It is NOT evidence that
Sting1 goes down. Step 13's unanchored probe already pointed this way (J0U dSting1 +0.003,
J0F +0.001).

    python scripts/run_step29_no_prior.py            # flip to T1, rebuild, report at init
    python scripts/run_step29_no_prior.py --revert   # flip back to T0 and rebuild
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NODES = ["F2rl1", "Gnaq", "Mapk1", "Mapk3", "Mcl1", "Bcl2l11", "Bax", "Bak1",
         "SE_IP3", "SE_CA2", "Vdac1", "SE_CYTO_MTDNA", "Cgas", "Sting1", "Tbk1", "Irf3", "Src"]


def set_tier(value):
    p = ROOT / "config.yaml"
    s = p.read_text(encoding="utf-8")
    for old in ("route_tier: T0", "route_tier: T1"):
        if old in s:
            p.write_text(s.replace(old, f"route_tier: {value}", 1), encoding="utf-8")
            return old.split()[-1]
    sys.exit("route_tier not found in config.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--no-rebuild", action="store_true")
    a = ap.parse_args()

    target = "T0" if a.revert else "T1"
    was = set_tier(target)
    print(f"route_tier {was} -> {target}")

    if not a.no_rebuild:
        tag = "step29_restore_prior" if a.revert else "step29_no_prior"
        r = subprocess.run([sys.executable, "scripts/rebuild_and_report.py", "--tag", tag], cwd=ROOT)
        if r.returncode != 0:
            print("[rebuild] non-zero exit; the tier is already flipped, so either fix the "
                  "failure or run --revert")

    from lembas_par2 import load_config
    from lembas_par2.model import build_model
    from lembas_par2.step13 import ko_delta
    import pandas as pd

    cfg = load_config()
    e = pd.read_csv(cfg.paths.output_dir / "step05_signalling_edges.csv.gz",
                    usecols=["tier"], low_memory=False)
    print(f"\ntier counts: {e.tier.value_counts().to_dict()}")

    m = build_model(cfg, device="cpu")
    m.initialize()
    d = ko_delta(m, cfg.step13["contexts"]["lps"], cfg.step13["ko_kappa"]).numpy()
    print(f"\nPAR2 KO (LPS), INITIALISED model, route_tier = {target}:\n")
    for n in NODES:
        if n in m.idx:
            print(f"   {n:16s} {d[m.idx[n]]:+.5f}")
    sting, mapk = float(d[m.idx["Sting1"]]), float(d[m.idx["Mapk1"]])
    print(f"\n   dSting1 / |dMapk1| = {sting / abs(mapk):+.4f}"
          if abs(mapk) > 1e-9 else "\n   dMapk1 is zero; the route does not transmit at all")
    print(f"   dSting1 sign: {'POSITIVE' if sting > 0 else 'NEGATIVE' if sting < 0 else 'ZERO'}")
    print("\n   Initialisation only - the fitted answer needs the retrain below, and the sign at\n"
          "   initialisation is NOT the result. Read the rules in this script's docstring first.")
    print("\nnext:\n  python scripts/run_step23_retrain.py\n"
          "  python scripts/run_node_contributions.py --node Sting1\n"
          f"  python scripts/run_step29_no_prior.py --revert   # to put the prior back")


if __name__ == "__main__":
    main()
