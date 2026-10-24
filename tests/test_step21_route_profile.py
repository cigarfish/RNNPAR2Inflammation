"""Step 21 checks: the route profile is coherent, and the "most-changed" claim is as stated.

Guards the two things panel D now asserts that could quietly become false: that every step of the
route moves in the direction its curated sign requires, and that the route's dominance of the
network response is reported with its circularity rather than without it.

    python tests/test_step21_route_profile.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
o = cfg.paths.output_dir / "step21_route"
if not (o / "route.csv").exists():
    sys.exit(f"missing {o}; run: python scripts/run_route_profile.py")

R = pd.read_csv(o / "route.csv").sort_values("order")
rep = json.loads((o / "report.json").read_text())

print("== the route is complete and ordered from the receptor")
check("the route starts at the receptor and ends at IRF3",
      R.node.iloc[0] == "F2rl1" and R.node.iloc[-1] == "Irf3",
      f"{R.node.iloc[0]} ... {R.node.iloc[-1]}")
check("both arms from PAR2 to ERK are present",
      {"Arrb1", "Arrb2"} <= set(R.node) and {"Gnaq", "Plcb1", "Kras"} <= set(R.node))
check("the MAPK core the user asked for is present (RAF, MEK, ERK)",
      {"Raf1", "Braf", "Map2k1", "Map2k2", "Mapk1", "Mapk3"} <= set(R.node))
check("every route node resolved to a network node", R.delta_lps.notna().all())

print("\n== direction: no step contradicts its curated sign")
d = R.set_index("node").delta_lps
down = ["F2rl1", "Gnaq", "Plcb1", "Plcb3", "SE_DAG", "Kras", "Hras", "Raf1", "Braf",
        "Map2k1", "Map2k2", "Mapk1", "Mapk3", "Rps6ka1", "Rps6ka3", "Mcl1", "Bcl2", "Bcl2l1"]
up = ["Gsk3b", "Bad", "Bcl2l11", "Bax", "Bak1", "SE_CYTO_MTDNA", "Cgas", "Sting1", "Tbk1", "Irf3"]
bad_down = [n for n in down if n in d.index and d[n] > 0]
bad_up = [n for n in up if n in d.index and d[n] < 0]
check("ERK and everything it activates goes DOWN on PAR2 KO", not bad_down, bad_down)
check("everything ERK inhibits, and the whole STING arm, goes UP", not bad_up, bad_up)
check("the claim arrives: Sting1 is positive", float(d["Sting1"]) > 0, f"{d['Sting1']:+.4f}")

print("\n== attenuation and ranking, reported with the caveat")
atten = abs(float(d["F2rl1"])) / abs(float(d["Sting1"]))
check("the signal attenuates from receptor to STING (it does not grow)", atten > 1,
      f"|PAR2| / |STING| = {atten:.1f}x")
top = pd.DataFrame(rep["top25_lps"])
n_route = int(top.on_route.sum())
check("most of the 25 most-changed nodes are on the route", n_route >= 15,
      f"{n_route} of 25; median route rank {int(R.rank_lps.median())} of {rep['n_nodes']}")
src = (ROOT / "figures" / "make_panel_d_route.py").read_text(encoding="utf-8")
check("the panel states that this ranking is PARTLY BUILT IN, not a clean finding",
      "partly built in" in src and "T0" in src)
check("the panel states which model it uses and why it is not the frozen one",
      "frozen J0" in src and "0.991" in src)
check("the panel says the ablation bars are not contribution shares",
      "NOT contribution shares" in src and "necessity" in src)

print("\n== the beta-arrestin asymmetry is surfaced, not buried")
check("ARRB1 and ARRB2 differ by more than 10x, and the panel flags it",
      abs(float(d["Arrb1"])) > 10 * abs(float(d["Arrb2"])) and "ARRB2" in src,
      f"ARRB1 {d['Arrb1']:+.4f} vs ARRB2 {d['Arrb2']:+.4f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
