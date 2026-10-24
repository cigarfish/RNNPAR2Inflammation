"""Step 12n validation: consolidated comparison + REPORT.md update.

    python scripts/consolidate_12n.py
    python tests/test_step12n_consolidation.py
"""
import json
import re
import sys
from pathlib import Path

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
o = cfg.paths.output_dir
out = o / "step12n"
if not check("12n outputs present", all((out / f).exists() for f in ["comparison.csv", "comparison.md", "summary.json"])):
    sys.exit(1)
agg = pd.read_csv(out / "comparison.csv")
runs = pd.read_csv(out / "runs.csv")
S = json.loads((out / "summary.json").read_text())

print("\n== comparison table")
harness = sorted(p.stem for p in (o / "harness").glob("*.json"))
check("every harness result is in the table (and nothing from sealed/)",
      sorted(runs.run) == harness, f"{len(runs)} runs")
need = ["12f_S5", "12f_C0", "12f_C4", "12f_stageA", "12j_J0", "12j_J1", "12j_J2", "12j_J3", "12j_J4", "12j_J5"]
check("all pre-12n variants present", set(need) <= set(agg.variant), sorted(set(need) - set(agg.variant)))
seeds = dict(zip(agg.variant, agg.n_seeds))
check("3 seeds for J0, J3, J5", all(seeds.get(v) == 3 for v in ["12j_J0", "12j_J3", "12j_J5"]),
      {v: seeds.get(v) for v in ["12j_J0", "12j_J3", "12j_J5"]})
unanch = [v for v in ["12j_J0U", "12j_J3U"] if v in seeds]
check("anchored vs unanchored runs present, 3 seeds each", unanch == ["12j_J0U", "12j_J3U"]
      and all(seeds[v] == 3 for v in unanch), {v: seeds.get(v) for v in ["12j_J0U", "12j_J3U"]})
if unanch:
    modes = runs[runs.variant.isin(unanch)]["mode"].unique().tolist()
    check("unanchored runs really trained unanchored", modes == ["unanchored"], modes)
check("verdicts carried into the summary", {"verdict.json", "verdict_geo.json", "selection.json"} <= set(S))

print("\n== REPORT.md")
rep = (ROOT / "REPORT.md").read_text(encoding="utf-8")
check("REPORT covers steps up to 12n", re.search(r"steps 0–12n", rep) is not None)
check("falsified-predictions section present", "## Predictions the measurements falsified" in rep)
ids = ["12d.1", "12e.1", "floor", "√d", "1.62", "GEO fit", "drug fit", "2.29", "negative transfer", "bulk"]
section = rep.split("## Predictions the measurements falsified")[-1].lower()
missing = [i for i in ids if i.lower() not in section]
check("each recorded falsification is listed", not missing, missing)
check("open blockers still listed as open (user decision 2026-10-16)",
      all(k in rep for k in ["VERIFY", "PAR2_26", "putative_isg"]))
check("REPORT quotes the consolidated numbers file", "outputs/step12n/comparison.md" in rep)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
