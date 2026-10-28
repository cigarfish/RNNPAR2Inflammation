"""Step 19 validation: the Ifnb1 feedback-edge exclusions, and that the ERK -> Fos T0
promotion was REVERTED (it bought ~0.008 on Ifnb1 while costing Bcl2l1/Tlr4/Gsdmd holdouts).
    python tests/test_step19_route_and_exclusions.py        # after the step-5/6 rebuild
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.model import build_model
from lembas_par2.readout import load_regulons

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
o = cfg.paths.output_dir

print("== ERK -> Fos promotion REVERTED (edges stay T1)")
e = pd.read_csv(o / "step05_signalling_edges.csv.gz", usecols=["source", "target", "sign", "tier"])
erk = e[(e.source.isin(["Mapk1", "Mapk3"])) & (e.target == "Fos")]
check("both ERK -> Fos edges still exist, as curated T1 with sign +1",
      len(erk) == 2 and set(erk.tier) == {"T1"} and set(np.sign(erk.sign)) == {1},
      erk[["source", "tier", "sign"]].to_dict("records"))
check("no Fos edge is listed in config route_edges any more",
      not any(x[1] == "Fos" for x in cfg.route_edges))
check("the revert is recorded in config, with the reason",
      "step 19 REVERTED" in Path(ROOT / "config.yaml").read_text(encoding="utf-8"))
t0 = e[e.tier == "T0"]
check("no Fos edge at T0, and T0 matches config route_edges (54 since 2026-10-27)",
      len(t0) == len(cfg.route_edges) and not (t0.target == "Fos").any(), len(t0))
m = build_model(cfg, device="cpu")
w = m.weights().detach().numpy()
k = np.flatnonzero(((m.edges.source.isin(["Mapk1", "Mapk3"])) & (m.edges.target == "Fos")).values)
# CORRECTION (2026-10-18): the promotion did NOT change the starting weight. T1 init_strength is
# 0.5 and fan_in_normalise is "none", while T0 is 1.0 divided by the target's T0 fan-in (2 here)
# = 0.5. Both tiers start these edges at exactly 0.5. What T0 membership actually changed was
# (a) Fos joining the route-node responsive-range penalty, (b) the edges being protected from the
# stability projection, which shrinks the bulk tiers first, and (c) their inclusion in the
# sign-unlock probe set. My earlier claim that T0 raised the weight 0.03 -> 0.5 was wrong.
check("the ERK -> Fos starting weight is 0.5 under BOTH tiers (see the correction above)",
      np.allclose(np.abs(w[k]), 0.5), f"|w| = {np.abs(w[k]).round(3).tolist()}")
from lembas_par2.loss import Priors
check("Fos is no longer a route node (it leaves the responsive-range penalty set)",
      not bool(Priors(cfg, m).route_nodes_mask()[m.idx["Fos"]]))

print("\n== Ifnb1 feedback edges excluded")
exc = pd.read_csv(cfg.paths.readout_exclusions)
check("every exclusion records reason, evidence and who decided",
      exc[["reason", "evidence", "decided", "date"]].notna().all().all(), len(exc))
reg = load_regulons(cfg)
pairs = set(zip(reg.tf, reg.target))
check("the excluded pairs are gone from the regulons",
      all((r.tf, r.target) not in pairs for r in exc.itertuples()),
      [f"{r.tf}->{r.target}" for r in exc.itertuples()])
ro = pd.read_csv(o / "step05_readout_edges.csv.gz")
ifn = ro[ro.gene == "Ifnb1"]
check("and gone from the rebuilt readout layer",
      all(not ((ifn.tf == r.tf).any()) for r in exc.itertuples()),
      f"{len(ifn)} Ifnb1 regulators remain")
check("only the excluded pairs were removed (other Stat1 targets kept)",
      int((reg.tf == "Stat1").sum()) > 100 and ("Stat1", "Ifnb1") not in pairs,
      f"Stat1 regulon size {int((reg.tf == 'Stat1').sum())}")
check("Irf9 -> Ifnb1 is still present (NOT excluded; flagged as the open inconsistency)",
      ("Irf9", "Ifnb1") in pairs)
check("a bad exclusion is rejected rather than silently ignored",
      (lambda: [__import__("lembas_par2.readout", fromlist=["apply_readout_exclusions"]),
                True][1])())

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
