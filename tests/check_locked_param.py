"""Step 14 checker: adopt model.locked_param = abs? Rule in config `locked_param_test`, fixed
before running; harness JSON + checkpoints only, holdouts never read.

    python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants J0A
    python tests/check_locked_param.py
ADOPT iff (1) every candidate seed is eligible under the harness rules, (2) non-inferior on the
seed medians (drug residualised r on signal drugs, GEO residualised r), and (3) the median relative
movement of locked T1 + signed readout edges is >= mobility_gain_min x the baseline's.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.eval_harness import default_eligibility, load_model, result_path
from lembas_par2.model import build_model

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
lt = cfg.locked_param_test
o = cfg.paths.output_dir


def run_name(v, s):
    return v if s == 0 else f"{v}_s{s}"


def mobility(v, s):
    """Median relative change from initialisation of locked T1 signalling + signed readout edges."""
    ck = o / "source_variants" / run_name(v, s) / "checkpoint.pt"
    m, meta = load_model(cfg, ck, "cpu")
    c0 = load_config()
    c0["model"]["fan_in_normalise"] = cfg.source_test["base"]["fan_in_normalise"]
    c0["model"]["locked_param"] = m.locked_param
    m0 = build_model(c0, device="cpu", seed=s)
    m0.initialize()
    w0, w = m0.weights().detach().numpy(), m.weights().detach().numpy()
    r0, r = m0.readout_weights().detach().numpy(), m.readout_weights().detach().numpy()
    li = m.locked_idx.numpy()
    t1 = li[m.tier[li] == "T1"]
    rl = m.r_locked_idx.numpy()
    rel = np.concatenate([np.abs(w[t1] - w0[t1]) / np.abs(w0[t1]), np.abs(r[rl] - r0[rl]) / np.abs(r0[rl])])
    return float(np.median(rel))


rows = []
for v in [lt["baseline"], lt["candidate"]]:
    for s in lt["seeds"]:
        p = result_path(cfg, "12j", run_name(v, s))
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        ok, why = default_eligibility(r, rho_max=cfg.source_test["rho_max"], transmit_min=cfg.source_test["transmit_min"])
        rows.append(dict(variant=v, seed=s, eligible=ok, why="; ".join(why), mobility=mobility(v, s),
                         **{k: r[k] for k in ["r_tf_drug_resid_signal", "r_tf_geo_resid", "r_tf_drug", "r_tf_geo",
                                              "route_dSting1", "rho_absA"]}))
t = pd.DataFrame(rows)
need = len(lt["seeds"])
n = t.groupby("variant").size().to_dict()
if not check("all seeds present for baseline and candidate", all(n.get(v, 0) >= need for v in [lt["baseline"], lt["candidate"]]), n):
    print(f"run: python scripts/run_source_variants.py --seeds {' '.join(map(str, lt['seeds']))} --variants {lt['candidate']}")
    sys.exit(1)
print(t.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
med = t.groupby("variant").median(numeric_only=True)
b, c = med.loc[lt["baseline"]], med.loc[lt["candidate"]]

c1 = bool(t[t.variant == lt["candidate"]].eligible.all())
drops = {k: float(b[k] - c[k]) for k in lt["noninferiority"]}
c2 = all(drops[k] <= tol for k, tol in lt["noninferiority"].items())
gain = float(c["mobility"] / b["mobility"]) if b["mobility"] > 0 else np.inf
c3 = gain >= lt["mobility_gain_min"]
print(f"\n[INFO] (1) every candidate seed eligible: {c1}")
print(f"[INFO] (2) non-inferior: " + ", ".join(f"{k} {b[k]:+.3f} -> {c[k]:+.3f} (drop {drops[k]:+.3f}, max {lt['noninferiority'][k]})" for k in drops) + f" -> {c2}")
print(f"[INFO] (3) locked-edge mobility {b['mobility']:.4f} -> {c['mobility']:.4f} ({gain:.1f}x, need {lt['mobility_gain_min']}x) -> {c3}")
print(f"[INFO] for reference: route dSting1 {b['route_dSting1']:+.3f} -> {c['route_dSting1']:+.3f}; "
      f"raw drug r {b['r_tf_drug']:+.3f} -> {c['r_tf_drug']:+.3f}; raw GEO r {b['r_tf_geo']:+.3f} -> {c['r_tf_geo']:+.3f}")
verdict = (f"ADOPT locked_param = abs ({lt['candidate']})" if (c1 and c2 and c3) else
           f"DO NOT ADOPT: " + "; ".join(x for x, ok in [("ineligible seed", c1), ("inferior fit", c2), ("mobility gain too small", c3)] if not ok))
print(f"\n[VERDICT] {verdict}")
(o / "step14").mkdir(exist_ok=True)
(o / "step14" / "verdict.json").write_text(json.dumps(dict(verdict=verdict, eligible=c1, noninferior=c2, drops=drops,
                                                           mobility_gain=gain, medians=med.to_dict()), indent=2, default=float))
t.to_csv(o / "step14" / "runs.csv", index=False)
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
