"""Step 12f validation: the shared evaluation harness (lembas_par2/eval_harness.py).

Run from the LEMBAS folder after `python scripts/run_harness.py`:
    python tests/test_step12f_harness.py
"""
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from lembas_par2 import load_config
from lembas_par2.eval_harness import SCHEMA_KEYS, evaluate_run, load_model, result_path, select

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
o = cfg.paths.output_dir
runs = ["S5", "C0", "C4"]
paths = {v: result_path(cfg, "12f", v) for v in runs}
if not check("harness results present", all(p.exists() for p in paths.values()), {v: p.exists() for v, p in paths.items()}):
    sys.exit(1)
R = {v: json.loads(p.read_text()) for v, p in paths.items()}

# ---------- 1. schema ----------
print("\n== schema")
check("every result has exactly the schema keys", all(set(r) == set(SCHEMA_KEYS) for r in R.values()))
check("validation split: 12 drugs (9 with r_max >= 0.3), 10 GEO",
      all((r["n_val_drug"], r["n_val_drug_signal"], r["n_val_geo"]) == (12, 9, 10) for r in R.values()),
      {v: (r["n_val_drug"], r["n_val_drug_signal"], r["n_val_geo"]) for v, r in R.items()})
check("no holdout condition in any per-condition table",
      not any(c["source"] == "holdout" or c["condition_id"].startswith(("PAR2_26", "PAR2_29", "PAR2_30"))
              for r in R.values() for c in r["per_condition"]))

# ---------- 2. reproduces the recorded numbers ----------
print("\n== reproduces recorded results (tol 2e-3: recorded on GPU)")
ob = pd.read_csv(o / "objective_variants" / "summary.csv").set_index("variant")
cs = pd.read_csv(o / "composition_variants" / "summary.csv").set_index("variant")
tol = 2e-3
s5 = R["S5"]
check("S5: val r(TF), drug, GEO", abs(s5["r_tf_val"] - ob.at["S5", "val_r_tf"]) < tol
      and abs(s5["r_tf_drug"] - ob.at["S5", "val_r_tf_drug"]) < tol and abs(s5["r_tf_geo"] - ob.at["S5", "val_r_tf_geo"]) < tol,
      f"{s5['r_tf_val']:.3f}/{s5['r_tf_drug']:.3f}/{s5['r_tf_geo']:.3f} vs "
      f"{ob.at['S5', 'val_r_tf']:.3f}/{ob.at['S5', 'val_r_tf_drug']:.3f}/{ob.at['S5', 'val_r_tf_geo']:.3f}")
check("S5: rho(|A|) as recorded", abs(s5["rho_absA"] - ob.at["S5", "rho"]) < tol, f"{s5['rho_absA']:.4f} vs {ob.at['S5', 'rho']:.4f}")
for v in ["C0", "C4"]:
    check(f"{v}: val r drug / GEO / route dSting1", abs(R[v]["r_tf_drug"] - cs.at[v, "val_r_tf_drug"]) < tol
          and abs(R[v]["r_tf_geo"] - cs.at[v, "val_r_tf_geo"]) < tol
          and abs(R[v]["route_dSting1"] - cs.at[v, "route_dSting1"]) < tol,
          f"{R[v]['r_tf_drug']:.3f}/{R[v]['r_tf_geo']:.3f}/{R[v]['route_dSting1']:+.3f}")
f0 = json.loads((o / "step12f0" / "report.json").read_text())
check("drug mean-profile baseline matches step 12f0", abs(s5["baseline_drug_mean"] - f0["baselines"]["drug:B_mean"]["val"]) < 1e-6,
      f"{s5['baseline_drug_mean']:.3f}")
check("GEO mean-profile baseline matches step 12f0", abs(s5["baseline_geo_mean"] - f0["baselines"]["geo:B_mean"]["val"]) < 1e-6,
      f"{s5['baseline_geo_mean']:.3f}")

# ---------- 3. determinism + invariance on the real code path ----------
print("\n== determinism and residual invariance (S5, re-evaluated here)")
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = ROOT / "outputs" / "objective_variants" / "S5" / "checkpoint.pt"
m, meta = load_model(cfg, ck, dev)
a = evaluate_run(cfg, ck, step="test", variant="S5", device=dev, model=m, meta=meta)
b = evaluate_run(cfg, ck, step="test", variant="S5", device=dev, model=m, meta=meta)
scal = [k for k in SCHEMA_KEYS if isinstance(a[k], (int, float)) and not isinstance(a[k], bool)]
# relative to max(1, |x|): the fitted gains are ~1e2 and GPU float32 reductions are not bit-reproducible
diff = max(abs(a[k] - b[k]) / max(1.0, abs(a[k])) for k in scal if np.isfinite(a[k]))
check("two evaluations agree to 1e-5 (relative to max(1,|x|)) on every scalar metric", diff <= 1e-5,
      f"max diff {diff:.1e} over {len(scal)} metrics on {dev}")

# add a fixed random profile per source to every prediction: raw r must change, residualised r must not
src = m.design.cond.source.values
g = torch.Generator().manual_seed(0)
const = {s: (torch.randn(len(m.tf_outputs), generator=g) * 1.0).to(dev, m.dtype) for s in ("drug", "geo", "perturb")}
orig_forward = m.forward


def shifted(conditions=None, info=None):
    out = orig_forward(conditions=conditions, info=info)
    scale = out["tf_pred"].abs().mean()
    add = torch.stack([const[src[i]] if src[i] in const else torch.zeros_like(const["drug"])
                       for i in out["conditions"]], dim=1) * scale * 5
    out["tf_pred"] = out["tf_pred"] + add
    return out


m.forward = shifted
c = evaluate_run(cfg, ck, step="test", variant="S5", device=dev, model=m, meta=meta)
m.forward = orig_forward
check("raw r changes when a constant per-source profile is added", abs(c["r_tf_drug"] - a["r_tf_drug"]) > 0.01,
      f"{a['r_tf_drug']:+.3f} -> {c['r_tf_drug']:+.3f}")
check("residualised r unchanged (drug, drug-signal, GEO) to 1e-6",
      max(abs(c[k] - a[k]) for k in ["r_tf_drug_resid", "r_tf_drug_resid_signal", "r_tf_geo_resid"]) < 1e-6)

# ---------- 4. sealed holdouts and selection ----------
print("\n== sealed holdouts and selection guard")
env_code = f"import sys; sys.path.insert(0, r'{ROOT}')\n"
p1 = subprocess.run([sys.executable, "-c", env_code + "import lembas_par2.eval_sealed\n"
                     "from lembas_par2.eval_harness import select\nselect([])"], capture_output=True, text=True)
check("select() refuses to run after eval_sealed was imported", p1.returncode != 0 and "eval_sealed" in p1.stderr,
      p1.stderr.strip().splitlines()[-1] if p1.stderr else "no error raised")
p2 = subprocess.run([sys.executable, "-c", env_code + "import lembas_par2.eval_harness as h\nh._SELECTING = True\n"
                     "import lembas_par2.eval_sealed"], capture_output=True, text=True)
check("importing eval_sealed during selection raises", p2.returncode != 0 and "during selection" in p2.stderr,
      p2.stderr.strip().splitlines()[-1] if p2.stderr else "no error raised")
fake = o / "harness" / "sealed" / "12f_fake_holdout.json"
fake.parent.mkdir(parents=True, exist_ok=True)
fake.write_text(json.dumps(s5))
try:
    select([fake])
    ok = False
except RuntimeError as e:
    ok = "sealed" in str(e)
fake.unlink()
check("select() refuses to read a file under sealed/", ok)

chosen, table = select(list(paths.values()))
print(table.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
check("select() is mechanical: returns an eligible variant or None with reasons",
      (chosen is None and table.reasons.str.len().gt(0).all()) or chosen in set(table.variant[table.eligible]))
print(f"[INFO] selection on r_tf_drug_resid_signal among {list(paths)} -> {chosen} "
      "(illustration only; 12j defines its own candidate set)")

# ---------- 5. headline (information) ----------
print("\n== headline")
for v, r in R.items():
    print(f"[INFO] {v}: drug r {r['r_tf_drug']:+.3f} -> resid {r['r_tf_drug_resid']:+.3f} "
          f"(signal {r['r_tf_drug_resid_signal']:+.3f} of ceiling {r['ceiling_drug_rmax_signal']:.3f}); "
          f"GEO r {r['r_tf_geo']:+.3f} -> resid {r['r_tf_geo_resid']:+.3f}; route dSting1 {r['route_dSting1']:+.3f}; "
          f"train gains drug {r['gain_drug']:.1f} / GEO {r['gain_geo']:.1f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
