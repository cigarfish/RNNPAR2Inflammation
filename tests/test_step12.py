"""Step 12 validation: checks the outputs of scripts/train_stage_b.py.

Run from the LEMBAS folder after training:
    python tests/test_step12.py                  # outputs/stage_b
    python tests/test_step12.py --run stage_b_loco_Viral
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.model import build_model
from lembas_par2.train import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="stage_b")
a = ap.parse_args()
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
sb = cfg.stage_b
run = cfg.paths.output_dir / a.run
print(f"== stage B run: {run}")
files = {f: (run / f).exists() for f in ["checkpoint.pt", "log.jsonl", "report.json"]}
if not check("run outputs present", all(files.values()), files):
    sys.exit(1)
rep = json.loads((run / "report.json").read_text())
log = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
evals = [r for r in log if "val_loss" in r]
print(f"[INFO] mode {rep['mode']}, {rep['steps_run']} steps in {rep['seconds']:.0f} s on {rep['device']}; "
      f"train {rep['n_train']} / val {rep['n_val']}"
      + (f" / held out {rep['n_held']} ({rep['loco']})" if rep["loco"] else "")
      + f"; early stop: {bool(log[-1].get('early_stop'))}")

# ---------- 1. setup ----------
print("\n== setup")
check("continued from the stage A checkpoint", rep["init_from"].endswith("checkpoint.pt")
      and sb["init_from"] in rep["init_from"], rep["init_from"])
check("configured stage B priors used", rep["priors"] == dict(sb["priors"]), rep["priors"])
check("all three condition sources trained", set(rep["per_source"]) == set(sb["sources"]),
      {k: rep["per_source"][k]["r_tf_median"] for k in rep["per_source"]})

dev = "cuda" if torch.cuda.is_available() else "cpu"
m_init = build_model(cfg, device=dev)
m_init.initialize()
m = build_model(cfg, device=dev)
meta = load_checkpoint(m, run / "checkpoint.pt")
w0, w1 = m_init.weights().detach(), m.weights().detach()
t3 = torch.tensor(m.tier == "T3", device=dev)
check("whole network was trainable: T3 weights moved", not torch.equal(w0[t3], w1[t3]),
      f"mean |w| T3 {rep['mean_abs_w']['T3']:.5f}, {rep['t3_nonzero_frac']:.1%} above 0.01")
check("curated (T0/T1) signs unchanged", torch.equal(torch.sign(w0[m.locked_idx]), torch.sign(w1[m.locked_idx])))
check("T0 route weights still near their prior strength (anchored)",
      abs(rep["mean_abs_w"]["T0"] - float(w0[torch.tensor(m.tier == 'T0', device=dev)].abs().mean())) < 0.25,
      f"prior {float(w0[torch.tensor(m.tier == 'T0', device=dev)].abs().mean()):.3f} -> {rep['mean_abs_w']['T0']:.3f}")

# ---------- 2. fit ----------
print("\n== fit")
tb, ta, vb, va = rep["train_before"], rep["train_after"], rep["val_before"], rep["val_after"]
check("training loss decreased", ta["loss"] < tb["loss"], f"{tb['loss']:.3f} -> {ta['loss']:.3f}")
check("validation loss decreased vs the stage A starting point", va["loss"] < vb["loss"],
      f"{vb['loss']:.3f} -> {va['loss']:.3f}")
check("validation median r(TF) improved vs stage A", va["r_tf_median"] > vb["r_tf_median"],
      f"{vb['r_tf_median']:+.3f} -> {va['r_tf_median']:+.3f}")
for k in ["val_geo", "val_drug"]:
    if rep.get(k) and rep[k]["r_tf_median"] is not None:
        check(f"{k}: positive median r(TF)", rep[k]["r_tf_median"] > 0,
              f"r_tf {rep[k]['r_tf_median']:+.3f}, loss {rep[k]['loss']:.3f}")
fmt = lambda v, s="{:+.3f}": "n/a" if v is None else s.format(v)
print("[INFO] per source after training: " + ", ".join(
    f"{s} loss {fmt(v['loss'], '{:.3f}')} r_tf {fmt(v['r_tf_median'])} "
    f"direction agreement {fmt(v['direction_agree'], '{:.2f}')}" for s, v in rep["per_source"].items())
    + "   (perturbation papers have direction constraints only, no measured TF matrix)")
if rep.get("held_out"):
    h = rep["held_out"]
    print(f"[INFO] held-out {rep['loco']} GEO: loss {fmt(h['loss'], '{:.3f}')}, r_tf {fmt(h['r_tf_median'])}")
check("stability kept (rho(|A|) <= 0.97)", rep["rho"] <= 0.97, f"rho {rep['rho']:.3f}")
check("nodes stayed in the responsive range (leak + saturation < 10%)",
      rep["leak_frac"] + rep["sat_frac"] < 0.10, f"leak {rep['leak_frac']:.3f}, saturated {rep['sat_frac']:.3f}")

# ---------- 3. the step-11b failure mode did not return ----------
print("\n== drift watch (step-11b failure mode)")
print("[INFO] bias drift vs the initialised model: " + ", ".join(f"{k} {v:+.3f}" for k, v in rep["bias_drift_watch"].items()))
print("[INFO] STING-arm pre-activation: " + ", ".join(f"{k} {v:+.2f}" for k, v in rep["sting_arm_preactivation"].items()))
check("STING arm inside the responsive range (pre-activation 0..1)",
      all(0.0 <= v <= 1.0 for v in rep["sting_arm_preactivation"].values()), rep["sting_arm_preactivation"])
rd = rep["route_delta_MEKi"]
print("[INFO] route under simulated MEK inhibition: " + ", ".join(f"{k} {v:+.3f}" for k, v in rd.items()))
if rep["mode"] == "anchored":
    check("route constraint holds: ERK down; Bax, Bak1, mtDNA up",
          rd["Mapk1"] < 0 and rd["Bax"] > 0 and rd["Bak1"] > 0 and rd["SE_CYTO_MTDNA"] > 0)
    check("route stays responsive: Sting1 rises >= 0.02 under simulated MEK inhibition", rd["Sting1"] >= 0.02,
          f"dSting1 {rd['Sting1']:+.3f} (was {rep['monitor_before']['route_dSting1']:+.3f} at stage A)")
else:
    print(f"[INFO] unanchored mode: route term off; dSting1 {rd['Sting1']:+.3f} comes from the data alone")
if evals:
    traj = [(r["step"] + 1, r.get("route_dSting1"), r.get("val_r_tf")) for r in evals]
    keep = traj[:: max(1, len(traj) // 8)]
    print("[INFO] trajectory (step, route dSting1, val r_tf): "
          + ", ".join(f"({s}, {d:+.3f}, {r:+.3f})" for s, d, r in keep if d is not None and r is not None))

# ---------- 4. holdouts: reported only ----------
print("\n== holdout PAR2-KO Sting1 delta (never trained; step 14 scores these)")
print("[INFO] " + ", ".join(f"{k} {v:+.3f}" for k, v in rep["holdout_sting1_delta"].items()))

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 12:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
