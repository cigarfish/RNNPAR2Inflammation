"""Step 11b validation: checks the outputs of scripts/train_stage_a.py.

Run from the LEMBAS folder after training:
    python tests/test_step11b.py               # main run  (outputs/stage_a)
    python tests/test_step11b.py --loo DRUG    # leave-one-drug-out run
With the V6 default priors it also checks that the route stays responsive
(Sting1 up >= 0.02 under simulated MEK inhibition) and the STING arm sits in 0..1.
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
from lembas_par2.train import load_checkpoint, route_neighbourhood_mask

ap = argparse.ArgumentParser()
ap.add_argument("--loo")
ap.add_argument("--run", help="run folder (default outputs/stage_a[_loo_DRUG])")
a = ap.parse_args()
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
run = Path(a.run) if a.run else cfg.paths.output_dir / ("stage_a" + (f"_loo_{a.loo}" if a.loo else ""))
print(f"== stage A run: {run}")
files = {f: (run / f).exists() for f in ["init.pt", "checkpoint.pt", "log.jsonl", "report.json"]}
if not check("run outputs present", all(files.values()), files):
    sys.exit(1)
rep = json.loads((run / "report.json").read_text())
log = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
print(f"[INFO] {rep['steps_run']} steps in {rep['seconds']:.0f} s on {rep['device']}; "
      f"train {rep['n_train']} / val {rep['n_val']} conditions; trainable edges {rep['mask']['edges']} "
      f"over {rep['mask']['nodes']} nodes")

dev = "cuda" if torch.cuda.is_available() else "cpu"
m0 = build_model(cfg, device=dev)
m1 = build_model(cfg, device=dev)
load_checkpoint(m0, run / "init.pt")
meta = load_checkpoint(m1, run / "checkpoint.pt")
w0, w1 = m0.weights().detach(), m1.weights().detach()

# ---------- 1. what was allowed to change ----------
print("\n== frozen vs trained")
mask, _ = route_neighbourhood_mask(m1, hops=meta["hops"], tiers=meta["tiers"])
edge_tr = torch.zeros(len(m1.edges), dtype=torch.bool, device=dev)
edge_tr[m1.locked_idx[mask["theta_locked"]]] = True
edge_tr[m1.free_idx[mask["theta_free"]]] = True
t3 = torch.tensor(m1.tier == "T3", device=dev)
check("STRING-only (T3) edges unchanged", torch.equal(w0[t3], w1[t3]))
check("edges outside the curated neighbourhood unchanged", torch.equal(w0[~edge_tr], w1[~edge_tr]))
check("curated (T0/T1) signs unchanged", torch.equal(torch.sign(w0[m1.locked_idx]), torch.sign(w1[m1.locked_idx])))
moved = (w0[edge_tr] != w1[edge_tr]).float().mean().item()
print(f"[INFO] trainable edges that moved: {moved:.0%}; T0 |w| {w0[torch.tensor(m1.tier == 'T0', device=dev)].abs().mean():.3f} -> "
      f"{w1[torch.tensor(m1.tier == 'T0', device=dev)].abs().mean():.3f}")

# ---------- 2. learning ----------
print("\n== learning")
tb, ta, vb, va = rep["train_before"], rep["train_after"], rep["val_before"], rep["val_after"]
check("training loss decreased by >= 10%", ta["loss"] < 0.9 * tb["loss"], f"{tb['loss']:.3f} -> {ta['loss']:.3f}")
check("validation median r(TF activity) improved", va["r_tf_median"] > vb["r_tf_median"],
      f"{vb['r_tf_median']:+.3f} -> {va['r_tf_median']:+.3f}")
print(f"[INFO] train median r(TF) {tb['r_tf_median']:+.3f} -> {ta['r_tf_median']:+.3f}; "
      f"val loss {vb['loss']:.3f} -> {va['loss']:.3f}; early stop: {bool(log[-1].get('early_stop'))}")
check("stability kept (rho(|A|) <= 0.97)", rep["rho"] <= 0.97, f"rho {rep['rho']:.3f}")

# ---------- 3. ERK axis ----------
print("\n== ERK axis (MEK / ERK / BRAF inhibitors)")
print(f"  {'drug':12s} {'split':8s} {'MPAS':>6s} {'dERK':>7s} " + " ".join(f"{t + ' meas/pred':>18s}" for t in ["Elk1", "Etv4", "Etv5"]))
for d, r in rep["erk_axis"].items():
    print(f"  {d:12s} {r['split']:8s} {r['MPAS_measured']:+6.2f} {r['dMapk1_pred']:+7.3f} " + " ".join(
        f"{(r[t + '_measured'] if r[t + '_measured'] is not None else float('nan')):+8.2f}/{r[t + '_pred']:+8.2f}"
        for t in ["Elk1", "Etv4", "Etv5"]))
e = rep["erk_axis"]
check("MEK inhibitors lower ERK activity in the model (Pimasertib, Trametinib)",
      e["Pimasertib"]["dMapk1_pred"] < 0 and e["Trametinib"]["dMapk1_pred"] < 0)
check("predicted Elk1 activity down for Pimasertib and Trametinib (measured down)",
      e["Pimasertib"]["Elk1_pred"] < 0 and e["Trametinib"]["Elk1_pred"] < 0,
      f"{e['Pimasertib']['Elk1_pred']:+.2f}, {e['Trametinib']['Elk1_pred']:+.2f}")

# ---------- 4. route after training ----------
print("\n== route under simulated MEK inhibition (pooled reference)")
rd = rep["route_delta_MEKi"]
print("  " + ", ".join(f"{k} {v:+.3f}" for k, v in rd.items()))
check("route constraint holds after training: ERK down; Bax, Bak1, mtDNA up",
      rd["Mapk1"] < 0 and rd["Bax"] > 0 and rd["Bak1"] > 0 and rd["SE_CYTO_MTDNA"] > 0)
print(f"[INFO] Sting1 under simulated MEK inhibition: {rd['Sting1']:+.3f}")
if rep.get("priors"):        # V6 default (stage_a.priors); a --priors base run skips these
    check("run used the configured stage A priors (V6)", rep["priors"] == dict(cfg.stage_a["priors"]), rep["priors"])
    check("route stays responsive: Sting1 rises >= 0.02 under simulated MEK inhibition", rd["Sting1"] >= 0.02,
          f"{rd['Sting1']:+.3f}")
    za = rep["sting_arm_preactivation"]
    check("STING arm (mtDNA, Cgas, Sting1, Tbk1) inside the responsive range (pre-activation 0..1)",
          all(0.0 <= v <= 1.0 for v in za.values()), ", ".join(f"{k} {v:+.2f}" for k, v in za.items()))
else:
    print("[INFO] baseline priors (no stage_a.priors): responsiveness checks skipped")
print(f"[INFO] holdout Sting1 delta (PAR2 KO; not trained): "
      + ", ".join(f"{k} {v:+.3f}" for k, v in rep["holdout_sting1_delta"].items()))
print(f"[INFO] learned kappa: " + ", ".join(f"{d} {rep['kappa'][d]:.2f}" for d in
                                         ["Pimasertib", "Trametinib", "Selumetinib", "Refametinib", "Ruxolitinib", "BGT226"]))

# ---------- 5. leave-one-out ----------
if a.loo:
    print(f"\n== held-out drug: {a.loo}")
    r = e.get(a.loo)
    if r is not None and r["Elk1_measured"] is not None and abs(r["Elk1_measured"]) > 1:
        check(f"{a.loo}: predicted Elk1 direction matches measured (never trained on it)",
              np.sign(r["Elk1_pred"]) == np.sign(r["Elk1_measured"]), f"measured {r['Elk1_measured']:+.2f}, predicted {r['Elk1_pred']:+.2f}")
    print(f"[INFO] held-out r(TF) {va['r_tf_median']:+.3f}, loss {va['loss']:.3f}")

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 11b:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
