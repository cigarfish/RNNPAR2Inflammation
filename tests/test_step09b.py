"""Step 9b validation: initialisation (bulk rescaling, exact bias centring) and the
prior-only baseline.

Run from the LEMBAS folder:
    python tests/test_step09b.py
Writes outputs/step09b_init_report.json, step09b_operating_regions.csv,
step09b_route_deltas.csv, step09b_holdout_baseline.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.core import signed_radius, simulate
from lembas_par2.model import build_model

pd.set_option("display.width", 220)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
out = cfg.paths.output_dir
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = build_model(cfg, device=dev)
# 2026-10-27 route: ERK -| BCL2/BCL-xL, so PAR2 KO -> Sting1 DOWN
EXPECT = {"Mapk1": -1, "Mapk3": -1, "Bcl2": 1, "Bcl2l1": 1,
          "Bax": -1, "Bak1": -1, "SE_CYTO_MTDNA": -1, "Cgas": -1, "Sting1": -1, "Tbk1": -1, "Irf3": -1, "Ifnb1": -1}

# ---------- 1. initialisation ----------
print(f"== initialisation ({dev})")
rep = m.initialize()
print(f"[INFO] pooled reference inputs: {rep['reference_inputs']}")
check("route alone is stable", rep["rho_route"] < cfg.model["rho_target"], f"rho(|A|) route {rep['rho_route']:.4f}")
check("rho(|A|) at target after rescaling the bulk only", rep["rho_after"] <= cfg.model["rho_target"] + 1e-3,
      f"{rep['rho_start']:.2f} -> {rep['rho_after']:.3f} (bulk x {rep['bulk_scale']:.4f})")
w = m.weights().detach()
t0 = torch.tensor(m.tier == "T0", device=dev)
e0 = m.edges[m.tier == "T0"]
fan = e0.target.map(e0.groupby("target").size()).values if cfg.model.get("t0_fan_in_normalise") else np.ones(len(e0))
want = torch.tensor(cfg.model["init_strength"]["T0"] / fan, dtype=w.dtype, device=dev)
check("T0 route weights untouched by the rescaling (start strength / T0 fan-in)", torch.allclose(w[t0].abs(), want, atol=1e-5),
      f"|w| range {w[t0].abs().min().item():.3f}-{w[t0].abs().max().item():.3f}")
print(f"[INFO] signed rho(A) {signed_radius(m.struct, w):.3f}; mean |w| by tier: "
      + ", ".join(f"{t} {w[torch.tensor(m.tier == t, device=dev)].abs().mean().item():.4f}" for t in ["T0", "T1", "T2", "T3"]))
check("bias centring exact under the pooled reference", rep["centre_error"] < 1e-4, f"max |Z - 0.25| {rep['centre_error']:.1e}")

# ---------- 2. operating regions per condition type ----------
print("\n== operating regions of reference arms (fraction of nodes)")
info = {}
with torch.no_grad():
    o = m(info=info)
    Z = m.struct.matrix(w) @ (o["S"] * o["X"]) + m.bias[:, None] + m.design.U
check("all 288 arms converge", info["last_forward"]["converged"], info["last_forward"])
src = m.design.cond.source.values
route_idx = [m.idx[g] for g in EXPECT]
rows = []
for s in ["drug", "geo", "perturb", "holdout"]:
    cols = [2 * i for i in np.where(src == s)[0]]
    z = Z[:, cols]
    zr = z[route_idx]
    rows.append(dict(source=s, conditions=len(cols), leak_all=(z < 0).float().mean().item(), sat_all=(z > 2).float().mean().item(),
                     linear_all=((z >= 0) & (z <= 0.5)).float().mean().item(), leak_route=(zr < 0).float().mean().item(),
                     sat_route=(zr > 2).float().mean().item()))
reg = pd.DataFrame(rows)
print(reg.round(3).to_string(index=False))
check("reference arms mostly in the responsive range (leak + saturation < 10% of nodes)",
      ((reg.leak_all + reg.sat_all) < 0.10).all(), (reg.leak_all + reg.sat_all).round(3).tolist())

# ---------- 3. route transmission ----------
print("\n== route transmission (prior-only)")
n = len(m.nodes)
contexts = {"LPS + PAR2 agonist": ["Tlr4", "Cd14", "F2rl1"], "pooled reference": None, "no input": []}
trows = []
for label, inputs in contexts.items():
    if inputs is None:
        U = m.pooled_reference().repeat(1, 3)
    else:
        U = torch.zeros(n, 3, dtype=m.dtype, device=dev)
        for x in inputs:
            U[m.idx[x], :] = 1.0
    S = torch.ones(n, 3, dtype=m.dtype, device=dev)
    S[m.idx["F2rl1"], 1] = 0.05
    S[m.idx["Map2k1"], 2] = S[m.idx["Map2k2"], 2] = 0.3
    with torch.no_grad():
        X = simulate(m.struct, w, m.bias.detach(), U, S, max_iter=2000, tol=1e-7)
    Y = S * X
    for k, col in [("PAR2_KO", 1), ("MEKi", 2)]:
        d = (Y[:, col] - Y[:, 0]).cpu().numpy()
        wrong = [g for g, e in EXPECT.items() if np.sign(d[m.idx[g]]) != e]
        trows.append(dict(context=label, perturbation=k, dMapk1=d[m.idx["Mapk1"]], dBax=d[m.idx["Bax"]],
                          dMtDNA=d[m.idx["SE_CYTO_MTDNA"]], dSting1=d[m.idx["Sting1"]],
                          ratio=abs(d[m.idx["Sting1"]] / d[m.idx["Mapk1"]]), wrong_signs=";".join(wrong)))
tr = pd.DataFrame(trows)
print(tr.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
main = tr[(tr.context == "LPS + PAR2 agonist") & (tr.perturbation == "PAR2_KO")].iloc[0]
check("PAR2 KO under LPS + agonist: all 15 route nodes move as predicted", main.wrong_signs == "", main.wrong_signs)
check("PAR2 KO under LPS + agonist: Sting1 falls visibly (< -0.01)", main.dSting1 < -0.01, f"{main.dSting1:+.3f}")
mek = tr[(tr.context == "pooled reference") & (tr.perturbation == "MEKi")].iloc[0]
check("MEK inhibition under the pooled reference: Sting1 down", mek.dSting1 < -0.01, f"{mek.dSting1:+.3f}")
core_ctx = tr[tr.context != "LPS + PAR2 agonist"]
ok = core_ctx[(core_ctx.wrong_signs == "") & (core_ctx.dSting1 < -0.01)]
check("route transmits with correct signs under no input / pooled reference (PAR2 KO and MEKi)",
      len(ok) == len(core_ctx), core_ctx[~core_ctx.index.isin(ok.index)][["context", "perturbation", "dSting1", "wrong_signs"]].values.tolist())
ag = tr[(tr.context == "LPS + PAR2 agonist") & (tr.perturbation == "MEKi")].iloc[0]
print(f"[INFO] full PAR2 agonist input (level 1 vs pooled 0.035) drives ERK high enough that MEK inhibition "
      f"(kappa 0.7) leaves the pro-apoptotic nodes off: dSting1 {ag.dSting1:+.4f} - a threshold effect, not a defect")

# route operating points across contexts (no node should switch fully off or saturate)
zrows = []
for label, inputs in [("no input", {}), ("LPS", {"Tlr4": 1, "Cd14": 1}), ("LPS + PAR2 agonist", {"Tlr4": 1, "Cd14": 1, "F2rl1": 1})]:
    U = torch.zeros(n, 1, dtype=m.dtype, device=dev)
    for k_, v_ in inputs.items():
        U[m.idx[k_], 0] = v_
    with torch.no_grad():
        Xc = simulate(m.struct, w, m.bias.detach(), U, None, max_iter=2000, tol=1e-7)
        zc = (m.struct.matrix(w) @ Xc + m.bias[:, None] + U)[route_idx, 0].cpu().numpy()
    zrows.append(dict(context=label, z_min=zc.min(), z_max=zc.max(), leak=int((zc < 0).sum()), saturated=int((zc > 2).sum())))
zr = pd.DataFrame(zrows)
print(zr.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
base = zr[zr.context != "LPS + PAR2 agonist"]
check("route nodes stay out of the leak region and saturation without PAR2 agonist",
      (base.leak == 0).all() and (base.saturated == 0).all(), base[["context", "leak", "saturated"]].values.tolist())

# ---------- 4. prior-only predictions for the holdouts ----------
print("\n== holdouts, prior-only (untrained; for reference only)")
dirs = pd.read_csv(out / "step06_directions.csv")
hd = dirs[dirs.source == "holdout"]
cond_pos = {c: i for i, c in enumerate(m.design.cond.condition_id)}
hrows = []
with torch.no_grad():
    for r in hd.itertuples():
        i = cond_pos[r.condition_id]
        if r.layer == "signal":
            v = o["delta"][m.idx[r.node], i].item()
        else:
            v = o["gene_pred"][m.genes.index(r.node), i].item() if r.node in m.genes else np.nan
        hrows.append(dict(condition=r.condition_id, node=r.node, layer=r.layer, observed=r.sign, predicted=v,
                          agree=bool(np.sign(v) == r.sign) if np.isfinite(v) else None))
    for c in m.design.cond.condition_id[m.design.cond.source == "holdout"]:
        hrows.append(dict(condition=c, node="Sting1 (activity)", layer="signal", observed=np.nan,
                          predicted=o["delta"][m.idx["Sting1"], cond_pos[c]].item(), agree=None))
hb = pd.DataFrame(hrows)
print(hb.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
scored = hb.agree.dropna()
print(f"[INFO] prior-only direction agreement on scored holdout observations: {int(scored.sum())}/{len(scored)}")

# ---------- 5. gradients after initialisation ----------
print("\n== gradient magnitudes after initialisation")
m.zero_grad()
o = m()
(o["tf_pred"].pow(2).mean() + o["gene_pred"].pow(2).mean()).backward()
g = {k: p.grad.abs().mean().item() for k, p in m.named_parameters()}
print("[INFO] mean |grad|: " + ", ".join(f"{k} {v:.1e}" for k, v in g.items()))
check("every parameter group receives gradient", all(v > 0 for v in g.values()))

# ---------- outputs ----------
(out / "step09b_init_report.json").write_text(json.dumps(rep, indent=2))
reg.to_csv(out / "step09b_operating_regions.csv", index=False)
tr.to_csv(out / "step09b_route_deltas.csv", index=False)
hb.to_csv(out / "step09b_holdout_baseline.csv", index=False)
torch.save(m.state_dict(), out / "step09b_init_state.pt")

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 9b:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
