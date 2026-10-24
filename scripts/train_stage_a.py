"""Training stage A: route + curated 1-hop neighbourhood on the drug conditions.

Priors: config `loss` plus `stage_a.priors` (variant V6 from step 11b+: bias penalty 1.0,
anchor 0.1 on T1 edges leaving drug / KO targets, responsive-range penalty on the
trainable nodes). --priors base runs the original baseline (V0) instead.

Run from the LEMBAS folder (GPU recommended):
    python scripts/train_stage_a.py                      # main run (drug fold 0 = validation)
    python scripts/train_stage_a.py --loo Pimasertib     # leave one drug out entirely
    python scripts/train_stage_a.py --steps 20           # quick smoke run
Options: --device cpu|cuda, --steps, --lr, --batch, --out DIR, --priors stage_a|base

Writes to outputs/stage_a/ (or outputs/stage_a_loo_<DRUG>/):
    init.pt, checkpoint.pt   initialised and trained model states
    log.jsonl                one line per step
    report.json              metrics, ERK-axis fit, route deltas, kappa, holdout Sting1
Then validate with:  python tests/test_step11b.py [--loo DRUG]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.core import mml_inverse, perron_radius, simulate
from lembas_par2.loss import Priors, Targets, balance_data_terms
from lembas_par2.model import build_model
from lembas_par2.train import (Trainer, apply_prior_spec, evaluate, mask_summary, route_neighbourhood_mask,
                               save_checkpoint)

STING_ARM = ["SE_CYTO_MTDNA", "Cgas", "Sting1", "Tbk1"]


def summarise(ev):
    return {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in ev.items() if k in ["loss", "tf", "gene", "direction", "r_tf_median", "r_gene_median", "direction_agree"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--loo", help="drug name to leave out of training entirely")
    ap.add_argument("--out")
    ap.add_argument("--priors", choices=["stage_a", "base"], default="stage_a")
    a = ap.parse_args()

    cfg = load_config()
    sa = dict(cfg.stage_a)
    for k in ["steps", "lr", "batch"]:
        if getattr(a, k) is not None:
            sa[k] = getattr(a, k)
    out = Path(a.out) if a.out else cfg.paths.output_dir / ("stage_a" + (f"_loo_{a.loo}" if a.loo else ""))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(sa["seed"])
    t_start = time.time()

    print(f"[stage A] device {a.device}, output {out}")
    m = build_model(cfg, device=a.device, seed=sa["seed"])
    init_rep = m.initialize()
    T = Targets(cfg, m)
    lam = balance_data_terms(m, T)
    save_checkpoint(m, out / "init.pt", stage="init")

    cond = m.design.cond
    is_src = cond.source.isin(sa["sources"]).values
    if a.loo:
        if a.loo not in set(cond.name):
            sys.exit(f"unknown drug {a.loo}")
        held = np.where(cond.name.values == a.loo)[0].tolist()
        train = [i for i in np.where(is_src)[0] if i not in held]
        val = held
    else:
        val = [i for i in np.where(is_src)[0] if cond.fold.iloc[i] == sa["val_fold"]]
        train = [i for i in np.where(is_src)[0] if i not in val]
    mask, minfo = route_neighbourhood_mask(m, hops=sa["hops"], tiers=sa["tiers"])
    print(f"[stage A] train {len(train)} conditions, validation {len(val)}; trainable: {mask_summary(m, mask)}")
    prior_spec = dict(sa["priors"]) if a.priors == "stage_a" else {}
    P = Priors(cfg, m, mode="anchored")
    apply_prior_spec(P, prior_spec, mask["bias"], cfg.stage_a_activation_range)
    print(f"[stage A] priors: {a.priors} {prior_spec}")

    ev_tr0, ev_va0 = evaluate(m, T, train), evaluate(m, T, val)
    tr = Trainer(m, T, P, mask=mask, lr=sa["lr"], warmup=sa["warmup"], seed=sa["seed"], log_path=out / "log.jsonl")
    tr.fit(train, val_pos=val, steps=sa["steps"], batch=sa["batch"], eval_every=sa["eval_every"],
           patience=sa["patience"], seed=sa["seed"])
    save_checkpoint(m, out / "checkpoint.pt", stage="A", loo=a.loo, train=train, val=val, mask_info=minfo,
                    tiers=list(sa["tiers"]), hops=sa["hops"], priors=prior_spec)
    ev_tr1, ev_va1 = evaluate(m, T, train), evaluate(m, T, val)

    # ERK-axis fit for MEK/ERK/BRAF drugs (measured vs predicted TF activity)
    progs = __import__("pandas").read_csv(cfg.paths.output_dir / "step06_programs.csv", index_col=0)
    erk = {}
    with torch.no_grad():
        for d in sa["mek_axis_drugs"]:
            i = int(np.where(cond.name.values == d)[0][0])
            o = m(conditions=[i])
            row = {"split": "val" if i in val else ("train" if i in train else "excluded"),
                   "MPAS_measured": float(progs.loc[cond.condition_id.iloc[i], "MPAS"]),
                   "dMapk1_pred": o["delta"][m.idx["Mapk1"], 0].item()}
            for tf in ["Elk1", "Etv4", "Etv5"]:
                j = m.tf_outputs.index(tf)
                row[f"{tf}_measured"] = T.tf[j, i].item() if T.tf_mask[j, i] > 0 else None
                row[f"{tf}_pred"] = o["tf_pred"][j, 0].item()
            erk[d] = row
        rd = P.route_delta(m)
        route = {g: rd[m.idx[g]].item() for g in ["Mapk1", "Mapk3", "Gsk3b", "Bcl2l11", "Bad", "Mcl1", "Bax", "Bak1",
                                                    "SE_CYTO_MTDNA", "Cgas", "Sting1", "Irf3"]}
        hold = [i for i, s in enumerate(cond.source) if s == "holdout"]
        oh = m(conditions=hold)
        holdout_sting = {cond.condition_id.iloc[i]: oh["delta"][m.idx["Sting1"], k].item() for k, i in enumerate(hold)}
        rho = perron_radius(m.struct, m.weights(), max_iter=3000, tol=1e-6).item()
        X = simulate(m.struct, m.weights(), m.bias, m.pooled_reference(), None, max_iter=3000, tol=1e-7)
        z = mml_inverse(X[:, 0].clamp(max=0.999))
        sting_arm_z = {g: z[m.idx[g]].item() for g in STING_ARM}
    kap = {d: m.design.kappa()[m.design.drugs.index(d)].item() for d in m.design.drugs}

    report = dict(stage="A", loo=a.loo, device=a.device, priors=prior_spec, seconds=time.time() - t_start, steps_run=len(tr.history),
                  init=init_rep, lambdas=lam, mask=minfo, n_train=len(train), n_val=len(val),
                  train_before=summarise(ev_tr0), train_after=summarise(ev_tr1),
                  val_before=summarise(ev_va0), val_after=summarise(ev_va1),
                  erk_axis=erk, route_delta_MEKi=route, holdout_sting1_delta=holdout_sting, rho=rho, kappa=kap,
                  sting_arm_preactivation=sting_arm_z)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"[stage A] done in {report['seconds']:.0f} s, {len(tr.history)} steps")
    print(f"  train loss {ev_tr0['loss']:.3f} -> {ev_tr1['loss']:.3f}, val loss {ev_va0['loss']:.3f} -> {ev_va1['loss']:.3f}")
    print(f"  val median r(TF) {ev_va0['r_tf_median']:+.3f} -> {ev_va1['r_tf_median']:+.3f}")
    print(f"  route (simulated MEK inhibition): " + ", ".join(f"{k} {v:+.3f}" for k, v in route.items()))
    print(f"  STING-arm pre-activation: " + ", ".join(f"{k} {v:+.2f}" for k, v in sting_arm_z.items()))
    print(f"  holdout Sting1 delta: {holdout_sting}")
    print(f"  validate with: python tests/test_step11b.py" + (f" --loo {a.loo}" if a.loo else ""))


if __name__ == "__main__":
    main()
