"""Training stage B: whole network, all trained conditions, continuing from stage A.

Differences from stage A:
  * every parameter is trainable (T3 STRING backbone included)
  * conditions = drug + GEO + perturbation papers (holdouts keep weight 0)
  * starts from outputs/<stage_b.init_from>/checkpoint.pt
  * priors still anchor to the step-9b initialised model, so T0/T1 are pulled towards
    prior strength and T2/T3 towards sparsity - not towards the stage A result
  * the responsive-range (activation) penalty now covers all nodes
  * route responsiveness and the step-11b drift nodes are logged at every eval

Run from the LEMBAS folder (GPU recommended):
    python scripts/train_stage_b.py                     # main run (fold 0 = validation)
    python scripts/train_stage_b.py --loco Viral        # leave one stimulus class out
    python scripts/train_stage_b.py --steps 30          # quick smoke run
    python scripts/train_stage_b.py --mode unanchored   # comparison model for step 13
Options: --device, --steps, --lr, --batch, --init-from, --out

Writes outputs/stage_b[...]/{checkpoint.pt, log.jsonl, report.json}; validate with
    python tests/test_step12.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.core import mml_inverse, perron_radius, simulate
from lembas_par2.loss import Priors, Targets, balance_data_terms
from lembas_par2.model import build_model
from lembas_par2.train import (Trainer, all_trainable, apply_prior_spec, evaluate, load_checkpoint,
                               route_neighbourhood_mask, save_checkpoint)

ROUTE_REPORT = ["Mapk1", "Mapk3", "Gsk3b", "Bcl2l11", "Bad", "Mcl1", "Bax", "Bak1",
                "SE_CYTO_MTDNA", "Cgas", "Sting1", "Irf3"]


def summarise(ev):
    keep = ["loss", "tf", "gene", "direction", "r_tf_median", "r_gene_median", "direction_agree"]
    return {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
            for k, v in ev.items() if k in keep}


def preactivation(m):
    with torch.no_grad():
        X = simulate(m.struct, m.weights(), m.bias, m.pooled_reference(), None, max_iter=3000, tol=1e-7)
        return mml_inverse(X[:, 0].clamp(max=0.999))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--eval-every", dest="eval_every", type=int)
    ap.add_argument("--init-from", dest="init_from")
    ap.add_argument("--loco", help="GEO stimulus class to hold out entirely (Bacterial / Viral / Chemical)")
    ap.add_argument("--mode", choices=["anchored", "unanchored"], default="anchored")
    ap.add_argument("--allow-any-init", action="store_true",
                    help="start from a checkpoint whose priors differ from the configured ones")
    ap.add_argument("--out")
    a = ap.parse_args()

    cfg = load_config()
    sb = dict(cfg.stage_b)
    for k in ["steps", "lr", "batch", "init_from", "eval_every"]:
        if getattr(a, k) is not None:
            sb[k] = getattr(a, k)
    tag = "stage_b" + (f"_loco_{a.loco}" if a.loco else "") + ("_unanchored" if a.mode == "unanchored" else "")
    out = Path(a.out) if a.out else cfg.paths.output_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(sb["seed"])
    t_start = time.time()
    print(f"[stage B] device {a.device}, mode {a.mode}, output {out}")

    # 1. prior state: build and initialise exactly as in step 9b, so the priors anchor there
    m = build_model(cfg, device=a.device, seed=sb["seed"])
    init_rep = m.initialize()
    T = Targets(cfg, m)
    lam = balance_data_terms(m, T)
    P = Priors(cfg, m, mode=a.mode)
    spec = dict(sb["priors"])
    which = sb["activation_nodes"]
    nodes = {"route": lambda: P.route_nodes_mask(),
             "trainable": lambda: route_neighbourhood_mask(m, hops=1, tiers=["T0", "T1", "T2"])[0]["bias"],
             "all": lambda: torch.ones(len(m.nodes), dtype=torch.bool, device=a.device)}[which]()
    apply_prior_spec(P, spec, nodes, cfg.stage_a_activation_range, reduce=sb["activation_reduce"])
    print(f"[stage B] priors {spec}; responsive-range penalty on {int(nodes.sum())} {which} nodes "
          f"({sb['activation_reduce']}, range {cfg.stage_a_activation_range})"
          + ("; UNANCHORED: route and responsive-range terms off" if a.mode == "unanchored" else ""))
    b_prior = m.bias.detach().clone()

    # 2. load the stage A result as the starting point
    src = cfg.paths.output_dir / sb["init_from"] / "checkpoint.pt"
    if not src.exists():
        sys.exit(f"missing starting checkpoint {src}; run scripts/train_stage_a.py first")
    meta_a = load_checkpoint(m, src)
    print(f"[stage B] started from {src} (stage {meta_a.get('stage')}, priors {meta_a.get('priors')})")
    # guard against the step-11b failure mode: a stage A checkpoint trained without
    # bias control has drifted biases and a route that no longer transmits
    init_priors = meta_a.get("priors") or meta_a.get("spec") or {}
    if float(init_priors.get("lambda_bias", 0)) < 1.0 and not a.allow_any_init:
        sys.exit(f"[stage B] refusing to start: {src.parent.name} was trained with priors {init_priors or 'none'},\n"
                 f"  i.e. without bias control, so it has the step-11b drifted biases and a dead route.\n"
                 f"  Re-run 'python scripts/train_stage_a.py' (now V6 by default), or pass --allow-any-init.")

    # 3. conditions
    cond = m.design.cond
    is_src = cond.source.isin(sb["sources"]).values
    held = []
    if a.loco:
        if a.loco not in set(cond.cls[cond.source == "geo"]):
            sys.exit(f"unknown GEO stimulus class {a.loco}")
        held = [i for i in np.where(is_src)[0] if cond.source.iloc[i] == "geo" and cond.cls.iloc[i] == a.loco]
    val = [i for i in np.where(is_src)[0]
           if i not in held and cond.source.iloc[i] in ("drug", "geo") and cond.fold.iloc[i] == sb["val_fold"]]
    train = [i for i in np.where(is_src)[0] if i not in val and i not in held]
    by = lambda idx: {s: int(sum(cond.source.iloc[i] == s for i in idx)) for s in sb["sources"]}
    msg = f"[stage B] train {len(train)} {by(train)}, validation {len(val)} {by(val)}"
    if a.loco:
        msg += f", held out {len(held)} {a.loco} GEO conditions"
    print(msg)
    mask = all_trainable(m)

    # re-balance the data terms at the stage B starting point (see config comment)
    lam_b = dict(lam)
    if sb["rebalance_at_start"]:
        lam_b = dict(balance_data_terms(m, T, conditions=train))
        print(f"[stage B] lambdas re-balanced on the training set: "
              + ", ".join(f"{k} {v:.4g}" for k, v in lam_b.items()) + f"  (step-9b values: "
              + ", ".join(f"{k} {v:.4g}" for k, v in lam.items()) + ")")

    # 4. monitor: route responsiveness and the step-11b drift nodes
    def monitor(model):
        with torch.no_grad():
            rd = P.route_delta(model)
            z = preactivation(model)
        return dict(route_dSting1=rd[model.idx["Sting1"]].item(), route_dMapk1=rd[model.idx["Mapk1"]].item(),
                    **{f"z_{n}": z[model.idx[n]].item() for n in sb["watch_nodes"]})

    ev_tr0, ev_va0 = evaluate(m, T, train), evaluate(m, T, val)
    mon0 = monitor(m)
    print(f"[stage B] start: train loss {ev_tr0['loss']:.3f}, val loss {ev_va0['loss']:.3f}, "
          f"val r_tf {ev_va0['r_tf_median']:+.3f}, route dSting1 {mon0['route_dSting1']:+.3f}")

    tr = Trainer(m, T, P, mask=mask, lr=sb["lr"], warmup=sb["warmup"], seed=sb["seed"], log_path=out / "log.jsonl",
                 project=sb["project_stability"])
    tr.fit(train, val_pos=val, steps=sb["steps"], batch=sb["batch"], eval_every=sb["eval_every"],
           patience=sb["patience"], seed=sb["seed"], monitor=monitor)
    save_checkpoint(m, out / "checkpoint.pt", stage="B", mode=a.mode, init_from=str(src), loco=a.loco,
                    train=train, val=val, held=held, priors=spec)

    # 5. report
    ev_tr1, ev_va1 = evaluate(m, T, train), evaluate(m, T, val)
    mon1 = monitor(m)
    pos_of = lambda s: [i for i in train + val if cond.source.iloc[i] == s]
    per_source = {s: summarise(evaluate(m, T, pos_of(s))) for s in sb["sources"] if pos_of(s)}
    val_geo = [i for i in val if cond.source.iloc[i] == "geo"]
    val_drug = [i for i in val if cond.source.iloc[i] == "drug"]
    z = preactivation(m)
    w1 = m.weights().detach()
    with torch.no_grad():
        rd = P.route_delta(m)
        route = {g: rd[m.idx[g]].item() for g in ROUTE_REPORT}
        hold = [i for i, s in enumerate(cond.source) if s == "holdout"]
        oh = m(conditions=hold)
        holdout_sting = {cond.condition_id.iloc[i]: oh["delta"][m.idx["Sting1"], k].item() for k, i in enumerate(hold)}
        rho = perron_radius(m.struct, w1, max_iter=3000, tol=1e-6).item()
        tiers = {t: float(w1[torch.tensor(m.tier == t, device=a.device)].abs().mean()) for t in ["T0", "T1", "T2", "T3"]}
        nz_t3 = float((w1[torch.tensor(m.tier == "T3", device=a.device)].abs() > 0.01).float().mean())
    report = dict(stage="B", mode=a.mode, loco=a.loco, device=a.device, priors=spec, init_from=str(src),
                  seconds=time.time() - t_start, steps_run=len(tr.history), lambdas=lam_b, lambdas_init=lam,
                  rebalanced=bool(sb["rebalance_at_start"]), projected=bool(sb["project_stability"]), init=init_rep,
                  n_train=len(train), n_val=len(val), n_held=len(held),
                  train_before=summarise(ev_tr0), train_after=summarise(ev_tr1),
                  val_before=summarise(ev_va0), val_after=summarise(ev_va1),
                  val_geo=summarise(evaluate(m, T, val_geo)) if val_geo else None,
                  val_drug=summarise(evaluate(m, T, val_drug)) if val_drug else None,
                  held_out=summarise(evaluate(m, T, held)) if held else None,
                  per_source=per_source, route_delta_MEKi=route, monitor_before=mon0, monitor_after=mon1,
                  holdout_sting1_delta=holdout_sting, rho=rho, mean_abs_w=tiers, t3_nonzero_frac=nz_t3,
                  sting_arm_preactivation={n: z[m.idx[n]].item() for n in sb["watch_nodes"]},
                  bias_drift_watch={n: (m.bias[m.idx[n]] - b_prior[m.idx[n]]).item() for n in sb["watch_nodes"]},
                  leak_frac=float((z < 0).float().mean()), sat_frac=float((z > 2).float().mean()))
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))

    print(f"[stage B] done in {report['seconds']:.0f} s, {report['steps_run']} steps")
    print(f"  train loss {ev_tr0['loss']:.3f} -> {ev_tr1['loss']:.3f}, val loss {ev_va0['loss']:.3f} -> {ev_va1['loss']:.3f}")
    print(f"  val median r(TF) {ev_va0['r_tf_median']:+.3f} -> {ev_va1['r_tf_median']:+.3f}"
          + (f"  (GEO {report['val_geo']['r_tf_median']:+.3f}" if report["val_geo"] else "")
          + (f", drug {report['val_drug']['r_tf_median']:+.3f})" if report["val_drug"] else ")"))
    if report["held_out"]:
        print(f"  held-out {a.loco}: loss {report['held_out']['loss']:.3f}, r_tf {report['held_out']['r_tf_median']:+.3f}")
    print(f"  route (simulated MEK inhibition): " + ", ".join(f"{k} {v:+.3f}" for k, v in route.items()))
    print(f"  STING-arm pre-activation: " + ", ".join(f"{k} {v:+.2f}" for k, v in report['sting_arm_preactivation'].items()))
    print(f"  rho {rho:.3f}; mean |w| " + ", ".join(f"{k} {v:.4f}" for k, v in tiers.items())
          + f"; T3 |w|>0.01 {nz_t3:.1%}")
    print(f"  holdout Sting1 delta: " + ", ".join(f"{k} {v:+.3f}" for k, v in holdout_sting.items()))
    print(f"  validate with: python tests/test_step12.py" + (f" --run {out.name}" if out.name != "stage_b" else ""))


if __name__ == "__main__":
    main()
