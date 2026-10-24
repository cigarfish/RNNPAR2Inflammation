"""Fix signal decay: compare network density and per-node input normalisation
(config: transmission_variants).

The decay at this point is not the activation slope - bias centring fixed that and the
slope is 1.000 everywhere, including at the TF outputs - but the edge weights, which
rho(|A|) < 1 caps at ~0.007 given 254 k edges. Each variant changes how that budget is
distributed, then trains a short stage B.

Reported per variant: the admissible weight after initialisation, |delta| at the TF
output nodes (the quantity that was 1.6e-04 against a measured |t| of ~0.9), validation
fit, route responsiveness and stability.

Run from the LEMBAS folder (GPU recommended; ~1-2 min per variant):
    python scripts/run_transmission_variants.py                  # all variants
    python scripts/run_transmission_variants.py --steps 0        # initialisation only (fast)
    python scripts/run_transmission_variants.py --variants N0 N4
Writes outputs/transmission_variants/{summary.csv, <ID>/report.json}; then:
    python tests/check_transmission_variants.py
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
from lembas_par2.core import mml_grad, mml_inverse, perron_radius, simulate, steady_state
from lembas_par2.loss import Priors, Targets, balance_data_terms
from lembas_par2.model import build_model
from lembas_par2.train import Trainer, all_trainable, apply_prior_spec, evaluate, load_checkpoint, save_checkpoint


def transmission_stats(m, T, train):
    """|delta| at the TF outputs and the per-hop gain along the route."""
    with torch.no_grad():
        o = m(conditions=train)
        d = o["delta"][m.tf_idx].abs().flatten()
        s = steady_state(m.struct, m.weights(), m.bias, m.pooled_reference(), max_iter=3000, tol=1e-6)
        slope = mml_grad(s.Z[:, 0])
        w = m.weights()
        E = m.edges
        key = {(a, b): i for i, (a, b) in enumerate(zip(E.source, E.target))}
        hops = [("Map2k1", "Mapk1"), ("Mapk1", "Bcl2l11"), ("Bcl2l11", "Bax"),
                ("Bax", "SE_CYTO_MTDNA"), ("SE_CYTO_MTDNA", "Cgas"), ("Cgas", "Sting1")]
        gains = {f"{a}->{b}": (w[key[(a, b)]] * slope[m.idx[b]]).item() for a, b in hops if (a, b) in key}
    return dict(tf_delta_median=float(d.median()), tf_delta_p90=float(d.kthvalue(int(0.9 * d.numel())).values),
                slope_median_tf=float(slope[m.tf_idx].median()), route_hop_gains=gains)


def run_variant(cfg, vid, spec, dev, steps, out_root, init_src, watch=None):
    out = out_root / vid
    out.mkdir(parents=True, exist_ok=True)
    watch = watch or cfg.transmission_test["watch"]
    sb = dict(cfg.stage_b)
    sb["deactivate_tiers"] = spec.get("deactivate_tiers", sb["deactivate_tiers"])
    cfg2 = load_config()                      # fresh copy so model config edits do not leak
    cfg2["model"]["fan_in_normalise"] = spec.get("fan_in_normalise", cfg.model.get("fan_in_normalise", "none"))
    tf_mode = spec.get("tf_mode", cfg.loss.get("tf_mode", "absolute"))
    torch.manual_seed(sb["seed"])
    t0 = time.time()
    print(f"\n=== {vid} {spec or 'baseline'}")

    m = build_model(cfg2, device=dev, seed=sb["seed"])
    active = m.deactivate_tiers(sb["deactivate_tiers"]) if sb["deactivate_tiers"] else \
        torch.ones(len(m.edges), dtype=torch.bool, device=dev)
    init_rep = m.initialize()
    if init_src is not None:
        load_checkpoint(m, init_src)
    T = Targets(cfg2, m).configure_tf(tf_mode, spec.get("tf_gain_nonneg", cfg.loss.get("tf_gain_nonneg", True)))
    balance_data_terms(m, T)
    P = Priors(cfg2, m, mode="anchored")
    apply_prior_spec(P, dict(sb["priors"]), P.route_nodes_mask(), cfg2.stage_a_activation_range,
                     reduce=sb["activation_reduce"])

    cond = m.design.cond
    is_src = cond.source.isin(sb["sources"]).values
    val = [i for i in np.where(is_src)[0]
           if cond.source.iloc[i] in ("drug", "geo") and cond.fold.iloc[i] == sb["val_fold"]]
    train = [i for i in np.where(is_src)[0] if i not in val]
    lam = dict(balance_data_terms(m, T, conditions=train))

    w = m.weights().detach()
    tiers = {t: float(w[torch.tensor(m.tier == t, device=dev)].abs().median()) for t in ["T0", "T1", "T2", "T3"]}
    st0 = transmission_stats(m, T, train)
    ev0 = evaluate(m, T, val)
    rep = dict(variant=vid, spec=json.dumps(spec), tf_mode=tf_mode, steps=steps, bulk_scale=init_rep["bulk_scale"],
               rho_init=init_rep["rho_after"], centre_error=init_rep["centre_error"],
               n_active_edges=int(active.sum()), median_w_T1=tiers["T1"], median_w_T2=tiers["T2"],
               median_w_T3=tiers["T3"], tf_delta_median_init=st0["tf_delta_median"],
               tf_delta_p90_init=st0["tf_delta_p90"], slope_median_tf=st0["slope_median_tf"],
               val_r_tf_init=ev0["r_tf_median"], val_loss_init=ev0["loss"], route_hop_gains=st0["route_hop_gains"])
    print(f"  init: bulk scale {init_rep['bulk_scale']:.3f}, rho {init_rep['rho_after']:.3f}, "
          f"active edges {int(active.sum())}, median |w| T1 {tiers['T1']:.4f} T2 {tiers['T2']:.4f} "
          f"T3 {tiers['T3']:.5f}\n        |delta| at TF nodes: median {st0['tf_delta_median']:.2e}, "
          f"90th {st0['tf_delta_p90']:.2e}  (target |t| ~0.9);  val r_tf {ev0['r_tf_median']:+.3f}", flush=True)

    if steps > 0:
        mask = all_trainable(m)
        if sb["deactivate_tiers"]:
            mask["theta_free"] = active[m.free_idx]
            mask["theta_locked"] = active[m.locked_idx]
        tr = Trainer(m, T, P, mask=mask, lr=sb["lr"], warmup=min(sb["warmup"], max(1, steps // 10)),
                     seed=sb["seed"], log_path=out / "log.jsonl", project=sb["project_stability"])
        tr.fit(train, val_pos=val, steps=steps, batch=sb["batch"], eval_every=max(10, steps // 6),
               patience=None, seed=sb["seed"], verbose=False)
        ev1 = evaluate(m, T, val)
        vg = [i for i in val if cond.source.iloc[i] == "geo"]
        vd = [i for i in val if cond.source.iloc[i] == "drug"]
        st1 = transmission_stats(m, T, train)
        w1 = m.weights().detach()
        with torch.no_grad():
            rd = P.route_delta(m)
            rho = perron_radius(m.struct, w1, max_iter=3000, tol=1e-6).item()
            z = mml_inverse(simulate(m.struct, w1, m.bias, m.pooled_reference(), None,
                                     max_iter=3000, tol=1e-7)[:, 0].clamp(max=0.999))
            hold = [i for i, s_ in enumerate(cond.source) if s_ == "holdout"]
            oh = m(conditions=hold)
        rep.update(val_r_tf=ev1["r_tf_median"], val_loss=ev1["loss"],
                   val_r_tf_geo=evaluate(m, T, vg)["r_tf_median"] if vg else None,
                   val_r_tf_drug=evaluate(m, T, vd)["r_tf_median"] if vd else None,
                   tf_delta_median=st1["tf_delta_median"], tf_delta_p90=st1["tf_delta_p90"],
                   median_w_T1_after=float(w1[torch.tensor(m.tier == "T1", device=dev)].abs().median()),
                   route_dSting1=rd[m.idx["Sting1"]].item(), route_dMapk1=rd[m.idx["Mapk1"]].item(),
                   rho=rho, tf_gain=T.last_tf_gain, gain_frac_negative=T.last_gain_frac_negative,
                   **{f"z_{k}": z[m.idx[k]].item() for k in watch},
                   **{f"holdout_{cond.condition_id.iloc[i]}_dSting1": oh["delta"][m.idx["Sting1"], k].item()
                      for k, i in enumerate(hold)})
        save_checkpoint(m, out / "checkpoint.pt", stage="B_transmission", variant=vid, spec=spec)
        print(f"  after {steps} steps: |delta| TF median {st1['tf_delta_median']:.2e}; "
              f"val r_tf {ev0['r_tf_median']:+.3f} -> {ev1['r_tf_median']:+.3f} "
              f"(drug {rep['val_r_tf_drug']:+.3f}, GEO {rep['val_r_tf_geo']:+.3f}); "
              f"T1 |w| {tiers['T1']:.4f} -> {rep['median_w_T1_after']:.4f}; "
              f"route dSting1 {rep['route_dSting1']:+.3f}; rho {rho:.3f}", flush=True)
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--init-from", dest="init_from")
    ap.add_argument("--out")
    ap.add_argument("--objective", action="store_true",
                    help="run objective_variants (step 12e) instead of transmission_variants")
    a = ap.parse_args()
    cfg = load_config()
    tt = cfg.objective_test if a.objective else cfg.transmission_test
    variants = cfg.objective_variants if a.objective else cfg.transmission_variants
    base = dict(tt.get("base", {}))
    steps = tt["steps"] if a.steps is None else a.steps
    name = a.init_from or tt.get("init_from", "none")
    src = None
    if name and name != "none":
        src = cfg.paths.output_dir / name / "checkpoint.pt"
        if not src.exists():
            sys.exit(f"missing {src}")
    out_root = Path(a.out) if a.out else cfg.paths.output_dir / (
        "objective_variants" if a.objective else "transmission_variants")
    ids = a.variants or list(variants)
    print(f"[{'objective' if a.objective else 'transmission'} variants] device {a.device}, {steps} steps each, "
          f"start {'step-9b initialisation' if src is None else src}")
    rows = [run_variant(cfg, v, {**base, **dict(variants[v])}, a.device, steps, out_root, src, tt["watch"])
            for v in ids]
    summ = pd.DataFrame(rows).drop(columns=["route_hop_gains"])
    p = out_root / "summary.csv"
    if p.exists() and a.variants:
        old = pd.read_csv(p)
        summ = pd.concat([old[~old.variant.isin(summ.variant)], summ], ignore_index=True).sort_values("variant")
    summ.to_csv(p, index=False)
    checker = "check_objective_variants.py" if a.objective else "check_transmission_variants.py"
    print(f"\nwrote {p}\nnext: python tests/{checker}")


if __name__ == "__main__":
    main()
