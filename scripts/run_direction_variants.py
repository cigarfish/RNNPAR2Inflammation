"""Compare formulations of the perturbation-paper direction term (config: direction_variants).

For each variant, at the stage A starting point:
  static   lambda_direction after re-balancing, how concentrated the term is (top-3 share,
           Gini), the fraction of constraints satisfied, and the gradient the term puts on
           the ERK biases - the step-12 failure mode was ERK being dragged into the leak
           region by three far-off mRNA constraints
  short    a stage-B-style run (whole network, drug + GEO + papers, stability projection)
           reporting fit, route responsiveness and where the watched nodes end up

Run from the LEMBAS folder (GPU recommended; ~1 min per variant at the default 300 steps):
    python scripts/run_direction_variants.py                 # all variants
    python scripts/run_direction_variants.py --steps 0       # static diagnostics only (fast)
    python scripts/run_direction_variants.py --variants D0 D5
Writes outputs/direction_variants/{summary.csv, <ID>/report.json}; then:
    python tests/check_direction_variants.py
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
from lembas_par2.loss import Priors, Targets, balance_data_terms, data_loss
from lembas_par2.model import build_model
from lembas_par2.train import Trainer, all_trainable, apply_prior_spec, evaluate, load_checkpoint, save_checkpoint


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def preactivation(m):
    with torch.no_grad():
        X = simulate(m.struct, m.weights(), m.bias, m.pooled_reference(), None, max_iter=3000, tol=1e-7)
        return mml_inverse(X[:, 0].clamp(max=0.999))


def constraint_table(m, T, pos):
    """Per-constraint value, residual and weight for the given condition positions."""
    col = {p: k for k, p in enumerate(pos)}
    with torch.no_grad():
        o = m(conditions=pos)
        rows = []
        for j in range(len(T.d_sign)):
            c = int(T.d_cond[j])
            if c not in col:
                continue
            k, i, mrna, s = col[c], int(T.d_index[j]), bool(T.d_mrna[j]), float(T.d_sign[j])
            v = (o["gene_pred"] if mrna else o["delta"])[i, k].item()
            r = T.direction_residuals(o, torch.tensor([v], device=m.device, dtype=m.dtype),
                                      torch.tensor([s], device=m.device, dtype=m.dtype))[0].item()
            w = float(T.d_weight[j] * T.d_layer_w[j])
            rows.append(dict(cond=int(c), layer="mrna" if mrna else "signal", sign=s, value=v,
                             residual=r, weight=w, contribution=r * w,
                             satisfied=bool((s != 0 and np.sign(v) == s) or (s == 0 and abs(v) <= T.margin))))
    return pd.DataFrame(rows)


def run_variant(cfg, vid, spec, dt, dev, steps, out_root, src):
    out = out_root / vid
    out.mkdir(parents=True, exist_ok=True)
    sb = cfg.stage_b
    torch.manual_seed(sb["seed"])
    t0 = time.time()

    m = build_model(cfg, device=dev, seed=sb["seed"])
    m.initialize()
    T = Targets(cfg, m).configure_direction(**spec)
    balance_data_terms(m, T)
    P = Priors(cfg, m, mode="anchored")
    apply_prior_spec(P, dict(sb["priors"]), P.route_nodes_mask(), cfg.stage_a_activation_range,
                     reduce=sb["activation_reduce"])
    load_checkpoint(m, src)

    cond = m.design.cond
    is_src = cond.source.isin(sb["sources"]).values
    val = [i for i in np.where(is_src)[0]
           if cond.source.iloc[i] in ("drug", "geo") and cond.fold.iloc[i] == sb["val_fold"]]
    train = [i for i in np.where(is_src)[0] if i not in val]
    lam = dict(balance_data_terms(m, T, conditions=train))

    # ---- static diagnostics at the stage A state ----
    ct = constraint_table(m, T, train)
    share = (ct.contribution / ct.contribution.sum()).sort_values(ascending=False)
    top3 = ct.assign(s=share).nlargest(3, "s")
    m.zero_grad()
    (lam["direction"] * data_loss(m(conditions=train), T)["direction"]).backward()
    g_dir = {n: m.bias.grad[m.idx[n]].item() for n in ["Mapk1", "Mapk3"]}
    m.zero_grad()
    L = data_loss(m(conditions=train), T)
    (lam["tf"] * L["tf"] + lam["gene"] * L["gene"]).backward()
    g_meas = {n: m.bias.grad[m.idx[n]].item() for n in ["Mapk1", "Mapk3"]}
    static = dict(lambda_direction=lam["direction"], top3_share=float(share.head(3).sum()),
                  gini=gini(ct.contribution), frac_satisfied=float(ct.satisfied.mean()),
                  n_constraints=len(ct), grad_erk_direction=g_dir, grad_erk_measured=g_meas,
                  erk_grad_ratio=abs(g_dir["Mapk1"]) / max(abs(g_meas["Mapk1"]), 1e-12),
                  top3=[dict(cond=cond.condition_id.iloc[int(r.cond)], layer=r.layer, sign=r.sign,
                             value=r.value, share=float(r.s)) for r in top3.itertuples()])
    print(f"\n=== {vid} {spec}")
    print(f"  static: lambda_dir {static['lambda_direction']:.1f}, top-3 share {static['top3_share']:.1%}, "
          f"Gini {static['gini']:.2f}, satisfied {static['frac_satisfied']:.0%}, "
          f"ERK grad direction {g_dir['Mapk1']:+.3f} vs measured {g_meas['Mapk1']:+.4f} "
          f"(ratio {static['erk_grad_ratio']:.0f}x)", flush=True)

    rep = dict(variant=vid, spec=json.dumps(spec), direction=T.direction_spec(), steps=steps, **static)
    if steps > 0:
        ev0 = evaluate(m, T, val)
        tr = Trainer(m, T, P, mask=all_trainable(m), lr=sb["lr"], warmup=min(sb["warmup"], max(1, steps // 10)),
                     seed=sb["seed"], log_path=out / "log.jsonl", project=sb["project_stability"])
        tr.fit(train, val_pos=val, steps=steps, batch=sb["batch"], eval_every=max(10, steps // 6),
               patience=None, seed=sb["seed"], verbose=False)
        ev1 = evaluate(m, T, val)
        z = preactivation(m)
        with torch.no_grad():
            rd = P.route_delta(m)
            rho = perron_radius(m.struct, m.weights(), max_iter=3000, tol=1e-6).item()
            hold = [i for i, s in enumerate(cond.source) if s == "holdout"]
            oh = m(conditions=hold)
        ct1 = constraint_table(m, T, train)
        rep.update(val_loss_before=ev0["loss"], val_loss=ev1["loss"], val_r_tf_before=ev0["r_tf_median"],
                   val_r_tf=ev1["r_tf_median"], direction_agree=ev1["direction_agree"],
                   frac_satisfied_after=float(ct1.satisfied.mean()), rho=rho,
                   route_dSting1=rd[m.idx["Sting1"]].item(), route_dMapk1=rd[m.idx["Mapk1"]].item(),
                   route_dBax=rd[m.idx["Bax"]].item(),
                   **{f"z_{n}": z[m.idx[n]].item() for n in cfg.direction_test["watch"]},
                   **{f"holdout_{cond.condition_id.iloc[i]}_dSting1": oh["delta"][m.idx["Sting1"], k].item()
                      for k, i in enumerate(hold)})
        save_checkpoint(m, out / "checkpoint.pt", stage="B_direction_test", variant=vid, direction=T.direction_spec())
        print(f"  after {steps} steps: val r_tf {ev0['r_tf_median']:+.3f} -> {ev1['r_tf_median']:+.3f}, "
              f"val loss {ev0['loss']:.3f} -> {ev1['loss']:.3f}, satisfied {rep['frac_satisfied_after']:.0%}, "
              f"route dSting1 {rep['route_dSting1']:+.3f}, z(ERK) {rep['z_Mapk1']:+.2f}, rho {rho:.3f}", flush=True)
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
    a = ap.parse_args()
    cfg = load_config()
    dt = cfg.direction_test
    steps = dt["steps"] if a.steps is None else a.steps
    name = a.init_from or dt["init_from"]
    src = cfg.paths.output_dir / name / "checkpoint.pt"
    if not src.exists():
        alt = cfg.paths.output_dir / "stage_a_variants" / "V6" / "checkpoint.pt"
        if not alt.exists():
            sys.exit(f"missing {src}; run scripts/train_stage_a.py first")
        print(f"[note] {src} not found, using {alt}")
        src = alt
    meta = torch.load(src, map_location="cpu", weights_only=False)["meta"]
    init_priors = meta.get("priors") or meta.get("spec") or {}
    if float(init_priors.get("lambda_bias", 0)) < 1.0:
        print(f"[warning] {src.parent.name} was trained without bias control (priors {init_priors or 'none'}),\n"
              f"          so it carries the step-11b drifted biases. The comparison still works, but re-run\n"
              f"          'python scripts/train_stage_a.py' first to start from the V6 state.")
    out_root = Path(a.out) if a.out else cfg.paths.output_dir / "direction_variants"
    ids = a.variants or list(cfg.direction_variants)
    print(f"[direction variants] device {a.device}, {steps} steps each, start {src}")
    rows = [run_variant(cfg, v, dict(cfg.direction_variants[v]), dt, a.device, steps, out_root, src) for v in ids]
    summ = pd.DataFrame(rows).drop(columns=["top3", "grad_erk_direction", "grad_erk_measured", "direction"])
    p = out_root / "summary.csv"
    if p.exists() and a.variants:
        old = pd.read_csv(p)
        summ = pd.concat([old[~old.variant.isin(summ.variant)], summ], ignore_index=True).sort_values("variant")
    summ.to_csv(p, index=False)
    print(f"\nwrote {p}\nnext: python tests/check_direction_variants.py")


if __name__ == "__main__":
    main()
