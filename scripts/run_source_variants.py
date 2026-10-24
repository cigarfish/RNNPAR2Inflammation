"""Step 12j: per-source readout intercept (and gain), scored by the step-12f harness.

Same start and schedule as the step-12e S5 run (step-9b initialisation, fan_in sqrt,
gain >= 0, V6 stage-B priors, 400 steps, batch 32, best-val-loss state kept), with
    y_hat = a_s * pred + mu_s            (TF term)
    gene_pred + mu_gene_s                (gene term)
for the sources listed in `intercept` (config `source_variants`, `source_test`).

Run from the LEMBAS folder (GPU; ~1-2 min per run):
    python scripts/run_source_variants.py                         # J0-J3, seed 0
    python tests/check_source_variants.py                         # selects among J1-J3 -> writes selection.json
    python scripts/run_source_variants.py --variants J4           # J4 = selected + per-source balancing
    python scripts/run_source_variants.py --seeds 1 2 --variants J0 <winner>   # seed check
    python tests/check_source_variants.py                         # final verdict
Writes outputs/source_variants/<ID>[_s<seed>]/{checkpoint.pt, log.jsonl, report.json} and
outputs/harness/12j_<ID>[_s<seed>].json (+ sealed/ with --sealed).
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
from lembas_par2.eval_harness import evaluate_run, write_result
from lembas_par2.loss import Priors, Targets, balance_data_terms
from lembas_par2.model import build_model
from lembas_par2.train import Trainer, all_trainable, apply_prior_spec, save_checkpoint

OUT = "source_variants"


def resolve_spec(cfg, vid, base_override=None):
    spec = dict(cfg.source_variants[vid])
    if spec.get("base") == "selected":
        b = base_override
        if b is None:
            sel = cfg.paths.output_dir / OUT / "selection.json"
            if not sel.exists():
                sys.exit(f"{vid} builds on the J1-J3 winner: run tests/check_source_variants.py first, or pass --base")
            b = json.loads(sel.read_text())["selected"]
            if b is None:
                sys.exit("no J1-J3 variant was eligible, so J4 has no base (recorded in selection.json)")
        spec = {**dict(cfg.source_variants[b]), **{k: v for k, v in spec.items() if k != "base"}, "base_variant": b}
    return spec


def run(cfg, vid, spec, seed, dev, steps, sealed, suffix="", force=False, val_fold=None):
    st = cfg.source_test
    # step 20: --val-fold rotates the held-out CV fold. The fold goes into the run NAME so the
    # folds cannot overwrite each other, and into the EVAL config as well as the training split,
    # since evaluate_run reads cfg.stage_b.val_fold to decide what it is scoring.
    if val_fold is not None:
        suffix = f"_f{val_fold}" + suffix
    name = (vid if seed == 0 else f"{vid}_s{seed}") + suffix
    out = cfg.paths.output_dir / OUT / name
    old = out / "report.json"
    if old.exists() and not force:
        prev = json.loads(old.read_text()).get("steps", 0)
        if prev > steps:
            sys.exit(f"refusing to overwrite {name} ({prev} steps) with a {steps}-step run; "
                     f"use --suffix _smoke for a test run, or --force")
    out.mkdir(parents=True, exist_ok=True)
    base = dict(st["base"])
    cfg2 = load_config()
    cfg2["model"]["fan_in_normalise"] = base.get("fan_in_normalise", "none")
    cfg2["model"]["unlock_signs"] = spec.get("unlock_signs", "none")      # step 13b sign-free probe
    cfg2["model"]["locked_param"] = spec.get("locked_param", "softplus")  # step 14 reparameterisation
    if spec.get("unsigned_init") is not None:                             # step 18
        cfg2["model"]["readout_unsigned_init"] = float(spec["unsigned_init"])
    sb = dict(cfg.stage_b)
    if val_fold is not None:
        sb["val_fold"] = int(val_fold)
        cfg2["stage_b"]["val_fold"] = int(val_fold)
    torch.manual_seed(seed)
    t0 = time.time()
    print(f"\n=== {name} {spec or 'J0 = S5 readout'}  (seed {seed})", flush=True)

    m = build_model(cfg2, device=dev, seed=seed)
    init_rep = m.initialize()
    T = Targets(cfg2, m).configure_tf(base["tf_mode"], base.get("tf_gain_nonneg", True))
    cond = m.design.cond
    is_src = cond.source.isin(sb["sources"]).values
    val = [int(i) for i in np.where(is_src)[0]
           if cond.source.iloc[i] in ("drug", "geo") and cond.fold.iloc[i] == sb["val_fold"]]
    train = [int(i) for i in np.where(is_src)[0] if i not in set(val)]
    if spec.get("exclude_sources"):
        train = [i for i in train if cond.source.iloc[i] not in set(spec["exclude_sources"])]
        print(f"  training without {spec['exclude_sources']}: {len(train)} conditions")

    nuis = bool(spec.get("intercept")) or spec.get("gain") == "source"
    if nuis:
        T.configure_nuisance(intercept=spec.get("intercept", ()), gain=spec.get("gain", "global"),
                             gene_intercept=st["gene_intercept"], gain_nonneg=base.get("tf_gain_nonneg", True))
        T.refit_nuisance(m, train)
    balance_data_terms(m, T)
    # 'unanchored' (step 12n / 13 honesty check): route term off, T0 anchored like T1, and the
    # route responsive-range penalty dropped (Priors.configure ignores it in that mode)
    P = Priors(cfg2, m, mode=spec.get("mode", "anchored"))
    apply_prior_spec(P, dict(sb["priors"]), P.route_nodes_mask(), cfg2.stage_a_activation_range,
                     reduce=sb["activation_reduce"])
    if spec.get("unsigned_l1_mult") is not None:                          # step 18
        P.configure(readout_free_mult=spec["unsigned_l1_mult"])
    bal = T.balance_sources(m, train) if spec.get("balance_sources") else {}
    if bal:
        print("  source balancing: " + ", ".join(f"{k} x{v['factor']:.3f} (loss {v['loss']:.3f})" for k, v in bal.items()))
        if nuis:
            T.refit_nuisance(m, train)
    lam = dict(balance_data_terms(m, T, conditions=train))

    k = int(st["refresh_every"])
    pre = (lambda model, s: T.refit_nuisance(model, train) if s % k == 0 else None) if nuis else None
    lr_mult = {"W_free": float(spec["unsigned_lr_mult"])} if spec.get("unsigned_lr_mult") else None
    tr = Trainer(m, T, P, mask=all_trainable(m), lr=sb["lr"], warmup=min(sb["warmup"], max(1, steps // 10)),
                 seed=seed, log_path=out / "log.jsonl", project=sb["project_stability"], lr_mult=lr_mult)
    if steps > 0:
        tr.fit(train, val_pos=val, steps=steps, batch=sb["batch"], eval_every=max(10, steps // 6),
               patience=None, seed=seed, verbose=False, pre_step=pre)
    if nuis:
        T.refit_nuisance(m, train)            # the kept (best-val-loss) state, refit on training only
    ns = T.nuisance_state()
    ck = out / "checkpoint.pt"
    save_checkpoint(m, ck, stage="B_source", variant=vid, seed=seed, spec=spec, nuisance=ns, mode=P.mode,
                    balance={k_: v["factor"] for k_, v in bal.items()}, lambdas=lam, train=train)
    if nuis:
        torch.save(dict(mu_tf=T.mu_tf.cpu(), mu_gene=T.mu_gene.cpu(), a_src=T.a_src.cpu()), out / "nuisance.pt")

    eval_cfg = cfg if val_fold is None else cfg2
    res = evaluate_run(eval_cfg, ck, step="12j", variant=name, device=dev, model=m,
                       meta=dict(stage="B_source", variant=vid, seed=seed, spec=spec, nuisance=ns, mode=P.mode))
    p = write_result(cfg, res)
    if sealed:
        from lembas_par2.eval_sealed import evaluate_holdouts
        evaluate_holdouts(cfg, ck, step="12j", variant=name, device=dev, model=m)
    rep = dict(variant=vid, name=name, seed=seed, spec=spec, steps=steps, val_fold=val_fold, rho_init=init_rep["rho_after"],
               nuisance=ns, balance=bal, lambdas=lam, harness=str(p), seconds=round(time.time() - t0, 1))
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(f"  drug r {res['r_tf_drug']:+.3f} -> resid {res['r_tf_drug_resid']:+.3f} | "
          f"**signal resid {res['r_tf_drug_resid_signal']:+.3f}** (ceiling {res['ceiling_drug_rmax_signal']:.3f}) | "
          f"GEO r {res['r_tf_geo']:+.3f} resid {res['r_tf_geo_resid']:+.3f} | route dSting1 {res['route_dSting1']:+.3f} | "
          f"rho {res['rho_absA']:.3f}" + (f" | a_drug {ns['a_drug']:.2f} a_geo {ns['a_geo']:.2f}" if ns else "")
          + f"  [{rep['seconds']:.0f} s]", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variants", nargs="*", help="default: J0 J1 J2 J3")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--base", help="J4's base variant (default: outputs/source_variants/selection.json)")
    ap.add_argument("--sealed", action="store_true", help="also write sealed holdout metrics")
    ap.add_argument("--suffix", default="", help="appended to run names, e.g. _smoke for test runs")
    ap.add_argument("--force", action="store_true", help="allow overwriting a longer run")
    ap.add_argument("--val-fold", type=int, dest="val_fold",
                    help="step 20: hold out this CV fold instead of stage_b.val_fold")
    a = ap.parse_args()
    cfg = load_config()
    steps = cfg.source_test["steps"] if a.steps is None else a.steps
    ids = a.variants or ["J0", "J1", "J2", "J3"]
    print(f"[12j] device {a.device}, {steps} steps, seeds {a.seeds}, variants {ids}")
    for s in a.seeds:
        for v in ids:
            run(cfg, v, resolve_spec(cfg, v, a.base), s, a.device, steps, a.sealed, a.suffix,
                a.force, a.val_fold)
    print("\nnext: python tests/check_source_variants.py")


if __name__ == "__main__":
    main()
