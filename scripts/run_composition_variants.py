"""Does removing the cell-composition signal from the bulk GEO contexts recover the drug fit?

Stage A, trained on drug conditions alone, reached val r(drug) = 0.224. Stage B, which adds
the 48 GEO contexts, drops it to <= 0.033 while GEO rises to 0.52-0.62 (step 12e). The bulk
GEO log2FCs are heavily contaminated by neutrophil infiltration (marker panel +1.70 median,
+3.15 bacterial, up to +4.9), which a cell-intrinsic signalling model cannot represent. This
tests whether that contamination is the cause.

The validation set is IDENTICAL in every variant (drug + GEO fold 0); only the training set
changes, so val r(drug) is directly comparable. Variants: config `composition_variants`.

Run from the LEMBAS folder (GPU recommended; ~2-3 min per variant):
    python scripts/run_composition_variants.py                 # all variants
    python scripts/run_composition_variants.py --steps 0       # scores only, no training
    python scripts/run_composition_variants.py --variants C0 C1
Writes outputs/composition_variants/{summary.csv, scores.csv, <ID>/report.json}; then:
    python tests/check_composition_variants.py
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
from lembas_par2.train import Trainer, all_trainable, apply_prior_spec, evaluate, save_checkpoint

SC_SETS = {"GSE169325epi", "GSE169325mph", "GSE289734t"}


def composition_scores(cfg):
    """Mean log2FC of each marker panel per GEO condition; `score_panel` is the one used."""
    lfc = pd.read_csv(cfg.paths.output_dir / "step06_gene_lfc.csv.gz", index_col=0)
    cond = pd.read_csv(cfg.paths.output_dir / "step06_conditions.csv")
    panels = {k: [g for g in v if g in lfc.index] for k, v in cfg.composition["marker_panels"].items()}
    rows = []
    for r in cond[cond.source == "geo"].itertuples():
        if r.condition_id not in lfc.columns:
            continue
        col = lfc[r.condition_id]
        row = dict(condition_id=r.condition_id, cls=r.cls, name=r.name,
                   kind="single-cell" if r.condition_id in SC_SETS else "bulk",
                   overall_abs_median=float(col.abs().median()))
        for k, gs in panels.items():
            v = col.reindex(gs).dropna()
            row[k] = float(v.mean()) if len(v) else np.nan
        rows.append(row)
    d = pd.DataFrame(rows)
    d["score"] = d[cfg.composition["score_panel"]]
    return d.sort_values("score", ascending=False)


def run_variant(cfg, vid, spec, dev, steps, out_root, scores):
    out = out_root / vid
    out.mkdir(parents=True, exist_ok=True)
    sb = dict(cfg.stage_b)
    torch.manual_seed(sb["seed"])
    t0 = time.time()
    print(f"\n=== {vid} {spec or 'baseline: all GEO contexts'}")

    m = build_model(cfg, device=dev, seed=sb["seed"])
    init_rep = m.initialize()
    T = Targets(cfg, m)
    balance_data_terms(m, T)
    P = Priors(cfg, m, mode="anchored")
    apply_prior_spec(P, dict(sb["priors"]), P.route_nodes_mask(), cfg.stage_a_activation_range,
                     reduce=sb["activation_reduce"])

    cond = m.design.cond
    is_src = cond.source.isin(sb["sources"]).values
    # validation is fixed across variants: drug + GEO fold 0
    val = [i for i in np.where(is_src)[0]
           if cond.source.iloc[i] in ("drug", "geo") and cond.fold.iloc[i] == sb["val_fold"]]
    train_all = [i for i in np.where(is_src)[0] if i not in val]
    sc = scores.set_index("condition_id").score
    geo_pos = {i for i in train_all if cond.source.iloc[i] == "geo"}
    dropped = []
    if spec.get("drug_only"):
        dropped = sorted(geo_pos)
    elif spec.get("exclude_above") is not None:
        thr = float(spec["exclude_above"])
        dropped = sorted(i for i in geo_pos if float(sc.get(cond.condition_id.iloc[i], 0.0)) > thr)
    train = [i for i in train_all if i not in set(dropped)]
    if spec.get("downweight"):
        with torch.no_grad():
            for i in geo_pos:
                s = max(0.0, float(sc.get(cond.condition_id.iloc[i], 0.0)))
                T.weight[i] = T.weight[i] / (1.0 + s)
    n_geo = sum(cond.source.iloc[i] == "geo" for i in train)
    print(f"  training on {len(train)} conditions: {sum(cond.source.iloc[i]=='drug' for i in train)} drug, "
          f"{n_geo} GEO, {sum(cond.source.iloc[i]=='perturb' for i in train)} papers"
          + (f"; dropped {len(dropped)} GEO" if dropped else "")
          + ("; GEO down-weighted by 1/(1+score)" if spec.get("downweight") else ""), flush=True)
    lam = dict(balance_data_terms(m, T, conditions=train))

    vd = [i for i in val if cond.source.iloc[i] == "drug"]
    vg = [i for i in val if cond.source.iloc[i] == "geo"]
    ev0, ev0d = evaluate(m, T, val), evaluate(m, T, vd)
    rep = dict(variant=vid, spec=json.dumps(spec), steps=steps, n_train=len(train), n_train_geo=n_geo,
               n_dropped_geo=len(dropped), dropped=[cond.condition_id.iloc[i] for i in dropped],
               val_r_tf_init=ev0["r_tf_median"], val_r_tf_drug_init=ev0d["r_tf_median"],
               rho_init=init_rep["rho_after"], lambdas=lam)
    if steps > 0:
        tr = Trainer(m, T, P, mask=all_trainable(m), lr=sb["lr"],
                     warmup=min(sb["warmup"], max(1, steps // 10)), seed=sb["seed"],
                     log_path=out / "log.jsonl", project=sb["project_stability"])
        tr.fit(train, val_pos=val, steps=steps, batch=min(sb["batch"], len(train)),
               eval_every=max(10, steps // 6), patience=None, seed=sb["seed"], verbose=False)
        ev1, ev1d, ev1g = evaluate(m, T, val), evaluate(m, T, vd), evaluate(m, T, vg)
        with torch.no_grad():
            rd = P.route_delta(m)
            rho = perron_radius(m.struct, m.weights(), max_iter=3000, tol=1e-6).item()
            z = mml_inverse(simulate(m.struct, m.weights(), m.bias, m.pooled_reference(), None,
                                     max_iter=3000, tol=1e-7)[:, 0].clamp(max=0.999))
            hold = [i for i, s_ in enumerate(cond.source) if s_ == "holdout"]
            oh = m(conditions=hold)
        rep.update(val_r_tf=ev1["r_tf_median"], val_r_tf_drug=ev1d["r_tf_median"],
                   val_r_tf_geo=ev1g["r_tf_median"], val_loss=ev1["loss"], tf_gain=T.last_tf_gain,
                   route_dSting1=rd[m.idx["Sting1"]].item(), route_dMapk1=rd[m.idx["Mapk1"]].item(), rho=rho,
                   **{f"z_{k}": z[m.idx[k]].item() for k in cfg.composition_test["watch"]},
                   **{f"holdout_{cond.condition_id.iloc[i]}_dSting1": oh["delta"][m.idx["Sting1"], k].item()
                      for k, i in enumerate(hold)})
        save_checkpoint(m, out / "checkpoint.pt", stage="B_composition", variant=vid, spec=spec, train=train)
        print(f"  after {steps} steps: val r(TF) {ev0['r_tf_median']:+.3f} -> {ev1['r_tf_median']:+.3f}; "
              f"**drug {ev0d['r_tf_median']:+.3f} -> {ev1d['r_tf_median']:+.3f}**; GEO {ev1g['r_tf_median']:+.3f}; "
              f"route dSting1 {rep['route_dSting1']:+.3f}; rho {rho:.3f}", flush=True)
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_config()
    steps = cfg.composition_test["steps"] if a.steps is None else a.steps
    out_root = Path(a.out) if a.out else cfg.paths.output_dir / "composition_variants"
    out_root.mkdir(parents=True, exist_ok=True)
    scores = composition_scores(cfg)
    scores.to_csv(out_root / "scores.csv", index=False)
    panel = cfg.composition["score_panel"]
    print(f"[composition] device {a.device}, {steps} steps each; contamination = mean log2FC of the "
          f"{panel} panel")
    print(scores.groupby("kind")[["score", "overall_abs_median"]].agg(["size", "median"]).round(2).to_string())
    print(f"  GEO contexts above +2: {int((scores.score > 2).sum())}, above +1: {int((scores.score > 1).sum())}")
    ids = a.variants or list(cfg.composition_variants)
    rows = [run_variant(cfg, v, dict(cfg.composition_variants[v]), a.device, steps, out_root, scores)
            for v in ids]
    summ = pd.DataFrame(rows).drop(columns=["dropped", "lambdas"])
    p = out_root / "summary.csv"
    if p.exists() and a.variants:
        old = pd.read_csv(p)
        summ = pd.concat([old[~old.variant.isin(summ.variant)], summ], ignore_index=True).sort_values("variant")
    summ.to_csv(p, index=False)
    print(f"\nwrote {p}\nnext: python tests/check_composition_variants.py")


if __name__ == "__main__":
    main()
