"""Step 12f0: drug noise ceiling + trivial baselines, against the model's val r(drug).

Needs no GPU and no training. See lembas_par2/noise_ceiling.py for the method and the
decision rules (fixed before running).

Run from the LEMBAS folder:
    python scripts/run_noise_ceiling.py                  # ceiling + baselines (~1-3 min, CPU)
    python scripts/run_noise_ceiling.py --model C4 C0    # + per-drug model r from those checkpoints
Writes outputs/step12f0/{ceiling.csv, baselines.csv, baselines_per_condition.csv,
                         model_per_condition.csv, report.json}; then:
    python tests/test_step12f0.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.noise_ceiling import corr, evaluate_baselines, shared_reference, split_half


def model_r(cfg, variants, val_ids, train_ids, tfa):
    """Per-condition val r(TF) of saved composition-variant checkpoints."""
    import torch
    from lembas_par2.loss import Targets
    from lembas_par2.model import build_model
    from lembas_par2.train import evaluate, load_checkpoint
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for v in variants:
        ck = cfg.paths.output_dir / "composition_variants" / v / "checkpoint.pt"
        if not ck.exists():
            print(f"  [skip] {ck} not found")
            continue
        m = build_model(cfg, device=dev, seed=cfg.stage_b["seed"])
        load_checkpoint(m, ck)
        T = Targets(cfg, m)
        pos = [m.design.cond.condition_id.tolist().index(c) for c in val_ids]
        ev = evaluate(m, T, pos)
        out[v] = pd.Series(ev["r_tf"], index=val_ids)
        # drug-specific part: remove the training-drug mean profile from the target
        pred = pd.DataFrame(ev["out"]["tf_pred"].detach().cpu().numpy(), index=m.tf_outputs, columns=val_ids)
        tgt = tfa.reindex(index=m.tf_outputs, columns=val_ids)
        resid = tgt.sub(tfa.reindex(m.tf_outputs)[train_ids].mean(axis=1), axis=0)
        out[f"{v}_resid"] = pd.Series({c: corr(pred[c].values, resid[c].values) for c in val_ids})
        print(f"  model {v}: median val r(drug) {out[v].median():+.3f}; "
              f"against target minus training-drug mean {out[f'{v}_resid'].median():+.3f}")
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="*", default=[], help="composition variants to score per drug, e.g. C4 C0")
    ap.add_argument("--max-splits", type=int, default=None, help="cap splits per drug (smoke runs)")
    a = ap.parse_args()
    cfg = load_config()
    out = cfg.paths.output_dir / "step12f0"
    out.mkdir(parents=True, exist_ok=True)
    o = cfg.paths.output_dir
    clip = cfg.loss["tf_target_clip"]
    tfs = list(pd.read_csv(o / "step05_tf_outputs.csv").tf)
    reg = pd.read_csv(o / "step05_regulons.csv.gz")
    cond = pd.read_csv(o / "step06_conditions.csv")
    tfa = pd.read_csv(o / "step06_tf_activity.csv.gz", index_col=0).reindex(tfs).clip(-clip, clip)
    val_fold = cfg.stage_b["val_fold"]
    drugs = cond[cond.source == "drug"]
    val_drug = list(drugs.condition_id[drugs.fold == val_fold])
    train_drug = [c for c in drugs.condition_id[drugs.fold != val_fold] if c in tfa.columns]

    # ---- 1. noise ceiling
    print("[12f0] split-half noise ceiling, GSE219229 ...", flush=True)
    ceil, full = split_half(cfg, reg, tfs, min_targets=cfg.conditions["min_tf_targets_condition"],
                            clip=clip, max_splits=a.max_splits)
    ceil = ceil.merge(drugs[["condition_id", "cls", "weight", "n_sig", "fold"]], on="condition_id", how="left")
    ceil.to_csv(out / "ceiling.csv", index=False)
    full.to_csv(out / "approx_full_lfc.csv.gz")
    cv = ceil[ceil.condition_id.isin(val_drug)]
    print(ceil[["drug", "cls", "weight", "n_sig", "r12", "r_max"]].sort_values("r_max")
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- 2. baselines
    print("\n[12f0] trivial baselines (training-fold conditions only)", flush=True)
    bd, pd_ = evaluate_baselines(tfa, cond, "drug", val_fold)
    bg, pg = evaluate_baselines(tfa, cond, "geo", val_fold)
    bd["source"], bg["source"] = "drug", "geo"
    base = pd.concat([bd, bg]).reset_index()
    base.to_csv(out / "baselines.csv", index=False)
    pd.concat([pd_.assign(source="drug"), pg.assign(source="geo")]).to_csv(
        out / "baselines_per_condition.csv", index=False)
    print(base.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n[12f0] shared-DMSO-reference decomposition of the mean-profile baseline", flush=True)
    sh = shared_reference(cfg, reg, tfs, min_targets=cfg.conditions["min_tf_targets_condition"], clip=clip)
    sh.to_csv(out / "shared_reference.csv", index=False)
    print(f"  mean of other drugs vs target, median r: shared DMSO {sh.r_shared.median():+.3f}, "
          f"disjoint DMSO {sh.r_disjoint.median():+.3f}")

    # ---- 3. model per drug (optional)
    mr = model_r(cfg, a.model, val_drug, train_drug, tfa) if a.model else pd.DataFrame(index=val_drug)
    if len(mr.columns):
        mr.join(ceil.set_index("condition_id")[["r_max", "weight"]]).to_csv(out / "model_per_condition.csv")

    # ---- report
    ref = {}
    cs = o / "composition_variants" / "summary.csv"
    if cs.exists():
        s = pd.read_csv(cs).set_index("variant")
        ref = {f"model_{v}_drug": float(s.at[v, "val_r_tf_drug"]) for v in s.index}
        ref.update({f"model_{v}_geo": float(s.at[v, "val_r_tf_geo"]) for v in s.index})
    sa = o / "stage_a" / "report.json"
    if sa.exists():
        ref["model_stageA_drug"] = json.loads(sa.read_text())["val_after"]["r_tf_median"]
    signal = cv[cv.weight > 0.05]
    rep = dict(
        n_drugs_ceiling=len(ceil), val_fold=val_fold, n_val_drug=len(val_drug), n_val_with_ceiling=len(cv),
        r_max_median_all=float(ceil.r_max.median()), r_max_median_val=float(cv.r_max.median()),
        r_max_median_val_signal=float(signal.r_max.median()) if len(signal) else None,
        r12_median_val=float(cv.r12.median()),
        shared_reference=dict(r_shared_median=float(sh.r_shared.median()), r_disjoint_median=float(sh.r_disjoint.median())), n_val_r_max_ge_0_5=int((cv.r_max >= 0.5).sum()),
        baselines={f"{r.source}:{r.baseline}": dict(val=float(r.val_fold_median), oof=float(r.oof_median))
                   for r in base.itertuples()},
        reference=ref,
        model_per_condition={v: dict(median=float(mr[v].median()),
                                     median_signal_drugs=float(mr[v][mr.index.isin(signal.condition_id)].median()))
                             for v in mr.columns},
        baseline_B_mean_resid_r=0.0,  # a constant profile has no drug-specific skill by construction
    )
    c4 = ref.get("model_C4_drug", 0.180)
    bmean = rep["baselines"]["drug:B_mean"]["val"]
    best_drug_base = max(v["val"] for k, v in rep["baselines"].items() if k.startswith("drug:"))
    rep["verdict"] = dict(
        constant_beats_or_ties_C4=bool(bmean >= c4),
        common_response_beats_C4=bool(sh.r_disjoint.median() >= c4),
        best_trivial_drug_baseline=best_drug_base,
        ceiling_low=bool(rep["r_max_median_val"] < 0.40),
        C4_fraction_of_ceiling=float(c4 / rep["r_max_median_val"]) if rep["r_max_median_val"] > 0 else None,
        geo_B_mean_vs_C0=(rep["baselines"]["geo:B_mean"]["val"], ref.get("model_C0_geo")),
    )
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print("\n[12f0] summary")
    print(f"  noise ceiling r_max, val drugs: median {rep['r_max_median_val']:.3f} "
          f"(signal-bearing, weight > 0.05: {rep['r_max_median_val_signal']}); all drugs {rep['r_max_median_all']:.3f}")
    print(f"  drug: B_mean {bmean:+.3f}, best trivial {best_drug_base:+.3f}  vs  model C4 {c4:+.3f}, "
          f"stage A {ref.get('model_stageA_drug', float('nan')):+.3f}, C0 {ref.get('model_C0_drug', float('nan')):+.3f}")
    print(f"  GEO : B_mean {rep['baselines']['geo:B_mean']['val']:+.3f}  vs  model C0 {ref.get('model_C0_geo', float('nan')):+.3f}")
    print(f"wrote {out}\nnext: python tests/test_step12f0.py")


if __name__ == "__main__":
    main()
