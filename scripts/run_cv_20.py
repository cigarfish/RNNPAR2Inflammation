"""Step 20: 5-fold cross-validation of LEMBAS, so it is scored the way its rivals already were.

Every method in the step-16 benchmark got a 5-fold out-of-fold number; LEMBAS got one fold
(fold 0), because it has to be retrained for each. That asymmetry favoured LEMBAS on the first
thing a reviewer checks - whether the single fold was lucky. This rotates the held-out fold,
retrains, and reports the spread.

This is EVALUATION, not selection. The rules live in `config.yaml: cv_test` and were fixed
before the run; nothing here may change a model, prior, sign or curation decision, and the
sealed holdouts are neither touched nor read.

    python scripts/run_cv_20.py                      # 5 folds x 3 seeds for R4, ~25 min on a GPU
    python scripts/run_cv_20.py --seeds 0 1 2 3 4    # tighter, ~40 min
    python scripts/run_cv_20.py --steps 6 --suffix _smoke   # smoke test, throws the numbers away
    python scripts/run_cv_20.py --collect            # re-aggregate without retraining
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.eval_harness import result_path
sys.path.insert(0, str(ROOT / "scripts"))
from run_source_variants import resolve_spec, run


def collect(cfg, variants, folds, seeds, suffix):
    rows = []
    for v in variants:
        for f in folds:
            for s in seeds:
                name = (v if s == 0 else f"{v}_s{s}") + f"_f{f}" + suffix
                p = result_path(cfg, "12j", name)
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                keep = [cfg.cv_test["primary"]] + list(cfg.cv_test["also_report"])
                rows.append(dict(variant=v, fold=f, seed=s, run=name,
                                 n_val_drug=r.get("n_val_drug"), n_val_geo=r.get("n_val_geo"),
                                 **{k: r.get(k) for k in keep}))
    return pd.DataFrame(rows)


def main():
    cv = load_config().cv_test
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variants", nargs="*", default=list(cv["variants"]))
    ap.add_argument("--folds", nargs="*", type=int, default=list(cv["folds"]))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(cv["seeds"]))
    ap.add_argument("--steps", type=int)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--collect", action="store_true", help="aggregate existing runs only")
    a = ap.parse_args()

    cfg = load_config()
    steps = cfg.source_test["steps"] if a.steps is None else a.steps
    n = len(a.variants) * len(a.folds) * len(a.seeds)
    if not a.collect:
        print(f"[20] 5-fold CV: {n} runs ({a.variants} x folds {a.folds} x seeds {a.seeds}), "
              f"{steps} steps, device {a.device}", flush=True)
        for v in a.variants:
            spec = resolve_spec(cfg, v)
            for f in a.folds:
                for s in a.seeds:
                    run(cfg, v, spec, s, a.device, steps, False, a.suffix, a.force, val_fold=f)

    t = collect(cfg, a.variants, a.folds, a.seeds, a.suffix)
    if not len(t):
        sys.exit("no runs found; drop --collect to train them")
    out = cfg.paths.output_dir / "step20_cv"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "runs.csv", index=False)

    prim = cv["primary"]
    per_fold = t.groupby(["variant", "fold"]).median(numeric_only=True).drop(columns="seed")
    print("\n== per fold (median over seeds)")
    print(per_fold.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\n== pooled over folds")
    summ = []
    for v in a.variants:
        sub = per_fold.loc[v]
        ref = cv["fold0_reference"]
        band = float(cv["consistency_band"])
        inside = [int(f) for f in sub.index
                  if abs(sub.loc[f, prim] - ref[prim]) <= band]
        row = dict(variant=v,
                   cv_median=float(sub[prim].median()), cv_min=float(sub[prim].min()),
                   cv_max=float(sub[prim].max()), fold0_reference=float(ref[prim]),
                   geo_cv_median=float(sub["r_tf_geo_resid"].median()),
                   geo_fold0_reference=float(ref["r_tf_geo_resid"]),
                   baseline_median=float(sub["baseline_drug_mean_signal"].median()),
                   ceiling_median=float(sub["ceiling_drug_rmax_signal"].median()),
                   folds_within_band=len(inside), n_folds=len(sub))
        summ.append(row)
        print(f"  {v}: drug resid (signal) median {row['cv_median']:+.4f} "
              f"[{row['cv_min']:+.4f}, {row['cv_max']:+.4f}] vs fold-0 {row['fold0_reference']:+.4f}; "
              f"{row['folds_within_band']}/{row['n_folds']} folds within {band:.2f}")
        print(f"      GEO resid median {row['geo_cv_median']:+.4f} vs fold-0 "
              f"{row['geo_fold0_reference']:+.4f}")
        print(f"      per-fold constant-mean baseline {row['baseline_median']:+.3f}, "
              f"noise ceiling {row['ceiling_median']:+.3f}")
    s = pd.DataFrame(summ)
    s.to_csv(out / "summary.csv", index=False)
    (out / "report.json").write_text(json.dumps(
        dict(rules=dict(cv), runs=len(t), per_fold=per_fold.reset_index().to_dict("records"),
             summary=summ), indent=2, default=float))
    print(f"\nNOTE the drug baseline above is expected to BEAT the model (step 12f0); this step\n"
          f"asks only whether fold 0 was representative, and states the spread.\n"
          f"wrote {out}\nnext: python tests/test_step20_cv.py")


if __name__ == "__main__":
    main()
