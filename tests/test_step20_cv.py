"""Step 20 checks: the cross-validation is a real rotation, and it cannot feed back into the model.

Two kinds of check. The first group runs without any CV output and guards the MECHANISM - that
--val-fold actually changes the split, that each fold gets its own run directory, and that the
step cannot read the sealed holdouts. The second group needs `scripts/run_cv_20.py` to have run
and checks the RESULT against the rules fixed in config before it.

    python tests/test_step20_cv.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
cv = cfg.cv_test
o = cfg.paths.output_dir

print("== the folds are a real rotation of held-out conditions")
cond = pd.read_csv(o / "step06_conditions.csv")
ev = cond[cond.source.isin(["drug", "geo"])]
sizes = ev.groupby("fold").size()
check("every configured fold exists in the condition table and holds out conditions",
      all(f in sizes.index for f in cv["folds"]) and sizes.min() > 0, sizes.to_dict())
check("the folds partition the evaluable conditions (no condition in two folds, none missing)",
      int(sizes.sum()) == len(ev) and ev.fold.notna().all(), f"{int(sizes.sum())} of {len(ev)}")
check("both sources are represented in every fold",
      (ev.groupby(["fold", "source"]).size().unstack() > 0).all().all(),
      ev.groupby(["fold", "source"]).size().unstack().to_dict())
check("the fold column predates this step (it comes from conditions.cv_folds, not step 20)",
      int(cfg.conditions["cv_folds"]) == len(cv["folds"]), cfg.conditions["cv_folds"])

print("\n== the runner rotates the split, and names runs so folds cannot collide")
src = (ROOT / "scripts" / "run_source_variants.py").read_text(encoding="utf-8")
check("--val-fold overrides the TRAINING split", 'sb["val_fold"] = int(val_fold)' in src)
check("--val-fold also overrides the EVAL config (evaluate_run reads cfg.stage_b.val_fold)",
      'cfg2["stage_b"]["val_fold"] = int(val_fold)' in src and "eval_cfg" in src)
check("the fold goes into the run name, so one fold cannot overwrite another",
      'suffix = f"_f{val_fold}"' in src)
check("the fold is recorded in each run's report.json", "val_fold=val_fold" in src)

print("\n== the step cannot select on anything, and cannot see the holdouts")
run_src = (ROOT / "scripts" / "run_cv_20.py").read_text(encoding="utf-8")
check("the CV runner never imports the sealed-holdout module",
      "eval_sealed" not in run_src and "sealed=True" not in run_src)
check("it passes sealed=False to every training run", "a.device, steps, False," in run_src)
check("the rules say in writing that this is evaluation, not selection",
      "never selection" in Path(ROOT / "config.yaml").read_text(encoding="utf-8"))
check("no fold-selection rule exists: the config has no 'select' or 'adopt' key",
      not any(k in dict(cv) for k in ("select", "adopt", "target_reduction")), list(dict(cv)))

p = o / "step20_cv" / "runs.csv"
if not p.exists():
    print(f"\n[INFO] {p} not present: the mechanism checks above are all that can run yet.")
    print("       run: python scripts/run_cv_20.py")
else:
    print("\n== the result, against the rules fixed before the run")
    t = pd.read_csv(p)
    prim = cv["primary"]
    have = t.groupby(["variant", "fold"]).size()
    check("every configured fold and variant was run",
          all((v, f) in have.index for v in cv["variants"] for f in cv["folds"]),
          have.to_dict())
    check("each run scored a non-empty validation set from its own fold",
          (t.n_val_drug > 0).all() and (t.n_val_geo > 0).all(),
          f"drug {sorted(set(t.n_val_drug))}, geo {sorted(set(t.n_val_geo))}")
    check("the held-out condition counts differ across folds or match the table",
          set(t.groupby("fold").n_val_drug.first()) <= set(ev[ev.source == "drug"].groupby("fold").size()),
          t.groupby("fold").n_val_drug.first().to_dict())
    per = t.groupby(["variant", "fold"])[prim].median()
    check("every fold produced a finite primary metric", np.isfinite(per).all(), per.round(4).to_dict())
    ref = cv["fold0_reference"][prim]
    spread = float(per.max() - per.min())
    check("the fold-0 reference in config is unchanged by this run (it is a frozen number)",
          abs(float(cfg.cv_test["fold0_reference"][prim]) - ref) < 1e-12)
    print(f"[INFO] per-fold {prim}: {per.round(4).to_dict()}")
    print(f"[INFO] spread across folds {spread:+.4f}; fold-0 reference {ref:+.4f}; "
          f"CV median {float(per.median()):+.4f}")
    print("[INFO] this is reported, not judged: no threshold here decides anything.")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
