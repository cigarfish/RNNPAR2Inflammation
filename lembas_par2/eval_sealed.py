"""Step 12f: sealed holdout metrics (PAR2_26 / 29 / 30). Never used for selection.

Writes outputs/harness/sealed/<step>_<variant>_holdout.json. Importing this module while
`eval_harness.select()` is running raises, and `select()` refuses to run in any process
that has imported it - so a selection script cannot read holdout numbers by accident.
"""
import json

import numpy as np
import pandas as pd
import torch

from . import eval_harness as _h

if _h._SELECTING:
    raise ImportError("lembas_par2.eval_sealed imported during selection")


def sealed_path(cfg, step, variant):
    return cfg.paths.output_dir / "harness" / "sealed" / f"{step}_{variant}_holdout.json"


@torch.no_grad()
def evaluate_holdouts(cfg, checkpoint, *, step, variant, device=None, model=None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = model if model is not None else _h.load_model(cfg, checkpoint, dev)[0]
    cond = m.design.cond
    hold = [i for i, s in enumerate(cond.source) if s == "holdout"]
    out = m(conditions=hold)
    col = {cond.condition_id.iloc[i]: j for j, i in enumerate(hold)}
    gpos = {g: k for k, g in enumerate(m.genes)}
    d = pd.read_csv(cfg.paths.output_dir / "step06_directions.csv")
    d = d[d.source == "holdout"]
    rows = []
    for r in d.itertuples():
        if r.condition_id not in col:
            continue
        j = col[r.condition_id]
        if r.layer == "signal" and r.node in m.idx:
            v = out["delta"][m.idx[r.node], j].item()
        elif r.layer == "mrna" and r.node in gpos:
            v = out["gene_pred"][gpos[r.node], j].item()
        else:
            continue
        rows.append(dict(observation_id=r.observation_id, condition_id=r.condition_id, node=r.node,
                         layer=r.layer, sign=int(r.sign), predicted=v,
                         agree=bool(np.sign(v) == r.sign) if r.sign != 0 else None))
    obs = pd.DataFrame(rows)
    scored = obs[obs.agree.notna()] if len(obs) else obs
    res = dict(step=step, variant=variant, checkpoint=str(checkpoint),
               dSting1={cond.condition_id.iloc[i]: out["delta"][m.idx["Sting1"], j].item() for j, i in enumerate(hold)},
               direction_agree=float(scored.agree.mean()) if len(scored) else float("nan"),
               n_scored=int(len(scored)), observations=obs.to_dict(orient="records"))
    p = sealed_path(cfg, step, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    return res, p
