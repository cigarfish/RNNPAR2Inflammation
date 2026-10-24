"""Step 12f: one evaluation schema for every later variant.

`evaluate_run(cfg, checkpoint)` loads a saved model and returns a flat dict with a fixed
set of keys (SCHEMA_KEYS), written by `write_result` to outputs/harness/<step>_<variant>.json.

Validation split: drug + GEO conditions of `stage_b.val_fold` (the split used by every
step-12 variant runner). The literature perturbation papers are train-only (fold -1) and
have no TF matrix, so they contribute direction agreement on the training set only.

Metrics, per source s in {drug, geo}:
  r_tf_<s>              median per-condition Pearson r(prediction, target) over TFs
                        (the metric REPORT.md used so far)
  r_tf_<s>_resid        median r(prediction - mean_s(prediction), target - mean_s(target)),
                        both means over the *training-fold* conditions of source s.
                        Invariant to adding any constant profile per source, so it measures
                        skill beyond "emit the source's average response" (step 12f0: the
                        drug r 0.18 of C4 was entirely that average). SELECTION METRIC (drug).
  r_tf_<s>_resid_tgt    target residualised only (continuity with step 12f0's -0.031)
  baseline_<s>_mean     r of the constant training-fold mean profile (step 12f0 B_mean)
For drug, every r is also given on the "signal" subset: validation drugs whose step-12f0
noise ceiling r_max >= SIGNAL_RMAX (`_signal` suffix), plus the ceiling itself. The
selection metric is r_tf_drug_resid_signal.

Holdouts: never computed here. `lembas_par2.eval_sealed` writes them to
outputs/harness/sealed/; `select()` refuses to run once that module is imported in the
same process and refuses to read anything under sealed/.
"""
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCHEMA_VERSION = 1
SIGNAL_RMAX = 0.30
_SELECTING = False

SCHEMA_KEYS = [
    "schema_version", "step", "variant", "checkpoint", "checkpoint_meta", "device",
    "n_val_drug", "n_val_drug_signal", "n_val_geo", "n_train",
    "r_tf_drug", "r_tf_drug_signal", "r_tf_drug_resid", "r_tf_drug_resid_signal",
    "r_tf_drug_resid_tgt", "r_tf_drug_resid_tgt_signal",
    "r_tf_geo", "r_tf_geo_resid", "r_tf_geo_resid_tgt", "r_tf_val",
    "r_gene_drug", "r_gene_geo",
    "baseline_drug_mean", "baseline_drug_mean_signal", "baseline_geo_mean",
    "ceiling_drug_rmax", "ceiling_drug_rmax_signal", "frac_of_ceiling_drug_signal",
    "direction_agree_train", "n_directions_train",
    "rho_absA", "rho_signedA",
    "delta_tf_median", "delta_tf_p10", "delta_tf_p90",
    "delta_tf_median_drug", "delta_tf_median_geo",
    "route_dSting1", "route_dMapk1",
    "gain_global", "gain_drug", "gain_geo", "gains_negative",
    "solver_residual_max", "solver_converged",
    "per_condition", "provenance",
]


# ---------- helpers ----------
def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def _med(x):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    return float(np.median(x)) if len(x) else float("nan")


def _gain(p, y):
    """Closed-form least-squares scalar gain, unconstrained (its sign is reported)."""
    m = np.isfinite(p) & np.isfinite(y)
    den = float((p[m] ** 2).sum())
    return float((p[m] * y[m]).sum() / den) if den > 0 else float("nan")


def split(cfg, cond):
    """(val positions, train positions, data-split training positions per source)."""
    vf = cfg.stage_b["val_fold"]
    src = cond.source.values
    val = [i for i in range(len(cond)) if src[i] in ("drug", "geo") and cond.fold.iloc[i] == vf]
    train = [i for i in range(len(cond)) if src[i] != "holdout" and i not in set(val)]
    by_src = {s: [i for i in train if src[i] == s] for s in ("drug", "geo")}
    return val, train, by_src


def load_model(cfg, checkpoint, device):
    from .config import load_config
    from .model import build_model
    from .train import load_checkpoint
    # a sign-free (step 13b) checkpoint has a different locked / free parameter split, so the
    # model must be built with the same unlock setting before the state can be loaded
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    spec = (ck.get("meta") or {}).get("spec") or {}
    want = dict(unlock_signs=spec.get("unlock_signs", "none"), locked_param=spec.get("locked_param", "softplus"))
    if any(want[k] != cfg.model.get(k, d) for k, d in [("unlock_signs", "none"), ("locked_param", "softplus")]):
        cfg = load_config()
        cfg["model"].update(want)
    m = build_model(cfg, device=device, seed=cfg.stage_b["seed"])
    meta = load_checkpoint(m, checkpoint)
    m.eval()
    return m, meta


def _config_hash(cfg):
    p = Path(cfg.root) / "config.yaml"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


# ---------- main entry ----------
@torch.no_grad()
def evaluate_run(cfg, checkpoint, *, step, variant, device=None, model=None, meta=None):
    from .core import perron_radius, signed_radius
    from .loss import Priors, Targets
    from .train import evaluate

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model, meta = load_model(cfg, checkpoint, dev)
    m = model
    cond = m.design.cond
    T = Targets(cfg, m)
    val, train, by_src = split(cfg, cond)
    ids = list(cond.condition_id)
    src = cond.source.values
    vd = [i for i in val if src[i] == "drug"]
    vg = [i for i in val if src[i] == "geo"]

    ceil_p = cfg.paths.output_dir / "step12f0" / "ceiling.csv"
    rmax = pd.read_csv(ceil_p).set_index("condition_id").r_max if ceil_p.exists() else pd.Series(dtype=float)
    vd_sig = [i for i in vd if float(rmax.get(ids[i], np.nan)) >= SIGNAL_RMAX]

    # one forward over every non-holdout condition: predictions, solver diagnostics
    pos = sorted(set(val) | set(train))
    info = {}
    out = m(conditions=pos, info=info)
    col = {p: j for j, p in enumerate(pos)}
    P = out["tf_pred"].double().cpu().numpy()
    Y = T.tf.double().cpu().numpy()
    M = T.tf_mask.double().cpu().numpy() > 0
    Yn = np.where(M, Y, np.nan)
    Pn = {i: np.where(M[:, i], P[:, col[i]], np.nan) for i in pos}
    D = out["delta"][m.tf_idx].abs().double().cpu().numpy()
    fwd = info.get("last_forward", {})

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # TFs never measured in a source -> NaN mean
        mu_y = {s: np.nanmean(Yn[:, by_src[s]], axis=1) for s in by_src}
        mu_p = {s: np.nanmean(np.stack([Pn[i] for i in by_src[s]], axis=1), axis=1) for s in by_src}

    rows = []
    for i in val:
        s = src[i]
        y, p = Yn[:, i], Pn[i]
        rows.append(dict(condition_id=ids[i], source=s, weight=float(cond.weight.iloc[i]),
                         r_max=float(rmax.get(ids[i], np.nan)) if s == "drug" else np.nan,
                         r=_corr(p, y), r_resid=_corr(p - mu_p[s], y - mu_y[s]),
                         r_resid_tgt=_corr(p, y - mu_y[s]), r_baseline_mean=_corr(mu_y[s], y),
                         delta_tf_median=float(np.median(D[:, col[i]]))))
    pc = pd.DataFrame(rows).set_index("condition_id")

    def med(col_, subset):
        return _med(pc.loc[[ids[i] for i in subset], col_]) if subset else float("nan")

    ev_d, ev_g = evaluate(m, T, vd), evaluate(m, T, vg)
    ev_tr = evaluate(m, T, train)
    ev_val = evaluate(m, T, val)
    Pr = Priors(cfg, m)
    rd = Pr.route_delta(m)

    def src_gain(s):
        idx = by_src[s]
        return _gain(np.concatenate([Pn[i] for i in idx]), np.concatenate([Yn[:, i] for i in idx]))

    g_all = _gain(np.concatenate([Pn[i] for i in train if src[i] in ("drug", "geo")]),
                  np.concatenate([Yn[:, i] for i in train if src[i] in ("drug", "geo")]))
    gains = dict(gain_global=g_all, gain_drug=src_gain("drug"), gain_geo=src_gain("geo"))
    dval = D[:, [col[i] for i in val]]

    res = dict(
        schema_version=SCHEMA_VERSION, step=step, variant=variant, checkpoint=str(checkpoint),
        checkpoint_meta={k: v for k, v in (meta or {}).items() if k != "train"}, device=dev,
        n_val_drug=len(vd), n_val_drug_signal=len(vd_sig), n_val_geo=len(vg), n_train=len(train),
        r_tf_drug=med("r", vd), r_tf_drug_signal=med("r", vd_sig),
        r_tf_drug_resid=med("r_resid", vd), r_tf_drug_resid_signal=med("r_resid", vd_sig),
        r_tf_drug_resid_tgt=med("r_resid_tgt", vd), r_tf_drug_resid_tgt_signal=med("r_resid_tgt", vd_sig),
        r_tf_geo=med("r", vg), r_tf_geo_resid=med("r_resid", vg), r_tf_geo_resid_tgt=med("r_resid_tgt", vg),
        r_tf_val=float(ev_val["r_tf_median"]),
        r_gene_drug=float(ev_d["r_gene_median"]), r_gene_geo=float(ev_g["r_gene_median"]),
        baseline_drug_mean=med("r_baseline_mean", vd), baseline_drug_mean_signal=med("r_baseline_mean", vd_sig),
        baseline_geo_mean=med("r_baseline_mean", vg),
        ceiling_drug_rmax=med("r_max", vd), ceiling_drug_rmax_signal=med("r_max", vd_sig),
        direction_agree_train=ev_tr["direction_agree"], n_directions_train=ev_tr["n_directions"],
        rho_absA=perron_radius(m.struct, m.weights(), max_iter=3000, tol=1e-8).item(),
        rho_signedA=signed_radius(m.struct, m.weights()),
        delta_tf_median=float(np.median(dval)), delta_tf_p10=float(np.percentile(dval, 10)),
        delta_tf_p90=float(np.percentile(dval, 90)),
        delta_tf_median_drug=med("delta_tf_median", vd), delta_tf_median_geo=med("delta_tf_median", vg),
        route_dSting1=rd[m.idx["Sting1"]].item(), route_dMapk1=rd[m.idx["Mapk1"]].item(),
        **gains, gains_negative=int(sum(np.isfinite(v) and v < 0 for v in gains.values())),
        solver_residual_max=float(fwd.get("residual", np.nan)), solver_converged=bool(fwd.get("converged", False)),
        per_condition=pc.reset_index().to_dict(orient="records"),
        provenance=dict(config_sha256_16=_config_hash(cfg), python=platform.python_version(),
                        torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                        git="not a git repository", signal_rmax=SIGNAL_RMAX, ceiling_file=str(ceil_p)),
    )
    res["frac_of_ceiling_drug_signal"] = (res["r_tf_drug_resid_signal"] / res["ceiling_drug_rmax_signal"]
                                         if res["ceiling_drug_rmax_signal"] > 0 else float("nan"))
    missing = set(SCHEMA_KEYS) ^ set(res)
    if missing:
        raise AssertionError(f"schema mismatch: {sorted(missing)}")
    return res


def result_path(cfg, step, variant):
    return cfg.paths.output_dir / "harness" / f"{step}_{variant}.json"


def write_result(cfg, res):
    p = result_path(cfg, res["step"], res["variant"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    return p


# ---------- selection (mechanical, from the JSON files only) ----------
def default_eligibility(res, rho_max=0.95, transmit_min=0.002, route_min=None):
    """Fixed criteria for step-12j-style selections; returns (eligible, reasons).

    RETIRED 2026-10-19: the guard used to be `route_dSting1 >= 0.02`, which rejected any model
    that did not predict Sting1 UP, and strongly. That threshold was set when the PAR2 -> STING
    direction was assumed; PAR2_29 has since measured pSTING DOWN, so the guard was rejecting
    models for failing to make what is now the disputed prediction. It is replaced by a
    DIRECTION-FREE transmission check upstream of the dispute: the receptor signal must reach
    ERK at all (|route_dMapk1| >= transmit_min). That still rejects a dead or disconnected model,
    which is what the guard was for, without asserting anything about Sting1.

    `route_min` is accepted and ignored so old callers do not silently change meaning.
    """
    why = []
    if res["gains_negative"] > 0:
        why.append(f"negative fitted gain ({res['gain_drug']:+.3g} drug, {res['gain_geo']:+.3g} geo)")
    if not res["rho_absA"] <= rho_max + 1e-4:
        why.append(f"rho {res['rho_absA']:.3f} > {rho_max}")
    if not abs(res["route_dMapk1"]) >= transmit_min:
        why.append(f"route does not transmit: |dMapk1| {abs(res['route_dMapk1']):.4f} < {transmit_min}")
    if not res["solver_converged"]:
        why.append("solver did not converge")
    return not why, why


def select(paths, metric="r_tf_drug_resid_signal", eligibility=default_eligibility):
    """Pick the eligible result with the highest `metric`. Reads harness JSON only; refuses
    sealed holdout files and refuses to run once the sealed module has been imported."""
    global _SELECTING
    if "lembas_par2.eval_sealed" in sys.modules:
        raise RuntimeError("selection must not run in a process that imported lembas_par2.eval_sealed")
    _SELECTING = True
    try:
        table = []
        for p in map(Path, paths):
            if "sealed" in p.parts:
                raise RuntimeError(f"selection may not read sealed holdout results: {p}")
            r = json.loads(p.read_text())
            ok, why = eligibility(r)
            table.append(dict(variant=r["variant"], eligible=ok, reasons="; ".join(why), metric=r[metric]))
        t = pd.DataFrame(table)
        el = t[t.eligible & t.metric.notna()]
        chosen = el.sort_values("metric", ascending=False).variant.iloc[0] if len(el) else None
        return chosen, t
    finally:
        _SELECTING = False
