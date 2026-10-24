"""Step 13: PAR2-KO prediction, holdout scoring, sign-free probe, branch ablation, attribution.
Method and verdict rules: lembas_par2/step13.py and config `step13` (fixed before running).
Holdouts are read here for the first time; nothing in this step selects a model.

Train the missing seeds first (user, GPU; ~100 s per run):
    python scripts/run_source_variants.py --seeds 3 4 --variants J0 J0U J3 J3U
    python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants J0F J3F
Then:
    python scripts/run_step13.py                 # all available checkpoints + prior-only
    python scripts/run_step13.py --perm 20       # fewer route-sign permutations (smoke)
    python tests/test_step13.py
Writes outputs/step13/{report.json, runs.csv, holdout_observations.csv, ablation.csv,
                       attribution_top.csv, sign_null.csv}.
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
from lembas_par2.eval_harness import load_model
from lembas_par2.model import build_model
from lembas_par2.step13 import (ablation, binom_tail, branch_masks, flipped_route_edges, holdout_observations,
                                ko_attribution, ko_delta, score_holdouts)

VARIANTS = ["J0", "J3", "J0U", "J3U", "J0F", "J3F", "J0A", "J0AU", "J0AF", "R4"]
ROUTE_NODES = ["Mapk1", "Mapk3", "Mcl1", "Bcl2l11", "Bax", "Bak1", "SE_CYTO_MTDNA", "Cgas", "Sting1", "Irf3"]


def evaluate(cfg, m, run, variant, seed, obs, st, attrib=True):
    s, tab = score_holdouts(m, obs, st["call_threshold"], st["route_subset"])
    row = dict(run=run, variant=variant, seed=seed, holdout_k=s["k"], holdout_n=s["n"], holdout_p=s["p"],
               holdout_k_sign_only=s["k_sign_only"], route_k=s["route_k"], route_n=s["route_n"],
               **{f"holdout_dSting1_{c}": v for c, v in s["dSting1_holdouts"].items()})
    for ctx, inp in st["contexts"].items():
        d = ko_delta(m, inp, st["ko_kappa"])
        for n in ROUTE_NODES:
            row[f"ko_{ctx}_{n}"] = d[m.idx[n]].item()
    ab = ablation(m, st["branches"], "pooled", st["ko_kappa"])
    for k, v in ab.items():
        if k != "base":
            row[f"ablate_{k}_retained"] = v["retained"]
    fl = flipped_route_edges(m)
    if fl:
        row.update(n_unlocked=fl["n_unlocked"], n_flipped=fl["n_flipped"], n_flipped_T0=fl["n_flipped_T0"],
                   flipped=";".join(fl["flipped"]))
    top = None
    if attrib:
        d, contrib = ko_attribution(m, "Sting1", "pooled", st["ko_kappa"])
        e = m.edges
        masks = {k: v.cpu().numpy() for k, v in branch_masks(m, st["branches"]).items()}
        t0 = m.tier == "T0"
        row["attr_total_first_order"] = float(contrib.sum())
        row["attr_T0"] = float(contrib[t0].sum())
        row["attr_nonT0"] = float(contrib[~t0].sum())
        for k, mk in masks.items():
            if k != "all_four":
                row[f"attr_{k}"] = float(contrib[mk].sum())
        o = np.argsort(-np.abs(contrib))[:15]
        top = pd.DataFrame(dict(run=run, edge=[f"{a}->{b}" for a, b in zip(e.source.values[o], e.target.values[o])],
                                tier=m.tier[o], contribution=contrib[o]))
    tab.insert(0, "run", run)
    return row, tab, top


@torch.no_grad()
def sign_null(cfg, obs, st, n_perm, dev, seed=0):
    """Prior-only model with random signs on the T0 route edges (re-initialised each time)."""
    m = build_model(cfg, device=dev, seed=0)
    th_l, th_f, b = m.theta_locked.clone(), m.theta_free.clone(), m.bias.clone()
    sign0 = m.locked_sign.clone()
    lt = torch.tensor(m.tier == "T0", device=dev)[m.locked_idx]
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_perm):
        m.theta_locked.copy_(th_l)
        m.theta_free.copy_(th_f)
        m.bias.copy_(b)
        s = sign0.clone()
        s[lt] = torch.tensor(rng.choice([-1.0, 1.0], int(lt.sum())), dtype=s.dtype, device=dev)
        m.locked_sign.copy_(s)
        m.initialize()
        sc, _ = score_holdouts(m, obs, st["call_threshold"], st["route_subset"])
        rows.append(dict(perm=p, holdout_k=sc["k"], ko_pooled_dSting1=ko_delta(m, "pooled", st["ko_kappa"])[m.idx["Sting1"]].item()))
    m.locked_sign.copy_(sign0)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--perm", type=int, help="route-sign permutations (default config step13.sign_null_permutations)")
    ap.add_argument("--primary", help="primary variant for the verdict (default config step13.primary; "
                                      "J0A for the step-14 sensitivity analysis)")
    ap.add_argument("--out", default="step13", help="output folder under outputs/")
    ap.add_argument("--variants", nargs="*", help="restrict to these variants (default: all available)")
    ap.add_argument("--no-prior", action="store_true", help="skip the prior-only model")
    a = ap.parse_args()
    cfg = load_config()
    st = dict(cfg.step13)
    if a.primary:
        st["primary"] = a.primary
    o = cfg.paths.output_dir
    out = o / a.out
    out.mkdir(parents=True, exist_ok=True)
    obs = holdout_observations(cfg)
    rows, tabs, tops = [], [], []

    print(f"[13] {len(obs)} unique scored holdout observations; device {a.device}", flush=True)
    if not a.no_prior:
        m = build_model(cfg, device=a.device, seed=0)
        m.initialize()
        r, t, tp = evaluate(cfg, m, "prior_only", "prior_only", 0, obs, st)
        rows.append(r), tabs.append(t), tops.append(tp)
    for v in (a.variants or VARIANTS):
        for s in st["seeds"]:
            run = v if s == 0 else f"{v}_s{s}"
            ck = o / "source_variants" / run / "checkpoint.pt"
            if not ck.exists():
                continue
            m, _ = load_model(cfg, ck, a.device)
            r, t, tp = evaluate(cfg, m, run, v, s, obs, st)
            rows.append(r), tabs.append(t), tops.append(tp)
            print(f"  {run:8s} holdouts {r['holdout_k']}/{r['holdout_n']} (p {r['holdout_p']:.3f}); "
                  f"KO dSting1 pooled {r['ko_pooled_Sting1']:+.4f}, LPS {r['ko_lps_Sting1']:+.4f}"
                  + (f"; flipped route edges {r['n_flipped']}/{r['n_unlocked']} (T0 {r['n_flipped_T0']})"
                     if 'n_flipped' in r else ""), flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    pd.concat(tabs).to_csv(out / "holdout_observations.csv", index=False)
    pd.concat([x for x in tops if x is not None]).to_csv(out / "attribution_top.csv", index=False)
    abl = runs[["run", "variant"] + [c for c in runs.columns if c.startswith("ablate_")]]
    abl.to_csv(out / "ablation.csv", index=False)

    n_perm = a.perm if a.perm is not None else st["sign_null_permutations"]
    print(f"\n[13a] secondary null: {n_perm} random route-sign assignments (prior-only)", flush=True)
    nul = sign_null(cfg, obs, st, n_perm, a.device)
    nul.to_csv(out / "sign_null.csv", index=False)

    # ---------- aggregate + verdict ----------
    agg = {}
    for v, g in runs.groupby("variant"):
        agg[v] = dict(n_seeds=len(g), holdout_k_median=float(g.holdout_k.median()), holdout_n=int(g.holdout_n.iloc[0]),
                      holdout_k_range=[int(g.holdout_k.min()), int(g.holdout_k.max())],
                      ko_pooled_dSting1_median=float(g.ko_pooled_Sting1.median()),
                      ko_lps_dSting1_median=float(g.ko_lps_Sting1.median()),
                      route_k_median=float(g.route_k.median()), route_n=int(g.route_n.iloc[0]))
        agg[v]["holdout_p_at_median"] = binom_tail(int(np.ceil(agg[v]["holdout_k_median"])), agg[v]["holdout_n"])
        if "n_flipped" in g:
            agg[v]["n_flipped_median"] = float(g.n_flipped.median())
            agg[v]["n_flipped_T0_median"] = float(g.n_flipped_T0.median())
    prim, fprim = st["primary"], st["primary"] + "F"
    need = len(st["seeds"])
    complete = agg.get(prim, {}).get("n_seeds", 0) >= need and agg.get(fprim, {}).get("n_seeds", 0) >= need
    thr = st["reversal_threshold"]
    if not complete:
        verdict = (f"PENDING: need {need} seeds of {prim} and {fprim} "
                   f"(have {agg.get(prim, {}).get('n_seeds', 0)} / {agg.get(fprim, {}).get('n_seeds', 0)})")
    elif agg[fprim]["ko_pooled_dSting1_median"] < -thr:
        verdict = "CONTRADICTED: with free route signs the data push PAR2-KO dSting1 negative -> back to data"
    elif agg[prim]["holdout_p_at_median"] < st["holdout_alpha"]:
        verdict = "CONSISTENT: holdouts beat random signs and the data do not reverse the route"
    else:
        verdict = "WEAKENED: holdouts do not beat random signs (direction not reversed) -> back to data"
    pk = int(runs[runs.variant == "prior_only"].holdout_k.iloc[0])
    null_frac = float((nul.holdout_k >= agg.get(prim, {}).get("holdout_k_median", pk)).mean()) if len(nul) else np.nan
    rep = dict(n_observations=len(obs), variants=agg, verdict=verdict,
               sign_null=dict(n=len(nul), holdout_k_median=float(nul.holdout_k.median()) if len(nul) else None,
                              frac_ge_primary=null_frac,
                              frac_dSting1_positive=float((nul.ko_pooled_dSting1 > 0).mean()) if len(nul) else None),
               rules=dict(st))
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))

    print("\n== 13a/13b summary (median over seeds)")
    print(f"  {'variant':11s} {'seeds':>5s} {'holdouts':>9s} {'p':>6s} {'route':>6s} {'KO dSting1 pooled':>18s} {'LPS':>8s}  flips")
    for v in ["prior_only"] + VARIANTS:
        if v in agg:
            x = agg[v]
            print(f"  {v:11s} {x['n_seeds']:5d} {x['holdout_k_median']:5.1f}/{x['holdout_n']:<3d} {x['holdout_p_at_median']:6.3f} "
                  f"{x['route_k_median']:3.0f}/{x['route_n']:<2d} {x['ko_pooled_dSting1_median']:+18.4f} {x['ko_lps_dSting1_median']:+8.4f}  "
                  + (f"{x['n_flipped_median']:.0f} (T0 {x['n_flipped_T0_median']:.0f})" if 'n_flipped_median' in x else ""))
    print(f"  route-sign null (prior-only, {len(nul)} perms): holdout k median {rep['sign_null']['holdout_k_median']}, "
          f"P(k >= primary) {null_frac:.3f}; P(PAR2-KO dSting1 > 0) {rep['sign_null']['frac_dSting1_positive']:.2f}")
    print("\n== 13c branch ablation (share of PAR2-KO dSting1 retained, median over seeds)")
    print(abl.groupby("variant").median(numeric_only=True).to_string(float_format=lambda x: f"{x:+.2f}"))
    print(f"\n[VERDICT] {verdict}\nwrote {out}\nnext: python tests/test_step13.py")


if __name__ == "__main__":
    main()
