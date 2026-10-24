"""Step 12o: network-propagation oracle and target-identity baseline (see lembas_par2/oracle.py
for the method and the decision rules, fixed before running).

    python scripts/run_oracle_12o.py
Writes outputs/step12o/{report.json, oracle_grid.csv, per_drug.csv}; then:
    python tests/test_step12o_oracle.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.eval_harness import SIGNAL_RMAX, load_model
from lembas_par2.loss import Targets
from lembas_par2.oracle import (ALPHAS, KMAX, adjacency, hops, perm_p, predict, resid_scores,
                                target_identity)


def main():
    cfg = load_config()
    o = cfg.paths.output_dir
    out = o / "step12o"
    out.mkdir(parents=True, exist_ok=True)
    m, _ = load_model(cfg, o / "source_variants" / "J3" / "checkpoint.pt", "cpu")
    cond = m.design.cond
    T = Targets(cfg, m)
    Y_all = np.where(T.tf_mask.cpu().numpy() > 0, T.tf.cpu().numpy(), np.nan).astype(float)
    rmax = pd.read_csv(o / "step12f0" / "ceiling.csv").set_index("condition_id").r_max
    drugs = [i for i in range(len(cond)) if cond.source.iloc[i] == "drug"]
    ids = list(cond.condition_id)
    fold = {i: int(cond.fold.iloc[i]) for i in drugs}
    signal = {i for i in drugs if float(rmax.get(ids[i], np.nan)) >= SIGNAL_RMAX}
    vf = cfg.stage_b["val_fold"]
    col = {i: j for j, i in enumerate(drugs)}           # columns of the drug-only matrices
    Y = Y_all[:, drugs]
    targets = [set(cond.trt_inhibit.iloc[i]) for i in drugs]
    classes = [cond.cls.iloc[i] for i in drugs]
    n = len(m.nodes)
    delta = np.zeros((n, len(drugs)))
    for j, i in enumerate(drugs):
        for t in cond.trt_inhibit.iloc[i]:
            delta[m.idx[t], j] = -1.0
    tf_idx = m.tf_idx.cpu().numpy()

    graphs = {g: hops(adjacency(m, g, m.weights()), delta) for g in ["curated", "curated_indeg", "trained"]}

    def split(f):
        tr = [col[i] for i in drugs if fold[i] != f]
        ev = [col[i] for i in drugs if fold[i] == f and i in signal]
        tr_sig = [col[i] for i in drugs if fold[i] != f and i in signal]
        return tr, ev, tr_sig

    def fit_and_score(f, record=None):
        """Choose (graph, K, alpha) on fold-f training drugs (inner score: training signal
        drugs residualised against the training mean); return the eval predictions."""
        tr, ev, tr_sig = split(f)
        best = (-np.inf, None)
        for g, H in graphs.items():
            scale = [np.sqrt(np.nanmean(h[tf_idx][:, tr] ** 2)) or 1.0 for h in H]
            for K in range(1, KMAX + 1):
                for a in ALPHAS:
                    P = predict(H, tf_idx, a, K, scale)
                    s_tr = float(np.nanmedian(resid_scores(P, Y, tr, tr_sig)))
                    s_ev = float(np.nanmedian(resid_scores(P, Y, tr, ev)))
                    if record is not None:
                        record.append(dict(fold=f, graph=g, K=K, alpha=a, train_score=s_tr, val_score=s_ev))
                    if s_tr > best[0]:
                        best = (s_tr, (g, K, a, P))
        return best, tr, ev

    # ---- O: oracle on the validation fold (+ the per-graph table for transparency)
    grid = []
    (s_tr, (g, K, a, P)), tr, ev = fit_and_score(vf, grid)
    o_val = float(np.nanmedian(resid_scores(P, Y, tr, ev)))
    o_obs, o_off, o_p = perm_p(P, Y, tr, ev)
    gt = pd.DataFrame(grid)
    gt.to_csv(out / "oracle_grid.csv", index=False)
    per_graph = (gt[gt.fold == vf].sort_values("train_score", ascending=False).groupby("graph").head(1)
                 .set_index("graph")[["K", "alpha", "train_score", "val_score"]])
    # out-of-fold: each fold selected on its own training drugs
    oof = []
    for f in sorted(set(fold.values())):
        (_, (_, _, _, Pf)), trf, evf = fit_and_score(f)
        oof += list(resid_scores(Pf, Y, trf, evf))
    o_oof = float(np.nanmedian(oof))

    # ---- T: target identity
    t_res = {}
    for mode in ["target", "class"]:
        Pt = target_identity(Y, targets, classes, tr, ev, mode)
        sc = resid_scores(np.nan_to_num(Pt, nan=0.0), Y, tr, ev)   # mean residual is already centred
        have = [k for k, j in enumerate(ev) if np.isfinite(Pt[:, j]).any()]
        obs, off, p = perm_p(np.nan_to_num(Pt, nan=0.0), Y, tr, ev)
        oofm = []
        for f in sorted(set(fold.values())):
            trf, evf, _ = split(f)
            Pf = target_identity(Y, targets, classes, trf, evf, mode)
            oofm += [s for s, j in zip(resid_scores(np.nan_to_num(Pf, nan=0.0), Y, trf, evf), evf)
                     if np.isfinite(Pf[:, j]).any()]
        t_res[mode] = dict(val=float(np.nanmedian(sc[have])) if have else float("nan"), n_with_match=len(have),
                           n_eval=len(ev), matched=obs, mismatched=off, perm_p=p, oof=float(np.nanmedian(oofm)))

    # ---- per-drug table
    rows = []
    for k, j in enumerate(ev):
        i = drugs[j]
        rows.append(dict(condition_id=ids[i], r_max=float(rmax.get(ids[i])), cls=classes[j],
                         targets=";".join(sorted(targets[j])), oracle=resid_scores(P, Y, tr, [j])[0]))
    pd.DataFrame(rows).to_csv(out / "per_drug.csv", index=False)

    # ---- verdict (rules fixed in lembas_par2/oracle.py)
    def eff(v, p):
        return v if p < 0.05 else min(v, 0.0499)
    O = eff(o_val, o_p)
    Tt = max(eff(t_res["target"]["val"], t_res["target"]["perm_p"]), eff(t_res["class"]["val"], t_res["class"]["perm_p"]))
    if O >= 0.15:
        verdict = "TRANSMISSION IS WORTH FIXING: curated topology carries drug-specific signal without the cap"
    elif O < 0.05 and Tt >= 0.15:
        verdict = "GRAPH/KNOWLEDGE LIMIT: target identity is informative but curated paths do not carry it; a transmission fix will not help"
    elif O < 0.05 and Tt < 0.05:
        verdict = "IDENTIFIABILITY / DATA LIMIT: neither uncapped propagation nor target identity predicts drug-specific responses; a transmission fix cannot help"
    else:
        verdict = "AMBIGUOUS: report, do not build"
    rep = dict(oracle=dict(val=o_val, selected=dict(graph=g, K=K, alpha=a, train_score=s_tr), matched=o_obs,
                           mismatched=o_off, perm_p=o_p, oof=o_oof, per_graph=per_graph.reset_index().to_dict(orient="records")),
               target_identity=t_res, model_J3_reference=0.063, ceiling=float(np.median([rmax[ids[drugs[j]]] for j in ev])),
               effective=dict(O=O, T=Tt), verdict=verdict, n_val_signal=len(ev))
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))

    print(f"[12o] validation: {len(ev)} signal drugs (fold {vf}); ceiling median {rep['ceiling']:.3f}; J3 model 0.063")
    print("\nO  oracle, best setting per graph (chosen on training drugs):")
    print(per_graph.to_string(float_format=lambda x: f"{x:+.3f}"))
    print(f"   selected {g}, K={K}, alpha={a}: val resid r {o_val:+.3f} (matched {o_obs:+.3f} vs mismatched {o_off:+.3f}, "
          f"perm p {o_p:.3f}); out-of-fold {o_oof:+.3f}")
    for mode, d in t_res.items():
        print(f"T  target identity by {mode}: val resid r {d['val']:+.3f} on {d['n_with_match']}/{d['n_eval']} drugs with a match "
              f"(perm p {d['perm_p']:.3f}); out-of-fold {d['oof']:+.3f}")
    print(f"\n[VERDICT] {verdict}")
    print(f"wrote {out}\nnext: python tests/test_step12o_oracle.py")


if __name__ == "__main__":
    main()
