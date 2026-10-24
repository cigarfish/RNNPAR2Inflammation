"""Step 12q: re-run the step-12o checks with MEASURED drug target profiles.

Target profiles (lembas_par2/kinome.py, Gualdrini 2024 Dataset EV1): occupancy at the dose
actually used, occ = c / (c + EC50) over every kinobeads target. Variants, fixed before running:
    union      (PRIMARY) measured occupancy, plus the annotated targets at occ 0.9 where the sheet
               has nothing higher. Every drug was dosed far above its textbook IC50, and kinobeads
               cannot see PI3K/mTOR and misses some high-affinity targets (e.g. BMS911543 lists
               only NQO2 at 5.8 uM), so measured-only under-states what is inhibited.
    measured   measured occupancy only
    annotated  the step-12o input (annotated targets at 1.0), on the same drug set
Checks (same metric, folds, null and thresholds as 12o; trametinib excluded - no profile):
    T  similarity of target profiles: a validation drug is predicted by the cosine-weighted
       mean residual profile of training drugs with positive profile similarity
    O  uncapped propagation oracle with delta = -occupancy at network nodes
Rules: O or T >= 0.15 with permutation p < 0.05 -> measured targets carry drug-specific signal
       -> re-specify the drug inputs in the model (step 12r). Both < 0.05 -> polypharmacology is
       not the explanation. Otherwise ambiguous.

    python scripts/run_kinome_12q.py
Writes outputs/step12q/{report.json, selected_vs_annotated.csv, occupancy.csv}.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.eval_harness import SIGNAL_RMAX, load_model
from lembas_par2.genes import symbol_map
from lembas_par2.kinome import occupancy_matrix, selected_vs_annotated
from lembas_par2.loss import Targets
from lembas_par2.oracle import ALPHAS, KMAX, adjacency, hops, perm_p, predict, resid_scores

UNION_OCC = 0.9


def similarity_predict(Y, X, train_cols, eval_cols):
    """Cosine-weighted mean of training residual profiles (positive similarities only)."""
    with np.errstate(all="ignore"):
        my = np.nanmean(Y[:, train_cols], axis=1)
    R = Y - my[:, None]
    nrm = np.linalg.norm(X, axis=1)
    P = np.full_like(Y, np.nan)
    for j in eval_cols:
        s = np.array([X[i] @ X[j] / (nrm[i] * nrm[j]) if nrm[i] > 0 and nrm[j] > 0 else 0.0 for i in train_cols])
        keep = s > 0
        if keep.any():
            P[:, j] = np.nansum(R[:, np.array(train_cols)[keep]] * s[keep], axis=1) / s[keep].sum()
    return P


def main():
    cfg = load_config()
    o = cfg.paths.output_dir
    out = o / "step12q"
    out.mkdir(parents=True, exist_ok=True)
    sva = selected_vs_annotated(cfg)
    sva.to_csv(out / "selected_vs_annotated.csv", index=False)

    m, _ = load_model(cfg, o / "source_variants" / "J3" / "checkpoint.pt", "cpu")
    cond = m.design.cond
    T = Targets(cfg, m)
    Yall = np.where(T.tf_mask.cpu().numpy() > 0, T.tf.cpu().numpy(), np.nan).astype(float)
    rmax = pd.read_csv(o / "step12f0" / "ceiling.csv").set_index("condition_id").r_max
    ids = list(cond.condition_id)
    occ = occupancy_matrix(cfg, symbol_map(cfg))
    drugs = [i for i in range(len(cond)) if cond.source.iloc[i] == "drug" and ids[i] in occ.index]
    vf = cfg.stage_b["val_fold"]
    Y = Yall[:, drugs]
    folds = [int(cond.fold.iloc[i]) for i in drugs]
    sig = [float(rmax.get(ids[i], np.nan)) >= SIGNAL_RMAX for i in drugs]

    genes = sorted(set(occ.columns) | {t for i in drugs for t in cond.trt_inhibit.iloc[i]}
                   | {t for i in drugs for t in pd.read_csv(cfg.root / "curation" / "drug_targets.csv")
                      .set_index("drug").targets.get(cond.name.iloc[i], "").split(";") if t})
    gi = {g: k for k, g in enumerate(genes)}
    ann = pd.read_csv(cfg.root / "curation" / "drug_targets.csv").set_index("drug").targets
    variants = {}
    for v in ["union", "measured", "annotated"]:
        X = np.zeros((len(drugs), len(genes)))
        for j, i in enumerate(drugs):
            if v in ("union", "measured"):
                row = occ.loc[ids[i]]
                for g, val in row[row > 0].items():
                    X[j, gi[g]] = val
            if v in ("union", "annotated"):
                for t in ann.get(cond.name.iloc[i], "").split(";"):
                    if t:
                        X[j, gi[t]] = max(X[j, gi[t]], UNION_OCC if v == "union" else 1.0)
        variants[v] = X
    pd.DataFrame(variants["union"], index=[ids[i] for i in drugs], columns=genes).to_csv(out / "occupancy.csv")

    tf_idx = m.tf_idx.cpu().numpy()
    graphs = {g: adjacency(m, g, m.weights()) for g in ["curated", "curated_indeg", "trained"]}

    def cols(f):
        tr = [j for j in range(len(drugs)) if folds[j] != f]
        ev = [j for j in range(len(drugs)) if folds[j] == f and sig[j]]
        trs = [j for j in tr if sig[j]]
        return tr, ev, trs

    rep = dict(n_drugs=len(drugs), selected_in_annotation=int(sva.selected_in_annotation.sum()),
               annotated_occ_ge_half=int(sva.annotated_occ_ge_half.sum()), n_sva=len(sva), variants={})
    for v, X in variants.items():
        delta = np.zeros((len(m.nodes), len(drugs)))
        for j in range(len(drugs)):
            for g, k in gi.items():
                if X[j, k] > 0 and g in m.idx:
                    delta[m.idx[g], j] = -X[j, k]
        H = {g: hops(A, delta) for g, A in graphs.items()}
        res = {}
        # T: profile similarity
        oofT = []
        for f in sorted(set(folds)):
            tr, ev, _ = cols(f)
            Pf = similarity_predict(Y, X, tr, ev)
            oofT += [s for s, j in zip(resid_scores(np.nan_to_num(Pf), Y, tr, ev), ev) if np.isfinite(Pf[:, j]).any()]
        tr, ev, trs = cols(vf)
        Pt = similarity_predict(Y, X, tr, ev)
        have = [k for k, j in enumerate(ev) if np.isfinite(Pt[:, j]).any()]
        sc = resid_scores(np.nan_to_num(Pt), Y, tr, ev)
        obs, off, p = perm_p(np.nan_to_num(Pt), Y, tr, ev)
        res["T"] = dict(val=float(np.nanmedian(sc[have])) if have else float("nan"), n_with_match=len(have),
                        perm_p=p, matched=obs, mismatched=off, oof=float(np.nanmedian(oofT)))

        # O: oracle, settings chosen on training drugs
        def fit(f):
            tr_, ev_, trs_ = cols(f)
            best = (-np.inf, None)
            for g, Hg in H.items():
                scale = [np.sqrt(np.nanmean(h[tf_idx][:, tr_] ** 2)) or 1.0 for h in Hg]
                for K in range(1, KMAX + 1):
                    for a in ALPHAS:
                        P = predict(Hg, tf_idx, a, K, scale)
                        s = float(np.nanmedian(resid_scores(P, Y, tr_, trs_)))
                        if s > best[0]:
                            best = (s, (g, K, a, P))
            return best, tr_, ev_
        (s_tr, (g, K, a, P)), tr, ev = fit(vf)
        obs, off, p = perm_p(P, Y, tr, ev)
        oofO = []
        for f in sorted(set(folds)):
            (_, (_, _, _, Pf)), trf, evf = fit(f)
            oofO += list(resid_scores(Pf, Y, trf, evf))
        res["O"] = dict(val=float(np.nanmedian(resid_scores(P, Y, tr, ev))), selected=dict(graph=g, K=K, alpha=a),
                        train_score=s_tr, perm_p=p, matched=obs, mismatched=off, oof=float(np.nanmedian(oofO)))
        rep["variants"][v] = res

    u = rep["variants"]["union"]
    best = max(((u[c]["val"], u[c]["perm_p"]) for c in ["T", "O"]), key=lambda t: t[0])
    if any(u[c]["val"] >= 0.15 and u[c]["perm_p"] < 0.05 for c in ["T", "O"]):
        verdict = "MEASURED TARGETS CARRY DRUG-SPECIFIC SIGNAL -> re-specify the drug inputs (step 12r)"
    elif all(u[c]["val"] < 0.05 or u[c]["perm_p"] >= 0.05 for c in ["T", "O"]) and best[0] < 0.05:
        verdict = "POLYPHARMACOLOGY IS NOT THE EXPLANATION: measured profiles do not predict drug-specific responses either"
    else:
        verdict = "AMBIGUOUS: report, do not build"
    rep["verdict"] = verdict
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))

    print(f"[12q] {len(drugs)} drugs with EV1 profiles; validation: {len(ev)} signal drugs (fold {vf})")
    print(f"  dosed-for kinase in our annotation: {rep['selected_in_annotation']}/{rep['n_sva']}; "
          f"any annotated target at occupancy >= 0.5: {rep['annotated_occ_ge_half']}/{rep['n_sva']}")
    print(f"\n  {'variant':10s} {'T val':>7s} {'p':>6s} {'oof':>7s} | {'O val':>7s} {'p':>6s} {'oof':>7s}  O setting")
    for v, r in rep["variants"].items():
        print(f"  {v:10s} {r['T']['val']:+7.3f} {r['T']['perm_p']:6.3f} {r['T']['oof']:+7.3f} | "
              f"{r['O']['val']:+7.3f} {r['O']['perm_p']:6.3f} {r['O']['oof']:+7.3f}  {r['O']['selected']}")
    print(f"\n[VERDICT] {verdict}\nwrote {out}")


if __name__ == "__main__":
    main()
