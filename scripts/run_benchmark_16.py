"""Step 16: method comparison. Rules in config `benchmark`, fixed before running; method details in
lembas_par2/benchmark.py. Needs scikit-learn; does NOT need torch.

    python scripts/run_benchmark_16.py                 # all methods, ~1-5 min on CPU
    python scripts/run_benchmark_16.py --methods ridge gradient_boosting
    python scripts/run_benchmark_16.py --no-oof        # skip the 5-fold secondary table
Writes outputs/step16/{summary.csv, per_drug.csv, oof.csv, oof_per_drug.csv,
                       paired_vs_lembas.csv, paired_vs_lembas_oof.csv,
                       report.json}; then:
    python tests/test_step16_benchmark.py
"""
import argparse
import json
import sys
import time

NL = chr(10)
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.benchmark import lembas_oof_scores, METHODS, drug_table, lembas_scores, permutation_p, score, sensitivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--no-oof", action="store_true")
    ap.add_argument("--no-sensitivity", action="store_true", help="skip the planted-signal power check")
    ap.add_argument("--seed", type=int, default=0)
    # The registered default is `benchmark.lembas_variant` (J0, the frozen model). This flag
    # benchmarks a DIFFERENT variant without editing that registered setting, so the frozen
    # comparison stays reproducible and the current model can be reported beside it. Added
    # 2026-10-20 for the post-step-41 figure rebuild; pass --lembas-variant R4 for the current
    # tree. Results land in a suffixed directory so neither run overwrites the other.
    ap.add_argument("--lembas-variant", default=None,
                    help="override benchmark.lembas_variant (default: the config value)")
    a = ap.parse_args()
    cfg = load_config()
    bc = dict(cfg.benchmark)
    lv_default = bc["lembas_variant"]
    if a.lembas_variant:
        bc["lembas_variant"] = a.lembas_variant
    out = cfg.paths.output_dir / ("step16" if not a.lembas_variant
                                  else f"step16_{a.lembas_variant}")
    out.mkdir(parents=True, exist_ok=True)
    if a.lembas_variant and a.lembas_variant != lv_default:
        print(f"[16] LEMBAS variant OVERRIDDEN: {lv_default} (registered) -> "
              f"{a.lembas_variant}; writing to {out.name}/ so the registered run is not "
              f"overwritten. The non-LEMBAS methods are unaffected by this flag.")

    X, Y, folds, ids, rmax = drug_table(cfg)
    vf = bc["primary_fold"]
    test = [i for i in range(len(ids)) if folds[i] == vf and rmax[i] >= bc["signal_rmax"]]
    test_all = [i for i in range(len(ids)) if folds[i] == vf]
    train = [i for i in range(len(ids)) if folds[i] != vf]
    print(f"[16] {len(ids)} drugs with a kinome profile, {X.shape[1]} target features, {Y.shape[0]} TF outputs")
    print(f"     primary fold {vf}: {len(test)} signal drugs of {len(test_all)}; training on {len(train)}")
    print(f"     noise ceiling on the signal drugs: {np.median(rmax[test]):.3f}\n")

    rows, per_drug = [], []
    for name in a.methods:
        t0 = time.time()
        P, info = METHODS[name](cfg, X, Y, train, test, a.seed)
        r_res = score(P, Y, train, test)
        r_raw = score(P, Y, train, test, resid=False)
        obs, mism, p = permutation_p(P, Y, train, test, bc["permutations"], a.seed)
        sens = np.nan
        if not a.no_sensitivity:
            sens, _ = sensitivity(cfg, name, X, Y, train, test, seed=a.seed)
        rows.append(dict(method=name, resid_r=float(np.nanmedian(r_res)), raw_r=float(np.nanmedian(r_raw)),
                         perm_p=p, sensitivity=sens, mismatched_r=mism,
                         seconds=round(time.time() - t0, 1), info=json.dumps(info)))
        per_drug += [dict(method=name, condition_id=ids[j], r_max=rmax[j], resid_r=v)
                     for j, v in zip(test, r_res)]
        print(f"  {name:24s} resid r {rows[-1]['resid_r']:+.3f} (perm p {p:.3f})  raw r {rows[-1]['raw_r']:+.3f}"
              f"  power {sens:+.3f}  [{rows[-1]['seconds']:.1f}s] {info}")

    # LEMBAS, from its frozen harness results (fold 0, 5 seeds)
    L = lembas_scores(cfg, bc["lembas_variant"], bc["lembas_seeds"])
    if len(L):
        sig = L[L.r_max >= bc["signal_rmax"]]
        med = sig.groupby("seed").r_resid.median()
        lbl = f"LEMBAS {bc['lembas_variant']}" + (" (frozen)" if bc["lembas_variant"] == lv_default else " (current tree)")
        rows.append(dict(method=lbl, resid_r=float(med.median()),
                         raw_r=float(sig.groupby("seed").r.median().median()), perm_p=np.nan, mismatched_r=np.nan,
                         sensitivity=np.nan, seconds=0.0,
                         info=json.dumps(dict(seeds=len(med), resid_r_range=[float(med.min()), float(med.max())]))))
        per_drug += [dict(method="LEMBAS", condition_id=c, r_max=float(g.r_max.iloc[0]), resid_r=float(g.r_resid.median()))
                     for c, g in sig.groupby("condition_id")]
        print(f"\n  {lbl:24s} resid r {rows[-1]['resid_r']:+.3f} "
              f"(5 seeds {med.min():+.3f} to {med.max():+.3f})  raw r {rows[-1]['raw_r']:+.3f}")

    s = pd.DataFrame(rows).sort_values("resid_r", ascending=False)
    s.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(per_drug).to_csv(out / "per_drug.csv", index=False)

    # secondary: 5-fold out-of-fold for the learners (not available for LEMBAS)
    oof, oof_per_drug = [], []
    if not a.no_oof:
        print("\n[16] secondary: 5-fold out-of-fold (learners only)")
        for name in a.methods:
            vals = []
            for f in sorted(set(folds)):
                te = [i for i in range(len(ids)) if folds[i] == f and rmax[i] >= bc["signal_rmax"]]
                tr = [i for i in range(len(ids)) if folds[i] != f]
                if not te:
                    continue
                P, _ = METHODS[name](cfg, X, Y, tr, te, a.seed)
                v = list(score(P, Y, tr, te))
                vals += v
                oof_per_drug += [dict(method=name, condition_id=ids[j], r_max=rmax[j],
                                      fold=int(f), resid_r=val) for j, val in zip(te, v)]
            oof.append(dict(method=name, oof_resid_r=float(np.nanmedian(vals)), n=len(vals)))
            print(f"  {name:24s} {oof[-1]['oof_resid_r']:+.3f} over {oof[-1]['n']} drugs")
        # LEMBAS's own out-of-fold values, from the step-20 CV runs: every drug scored by the
        # model that held its fold out, so the pairing below compares like with like.
        # NOTE the out-of-fold LEMBAS is the step-20 CV variant (cv_test.variants[0]), NOT the
        # frozen `benchmark.lembas_variant`: the frozen model was trained once with fold 0 held
        # out and cannot be cross-validated without retraining, and its checkpoints no longer
        # load in the current tree. The label below says so, because the two are different
        # models and the comparison must not be read as if the frozen one had been rotated.
        cvv = list(cfg.cv_test["variants"])[0]
        LO = lembas_oof_scores(cfg, cvv, tuple(cfg.cv_test["seeds"]))
        if len(LO):
            sg = LO[LO.r_max >= bc["signal_rmax"]]
            oof.append(dict(method="LEMBAS",
                            oof_resid_r=float(sg.groupby("seed").r_resid.median().median()),
                            n=int(sg.condition_id.nunique())))
            oof_per_drug += [dict(method="LEMBAS", condition_id=c, r_max=float(g.r_max.iloc[0]),
                                  fold=int(g.fold.iloc[0]), resid_r=float(g.r_resid.median()))
                             for c, g in sg.groupby("condition_id")]
            print(f"  {'LEMBAS ' + cvv + ' (step-20 CV)':24s} {oof[-1]['oof_resid_r']:+.3f} "
                  f"over {oof[-1]['n']} drugs")
        else:
            print("  LEMBAS out-of-fold: no step-20 runs found; run scripts/run_cv_20.py")
        pd.DataFrame(oof).to_csv(out / "oof.csv", index=False)
        pd.DataFrame(oof_per_drug).to_csv(out / "oof_per_drug.csv", index=False)

    # paired per-drug comparison against LEMBAS: comparing medians of the summary table is not a
    # test, and the per-drug pairing can reverse the apparent ranking (it does here).
    # which methods beat chance on the primary fold, and which had the power to (restored
    # 2026-10-18: this block was deleted by mistake when the paired test was rewritten, and
    # rebuilt from `benchmark.beats_chance` plus the previous run's report.json, which it
    # reproduces exactly)
    bch = bc["beats_chance"]
    winners = [x.method for x in s.itertuples()
               if not str(x.method).startswith("LEMBAS")
               and float(x.resid_r) >= bch["min_resid_r"]
               and np.isfinite(x.perm_p) and float(x.perm_p) < bch["max_perm_p"]]
    powered = [x.method for x in s.itertuples()
               if np.isfinite(x.sensitivity) and float(x.sensitivity) >= bch["min_resid_r"]]
    if winners:
        verdict = ("BEATEN ON PREDICTION: " + ", ".join(winners) + " beat chance on the primary "
                   "fold; what carries the mechanism is a separate decision (config benchmark.verdict)")
        gnn = "a GNN may be worth building (a simple learner beat chance; config benchmark.gnn)"
    else:
        verdict = ("DATA LIMIT: no method tested extracts drug-specific signal from this screen "
                   "(every residualised r below " + f"{bch['min_resid_r']}" + " or permutation p >= "
                   + f"{bch['max_perm_p']}" + "). Methods with demonstrated power to detect planted "
                   "signal at the noise ceiling: " + (", ".join(powered) or "none")
                   + "; the others are underpowered at this sample size and their null result is weaker.")
        gnn = "do NOT build a GNN (no simple learner beat chance; config benchmark.gnn)"

    def pair(frame, label, fname):
        rowsp = []
        pdw = pd.DataFrame(frame).pivot_table(index="condition_id", columns="method",
                                              values="resid_r")
        if "LEMBAS" not in pdw:
            print(f"{NL}[skip] {label}: no LEMBAS column")
            return []
        from scipy.stats import wilcoxon
        for m in [c for c in pdw.columns if c != "LEMBAS"]:
            both = pdw[[m, "LEMBAS"]].dropna()
            try:
                w = float(wilcoxon(both[m], both["LEMBAS"]).pvalue)
            except Exception:
                w = float("nan")
            rowsp.append(dict(method=m, n=len(both),
                              median_diff_vs_lembas=float((both[m] - both["LEMBAS"]).median()),
                              wins=int((both[m] > both["LEMBAS"]).sum()), wilcoxon_p=w))
        pdf = pd.DataFrame(rowsp).sort_values("median_diff_vs_lembas", ascending=False)
        pdf.to_csv(out / fname, index=False)
        print(f"{NL}== {label}")
        print(pdf.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
        better = [x.method for x in pdf.itertuples()
                  if x.median_diff_vs_lembas > 0 and x.wilcoxon_p < 0.05]
        print(f"   better than LEMBAS per drug (p < 0.05): {better or 'none'}")
        return rowsp

    # Two paired tests. The fold-0 one is kept for continuity with what has been reported; the
    # OUT-OF-FOLD one is the comparable test and the one to quote, because every method is then
    # scored on all 43 signal drugs by a model that did not see them.
    paired = pair(per_drug, "paired per-drug comparison, FOLD 0 ONLY (n = 9 drugs)",
                  "paired_vs_lembas.csv")
    paired_oof = (pair(oof_per_drug, "paired per-drug comparison, OUT-OF-FOLD (quote this one)",
                       "paired_vs_lembas_oof.csv") if oof_per_drug else [])
    rep = dict(paired_vs_lembas=paired, paired_vs_lembas_oof=paired_oof, powered_methods=powered, n_drugs=len(ids), n_features=int(X.shape[1]), n_tf=int(Y.shape[0]), primary_fold=vf,
               n_signal_drugs=len(test), ceiling=float(np.median(rmax[test])), rules=dict(bc),
               summary=s.to_dict(orient="records"), oof=oof, verdict=verdict, gnn_decision=gnn)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print("\n== summary (primary fold, signal drugs)")
    print(s[["method", "resid_r", "raw_r", "perm_p", "sensitivity"]].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("  sensitivity = what the method recovers when a linear response to the SAME inputs is planted\n"
          "                and noised down to the measured ceiling (0.55); low = underpowered, not 'no signal'")
    print(f"\n[VERDICT] {verdict}\n[GNN] {gnn}\nwrote {out}\nnext: python tests/test_step16_benchmark.py")


if __name__ == "__main__":
    main()
