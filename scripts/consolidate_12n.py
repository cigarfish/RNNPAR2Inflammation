"""Step 12n: one comparison table across every scored variant (harness JSON only).

Reads outputs/harness/<step>_<variant>.json (never sealed/), the step-12f0 references, the
step-12h summaries, and the 12j selection / verdict / permutation files. Seeds of the same
variant (<ID>_s<k>) are aggregated as median [min, max].

    python scripts/consolidate_12n.py
Writes outputs/step12n/{comparison.csv, comparison.md, summary.json}; then:
    python tests/test_step12n_consolidation.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config

KEYS = ["r_tf_drug", "r_tf_drug_resid_signal", "r_tf_drug_signal", "r_tf_geo", "r_tf_geo_resid", "r_tf_val",
        "route_dSting1", "rho_absA", "rho_signedA", "delta_tf_median", "gain_drug", "gain_geo"]
LABEL = {
    "12f_stageA": "stage A V6 (drug only, route neighbourhood)",
    "12f_C4": "C4: no GEO (step 12f composition test)",
    "12f_C0": "C0: all GEO (step 12f composition test)",
    "12f_S5": "S5: frozen baseline (step 12e)",
    "12j_J0": "J0 = S5 readout, rerun",
    "12j_J1": "J1 + intercept drug",
    "12j_J2": "J2 + intercept drug, GEO",
    "12j_J3": "J3 + intercept drug, GEO + per-source gain",
    "12j_J4": "J4 = J3 + per-source balancing",
    "12j_J5": "J5 = J3 without GEO",
    "12j_J0U": "J0U = J0 unanchored",
    "12j_J3U": "J3U = J3 unanchored",
}


def base_of(stem):
    return re.sub(r"_s\d+$", "", stem)


def fmt(med, lo, hi, n):
    if not np.isfinite(med):
        return "—"
    return f"{med:+.3f}" if n == 1 else f"{med:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    cfg = load_config()
    o = cfg.paths.output_dir
    hd = o / "harness"
    files = sorted(p for p in hd.glob("*.json") if "sealed" not in p.parts)
    rows = []
    for p in files:
        r = json.loads(p.read_text())
        meta = r.get("checkpoint_meta") or {}
        rows.append(dict(run=p.stem, variant=base_of(p.stem), seed=meta.get("seed", 0),
                         mode=meta.get("mode", "anchored"), **{k: r[k] for k in KEYS}))
    runs = pd.DataFrame(rows)
    agg = []
    for v, g in runs.groupby("variant", sort=False):
        row = dict(variant=v, label=LABEL.get(v, v), n_seeds=len(g), mode=g["mode"].iloc[0])
        for k in KEYS:
            x = g[k].astype(float)
            row[f"{k}_median"], row[f"{k}_min"], row[f"{k}_max"] = x.median(), x.min(), x.max()
        agg.append(row)
    agg = pd.DataFrame(agg)
    order = [k for k in LABEL if k in set(agg.variant)] + [v for v in agg.variant if v not in LABEL]
    agg = agg.set_index("variant").loc[order].reset_index()
    out = o / "step12n"
    out.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out / "comparison.csv", index=False)
    runs.to_csv(out / "runs.csv", index=False)

    f0 = json.loads((o / "step12f0" / "report.json").read_text())
    ref = dict(ceiling_signal=None, drug_B_mean=f0["baselines"]["drug:B_mean"]["val"],
               geo_B_mean=f0["baselines"]["geo:B_mean"]["val"], ceiling_val=f0["r_max_median_val"],
               shared_reference=f0["shared_reference"])
    any_res = json.loads(files[0].read_text())
    ref["ceiling_signal"] = any_res["ceiling_drug_rmax_signal"]

    def cell(v, k):
        r = agg[agg.variant == v].iloc[0]
        return fmt(r[f"{k}_median"], r[f"{k}_min"], r[f"{k}_max"], r.n_seeds)

    md = ["# Step 12n — consolidated comparison", "",
          f"Validation fold {cfg.stage_b['val_fold']}: 12 drugs (9 with noise ceiling r_max ≥ 0.3), 10 GEO contexts. "
          "Seeds aggregated as median [min, max]. Holdouts are sealed and not shown.", "",
          "## References (no model)", "",
          "| reference | value |", "|---|---|",
          f"| drug noise ceiling, signal drugs (r_max median) | {ref['ceiling_signal']:.3f} |",
          f"| drug: constant mean training-drug profile, raw r | {ref['drug_B_mean']:+.3f} |",
          f"| drug: mean of other drugs, shared vs disjoint DMSO | {ref['shared_reference']['r_shared_median']:+.3f} vs "
          f"{ref['shared_reference']['r_disjoint_median']:+.3f} |",
          f"| GEO: constant mean training-GEO profile, raw r | {ref['geo_B_mean']:+.3f} |",
          "| any constant profile, residualised r | 0 by construction |", "",
          "## All variants", "",
          "| variant | seeds | drug raw | **drug resid (signal)** | GEO raw | GEO resid | route ΔSting1 | ρ(\\|A\\|) | signed ρ |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v in agg.variant:
        r = agg[agg.variant == v].iloc[0]
        md.append(f"| {r.label} | {r.n_seeds} | {cell(v, 'r_tf_drug')} | {cell(v, 'r_tf_drug_resid_signal')} | "
                  f"{cell(v, 'r_tf_geo')} | {cell(v, 'r_tf_geo_resid')} | {cell(v, 'route_dSting1')} | "
                  f"{cell(v, 'rho_absA')} | {cell(v, 'rho_signedA')} |")

    pairs = [(a, b) for a, b in [("12j_J0", "12j_J0U"), ("12j_J3", "12j_J3U")] if {a, b} <= set(agg.variant)]
    md += ["", "## Anchored vs unanchored (honesty check for step 13)", ""]
    if pairs:
        md += ["| comparison | route ΔSting1 anchored | route ΔSting1 unanchored | drug resid (signal) anchored | unanchored | GEO resid anchored | unanchored |",
               "|---|---|---|---|---|---|---|"]
        for a, b in pairs:
            md.append(f"| {LABEL[a].split(' ')[0]} vs {LABEL[b].split(' ')[0]} | {cell(a, 'route_dSting1')} | {cell(b, 'route_dSting1')} | "
                      f"{cell(a, 'r_tf_drug_resid_signal')} | {cell(b, 'r_tf_drug_resid_signal')} | "
                      f"{cell(a, 'r_tf_geo_resid')} | {cell(b, 'r_tf_geo_resid')} |")
    else:
        md.append("_Pending: run J0U and J3U (`python scripts/run_source_variants.py --seeds 0 1 2 --variants J0U J3U`)._")

    extra = {}
    for f in ["selection.json", "verdict.json", "verdict_geo.json"]:
        p = o / "source_variants" / f
        if p.exists():
            extra[f] = json.loads(p.read_text())
    pn = o / "source_variants" / "permutation_null.csv"
    if pn.exists():
        d = pd.read_csv(pn)
        extra["permutation_null"] = d.to_dict(orient="records")
        md += ["", "## Drug-label permutation null (12j)", "",
               "| run | matched r | mismatched r | permutation p |", "|---|---|---|---|"]
        md += [f"| {r.run} | {r.matched_median:+.3f} | {r.mismatched_median:+.3f} | {r.perm_p:.3f} |" for r in d.itertuples()]
    for n in ["S5", "J3"]:
        p = o / "step12h" / f"{n}_summary.json"
        if p.exists():
            S = json.loads(p.read_text())
            extra[f"spectral_{n}"] = {k: S[k] for k in ["rho", "rho_signedA", "headroom_absA_over_signed",
                                                         "giant_scc_below_half_at_T3_fraction"]}
            extra[f"spectral_{n}"]["route_T0_loading"] = S["route_T0"]["loading_share"]
    md += ["", "## Verdicts", ""]
    if "verdict.json" in extra:
        md.append(f"* 12j: {extra['verdict.json']['verdict']}")
    if "verdict_geo.json" in extra:
        md.append(f"* J5: {extra['verdict_geo.json']['verdict']}")
    for n in ["S5", "J3"]:
        if f"spectral_{n}" in extra:
            s = extra[f"spectral_{n}"]
            md.append(f"* 12h {n}: headroom {s['headroom_absA_over_signed']:.2f}x, T0 route carries "
                      f"{s['route_T0_loading']:.1%} of ρ, giant SCC < 50 % at {s['giant_scc_below_half_at_T3_fraction']} of T3 cut")
    md.append("* 12k: dropped by the plan's rule on S5 (headroom 1.35 < 1.5), decided with the user 2026-10-16.")
    md.append("* 12l: not warranted (needs the giant SCC to break at < 5 % of T3 cut).")
    (out / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(dict(references=ref, variants=list(agg.variant),
                                                      n_runs=len(runs), **extra), indent=2, default=float))
    print("\n".join(md))
    print(f"\nwrote {out}\nnext: python tests/test_step12n_consolidation.py")


if __name__ == "__main__":
    main()
