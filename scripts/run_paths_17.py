"""Step 17: signed curated paths F2rl1 -> Sting1 / Ifnb1. Rules: config `paths_test`, fixed before
running; method in lembas_par2/paths.py. Read-only, no training. Torch is used only to read the
LEARNED signs for the unsigned Ifnb1 regulators; without it the 'all' set falls back to curated.

    python scripts/run_paths_17.py                 # ~1-3 min on CPU
    python scripts/run_paths_17.py --no-null       # skip the sign permutations
Writes outputs/step17/{summary.csv, regulators.csv, paths_<layer>_<endpoint>.csv.gz, report.json}.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.paths import (enumerate_paths, ifnb1_regulators, learned_readout_signs, mrna_paths,
                               path_edge_matrix, sign_null, signed_adjacency, summarise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--permutations", type=int)
    a = ap.parse_args()
    cfg = load_config()
    pt = cfg.paths_test
    out = cfg.paths.output_dir / "step17"
    out.mkdir(parents=True, exist_ok=True)
    n_perm = a.permutations or pt["permutations"]

    learned = learned_readout_signs(cfg)
    regs = ifnb1_regulators(cfg, learned)
    regs.to_csv(out / "regulators.csv", index=False)
    n_cur = int((regs.curated_sign != 0).sum())
    print(f"[17] Ifnb1 readout regulators: {len(regs)} total, {n_cur} with a curated sign, "
          f"{len(regs) - n_cur} unsigned (learned sign used in the 'all' set)")
    uns = regs[regs.curated_sign == 0]
    if len(uns):
        print("     unsigned: " + ", ".join(f"{r.tf} (learned {r.learned_sign:+d})" for r in uns.itertuples()))

    rows, reports = [], {}
    for lname, spec in pt["layers"].items():
        adj, n_edges = signed_adjacency(cfg, spec["tiers"])
        tf_targets = set(regs.tf) | {"Sting1", "Ifnb1"}
        t0 = time.time()
        df, explored, capped, elists = enumerate_paths(adj, tf_targets, spec["max_depth"], return_edges=True)
        M = path_edge_matrix(elists, n_edges)
        print(f"\n[{lname}] {n_edges} signed edges, depth <= {spec['max_depth']}: "
              f"{len(df)} paths to any endpoint, {explored} explored{' (CAP HIT)' if capped else ''} "
              f"[{time.time() - t0:.1f}s]")

        # each endpoint is a row subset of the enumerated path set, so the null re-signs the SAME
        # paths (one sparse product per permutation) instead of re-enumerating the graph
        ends = {}
        for ename, mask in [("Sting1_signal", (df.target == "Sting1").values),
                            ("Ifnb1_signal", (df.target == "Ifnb1").values)]:
            ends[ename] = (df[mask], M[mask], None)
        for rs in pt["regulator_sets"]:
            col = "curated_sign" if rs == "curated" else "sign_used_all"
            e = mrna_paths(df, regs, col)
            km = e.attrs["keep_mask"]
            # the readout sign flips a path's net sign, so fold it into the parity as a constant
            flip = (e.readout_sign.values < 0).astype(np.int8)
            ends[f"Ifnb1_mrna_{rs}"] = (e, M[km], flip)
        for ename, (e, Me, flip) in ends.items():
            s = summarise(e, ename)
            s.update(layer=lname, n_signed_edges=n_edges, capped=capped)
            if len(e) and not a.no_null:
                null = sign_null(Me, n_perm, extra_parity=flip)
                s["null_frac_up_median"] = float(np.nanmedian(null))
                s["null_sd"] = float(np.nanstd(null))
                # BOTH sides: a one-sided 'up' p of 1.0 means a significant lean DOWN, which was
                # misread once (see HANDOFF step 17, correction 2026-10-18)
                s["p_up"] = float(np.nanmean(null >= s["frac_up"]))
                s["p_down"] = float(np.nanmean(null <= s["frac_up"]))
                s["direction"] = ("up" if s["p_up"] < 0.05 else "down" if s["p_down"] < 0.05 else "none")
            rows.append(s)
            e.drop(columns=[c for c in ["readout_sign"] if c in e], errors="ignore").to_csv(
                out / f"paths_{lname}_{ename}.csv.gz", index=False)
            print(f"  {ename:22s} {s['n_paths']:6d} paths | predict UP {s['n_predict_up']:6d} "
                  f"({s['frac_up']:.1%}) | depth {s['depth_min']}-{s['depth_max']}"
                  + (f" | null {s['null_frac_up_median']:.1%}+-{s['null_sd']:.1%}"
                     f" p(up) {s['p_up']:.3f} p(down) {s['p_down']:.3f} -> favours {s['direction'].upper()}"
                     if "p_up" in s else ""))

        # does any UP path to Ifnb1 avoid the STING arm?
        ifn = ends["Ifnb1_signal"][0]
        if len(ifn):
            up = ifn[ifn.net_sign < 0]
            bypass = up[~up.path.str.contains("Sting1")]
            reports[lname] = dict(n_up=len(up), n_up_bypassing_STING=len(bypass),
                                  shortest_bypass=(bypass.sort_values("depth").path.iloc[0] if len(bypass) else None),
                                  shortest_up=(up.sort_values("depth").path.iloc[0] if len(up) else None))
            print(f"  Ifnb1-up paths avoiding Sting1: {len(bypass)} of {len(up)}"
                  + (f"; shortest: {reports[lname]['shortest_bypass']}" if len(bypass) else ""))

    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)
    (out / "report.json").write_text(json.dumps(dict(rules=dict(pt), n_regulators=len(regs),
                                                     n_curated=n_cur, summary=rows, bypass=reports,
                                                     learned_signs=learned), indent=2, default=float))
    print("\n== summary (frac_up = fraction of paths predicting the target UP on PAR2 loss)")
    cols = ["layer", "label", "n_paths", "n_predict_up", "frac_up"] + \
           [c for c in ["null_frac_up_median", "null_sd", "p_up", "p_down", "direction"] if c in s]
    print(s[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {out}\nnext: python tests/test_step17_paths.py")


if __name__ == "__main__":
    main()
