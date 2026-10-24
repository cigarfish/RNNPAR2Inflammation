"""Step 45: the Panel C table — model predictions for the PAR2_26 / 29 / 30 holdouts.

Each study is a DIFFERENT perturbation and must be run as one (protocols: prereg section 7c):

  PAR2_26  4T1 / Raw264.7, pure PAR2 KO          -> F2rl1 KO alone, NO TLR4 input
  PAR2_30  shPAR2 vs shCtrl, pure PAR2 KO        -> same as 26
  PAR2_29  albumin-delivered editing in vivo     -> F2rl1 KO + Tlr4 knockdown (the CARRIER)

Both contexts are printed for 26/30 (no LPS input, and LPS on) because that is a judgement call
and the difference should be visible rather than chosen silently.

LAYER MAPPING is not invented here: it is `conditions.readout_layers` from config.yaml, the same
mapping `lembas_par2/step13.py` scores with --
  signal : phosphoprotein, signal_activity, enzyme_activity, protein_cleavage,
           site_phosphorylation, total_protein
  mrna   : transcript, transcript/protein, secretion, transcript/secretion
A prediction with |value| < call_threshold (1e-3) is NO CALL and counts as a disagreement, which
is how step 13 scores it -- a correct sign below the threshold is not a success.

    python scripts/run_step45_panelC_table.py
    python scripts/run_step45_panelC_table.py --kappa-tlr4 0.77    # in-vivo TLR4 reduction
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.eval_harness import load_model
from lembas_par2.core import simulate

torch.set_grad_enabled(False)
STUDIES = ("PAR2_26", "PAR2_29", "PAR2_30")
SIGN = {"up": 1, "down": -1}


@torch.no_grad()
def deltas(m, cfg, inputs, kos):
    n = len(m.nodes)
    u = torch.zeros(n, 1, dtype=m.dtype)
    for x in inputs:
        u[m.idx[x], 0] = cfg.perturbation["input_level"]
    U = u.repeat(1, 2)
    S = torch.ones(n, 2, dtype=m.dtype)
    for node, kap in kos.items():
        S[m.idx[node], 1] = 1.0 - kap
    X = simulate(m.struct, m.weights(), m.bias, U, S, max_iter=3000, tol=1e-8)
    d = ((S * X)[:, 1] - (S * X)[:, 0])
    g = torch.zeros(len(m.genes), dtype=d.dtype).index_add(
        0, m.r_gene, m.readout_weights() * d[m.r_tf])
    return d, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="R4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--kappa-tlr4", type=float, default=0.60)
    a = ap.parse_args()

    cfg = load_config()
    thr = cfg.step13["call_threshold"]
    layers = cfg.conditions["readout_layers"]
    lps = cfg.step13["contexts"]["lps"]
    kap = cfg.step13["ko_kappa"]

    obs = pd.read_csv(ROOT / "data/perturbation/perturbation_observations_v0_2.csv")
    obs = obs[obs.contrast_id.str.startswith(STUDIES)].copy()
    obs["study"] = obs.contrast_id.str.split("_C").str[0]
    obs["layer"] = obs.readout_type.map(layers)
    obs["obs_sign"] = obs.effect_direction.map(SIGN)

    setups = {
        "26/30 pure PAR2 KO (no TLR4 input)": ([], {"F2rl1": kap}),
        "26/30 pure PAR2 KO (LPS on)": (lps, {"F2rl1": kap}),
        f"29 PAR2 KO + Tlr4 KD {a.kappa_tlr4}": (lps, {"F2rl1": kap, "Tlr4": a.kappa_tlr4}),
    }
    per_seed = {k: [] for k in setups}
    for s in a.seeds:
        run = a.variant if s == 0 else f"{a.variant}_s{s}"
        ck = cfg.paths.output_dir / "source_variants" / run / "checkpoint.pt"
        if not ck.exists():
            continue
        try:
            m, _ = load_model(cfg, ck, "cpu")
        except (RuntimeError, ValueError) as exc:
            sys.exit(f"{run}: checkpoint does not match the network ({exc}).\nRetrain first:\n"
                     f"  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
        gpos = {g: i for i, g in enumerate(m.genes)}
        for lab, (inp, kos) in setups.items():
            d, g = deltas(m, cfg, inp, kos)
            vals = {}
            for r in obs.itertuples():
                if r.layer == "signal" and r.graph_target_id in m.idx:
                    vals[r.observation_id] = float(d[m.idx[r.graph_target_id]])
                elif r.layer == "mrna" and r.graph_target_id in gpos:
                    vals[r.observation_id] = float(g[gpos[r.graph_target_id]])
                else:
                    vals[r.observation_id] = np.nan
            per_seed[lab].append(vals)
        print(f"  {run}: ok", flush=True)
    if not any(per_seed.values()):
        sys.exit("no checkpoints found. Train first:" 
                 "\n  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")

    out = cfg.paths.output_dir / "step45_panelC"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in obs.itertuples():
        lab = (f"29 PAR2 KO + Tlr4 KD {a.kappa_tlr4}" if r.study == "PAR2_29"
               else "26/30 pure PAR2 KO (no TLR4 input)")
        v = [sd[r.observation_id] for sd in per_seed[lab]]
        med = float(np.nanmedian(v))
        alt = float(np.nanmedian([sd[r.observation_id] for sd in
                                  per_seed["26/30 pure PAR2 KO (LPS on)"]])) \
            if r.study != "PAR2_29" else np.nan
        call = np.isfinite(med) and abs(med) >= thr
        agree = bool(call and np.isfinite(r.obs_sign) and np.sign(med) == r.obs_sign)
        # PAR2_29's Tlr4 is IMPOSED under this setup, so it cannot be scored as a prediction
        imposed = (r.study == "PAR2_29" and r.graph_target_id == "Tlr4")
        rows.append(dict(
            study=r.study, observation=r.observation_id, target=r.graph_target_id,
            readout=r.readout_type, layer=r.layer, figure=r.source_figure,
            observed=r.effect_direction, predicted=med, alt_LPS_on=alt,
            seeds_same_sign=int(np.nansum(np.sign(v) == np.sign(med))),
            call="yes" if call else "NO CALL",
            verdict=("INPUT, not a prediction" if imposed
                     else "unscored (direction unknown)" if not np.isfinite(r.obs_sign)
                     else "agree" if agree else "disagree")))
    T = pd.DataFrame(rows).sort_values(["study", "observation"])
    T.to_csv(out / "panelC_table.csv", index=False)

    pd.set_option("display.width", 250)
    for st in STUDIES:
        sub = T[T.study == st]
        print(f"\n===== {st} =====")
        print(sub[["observation", "target", "readout", "layer", "figure", "observed",
                   "predicted", "seeds_same_sign", "call", "verdict"]]
              .to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

    scored = T[T.verdict.isin(["agree", "disagree"])]
    # step 13 de-duplicates on (condition, node, layer); mirror that so the count matches
    dedup = scored.drop_duplicates(["study", "target", "layer"])
    k, n = int((dedup.verdict == "agree").sum()), len(dedup)
    print(f"\n=== SCORED {k}/{n} (after de-duplicating on study+target+layer, as step 13 does) ===")
    print(f"    excluded: {int((T.verdict == 'INPUT, not a prediction').sum())} imposed input, "
          f"{int((T.verdict == 'unscored (direction unknown)').sum())} direction unknown, "
          f"{len(scored) - len(dedup)} duplicate (same target+layer, second assay)")
    print(f"    call threshold {thr}; a correct sign below it is NO CALL and counts as disagree.")
    print(f"\nwrote {out / 'panelC_table.csv'}")


if __name__ == "__main__":
    main()
