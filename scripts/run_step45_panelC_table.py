"""Step 45: the Panel C table — model predictions for the PAR2_29 holdout.

2026-10-27 (user): PAR2_26 and PAR2_30 left the validation set, and TLR4 left the story. PAR2_29
is simulated as a plain F2rl1 KO under its LPS background -- NO Tlr4 knockdown for the albumin
carrier (the step-34/45 setup B is retired). Its Tlr4 observation is therefore a prediction again
and is scored like any other.

  PAR2_29  albumin-delivered editing, Raw264.7 / in vivo  -> F2rl1 KO, LPS on

LAYER MAPPING is not invented here: it is `conditions.readout_layers` from config.yaml, the same
mapping `lembas_par2/step13.py` scores with --
  signal : phosphoprotein, signal_activity, enzyme_activity, protein_cleavage,
           site_phosphorylation, total_protein
  mrna   : transcript, transcript/protein, secretion, transcript/secretion
A prediction with |value| < call_threshold (1e-3) is NO CALL and counts as a disagreement, which
is how step 13 scores it -- a correct sign below the threshold is not a success.

    python scripts/run_step45_panelC_table.py
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
STUDIES = ("PAR2_29",)
SETUP = "29 PAR2 KO (LPS on, no Tlr4 KD)"
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

    setups = {SETUP: (lps, {"F2rl1": kap})}
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
                 "
  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")

    out = cfg.paths.output_dir / "step45_panelC"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in obs.itertuples():
        v = [sd[r.observation_id] for sd in per_seed[SETUP]]
        med = float(np.nanmedian(v))
        call = np.isfinite(med) and abs(med) >= thr
        agree = bool(call and np.isfinite(r.obs_sign) and np.sign(med) == r.obs_sign)
        rows.append(dict(
            study=r.study, observation=r.observation_id, target=r.graph_target_id,
            readout=r.readout_type, layer=r.layer, figure=r.source_figure,
            observed=r.effect_direction, predicted=med,
            seeds_same_sign=int(np.nansum(np.sign(v) == np.sign(med))),
            call="yes" if call else "NO CALL",
            verdict=("unscored (direction unknown)" if not np.isfinite(r.obs_sign)
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
    print(f"    excluded: {int((T.verdict == 'unscored (direction unknown)').sum())} direction unknown, "
          f"{len(scored) - len(dedup)} duplicate (same target+layer, second assay)")
    print(f"    call threshold {thr}; a correct sign below it is NO CALL and counts as disagree.")
    print(f"\nwrote {out / 'panelC_table.csv'}")


if __name__ == "__main__":
    main()
