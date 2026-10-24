"""Step 34: the two published studies as two DIFFERENT perturbations.

Protocols pre-registered in curation/TLR4_T0_prereg_2026-10-19.md section 7c. Read it first.

  A  PAR2_26, 4T1 tumour cell    : F2rl1 KO alone, NO TLR4/CD14 input.      Observed pSTING UP.
  B  PAR2_29, Raw264.7 macrophage: F2rl1 KO + Tlr4 knockdown, LPS context.  Observed pSTING DOWN.

The `Tlr4` knockdown in B represents the ALBUMIN CARRIER, which PAR2_29's own conclusion says
"colocalize[s] with endosomal TLR4 and subsequently inhibited TLR4 downstream signalling". It is
NOT part of PAR2 biology and must never be described as such.

kappa_TLR4 is a free knob, so it is reported as a DOSE-RESPONSE CURVE, never as a single tuned
point. The paper's measured reductions are marked on it: 0.40 albumin alone, 0.60 hAlbumin/cgPAR2
in vitro (the arm Fig 3N shows -> the primary comparison), 0.77 in vivo lung. If dSting1 only
turns negative far above 0.60, the explanation FAILS AT THE MEASURED DOSE and the run says so.

CIRCULARITY, declared in the prereg: PAR2_29_C1's `Tlr4` observation is currently scored as one of
the eight holdout successes. Under B, TLR4 falling is an IMPOSED INPUT, not a prediction, so that
observation must be dropped from the holdout score whenever B is the setup.

MODALITY: pSTING is activity -> compare the SIGNAL node. The mRNA readout is printed beside it
because the collaborators' own dynasore data is RNA-seq.

    python scripts/run_step34_studies.py
    python scripts/run_step34_studies.py --variant R0 --seeds 0 1 2
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
from lembas_par2.core import simulate

WATCH = ["Mapk1", "Mapk3", "Bak1", "Bax", "SE_CYTO_MTDNA", "Cgas", "Tlr4", "Ticam1",
         "Tbk1", "Irf3", "Sting1", "Ifnb1", "Nfkb1", "Il1b", "Il6"]
ANCHORS = {0.40: "albumin alone", 0.60: "hAlbumin/cgPAR2 in vitro (Fig 3N) <-- PRIMARY",
           0.77: "hAlbumin/cgPAR2 in vivo lung"}


@torch.no_grad()
def multi_ko(m, inputs, kos):
    """Steady-state delta for knocking down SEVERAL nodes at once. kos = {node: kappa}."""
    n = len(m.nodes)
    u = torch.zeros(n, 1, dtype=m.dtype)
    for x in inputs:
        u[m.idx[x], 0] = m.cfg.perturbation["input_level"]
    U = u.repeat(1, 2)
    S = torch.ones(n, 2, dtype=m.dtype)
    for node, kap in kos.items():
        S[m.idx[node], 1] = 1.0 - kap
    X = simulate(m.struct, m.weights(), m.bias, U, S, max_iter=3000, tol=1e-8)
    Y = S * X
    return (Y[:, 1] - Y[:, 0]).detach()


@torch.no_grad()
def gene_delta(m, d):
    c = m.readout_weights() * d[m.r_tf]
    return torch.zeros(len(m.genes), dtype=c.dtype).index_add(0, m.r_gene, c)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="R4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    cfg = load_config()
    out = cfg.paths.output_dir / "step34_studies"
    out.mkdir(parents=True, exist_ok=True)
    kap = cfg.step13["ko_kappa"]
    lps = cfg.step13["contexts"]["lps"]
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.77, 0.85, 0.95]

    rows, dose = [], []
    for s in a.seeds:
        run = a.variant if s == 0 else f"{a.variant}_s{s}"
        ck = cfg.paths.output_dir / "source_variants" / run / "checkpoint.pt"
        if not ck.exists():
            print(f"[skip] {run}")
            continue
        try:
            m, _ = load_model(cfg, ck, "cpu")
        except (RuntimeError, ValueError) as exc:
            sys.exit(f"{run}: checkpoint does not match the network ({exc}). Retrain first:\n"
                     f"  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
        gidx = {g: i for i, g in enumerate(m.genes)}

        setups = {
            "A PAR2_26 4T1 (no TLR4 input)": ([], {"F2rl1": kap}),
            "A' PAR2_26 with LPS on (check)": (lps, {"F2rl1": kap}),
            "B PAR2_29 macrophage (kTLR4=.60)": (lps, {"F2rl1": kap, "Tlr4": 0.60}),
        }
        for lab, (inp, kos) in setups.items():
            d = multi_ko(m, inp, kos)
            gp = gene_delta(m, d)
            for n in WATCH:
                rows.append(dict(seed=s, setup=lab, node=n, signal=float(d[m.idx[n]]),
                                 mrna=float(gp[gidx[n]]) if n in gidx else np.nan))
        for k in grid:
            d = multi_ko(m, lps, {"F2rl1": kap, "Tlr4": k})
            gp = gene_delta(m, d)
            dose.append(dict(seed=s, kappa_tlr4=k, sting_signal=float(d[m.idx["Sting1"]]),
                             sting_mrna=float(gp[gidx["Sting1"]]),
                             tlr4_signal=float(d[m.idx["Tlr4"]])))
        print(f"  {run}: done", flush=True)

    if not rows:
        sys.exit("no checkpoints found. Train first:" 
                 "\n  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
    D, G = pd.DataFrame(rows), pd.DataFrame(dose)
    D.to_csv(out / "setups.csv", index=False); G.to_csv(out / "dose_response.csv", index=False)
    ns = D.seed.nunique()

    for lab in D.setup.unique():
        sub = D[D.setup == lab].groupby("node").agg(
            signal=("signal", "median"), mrna=("mrna", "median"),
            seeds_up=("signal", lambda x: int((x > 0).sum())))
        print(f"\n=== {lab} === ({a.variant}, median of {ns} seeds)")
        print(sub.reindex([n for n in WATCH if n in sub.index]).round(5).to_string())

    print(f"\n=== B: dose-response in kappa_TLR4 (LPS context, F2rl1 KO held at {kap}) ===")
    g = G.groupby("kappa_tlr4").agg(sting_signal=("sting_signal", "median"),
                                    sting_mrna=("sting_mrna", "median"),
                                    seeds_up=("sting_signal", lambda x: int((x > 0).sum())))
    g["anchor"] = [ANCHORS.get(k, "") for k in g.index]
    print(g.round(5).to_string())

    # A SIGN IS NOT A CALL. The project's call threshold is 1e-3; step 29 established that a
    # delta below it is numerical silence, not a direction ("the sign at 4e-5 is numerical
    # residue, not a prediction"). An earlier version of this verdict tested the sign alone and
    # reported a -0.00015 median as reproducing PAR2_29. It does not.
    CALL = 1e-3
    neg = g[g.sting_signal < 0]
    flip = float(neg.index.min()) if len(neg) else None
    called = g[g.sting_signal < -CALL]
    firm = float(called.index.min()) if len(called) else None
    at60 = float(g.sting_signal.get(0.60, float("nan")))
    n60 = int(g.seeds_up.get(0.60, -1))
    print()
    print(f"At the measured kappa_TLR4 = 0.60: dSting1 = {at60:+.5f}, {n60}/5 seeds up.")
    if abs(at60) < CALL:
        print(f"  |{at60:+.5f}| < call threshold {CALL} -> this is SILENCE, not a DOWN call. "
              f"The sign alone must not be reported as reproducing PAR2_29.")
    if flip is None:
        print("VERDICT: never negative on this grid -> does NOT reproduce PAR2_29 at any dose.")
    else:
        print(f"VERDICT: sign turns negative at kappa_TLR4 = {flip:.2f}; first dose clearing the "
              f"call threshold is {firm if firm is not None else 'NONE on this grid'}.")
        if firm is not None and firm > 0.60:
            print(f"  A confident DOWN needs kappa_TLR4 >= {firm:.2f}, ABOVE the measured in "
                  f"vitro 0.60 (prereg prediction 10 is NOT met on magnitude).")
    print("\nREMINDER: PAR2_29's Tlr4 holdout observation is an INPUT here, not a prediction. "
          "Drop it from the holdout score under setup B.")
    (out / "report.json").write_text(json.dumps(dict(
        variant=a.variant, seeds=sorted(int(s) for s in D.seed.unique()),
        flip_kappa=flip, dose=g.drop(columns="anchor").to_dict("index")), indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
