"""Step 41 verification: did dropping the six STRING shortcuts fix Ikbkb and Nfkb1?

Run AFTER retraining. States the expected outcome up front and reports PASS/FAIL against it, so
the result is not open to interpretation after the fact.

What was changed (step 41): six STRING-only PPI edges into Ikbkb were dropped via
curation/sign_decisions.csv (action: drop) -- Tlr4, Ticam1, Ticam2, Myd88, Irak4, Irak1 -> Ikbkb.
All had n_dbs=0 and a free learned sign that fitted NEGATIVE, together supplying +0.02729 against
the curated Map3k7/Traf6 route's -0.02063.

EXPECTED, fixed before the run:
  Ikbkb  delta  NEGATIVE  (was +0.0081; first-order projection ~ -0.019)
  Nfkbia delta  POSITIVE  (was -0.0172) -- because Ikbkb -| Nfkbia
  Nfkb1  delta  NEGATIVE  (was +0.0116) -- because Nfkbia -| Nfkb1; PAR2_29 Fig 3L says DOWN
  Sting1          reported, NOT a success criterion for this step. Map3k7 is one of its three T0
                  inputs and sits in the same cascade, so it may move; that is an observation,
                  not the goal. This step was justified on curation grounds alone.

    python scripts/run_step41_verify.py
    python scripts/run_step41_verify.py --kappa-tlr4 0.77
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.eval_harness import load_model
from lembas_par2.core import simulate

torch.set_grad_enabled(False)

# (node, expected sign, value before step 41)
EXPECT = [("Ikbkb", -1, +0.00807), ("Nfkbia", +1, -0.01724), ("Nfkb1", -1, +0.01161)]
REPORT = ["Tlr4", "Ticam1", "Traf3", "Tbk1", "Map3k7", "Cgas", "Sting1", "Ifnb1", "Il1b", "Il6"]
GONE = [("Tlr4", "Ikbkb"), ("Ticam1", "Ikbkb"), ("Ticam2", "Ikbkb"),
        ("Myd88", "Ikbkb"), ("Irak4", "Ikbkb"), ("Irak1", "Ikbkb")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="R4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--kappa-tlr4", type=float, default=0.60)
    a = ap.parse_args()

    cfg = load_config()
    lps = cfg.step13["contexts"]["lps"]
    rows, checked_edges = [], False
    for s in a.seeds:
        run = a.variant if s == 0 else f"{a.variant}_s{s}"
        ck = cfg.paths.output_dir / "source_variants" / run / "checkpoint.pt"
        if not ck.exists():
            print(f"[skip] no checkpoint {run}")
            continue
        try:
            m, _ = load_model(cfg, ck, "cpu")
        except (RuntimeError, ValueError) as exc:
            sys.exit(f"{run}: checkpoint does not match the network ({exc}).\nRetrain first:\n"
                     f"  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
        if not checked_edges:
            pairs = set(zip(m.edges.source, m.edges.target))
            still = [f"{s_}->{t_}" for s_, t_ in GONE if (s_, t_) in pairs]
            print(f"[edges] the 6 dropped shortcuts still present: "
                  f"{still if still else 'NONE - removal confirmed in the trained model'}")
            checked_edges = True
        n = len(m.nodes)
        u = torch.zeros(n, 1, dtype=m.dtype)
        for x in lps:
            u[m.idx[x], 0] = cfg.perturbation["input_level"]
        U = u.repeat(1, 2)
        S = torch.ones(n, 2, dtype=m.dtype)
        S[m.idx["F2rl1"], 1] = 1 - cfg.step13["ko_kappa"]
        S[m.idx["Tlr4"], 1] = 1 - a.kappa_tlr4
        X = simulate(m.struct, m.weights(), m.bias, U, S, max_iter=3000, tol=1e-8)
        d = (S * X)[:, 1] - (S * X)[:, 0]
        for nd in [e[0] for e in EXPECT] + REPORT:
            rows.append(dict(seed=s, node=nd, delta=float(d[m.idx[nd]])))
        print(f"  {run}: ok", flush=True)

    if not rows:
        sys.exit("no checkpoints found. Train first:" 
                 "\n  python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R4 R0")
    D = pd.DataFrame(rows)
    ns = D.seed.nunique()

    print(f"\n=== THE CHECK (setup B, kappa_TLR4={a.kappa_tlr4}, {a.variant}, {ns} seeds) ===")
    ok = True
    for nd, want, before in EXPECT:
        v = D[D.node == nd].delta
        med, agree = v.median(), int((v > 0).sum() if want > 0 else (v < 0).sum())
        good = (med > 0) if want > 0 else (med < 0)
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {nd:7s} {med:+.5f} "
              f"(was {before:+.5f}, wanted {'POSITIVE' if want > 0 else 'NEGATIVE'}), "
              f"{agree}/{ns} seeds agree")
    print(f"\n  {'ALL THREE FIXED' if ok else 'NOT FIXED - do not proceed to the Cgas levers'}")

    print("\n=== reported, NOT success criteria for this step ===")
    r = D[D.node.isin(REPORT)].groupby("node").agg(
        delta=("delta", "median"), seeds_neg=("delta", lambda x: int((x < 0).sum())))
    print(r.reindex([n for n in REPORT if n in r.index]).round(5).to_string())
    print(f"\n  Sting1 is NOT the goal here. Step 41 was justified on curation grounds "
          f"(n_dbs=0 shortcuts collapsing a modelled cascade), not on its sign.")


if __name__ == "__main__":
    main()
