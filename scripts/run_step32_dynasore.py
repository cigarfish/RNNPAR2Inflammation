"""Does an EDGE-level block transmit at T1? The prerequisite for simulating dynasore.

Dynasore is not a node KO. It blocks dynamin-dependent internalisation, so the right
representation is to cut the edges that represent ENDOSOMAL output while leaving the membrane
ones intact:
  * Arrb1/Arrb2 -> Raf1/Map2k1/Map2k2/Mapk1/Mapk3   (the endosomal ERK scaffold)
  * Tlr4 -> Ticam1/Ticam2                            (TRIF; requires TLR4 endocytosis)
and leaving F2rl1 -> Arrb1/2 (recruitment, desensitisation) and Arrb1 -> Src intact.

An Arrb1 NODE KO is wrong for two separate reasons, both measured here: it removes
beta-arrestin-mediated receptor desensitisation (so F2rl1 goes UP, which dynasore should not do)
and it removes the Arrb1 -> Src arm.

This probe answers one question before any tier is changed: does cutting those edges move
anything downstream at their CURRENT tiers, or does T1 swallow the perturbation?
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config  # noqa: E402
from lembas_par2.eval_harness import load_model  # noqa: E402
from lembas_par2.core import simulate  # noqa: E402

ENDO_ERK = [("Arrb1", t) for t in ("Raf1", "Map2k1", "Map2k2", "Mapk1", "Mapk3")] + \
           [("Arrb2", t) for t in ("Raf1", "Map2k1", "Map2k2", "Mapk1", "Mapk3")]
TRIF = [("Tlr4", "Ticam1"), ("Tlr4", "Ticam2")]
WATCH = ["F2rl1", "Arrb1", "Gnaq", "Prkca", "Raf1", "Map2k1", "Mapk1", "Mapk3", "Src",
         "Ticam1", "Tbk1", "Irf3", "Ifnb1", "Nfkb1", "Bak1", "Bax", "SE_CYTO_MTDNA",
         "Cgas", "Sting1"]


@torch.no_grad()
def steady(m, w, u):
    n = len(m.nodes)
    return simulate(m.struct, w, m.bias, u, torch.ones(n, 1, dtype=m.dtype),
                    max_iter=3000, tol=1e-8)


@torch.no_grad()
def main():
    cfg = load_config()
    m, _ = load_model(cfg, cfg.paths.output_dir / "source_variants" / "R4" / "checkpoint.pt", "cpu")
    E = m.edges
    n = len(m.nodes)
    u = torch.zeros(n, 1, dtype=m.dtype)
    for x in cfg.step13["contexts"]["lps"]:
        u[m.idx[x], 0] = cfg.perturbation["input_level"]

    key = {(s, t): i for i, (s, t) in enumerate(zip(E.source, E.target))}

    def block(pairs, label):
        idx, tiers = [], []
        for p in pairs:
            if p in key:
                idx.append(key[p])
                tiers.append(E.tier.iloc[key[p]])
            else:
                print(f"    [absent] {p[0]} -> {p[1]}")
        w = m.weights().clone()
        before = w[idx].abs().mean().item() if idx else float("nan")
        w[idx] = 0.0
        print(f"  {label}: {len(idx)} edges cut, tiers={sorted(set(tiers))}, "
              f"mean |w| cut = {before:.5f}")
        return w

    X0 = steady(m, m.weights(), u)
    base = {nd: float(X0[m.idx[nd], 0]) for nd in WATCH if nd in m.idx}

    scenarios = {
        "endosomal ERK scaffold only": ENDO_ERK,
        "TRIF only (Tlr4 -> Ticam1/2)": TRIF,
        "DYNASORE = both": ENDO_ERK + TRIF,
    }
    out = {}
    for label, pairs in scenarios.items():
        print(f"\n[{label}]")
        w = block(pairs, "cut")
        X = steady(m, w, u)
        out[label] = {nd: float(X[m.idx[nd], 0]) - base[nd] for nd in base}

    print(f"\n{'node':16s} {'baseline':>10s} " + " ".join(f"{k[:14]:>15s}" for k in out))
    print("-" * (28 + 16 * len(out)))
    for nd in WATCH:
        if nd not in base:
            continue
        row = f"{nd:16s} {base[nd]:+10.5f} "
        row += " ".join(f"{out[k][nd]:+15.5f}" for k in out)
        print(row)
    print("\n(columns are the CHANGE from baseline caused by cutting those edges)")


if __name__ == "__main__":
    main()
