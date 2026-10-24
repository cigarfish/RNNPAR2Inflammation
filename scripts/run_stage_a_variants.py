"""Stage A variants against bias drift (config: stage_a_variants).

Every variant uses the same seed, split, mask and schedule as stage A; only the
prior / learning-rate settings differ.

Run from the LEMBAS folder (GPU recommended, ~1-2 min per variant):
    python scripts/run_stage_a_variants.py                    # all variants
    python scripts/run_stage_a_variants.py --variants V0 V5   # a subset
    python scripts/run_stage_a_variants.py --steps 20         # quick smoke run
Writes outputs/stage_a_variants/<ID>/{checkpoint.pt, log.jsonl, report.json}
and outputs/stage_a_variants/summary.csv. Then:
    python tests/check_stage_a_variants.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.core import mml_inverse, perron_radius, simulate
from lembas_par2.loss import Priors, Targets, balance_data_terms
from lembas_par2.model import build_model
from lembas_par2.train import Trainer, apply_prior_spec, evaluate, route_neighbourhood_mask, save_checkpoint

STING_ARM = ["SE_CYTO_MTDNA", "Cgas", "Sting1", "Tbk1"]
JAK_STAT = [(j, s) for j in ["Jak1", "Jak2", "Tyk2"] for s in ["Stat1", "Stat2", "Stat3", "Stat6"]]


def pooled_z(m):
    with torch.no_grad():
        X = simulate(m.struct, m.weights(), m.bias, m.pooled_reference(), None, max_iter=3000, tol=1e-7)
        return mml_inverse(X[:, 0].clamp(max=0.999))


def run_variant(cfg, vid, spec, sa, dev, out_root):
    out = out_root / vid
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(sa["seed"])
    t0 = time.time()
    m = build_model(cfg, device=dev, seed=sa["seed"])
    m.initialize()
    b_init = m.bias.detach().clone()
    T = Targets(cfg, m)
    balance_data_terms(m, T)
    mask, minfo = route_neighbourhood_mask(m, hops=sa["hops"], tiers=sa["tiers"])
    P = Priors(cfg, m, mode="anchored")
    apply_prior_spec(P, spec, mask["bias"], cfg.stage_a_activation_range)
    cond = m.design.cond
    is_src = cond.source.isin(sa["sources"]).values
    val = [i for i in np.where(is_src)[0] if cond.fold.iloc[i] == sa["val_fold"]]
    train = [i for i in np.where(is_src)[0] if i not in val]
    tr = Trainer(m, T, P, mask=mask, lr=sa["lr"], warmup=sa["warmup"], seed=sa["seed"], log_path=out / "log.jsonl",
                 lr_mult={"bias": spec.get("bias_lr_mult", 1.0)})
    print(f"\n=== {vid} {spec}")
    tr.fit(train, val_pos=val, steps=sa["steps"], batch=sa["batch"], eval_every=sa["eval_every"],
           patience=sa["patience"], seed=sa["seed"], verbose=False)
    save_checkpoint(m, out / "checkpoint.pt", stage="A", variant=vid, spec=spec, train=train, val=val,
                    tiers=list(sa["tiers"]), hops=sa["hops"])

    ev_tr, ev_va = evaluate(m, T, train), evaluate(m, T, val)
    jak = [i for i in train + val if cond.cls.iloc[i] in ("JAK", "JAK2/FLT3/IRAK1")]
    ev_jak = evaluate(m, T, jak)
    w = m.weights().detach()
    E = m.edges
    key = {(a, b): i for i, (a, b) in enumerate(zip(E.source, E.target))}
    js = [abs(w[key[p]].item()) for p in JAK_STAT if p in key]
    z = pooled_z(m)
    nodes_tr = mask["bias"]
    with torch.no_grad():
        rd = P.route_delta(m)
        hold = [i for i, s in enumerate(cond.source) if s == "holdout"]
        oh = m(conditions=hold)
        rho = perron_radius(m.struct, w, max_iter=3000, tol=1e-6).item()
    rep = dict(variant=vid, spec=json.dumps(spec), steps=len(tr.history), seconds=round(time.time() - t0, 1),
               train_loss=ev_tr["loss"], val_loss=ev_va["loss"], train_r_tf=ev_tr["r_tf_median"], val_r_tf=ev_va["r_tf_median"],
               jak_r_tf=ev_jak["r_tf_median"], jak_stat_w=float(np.mean(js)), rho=rho,
               leak_frac=float((z[nodes_tr] < 0).float().mean()), sat_frac=float((z[nodes_tr] > 2).float().mean()),
               **{f"z_{n}": z[m.idx[n]].item() for n in STING_ARM},
               **{f"dbias_{n}": (m.bias[m.idx[n]] - b_init[m.idx[n]]).item() for n in STING_ARM},
               route_dMapk1=rd[m.idx["Mapk1"]].item(), route_dBax=rd[m.idx["Bax"]].item(),
               route_dMtDNA=rd[m.idx["SE_CYTO_MTDNA"]].item(), route_dCgas=rd[m.idx["Cgas"]].item(),
               route_dSting1=rd[m.idx["Sting1"]].item(),
               **{f"holdout_{cond.condition_id.iloc[i]}_dSting1": oh["delta"][m.idx["Sting1"], k].item()
                  for k, i in enumerate(hold)})
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(f"  {rep['steps']} steps, {rep['seconds']} s | val loss {rep['val_loss']:.3f}, val r_tf {rep['val_r_tf']:+.3f}, "
          f"JAK r_tf {rep['jak_r_tf']:+.3f} | JAK->STAT |w| {rep['jak_stat_w']:.3f} | "
          f"z mtDNA/Cgas/Sting1/Tbk1 " + "/".join(f"{rep['z_' + n]:.2f}" for n in STING_ARM)
          + f" | route dSting1 {rep['route_dSting1']:+.3f}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_config()
    sa = dict(cfg.stage_a)
    if a.steps:
        sa["steps"] = a.steps
    variants = cfg.stage_a_variants
    ids = a.variants or list(variants)
    out_root = Path(a.out) if a.out else cfg.paths.output_dir / "stage_a_variants"
    rows = [run_variant(cfg, v, dict(variants[v]), sa, a.device, out_root) for v in ids]
    summ = pd.DataFrame(rows)
    p = out_root / "summary.csv"
    if p.exists() and a.variants:
        old = pd.read_csv(p)
        summ = pd.concat([old[~old.variant.isin(summ.variant)], summ], ignore_index=True).sort_values("variant")
    summ.to_csv(p, index=False)
    print(f"\nwrote {p}\nnext: python tests/check_stage_a_variants.py")


if __name__ == "__main__":
    main()
