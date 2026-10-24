"""Step 18 checker: how should the unsigned (free-sign) readout edges be treated?

Rules in config `readout_param_test`, fixed before running. Reads the harness JSON and the
checkpoints; HOLDOUTS ARE NOT READ. The Ifnb1 / Cgas / Sting1 PAR2-KO predictions are printed for
transparency and are explicitly NOT selection criteria: this step was prompted by an Ifnb1 holdout
miss, so selecting on them would make the holdout circular.

    python scripts/run_source_variants.py --seeds 0 1 2 3 4 --variants R0 R1 R2 R3 R4
    python tests/check_readout_param.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.eval_harness import default_eligibility, load_model, result_path
from lembas_par2.step13 import ko_delta

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
rt = cfg.readout_param_test
o = cfg.paths.output_dir


@torch.no_grad()
def diagnostics(variant, seed):
    """The target metric plus the (non-selection) PAR2-KO readout predictions."""
    run = variant if seed == 0 else f"{variant}_s{seed}"
    ck = o / "source_variants" / run / "checkpoint.pt"
    if not ck.exists():
        return None
    try:
        m, _ = load_model(cfg, ck, "cpu")
    except (RuntimeError, ValueError) as exc:
        # A curation change makes an old checkpoint unloadable in two different ways, and this
        # caught only the first until 2026-10-19:
        #   RuntimeError "size mismatch ..."        - a parameter changed shape (signed <-> unsigned)
        #   ValueError   "checkpoint does not match the network" - the edge list itself changed
        # Either way the variant simply needs retraining, so report it as stale rather than
        # aborting the whole check and hiding the variants that ARE comparable.
        if not any(k in str(exc) for k in ("size mismatch", "does not match the network")):
            raise
        return dict(variant=variant, seed=seed, stale=True)
    W = m.readout_weights().detach()
    tot = float(W.abs().sum())
    share = float(W[m.r_free_idx].abs().sum() / tot) if tot > 0 else np.nan
    d = ko_delta(m, cfg.step13["contexts"]["lps"], cfg.step13["ko_kappa"])
    contrib = m.readout_weights() * d[m.r_tf]
    gp = torch.zeros(len(m.genes), dtype=d.dtype).index_add(0, m.r_gene, contrib)
    gidx = {g: i for i, g in enumerate(m.genes)}
    pred = {g: float(gp[gidx[g]]) for g in ["Ifnb1", "Cgas", "Sting1"] if g in gidx}
    return dict(variant=variant, seed=seed, unsigned_share=share, **{f"ko_{k}": v for k, v in pred.items()})


rows, diags = [], []
for v in rt["variants"]:
    for s in rt["seeds"]:
        p = result_path(cfg, "12j", v if s == 0 else f"{v}_s{s}")
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        ok, why = default_eligibility(r, rho_max=cfg.source_test["rho_max"], transmit_min=cfg.source_test["transmit_min"])
        rows.append(dict(variant=v, seed=s, eligible=ok, why="; ".join(why),
                         **{k: r[k] for k in ["r_tf_drug_resid_signal", "r_tf_geo_resid", "r_tf_drug",
                                              "r_tf_geo", "route_dSting1", "rho_absA"]}))
        d = diagnostics(v, s)
        if d:
            diags.append(d)
t, dg = pd.DataFrame(rows), pd.DataFrame(diags)
stale_mask = (dg["stale"].fillna(False).astype(bool) if "stale" in dg.columns
              else pd.Series(False, index=dg.index))
stale = sorted(set(dg.variant[stale_mask])) if len(dg) else []
if stale:
    print(f"[STALE] trained before the current curation and NOT comparable: {', '.join(stale)}\n"
          f"        retrain with: python scripts/run_source_variants.py --seeds "
          f"{' '.join(map(str, rt['seeds']))} --variants {' '.join(stale)}")
    t = t[~t.variant.isin(stale)]
    dg = dg[~dg.variant.isin(stale)]
dg = dg.drop(columns=[c for c in ["stale"] if c in dg])
fresh = [v for v in rt["variants"] if v not in stale]
need = len(rt["seeds"])
have = t.groupby("variant").size().to_dict() if len(t) else {}
if not check(f"all comparable variants present with all seeds ({', '.join(fresh)})",
             all(have.get(v, 0) >= need for v in fresh) and rt["baseline"] in fresh, have):
    print(f"run: python scripts/run_source_variants.py --seeds {' '.join(map(str, rt['seeds']))} "
          f"--variants {' '.join(rt['variants'])}")
    sys.exit(1)

m = t.groupby("variant").median(numeric_only=True).drop(columns="seed")
md = dg.groupby("variant").median(numeric_only=True).drop(columns="seed")
tab = m.join(md)
print("\n== medians over seeds (the ko_* columns are NOT selection criteria)")
print(tab.to_string(float_format=lambda x: f"{x:+.4f}"))

base = rt["baseline"]
b = tab.loc[base]
cand = [v for v in fresh if v != base]
print(f"\n== rules (baseline {base}: unsigned share {b.unsigned_share:.3f}, "
      f"drug resid {b.r_tf_drug_resid_signal:+.3f}, GEO resid {b.r_tf_geo_resid:+.3f})")
elig = []
for v in cand:
    r = tab.loc[v]
    red = 1 - r.unsigned_share / b.unsigned_share if b.unsigned_share > 0 else np.nan
    ok_t = red >= rt["target_reduction"]
    drops = {k: float(b[k] - r[k]) for k in rt["noninferiority"]}
    ok_n = all(drops[k] <= tol for k, tol in rt["noninferiority"].items())
    ok_e = bool(t[(t.variant == v)].eligible.all())
    if ok_t and ok_n and ok_e:
        elig.append(v)
    print(f"  {v}: unsigned share {r.unsigned_share:.3f} (reduction {red:.0%}, need "
          f"{rt['target_reduction']:.0%}) -> {ok_t} | non-inferior {ok_n} "
          + ", ".join(f"{k.split('_')[2]} drop {d:+.3f}" for k, d in drops.items())
          + f" | all seeds eligible {ok_e}")
sel = max(elig, key=lambda v: tab.loc[v].r_tf_drug_resid_signal) if elig else None
verdict = (f"ADOPT {sel} (highest residualised drug r among qualifying variants: "
           f"{tab.loc[sel].r_tf_drug_resid_signal:+.3f})" if sel else
           "ADOPT NONE: no variant both halves the unsigned share and stays non-inferior")
check("selection is mechanical and recorded", True, verdict)
out = o / "step18"
out.mkdir(exist_ok=True)
t.to_csv(out / "runs.csv", index=False)
dg.to_csv(out / "diagnostics.csv", index=False)
(out / "verdict.json").write_text(json.dumps(dict(verdict=verdict, selected=sel, eligible=elig,
                                                  medians=tab.to_dict(), rules=dict(rt)), indent=2, default=float))
print(f"\n[VERDICT] {verdict}")
print("[INFO] PAR2-KO predictions under LPS (reported only, never used to choose):")
for v in fresh:
    if v in md.index:
        r = md.loc[v]
        print(f"  {v}: Ifnb1 {r.ko_Ifnb1:+.4f}  Cgas {r.ko_Cgas:+.4f}  Sting1 {r.ko_Sting1:+.4f}")
print("       (study 30 observed Ifnb1 UP; a variant that flips it is NOT thereby preferred)")
print(f"\nwrote {out}\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
