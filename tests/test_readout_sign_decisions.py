"""Readout (TF -> gene) sign decisions (curation/readout_sign_decisions.csv, 2026-10-17).

    python tests/test_readout_sign_decisions.py      # after tests/test_step05.py rebuilt step 5
"""
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.model import build_model
from lembas_par2.readout import load_regulons

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
dec = pd.read_csv(cfg.paths.readout_sign_decisions)
check("every decision records reason, evidence and who decided",
      dec[["reason", "evidence", "decided", "date"]].notna().all().all(), len(dec))
reg = load_regulons(cfg)
key = reg.set_index(["tf", "target"])
ok = all(int(key.at[(r.tf, r.target), "sign"]) == int(r.sign) and key.at[(r.tf, r.target), "decision"] == "readout_sign_decision"
         for r in dec.itertuples())
check("decisions applied by load_regulons", ok)
# a decision may override a CONFLICTED annotation (Irf3, Rela, Nfkb1) or a single-source one on
# experimental grounds (Irf2, Fos, Relb, Rel, Xbp1) - what must hold is that deciding never rewrites
# the record of what the sources said
import lembas_par2.readout as _R
raw = _R.pd.concat([_R.load_collectri(cfg), _R.load_trrust_regulon(cfg)], ignore_index=True)
m_ = _R.symbol_map(cfg)
raw["tf"] = raw.tf.map(lambda x: m_.get(x, x)); raw["target"] = raw.target.map(lambda x: m_.get(x, x))
raw = raw[raw.tf != raw.target]
g_ = raw.groupby(["tf", "target"], sort=False)
base = g_.agg(sign=("sign", lambda s: (lambda nz: nz.pop() if len(nz) == 1 else 0)(set(s[s != 0]))),
              dbs=("db", lambda s: ";".join(sorted(set(s))))).reset_index()
base["conflict"] = g_.sign.apply(lambda s: s.gt(0).any() and s.lt(0).any()).values
bk = base.set_index(["tf", "target"])
check("deciding a sign never rewrites what the sources said (dbs and conflict unchanged)",
      all(bk.at[(r.tf, r.target), "conflict"] == key.at[(r.tf, r.target), "conflict"]
          and bk.at[(r.tf, r.target), "dbs"] == key.at[(r.tf, r.target), "dbs"] for r in dec.itertuples()))
check("a decision that sets sign 0 returns the edge to sign-unknown",
      all(int(key.at[(r.tf, r.target), "sign"]) == 0 for r in dec.itertuples() if int(r.sign) == 0))
check("only the decided pairs differ from the plain consensus", int((reg.decision != "").sum()) == len(dec))
ro = pd.read_csv(cfg.paths.output_dir / "step05_readout_edges.csv.gz").set_index(["tf", "gene"])
check("rebuilt readout layer carries every decision",
      all(int(ro.at[(r.tf, r.target), "sign"]) == int(r.sign) for r in dec.itertuples() if (r.tf, r.target) in ro.index))
m = build_model(cfg, device="cpu")
W = m.readout_weights().detach()
rr = m.readout.reset_index(drop=True)
k = rr.index[(rr.tf == "Irf3") & (rr.gene == "Ifnb1")]
check("model starts Irf3 -> Ifnb1 positive and sign-locked",
      len(k) == 1 and float(W[k[0]]) > 0 and int(k[0]) in set(m.r_locked_idx.tolist()), f"w = {float(W[k[0]]):+.4f}")
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
