"""Step 11a validation: training machinery (masks, metrics, schedule, early stopping,
checkpoints, reproducibility).

Run from the LEMBAS folder:
    python tests/test_step11a.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.loss import Priors, Targets, balance_data_terms
from lembas_par2.model import build_model
from lembas_par2.train import (Trainer, evaluate, load_checkpoint, lr_at, mask_summary, pearson_cols,
                               route_neighbourhood_mask, save_checkpoint)

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = cfg.paths.output_dir


def fresh():
    m = build_model(cfg, device=dev)
    m.initialize()
    T = Targets(cfg, m)
    balance_data_terms(m, T)
    return m, T, Priors(cfg, m)


m, T, P = fresh()
cond = m.design.cond
pick = [int(np.where(cond.name == n)[0][0]) for n in ["Pimasertib", "Trametinib", "Ruxolitinib"]] + \
       [int(np.where(cond.source == "geo")[0][0])]

# ---------- 1. metrics ----------
print("== metrics")
g = torch.Generator().manual_seed(0)
tgt = torch.randn(50, 4, generator=g, dtype=torch.float64)
msk = torch.ones(50, 4, dtype=torch.float64); msk[:10, 1] = 0; msk[:, 3] = 0; msk[:2, 3] = 1
r = pearson_cols(2 * tgt + 1, tgt, msk)
check("Pearson r = 1 for a positive linear relation (masked entries ignored)", np.allclose(r[:3], 1))
check("Pearson r = NaN with < 3 entries", np.isnan(r[3]))
check("Pearson r = -1 for a sign flip", np.allclose(pearson_cols(-tgt, tgt, torch.ones_like(tgt)), -1))
ev = evaluate(m, T, pick)
check("evaluate returns finite loss terms and per-condition r", all(np.isfinite(ev[k]) for k in ["loss", "tf", "gene"])
      and len(ev["r_tf"]) == len(pick), f"r_tf {np.round(ev['r_tf'], 3).tolist()}, direction agreement {ev['direction_agree']}")

# ---------- 2. schedule ----------
lrs = [lr_at(s, 100, 0.01, 10) for s in range(100)]
check("learning rate: linear warm-up then cosine decay", lrs[0] < lrs[9] == 0.01 and lrs[-1] < 1e-4 and
      all(a >= b for a, b in zip(lrs[9:], lrs[10:])), f"{lrs[0]:.4f} -> {lrs[9]:.4f} -> {lrs[-1]:.6f}")

# ---------- 3. parameter masks ----------
print("\n== parameter masks")
mask, info = route_neighbourhood_mask(m, hops=1)
print(f"[INFO] route + 1-hop neighbourhood: {info['nodes']} nodes, {info['edges']} edges; trainable: {mask_summary(m, mask)}")
check("all 77 route edges are trainable in the neighbourhood mask",
      bool(torch.tensor(m.tier == "T0", device=dev)[torch.cat([m.locked_idx[mask['theta_locked']]])].sum() == 77))
before = {k: p.detach().clone() for k, p in m.named_parameters()}
tr = Trainer(m, T, P, mask=mask, lr=0.01, warmup=1)
tr.fit(pick, steps=10, verbose=False)
frozen_ok, moved = True, True
for k, p in m.named_parameters():
    f = ~mask[k]
    if f.any():
        frozen_ok &= torch.equal(p.detach()[f], before[k][f])
    if mask[k].any():
        moved &= not torch.equal(p.detach()[mask[k]], before[k][mask[k]])
check("frozen entries stay bit-identical after training", frozen_ok)
check("trainable entries of every group change", moved)

# ---------- 4. early stopping ----------
print("\n== early stopping")
m2, T2, P2 = fresh()
tr2 = Trainer(m2, T2, P2, lr=0.0, warmup=1)
h = tr2.fit(pick, val_pos=pick, steps=50, eval_every=1, patience=1, verbose=False)
check("stops when validation loss stops improving (lr 0: flat loss)", len(h) == 2 and h[-1].get("early_stop"), f"{len(h)} steps")

# ---------- 5. checkpoints ----------
print("\n== checkpoints")
ck = out / "checkpoints" / "test_step11a.pt"
save_checkpoint(m, ck, note="test")
m3, T3, _ = fresh()
meta = load_checkpoint(m3, ck)
check("checkpoint round trip restores every parameter and buffer bit-identically",
      all(torch.equal(x, y) for x, y in zip(m.state_dict().values(), m3.state_dict().values())) and meta["note"] == "test")
with torch.no_grad():
    a, b, a2 = m(conditions=pick), m3(conditions=pick), m(conditions=pick)
noise = (a["tf_pred"] - a2["tf_pred"]).abs().max().item()       # same model twice: GPU scatter-add order
diff = (a["tf_pred"] - b["tf_pred"]).abs().max().item()
check("restored model predicts the same (within GPU run-to-run noise)", diff <= max(noise, 1e-6) * 10,
      f"diff {diff:.1e}, run-to-run noise {noise:.1e}")
ck.unlink()

# ---------- 6. reproducibility ----------
print("\n== reproducibility")
runs = []
for _ in range(2):
    mr, Tr, Pr = fresh()
    h = Trainer(mr, Tr, Pr, lr=0.01, warmup=1, seed=1).fit(pick, steps=5, verbose=False)
    runs.append([r["loss"] for r in h])
diff = max(abs(x - y) for x, y in zip(*runs))
check("same seed -> same loss trajectory (up to GPU float noise)", diff < 1e-4, f"max diff {diff:.1e}")

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 11a:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
