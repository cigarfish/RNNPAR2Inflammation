"""Step 7e: benchmark the LEMBAS core on the real signalling network.

Uses outputs/step05_signalling_edges.csv.gz and outputs/step06_conditions.csv.
Builds every condition arm (reference + treated, ~290 simulations), initialises
weights by tier (T0 1.0, T1 0.5 with curated sign; T2 +-0.1, T3 +-0.01 random),
rescales to rho(|A|) = 0.9, and times forward + backward on each device and for
two STRING thresholds (T3 score >= 700 and >= 900).

Run from the LEMBAS folder:
    python tests/test_step07e.py
Writes outputs/step07e_benchmark.json
"""
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
from lembas_par2.core import Structure, perron_radius, simulate

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
out = cfg.paths.output_dir
E = pd.read_csv(out / "step05_signalling_edges.csv.gz", usecols=["source", "target", "sign", "tier", "string_score"],
                low_memory=False)
cond = pd.read_csv(out / "step06_conditions.csv")
for col in ["ref_inputs", "trt_inputs", "trt_inhibit"]:
    cond[col] = cond[col].map(json.loads)
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
print(f"torch {torch.__version__}, devices: {devices}"
      + (f", GPU {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))


def build(E, seed=0):
    nodes = sorted(set(E.source) | set(E.target))
    idx = {n: i for i, n in enumerate(nodes)}
    g = np.random.default_rng(seed)
    init = {"T0": 1.0, "T1": 0.5}
    w = np.where(E.tier.isin(["T0", "T1"]), E.sign * E.tier.map(init).fillna(0),
                 np.where(E.tier == "T2", g.choice([-1, 1], len(E)) * 0.1, g.choice([-1, 1], len(E)) * 0.01))
    return nodes, idx, E.source.map(idx).values, E.target.map(idx).values, w.astype(float)


def arms(cond, idx, n, inhibit_scale=0.3):
    """U (inputs) and S (output scale) for reference and treated arms of every condition."""
    cols_U, cols_S = [], []
    for r in cond.itertuples():
        for inp, inh in [(r.ref_inputs, []), (r.trt_inputs, r.trt_inhibit)]:
            u = np.zeros(n); s = np.ones(n)
            u[[idx[x] for x in inp if x in idx]] = 1.0
            s[[idx[x] for x in inh if x in idx]] = inhibit_scale
            cols_U.append(u); cols_S.append(s)
    return np.column_stack(cols_U), np.column_stack(cols_S)


report = {}
for label, sub in [("STRING>=700", E), ("STRING>=900", E[(E.tier != "T3") | (E.string_score >= 900)])]:
    nodes, idx, src, tgt, w = build(sub.reset_index(drop=True))
    n = len(nodes)
    U, S = arms(cond, idx, n)
    print(f"\n== {label}: {n} nodes, {len(src)} edges ({sub.tier.value_counts().to_dict()}), {U.shape[1]} simulations")
    loss_dir = np.random.default_rng(1).normal(size=(n, U.shape[1]))
    ref_X = None
    for dev in devices:
        st = Structure.build(src, tgt, n, dev)
        wt = torch.tensor(w, dtype=torch.float32, device=dev)
        rho0 = perron_radius(st, wt, max_iter=5000, tol=1e-7).item()
        wt = (wt * 0.9 / rho0).requires_grad_(True)
        b = torch.zeros(n, dtype=torch.float32, device=dev, requires_grad=True)
        Ut = torch.tensor(U, dtype=torch.float32, device=dev)
        St = torch.tensor(S, dtype=torch.float32, device=dev)
        C = torch.tensor(loss_dir, dtype=torch.float32, device=dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        info = {}
        t0 = time.time()
        X = simulate(st, wt, b, Ut, St, info=info, max_iter=500, tol=1e-5)
        if dev == "cuda":
            torch.cuda.synchronize()
        t_fwd = time.time() - t0
        t0 = time.time()
        (X * C).sum().backward()
        if dev == "cuda":
            torch.cuda.synchronize()
        t_bwd = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else None
        fi, bi = info["last_forward"], info["last_backward"]
        check(f"[{label} {dev}] forward and backward converge", fi["converged"] and bi["converged"],
              f"forward {fi['iterations']} it ({t_fwd:.2f} s), backward {bi['iterations']} it ({t_bwd:.2f} s)"
              + (f", peak GPU memory {mem:.0f} MB" if mem else ""))
        check(f"[{label} {dev}] states bounded, gradients finite",
              bool((X < 1).all() and torch.isfinite(X).all() and torch.isfinite(wt.grad).all() and torch.isfinite(b.grad).all()),
              f"X range [{X.min().item():.3f}, {X.max().item():.3f}], rho(|A|) start {rho0:.2f} -> 0.9")
        if ref_X is None:
            ref_X = X.detach().cpu()
        else:
            check(f"[{label}] CPU and GPU steady states agree", torch.allclose(ref_X, X.detach().cpu(), atol=1e-4),
                  f"max diff {(ref_X - X.detach().cpu()).abs().max().item():.1e}")
        report[f"{label}|{dev}"] = dict(nodes=n, edges=len(src), sims=U.shape[1], forward_s=t_fwd, backward_s=t_bwd,
                                        forward_iter=fi["iterations"], backward_iter=bi["iterations"], gpu_mem_mb=mem)

print("\n== timing summary (one forward + backward over all simulations)")
for k, v in report.items():
    print(f"[INFO] {k:18s} {v['forward_s'] + v['backward_s']:6.2f} s  "
          f"({v['edges']} edges, {v['forward_iter']}+{v['backward_iter']} iterations)")
(out / "step07e_benchmark.json").write_text(json.dumps(report, indent=2))

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 7e:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
