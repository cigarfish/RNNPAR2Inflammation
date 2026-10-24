"""Step 7d validation: Perron radius rho(|A|), its gradient, and the stability guarantee.

Run from the LEMBAS folder:
    python tests/test_step07d.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2.core import Structure, perron_radius, signed_radius, simulate, steady_state

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


DT = torch.float64
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
print(f"torch {torch.__version__}, devices: {devices}")
TIGHT = dict(max_iter=20000, tol=1e-14)


def rand_edges(n, density, seed, signed=True):
    g = np.random.default_rng(seed)
    m = g.random((n, n)) < density
    np.fill_diagonal(m, False)
    tgt, src = np.nonzero(m)
    w = g.normal(size=len(src)) if signed else g.uniform(0.1, 1, len(src))
    return src, tgt, w


def dense(src, tgt, w, n):
    A = np.zeros((n, n)); A[tgt, src] = w
    return A


# ---------- 1. value vs numpy ----------
print("== rho(|A|) value")
errs, bracket = [], True
for seed in range(10):
    n = 30 + 5 * seed
    src, tgt, w = rand_edges(n, 0.1, seed)
    exact = np.abs(np.linalg.eigvals(np.abs(dense(src, tgt, w, n)))).max()
    info = {}
    r = perron_radius(Structure.build(src, tgt, n), torch.tensor(w, dtype=DT), info=info, **TIGHT).item()
    errs.append(abs(r - exact) / exact)
    bracket &= info["last"]["lower"] <= exact + 1e-9 and exact <= info["last"]["upper"] + 1e-9
check("matches numpy Perron root on 10 random signed matrices", max(errs) < 1e-8, f"max rel err {max(errs):.1e}")
check("Collatz-Wielandt bounds bracket the true value", bracket)

# reducible: a chain feeding a 3-cycle, plus a sink; periodic: a 2-cycle
cases = {
    "periodic 2-cycle (rho = sqrt(ab))": ([0, 1], [1, 0], [0.8, 0.5], 2, np.sqrt(0.4)),
    "reducible chain -> 3-cycle -> sink": ([0, 1, 2, 3, 3], [1, 2, 3, 1, 4], [2.0, 0.9, 0.7, 0.6, 5.0], 5, (0.9 * 0.7 * 0.6) ** (1 / 3)),
}
for name, (s, t, w, n, exact) in cases.items():
    r = perron_radius(Structure.build(s, t, n), torch.tensor(w, dtype=DT), **TIGHT).item()
    check(name, abs(r - exact) < 1e-6, f"{r:.6f} vs {exact:.6f}")
# acyclic (nilpotent) networks: rho = 0 is approached only ~1/iterations from above;
# the estimate stays a valid upper bound (the safe side for a stability penalty)
r = perron_radius(Structure.build([0, 1, 0], [1, 2, 2], 3), torch.tensor([3.0, -2.0, 1.0], dtype=DT), **TIGHT).item()
check("acyclic network (rho = 0): upper bound within 1e-3", 0 <= r < 1e-3, f"{r:.2e}")

# ---------- 2. gradient ----------
print("\n== gradient")
for dev in devices:
    src, tgt, w = rand_edges(15, 0.25, 3)
    st = Structure.build(src, tgt, 15, dev)
    wt = torch.tensor(w, dtype=DT, device=dev, requires_grad=True)
    ok = torch.autograd.gradcheck(lambda x: perron_radius(st, x, **TIGHT), (wt,), eps=1e-6, atol=1e-6, rtol=1e-4,
                                  raise_exception=False)
    check(f"[{dev}] gradcheck of rho(|A|) w.r.t. signed weights", ok, f"{len(w)} edges")
st = Structure.build(src, tgt, 15)
wt = torch.tensor(w, dtype=DT, requires_grad=True)
perron_radius(st, wt, **TIGHT).backward()
check("gradient sign follows weight sign (shrinking |w| lowers rho)",
      bool((torch.sign(wt.grad[wt.grad != 0]) == torch.sign(wt.detach()[wt.grad != 0])).all()))
# a gradient step on rho reduces it
w2 = (wt - 0.05 * wt.grad).detach()
check("one gradient step lowers rho", perron_radius(st, w2, **TIGHT).item() < perron_radius(st, wt.detach(), **TIGHT).item())

# ---------- 3. bound and guarantee ----------
print("\n== stability guarantee")
ok_bound, ok_conv, iters = True, True, []
for seed in range(20):
    n = 300
    src, tgt, w = rand_edges(n, 0.02, 100 + seed)
    st = Structure.build(src, tgt, n)
    wt = torch.tensor(w, dtype=DT)
    wt = wt * 0.95 / perron_radius(st, wt, **TIGHT)
    rs = signed_radius(st, wt)
    ok_bound &= rs <= 0.95 + 1e-6
    g = np.random.default_rng(seed)
    U = torch.tensor(g.normal(scale=2.0, size=(n, 4)), dtype=DT)
    S = torch.tensor(g.uniform(0, 1, size=(n, 4)), dtype=DT)
    sol = steady_state(st, wt, torch.tensor(g.normal(size=n), dtype=DT), U, S, max_iter=3000, tol=1e-10)
    ok_conv &= sol.converged
    iters.append(sol.iterations)
check("signed radius rho(A) <= rho(|A|) (20 random networks)", ok_bound)
check("rho(|A|) = 0.95 -> forward converges for random inputs, biases and inhibition", ok_conv,
      f"iterations {min(iters)}-{max(iters)}")
src, tgt, w = rand_edges(300, 0.02, 7)
st = Structure.build(src, tgt, 300)
wt = torch.tensor(w, dtype=DT)
wt = wt * 0.95 / perron_radius(st, wt, **TIGHT)
info = {}
X = simulate(st, wt.clone().requires_grad_(True), torch.zeros(300, dtype=DT), torch.randn(300, 3, dtype=DT), info=info,
             max_iter=3000, tol=1e-10)
X.sum().backward()
check("adjoint (backward) converges too", info["last_backward"]["converged"], info["last_backward"])

# ---------- 4. scale: random network the size of ours ----------
print("\n== scale")
n, E = 8650, 254000
g = np.random.default_rng(0)
src = g.integers(0, n, E * 2); tgt = g.integers(0, n, E * 2)
keep = src != tgt
key = np.unique(tgt[keep] * n + src[keep])[:E]
tgt, src = key // n, key % n
w = g.normal(scale=0.02, size=len(src))
for dev in devices:
    st = Structure.build(src, tgt, n, dev)
    wt = torch.tensor(w, dtype=torch.float32, device=dev)
    t0 = time.time()
    info = {}
    r = perron_radius(st, wt, max_iter=3000, tol=1e-6, info=info).item()
    if dev == "cuda":
        torch.cuda.synchronize()
    check(f"[{dev}] {n} nodes / {len(src)} edges: rho(|A|) computed", np.isfinite(r),
          f"rho {r:.4f}, {info['last']['iterations']} iterations, {time.time() - t0:.2f} s")

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 7d:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
