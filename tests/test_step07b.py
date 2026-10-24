"""Step 7b validation: sparse structure and forward steady-state solver.

Run from the LEMBAS folder:
    python tests/test_step07b.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2.core import LEAK, Structure, mml, steady_state

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


DT = torch.float64
f = lambda z: float(mml(torch.tensor(z, dtype=DT)))
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
print(f"torch {torch.__version__}, devices: {devices}")


def solve(src, tgt, w, n, b, U, dev="cpu", **kw):
    s = Structure.build(src, tgt, n, dev)
    return steady_state(s, torch.tensor(w, dtype=DT, device=dev), torch.tensor(b, dtype=DT, device=dev),
                        torch.tensor(U, dtype=DT, device=dev), **kw)


# ---------- 1. structure ----------
print("== structure")
s = Structure.build([0, 1, 2, 0], [1, 2, 0, 2], 3)
w = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DT)
dense = s.matrix(w).to_dense()
exp = torch.zeros(3, 3, dtype=DT)
for a, t, v in zip([0, 1, 2, 0], [1, 2, 0, 2], [1.0, 2.0, 3.0, 4.0]):
    exp[t, a] = v
check("A[tgt, src] = w, any edge order", torch.equal(dense, exp))
check("A^T matrix is the transpose", torch.equal(s.matrix_T(w).to_dense(), exp.T))
for bad, why in [(([0, 0], [1, 1], 3), "duplicate edges"), (([0], [5], 3), "out of range")]:
    try:
        Structure.build(*bad)
        check(f"rejects {why}", False)
    except ValueError:
        check(f"rejects {why}", True)

# ---------- 2. toy networks with exact answers ----------
print("\n== toy networks")
# chain 0 -> 1 -> 2 (+0.8, +0.6), input 0.4 on node 0, biases 0
r = solve([0, 1], [1, 2], [0.8, 0.6], 3, [0, 0, 0], [[0.4], [0], [0]])
x0 = f(0.4); x1 = f(0.8 * x0); x2 = f(0.6 * x1)
check("chain: exact steady state", np.allclose(r.X[:, 0].numpy(), [x0, x1, x2], atol=1e-9) and r.converged,
      f"{r.X[:, 0].numpy().round(6)} vs {[round(v, 6) for v in (x0, x1, x2)]}")
# inhibitory step 0 -| 1 with bias on 1
r = solve([0], [1], [-1.5], 2, [0, 0.3], [[0.6], [0]])
x0 = f(0.6); x1 = f(0.3 - 1.5 * x0)
check("inhibition: exact steady state (incl. leak)", np.allclose(r.X[:, 0].numpy(), [x0, x1], atol=1e-9),
      f"node1 = {r.X[1, 0].item():+.5f} (pre-activation {0.3 - 1.5 * x0:+.4f})")
# saturation: strong input
r = solve([0], [1], [1.0], 2, [0, 0], [[10.0], [0]])
check("saturating branch: f(10) = 0.975", abs(r.X[0, 0].item() - 0.975) < 1e-12)
# feedback loop 0 <-> 1: fixed point solved and self-consistent
r = solve([0, 1], [1, 0], [0.5, 0.4], 2, [0.1, 0.0], [[0.3], [0]])
Z = np.array([0.1 + 0.3 + 0.4 * r.X[1, 0].item(), 0.5 * r.X[0, 0].item()])
check("feedback loop: X = f(AX + b + U)", np.allclose(r.X[:, 0].numpy(), mml(torch.tensor(Z)).numpy(), atol=1e-8) and r.converged,
      f"{r.iterations} iterations, residual {r.residual:.1e}")
# output scale S = 0 on node 0 blocks its signal
s = Structure.build([0, 1], [1, 2], 3)
Sm = torch.tensor([[0.0], [1.0], [1.0]], dtype=DT)
r = steady_state(s, torch.tensor([0.8, 0.6], dtype=DT), torch.zeros(3, dtype=DT), torch.tensor([[0.4], [0], [0]], dtype=DT), S=Sm)
check("output scale S=0 on node 0 silences downstream", r.X[0, 0].item() > 0 and abs(r.X[1, 0].item()) < 1e-12 and abs(r.X[2, 0].item()) < 1e-12)


# ---------- 3. random networks vs dense numpy reference ----------
def random_net(n, density, rho, seed):
    g = np.random.default_rng(seed)
    m = g.random((n, n)) < density
    np.fill_diagonal(m, False)
    tgt, src = np.nonzero(m)
    w = g.normal(size=len(src))
    Ad = np.zeros((n, n)); Ad[tgt, src] = w
    scale = rho / max(abs(np.linalg.eigvals(Ad)))
    return src, tgt, w * scale, Ad * scale, g


def numpy_ref(Ad, b, U, iters=2000):
    X = np.zeros_like(U)
    for _ in range(iters):
        Z = Ad @ X + b[:, None] + U
        X = np.where(Z < 0, LEAK * Z, np.where(Z > 0.5, 1 - 0.25 / np.maximum(Z, 0.5), Z))
    return X


print("\n== random networks")
n, B = 400, 6
src, tgt, w, Ad, g = random_net(n, 0.02, 0.8, 1)
b = g.normal(scale=0.1, size=n)
U = np.zeros((n, B)); U[g.integers(0, n, 20), g.integers(0, B, 20)] = 1.0
ref = numpy_ref(Ad, b, U)
for dev in devices:
    r = solve(src, tgt, w, n, b, U, dev, tol=1e-9, max_iter=2000)
    check(f"[{dev}] matches dense numpy reference", r.converged and np.allclose(r.X.cpu().numpy(), ref, atol=1e-7),
          f"{r.iterations} iterations, residual {r.residual:.1e}, max diff {np.abs(r.X.cpu().numpy() - ref).max():.1e}")
r_all = solve(src, tgt, w, n, b, U, tol=1e-10, max_iter=2000)
sep = np.column_stack([solve(src, tgt, w, n, b, U[:, [k]], tol=1e-10, max_iter=2000).X[:, 0].numpy() for k in range(B)])
check("batched simulations equal separate ones", np.allclose(r_all.X.numpy(), sep, atol=1e-9))
s32 = Structure.build(src, tgt, n)
r32 = steady_state(s32, torch.tensor(w, dtype=torch.float32), torch.tensor(b, dtype=torch.float32),
                   torch.tensor(U, dtype=torch.float32), tol=1e-6, max_iter=2000)
check("float32 close to float64", r32.converged and np.allclose(r32.X.numpy(), ref, atol=1e-4), f"max diff {np.abs(r32.X.numpy() - ref).max():.1e}")

# ---------- 4. non-convergence is flagged ----------
src, tgt, w, Ad, g = random_net(200, 0.05, 3.0, 2)
r = solve(src, tgt, w, 200, np.zeros(200), g.normal(size=(200, 2)), max_iter=200)
check("unstable network (spectral radius 3) is flagged as not converged", not r.converged,
      f"residual {r.residual:.2e}, iterations {r.iterations}")

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 7b:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
