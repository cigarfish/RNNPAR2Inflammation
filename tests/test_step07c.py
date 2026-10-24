"""Step 7c validation: implicit (adjoint) backward pass of the steady state.

Run from the LEMBAS folder:
    python tests/test_step07c.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2.core import LEAK, Structure, mml, mml_grad, simulate

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


DT = torch.float64
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
print(f"torch {torch.__version__}, devices: {devices}")
TIGHT = dict(max_iter=5000, tol=1e-13)


def random_net(n, density, rho, seed, B=3):
    g = np.random.default_rng(seed)
    m = g.random((n, n)) < density
    np.fill_diagonal(m, False)
    tgt, src = np.nonzero(m)
    w = g.normal(size=len(src))
    Ad = np.zeros((n, n)); Ad[tgt, src] = w
    w = w * rho / max(abs(np.linalg.eigvals(Ad)))
    b = g.normal(scale=0.2, size=n)
    U = np.zeros((n, B)); U[g.integers(0, n, 2 * B), g.integers(0, B, 2 * B)] = g.uniform(0.3, 1.0, 2 * B)
    S = g.uniform(0.3, 1.0, size=(n, B))
    return src, tgt, w, b, U, S


def tensors(w, b, U, S, dev):
    t = lambda a: torch.tensor(a, dtype=DT, device=dev, requires_grad=True)
    return t(w), t(b), t(U), t(S)


# ---------- 1. analytic gradient on a chain ----------
print("== analytic chain")
s = Structure.build([0], [1], 2)
w = torch.tensor([0.8], dtype=DT, requires_grad=True)
b = torch.zeros(2, dtype=DT, requires_grad=True)
U = torch.tensor([[0.4], [0.0]], dtype=DT, requires_grad=True)
X = simulate(s, w, b, U, **TIGHT)
X[1, 0].backward()
x0 = mml(torch.tensor(0.4, dtype=DT))
d1 = mml_grad(0.8 * x0)
check("d x1 / d w = f'(w x0) x0", abs(w.grad.item() - (d1 * x0).item()) < 1e-12, f"{w.grad.item():.6f}")
check("d x1 / d b1 = f'(w x0)", abs(b.grad[1].item() - d1.item()) < 1e-12)
check("d x1 / d u0 = f'(w x0) w f'(u0)", abs(U.grad[0, 0].item() - (d1 * 0.8 * mml_grad(torch.tensor(0.4, dtype=DT))).item()) < 1e-12)


# ---------- 2. gradcheck on random networks (finite differences) ----------
def margin(src, tgt, w, b, U, S):
    """Smallest distance of any pre-activation to the activation kinks (0, 0.5)."""
    st = Structure.build(src, tgt, len(b))
    with torch.no_grad():
        X = simulate(st, *[torch.tensor(a, dtype=DT) for a in (w, b, U, S)], **TIGHT)
        Z = st.matrix(torch.tensor(w, dtype=DT)) @ (torch.tensor(S) * X) + torch.tensor(b)[:, None] + torch.tensor(U)
    return float(torch.minimum(Z.abs(), (Z - 0.5).abs()).min())


print("\n== gradcheck (float64, finite differences)")
seed = 0
while True:
    net = random_net(12, 0.25, 0.7, seed)
    if margin(*net) > 1e-3:
        break
    seed += 1
for dev in devices:
    src, tgt, w, b, U, S = net
    st = Structure.build(src, tgt, len(b), dev)
    fn = lambda w_, b_, U_, S_: simulate(st, w_, b_, U_, S_, **TIGHT)
    ok = torch.autograd.gradcheck(fn, tensors(w, b, U, S, dev), eps=1e-6, atol=1e-6, rtol=1e-5, raise_exception=False)
    check(f"[{dev}] gradcheck w.r.t. weights, biases, inputs and output scale", ok, f"n=12, {len(w)} edges, seed {seed}")


# ---------- 3. equals plain backprop through the unrolled iteration ----------
print("\n== implicit vs unrolled backprop")
src, tgt, w, b, U, S = random_net(300, 0.02, 0.8, 7, B=5)
n = len(b)
g = np.random.default_rng(3)
C = torch.tensor(g.normal(size=(n, 5)), dtype=DT)                 # random linear loss
for dev in devices:
    st = Structure.build(src, tgt, n, dev)
    wi, bi, Ui, Si = tensors(w, b, U, S, dev)
    info = {}
    (simulate(st, wi, bi, Ui, Si, info=info, **TIGHT) * C.to(dev)).sum().backward()
    wu, bu, Uu, Su = tensors(w, b, U, S, dev)
    A = torch.zeros(n, n, dtype=DT, device=dev).index_put((torch.tensor(tgt, device=dev), torch.tensor(src, device=dev)), wu)
    Xu = torch.zeros(n, 5, dtype=DT, device=dev)
    for _ in range(400):
        Xu = mml(A @ (Su * Xu) + bu[:, None] + Uu)
    (Xu * C.to(dev)).sum().backward()
    diffs = {k: (a.grad - c.grad).abs().max().item() / max(c.grad.abs().max().item(), 1e-12)
             for k, a, c in [("w", wi, wu), ("b", bi, bu), ("U", Ui, Uu), ("S", Si, Su)]}
    check(f"[{dev}] implicit gradients equal unrolled backprop (rel. diff < 1e-8)", max(diffs.values()) < 1e-8,
          ", ".join(f"{k} {v:.1e}" for k, v in diffs.items())
          + f"; forward {info['last_forward']['iterations']} it, adjoint {info['last_backward']['iterations']} it")

# ---------- 4. CPU and GPU agree; float32 close ----------
if "cuda" in devices:
    grads = []
    for dev in devices:
        st = Structure.build(src, tgt, n, dev)
        ws = tensors(w, b, U, S, dev)
        (simulate(st, *ws, **TIGHT) * C.to(dev)).sum().backward()
        grads.append(ws[0].grad.cpu())
    check("CPU and GPU weight gradients agree", torch.allclose(grads[0], grads[1], atol=1e-10))
st = Structure.build(src, tgt, n)
w64 = tensors(w, b, U, S, "cpu")
(simulate(st, *w64, **TIGHT) * C).sum().backward()
w32 = [torch.tensor(a, dtype=torch.float32, requires_grad=True) for a in (w, b, U, S)]
(simulate(st, *w32, max_iter=2000, tol=1e-7) * C.float()).sum().backward()
rel = ((w32[0].grad.double() - w64[0].grad).abs().max() / w64[0].grad.abs().max()).item()
check("float32 weight gradients close to float64 (rel < 1e-4)", rel < 1e-4, f"{rel:.1e}")

# ---------- 5. options ----------
print("\n== options")
wi, bi, Ui, _ = tensors(w, b, U, S, "cpu")
info = {}
(simulate(st, wi, bi, Ui, None, info=info, **TIGHT) * C).sum().backward()
check("works without output scale S", wi.grad is not None and torch.isfinite(wi.grad).all().item())
check("diagnostics reported", info["last_forward"]["converged"] and info["last_backward"]["converged"],
      f"forward {info['last_forward']}, backward {info['last_backward']}")
wi, bi, Ui, Si = tensors(w, b, U, S, "cpu")
(simulate(st, wi, bi, Ui, Si, clip=0.05, **TIGHT) * C * 100).sum().backward()
check("gradient clipping option keeps gradients finite", torch.isfinite(wi.grad).all().item())

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 7c:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
