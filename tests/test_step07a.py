"""Step 7a validation: MML activation, derivatives and inverse.

Run from the LEMBAS folder:
    python tests/test_step07a.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2.core import LEAK, mml, mml_grad, mml_grad_from_output, mml_inverse

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__}, device for tests: {dev}")
z = torch.linspace(-5, 20, 20001, dtype=torch.float64)

# ---------- values ----------
known = {-1.0: -LEAK, 0.0: 0.0, 0.25: 0.25, 0.5: 0.5, 1.0: 0.75, 5.0: 0.95}
got = {k: float(mml(torch.tensor(k, dtype=torch.float64))) for k in known}
check("known values", all(abs(got[k] - v) < 1e-12 for k, v in known.items()), got)
y = mml(z)
check("bounded above by 1", bool((y < 1).all()), f"max {float(y.max()):.6f}")
check("monotonically increasing", bool((torch.diff(y) > 0).all()))
eps = 1e-9
for p in [0.0, 0.5]:
    t = torch.tensor([p - eps, p + eps], dtype=torch.float64)
    check(f"continuous at z={p}", abs(float(mml(t)[1] - mml(t)[0])) < 1e-8)
t = torch.tensor([0.5 - eps, 0.5 + eps], dtype=torch.float64)
check("slope continuous at z=0.5 (both sides 1)", torch.allclose(mml_grad(t), torch.ones(2, dtype=torch.float64), atol=1e-6))

# ---------- derivatives ----------
zz = z[(z.abs() > 1e-6) & ((z - 0.5).abs() > 1e-6)].clone().requires_grad_(True)
mml(zz).sum().backward()
check("mml_grad equals torch autograd", torch.allclose(mml_grad(zz.detach()), zz.grad, atol=1e-12))
check("mml_grad_from_output equals mml_grad", torch.allclose(mml_grad_from_output(mml(z)), mml_grad(z), atol=1e-10))
h = 1e-6
fd = (mml(z + h) - mml(z - h)) / (2 * h)
mask = (z.abs() > 1e-4) & ((z - 0.5).abs() > 1e-4)
check("mml_grad equals finite differences", torch.allclose(mml_grad(z)[mask], fd[mask], atol=1e-6))

# ---------- inverse ----------
check("inverse round trip f^-1(f(z)) = z", torch.allclose(mml_inverse(mml(z)), z, atol=1e-8))

# ---------- dtype / device ----------
for dt in [torch.float32, torch.float64]:
    x = torch.randn(1000, 7, dtype=dt, device=dev) * 3
    out = mml(x)
    check(f"{dt} on {dev}: shape, dtype, device kept",
          out.shape == x.shape and out.dtype == dt and out.device.type == dev and torch.isfinite(out).all().item())

# ---------- matches the original LEMBAS implementation (lembas_erk/activation.py) ----------
src = ROOT / "lembas_erk" / "activation.py"
if src.exists():
    spec = importlib.util.spec_from_file_location("lembas_activation", src)
    ref = importlib.util.module_from_spec(spec)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(ref)
        zn = z.numpy()
        check("matches original MMLactivation", np.allclose(ref.MMLactivation(zn), mml(z).numpy(), atol=1e-12))
        check("matches original MMLDeltaActivation", np.allclose(ref.MMLDeltaActivation(zn), mml_grad(z).numpy(), atol=1e-12))
        yo = mml(z)
        check("matches original MMLoneStepDeltaActivationFactor",
              torch.allclose(ref.MMLoneStepDeltaActivationFactor(yo), mml_grad_from_output(yo), atol=1e-10))
else:
    print("[INFO] lembas_erk/activation.py not found; reference comparison skipped")

n_fail = results.count(False)
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 7a:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
