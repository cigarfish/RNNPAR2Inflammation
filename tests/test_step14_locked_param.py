"""Step 14 unit test: the 'abs' parameterisation of sign-locked edges (model.locked_param).

CPU, no training data needed beyond the step-5/6 outputs (~1 min):
    python tests/test_step14_locked_param.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.model import build_model
from lembas_par2.train import load_checkpoint, save_checkpoint

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


def model(param):
    cfg = load_config()
    cfg["model"]["locked_param"] = param
    m = build_model(cfg, device="cpu", seed=0)
    m.initialize()
    return m


a, b = model("softplus"), model("abs")
check("identical model at initialisation (weights, readout, biases)",
      torch.allclose(a.weights(), b.weights(), atol=1e-6) and torch.allclose(a.readout_weights(), b.readout_weights(), atol=1e-6)
      and torch.allclose(a.bias, b.bias, atol=1e-5))

# slope dw/dtheta: 1 for abs, sigmoid(theta) for softplus
for m, name in [(a, "softplus"), (b, "abs")]:
    m.zero_grad()
    m.weights()[m.locked_idx].sum().backward()
    g = m.theta_locked.grad.abs()
    if name == "abs":
        check("abs: dw/dtheta = 1 for every locked signalling edge", torch.allclose(g, torch.ones_like(g)))
    else:
        check("softplus: dw/dtheta = sigmoid(theta)", torch.allclose(g, torch.sigmoid(a.theta_locked.detach()), atol=1e-6),
              f"median {float(g.median()):.3f}")

# a large step that would cross zero: the clamp keeps every curated sign
sign0 = torch.sign(b.weights()[b.locked_idx]).clone()
with torch.no_grad():
    b.theta_locked.sub_(10.0)
    b.W_locked.sub_(10.0)
b.clamp_locked()
w = b.weights()[b.locked_idx]
check("clamp_locked keeps theta >= 0 (sign can only reach 0, never flip)",
      bool((b.theta_locked >= 0).all() and (b.W_locked >= 0).all() and ((torch.sign(w) == sign0) | (w == 0)).all()))
check("clamp_locked is a no-op in softplus mode", (lambda t: (a.clamp_locked(), torch.equal(a.theta_locked, t))[1])(a.theta_locked.clone()))

# checkpoint round trip through the harness loader (reads locked_param from the spec)
from lembas_par2.eval_harness import load_model
c = model("abs")
p = ROOT / "outputs" / "step14_test_ckpt.pt"
save_checkpoint(c, p, spec={"locked_param": "abs"})
m2, meta = load_model(load_config(), p, "cpu")
check("harness loader rebuilds an abs-mode model from the checkpoint spec",
      m2.locked_param == "abs" and torch.allclose(m2.weights(), c.weights()))
p.unlink()

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
