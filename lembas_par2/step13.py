"""Step 13: the PAR2-knockout prediction, scored against the held-out papers, with the
controls that make the answer interpretable. Rules fixed in config `step13` before any number
was computed.

13a  Holdout scoring. For each model, the three held-out PAR2-KO contrasts (PAR2_26/29/30) are
     simulated and every scored observation is compared by sign (signal layer: node delta;
     mrna layer: predicted log2FC). Unit = unique (condition, node, layer): 10 observations
     (PAR2_30's Ifnb1 is listed twice). |prediction| < call_threshold counts as a disagreement.
     Primary null: random predicted signs, one-sided binomial P(X >= k | n, 0.5).
     Secondary null: the prior-only model with random T0 (route) signs.
     Models: prior-only (step-9b initialisation), J0 (adopted baseline) / J3, their unanchored
     versions J0U / J3U, and the sign-free versions J0F / J3F; >= 5 seeds per trained variant.
13b  Sign-free probe. J0F / J3F: unanchored, and the signs of the 77 T0 edges plus the 58 T1
     edges between route nodes are free (start at the curated sign, may flip). Statistic: the
     simulated PAR2-KO dSting1 (F2rl1 kappa 0.95) in the pooled and LPS contexts, median over
     seeds; plus how many route edges ended with the opposite sign. Caveat: the magnitude
     anchor still applies, so a flip must pass through 0 against it.
13c  Branch ablation: zero the edges of one ERK -> BCL2-family branch (Mcl1, Gsk3b, Bad, Bim),
     and of all four, and report the share of PAR2-KO dSting1 that remains. Descriptive.
13d  Attribution: first-order edge contributions w_e * d(dSting1)/d(w_e) through the adjoint,
     summed by branch and listed for the top edges. Descriptive, and an attribution OF THE
     PRIOR for anchored models - always shown next to the unanchored numbers.

Verdict (primary model J0, seeds pooled by median):
    CONTRADICTED  sign-free J0F median dSting1 (pooled) < -reversal_threshold
    CONSISTENT    holdout agreement binomial p < holdout_alpha AND not contradicted
    WEAKENED      otherwise (holdouts do not beat random signs, direction not reversed)
"""
from math import comb

import numpy as np
import pandas as pd
import torch

from .core import simulate


# ---------- simulations ----------
@torch.no_grad()
def ko_delta(model, inputs="pooled", kappa=0.95, node="F2rl1", weights=None):
    """Steady-state delta (all nodes) of knocking out `node` in a given input context."""
    n = len(model.nodes)
    dev, dt = model.device, model.dtype
    if isinstance(inputs, str) and inputs == "pooled":
        u = model.pooled_reference()
    else:
        u = torch.zeros(n, 1, dtype=dt, device=dev)
        for x in inputs:
            u[model.idx[x], 0] = model.cfg.perturbation["input_level"]
    U = u.repeat(1, 2)
    S = torch.ones(n, 2, dtype=dt, device=dev)
    S[model.idx[node], 1] = 1.0 - kappa
    w = model.weights() if weights is None else weights
    X = simulate(model.struct, w, model.bias, U, S, max_iter=3000, tol=1e-8)
    Y = S * X
    return (Y[:, 1] - Y[:, 0]).detach()


def ko_attribution(model, target="Sting1", inputs="pooled", kappa=0.95):
    """w_e * d(delta_target)/d(w_e) for every edge (first-order contribution)."""
    n = len(model.nodes)
    dev, dt = model.device, model.dtype
    u = model.pooled_reference() if inputs == "pooled" else None
    if u is None:
        u = torch.zeros(n, 1, dtype=dt, device=dev)
        for x in inputs:
            u[model.idx[x], 0] = model.cfg.perturbation["input_level"]
    U = u.repeat(1, 2)
    S = torch.ones(n, 2, dtype=dt, device=dev)
    S[model.idx["F2rl1"], 1] = 1.0 - kappa
    w = model.weights().detach().clone().requires_grad_(True)
    X = simulate(model.struct, w, model.bias.detach(), U, S, max_iter=3000, tol=1e-8)
    d = (S * X)[model.idx[target], 1] - (S * X)[model.idx[target], 0]
    (g,) = torch.autograd.grad(d, w)
    return float(d), (w.detach() * g).cpu().numpy()


# ---------- 13a ----------
def binom_tail(k, n, p=0.5):
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def holdout_observations(cfg):
    d = pd.read_csv(cfg.paths.output_dir / "step06_directions.csv")
    d = d[(d.source == "holdout") & (d.sign != 0)]
    return d.drop_duplicates(["condition_id", "node", "layer"])


@torch.no_grad()
def score_holdouts(model, obs, call_threshold, route_subset):
    cond = model.design.cond
    hold = [i for i, s in enumerate(cond.source) if s == "holdout"]
    out = model(conditions=hold)
    col = {cond.condition_id.iloc[i]: j for j, i in enumerate(hold)}
    gpos = {g: k for k, g in enumerate(model.genes)}
    rows = []
    for r in obs.itertuples():
        j = col[r.condition_id]
        if r.layer == "signal" and r.node in model.idx:
            v = out["delta"][model.idx[r.node], j].item()
        elif r.layer == "mrna" and r.node in gpos:
            v = out["gene_pred"][gpos[r.node], j].item()
        else:
            v = np.nan
        call = np.isfinite(v) and abs(v) >= call_threshold
        rows.append(dict(condition_id=r.condition_id, node=r.node, layer=r.layer, observed=int(r.sign),
                         predicted=v, call=call, agree=bool(call and np.sign(v) == r.sign),
                         agree_sign_only=bool(np.isfinite(v) and np.sign(v) == r.sign)))
    t = pd.DataFrame(rows)
    k, n = int(t.agree.sum()), len(t)
    sub = t[t.node.isin(route_subset)]
    sting = {cond.condition_id.iloc[i]: out["delta"][model.idx["Sting1"], j].item() for j, i in enumerate(hold)}
    return dict(k=k, n=n, p=binom_tail(k, n), k_sign_only=int(t.agree_sign_only.sum()),
                route_k=int(sub.agree.sum()), route_n=len(sub), dSting1_holdouts=sting), t


# ---------- 13c ----------
def branch_masks(model, branches):
    e = model.edges
    key = {(a, b): i for i, (a, b) in enumerate(zip(e.source, e.target))}
    masks = {}
    for name, pairs in branches.items():
        idx = [key[(a, b)] for a, b in pairs if (a, b) in key]
        m = torch.zeros(len(e), dtype=torch.bool, device=model.device)
        m[idx] = True
        masks[name] = m
    masks["all_four"] = torch.stack(list(masks.values())).any(0)
    return masks


@torch.no_grad()
def ablation(model, branches, inputs="pooled", kappa=0.95):
    w = model.weights()
    base = ko_delta(model, inputs, kappa)[model.idx["Sting1"]].item()
    res = dict(base=base)
    for name, m in branch_masks(model, branches).items():
        d = ko_delta(model, inputs, kappa, weights=torch.where(m, torch.zeros_like(w), w))[model.idx["Sting1"]].item()
        res[name] = dict(dSting1=d, retained=d / base if abs(base) > 1e-12 else np.nan)
    return res


# ---------- sign-free bookkeeping ----------
def flipped_route_edges(model):
    if not getattr(model, "sign_unlocked", np.zeros(1, bool)).any():
        return None
    w = model.weights().detach().cpu().numpy()
    u = model.sign_unlocked
    flip = u & (np.sign(w) != model.curated_sign) & (np.abs(w) > 1e-6)
    t0 = model.tier == "T0"
    e = model.edges
    return dict(n_unlocked=int(u.sum()), n_flipped=int(flip.sum()), n_flipped_T0=int((flip & t0).sum()),
                flipped=[f"{a}->{b}" for a, b in zip(e.source[flip], e.target[flip])])
