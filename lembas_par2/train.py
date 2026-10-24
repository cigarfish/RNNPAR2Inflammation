"""Step 11: training machinery.

11a  Trainer
     - parameter masks: train only selected entries of each parameter tensor
       (gradients of frozen entries are zeroed before every optimiser step, and
       Adam never sees a non-zero gradient for them, so they stay bit-identical)
     - Adam with linear warm-up then cosine decay
     - evaluation: data-loss terms, per-condition Pearson r of predicted vs
       measured TF activity, sign agreement of perturbation constraints
     - early stopping on a validation set (patience, keep best state)
     - checkpoints (state dict + metadata) and a step log
"""
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .loss import data_loss, full_loss


# ---------- masks ----------
def all_trainable(model):
    return {k: torch.ones_like(p, dtype=torch.bool) for k, p in model.named_parameters()}


def route_neighbourhood_mask(model, hops=1, extra_nodes=(), tiers=("T0", "T1", "T2", "T3")):
    """Train T0 edges and every edge of the given tiers with both ends within `hops`
    (undirected, over those tiers) of a route node; biases of those nodes; all
    readout, TF scales and drug kappa."""
    import networkx as nx
    e = model.edges
    t0 = model.tier == "T0"
    in_tiers = np.isin(model.tier, list(tiers))
    route = set(e.source[t0]) | set(e.target[t0]) | set(extra_nodes)
    G = nx.Graph()
    G.add_edges_from(zip(e.source[in_tiers], e.target[in_tiers]))
    near = set(route)
    for _ in range(hops):
        near |= {nb for v in list(near) if v in G for nb in G.neighbors(v)}
    edge_ok = torch.tensor(t0 | (in_tiers & (e.source.isin(near) & e.target.isin(near)).values), device=model.device)
    node_ok = torch.tensor([n in near for n in model.nodes], device=model.device)
    mask = all_trainable(model)
    mask["theta_locked"] = edge_ok[model.locked_idx]
    mask["theta_free"] = edge_ok[model.free_idx]
    mask["bias"] = node_ok
    return mask, dict(nodes=len(near), edges=int(edge_ok.sum()))


def mask_summary(model, mask):
    return {k: f"{int(mask[k].sum())}/{mask[k].numel()}" for k, _ in model.named_parameters()}


def apply_prior_spec(P, spec, nodes, act_range, reduce="mean"):
    """Configure Priors from a spec dict (keys lambda_bias, target_edge_anchor,
    activation as in config stage_a.priors / stage_a_variants); the activation
    penalty acts on `nodes` (bool mask) within act_range = (lo, hi)."""
    lo, hi = act_range
    return P.configure(lambda_bias=spec.get("lambda_bias"), target_edge_anchor=spec.get("target_edge_anchor"),
                       activation=(dict(lam=spec["activation"], lo=lo, hi=hi, nodes=nodes, reduce=reduce)
                                   if spec.get("activation") else None))


# ---------- metrics ----------
def pearson_cols(pred, target, mask):
    """Per-column Pearson r over masked entries (NaN when < 3 entries)."""
    out = []
    for j in range(pred.shape[1]):
        m = mask[:, j] > 0
        if int(m.sum()) < 3:
            out.append(float("nan"))
            continue
        a, b = pred[m, j].double(), target[m, j].double()
        a, b = a - a.mean(), b - b.mean()
        den = (a.norm() * b.norm()).item()
        out.append((a @ b).item() / den if den > 0 else float("nan"))
    return np.array(out)


@torch.no_grad()
def evaluate(model, T, positions):
    out = model(conditions=positions)
    D = data_loss(out, T)
    pos = torch.tensor(positions, device=T.weight.device)
    r_tf = pearson_cols(out["tf_pred"], T.tf[:, pos], T.tf_mask[:, pos])
    r_gene = pearson_cols(out["gene_pred"], T.gene[:, pos], T.gene_mask[:, pos])
    col = {p: j for j, p in enumerate(positions)}
    agree, n = 0, 0
    for c, i, mr, s in zip(T.d_cond.tolist(), T.d_index.tolist(), T.d_mrna.tolist(), T.d_sign.tolist()):
        if c in col and s != 0:
            v = (out["gene_pred"] if mr else out["delta"])[i, col[c]].item()
            agree += int(np.sign(v) == s)
            n += 1
    return dict(loss=D["total"].item(), tf=D["tf"].item(), gene=D["gene"].item(), direction=D["direction"].item(),
                r_tf=r_tf, r_gene=r_gene, r_tf_median=float(np.nanmedian(r_tf)) if np.isfinite(r_tf).any() else float("nan"),
                r_gene_median=float(np.nanmedian(r_gene)) if np.isfinite(r_gene).any() else float("nan"),
                direction_agree=agree / n if n else float("nan"), n_directions=n, out=out)


# ---------- schedule ----------
def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


# ---------- trainer ----------
class Trainer:
    def __init__(self, model, T, P, mask=None, lr=0.01, warmup=20, seed=0, log_path=None, lr_mult=None,
                 project=False):
        """lr_mult: {parameter name: multiplier}, e.g. {'bias': 0.1} for slower biases.
        project: enforce rho(|A|) <= loss.rho_max by projection after every step."""
        self.project = project
        torch.manual_seed(seed)
        self.model, self.T, self.P = model, T, P
        self.mask = mask or all_trainable(model)
        self.params = [p for _, p in model.named_parameters()]
        self.names = [k for k, _ in model.named_parameters()]
        lr_mult = lr_mult or {}
        self.opt = torch.optim.Adam([dict(params=[p], lr=lr, mult=lr_mult.get(k, 1.0))
                                     for k, p in zip(self.names, self.params)], lr=lr)
        self.base_lr, self.warmup = lr, warmup
        self.log_path = Path(log_path) if log_path else None
        self.history = []

    def step(self, positions, step, total):
        for g in self.opt.param_groups:
            g["lr"] = lr_at(step, total, self.base_lr, self.warmup) * g["mult"]
        self.opt.zero_grad(set_to_none=False)
        info = {}
        loss, terms, _ = full_loss(self.model, self.T, self.P, conditions=positions, info=info)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at step {step}: " + ", ".join(f"{k} {float(v):.3g}" for k, v in terms.items()))
        loss.backward()
        for k, p in zip(self.names, self.params):
            if p.grad is not None:
                p.grad.mul_(self.mask[k])
        self.opt.step()
        if hasattr(self.model, "clamp_locked"):
            self.model.clamp_locked()
        if self.project:
            terms["rho_projected"], terms["bulk_scale"] = self.model.project_stability()
        return loss.item(), {k: float(v) for k, v in terms.items()}, info

    def fit(self, train_pos, val_pos=None, steps=200, batch=None, eval_every=20, patience=None, seed=0, verbose=True,
            monitor=None, pre_step=None):
        """Train; optional mini-batches of `batch` conditions; early stopping on val loss.
        monitor: optional callable(model) -> dict of extra scalars, logged at every eval.
        pre_step: optional callable(model, step) run before every optimiser step (step 12j
        uses it to refit the detached readout nuisance on the training conditions)."""
        rng = np.random.default_rng(seed)
        best, best_state, bad = float("inf"), None, 0
        t0 = time.time()
        for s in range(steps):
            pos = list(rng.choice(train_pos, size=batch, replace=False)) if batch and batch < len(train_pos) else list(train_pos)
            if pre_step is not None:
                pre_step(self.model, s)
            loss, terms, info = self.step(pos, s, steps)
            rec = dict(step=s, loss=loss, lr=lr_at(s, steps, self.base_lr, self.warmup), seconds=time.time() - t0,
                       fwd_it=info["last_forward"]["iterations"], fwd_conv=info["last_forward"]["converged"], **terms)
            if val_pos is not None and ((s + 1) % eval_every == 0 or s == steps - 1):
                ev = evaluate(self.model, self.T, val_pos)
                rec.update(val_loss=ev["loss"], val_r_tf=ev["r_tf_median"], val_dir=ev["direction_agree"])
                if monitor is not None:
                    rec.update(monitor(self.model))
                if ev["loss"] < best - 1e-6:
                    best, best_state, bad = ev["loss"], copy.deepcopy(self.model.state_dict()), 0
                else:
                    bad += 1
                if verbose:
                    print(f"  step {s + 1:4d}  loss {loss:.4f}  val {ev['loss']:.4f}  val r_tf {ev['r_tf_median']:+.3f}  "
                          f"rho {terms['rho']:.3f}", flush=True)
                if patience is not None and bad >= patience:
                    rec["early_stop"] = True
                    self.history.append(rec)
                    break
            self.history.append(rec)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w") as f:
                for r in self.history:
                    f.write(json.dumps(r) + "\n")
        return self.history


# ---------- checkpoints ----------
def save_checkpoint(model, path, **meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), meta=meta, n_nodes=len(model.nodes), n_edges=len(model.edges)), path)


def load_checkpoint(model, path):
    ck = torch.load(path, map_location=model.device, weights_only=False)
    if ck["n_nodes"] != len(model.nodes) or ck["n_edges"] != len(model.edges):
        raise ValueError("checkpoint does not match the network")
    model.load_state_dict(ck["state"])
    return ck["meta"]
