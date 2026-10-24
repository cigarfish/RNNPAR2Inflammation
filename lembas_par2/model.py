"""Step 9: the LEMBAS-style model.

9a  Assembly. Learnable pieces:
      signalling weights  T0/T1: w = sign * softplus(theta)  (sign locked)
                          T2/T3: w = theta                   (sign free)
      node biases b
      drug kappa          via perturb.Design (kappa = sigmoid(theta_d))
      TF-output scales    alpha = softplus(a) > 0, one per output TF
      gene readout        W[g, TF]: signed regulon edges sign-locked, unsigned free
    Forward: steady states of all condition arms -> effective deltas (n x C) ->
      tf_pred   (TF x C) = alpha * delta[TF]
      gene_pred (G x C)  = sum_TF W[g, TF] * delta[TF]
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .core import LEAK, Structure, simulate
from .perturb import Design


def softplus_inv(y):
    y = torch.as_tensor(y, dtype=torch.float64)
    return torch.where(y > 20, y, torch.log(torch.expm1(y)))


class LembasModel(torch.nn.Module):
    def __init__(self, cfg, edges, readout, tf_outputs, cond, device="cpu", dtype=torch.float32, seed=0):
        super().__init__()
        mc = cfg.model
        self.cfg, self.device, self.dtype = cfg, device, dtype
        # magnitude parameterisation of sign-locked edges (signalling T0/T1 and signed readout):
        # 'softplus' w = sign*softplus(theta) [default]; 'abs' w = sign*theta, theta >= 0 kept by
        # clamp_locked() after every optimiser step. softplus has slope sigmoid(theta) ~0.04-0.1 at
        # the small curated weights, so those edges learned 10-25x slower than free ones (step 14).
        self.locked_param = mc.get("locked_param", "softplus")
        if self.locked_param not in ("softplus", "abs"):
            raise ValueError(f"unknown locked_param {self.locked_param}")
        g = np.random.default_rng(seed)
        edges = edges.reset_index(drop=True)

        # ----- signalling layer -----
        self.nodes = sorted(set(edges.source) | set(edges.target))
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        self.edges = edges
        self.struct = Structure.build(edges.source.map(self.idx).values, edges.target.map(self.idx).values,
                                      len(self.nodes), device)
        tier = edges.tier.astype(str).values
        self.tier = tier
        locked = np.isin(tier, mc["sign_locked_tiers"])
        strength = np.array([mc["init_strength"][t] for t in tier], dtype=float)
        if mc.get("t0_fan_in_normalise", False):
            t0 = tier == "T0"
            fan_in = edges[t0].groupby("target").size()
            strength[t0] = strength[t0] / edges.target[t0].map(fan_in).values
        # Per-node input normalisation for the bulk tiers. rho(|A|) < 1 is roughly a
        # budget on each node's total input, so with a median in-degree of 18 (90th
        # percentile 72) a uniform strength gives every edge the hub's small share.
        # Dividing by the target's in-degree moves budget from hubs to sparsely
        # regulated nodes: it raises the 90th-percentile admissible weight ~3x at the
        # same rho (step 12d), which is where the target->TF paths run.
        norm = mc.get("fan_in_normalise", "none")
        if norm != "none":
            deg = edges.groupby("target").size()
            d = edges.target.map(deg).fillna(1.0).values.astype(float)
            f = np.sqrt(d) if norm == "sqrt" else d
            strength = np.where(tier == "T0", strength, strength / np.maximum(1.0, f))
        self.fan_in_normalise = norm
        # step 13b sign-free probe: 'route' frees the sign of every T0 edge and of every T1 edge
        # between two route nodes. They start at their curated sign but training may flip them,
        # so the data - not the prior - decide the route's direction.
        unlock = mc.get("unlock_signs", "none")
        if unlock not in ("none", "route"):
            raise ValueError(f"unknown unlock_signs {unlock}")
        if unlock == "route":
            t0 = tier == "T0"
            rn = set(edges.source[t0]) | set(edges.target[t0])
            unl = t0 | ((tier == "T1") & edges.source.isin(rn).values & edges.target.isin(rn).values)
        else:
            unl = np.zeros(len(edges), dtype=bool)
        self.sign_unlocked = unl
        self.curated_sign = np.sign(edges.sign.values).astype(float)
        locked = locked & ~unl
        sign = np.where(locked | unl, np.sign(edges.sign.values), g.choice([-1.0, 1.0], len(edges)))
        if (locked & (sign == 0)).any():
            raise ValueError("sign-locked edge without a sign")
        self.locked_idx = torch.tensor(np.where(locked)[0], device=device)
        self.free_idx = torch.tensor(np.where(~locked)[0], device=device)
        self.register_buffer("locked_sign", torch.tensor(sign[locked], dtype=dtype, device=device))
        self.theta_locked = torch.nn.Parameter(self._pos_inv(strength[locked]).to(device, dtype))
        self.theta_free = torch.nn.Parameter(torch.tensor(sign[~locked] * strength[~locked], dtype=dtype, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(len(self.nodes), dtype=dtype, device=device))

        # ----- perturbation design (inputs, inhibition, drug kappa) -----
        self.design = Design(cfg, cond, self.nodes, device=device, dtype=dtype)
        self.kappa_theta = torch.nn.Parameter(self.design.theta.detach().clone())
        self.design.theta = self.kappa_theta

        # ----- TF outputs -----
        self.tf_outputs = list(tf_outputs)
        self.register_buffer("tf_idx", torch.tensor([self.idx[t] for t in self.tf_outputs], device=device))
        self.alpha_raw = torch.nn.Parameter(
            softplus_inv(torch.full((len(self.tf_outputs),), mc["alpha_init"])).to(device, dtype))

        # ----- gene readout -----
        readout = readout[readout.tf.isin(self.idx)].reset_index(drop=True)
        self.genes = sorted(set(readout.gene))
        gidx = {g_: i for i, g_ in enumerate(self.genes)}
        self.readout = readout
        n_reg = readout.groupby("gene").tf.transform("size").values
        mag = mc["readout_init"] / np.sqrt(n_reg)
        rs = readout.sign.values.astype(float)
        r_locked = rs != 0
        # step 18: the 3.2 % of readout edges with no curated sign are the only freely-moving ones
        # (signed edges are sign-locked through softplus and learn ~10x slower), so they absorbed
        # the fit - for Ifnb1 the two unsigned regulators reached 0.45 / 0.41 against ~0.10 for the
        # 43 signed ones. Starting them at 0 makes an unsigned edge earn its weight instead.
        uinit = mc.get("readout_unsigned_init")
        if uinit is not None:
            mag = mag.copy()
            mag[~r_locked] = float(uinit)
        self.readout_unsigned_init = uinit
        r_sign = np.where(r_locked, rs, g.choice([-1.0, 1.0], len(readout)))
        self.register_buffer("r_gene", torch.tensor(readout.gene.map(gidx).values, device=device))
        self.register_buffer("r_tf", torch.tensor(readout.tf.map(self.idx).values, device=device))
        self.r_locked_idx = torch.tensor(np.where(r_locked)[0], device=device)
        self.r_free_idx = torch.tensor(np.where(~r_locked)[0], device=device)
        self.register_buffer("r_locked_sign", torch.tensor(r_sign[r_locked], dtype=dtype, device=device))
        self.W_locked = torch.nn.Parameter(self._pos_inv(mag[r_locked]).to(device, dtype))
        self.W_free = torch.nn.Parameter(torch.tensor(r_sign[~r_locked] * mag[~r_locked], dtype=dtype, device=device))
        self.sim_opts = dict(leak=LEAK, max_iter=500, tol=1e-5)

    # ----- locked-magnitude parameterisation -----
    def _pos(self, theta):
        return F.softplus(theta) if self.locked_param == "softplus" else theta.clamp(min=0.0)

    def _pos_inv(self, y):
        y = torch.as_tensor(y)
        return softplus_inv(y) if self.locked_param == "softplus" else y.clone().double()

    @torch.no_grad()
    def clamp_locked(self):
        """'abs' mode: keep every locked magnitude >= 0 (a projection, so the curated sign holds)."""
        if self.locked_param == "abs":
            self.theta_locked.clamp_(min=0.0)
            self.W_locked.clamp_(min=0.0)

    # ----- assembled parameters -----
    def weights(self):
        w = torch.empty(len(self.edges), dtype=self.dtype, device=self.device)
        w = w.index_put((self.locked_idx,), self.locked_sign * self._pos(self.theta_locked))
        return w.index_put((self.free_idx,), self.theta_free)

    def readout_weights(self):
        W = torch.empty(len(self.readout), dtype=self.dtype, device=self.device)
        W = W.index_put((self.r_locked_idx,), self.r_locked_sign * self._pos(self.W_locked))
        return W.index_put((self.r_free_idx,), self.W_free)

    def alpha(self):
        return F.softplus(self.alpha_raw)

    # ----- forward -----
    def forward(self, conditions=None, info=None):
        """conditions: list of row positions in the condition table (default: all).
        Returns dict X (n x 2c), S, delta (n x c), tf_pred (TF x c), gene_pred (G x c)."""
        d = self.design
        pos = list(range(d.C)) if conditions is None else list(conditions)
        cols = torch.tensor([j for i in pos for j in (2 * i, 2 * i + 1)], device=self.device)
        U = d.U[:, cols]
        S = d.S()[:, cols]
        X = simulate(self.struct, self.weights(), self.bias, U, S, info=info, **self.sim_opts)
        Y = S * X
        delta = Y[:, 1::2] - Y[:, 0::2]
        tf_pred = self.alpha()[:, None] * delta[self.tf_idx]
        contrib = self.readout_weights()[:, None] * delta[self.r_tf]
        gene_pred = torch.zeros(len(self.genes), len(pos), dtype=self.dtype, device=self.device)
        gene_pred = gene_pred.index_add(0, self.r_gene, contrib)
        return dict(X=X, S=S, delta=delta, tf_pred=tf_pred, gene_pred=gene_pred, conditions=pos)

    # ----- 9b: initialisation -----
    def pooled_reference(self):
        """Mean reference-arm input over training conditions (n x 1)."""
        d = self.design
        train = [i for i, s in enumerate(d.cond.source) if s != "holdout"]
        return d.U[:, [2 * i for i in train]].mean(dim=1, keepdim=True)

    @torch.no_grad()
    def initialize(self, rho_target=None, centre_target=None):
        """Rescale the bulk (T1-T3) so rho(|A|) <= rho_target, keeping the T0 route at
        its starting strength, then centre all biases exactly at the pooled
        reference input. Returns a report dict."""
        from .core import center_biases, perron_radius
        mc = self.cfg.model
        rho_target = rho_target or mc["rho_target"]
        centre_target = centre_target or mc["centre_target"]
        w0 = self.weights().clone()
        t0 = torch.tensor(self.tier == "T0", device=self.device)
        rho = lambda w: perron_radius(self.struct, w, max_iter=5000, tol=1e-7).item()
        route_only = torch.where(t0, w0, torch.zeros_like(w0))
        rho_route = rho(route_only)
        if rho_route >= rho_target:
            raise ValueError(f"route alone has rho(|A|) = {rho_route:.3f} >= {rho_target}")
        scaled = lambda s: torch.where(t0, w0, s * w0)
        rho_start = rho(w0)
        lo, hi = 0.0, 1.0
        if rho_start <= rho_target:
            # budget to spare (e.g. after deactivate_tiers): grow the bulk into it
            # instead of leaving the curated edges at a density-driven compromise
            lo = 1.0
            while hi < 64.0 and rho(scaled(hi)) <= rho_target:
                lo, hi = hi, 2.0 * hi
            if hi > 1.0:
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if rho(scaled(mid)) <= rho_target else (lo, mid)
        else:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if rho(scaled(mid)) <= rho_target else (lo, mid)
        w = scaled(lo)
        # write back into the parameters
        locked = w[self.locked_idx].abs()
        self.theta_locked.copy_(self._pos_inv(locked.clamp(min=1e-12)).to(self.dtype))
        self.theta_free.copy_(w[self.free_idx])
        U_ref = self.pooled_reference()
        b, err = center_biases(self.struct, self.weights(), U_ref, target=centre_target, leak=self.sim_opts["leak"])
        self.bias.copy_(b)
        return dict(rho_start=rho_start, rho_route=rho_route, bulk_scale=lo, rho_after=rho(self.weights()),
                    centre_error=err, reference_inputs={self.nodes[i]: round(v, 3) for i, v in
                                                        enumerate(U_ref[:, 0].tolist()) if v > 0})

    @torch.no_grad()
    def deactivate_tiers(self, tiers):
        """Set the weights of whole tiers to zero and report a mask of the edges left
        active. Zeroed edges contribute nothing to rho(|A|), so the stability budget
        they occupied becomes available to the tiers that remain: dropping T3 raises
        the admissible curated weight ~6x and the 3-hop gain ~200x (step 12d)."""
        drop = torch.tensor(np.isin(self.tier, list(tiers)), device=self.device)
        self.theta_free[drop[self.free_idx]] = 0.0
        if drop[self.locked_idx].any():
            self.theta_locked[drop[self.locked_idx]] = self._pos_inv(torch.tensor(1e-9)).to(self.dtype)
        return ~drop

    @torch.no_grad()
    def project_stability(self, rho_max=None, safety=0.99, rounds=3, order=None):
        """Hard stability projection: if rho(|A|) > rho_max, scale the non-T0 weights
        down until it is not. The soft spectral penalty is a nudge only; once the whole
        T3 backbone is trainable (stage B) the data gradient outruns it and rho passes 1,
        which makes the steady state - and therefore every prediction - undefined.
        rho is (almost exactly) linear in a uniform bulk scaling, so one corrective step
        usually suffices. Returns (rho_after, cumulative bulk scale)."""
        from .core import perron_radius
        rho_max = rho_max or self.cfg.loss["rho_max"]
        rho = lambda ww: perron_radius(self.struct, ww, max_iter=3000, tol=1e-6).item()
        r = rho(self.weights())
        if r <= rho_max:
            return r, 1.0
        # Shrink the least-supported tiers first. A uniform scaling of everything
        # non-T0 also shrinks the curated, anchored T1 edges, which is how stage B
        # lost 3.6x of its T1 strength and with it the drug fit (step 12c).
        groups = order or self.cfg.stage_b.get("projection_order", [["T3"], ["T2"], ["T1"]])
        total = 1.0
        for grp in groups:
            sel = torch.tensor(np.isin(self.tier, list(grp)), device=self.device)
            if not sel.any() or self.weights()[sel].abs().max() == 0:
                continue
            for _ in range(rounds):
                if r <= rho_max:
                    break
                s = max(1e-6, safety * rho_max / r)
                w = self.weights()
                w = torch.where(sel, s * w, w)
                self.theta_locked.copy_(self._pos_inv(w[self.locked_idx].abs().clamp(min=1e-12)).to(self.dtype))
                self.theta_free.copy_(w[self.free_idx])
                total *= s
                r = rho(self.weights())
            if r <= rho_max:
                break
        return r, total

    def parameter_summary(self):
        rows = [("signalling T" + t[-1], int((self.tier == t).sum())) for t in ["T0", "T1", "T2", "T3"]]
        rows += [("biases", self.bias.numel()), ("drug kappa", self.kappa_theta.numel()),
                 ("TF output scales", self.alpha_raw.numel()),
                 ("readout signed", self.W_locked.numel()), ("readout unsigned", self.W_free.numel())]
        return pd.DataFrame(rows, columns=["group", "n"])


def build_model(cfg, device="cpu", seed=0):
    """Model from the step 5 / step 6 outputs."""
    out = cfg.paths.output_dir
    edges = pd.read_csv(out / "step05_signalling_edges.csv.gz",
                        usecols=["source", "target", "sign", "tier", "context_dependent"], low_memory=False)
    readout = pd.read_csv(out / "step05_readout_edges.csv.gz")
    tfs = pd.read_csv(out / "step05_tf_outputs.csv").tf.tolist()
    cond = pd.read_csv(out / "step06_conditions.csv")
    dtype = getattr(torch, cfg.model.get("dtype", "float32"))
    return LembasModel(cfg, edges, readout, tfs, cond, device=device, dtype=dtype, seed=seed)
