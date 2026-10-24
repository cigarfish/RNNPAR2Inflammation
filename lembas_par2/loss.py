"""Step 10: loss.

10a  Data terms, each a weighted mean so terms are comparable, then scaled by lambda:
     TF    per condition: mean over measured TFs of (alpha*delta_TF - t_ULM)^2,
           averaged over conditions with weight w_c
     gene  per condition: padj-weighted mean over measured readout genes of
           (pred - log2FC)^2, averaged over conditions with weight w_c
     dir   perturbation sign constraints: (relu(margin - s * value))^2 for up/down,
           value^2 for 'unchanged'; value = node delta (signal layer) or predicted
           log2FC (mrna layer); weighted mean over constraints
     Holdout conditions have weight 0 and their constraints are excluded.

10b  Priors and regularisation (relative to the initialised model, so lambdas are
     comparable across tiers of very different weight scale):
     anchor   T0/T1: mean ((|w| - |w0|) / |w0|)^2       pulls curated edges to prior strength
     l1       T2/T3: mean |w| / mean |w0|               keeps the unsigned / STRING bulk sparse
     readout  mean |W| / mean |W0|
     bias     mean (b - b0)^2
     spectral relu(rho(|A|) - rho_max)^2                keeps the steady state guaranteed
     route    simulated MEK inhibition under the reference input: Bax, Bak1, mtDNA up
              and ERK down by >= margin (squared hinge) - deep supervision mid-route
     mode 'unanchored': T0 anchored like T1 and no route term (comparison model).
"""
from .core import perron_radius, simulate
import numpy as np
import pandas as pd
import torch


class Targets:
    """Step 6 targets aligned to a LembasModel (TF order, gene order, condition order)."""

    def __init__(self, cfg, model):
        lc = cfg.loss
        out = cfg.paths.output_dir
        dev, dt = model.device, model.dtype
        cond = model.design.cond
        self.condition_ids = list(cond.condition_id)
        C = len(cond)
        self.weight = torch.tensor(cond.weight.astype(float).values, dtype=dt, device=dev)
        self.weight[torch.tensor((cond.source == "holdout").values, device=dev)] = 0.0

        tfa = pd.read_csv(out / "step06_tf_activity.csv.gz", index_col=0)
        lfc = pd.read_csv(out / "step06_gene_lfc.csv.gz", index_col=0)
        padj = pd.read_csv(out / "step06_gene_padj.csv.gz", index_col=0)
        tf = tfa.reindex(index=model.tf_outputs, columns=self.condition_ids).clip(-lc["tf_target_clip"], lc["tf_target_clip"])
        g = lfc.reindex(index=model.genes, columns=self.condition_ids)
        p = padj.reindex(index=model.genes, columns=self.condition_ids)
        self.has_matrix = torch.tensor([c in tfa.columns for c in self.condition_ids], device=dev)
        t = lambda df: torch.tensor(df.values, dtype=dt, device=dev)
        self.tf, self.tf_mask = t(tf.fillna(0)), t(tf.notna().astype(float))
        self.gene, self.gene_mask = t(g.fillna(0)), t(g.notna().astype(float))
        pw = lc["gene_padj_weight"]
        gw = pw["floor"] + pw["gain"] * np.minimum(-np.log10(p.clip(lower=1e-300)), pw["cap"])
        self.gene_w = t(gw.fillna(pw["floor"]) * g.notna())

        d = pd.read_csv(out / "step06_directions.csv")
        d = d[d.source == "perturb"]
        pos = {c: i for i, c in enumerate(self.condition_ids)}
        gpos = {g_: i for i, g_ in enumerate(model.genes)}
        keep, rows = [], []
        for r in d.itertuples():
            if r.condition_id not in pos:
                continue
            if r.layer == "signal" and r.node in model.idx:
                rows.append((pos[r.condition_id], model.idx[r.node], 0, r.sign, r.weight))
            elif r.layer == "mrna" and r.node in gpos:
                rows.append((pos[r.condition_id], gpos[r.node], 1, r.sign, r.weight))
            else:
                continue
            keep.append(r.observation_id)
        self.dropped_directions = sorted(set(d.observation_id) - set(keep))
        a = np.array(rows, dtype=float).reshape(-1, 5)
        self.d_cond = torch.tensor(a[:, 0], dtype=torch.long, device=dev)
        self.d_index = torch.tensor(a[:, 1], dtype=torch.long, device=dev)
        self.d_mrna = torch.tensor(a[:, 2] == 1, device=dev)
        self.d_sign = torch.tensor(a[:, 3], dtype=dt, device=dev)
        self.d_weight = torch.tensor(a[:, 4], dtype=dt, device=dev)
        self.margin, self.unchanged = lc["direction_margin"], lc["unchanged_scale"]
        self.lambdas = dict(tf=lc["lambda_tf"], gene=lc["lambda_gene"], direction=lc["lambda_direction"])
        self.C = C
        self.configure_direction(**dict(lc.get("direction", {})))
        self.configure_tf(lc.get("tf_mode", "absolute"), lc.get("tf_gain_nonneg", True))
        self.last_tf_gain = None
        self.last_gain_frac_negative = None
        # per-condition source code for the step-12j readout nuisance (drug 0, geo 1, other 2)
        code = {"drug": 0, "geo": 1}
        self.source_code = torch.tensor([code.get(x, 2) for x in cond.source], device=dev)
        self.nuis = None

    def configure_direction(self, form="hinge2", normalize="global", cap=0.25, scale=0.1,
                            layer_weight=None):
        """How the perturbation-paper sign constraints enter the loss.

        form       hinge2          (relu(margin - s*v))^2, unbounded  [default, step 10a]
                   hinge2_capped   the violation is clipped at `cap` before squaring
                   logistic        softplus(-(s*v)/scale): bounded gradient, saturates
        normalize  global          one weighted mean over all constraints  [default]
                   per_condition   weighted mean within each condition, then a
                                   condition-weighted mean, as the TF and gene terms do
        layer_weight  extra factor per layer, e.g. {'mrna': 0.5}: a predicted log2FC
                   passes through a readout that is itself still being fitted, so it is
                   weaker evidence than a signalling-node direction.

        Step 12 found the default combination lets three far-off mRNA constraints
        (Snai2, Nanog, Csf2) make up 68% of the term and drag ERK into the leak region.
        """
        if form not in ("hinge2", "hinge2_capped", "logistic"):
            raise ValueError(f"unknown direction form {form}")
        if normalize not in ("global", "per_condition"):
            raise ValueError(f"unknown direction normalisation {normalize}")
        self.dir_form, self.dir_norm, self.dir_cap, self.dir_scale = form, normalize, cap, scale
        lw = {"signal": 1.0, "mrna": 1.0, **(layer_weight or {})}
        self.dir_layer_weight = lw
        self.d_layer_w = torch.where(self.d_mrna, torch.as_tensor(lw["mrna"], dtype=self.d_weight.dtype,
                                                                  device=self.d_weight.device),
                                     torch.as_tensor(lw["signal"], dtype=self.d_weight.dtype,
                                                     device=self.d_weight.device))
        return self

    def configure_tf(self, mode="absolute", gain_nonneg=True, gain_max=1e4):
        """How predicted TF activity is compared with the measured ULM t-values.

        absolute        mean squared error on the raw values  [default, step 10a]
        gain_global     one closed-form scalar gain for the whole batch, then MSE
        gain_condition  one closed-form gain per condition, then MSE
        standardized    z-score prediction and target within each condition, then MSE
                        (equals 2(1 - Pearson r), i.e. a correlation loss)

        Rationale (step 12d): a ULM t-value is a standardised statistic (sd 1.64, up to
        +-10) while delta is in arbitrary activity units that rho(|A|) < 1 caps near
        1e-4, so the two do not share a scale. Under 'absolute' the cheapest solution is
        to predict ~0 - every variant measured sat at loss ~= the target variance - and
        the leftover gradient went into whatever else could move, which produced the
        step-11b bias drift and the step-12c T1 shrinkage. The gain is fitted in closed
        form and detached (plug-in / alternating least squares), so it absorbs the unit
        conversion instead of the network having to manufacture 10^4 of gain.

        A negative gain would fit a condition by predicting the *opposite* pattern and
        letting the scalar flip it, which leaves the sign of the response unconstrained -
        measured at 24/119 conditions (38 % of the drug conditions) with gain_condition
        in step 12e. For a directional question that is not a fit, so the gain is held
        >= 0 by default.
        """
        if mode not in ("absolute", "gain_global", "gain_condition", "standardized"):
            raise ValueError(f"unknown tf_mode {mode}")
        self.tf_mode = mode
        self.gain_lo = 0.0 if gain_nonneg else -gain_max
        self.gain_hi = gain_max
        return self

    # ---------- step 12j: per-source readout nuisance ----------
    SOURCES = ("drug", "geo")

    def configure_nuisance(self, intercept=(), gain="global", gene_intercept=True, gain_nonneg=True):
        """Readout  y_hat[t, c] = a_s * pred[t, c] + mu_s[t]  for the condition's source s.

        intercept       sources that get a per-TF intercept profile mu_s (577-vector) and,
                        with gene_intercept, a per-gene intercept on the gene term (no gain
                        there: the trainable readout W already carries the scale)
        gain            'global' (one a for all sources) or 'source' (a_drug, a_geo)

        a and mu are the exact joint weighted least-squares solution (TF fixed effects plus
        one slope) over the TRAINING conditions given to refit_nuisance, with the loss's
        own weights (w_c / n_tf_c per measured TF; gene: w_c * padj weight / its sum per
        condition); detached, like the step-12e gain. Why: every drug target is computed
        against the same 6 DMSO samples and carries the same "-DMSO noise" vector, and the
        GEO targets share a generic inflammation profile (step 12f0: constant profiles
        score 0.57 / 0.71). With nowhere else to put them, the cheapest fit is for the
        signalling weights to emit those source means - which is all the drug fit was
        (residualised r ~0.03, step 12f). With mu_s the gradient only rewards each
        condition's deviation from its source mean.
        """
        bad = set(intercept) - set(self.SOURCES)
        if bad or gain not in ("global", "source"):
            raise ValueError(f"bad nuisance spec intercept={intercept} gain={gain}")
        dev, dt = self.tf.device, self.tf.dtype
        self.nuis = dict(intercept=tuple(intercept), gain=gain, gene_intercept=bool(gene_intercept),
                         gain_lo=0.0 if gain_nonneg else -1e4, ready=False, refits=0)
        self.mu_tf = torch.zeros(3, self.tf.shape[0], dtype=dt, device=dev)
        self.mu_gene = torch.zeros(3, self.gene.shape[0], dtype=dt, device=dev)
        self.a_src = torch.ones(3, dtype=dt, device=dev)
        self.tf_mode = "nuisance"
        return self

    @torch.no_grad()
    def refit_nuisance(self, model, train_pos):
        """Refit a and mu on `train_pos` (training conditions only - never validation)."""
        N = self.nuis
        pos = torch.tensor(list(train_pos), dtype=torch.long, device=self.weight.device)
        out = model(conditions=list(train_pos))
        sc = self.source_code[pos]
        w = self.weight[pos] * self.has_matrix[pos]
        m = self.tf_mask[:, pos]
        v = m * (w / m.sum(0).clamp(min=1)).unsqueeze(0)              # loss weight per (TF, condition)
        p, y = out["tf_pred"].to(v.dtype), self.tf[:, pos]
        pc, yc = p.clone(), y.clone()
        bar = {}
        for k, s_ in enumerate(self.SOURCES):
            if s_ not in N["intercept"]:
                continue
            cols = sc == k
            V = v[:, cols].sum(1).clamp(min=1e-20)
            pb, yb = (v[:, cols] * p[:, cols]).sum(1) / V, (v[:, cols] * y[:, cols]).sum(1) / V
            bar[k] = (pb, yb)
            pc[:, cols], yc[:, cols] = p[:, cols] - pb[:, None], y[:, cols] - yb[:, None]

        def slope(cols):
            num, den = (v[:, cols] * pc[:, cols] * yc[:, cols]).sum(), (v[:, cols] * pc[:, cols] ** 2).sum()
            a = (num / den.clamp(min=1e-30)).nan_to_num(0.0, 0.0, 0.0)
            return a.clamp(N["gain_lo"], 1e4)

        if N["gain"] == "global":
            self.a_src[:] = slope(sc < 2)
        else:
            for k in range(2):
                self.a_src[k] = slope(sc == k) if bool((sc == k).any()) else 1.0
            self.a_src[2] = 1.0
        self.mu_tf.zero_()
        for k, (pb, yb) in bar.items():
            self.mu_tf[k] = yb - self.a_src[k] * pb
        # gene term: intercept only; minimises sum_c (w_c / G_c) gw (gene_pred + mu - y)^2 in mu
        self.mu_gene.zero_()
        if N["gene_intercept"]:
            gw = self.gene_w[:, pos]
            u = gw * (w / gw.sum(0).clamp(min=1e-12)).unsqueeze(0)
            r = self.gene[:, pos] - out["gene_pred"].to(u.dtype)
            for k, s_ in enumerate(self.SOURCES):
                if s_ in N["intercept"]:
                    cols = sc == k
                    self.mu_gene[k] = (u[:, cols] * r[:, cols]).sum(1) / u[:, cols].sum(1).clamp(min=1e-20)
        N["ready"] = True
        N["refits"] += 1
        self.last_tf_gain = float(self.a_src[0])
        self.last_gain_frac_negative = float((self.a_src[:2] < 0).float().mean())
        return self.nuisance_state()

    def nuisance_state(self):
        """JSON-able summary (saved in checkpoint meta)."""
        if self.nuis is None:
            return None
        N = self.nuis
        return dict(intercept=list(N["intercept"]), gain=N["gain"], gene_intercept=N["gene_intercept"],
                    refits=N["refits"], a_drug=float(self.a_src[0]), a_geo=float(self.a_src[1]),
                    mu_tf_norm={s_: float(self.mu_tf[k].norm()) for k, s_ in enumerate(self.SOURCES)},
                    mu_gene_norm={s_: float(self.mu_gene[k].norm()) for k, s_ in enumerate(self.SOURCES)})

    def balance_sources(self, model, train_pos):
        """Scale condition weights so drug and GEO contribute equally to the TF + gene data
        loss on the training set (their total weight is preserved). Returns the factors."""
        with torch.no_grad():
            saved = dict(self.lambdas)
            cw = {}
            for k, s_ in enumerate(self.SOURCES):
                cols = [i for i in train_pos if int(self.source_code[i]) == k and bool(self.has_matrix[i])]
                if not cols:
                    continue
                self.lambdas = dict(tf=saved["tf"], gene=saved["gene"], direction=0.0)
                L = data_loss(model(conditions=cols), self)["total"].item()
                cw[k] = (cols, float(self.weight[cols].sum()), L)
            self.lambdas = saved
            if len(cw) < 2:
                return {}
            Wtot = sum(W for _, W, _ in cw.values())
            raw = {k: 1.0 / max(W * L, 1e-20) for k, (_, W, L) in cw.items()}
            norm = Wtot / sum(raw[k] * cw[k][1] for k in cw)
            fac = {}
            for k, (cols, W, L) in cw.items():
                f = raw[k] * norm
                idx = torch.tensor(cols, device=self.weight.device)
                self.weight[idx] = self.weight[idx] * f
                fac[self.SOURCES[k]] = dict(factor=f, weight=W, loss=L)
            return fac

    def direction_spec(self):
        return dict(form=self.dir_form, normalize=self.dir_norm, cap=self.dir_cap, scale=self.dir_scale,
                    layer_weight=dict(self.dir_layer_weight))

    def direction_residuals(self, out, v, s):
        """Per-constraint loss contribution, before weighting."""
        if self.dir_form == "hinge2":
            up_down = torch.relu(self.margin - s * v) ** 2
            unchanged = self.unchanged * v ** 2
        elif self.dir_form == "hinge2_capped":
            up_down = torch.clamp(torch.relu(self.margin - s * v), max=self.dir_cap) ** 2
            unchanged = self.unchanged * torch.clamp(v.abs(), max=self.dir_cap) ** 2
        else:  # logistic
            up_down = torch.nn.functional.softplus(-(s * v - self.margin) / self.dir_scale) * self.dir_scale
            unchanged = self.unchanged * self.dir_scale * (
                torch.nn.functional.softplus((v.abs() - self.margin) / self.dir_scale))
        return torch.where(s != 0, up_down, unchanged)


def _weighted_condition_mean(per_cond, w):
    s = w.sum()
    return (per_cond * w).sum() / s if s > 0 else per_cond.sum() * 0.0


def data_loss(out, T):
    """out = model forward output (with out['conditions'] = condition positions)."""
    pos = torch.tensor(out["conditions"], dtype=torch.long, device=T.weight.device)
    w = T.weight[pos] * T.has_matrix[pos]

    m = T.tf_mask[:, pos]
    p, y = out["tf_pred"], T.tf[:, pos]
    n_tf = m.sum(0)
    if T.tf_mode == "nuisance":
        if not T.nuis["ready"]:
            raise RuntimeError("readout nuisance not fitted: call Targets.refit_nuisance(model, train) first")
        sc = T.source_code[pos]
        resid = T.a_src[sc].unsqueeze(0) * p + T.mu_tf[sc].T - y
    elif T.tf_mode == "absolute":
        T.last_tf_gain = None
        resid = p - y
    elif T.tf_mode in ("gain_global", "gain_condition"):
        dim = None if T.tf_mode == "gain_global" else 0
        with torch.no_grad():
            num = (p * y * m).sum() if dim is None else (p * y * m).sum(0)
            den = (p * p * m).sum() if dim is None else (p * p * m).sum(0)
            a = num / den.clamp(min=1e-20)
            # a = r * sd(y)/sd(p): it explodes for a condition whose prediction is ~0,
            # which only amplifies float noise, so bound it
            a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).clamp(T.gain_lo, T.gain_hi)
            T.last_gain_frac_negative = float((a < 0).float().mean()) if a.ndim else float(a < 0)
        T.last_tf_gain = float(a.mean() if dim is not None else a)
        resid = (a * p if dim is None else a.unsqueeze(0) * p) - y
    else:  # standardized: 2 (1 - Pearson r) per condition, as a plug-in least squares
        T.last_tf_gain = None

        def z(x, detach):
            """Centre and scale within each condition over the measured TFs. The mean
            and sd of the prediction are detached: sqrt() has an infinite gradient at 0,
            which turned the whole run to NaN, and a detached normaliser makes this the
            same plug-in (alternating least squares) scheme as the fitted gain."""
            mu = (x * m).sum(0) / n_tf.clamp(min=1)
            xc = (x - mu.detach().unsqueeze(0) if detach else x - mu.unsqueeze(0)) * m
            var = (xc.detach() if detach else xc).pow(2).sum(0) / n_tf.clamp(min=1)
            sd = (var + 1e-12).sqrt()
            return xc / sd.unsqueeze(0)
        resid = z(p, True) - z(y, False)
    per = (resid ** 2 * m).sum(0) / n_tf.clamp(min=1)
    L_tf = _weighted_condition_mean(per, w * (n_tf >= (3 if T.tf_mode == "standardized" else 1)))

    gw = T.gene_w[:, pos]
    gp = out["gene_pred"]
    if T.tf_mode == "nuisance" and T.nuis["gene_intercept"]:
        gp = gp + T.mu_gene[T.source_code[pos]].T
    per = ((gp - T.gene[:, pos]) ** 2 * gw).sum(0) / gw.sum(0).clamp(min=1e-12)
    L_gene = _weighted_condition_mean(per, w * (gw.sum(0) > 0))

    col = torch.full((T.C,), -1, dtype=torch.long, device=pos.device)
    col[pos] = torch.arange(len(pos), device=pos.device)
    sel = col[T.d_cond] >= 0
    if sel.any():
        c = col[T.d_cond[sel]]
        idx, mrna, s = T.d_index[sel], T.d_mrna[sel], T.d_sign[sel]
        dw = T.d_weight[sel] * T.d_layer_w[sel]
        v_sig = out["delta"][idx.clamp(max=out["delta"].shape[0] - 1), c]
        v_mrna = out["gene_pred"][idx.clamp(max=out["gene_pred"].shape[0] - 1), c]
        v = torch.where(mrna, v_mrna, v_sig)
        lossr = T.direction_residuals(out, v, s)
        if T.dir_norm == "global":
            L_dir = (lossr * dw).sum() / dw.sum()
        else:  # per_condition: no single paper can dominate the term
            num = torch.zeros(len(pos), dtype=lossr.dtype, device=lossr.device).index_add(0, c, lossr * dw)
            den = torch.zeros(len(pos), dtype=lossr.dtype, device=lossr.device).index_add(0, c, dw)
            L_dir = _weighted_condition_mean(num / den.clamp(min=1e-12), T.weight[pos] * (den > 0))
    else:
        L_dir = out["tf_pred"].sum() * 0.0
    lam = T.lambdas
    total = lam["tf"] * L_tf + lam["gene"] * L_gene + lam["direction"] * L_dir
    return dict(total=total, tf=L_tf, gene=L_gene, direction=L_dir)


# ---------- 10b: priors, regularisation, full loss ----------
def balance_data_terms(model, T, conditions=None):
    """Set T.lambdas so that each data term equals 1 for the current model.
    `conditions` restricts the measurement to those condition positions (e.g. the
    training set of a stage); otherwise all conditions are used."""
    with torch.no_grad():
        saved = dict(T.lambdas)
        T.lambdas = dict(tf=1.0, gene=1.0, direction=1.0)
        L = data_loss(model(conditions=conditions), T)
        T.lambdas = {k: (1.0 / L[k].item() if L[k].item() > 0 else saved[k]) for k in ["tf", "gene", "direction"]}
    return T.lambdas


class Priors:
    def __init__(self, cfg, model, mode=None):
        lc = cfg.loss
        self.mode = mode or lc.get("mode", "anchored")
        if self.mode not in ("anchored", "unanchored"):
            raise ValueError(f"unknown mode {self.mode}")
        dev, dt = model.device, model.dtype
        with torch.no_grad():
            self.w0 = model.weights().detach().clone()
            self.W0_mean = model.readout_weights().abs().mean().item()
            self.b0 = model.bias.detach().clone()
        self.tier = {t: torch.tensor(model.tier == t, device=dev) for t in ["T0", "T1", "T2", "T3"]}
        self.l1_scale = {t: self.w0[self.tier[t]].abs().mean().item() for t in ["T2", "T3"]}
        self.lam_anchor = dict(lc["lambda_anchor"])
        rc = dict(lc["route_constraint"])
        if self.mode == "unanchored":
            self.lam_anchor["T0"] = self.lam_anchor["T1"]
            rc["lambda"] = 0.0
        self.lam = dict(l1=dict(lc["lambda_l1"]), readout=lc["lambda_readout_l1"], bias=lc["lambda_bias"],
                        spectral=lc["lambda_spectral"], route=rc["lambda"])
        self.rho_max = lc["rho_max"]
        # route-constraint experiment: reference vs MEK inhibition, same input
        n = len(model.nodes)
        if rc["context"] == "pooled":
            u = model.pooled_reference()
        else:
            u = torch.zeros(n, 1, dtype=dt, device=dev)
            for x in rc["context"]:
                u[model.idx[x], 0] = 1.0
        self.r_U = u.repeat(1, 2)
        S = torch.ones(n, 2, dtype=dt, device=dev)
        for x in rc["inhibit"]:
            S[model.idx[x], 1] = 1.0 - rc["kappa"]
        self.r_S = S
        self.r_up = torch.tensor([model.idx[x] for x in rc["up"]], device=dev)
        self.r_down = torch.tensor([model.idx[x] for x in rc["down"]], device=dev)
        self.r_margin = rc["margin"]
        # optional variant terms (configure)
        self.readout_free_mult = 1.0          # step 18: extra L1 on unsigned readout edges only
        self.t1_target = torch.zeros_like(self.tier["T1"])
        self.lam_target = None
        self.act = None
        self._model_ref = model

    def route_nodes_mask(self):
        """Bool mask of the T0 route's own nodes."""
        m = self._model_ref
        e = m.edges[m.tier == "T0"]
        route = set(e.source) | set(e.target)
        return torch.tensor([n in route for n in m.nodes], device=m.device)

    def configure(self, lambda_bias=None, target_edge_anchor=None, activation=None,
                  readout_free_mult=None):
        """Stage A variants.
        lambda_bias          override the bias penalty
        target_edge_anchor   anchor lambda for T1 edges leaving any inhibited node
                             (drug targets, KO targets); the rest of T1 keeps lambda_anchor T1
        activation           dict(lam, lo, hi, nodes=bool mask, reduce='sum'|'mean'):
                             penalise pre-activations under the pooled reference outside
                             [lo, hi] (squared). Use reduce='sum' over few nodes to keep
                             a single node from drifting: a mean over thousands of nodes
                             dilutes one violation to nothing (step 12 found ERK alone
                             falling into the leak region, which cost the route 100x gain
                             at its first hop). Being a statement about the route, it is
                             dropped in 'unanchored' mode together with the route term."""
        m = self._model_ref
        if lambda_bias is not None:
            self.lam["bias"] = lambda_bias
        if target_edge_anchor is not None:
            inh = {x for v in m.design.cond.trt_inhibit for x in v}
            src = torch.tensor(m.edges.source.isin(inh).values, device=m.device)
            self.t1_target = self.tier["T1"] & src
            self.lam_target = target_edge_anchor
        if readout_free_mult is not None:
            self.readout_free_mult = float(readout_free_mult)
        if activation is not None and self.mode != "unanchored":
            self.act = dict(activation)
            self.act.setdefault("reduce", "sum")
            self.act_U = m.pooled_reference()
        return self

    def terms(self, model):
        w = model.weights()
        t = {}
        m0 = self.tier["T0"]
        t["anchor_T0"] = (((w[m0].abs() - self.w0[m0].abs()) / self.w0[m0].abs().clamp(min=1e-12)) ** 2).mean()
        rel = lambda m: (((w[m].abs() - self.w0[m].abs()) / self.w0[m].abs().clamp(min=1e-12)) ** 2).mean()
        if self.lam_target is not None:
            t["anchor_T1"] = rel(self.tier["T1"] & ~self.t1_target)
            t["anchor_T1_targets"] = rel(self.t1_target)
        else:
            t["anchor_T1"] = rel(self.tier["T1"])
        for k in ["T2", "T3"]:
            t[f"l1_{k}"] = w[self.tier[k]].abs().mean() / max(self.l1_scale[k], 1e-12)
        W = model.readout_weights()
        t["readout_l1"] = W.abs().mean() / max(self.W0_mean, 1e-12)
        if self.readout_free_mult != 1.0 and len(model.r_free_idx):
            # the extra (mult - 1) share, so mult = 1 reproduces the baseline exactly
            t["readout_l1_free"] = W[model.r_free_idx].abs().mean() / max(self.W0_mean, 1e-12)
        t["bias"] = ((model.bias - self.b0) ** 2).mean()
        rho = perron_radius(model.struct, w, max_iter=3000, tol=1e-6)
        t["rho"] = rho.detach()
        t["spectral"] = torch.relu(rho - self.rho_max) ** 2
        if self.lam["route"] > 0:
            d = self.route_delta(model)
            viol = torch.cat([torch.relu(self.r_margin - d[self.r_up]), torch.relu(self.r_margin + d[self.r_down])])
            t["route"] = (viol ** 2).mean()
        else:
            t["route"] = w.sum() * 0.0
        if self.act is not None:
            from .core import mml_inverse
            X = simulate(model.struct, w, model.bias, self.act_U, None, **model.sim_opts)
            z = mml_inverse(X[:, 0].clamp(max=0.999))[self.act["nodes"]]
            viol = torch.relu(self.act["lo"] - z) ** 2 + torch.relu(z - self.act["hi"]) ** 2
            t["activation"] = viol.sum() if self.act["reduce"] == "sum" else viol.mean()
        return t

    def route_delta(self, model):
        X = simulate(model.struct, model.weights(), model.bias, self.r_U, self.r_S, **model.sim_opts)
        Y = self.r_S * X
        return Y[:, 1] - Y[:, 0]

    def weighted(self, t):
        lam = self.lam
        total = (self.lam_anchor["T0"] * t["anchor_T0"] + self.lam_anchor["T1"] * t["anchor_T1"]
                 + lam["l1"]["T2"] * t["l1_T2"] + lam["l1"]["T3"] * t["l1_T3"] + lam["readout"] * t["readout_l1"]
                 + lam["bias"] * t["bias"] + lam["spectral"] * t["spectral"] + lam["route"] * t["route"])
        if "anchor_T1_targets" in t:
            total = total + self.lam_target * t["anchor_T1_targets"]
        if "readout_l1_free" in t:
            total = total + lam["readout"] * (self.readout_free_mult - 1.0) * t["readout_l1_free"]
        if "activation" in t:
            total = total + self.act["lam"] * t["activation"]
        return total


def full_loss(model, T, P, conditions=None, info=None):
    """Forward + data terms + prior terms. Returns (total, dict of all terms, forward output)."""
    out = model(conditions=conditions, info=info)
    D = data_loss(out, T)
    R = P.terms(model)
    prior = P.weighted(R)
    total = D["total"] + prior
    terms = {**{f"data_{k}": v for k, v in D.items() if k != "total"}, "data": D["total"], "prior": prior, **R}
    return total, terms, out
