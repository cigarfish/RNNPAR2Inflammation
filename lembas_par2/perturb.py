"""Step 8: how experiments are simulated.

8a  Design: encode every condition as two simulation arms (columns).
        column 2c   = reference arm of condition c  (ref_inputs)
        column 2c+1 = treated arm                    (trt_inputs, trt_inhibit)
    U (n x 2C): input_level at input nodes.
    S (n x 2C): output scale, 1 - kappa at inhibited nodes, else 1.
        kappa fixed by intervention type (KO 0.95, siRNA/knockdown/antagonist 0.7, ...)
        or learned per drug: kappa_d = sigmoid(theta_d), initialised at drug_kappa_init.

8b  Two-arm effects: delta = effective activity (S*X) of the treated arm minus the
    reference arm. Inhibition removes the node's output, so downstream nodes move
    by the signs of the paths from it, and the inhibited node itself reads as down.
"""
import json

import numpy as np
import pandas as pd
import torch


def _logit(p):
    return float(np.log(p / (1 - p)))


def intervention_kinds(cfg, cond):
    """Kappa source for each condition's inhibited nodes: 'drug', 'geo_chemical' or
    the perturbation intervention_type of that contrast."""
    iv = pd.read_csv(cfg.paths.perturbation_interventions, encoding="utf-8-sig")
    by_contrast = {}
    for r in iv.itertuples():
        for t in str(r.graph_target_id).split(";"):
            by_contrast[(r.contrast_id, t)] = r.intervention_type
    kinds = []
    for r in cond.itertuples():
        if r.source == "drug":
            kinds.append({t: "drug" for t in r.trt_inhibit})
        elif r.source == "geo":
            kinds.append({t: "geo_chemical" for t in r.trt_inhibit})
        else:
            kinds.append({t: by_contrast.get((r.condition_id, t), "knockout") for t in r.trt_inhibit})
    return kinds


class Design:
    """Inputs and inhibition of all condition arms on a fixed node index."""

    def __init__(self, cfg, cond, nodes, device="cpu", dtype=torch.float32):
        pc = cfg.perturbation
        cond = cond.copy()
        for col in ["ref_inputs", "trt_inputs", "trt_inhibit"]:
            if isinstance(cond[col].iloc[0], str):
                cond[col] = cond[col].map(json.loads)
        self.cond = cond.reset_index(drop=True)
        self.nodes = list(nodes)
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        self.n, self.C = len(self.nodes), len(self.cond)
        self.device, self.dtype = device, dtype
        missing = sorted({x for col in ["ref_inputs", "trt_inputs", "trt_inhibit"]
                          for v in self.cond[col] for x in v} - set(self.idx))
        if missing:
            raise ValueError(f"design nodes missing from the network: {missing}")

        U = torch.zeros(self.n, 2 * self.C, dtype=dtype)
        for c, r in enumerate(self.cond.itertuples()):
            for x in r.ref_inputs:
                U[self.idx[x], 2 * c] = pc["input_level"]
            for x in r.trt_inputs:
                U[self.idx[x], 2 * c + 1] = pc["input_level"]
        self.U = U.to(device)

        kinds = intervention_kinds(cfg, self.cond)
        drugs = sorted({r.name for r in self.cond.itertuples() if r.source == "drug"})
        self.drugs = drugs
        self.theta = torch.full((len(drugs),), _logit(pc["drug_kappa_init"]), dtype=dtype, device=device,
                                requires_grad=bool(pc["learn_drug_kappa"]))
        fixed_rows, fixed_cols, fixed_val, drug_rows, drug_cols, drug_ids = [], [], [], [], [], []
        self.inhibition = []
        for c, (r, kd) in enumerate(zip(self.cond.itertuples(), kinds)):
            for t, kind in kd.items():
                row, col = self.idx[t], 2 * c + 1
                if kind == "drug":
                    drug_rows.append(row); drug_cols.append(col); drug_ids.append(drugs.index(r.name))
                    self.inhibition.append(dict(condition_id=r.condition_id, node=t, kind=kind, kappa="learned"))
                else:
                    k = pc["kappa"].get(kind)
                    if k is None:
                        raise ValueError(f"no kappa for intervention type {kind!r} ({r.condition_id})")
                    fixed_rows.append(row); fixed_cols.append(col); fixed_val.append(k)
                    self.inhibition.append(dict(condition_id=r.condition_id, node=t, kind=kind, kappa=k))
        t = lambda a: torch.tensor(a, dtype=torch.long, device=device)
        self._fixed = (t(fixed_rows), t(fixed_cols), torch.tensor(fixed_val, dtype=dtype, device=device))
        self._drug = (t(drug_rows), t(drug_cols), t(drug_ids))
        self.inhibition = pd.DataFrame(self.inhibition)

    # ----- kappa and S -----
    def kappa(self):
        """Per-drug inhibition strength in (0, 1)."""
        return torch.sigmoid(self.theta)

    def S(self):
        """Output-scale matrix (n x 2C); differentiable w.r.t. theta."""
        S = torch.ones(self.n, 2 * self.C, dtype=self.dtype, device=self.device)
        r, c, v = self._fixed
        S = S.index_put((r, c), 1 - v)
        r, c, d = self._drug
        if r.numel():
            S = S.index_put((r, c), 1 - self.kappa()[d])
        return S

    # ----- arms -----
    def ref_cols(self):
        return torch.arange(0, 2 * self.C, 2, device=self.device)

    def trt_cols(self):
        return torch.arange(1, 2 * self.C, 2, device=self.device)

    def delta(self, X, S=None):
        """Treated minus reference per condition (n x C). With S, the effective
        activity S*X is compared, so an inhibited node itself reads as reduced
        (its state X is upstream-driven; its output to the network is S*X)."""
        Y = X if S is None else S * X
        return Y[:, self.trt_cols()] - Y[:, self.ref_cols()]

    def subset(self, condition_ids):
        """Column indices (both arms) of selected conditions, e.g. one CV fold."""
        pos = [i for i, c in enumerate(self.cond.condition_id) if c in set(condition_ids)]
        return torch.tensor([j for i in pos for j in (2 * i, 2 * i + 1)], dtype=torch.long, device=self.device)
