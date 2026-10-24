"""Exploration: which loss terms and conditions push the biases of SE_CYTO_MTDNA,
Cgas, Sting1 and Tbk1 during stage A.

For each bias b and each model state (init, trained):
  * prior terms: d(lambda * term)/db by autograd, one term at a time
  * data terms: dL/db = sum over outputs of dL/dpred * dpred/db, with dpred/db from
    a central finite difference of the forward pass (exact chain rule; checked
    against autograd of the full data loss). This splits the gradient by
    condition and by output (TF activity / gene readout).
A negative gradient means gradient descent pushes the bias UP.

Run from the LEMBAS folder after scripts/train_stage_a.py:
    python tests/explore_11b_bias_drivers.py
Writes outputs/stage_a/bias_drivers_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.loss import Priors, Targets, balance_data_terms, data_loss
from lembas_par2.model import build_model
from lembas_par2.train import load_checkpoint

NODES = ["SE_CYTO_MTDNA", "Cgas", "Sting1", "Tbk1"]
EPS = 1e-3
cfg = load_config()
dev = "cuda" if torch.cuda.is_available() else "cpu"
run = cfg.paths.output_dir / "stage_a"
torch.set_default_dtype(torch.float32)

# lambdas exactly as in the run: balanced on the initial model
m_init = build_model(cfg, device=dev)
load_checkpoint(m_init, run / "init.pt")
T = Targets(cfg, m_init)
balance_data_terms(m_init, T)
lam = dict(T.lambdas)


def prior_grads(m, P):
    out = {}
    names = {"anchor_T0": P.lam_anchor["T0"], "anchor_T1": P.lam_anchor["T1"], "l1_T2": P.lam["l1"]["T2"],
             "l1_T3": P.lam["l1"]["T3"], "readout_l1": P.lam["readout"], "bias": P.lam["bias"],
             "spectral": P.lam["spectral"], "route": P.lam["route"]}
    for k, l in names.items():
        m.zero_grad()
        t = P.terms(m)[k]
        if t.requires_grad and l > 0:
            (l * t).backward()
        g = m.bias.grad
        out[k] = {n: (g[m.idx[n]].item() if g is not None else 0.0) for n in NODES}
    return out


def data_attribution(m, pos):
    """Per-condition, per-output contributions to dL_data/db for each bias in NODES."""
    with torch.no_grad():
        o = m(conditions=pos)
        p = torch.tensor(pos, device=dev)
        w = T.weight[p] * T.has_matrix[p]
        # dL/dpred for the TF term
        mt = T.tf_mask[:, p]
        wtf = w * (mt.sum(0) > 0)
        dtf = lam["tf"] * (wtf / wtf.sum()) * 2 * (o["tf_pred"] - T.tf[:, p]) * mt / mt.sum(0).clamp(min=1)
        gw = T.gene_w[:, p]
        wg = w * (gw.sum(0) > 0)
        dge = lam["gene"] * (wg / wg.sum()) * 2 * (o["gene_pred"] - T.gene[:, p]) * gw / gw.sum(0).clamp(min=1e-12)
        res = {}
        for n in NODES:
            i = m.idx[n]
            b0 = m.bias[i].item()
            m.bias[i] = b0 + EPS; up = m(conditions=pos)
            m.bias[i] = b0 - EPS; dn = m(conditions=pos)
            m.bias[i] = b0
            s_tf = (up["tf_pred"] - dn["tf_pred"]) / (2 * EPS)
            s_ge = (up["gene_pred"] - dn["gene_pred"]) / (2 * EPS)
            res[n] = dict(tf=(dtf * s_tf), gene=(dge * s_ge), d_delta=(up["delta"] - dn["delta"]) / (2 * EPS))
    return o, res


def analyse(label, ckpt):
    m = build_model(cfg, device=dev)
    meta = load_checkpoint(m, ckpt)
    P = Priors(cfg, m_init)                        # priors anchored to the initial model, as in training
    P.w0, P.b0 = P.w0, P.b0
    meta_run = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)["meta"]
    pos = meta_run["train"]
    cond = m.design.cond
    print(f"\n==================== {label} ====================")
    # autograd check of the total data gradient
    m.zero_grad()
    data_loss(m(conditions=pos), T)["total"].backward()
    g_auto = {n: m.bias.grad[m.idx[n]].item() for n in NODES}
    o, res = data_attribution(m, pos)
    pg = prior_grads(m, P)
    rows = []
    for n in NODES:
        tf_c = res[n]["tf"].sum(0).cpu().numpy()
        ge_c = res[n]["gene"].sum(0).cpu().numpy()
        tot = tf_c.sum() + ge_c.sum()
        print(f"\n-- {n}: bias {m.bias[m.idx[n]].item():+.3f};  dL/db data: TF {tf_c.sum():+.2e}, gene {ge_c.sum():+.2e} "
              f"(sum {tot:+.2e}, autograd {g_auto[n]:+.2e});  priors: "
              + ", ".join(f"{k} {v[n]:+.1e}" for k, v in pg.items() if abs(v[n]) > 0))
        # by condition
        dfc = pd.DataFrame(dict(condition=[cond.name.iloc[i] for i in pos], cls=[cond.cls.iloc[i] for i in pos],
                                tf=tf_c, gene=ge_c))
        dfc["total"] = dfc.tf + dfc.gene
        print("   conditions pushing the bias UP (most negative gradient):")
        print(dfc.sort_values("total").head(6).round(5).to_string(index=False))
        print("   conditions pushing it DOWN:")
        print(dfc.sort_values("total", ascending=False).head(3).round(5).to_string(index=False))
        # by output
        tf_o = res[n]["tf"].sum(1).cpu().numpy()
        ge_o = res[n]["gene"].sum(1).cpu().numpy()
        out_df = pd.concat([pd.DataFrame(dict(output=m.tf_outputs, kind="TF", grad=tf_o)),
                            pd.DataFrame(dict(output=m.genes, kind="gene", grad=ge_o))])
        print("   outputs pushing the bias UP:")
        top = out_df.sort_values("grad").head(8)
        for r in top.itertuples():
            if r.kind == "TF":
                j = m.tf_outputs.index(r.output)
                pp = torch.tensor(pos, device=dev)
                msk = T.tf_mask[j, pp] > 0
                meas = T.tf[j, pp][msk].mean().item()
                pred = o["tf_pred"][j][msk].mean().item()
                sens = res[n]["d_delta"][m.idx[r.output]].mean().item()
                print(f"     TF   {r.output:10s} grad {r.grad:+.2e}  mean measured {meas:+.2f}  mean predicted {pred:+.2f}  "
                      f"d(activity)/d(bias) {sens:+.3f}")
            else:
                print(f"     gene {r.output:10s} grad {r.grad:+.2e}")
        dfc.assign(node=n, state=label).to_csv(run / f"bias_drivers_{label}_{n}_by_condition.csv", index=False)
        out_df.assign(node=n, state=label).to_csv(run / f"bias_drivers_{label}_{n}_by_output.csv", index=False)
        rows.append(dict(state=label, node=n, data_tf=tf_c.sum(), data_gene=ge_c.sum(), autograd=g_auto[n],
                         **{f"prior_{k}": v[n] for k, v in pg.items()}))
    return rows


rows = analyse("init", run / "init.pt") + analyse("trained", run / "checkpoint.pt")
pd.DataFrame(rows).to_csv(run / "bias_drivers_summary.csv", index=False)
print("\nsummary (negative = pushes the bias up):")
print(pd.DataFrame(rows).round(6).to_string(index=False))
