"""Audit of the curated TF -> gene signs that J0A (locked_param = abs) drove to zero (step 14).

Per signed readout edge, over the 5 J0A seeds: how often it ended at exactly 0 (the sign clamp),
its J0 (softplus) weight, its source databases, and a model-free check: the correlation across
TRAINING conditions between the TF's measured activity (step-6 ULM target) and the gene's
measured log2FC. If the data oppose a curated sign, that correlation should tend to have the
opposite sign. (Caveat: the TF activity is computed partly from this gene, which biases the
correlation towards the curated sign, so a negative agreement is the stronger signal.)

    python tests/explore_readout_sign_audit.py
Writes outputs/step14/readout_audit.csv and readout_audit_route.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.eval_harness import load_model, split

ROUTE_GENES = ["Ifnb1", "Cgas", "Sting1", "Isg15", "Ifit1", "Ifit3", "Rsad2", "Mx1", "Oasl1", "Irf7", "Usp18", "Cxcl10"]


def main():
    cfg = load_config()
    o = cfg.paths.output_dir
    out = o / "step14"
    out.mkdir(exist_ok=True)
    W = {}
    for v in ["J0", "J0A"]:
        ws = []
        for s in range(5):
            m, _ = load_model(cfg, o / "source_variants" / (v if s == 0 else f"{v}_s{s}") / "checkpoint.pt", "cpu")
            ws.append(m.readout_weights().detach().numpy())
        W[v] = np.stack(ws)
    ro = m.readout.reset_index(drop=True).copy()
    signed = ro.sign.values != 0
    ro["zero_frac_J0A"] = (np.abs(W["J0A"]) < 1e-6).mean(0)
    ro["w_J0_median"] = np.median(W["J0"], 0)
    ro["w_J0A_median"] = np.median(W["J0A"], 0)

    # model-free: corr over training conditions of measured TF activity vs measured gene log2FC
    tfa = pd.read_csv(o / "step06_tf_activity.csv.gz", index_col=0)
    lfc = pd.read_csv(o / "step06_gene_lfc.csv.gz", index_col=0)
    cond = m.design.cond
    _, train, _ = split(cfg, cond)
    cols = [cond.condition_id.iloc[i] for i in train if cond.condition_id.iloc[i] in tfa.columns and cond.condition_id.iloc[i] in lfc.columns]
    A = tfa.reindex(columns=cols)
    L = lfc.reindex(columns=cols)

    def corr(tf, g):
        if tf not in A.index or g not in L.index:
            return np.nan
        a, b = A.loc[tf].values.astype(float), L.loc[g].values.astype(float)
        k = np.isfinite(a) & np.isfinite(b)
        if k.sum() < 10:
            return np.nan
        return float(np.corrcoef(a[k], b[k])[0, 1])

    ro["data_corr"] = [corr(t, g) for t, g in zip(ro.tf, ro.gene)]
    ro["data_agrees_with_sign"] = np.where(ro.sign != 0, np.sign(ro.data_corr) == ro.sign, np.nan)
    ro["source"] = ro.dbs.str.replace("collectri;trrust", "both")
    ro.to_csv(out / "readout_audit.csv", index=False)

    s = ro[signed].copy()
    s["zeroed"] = s.zero_frac_J0A >= 0.6          # zero in >= 3 of 5 seeds
    print(f"signed readout edges: {len(s)}; zeroed in >= 3/5 J0A seeds: {int(s.zeroed.sum())} ({s.zeroed.mean():.0%})")
    print("\n== by curated sign")
    print(s.groupby(s.sign.map({1: 'activation', -1: 'repression'})).zeroed.agg(["size", "mean"]).to_string(float_format=lambda x: f"{x:.2f}"))
    print("\n== by source")
    print(s.groupby("source").zeroed.agg(["size", "mean"]).to_string(float_format=lambda x: f"{x:.2f}"))
    print("\n== model-free check: does the data correlation agree with the curated sign?")
    g = s.dropna(subset=["data_corr"]).groupby("zeroed").agg(n=("data_corr", "size"),
                                                          agrees=("data_agrees_with_sign", "mean"),
                                                          median_signed_corr=("data_corr", lambda x: float(np.median(x * s.loc[x.index, "sign"]))))
    print(g.to_string(float_format=lambda x: f"{x:.3f}"))
    tfz = s.groupby("tf").agg(n=("zeroed", "size"), zeroed=("zeroed", "mean")).query("n >= 20").sort_values("zeroed", ascending=False)
    print("\n== TFs with the most zeroed edges (>= 20 signed targets)")
    print(tfz.head(10).to_string(float_format=lambda x: f"{x:.2f}"))

    r = ro[ro.gene.isin(ROUTE_GENES)].copy()
    r = r.sort_values(["gene", "sign", "tf"])
    r.to_csv(out / "readout_audit_route.csv", index=False)
    print("\n== route readout genes: signed edges zeroed in J0A, and the unsigned (learned-sign) edges")
    for gname in ROUTE_GENES:
        x = r[r.gene == gname]
        if not len(x):
            continue
        sg = x[x.sign != 0]
        z = sg[sg.zero_frac_J0A >= 0.6]
        un = x[x.sign == 0]
        print(f"{gname:7s} signed {len(sg):3d}, zeroed {len(z):3d}"
              + (f": {', '.join(f'{t}({int(sv):+d}, data r {c:+.2f})' for t, sv, c in zip(z.tf, z.sign, z.data_corr))}" if len(z) else "")
              + (f" | unsigned: {', '.join(f'{t} J0 {a:+.2f} / J0A {b:+.2f}' for t, a, b in zip(un.tf, un.w_J0_median, un.w_J0A_median))}" if len(un) else ""))
    print(f"\nwrote {out / 'readout_audit.csv'}")


if __name__ == "__main__":
    main()
