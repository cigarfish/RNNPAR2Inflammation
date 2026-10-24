"""Step 15: GSE275623 (Cgas-KO BMDM, LPS 100 ng/ml 2 h) - validation only. Rules: config `cgas_test`,
fixed 2026-10-17 before any data were loaded.

A1 sanity   C4 WT LPS vs WT untreated: Tnf, Cxcl1, Il1b, Ifnb1 each up >= 2 log2 (padj < 0.05), else stop.
            The paper's Cxcl1 / Tnf up in C1 (KO LPS vs WT LPS) is reported either way.
A2 STING    is cGAS -> STING active at 2 h LPS? ACTIVE iff in C1 the 8-gene ISG score < 0 with
            permutation p < 0.05 vs random 8-gene sets AND Ifnb1 log2FC < 0 with padj < 0.1.
A3 model    simulate Cgas KO (kappa 0.95, LPS inputs) in J0 and J0U (5 seeds each); sign agreement on
            the pre-listed readouts vs C1, call threshold 1e-3, one-sided binomial on the seed median.
C3 (ER-cGAS rescue, descriptive) is not run: the ER file has Ensembl ids only and no local
Ensembl -> symbol mapping covers the genes needed.

Needs pydeseq2 (WSL env gnn):
    python scripts/run_cgas_15.py
    python scripts/run_cgas_15.py --from-csv    # A1-A3 from saved DEG tables (e.g. where torch works)
Writes outputs/step15/{deg_C1.csv, deg_C2.csv, deg_C4.csv, model_C1.csv, report.json}.
"""
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from lembas_par2 import load_config
from lembas_par2.conditions import tf_activity, zscore
from lembas_par2.eval_harness import load_model
from lembas_par2.genes import symbol_map
from lembas_par2.step13 import binom_tail, ko_delta

SRC = ROOT / "data_external" / "GSE275623" / "GSE275623_BMDM_count_matrix.xlsx"
GROUPS = {"WT-UT": "WT_UT", "KO-UT": "KO_UT", "WT-LPS": "WT_LPS", "KO-LPS": "KO_LPS"}
CONTRASTS = {"C1": ("KO_LPS", "WT_LPS"), "C2": ("KO_UT", "WT_UT"), "C4": ("WT_LPS", "WT_UT")}


def deseq(cfg):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    d = pd.read_excel(SRC)
    m = symbol_map(cfg)
    d["Gene_symbol"] = d.Gene_symbol.astype(str).str.strip().map(lambda x: m.get(x, x))
    X = d.groupby("Gene_symbol").sum().T
    X = X.loc[:, X.sum() >= 10].astype(int)
    meta = pd.DataFrame({"group": [GROUPS[c.rsplit("-rep", 1)[0]] for c in X.index]}, index=X.index)
    assert meta.group.value_counts().eq(3).all(), meta.group.value_counts()
    dds = DeseqDataSet(counts=X, metadata=meta, design="~group", ref_level=["group", "WT_UT"], quiet=True)
    dds.deseq2()
    out = {}
    for k, (a, b) in CONTRASTS.items():
        st = DeseqStats(dds, contrast=["group", a, b], quiet=True)
        st.summary()
        r = st.results_df.rename_axis("gene")[["log2FoldChange", "padj", "baseMean"]]
        out[k] = r.rename(columns={"log2FoldChange": "log2fc"})
    return out


def isg_test(lfc, genes, n_perm, seed=0):
    z = zscore(lfc.dropna())
    have = [g for g in genes if g in z.index]
    obs = float(z[have].mean())
    rng = np.random.default_rng(seed)
    null = np.array([z.values[rng.choice(len(z), len(have), replace=False)].mean() for _ in range(n_perm)])
    return obs, float(np.mean(null <= obs)), have


@torch.no_grad()
def model_prediction(m, inputs, kappa):
    d = ko_delta(m, inputs, kappa, node="Cgas")
    contrib = m.readout_weights() * d[m.r_tf]
    gp = torch.zeros(len(m.genes), dtype=d.dtype).index_add(0, m.r_gene, contrib)
    return d, dict(zip(m.genes, gp.tolist()))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true",
                    help="reuse outputs/step15/deg_C*.csv instead of running DESeq2 (no pydeseq2 needed)")
    a = ap.parse_args()
    cfg = load_config()
    ct = cfg.cgas_test
    o = cfg.paths.output_dir
    out = o / "step15"
    out.mkdir(exist_ok=True)
    if a.from_csv:
        R = {k: pd.read_csv(out / f"deg_{k}.csv", index_col=0) for k in CONTRASTS}
    else:
        R = deseq(cfg)
        for k, r in R.items():
            r.to_csv(out / f"deg_{k}.csv")
    rep = {}

    # ---- A1
    c4, c1 = R["C4"], R["C1"]
    s = ct["sanity"]
    a1 = {g: dict(log2fc=float(c4.at[g, "log2fc"]), padj=float(c4.at[g, "padj"])) for g in s["lps_genes"] if g in c4.index}
    a1_ok = len(a1) == len(s["lps_genes"]) and all(v["log2fc"] >= s["lps_min_log2fc"] and v["padj"] < 0.05 for v in a1.values())
    paper = {g: dict(log2fc=float(c1.at[g, "log2fc"]), padj=float(c1.at[g, "padj"])) for g in s["paper_genes"] if g in c1.index}
    cg = {k: float(R[k].at["Cgas", "log2fc"]) if "Cgas" in R[k].index else None for k in R}
    rep["A1"] = dict(pass_=a1_ok, lps_genes_C4=a1, paper_genes_C1=paper, Cgas_log2fc=cg)
    print("[A1] LPS response (C4): " + ", ".join(f"{g} {v['log2fc']:+.2f} (padj {v['padj']:.1e})" for g, v in a1.items())
          + f" -> {'PASS' if a1_ok else 'FAIL'}")
    print("     paper genes in C1 (KO vs WT under LPS): " + ", ".join(f"{g} {v['log2fc']:+.2f} (padj {v['padj']:.1e})" for g, v in paper.items()))
    print(f"     Cgas itself: C1 {cg['C1']:+.2f}, C2 {cg['C2']:+.2f} log2 (knockout check)")
    if not a1_ok:
        rep["verdict"] = "STOP: the LPS sanity check failed"
        (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
        print(f"[VERDICT] {rep['verdict']}")
        return

    # ---- A2
    sa = ct["sting_arm"]
    obs, p, have = isg_test(c1.log2fc, sa["isg_genes"], 10000)
    ifnb = dict(log2fc=float(c1.at["Ifnb1", "log2fc"]), padj=float(c1.at["Ifnb1", "padj"]))
    active = obs < 0 and p < 0.05 and ifnb["log2fc"] < 0 and ifnb["padj"] < 0.1
    per_isg = {g: dict(log2fc=float(c1.at[g, "log2fc"]), padj=float(c1.at[g, "padj"])) for g in have}
    rep["A2"] = dict(isg_score=obs, perm_p=p, ifnb1=ifnb, per_gene=per_isg,
                     verdict="ACTIVE" if active else "NOT DETECTED")
    print(f"\n[A2] C1 ISG score {obs:+.3f} (permutation p {p:.4f}); Ifnb1 {ifnb['log2fc']:+.2f} (padj {ifnb['padj']:.2e})")
    print("     " + ", ".join(f"{g} {v['log2fc']:+.2f}" for g, v in per_isg.items()))
    print(f"     cGAS -> STING arm at 2 h LPS: {rep['A2']['verdict']}")

    # ---- A3
    mv = ct["model_validation"]
    reg = pd.read_csv(o / "step05_regulons.csv.gz")
    tfs = list(pd.read_csv(o / "step05_tf_outputs.csv").tf)
    tfa = tf_activity({"C1": c1[["log2fc"]]}, reg, tfs, cfg.conditions["min_tf_targets_condition"])["C1"]
    meas = {"Irf3_activity": float(tfa.get("Irf3", np.nan))}
    meas.update({g: float(c1.at[g, "log2fc"]) if g in c1.index else np.nan for g in mv["scored"] if g != "Irf3_activity"})
    rows = []
    for v in ["J0", "J0U"]:
        for sd in range(5):
            run = v if sd == 0 else f"{v}_s{sd}"
            m, _ = load_model(cfg, o / "source_variants" / run / "checkpoint.pt", "cpu")
            d, gp = model_prediction(m, mv["inputs"], mv["kappa"])
            pred = {"Irf3_activity": d[m.idx["Irf3"]].item()}
            pred.update({g: gp.get(g, np.nan) for g in mv["scored"] if g != "Irf3_activity"})
            agree = sum(1 for k in mv["scored"] if np.isfinite(pred[k]) and abs(pred[k]) >= mv["call_threshold"]
                        and np.isfinite(meas[k]) and np.sign(pred[k]) == np.sign(meas[k]))
            rows.append(dict(variant=v, seed=sd, agree=agree, n=len(mv["scored"]),
                             dSting1=d[m.idx["Sting1"]].item(), dCgas=d[m.idx["Cgas"]].item(),
                             **{f"pred_{k}": x for k, x in pred.items()}))
    t = pd.DataFrame(rows)
    t.to_csv(out / "model_C1.csv", index=False)
    rep["A3"] = dict(measured=meas, by_variant={})
    print(f"\n[A3] measured C1: " + ", ".join(f"{k} {x:+.2f}" for k, x in meas.items()))
    for v, g in t.groupby("variant"):
        k = int(np.median(g.agree))
        pv = binom_tail(k, int(g.n.iloc[0]))
        rep["A3"]["by_variant"][v] = dict(agree_median=k, n=int(g.n.iloc[0]), p=pv, agree_range=[int(g.agree.min()), int(g.agree.max())],
                                          dSting1_median=float(g.dSting1.median()))
        print(f"     {v:4s} agreement {k}/{int(g.n.iloc[0])} (range {g.agree.min()}-{g.agree.max()}, p {pv:.3f}); "
              f"simulated dSting1 {g.dSting1.median():+.4f}; predicted: "
              + ", ".join(f"{c[5:]} {g[c].median():+.4f}" for c in g.columns if c.startswith("pred_")))
    rep["verdict"] = f"A2 {rep['A2']['verdict']}; A3 J0 {rep['A3']['by_variant']['J0']['agree_median']}/{rep['A3']['by_variant']['J0']['n']} (p {rep['A3']['by_variant']['J0']['p']:.3f})"
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(f"\n[VERDICT] {rep['verdict']}\nwrote {out}")


if __name__ == "__main__":
    main()
