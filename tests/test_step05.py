"""Step 5 validation: signalling layer vs transcriptional readout layer.

Needs data/collectri_mouse.csv (run scripts/fetch_collectri.py first), unless
config readout.require_collectri is false.

Run from the LEMBAS folder:
    python tests/test_step05.py

Writes to outputs/:
    step05_signalling_edges.csv.gz  signalling layer (non-transcriptional edges)
    step05_nodes.csv                network nodes (universe genes, extension TFs, entities)
    step05_regulons.csv.gz          merged CollecTRI + TRRUST regulons
    step05_tf_outputs.csv           TF-activity outputs with signed measured target counts
    step05_readout_edges.csv.gz     TF -> gene readout edges (sign init)
    step05_extension_tfs.csv        TFs added beyond the universe and their curated inputs
    step05_log2fc.csv.gz / step05_padj.csv.gz   measured matrices for all readout genes
"""
import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.genes import load_universe
from lembas_par2.readout import build_layers

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
rc = cfg.readout
out = cfg.paths.output_dir
U = set(load_universe(cfg))
L = build_layers(cfg)
reg, S = L["regulons"], L["signalling"]

# ---------- 1. regulons ----------
print("== regulons")
by_db = reg.dbs.value_counts().to_dict()
n_col = int(reg.dbs.str.contains("collectri").sum())
if rc["require_collectri"]:
    check("CollecTRI loaded", n_col > 0, f"{n_col} pairs from {cfg.paths.collectri.name}")
else:
    print(f"[INFO] CollecTRI not required; pairs with CollecTRI: {n_col}")
print(f"[INFO] regulon pairs {len(reg)}, TFs {reg.tf.nunique()}, targets {reg.target.nunique()}, by source {by_db}")
print(f"[INFO] signed {int((reg.sign != 0).sum())}, unsigned {int((reg.sign == 0).sum())}, "
      f"pairs with both signs across/within sources (set unsigned) {int(reg.conflict.sum())}")

# ---------- 2. extension TFs ----------
print("\n== extension TFs (outside universe, with curated signalling input)")
ext = L["ext_tfs"]
check("extension TFs found", len(ext) > 0, f"{len(ext)} TFs")
nodes_net = set(S.source) | set(S.target)
for tf in ["Elk1", "Stat1", "Stat2"]:
    check(f"{tf} is a network node", tf in nodes_net, "extension" if tf in ext else ("universe" if tf in U else "missing"))
ei = L["ext_inputs"]
for s, t in [("Mapk1", "Elk1"), ("Mapk3", "Elk1")]:
    x = ei[(ei.source == s) & (ei.target == t)]
    check(f"curated {s} -> {t} present and positive", len(x) > 0 and (x.sign > 0).any(),
          x[["db", "sign"]].values.tolist())

# ---------- 3. signalling layer ----------
print("\n== signalling layer")
moved = L["moved_transcription"]
print(f"[INFO] signalling edges {len(S)}; transcription-only edges moved to readout {len(moved)}")
tx = set(rc["transcription_mechanisms"])
left = S[S.mechanism.map(lambda m: bool({t for t in str(m).split(';') if t}) and
                         {t for t in str(m).split(';') if t} <= tx) & (S.tier != "T0")]
check("no transcription-only edges left in signalling layer", left.empty, len(left))
rt = cfg.route_tier if hasattr(cfg, "route_tier") else "T0"
check(f"all route edges kept in signalling layer at the configured tier ({rt})",
      int((S.tier == rt).sum()) >= len(cfg.route_edges),
      f"{int((S.tier == rt).sum())} at {rt}, {len(cfg.route_edges)} configured")
# built from the CONFIGURED route edges, not from a whole tier: at route_tier=T1 the tier
# holds ~16 000 edges and all_simple_paths over it never terminates
_route = {(x[0], x[1]) for x in cfg.route_edges}
G0 = nx.DiGraph([(r.source, r.target, {"sign": int(r.sign)})
                 for r in S.itertuples() if (r.source, r.target) in _route])
signs = []
if G0.has_node("F2rl1") and G0.has_node("Sting1"):
    for p_ in nx.all_simple_paths(G0, "F2rl1", "Sting1"):
        sg = 1
        for u, v in zip(p_, p_[1:]):
            sg *= G0[u][v]["sign"]
        signs.append(sg)
# REPORTED, NOT ASSERTED since 2026-10-19. This used to require every prior-route path to give
# F2rl1 -| Sting1, i.e. Sting1 UP on PAR2 loss. PAR2_29 Fig 3N measures pSTING DOWN, so the
# direction is the open question and a test must not decide it. The enumeration is still printed
# because a change in it is worth noticing.
if signs:
    up = sum(1 for s in signs if s == -1)
    print(f"[INFO] prior-route paths F2rl1 -> Sting1 at {rt}: {len(signs)}, "
          f"{up} predict Sting1 UP ({up / len(signs):.0%}) -- reported, not asserted")
else:
    print(f"[INFO] no prior-route path F2rl1 -> Sting1 within {rt} "
          f"(expected when route_tier is T1, or when the terminal edges are demoted)")

# ---------- 4. readout layer ----------
print("\n== readout layer")
tfo = L["tf_outputs"]
check("TF-activity outputs defined", len(tfo) > 0,
      f"{len(tfo)} TFs with >= {rc['min_tf_targets']} signed targets measured in >= {rc['min_gene_contexts']} contexts")
missing_tf = [t for t in rc["key_tfs"] if t not in tfo]
check("key ERK / IFN / NF-kB / p53 TFs are outputs", not missing_tf,
      f"missing {missing_tf}" if missing_tf else ", ".join(f"{t}:{int(L['tf_n_targets'][t])}" for t in rc["key_tfs"]))
G = nx.DiGraph([(r.source, r.target) for r in S[S.tier.isin(["T0", "T1"])].itertuples()])
unreach = [t for t in tfo if t not in G or not (nx.has_path(G, "F2rl1", t) if "F2rl1" in G else False)]
print(f"[INFO] TF outputs reachable from F2rl1 on T0+T1: {len(tfo) - len(unreach)}/{len(tfo)}")
for t in ["Elk1", "Stat1", "Irf3"]:
    if t in G:
        check(f"{t} reachable from F2rl1 on T0+T1", nx.has_path(G, "F2rl1", t),
              " > ".join(nx.shortest_path(G, "F2rl1", t)) if nx.has_path(G, "F2rl1", t) else "no path")

rg = L["readout_genes"]
re_ = L["readout_edges"]
ncx = L["n_contexts"]
print(f"[INFO] readout genes {len(rg)} (measured in >= {rc['min_gene_contexts']} contexts and regulated by a network TF); "
      f"readout edges {len(re_)}, signed {int((re_.sign != 0).sum())}")
n_u_meas = int(sum(1 for g in U if ncx.get(g, 0) >= rc["min_gene_contexts"]))
print(f"[INFO] universe genes measured but without a network TF regulator: "
      f"{n_u_meas - len(set(rg) & U)} of {n_u_meas} (excluded from gene loss)")
for g in rc["key_genes"]:
    regs = re_[re_.gene == g]
    n_put = int((regs.dbs == "putative_isg").sum())
    check(f"readout gene {g}", len(regs) > 0,
          f"{len(regs)} TFs ({n_put} putative), {int(ncx.get(g, 0))} contexts; "
          + ", ".join(f"{a}{'+' if s > 0 else '-' if s < 0 else '?'}" for a, s in zip(regs.tf, regs.sign))[:120])
put = re_[re_.dbs == "putative_isg"]
check("putative regulator edges are unsigned and only for configured genes",
      (put.sign == 0).all() and set(put.gene) <= set(rc.get("putative_regulators", {})),
      put.groupby("gene").tf.apply(list).to_dict())

# ---------- outputs ----------
S.to_csv(out / "step05_signalling_edges.csv.gz", index=False)
nodes = L["network"]["nodes"].copy()
nodes["extension_tf"] = nodes.node.isin(ext)
nodes["tf_output"] = nodes.node.isin(tfo)
nodes.to_csv(out / "step05_nodes.csv", index=False)
reg.to_csv(out / "step05_regulons.csv.gz", index=False)
pd.DataFrame({"tf": tfo, "n_signed_measured_targets": [int(L["tf_n_targets"][t]) for t in tfo],
              "extension_tf": [t in ext for t in tfo]}).to_csv(out / "step05_tf_outputs.csv", index=False)
re_.to_csv(out / "step05_readout_edges.csv.gz", index=False)
ei.to_csv(out / "step05_extension_tfs.csv", index=False)
L["lfc"].loc[sorted(set(rg) | U)].to_csv(out / "step05_log2fc.csv.gz")
L["padj"].loc[sorted(set(rg) | U)].to_csv(out / "step05_padj.csv.gz")

n_fail = sum(not ok for _, ok, _ in results)
(out / "step05_report.json").write_text(json.dumps(dict(
    regulon_pairs=len(reg), extension_tfs=len(ext), tf_outputs=len(tfo), readout_genes=len(rg),
    signalling_edges=len(S), checks=[dict(name=n, ok=ok, detail=d) for n, ok, d in results]), indent=2))
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 5:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
