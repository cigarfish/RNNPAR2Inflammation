"""Step 4 validation: merged network, evidence tiers, convergence-node restriction.

Run from the LEMBAS folder:
    python tests/test_step04.py

Writes to outputs/:
    step04_edges.csv.gz             full tiered edge list
    step04_nodes.csv                genes + signal entities
    step04_route_check.csv          each T0 route edge vs curated data
    step04_convergence_inputs.csv   T0/T1 inputs of each convergence node, with sources
    step04_removed_convergence.csv  T2/T3 inputs removed from convergence nodes
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
from lembas_par2.network import build_network

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_rows", 300)
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
out = cfg.paths.output_dir
U = set(load_universe(cfg))
N = build_network(cfg, sorted(U))
E, nodes, rc = N["edges"], N["nodes"], N["route_check"]

# ---------- 1. merged network ----------
print("== merged network")
check("one row per directed pair", not E.duplicated(["source", "target"]).any())
ents = set(nodes[nodes.kind == "signal_entity"].node)
ends = set(E.source) | set(E.target)
check("every endpoint is a universe gene or entity", ends <= (U | ents), sorted(ends - U - ents)[:10])
check("no self-loops", (E.source != E.target).all())
tiers = E.groupby("tier", observed=False).agg(
    edges=("sign", "size"), pos=("sign", lambda s: int((s > 0).sum())),
    neg=("sign", lambda s: int((s < 0).sum())), unsigned=("sign", lambda s: int((s == 0).sum())))
print("[INFO] edges per tier:\n" + tiers.to_string())
print(f"[INFO] sign rules among curated pairs: {E[E.tier != 'T3'].sign_rule.value_counts().to_dict()}")
print(f"[INFO] STRING-only pairs by score: {N['string_stats']['only_pairs']} "
      f"(threshold {N['string_stats']['threshold']}, each pair added in both directions)")
check("T1 edges all signed, T2/T3 all unsigned",
      (E[E.tier == "T1"].sign != 0).all() and (E[E.tier.isin(["T2", "T3"])].sign == 0).all())
n_in = int(nodes.in_network.sum())
print(f"[INFO] nodes in network: {n_in}/{len(nodes)} "
      f"(isolated: {sorted(nodes[~nodes.in_network].node)[:20]})")

# ---------- 2. T0 route ----------
print("\n== T0 route edges")
ok_status = rc.status.isin(["data_agrees", "sign_from_prior", "added_from_prior"])
check(f"all {len(rc)} route edges confirmed by data (or prior-signed)", ok_status.all(),
      rc[~ok_status][["source", "target", "status", "db_signs"]].values.tolist())
prior = rc[rc.status.isin(["sign_from_prior", "added_from_prior"])]
check("prior-signed route edges are only the flagged ones",
      set(zip(prior.source, prior.target)) == set(zip(rc[rc.prior].source, rc[rc.prior].target)),
      prior[["source", "target"]].values.tolist())
print("[INFO] prior-signed route edges:\n" +
      prior[["source", "target", "route_sign", "status", "db_signs", "prior_source"]].to_string(index=False))
flagged_but_data = rc[rc.prior & (rc.status == "data_agrees")]
if len(flagged_but_data):
    print(f"[INFO] flagged 'prior' but data already agrees: {flagged_but_data[['source', 'target']].values.tolist()}")
check("T0 count == route edges", int((E.tier == "T0").sum()) == len(rc), int((E.tier == "T0").sum()))

# every T0 path from F2rl1 to Sting1 must give F2rl1 -> Sting1 sign -1 (KO raises Sting1)
G0 = nx.DiGraph()
for r in E[E.tier == "T0"].itertuples():
    G0.add_edge(r.source, r.target, sign=int(r.sign))
paths = list(nx.all_simple_paths(G0, "F2rl1", "Sting1"))
signs = []
for p in paths:
    s = 1
    for u, v in zip(p, p[1:]):
        s *= G0[u][v]["sign"]
    signs.append(s)
check("T0 paths F2rl1 -> Sting1 exist", len(paths) > 0, f"{len(paths)} simple paths")
check("every T0 path gives F2rl1 -| Sting1 (net sign -1)", all(s == -1 for s in signs),
      f"+1 paths: {[' > '.join(p) for p, s in zip(paths, signs) if s == 1][:5]}")
G01 = nx.DiGraph()
for r in E[E.tier.isin(["T0", "T1"])].itertuples():
    G01.add_edge(r.source, r.target, sign=int(r.sign))
check("Sting1 reachable from F2rl1 on T0+T1", nx.has_path(G01, "F2rl1", "Sting1"),
      f"shortest: {' > '.join(nx.shortest_path(G01, 'F2rl1', 'Sting1'))}")

# competing short curated paths (the dilution the route priors must outweigh)
census, examples = {}, {1: [], -1: []}
for p in nx.all_simple_paths(G01, "F2rl1", "Sting1", cutoff=4):
    s = 1
    for u, v in zip(p, p[1:]):
        s *= G01[u][v]["sign"]
    census[(len(p) - 1, s)] = census.get((len(p) - 1, s), 0) + 1
    if len(examples[s]) < 6:
        examples[s].append(" > ".join(p))
print(f"[INFO] T0+T1 paths F2rl1 -> Sting1 up to 4 steps, (length, net sign): {dict(sorted(census.items()))}")
print(f"[INFO]   net +1 (KO would lower Sting1): {examples[1]}")
print(f"[INFO]   net -1 (KO would raise Sting1): {examples[-1]}")

# ---------- 2b. curation effects ----------
print("\n== curation effects")
pairs = set(zip(E.source, E.target))
check("Gnai1/Gnas -> Src gone from every tier (incl. STRING)",
      not ({("Gnai1", "Src"), ("Gnas", "Src"), ("Src", "Gnai1"), ("Src", "Gnas")} & pairs))
sc = E[(E.source == "Src") & (E.target == "Cgas")]
check("Src -| Cgas present as T1 (manual, -1)", len(sc) == 1 and sc.iloc[0].sign == -1 and sc.iloc[0].tier == "T1",
      sc[["tier", "sign", "db_signs"]].values.tolist())
ctx = E[E.context_dependent]
check("context-dependent edges tagged", len(ctx) == len(cfg.context_dependent_edges),
      ctx[["source", "target", "tier", "sign", "db_signs"]].values.tolist())
G_beta = nx.DiGraph([(r.source, r.target) for r in E[E.tier.isin(["T0", "T1"])].itertuples()])
check("Src still reachable from F2rl1 via Gbg -> Pik3cg or Arrb1",
      nx.has_path(G_beta, "F2rl1", "Src"), " > ".join(nx.shortest_path(G_beta, "F2rl1", "Src")))

# ---------- 3. convergence nodes ----------
print("\n== convergence nodes (only T0/T1 inputs)")
conv = cfg.convergence_nodes
cin = E[E.target.isin(conv)]
check("no T2/T3 inputs into convergence nodes", cin.tier.isin(["T0", "T1"]).all(),
      cin[~cin.tier.isin(["T0", "T1"])][["source", "target", "tier"]].values.tolist())
rem = N["removed_convergence"]
print(f"[INFO] T2/T3 inputs removed: {rem.groupby(['target', 'tier'], observed=True).size().to_dict()}")
report = (cin.assign(tier=cin.tier.astype(str))
             .sort_values(["target", "tier", "sign", "source"])
             [["target", "source", "tier", "sign", "sign_rule", "db_signs", "mechanism", "string_score", "detail"]])
for node in conv:
    sub = report[report.target == node]
    print(f"\n-- {node}: {len(sub)} inputs "
          f"(T0 {int((sub.tier == 'T0').sum())}, T1 {int((sub.tier == 'T1').sum())})")
    print(sub[["source", "tier", "sign", "sign_rule", "db_signs", "mechanism"]].to_string(index=False))

# ---------- outputs ----------
E.to_csv(out / "step04_edges.csv.gz", index=False)
nodes.to_csv(out / "step04_nodes.csv", index=False)
rc.to_csv(out / "step04_route_check.csv", index=False)
report.to_csv(out / "step04_convergence_inputs.csv", index=False)
rem.to_csv(out / "step04_removed_convergence.csv", index=False)

n_fail = sum(not ok for _, ok, _ in results)
(out / "step04_report.json").write_text(json.dumps(dict(
    tiers=tiers.to_dict(), string=N["string_stats"], n_route_paths=len(paths),
    short_path_census={f"len{k[0]}_sign{k[1]:+d}": v for k, v in census.items()},
    checks=[dict(name=n, ok=ok, detail=d) for n, ok, d in results]), indent=2, default=str))
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 4:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
