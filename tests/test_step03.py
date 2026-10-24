"""Step 3 validation: signal-entity nodes and edges.

Run from the LEMBAS folder:
    python tests/test_step03.py
Writes outputs/step03_entity_nodes.csv and outputs/step03_entity_edges.csv.
Needs outputs/step02_edges_by_db.csv from step 2 for the route checks.
"""
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2 import entities as SE
from lembas_par2.genes import load_universe

pd.set_option("display.width", 250)
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail != "" else ""))
    return ok


cfg = load_config()
out = cfg.paths.output_dir
U = set(load_universe(cfg))

# ---------- 1. nodes ----------
print("== entity nodes")
nodes = SE.load_entity_nodes(cfg)
ids = set(nodes.node)
check("24 entity nodes", len(nodes) == 24, len(nodes))
check("entity IDs unique and prefixed SE_", nodes.node.is_unique and all(i.startswith("SE_") for i in ids))
check("no entity ID collides with a gene", not (ids & U), sorted(ids & U))
inputs = set(nodes[nodes.input_only].node)
print(f"[INFO] input-only entities: {sorted(inputs)}")

# ---------- 2. edges ----------
print("\n== entity edges")
e = SE.load_entity_edges(cfg, sorted(U))
check("edges loaded, duplicates merged", len(e) > 0 and not e.duplicated(["source", "target"]).any(),
      f"{len(e)} edges from {int(e.n_records.sum())} records")
ends = set(e.source) | set(e.target)
check("every endpoint is a universe gene or an entity", ends <= (U | ids), sorted(ends - U - ids))
check("every edge touches an entity", (e.source.isin(ids) | e.target.isin(ids)).all())
check("signs in {-1, +1}", set(e.sign) <= {-1, 1}, sorted(set(e.sign)))
check("no edges into input-only entities", not e.target.isin(inputs).any(),
      e[e.target.isin(inputs)][["source", "target"]].values.tolist())
produced = ids - inputs
no_prod = sorted(n for n in produced if not ((e.target == n) & (e.sign > 0)).any())
check("every produced entity has a positive producer", not no_prod, no_prod)
no_recv = sorted(n for n in ids if not (e.source == n).any())
check("every entity has a downstream receiver", not no_recv, no_recv)

lookup = {(r.source, r.target): r.sign for r in e.itertuples()}
expected = [
    ("Bax", "SE_CYTO_MTDNA", 1), ("Bak1", "SE_CYTO_MTDNA", 1), ("Vdac1", "SE_CYTO_MTDNA", 1),
    ("Bcl2", "SE_CYTO_MTDNA", -1), ("Bcl2l1", "SE_CYTO_MTDNA", -1), ("Mcl1", "SE_CYTO_MTDNA", -1),
    ("SE_CYTO_MTDNA", "Cgas", 1), ("Cgas", "SE_CGAMP", 1), ("SE_CGAMP", "Sting1", 1),
    ("Pik3ca", "SE_PIP3", 1), ("Pten", "SE_PIP3", -1), ("SE_PIP3", "Akt1", 1),
    ("SE_PIP2", "SE_IP3", 1), ("SE_IP3", "SE_CA2", 1), ("SE_CA2", "Calm1", 1),
    ("SE_CAMP", "Prkar1a", -1), ("SE_ROS", "Keap1", -1), ("SE_ROS", "Map3k5", 1),
]
for s, t, sg in expected:
    check(f"{s} -> {t} = {sg:+d}", lookup.get((s, t)) == sg, f"got {lookup.get((s, t))}")
print(f"[INFO] sign overrides applied: {e[e.overridden][['source', 'target', 'sign']].values.tolist()}")

# ---------- 3. route with curated edges ----------
print("\n== route through entities (with step-2 curated edges)")
cur_path = out / "step02_edges_by_db.csv"
if check("step-2 edge table present", cur_path.exists(), str(cur_path.name)):
    cur = pd.read_csv(cur_path)
    allE = pd.concat([cur[["source", "target", "sign", "db"]], e[["source", "target", "sign", "db"]]])
    allE = allE[allE.sign != 0]
    adj = defaultdict(list)
    for r in allE.itertuples():
        adj[r.source].append((r.target, r.sign, r.db))

    def signed_path(src, dst, via=None, max_len=6):
        """Shortest path src->dst (optionally through `via`), returning (nodes, sign product)."""
        q = deque([(src, [src], 1)])
        seen = {src}
        while q:
            n, path, sg = q.popleft()
            if len(path) > max_len:
                continue
            for m, s, _ in adj[n]:
                if m == dst and (via is None or via in path + [m]):
                    return path + [m], sg * s
                if m not in seen:
                    seen.add(m)
                    q.append((m, path + [m], sg * s))
        return None, None

    for src, via in [("Bax", "SE_CYTO_MTDNA"), ("Bak1", "SE_CYTO_MTDNA"), ("Mcl1", "SE_CYTO_MTDNA")]:
        p, sg = signed_path(src, "Sting1", via)
        exp = -1 if src == "Mcl1" else 1
        check(f"{src} reaches Sting1 via mtDNA with sign {exp:+d}", p is not None and sg == exp,
              f"{' -> '.join(p) if p else 'no path'} (sign {sg})")
    direct = cur[(cur.source == "Cgas") & (cur.target == "Sting1")]
    print(f"[INFO] Cgas -> Sting1 exists both directly ({', '.join(direct.db)}) and via SE_CGAMP; "
          "both kept, step 4 decides weights")
    pka = cur[(cur.source.str.startswith("Prkar")) & (cur.target.str.startswith("Prkac"))]
    print(f"[INFO] curated Prkar* -> Prkac* edges (needed for cAMP -> PKA): "
          f"{pka[['source', 'target', 'db', 'sign']].values.tolist()}")

# ---------- outputs ----------
nodes.to_csv(out / "step03_entity_nodes.csv", index=False)
e.to_csv(out / "step03_entity_edges.csv", index=False)
n_fail = sum(not ok for _, ok, _ in results)
(out / "step03_report.json").write_text(json.dumps(dict(
    n_nodes=len(nodes), n_edges=len(e),
    checks=[dict(name=n, ok=ok, detail=d) for n, ok, d in results]), indent=2))
print(f"\n== SUMMARY: {len(results) - n_fail}/{len(results)} passed, {n_fail} failed")
print("STEP 3:", "PASS" if n_fail == 0 else "FAIL")
sys.exit(1 if n_fail else 0)
