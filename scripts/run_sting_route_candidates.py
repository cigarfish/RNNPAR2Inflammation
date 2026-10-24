"""Step 24: which curated routes drive Sting1 DOWN on PAR2 loss, and what each costs to promote.

PAR2_29 measures pSTING DOWN; the model predicts UP. Over all curated signed edges (T0+T1) the
enumeration already favours DOWN (44.3 % of 8801 paths predict UP), so the routes that give the
observed direction exist in curated knowledge - they are just not in the prior.

This lists them, grouped by the distinct mechanism they run through, and reports for each:
  * how many enumerated paths it carries and what fraction predict DOWN,
  * the **promotion cost**: how many of its edges are not already T0,
  * the hop through which it reaches Sting1 (the final edge), which is the thing that actually
    distinguishes one mechanism from another.

Nothing is promoted here. The output is a shortlist to argue about, and the argument has to be
about the biology of each edge, not about which combination makes the number come out.

**Path counts are NOT influence.** A linear chain contributes a handful of paths while a branched
arm contributes thousands (step 22b: the Ca2+ arm is 2 paths against 2640). Read the
promotion-cost column and the mechanism, not the path count.

    python scripts/run_sting_route_candidates.py
    python scripts/run_sting_route_candidates.py --top 25 --max-cost 3
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config
from lembas_par2.paths import enumerate_paths, signed_adjacency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Sting1")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--max-cost", type=int, default=None, help="only routes needing <= this many promotions")
    a = ap.parse_args()

    cfg = load_config()
    out = cfg.paths.output_dir / "step24_sting_candidates"
    out.mkdir(parents=True, exist_ok=True)

    e = pd.read_csv(cfg.paths.output_dir / "step05_signalling_edges.csv.gz",
                    usecols=["source", "target", "sign", "tier"], low_memory=False)
    tier = {(s, t): tr for s, t, tr in zip(e.source, e.target, e.tier)}

    adj, n_edges = signed_adjacency(cfg, ["T0", "T1"])
    df, explored, capped = enumerate_paths(adj, {a.target}, a.depth)
    if capped:
        print("[warn] the enumeration hit its explore cap; counts are a lower bound")
    # net_sign +1 transmits a decrease as a decrease, i.e. predicts the target DOWN on PAR2 loss
    df["predicts"] = np.where(df.net_sign > 0, "down", "up")
    nodes = df.path.str.split("->")
    df["last_hop"] = nodes.map(lambda p: f"{p[-2]} -> {p[-1]}")
    df["n_non_t0"] = nodes.map(
        lambda p: sum(tier.get((p[i], p[i + 1]), "T3") != "T0" for i in range(len(p) - 1)))
    df["depth"] = nodes.map(len) - 1
    df.to_csv(out / "paths.csv.gz", index=False)

    tot = len(df)
    down = df[df.predicts == "down"]
    print(f"\n== F2rl1 -> {a.target}, curated T0+T1, depth <= {a.depth}")
    print(f"   {tot} paths: {len(down)} predict DOWN ({len(down) / tot:.1%}), "
          f"{tot - len(down)} predict UP")

    # group by the final hop: that is what distinguishes one mechanism from another
    g = (df.groupby("last_hop")
           .agg(paths=("predicts", "size"),
                frac_down=("predicts", lambda s: (s == "down").mean()),
                min_cost=("n_non_t0", "min"),
                min_depth=("depth", "min"))
           .reset_index())
    g = g[g.frac_down > 0.5].sort_values(["min_cost", "frac_down"], ascending=[True, False])
    if a.max_cost is not None:
        g = g[g.min_cost <= a.max_cost]
    print(f"\n== final hops into {a.target} whose paths mostly predict DOWN "
          f"(sorted by promotion cost)\n")
    print(g.head(a.top).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    g.to_csv(out / "candidate_hops.csv", index=False)

    print(f"\n== the cheapest concrete routes (fewest edges to promote)\n")
    best = down.sort_values(["n_non_t0", "depth"]).head(a.top)
    for r in best.itertuples():
        p = r.path.split("->")
        missing = [f"{p[i]}->{p[i + 1]}" for i in range(len(p) - 1)
                   if tier.get((p[i], p[i + 1]), "T3") != "T0"]
        print(f"  cost {r.n_non_t0}  depth {r.depth}  {r.path}")
        print(f"          promote: {', '.join(missing) if missing else '(already all T0)'}")

    print(f"\n== for contrast, what currently drives {a.target} UP from inside T0")
    up_t0 = df[(df.predicts == "up") & (df.n_non_t0 == 0)]
    print(f"   {len(up_t0)} UP paths are already entirely T0" if len(up_t0)
          else "   none: every UP path still needs at least one non-T0 edge")
    if len(up_t0):
        print("   their final hops: " + ", ".join(f"{k} x{v}" for k, v in
                                                  Counter(up_t0.last_hop).most_common(5)))
    print(f"\nwrote {out}")
    print("\nREAD THIS BEFORE PROMOTING ANYTHING: path count is combinatorial fan-out, not\n"
          "influence, and PAR2_29's pSTING has already been read, so nothing promoted now can\n"
          "be validated against it. Argue each edge on its own biology.")


if __name__ == "__main__":
    main()
