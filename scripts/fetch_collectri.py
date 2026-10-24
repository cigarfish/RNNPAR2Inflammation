"""Download CollecTRI mouse TF-target regulons via decoupler (OmniPath) and save them.

Run once from the LEMBAS folder (needs internet and decoupler):
    pip install decoupler        # if missing
    python scripts/fetch_collectri.py
Writes data/collectri_mouse.csv with columns source, target, weight[, PMID].
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "collectri_mouse.csv"

import decoupler as dc

print("decoupler", dc.__version__)
if hasattr(dc, "op") and hasattr(dc.op, "collectri"):          # decoupler >= 2.0
    net = dc.op.collectri(organism="mouse")
elif hasattr(dc, "get_collectri"):                               # decoupler 1.x
    net = dc.get_collectri(organism="mouse", split_complexes=False)
else:
    sys.exit("No CollecTRI accessor found in this decoupler version")

keep = [c for c in ["source", "target", "weight", "PMID", "resources", "sign_decision"] if c in net.columns]
net = net[keep]
net.to_csv(OUT, index=False)
print(f"saved {len(net)} edges, {net.source.nunique()} TFs, {net.target.nunique()} targets -> {OUT}")
print(net.head())
