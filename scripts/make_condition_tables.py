"""Emit the full condition listing as markdown, for the manuscript supplement.

One table per source family -- drug (57), GEO (48), perturbation paper (36), holdout (3) --
from outputs/step06_conditions.csv, which is the table the model is actually built from, so the
listing cannot drift from the fitted conditions.

`weight` is the condition's weight in the loss; holdouts are 0 by construction and their
direction constraints are excluded as well. `fold` is the cross-validation fold (drug only).
`n_sig` is the number of significant genes behind the condition where one was computed.

    python scripts/make_condition_tables.py            # -> outputs/condition_tables.md
    python scripts/make_condition_tables.py --stdout
"""
import argparse
import ast
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lembas_par2 import load_config


def nodes(cell):
    """The JSON-ish list columns -> 'A, B' or an em dash."""
    if not isinstance(cell, str) or not cell.strip():
        return "—"
    try:
        v = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return cell
    return ", ".join(f"`{x}`" for x in v) if v else "—"


def note_field(notes, key):
    if not isinstance(notes, str):
        return "—"
    m = re.search(rf"{key}=([^;]+)", notes)
    if m:
        return m.group(1).strip()
    return "—"


def background(notes):
    if not isinstance(notes, str):
        return "—"
    m = re.search(r"background='([^']*)'", notes)
    return (m.group(1) or "none") if m else "—"


def md(df, cols, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in df.itertuples():
        out.append("| " + " | ".join(str(getattr(r, c)) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    d = pd.read_csv(cfg.paths.output_dir / "step06_conditions.csv")
    d["inhibited"] = d.trt_inhibit.map(nodes)
    d["driven"] = d.trt_inputs.map(nodes)
    d["species"] = d.notes.map(lambda s: note_field(s, "species"))
    d["background"] = d.notes.map(background)
    d["study"] = d.dataset
    d["fold_s"] = d.fold.map(lambda v: "—" if pd.isna(v) else str(int(v)))
    d["nsig"] = d.n_sig.map(lambda v: "—" if pd.isna(v) else str(int(v)))
    d["w"] = d.weight.map(lambda v: f"{v:.3f}")

    L = []
    L.append("## Supplementary Table S1 — all model conditions\n")
    L.append(f"From `outputs/step06_conditions.csv` ({len(d)} conditions). `weight` is the "
             "condition's weight in the loss. **Holdout conditions carry weight 0 and their "
             "direction constraints are excluded from the loss**, so nothing in them reaches "
             "the fitted parameters.\n")

    g = d[d.source == "drug"].sort_values(["cls", "name"])
    L.append(f"\n### S1a — Drug conditions (n = {len(g)})\n")
    L.append("All from GSE219229, LPS-stimulated (`Tlr4`, `Cd14` driven in both arms); the drug "
             "arm additionally inhibits the listed nodes. `fold` is the cross-validation fold.\n")
    L.append(md(g, ["name", "cls", "inhibited", "fold_s", "nsig", "w"],
                ["drug", "target class", "inhibited node(s)", "fold", "n sig genes", "weight"]))

    g = d[d.source == "geo"].sort_values(["cls", "name", "dataset"])
    L.append(f"\n\n### S1b — GEO perturbation conditions (n = {len(g)})\n")
    L.append("Public contrasts. `driven` are nodes set as inputs in the treatment arm; "
             "`inhibited` are nodes suppressed.\n")
    L.append(md(g, ["study", "name", "cls", "driven", "inhibited", "w"],
                ["dataset", "perturbation", "class", "driven", "inhibited", "weight"]))

    g = d[d.source == "perturb"].sort_values(["study", "condition_id"])
    L.append(f"\n\n### S1c — Perturbation-paper conditions (n = {len(g)})\n")
    L.append("Directional constraints curated from the PAR2 literature. These enter the loss "
             "through the direction term only.\n")
    L.append(md(g, ["condition_id", "name", "species", "background", "inhibited", "w"],
                ["condition", "perturbation", "species", "background", "inhibited", "weight"]))

    g = d[d.source == "holdout"].sort_values("condition_id")
    L.append(f"\n\n### S1d — Sealed holdout conditions (n = {len(g)})\n")
    L.append("**Weight 0 throughout training; direction constraints excluded.** Scored only "
             "after the model was fixed. Observations are listed in Table S2.\n")
    L.append(md(g, ["condition_id", "study", "name", "species", "background", "inhibited"],
                ["condition", "study", "perturbation", "species", "background", "inhibited"]))

    txt = "\n".join(L) + "\n"
    if a.stdout:
        print(txt)
    else:
        p = cfg.paths.output_dir / "condition_tables.md"
        p.write_text(txt, encoding="utf-8")
        print(f"wrote {p}")
        for s, n in d.source.value_counts().items():
            print(f"   {s:8s} {n}")


if __name__ == "__main__":
    main()
