"""Step 17: does curated evidence say PAR2 loss raises Ifnb1, and by which route?

Enumerates simple signed paths F2rl1 -> ... -> target through the curated signalling layer and
multiplies their edge signs. A PAR2 knockout removes F2rl1, so a path whose net sign is NEGATIVE
predicts the target UP. Nothing is weighted: this is a statement about curated topology, not about
the fitted model, so no trained parameter enters.

Verified against the record: all simple T0 paths F2rl1 -> Sting1 number 2 640 (saturating at depth
16), the figure quoted in REPORT.md.

Two endpoints, because Ifnb1 exists twice in the model and the two disagree in the fitted model
(signal-layer Ifnb1 rises under PAR2-KO while predicted Ifnb1 mRNA falls):
  signal   paths to the Ifnb1 signalling node (reached by the T0 edge Irf3 -> Ifnb1)
  mrna     paths to any TF that regulates Ifnb1 in the readout layer, times that TF's readout sign
Two regulator sets for the mRNA endpoint:
  curated  only the Ifnb1 regulators with a curated consensus sign
  all      every Ifnb1 regulator; the ones with no curated sign (Stat1, Nfkb1, Yy1) take the sign
           the frozen model LEARNED for them, which is labelled as learned, not curated
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

SOURCE = "F2rl1"
MAX_EXPLORE = 60_000_000


def signed_adjacency(cfg, tiers):
    """({source: [(target, sign, edge_id)]}, n_edges) over the given tiers; unsigned (T2) edges are
    dropped, since a path's net sign is undefined through them. `edge_id` lets the permutation null
    re-sign an already-enumerated path set instead of re-enumerating the graph."""
    e = pd.read_csv(cfg.paths.output_dir / "step05_signalling_edges.csv.gz",
                    usecols=["source", "target", "sign", "tier"])
    e = e[e.tier.isin(tiers) & (e.sign != 0)].reset_index(drop=True)
    adj = defaultdict(list)
    for i, (s, t, sg) in enumerate(zip(e.source, e.target, np.sign(e.sign.values).astype(int))):
        adj[s].append((t, int(sg), i))
    return adj, len(e)


def enumerate_paths(adj, targets, max_depth, source=SOURCE, max_explore=MAX_EXPLORE, keep_paths=True,
                    return_edges=False):
    """Simple paths source -> each target, with the product of their edge signs.

    Returns (DataFrame[target, depth, net_sign, path], n_explored, capped), and with return_edges a
    4th element: the list of edge-id lists, one per path, for the permutation null. Iterative, so
    large depths do not hit Python's recursion limit."""
    targets = set(targets)
    rows, edges, explored, capped = [], [], 0, False
    stack = [(source, 1, [source], [], {source}, iter(adj.get(source, [])))]
    while stack:
        node, sign, path, eids, seen, it = stack[-1]
        nxt = next(it, None)
        if nxt is None:
            stack.pop()
            continue
        t, sg, eid = nxt
        if t in seen:
            continue
        explored += 1
        if explored > max_explore:
            capped = True
            break
        ns, np_, ne = sign * sg, path + [t], eids + [eid]
        if t in targets:
            rows.append(dict(target=t, depth=len(np_) - 1, net_sign=ns,
                             path="->".join(np_) if keep_paths else ""))
            if return_edges:
                edges.append(ne)
        if len(np_) - 1 < max_depth:
            stack.append((t, ns, np_, ne, seen | {t}, iter(adj.get(t, []))))
    df = pd.DataFrame(rows, columns=["target", "depth", "net_sign", "path"])
    return (df, explored, capped, edges) if return_edges else (df, explored, capped)


def summarise(df, label=""):
    """Counts of net-negative (= target UP on PAR2 loss) vs net-positive paths."""
    n = len(df)
    up = int((df.net_sign < 0).sum())
    return dict(label=label, n_paths=n, n_predict_up=up, n_predict_down=n - up,
                frac_up=up / n if n else np.nan,
                depth_min=int(df.depth.min()) if n else None, depth_max=int(df.depth.max()) if n else None)


def path_edge_matrix(edge_lists, n_edges):
    """Sparse 0/1 matrix [path, edge]: 1 where the path uses that edge."""
    rows = np.fromiter((i for i, es in enumerate(edge_lists) for _ in es), dtype=np.int64)
    cols = np.fromiter((e for es in edge_lists for e in es), dtype=np.int64)
    return sp.csr_matrix((np.ones(len(cols), dtype=np.int8), (rows, cols)),
                         shape=(len(edge_lists), n_edges))


def sign_null(M, n_perm=1000, seed=0, extra_parity=None):
    """Null: reassign every edge sign at random, keeping the topology AND the enumerated path set,
    and recount the fraction of paths predicting UP. A path's net sign is negative iff it carries an
    odd number of negative edges, so one sparse product per permutation replaces a re-enumeration.
    Answers 'do the curated signs favour UP, or does the shape of the graph alone?'"""
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    for k in range(n_perm):
        neg = (rng.random(M.shape[1]) < 0.5).astype(np.int8)
        par = (M @ neg) % 2
        if extra_parity is not None:           # a fixed readout-sign flip per path
            par = (par + extra_parity) % 2
        out[k] = float((par == 1).mean())
    return out


def ifnb1_regulators(cfg, model_signs=None):
    """Ifnb1's readout regulators with the sign to use. `curated` = consensus sign only; `all` adds
    the unsigned ones with the sign the frozen model learned (labelled as learned)."""
    ro = pd.read_csv(cfg.paths.output_dir / "step05_readout_edges.csv.gz")
    r = ro[ro.gene == "Ifnb1"].copy()
    r["curated_sign"] = np.sign(r.sign.values).astype(int)
    r["learned_sign"] = [int(np.sign(model_signs.get(tf, 0))) if model_signs else 0 for tf in r.tf]
    r["sign_used_all"] = np.where(r.curated_sign != 0, r.curated_sign, r.learned_sign)
    r["source_of_sign"] = np.where(r.curated_sign != 0, "curated", "learned")
    return r[["tf", "curated_sign", "learned_sign", "sign_used_all", "source_of_sign", "dbs", "conflict"]]


def learned_readout_signs(cfg, gene="Ifnb1", variant="J0", seeds=(0, 1, 2, 3, 4)):
    """Median learned readout weight per regulator of `gene`, from the frozen checkpoints."""
    try:
        import torch

        from .eval_harness import load_model
    except Exception as exc:                                     # torch unavailable (e.g. NumPy clash)
        print(f"[paths] learned signs unavailable ({exc}); the 'all' variant will use curated signs only",
              file=sys.stderr)
        return {}
    o = cfg.paths.output_dir
    acc = defaultdict(list)
    for s in seeds:
        p = o / "source_variants" / (variant if s == 0 else f"{variant}_s{s}") / "checkpoint.pt"
        if not p.exists():
            continue
        try:
            m, _ = load_model(cfg, p, "cpu")
        except Exception as exc:
            # a curation change moves edges between tiers or layers and the saved parameter
            # shapes stop matching. This enumeration is topology-only and read-only, so a stale
            # checkpoint must degrade to curated signs, not abort the whole step.
            print(f"[paths] skipping stale checkpoint {p.parent.name}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        with torch.no_grad():
            W = m.readout_weights().detach().cpu().numpy()
        rr = m.readout.reset_index(drop=True)
        for k in np.flatnonzero((rr.gene == gene).values):
            acc[rr.tf.iloc[k]].append(float(W[k]))
    if not acc:
        print("[paths] no loadable checkpoint: the 'all' variant falls back to curated signs",
              file=sys.stderr)
    return {tf: float(np.median(v)) for tf, v in acc.items()}


def mrna_paths(paths_to_tfs, regs, sign_col):
    """Extend paths that end at an Ifnb1 regulator by that regulator's readout sign."""
    s = dict(zip(regs.tf, regs[sign_col]))
    keep = paths_to_tfs.target.isin([t for t, v in s.items() if v != 0])
    df = paths_to_tfs[keep].copy()
    df.attrs["keep_mask"] = keep.values
    df["readout_sign"] = df.target.map(s).astype(int)
    df["net_sign"] = df.net_sign * df.readout_sign
    df["depth"] = df.depth + 1
    return df
