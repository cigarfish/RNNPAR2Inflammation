"""Step 12o: is transmission worth fixing? Two model-free checks on drug-specific skill.

Both are scored exactly like the harness selection metric: residualised r (prediction and
target each minus their training-fold drug mean) on the 9 validation drugs with step-12f0
noise ceiling r_max >= 0.3, plus out-of-fold over all 5 drug folds and a drug-label
permutation null.

O  Network-propagation oracle - transmission with no stability cap. A drug inhibits its
   targets (delta = -1 there) and the signal is propagated through a signed graph as a
   truncated series   x = sum_{k=0..K} alpha^k * H_k,   H_k = A^k delta  normalised per hop
   to unit RMS over the training drugs, so alpha sets the relative weight of hop k and no
   hop is attenuated by the rho < 1 cap. Prediction = x at the TF output nodes.
   Graphs: curated signed T0+T1 (raw signs, and divided by in-degree), and the trained J3
   network with all tiers (its learned signs and relative weights, but no cap).
   (graph, K, alpha) is chosen on the training drugs only.
T  Target-identity baseline: a validation drug is predicted by the mean residual TF
   profile of the training drugs that share at least one target with it (or, second
   variant, the same target class). Measures whether the target annotation carries
   drug-specific information at this data scale, with no network at all.

Decision rules (fixed before running):
   oracle O >= 0.15 (val signal median)                -> transmission is worth fixing
                                                           (per-node input sparsity, 12p)
   O < 0.05 and T >= 0.15                              -> the information is in target
                                                           identity but not reachable through
                                                           curated topology: a graph/knowledge
                                                           limit, not a transmission limit
   O < 0.05 and T < 0.05                               -> identifiability / data limit at this
                                                           scale; a transmission fix cannot help
   anything else                                        -> ambiguous; report, do not build
   A value counts only if its permutation p < 0.05 (otherwise it is treated as 0.05-band).
"""
import numpy as np
import scipy.sparse as sp
import torch

ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
KMAX = 6


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def adjacency(model, kind, weights=None):
    """Signed sparse A[target, source] over the model's nodes."""
    e = model.edges
    src = e.source.map(model.idx).values
    tgt = e.target.map(model.idx).values
    n = len(model.nodes)
    if kind in ("curated", "curated_indeg"):
        k = np.isin(model.tier, ["T0", "T1"])
        w = np.sign(e.sign.values[k]).astype(float)
        if kind == "curated_indeg":
            deg = np.bincount(tgt[k], minlength=n).astype(float)
            w = w / np.maximum(deg[tgt[k]], 1.0)
        return sp.csr_matrix((w, (tgt[k], src[k])), shape=(n, n))
    if kind == "trained":
        w = weights.detach().cpu().double().numpy()
        return sp.csr_matrix((w, (tgt, src)), shape=(n, n))
    raise ValueError(kind)


def hops(A, delta, kmax=KMAX):
    """[H_0 .. H_K], H_k = A^k delta (nodes x drugs)."""
    H = [delta]
    for _ in range(kmax):
        H.append(A @ H[-1])
    return H


def predict(H, tf_idx, alpha, K, scale):
    return sum((alpha ** k) * H[k][tf_idx] / scale[k] for k in range(K + 1))


def resid_scores(P, Y, train_cols, eval_cols):
    """Residualised r per eval column: both minus the training-column mean."""
    with np.errstate(all="ignore"):
        mp, my = np.nanmean(P[:, train_cols], axis=1), np.nanmean(Y[:, train_cols], axis=1)
    return np.array([corr(P[:, j] - mp, Y[:, j] - my) for j in eval_cols])


def perm_p(P, Y, train_cols, eval_cols, n=5000, seed=0):
    """Median matched residualised r vs random relabellings of the eval drugs."""
    with np.errstate(all="ignore"):
        mp, my = np.nanmean(P[:, train_cols], axis=1), np.nanmean(Y[:, train_cols], axis=1)
    R = np.array([[corr(P[:, i] - mp, Y[:, j] - my) for j in eval_cols] for i in eval_cols])
    obs = np.nanmedian(np.diag(R))
    rng = np.random.default_rng(seed)
    null = [np.nanmedian(R[np.arange(len(eval_cols)), rng.permutation(len(eval_cols))]) for _ in range(n)]
    off = R[~np.eye(len(eval_cols), dtype=bool)]
    return float(obs), float(np.nanmedian(off)), float(np.mean(np.array(null) >= obs))


def target_identity(Y, targets, classes, train_cols, eval_cols, mode="target"):
    """Predict each eval drug by the mean residual profile of training drugs sharing a
    target (mode 'target') or a target class (mode 'class'); NaN when none share."""
    with np.errstate(all="ignore"):
        my = np.nanmean(Y[:, train_cols], axis=1)
    R = Y - my[:, None]
    P = np.full_like(Y, np.nan)
    for j in eval_cols:
        share = [i for i in train_cols if (targets[i] & targets[j] if mode == "target" else classes[i] == classes[j])]
        if share:
            P[:, j] = np.nanmean(R[:, share], axis=1)
    return P
