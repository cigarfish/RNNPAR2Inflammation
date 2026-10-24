"""Step 16: can any method predict drug-specific TF-activity responses from drug target profiles?

Every method gets the same inputs (measured kinase occupancy at the dose used, step 12q union
profile) and the same scoring as the step-12f harness selection metric:

    residualised r = per-condition Pearson r after removing the TRAINING-fold drug mean profile
                     from both prediction and target

so a method that only reproduces the common drug response scores 0, and raw r is reported next to
it to make that visible. LEMBAS trained once with CV fold 0 held out, so the primary comparison is
fold 0 for every method; the learners also get a 5-fold out-of-fold number, which LEMBAS cannot
have without retraining per fold.

No torch: the signalling graph is read from the step-5 edge table with scipy, and LEMBAS's own
per-condition scores are read from its frozen harness JSON.
"""
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

SIGNAL_RMAX = 0.30          # same signal-drug subset as the harness
PCS = 20                    # latent dimensions for the tree ensembles


# ---------- data ----------
def drug_table(cfg):
    """(X occupancy, Y TF activity, folds, ids, r_max) over GSE219229 drugs with a kinome profile."""
    from .genes import symbol_map
    from .kinome import occupancy_matrix
    o = cfg.paths.output_dir
    occ = occupancy_matrix(cfg, symbol_map(cfg))
    cond = pd.read_csv(o / "step06_conditions.csv").set_index("condition_id")
    tfa = pd.read_csv(o / "step06_tf_activity.csv.gz", index_col=0)
    rmax = pd.read_csv(o / "step12f0" / "ceiling.csv").set_index("condition_id").r_max
    ann = pd.read_csv(cfg.root / "curation" / "drug_targets.csv").set_index("drug").targets
    ids = [c for c in occ.index if c in cond.index and c in tfa.columns and cond.at[c, "source"] == "drug"]
    X = occ.loc[ids].copy()
    for c in ids:                                     # union with the annotated targets (step 12q)
        for t in str(ann.get(cond.at[c, "name"], "")).split(";"):
            if t:
                X.loc[c, t] = max(float(X.get(t, pd.Series(0.0, index=ids)).get(c, 0.0)), 0.9)
    X = X.fillna(0.0)
    Y = tfa[ids].clip(-cfg.loss["tf_target_clip"], cfg.loss["tf_target_clip"])
    folds = np.array([int(cond.at[c, "fold"]) for c in ids])
    return X, Y, folds, ids, np.array([float(rmax.get(c, np.nan)) for c in ids])


def adjacency(cfg, tiers=("T0", "T1")):
    """Signed sparse A[target, source] and the node index, from the step-5 edge table."""
    o = cfg.paths.output_dir
    e = pd.read_csv(o / "step05_signalling_edges.csv.gz", usecols=["source", "target", "sign", "tier"])
    e = e[e.tier.isin(tiers)]
    nodes = sorted(set(e.source) | set(e.target))
    idx = {n: i for i, n in enumerate(nodes)}
    A = sp.csr_matrix((np.sign(e.sign.values).astype(float),
                       (e.target.map(idx).values, e.source.map(idx).values)), shape=(len(nodes),) * 2)
    return A, idx


# ---------- metric ----------
def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def score(P, Y, train, test, resid=True):
    """Per-test-column r; with resid, both sides minus the training-column mean profile."""
    Yv, Pv = Y.values, np.asarray(P)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        my = np.nanmean(Yv[:, train], axis=1)
        mp = np.nanmean(Pv[:, train], axis=1) if resid else 0.0
    return np.array([corr(Pv[:, j] - (mp if resid else 0.0), Yv[:, j] - (my if resid else 0.0)) for j in test])


def permutation_p(P, Y, train, test, n=5000, seed=0):
    """Median matched residualised r vs random relabellings of the test conditions."""
    Yv, Pv = Y.values, np.asarray(P)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        my, mp = np.nanmean(Yv[:, train], axis=1), np.nanmean(Pv[:, train], axis=1)
    R = np.array([[corr(Pv[:, i] - mp, Yv[:, j] - my) for j in test] for i in test])
    obs = np.nanmedian(np.diag(R))
    rng = np.random.default_rng(seed)
    null = [np.nanmedian(R[np.arange(len(test)), rng.permutation(len(test))]) for _ in range(n)]
    return float(obs), float(np.nanmedian(R[~np.eye(len(test), dtype=bool)])), float(np.mean(np.array(null) >= obs))


# ---------- methods ----------
def _resid_fit_space(Y, train):
    """Centre Y on the training mean profile: every learner predicts the DEVIATION from it, so no
    method can win by reproducing the common drug response."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        my = np.nanmean(Y.values[:, train], axis=1)
    R = np.nan_to_num(Y.values - my[:, None])
    return R, my


def fit_linear(kind, X, Y, train, test, alphas, seed=0):
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.model_selection import KFold
    R, my = _resid_fit_space(Y, train)
    Xt, Yt = X.values[train], R[:, train].T
    best, bo = None, -np.inf
    inner = KFold(n_splits=min(5, len(train)), shuffle=True, random_state=seed)
    for a in alphas:
        s = []
        for tr, va in inner.split(Xt):
            mdl = Ridge(alpha=a) if kind == "ridge" else ElasticNet(alpha=a, l1_ratio=0.5, max_iter=5000)
            mdl.fit(Xt[tr], Yt[tr])
            pv = mdl.predict(Xt[va])
            s += [corr(pv[i], Yt[va][i]) for i in range(len(va))]
        s = np.nanmedian(s)
        if s > bo:
            bo, best = s, a
    mdl = Ridge(alpha=best) if kind == "ridge" else ElasticNet(alpha=best, l1_ratio=0.5, max_iter=5000)
    mdl.fit(Xt, Yt)
    P = np.full(Y.shape, np.nan)
    P[:, test] = mdl.predict(X.values[test]).T
    P[:, train] = mdl.predict(Xt).T
    return P + my[:, None], dict(alpha=float(best), inner_score=float(bo))


def fit_pls(X, Y, train, test, comps, seed=0):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    R, my = _resid_fit_space(Y, train)
    Xt, Yt = X.values[train], R[:, train].T
    best, bo = None, -np.inf
    inner = KFold(n_splits=min(5, len(train)), shuffle=True, random_state=seed)
    for k in [c for c in comps if c < len(train) - 1]:
        s = []
        for tr, va in inner.split(Xt):
            m = PLSRegression(n_components=min(k, len(tr) - 1)).fit(Xt[tr], Yt[tr])
            pv = m.predict(Xt[va])
            s += [corr(pv[i], Yt[va][i]) for i in range(len(va))]
        s = np.nanmedian(s)
        if s > bo:
            bo, best = s, k
    m = PLSRegression(n_components=best).fit(Xt, Yt)
    P = np.full(Y.shape, np.nan)
    P[:, test] = m.predict(X.values[test]).T
    P[:, train] = m.predict(Xt).T
    return P + my[:, None], dict(n_components=int(best), inner_score=float(bo))


def fit_tree(kind, X, Y, train, test, seed=0, n_pcs=PCS):
    """Tree ensembles on the top principal components of the training deviations (577 separate
    models would be slow and no better); PCA is fitted on training conditions only."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    R, my = _resid_fit_space(Y, train)
    k = min(n_pcs, len(train) - 1)
    pca = PCA(n_components=k, random_state=seed).fit(R[:, train].T)
    Z = pca.transform(R[:, train].T)
    Xt = X.values[train]
    pred = np.zeros((len(test) + len(train), k))
    Xall = np.vstack([X.values[test], Xt])
    for c in range(k):
        m = (GradientBoostingRegressor(random_state=seed, n_estimators=300, max_depth=2, learning_rate=0.05)
             if kind == "gbm" else
             RandomForestRegressor(random_state=seed, n_estimators=500, min_samples_leaf=2, n_jobs=-1))
        m.fit(Xt, Z[:, c])
        pred[:, c] = m.predict(Xall)
    back = pca.inverse_transform(pred).T
    P = np.full(Y.shape, np.nan)
    P[:, test] = back[:, :len(test)]
    P[:, train] = back[:, len(test):]
    return P + my[:, None], dict(n_pcs=int(k), explained=float(pca.explained_variance_ratio_.sum()))


def fit_knn(X, Y, train, test, **_):
    """Cosine-similarity weighted mean of the training deviation profiles (positive weights only)."""
    R, my = _resid_fit_space(Y, train)
    n = np.linalg.norm(X.values, axis=1)
    P = np.full(Y.shape, np.nan)
    for j in list(test) + list(train):
        s = np.array([X.values[i] @ X.values[j] / (n[i] * n[j]) if n[i] > 0 and n[j] > 0 and i != j else 0.0
                      for i in train])
        k = s > 0
        P[:, j] = (R[:, np.array(train)[k]] @ s[k]) / s[k].sum() if k.any() else 0.0
    return P + my[:, None], dict(n_neighbours_mean=float(np.mean([(X.values[train] @ X.values[j] > 0).sum() for j in test])))


def fit_diffusion(cfg, X, Y, train, test, alphas=(0.1, 0.25, 0.5, 1.0), kmax=4, seed=0):
    """Propagate -occupancy through the curated signed graph (truncated series, no stability cap),
    then fit one scalar per hop depth by inner CV; prediction = the propagated TF-node values."""
    o = cfg.paths.output_dir
    A, idx = adjacency(cfg)
    tfs = [t for t in Y.index]
    rows = [idx.get(t, -1) for t in tfs]
    d = np.zeros((A.shape[0], X.shape[0]))
    for j, c in enumerate(X.index):
        for g, v in X.loc[c].items():
            if v > 0 and g in idx:
                d[idx[g], j] = -float(v)
    H = [d]
    for _ in range(kmax):
        H.append(A @ H[-1])
    keep = np.array([r >= 0 for r in rows])
    Hs = [np.where(keep[:, None], h[np.clip(rows, 0, None)], np.nan) for h in H]
    R, my = _resid_fit_space(Y, train)
    best, bo = None, -np.inf
    for a in alphas:
        for K in range(1, kmax + 1):
            P = sum((a ** k) * np.nan_to_num(Hs[k]) for k in range(K + 1))
            s = np.nanmedian([corr(P[:, j], R[:, j]) for j in train])
            if s > bo:
                bo, best = s, (a, K)
    a, K = best
    P = sum((a ** k) * np.nan_to_num(Hs[k]) for k in range(K + 1))
    sd = np.nanstd(P[:, train]) or 1.0
    P = P / sd * (np.nanstd(R[:, train]) or 1.0)
    return P + my[:, None], dict(alpha=float(a), K=int(K), inner_score=float(bo))


def fit_constant(X, Y, train, test, **_):
    """The training-fold mean profile: raw r high, residualised r 0 by construction."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        my = np.nanmean(Y.values[:, train], axis=1)
    return np.repeat(my[:, None], Y.shape[1], axis=1), {}


METHODS = {
    "constant_mean_profile": lambda cfg, X, Y, tr, te, sd: fit_constant(X, Y, tr, te),
    "ridge": lambda cfg, X, Y, tr, te, sd: fit_linear("ridge", X, Y, tr, te, [0.1, 1, 10, 100, 1000], sd),
    "elastic_net": lambda cfg, X, Y, tr, te, sd: fit_linear("elastic", X, Y, tr, te, [0.001, 0.01, 0.1, 1.0], sd),
    "pls": lambda cfg, X, Y, tr, te, sd: fit_pls(X, Y, tr, te, [1, 2, 3, 5, 8, 12], sd),
    "gradient_boosting": lambda cfg, X, Y, tr, te, sd: fit_tree("gbm", X, Y, tr, te, sd),
    "random_forest": lambda cfg, X, Y, tr, te, sd: fit_tree("rf", X, Y, tr, te, sd),
    "knn_occupancy_cosine": lambda cfg, X, Y, tr, te, sd: fit_knn(X, Y, tr, te),
    "diffusion_curated_graph": lambda cfg, X, Y, tr, te, sd: fit_diffusion(cfg, X, Y, tr, te, seed=sd),
}


def sensitivity(cfg, name, X, Y, train, test, r_target=0.55, n_rep=3, seed=0):
    """Detection power: plant a linear response to the target profiles, add noise so the planted
    truth is recoverable only up to r_target (the measured noise ceiling), and report what the
    method actually recovers. A method that scores ~0 on the real data AND low here is
    underpowered at this sample size, which is a different statement from 'no signal in the data'.
    """
    rng = np.random.default_rng(seed)
    k = r_target ** 2 / (1 - r_target ** 2)
    vals = []
    for rep in range(n_rep):
        W = rng.normal(size=(X.shape[1], Y.shape[0]))
        S = (X.values @ W).T
        Ysyn = pd.DataFrame(S + rng.normal(size=S.shape) * S.std() / np.sqrt(k), index=Y.index, columns=Y.columns)
        P, _ = METHODS[name](cfg, X, Ysyn, train, test, seed + rep)
        vals.append(float(np.nanmedian(score(P, Ysyn, train, test))))
    return float(np.median(vals)), vals


def lembas_oof_scores(cfg, variant="R4", seeds=(0, 1, 2), folds=(0, 1, 2, 3, 4)):
    """Per-drug OUT-OF-FOLD r of LEMBAS, from the step-20 cross-validation runs.

    `lembas_scores` below reads one model that held out a single fold, so its per-drug values
    cover that fold only. This reads the five models of step 20, each scored on the fold it
    held out, so every drug appears exactly once per seed - the same construction the learners'
    own out-of-fold numbers use, which is what makes a paired test against them legitimate."""
    import json
    o = cfg.paths.output_dir
    rows = []
    for f in folds:
        for sd in seeds:
            stem = (variant if sd == 0 else f"{variant}_s{sd}") + f"_f{f}"
            p = o / "harness" / f"12j_{stem}.json"
            if not p.exists():
                continue
            for c in json.loads(p.read_text())["per_condition"]:
                if c["source"] == "drug":
                    rows.append(dict(seed=sd, fold=f, condition_id=c["condition_id"],
                                     r=c["r"], r_resid=c["r_resid"], r_max=c["r_max"]))
    d = pd.DataFrame(rows)
    if len(d):
        dup = d.groupby(["seed", "condition_id"]).size()
        if (dup > 1).any():
            raise ValueError("a drug was scored in more than one held-out fold: "
                             + str(dup[dup > 1].to_dict()))
    return d


def lembas_scores(cfg, variant="J0", seeds=(0, 1, 2, 3, 4)):
    """Per-drug residualised and raw r of the frozen LEMBAS model, from its harness JSON."""
    import json
    o = cfg.paths.output_dir
    rows = []
    for s in seeds:
        p = o / "harness" / f"12j_{variant if s == 0 else f'{variant}_s{s}'}.json"
        if not p.exists():
            continue
        for c in json.loads(p.read_text())["per_condition"]:
            if c["source"] == "drug":
                rows.append(dict(seed=s, condition_id=c["condition_id"], r=c["r"], r_resid=c["r_resid"],
                                 r_max=c["r_max"]))
    return pd.DataFrame(rows)
