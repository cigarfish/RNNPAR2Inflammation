"""Step 12f0: how good can r(TF) on the drug conditions be, and what does nothing-clever score?

Two reference points the model's val r(drug) (C4 0.180, stage A 0.224, all-GEO 0.035) has never
been compared against:

1. **Noise ceiling** (GSE219229 only; it has the raw counts, 3 replicates per drug, 6 DMSO).
   Split each drug's contrast into two halves with *disjoint* samples
       half 1 = 1 drug replicate vs 3 DMSO,   half 2 = the other 2 drug replicates vs the other 3 DMSO
   compute the TF-activity profile of each half exactly as step 6 does (ULM t on z-scored
   log2FC, same regulons, same clip), and correlate them. Under signal variance S and an
   equal per-sample noise variance N, the halves carry noise 4N/3 and 5N/6 and the full
   contrast (3 vs 6) carries N/2, so with k = S/N
       r12 = k / sqrt((k + 4/3)(k + 5/6))          (solved for k)
       R   = k / (k + 1/2)                         reliability of the full-data target
       r_max = sqrt(R)                             best achievable r against that target
   Averaged over all 3 x 10 disjoint splits (Fisher z). log2FC here is log2 of DESeq2
   median-of-ratios normalised means (+1 pseudocount), not the pydeseq2 Wald LFC; the test
   checks that the two agree on the full data before the ceiling is trusted. The replicate-
   number batch (R1/R2/R3, ~27 % of residual variance) is removed first; see batch_corrected.

2. **Trivial baselines**, scored with the training metric (median over conditions of the
   per-condition Pearson r across TFs, TF set and clip as `Targets`), on the same validation
   fold as every step-12 variant. Each prediction uses training-fold conditions only:
       B_mean    mean TF profile of the training drugs
       B_class   mean of training drugs with the same target class (falls back to B_mean)
       B_target  mean of training drugs sharing >= 1 target gene (falls back to B_mean)
   and for GEO, B_mean over training GEO contexts (is GEO r 0.52 just "generic inflammation"?).
   All baselines are also scored out-of-fold over all 5 folds for a less noisy number.

Decision rules (fixed before running; see HANDOFF step 12f0):
   * B_mean(drug) >= model C4 (0.180): the drug fit does not beat a constant profile; the
     drug r numbers in REPORT.md must be restated against it.
   * median r_max over the validation drugs < 0.40: the drug targets are mostly noise; drug r
     is not a usable selection metric for 12j, restrict it to drugs with r_max >= 0.5.
   * otherwise report model r / r_max as the fraction of explainable signal captured.
"""
import itertools
import json
import re

import numpy as np
import pandas as pd

import scipy.sparse as sp

from .conditions import tf_activity, zscore

REF_GROUPS = ("DMSOs1", "DMSOs2")
PSEUDO = 0.5   # log2(mean + 0.5): closest to the pydeseq2 targets (TF profile r 0.98 vs 0.94 at +1)


# ---------- counts ----------
def load_counts(cfg):
    """GSE219229 counts as genes x samples (duplicate symbols summed, harmonised symbols)."""
    from .genes import symbol_map
    d = pd.read_csv(cfg.root / "curation" / "GSE219229.csv")
    d["Symbol"] = d.Symbol.astype(str).str.strip()
    m = symbol_map(cfg)
    d["Symbol"] = d.Symbol.map(lambda x: m.get(x, x))
    X = d.groupby("Symbol").sum()
    return X.loc[X.sum(axis=1) >= 10]


def sample_groups(columns):
    """{group: [columns]}; the two DMSO groups are merged into 'DMSO'."""
    out = {}
    for c in columns:
        g = re.sub(r"_2hLPS_R\d+$", "", c)
        out.setdefault("DMSO" if g in REF_GROUPS else g, []).append(c)
    return out


def size_factors(X):
    """DESeq2 median-of-ratios size factors over genes expressed in every sample."""
    with np.errstate(divide="ignore"):
        L = np.log(X.values.astype(float))
    ok = np.isfinite(L).all(axis=1)
    geo = L[ok].mean(axis=1, keepdims=True)
    return pd.Series(np.exp(np.median(L[ok] - geo, axis=0)), index=X.columns)


def log2fc(N, trt, ref, pseudo=PSEUDO):
    """log2 of normalised mean(trt) / mean(ref)."""
    return np.log2(N[trt].mean(axis=1) + pseudo) - np.log2(N[ref].mean(axis=1) + pseudo)


def replicate_of(columns):
    return pd.Series([re.search(r"_R(\d+)$", c).group(1) for c in columns], index=columns)


def batch_corrected(N, G, pseudo=PSEUDO):
    """Normalised counts with the replicate-number batch removed (multiplicatively).

    R1/R2/R3 of every drug and of DMSO were evidently processed together: the R-number
    explains ~27 % of the within-group residual variance (same-R residuals correlate +0.25
    across drugs, different-R -0.13). The full 3-vs-6 contrasts are balanced over R, so the
    batch cancels in the training targets, but a 1-vs-2 replicate split is not balanced and
    the batch then makes the halves anti-correlate. Estimated per gene as the mean
    within-group log2 residual of each R-number over all samples, divided out of the counts."""
    L = np.log2(N + pseudo)
    res = pd.concat([L[c].sub(L[c].mean(axis=1), axis=0) for c in G.values()], axis=1)[L.columns]
    rep = replicate_of(L.columns)
    b = res.T.groupby(rep).mean().T[rep.values].set_axis(L.columns, axis=1)
    return N * np.power(2.0, -b)


class ULM:
    """`conditions.tf_activity` for a fixed gene set, with the regulon design built once."""

    def __init__(self, genes, regulon, tfs, min_targets, clip):
        reg = regulon[(regulon.sign != 0) & regulon.tf.isin(tfs)]
        gi = {g: i for i, g in enumerate(genes)}
        ti = {t: i for i, t in enumerate(tfs)}
        r = reg[reg.target.isin(gi)]
        self.X = sp.csr_matrix((r.sign.astype(float).values, ([gi[g] for g in r.target], [ti[t] for t in r.tf])),
                               shape=(len(genes), len(tfs)))
        self.n = len(genes)
        self.sx = np.asarray(self.X.sum(axis=0)).ravel()
        self.sxx = np.asarray(self.X.multiply(self.X).sum(axis=0)).ravel() - self.sx ** 2 / self.n
        self.bad = np.asarray((self.X != 0).sum(axis=0)).ravel() < min_targets
        self.XT = self.X.T.tocsr()
        self.genes, self.tfs, self.clip = pd.Index(genes), list(tfs), clip

    def __call__(self, lfc):
        zv = zscore(lfc.reindex(self.genes)).values
        sxy = self.XT @ zv - self.sx * zv.mean()
        syy = ((zv - zv.mean()) ** 2).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            b = sxy / self.sxx
            t = b / np.sqrt(np.maximum(syy - b * sxy, 1e-12) / (self.n - 2) / self.sxx)
        t[self.bad] = np.nan
        return pd.Series(np.clip(t, -self.clip, self.clip), index=self.tfs)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


# ---------- ceiling ----------
def ceiling_from_r12(r12, n1=(1, 3), n2=(2, 3), full=(3, 6)):
    """Solve r12 = k / sqrt((k + a1)(k + a2)) for k = S/N, return (k, R_full, r_max).
    a = 1/n_drug + 1/n_ref for each half; full reliability R = k / (k + a_full)."""
    a1, a2, af = (1 / n1[0] + 1 / n1[1]), (1 / n2[0] + 1 / n2[1]), (1 / full[0] + 1 / full[1])
    if not np.isfinite(r12) or r12 <= 0:
        return 0.0, 0.0, 0.0
    r12 = min(r12, 1 - 1e-12)
    # r^2 (k + a1)(k + a2) = k^2  ->  (1 - r^2) k^2 - r^2 (a1 + a2) k - r^2 a1 a2 = 0
    A, B, C = 1 - r12 ** 2, -r12 ** 2 * (a1 + a2), -r12 ** 2 * a1 * a2
    k = (-B + np.sqrt(B * B - 4 * A * C)) / (2 * A)
    R = k / (k + af)
    return float(k), float(R), float(np.sqrt(R))


def split_half(cfg, regulon, tfs, drugs=None, min_targets=5, clip=10.0, max_splits=None):
    """Per drug: mean split-half r of TF profiles (Fisher z over disjoint splits) and the ceiling."""
    X = load_counts(cfg)
    N = X / size_factors(X)
    G = sample_groups(X.columns)
    Nc = batch_corrected(N, G)
    ref = sorted(G["DMSO"])
    ref_splits = [(list(a), [c for c in ref if c not in a])
                  for a in itertools.combinations(ref, len(ref) // 2) if ref[0] in a]  # 10 unordered halves
    ulm = ULM(Nc.index, regulon, tfs, min_targets, clip)
    rows, full = [], {}
    for drug in sorted(drugs or (set(G) - {"DMSO"})):
        reps = sorted(G.get(drug, []))
        if len(reps) != 3:
            continue
        full[drug] = log2fc(Nc, reps, ref)
        zs = []
        for i, (ra, rb) in itertools.product(range(3), ref_splits):
            for r_a, r_b in ((ra, rb), (rb, ra)):  # both ways round, the 1-rep half gets each DMSO half
                h1 = ulm(log2fc(Nc, [reps[i]], r_a))
                h2 = ulm(log2fc(Nc, [x for j, x in enumerate(reps) if j != i], r_b))
                zs.append(np.arctanh(np.clip(corr(h1.values, h2.values), -0.999999, 0.999999)))
                if max_splits and len(zs) >= max_splits:
                    break
            if max_splits and len(zs) >= max_splits:
                break
        r12 = float(np.tanh(np.nanmean(zs)))
        k, R, rmax = ceiling_from_r12(r12)
        rows.append(dict(drug=drug, condition_id=f"GSE219229_{drug}", n_splits=len(zs), r12=r12,
                         r12_sd=float(np.nanstd(np.tanh(zs))), snr_k=k, reliability=R, r_max=rmax))
    return pd.DataFrame(rows), pd.DataFrame(full)


def shared_reference(cfg, regulon, tfs, min_targets=5, clip=10.0):
    """How much of the mean-drug-profile baseline is the shared DMSO reference?

    Every drug contrast uses the same 6 DMSO samples, so each target carries the same
    '- DMSO noise' term and the mean of the other drugs predicts it. Split the DMSO into
    its two groups: target = drug vs DMSO half A, predictor = mean of the other drugs vs
    half A (shared) or vs half B (disjoint; reference noise independent). Both orientations.
    The disjoint r is the genuine common drug response; shared - disjoint is the artefact."""
    X = load_counts(cfg)
    N = X / size_factors(X)
    G = sample_groups(X.columns)
    Nc = batch_corrected(N, G)
    ulm = ULM(Nc.index, regulon, tfs, min_targets, clip)
    halves = [[c for c in G["DMSO"] if c.startswith(g)] for g in REF_GROUPS]
    drugs = [g for g in G if g != "DMSO" and len(G[g]) == 3]
    prof = [pd.DataFrame({g: ulm(log2fc(Nc, G[g], h)) for g in drugs}) for h in halves]
    rows = []
    for g in drugs:
        for a, b in ((0, 1), (1, 0)):
            t = prof[a][g].values
            rows.append(dict(drug=g, orientation=a,
                             r_shared=corr(t, prof[a].drop(columns=g).mean(axis=1).values),
                             r_disjoint=corr(t, prof[b].drop(columns=g).mean(axis=1).values)))
    return pd.DataFrame(rows).groupby("drug")[["r_shared", "r_disjoint"]].mean().reset_index()


# ---------- baselines ----------
def score(pred, target):
    """Per-condition Pearson r over TFs measured in the target (training metric)."""
    return pd.Series({c: corr(pred[c].values, target[c].values) for c in target.columns})


def baseline_predictions(tfa, cond, val_ids, train_ids):
    """{name: DataFrame TF x val condition} built from training conditions only."""
    tr = tfa[train_ids]
    mean = tr.mean(axis=1)
    info = cond.set_index("condition_id")
    def targets(v):
        return set(json.loads(v)) if isinstance(v, str) else set(v)
    tgt = {c: targets(info.at[c, "trt_inhibit"]) for c in list(val_ids) + list(train_ids)}
    out = {"B_mean": pd.DataFrame({c: mean for c in val_ids})}
    if (info.loc[list(val_ids), "source"] == "drug").all():
        cls, shr = {}, {}
        for c in val_ids:
            same = [t for t in train_ids if info.at[t, "cls"] == info.at[c, "cls"]]
            cls[c] = tr[same].mean(axis=1) if same else mean
            share = [t for t in train_ids if tgt[t] & tgt[c]]
            shr[c] = tr[share].mean(axis=1) if share else mean
        out["B_class"], out["B_target"] = pd.DataFrame(cls), pd.DataFrame(shr)
    return out


def evaluate_baselines(tfa, cond, source, val_fold=0, folds=None):
    """Median r of each baseline on `val_fold`, plus out-of-fold over all folds."""
    c = cond[(cond.source == source) & cond.fold.notna() & (cond.fold >= 0)]
    c = c[c.condition_id.isin(tfa.columns)]
    per = []
    for f in sorted(c.fold.unique()) if folds is None else folds:
        val = list(c.condition_id[c.fold == f])
        train = list(c.condition_id[c.fold != f])
        for name, P in baseline_predictions(tfa, cond, val, train).items():
            r = score(P, tfa[val])
            per += [dict(baseline=name, fold=int(f), condition_id=k, r=v) for k, v in r.items()]
    per = pd.DataFrame(per)
    summ = (per.groupby("baseline")
            .apply(lambda d: pd.Series(dict(val_fold_median=d.r[d.fold == val_fold].median(),
                                            oof_median=d.r.median(), n_val=int((d.fold == val_fold).sum()))),
                   include_groups=False))
    return summ, per
