"""Step 12h (items 2, 4, 5 of TASKS_12f-12n): spectral audit of a trained model. Read-only.

Items 1 and 3 (the three radii, headroom 1.62x, SCC census) were measured in the plan
review recorded in HANDOFF.md and are only re-reported here.

2  Perron sensitivities. With u, v the left / right Perron vectors of |A| (loop edges,
   float64), d rho / d|A_ij| = u_i v_j / (u . v). Since rho is homogeneous of degree 1
   in |A|, the loadings |w_e| * s_e sum exactly to rho (Euler) - the consistency check.
   Reported by tier, for edges touching the 34 route nodes, and for the top 1 %.
   This is what a Perron-targeted projection (12k) would act on.
4  Feedback-arc curve. Remove T3 edges in ascending STRING score - never T0/T1/T2, never
   an edge touching a route node - and report the giant SCC and rho(|A|) of the remaining
   (unrescaled) weights. Nothing is cut in the model.
5  Loop-gain headroom 1/(1 - rho) at the current rho and at 0.99 / 0.995.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

from .core import _perron_vector, loop_edge_mask

FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.0]


def route_nodes(model):
    e = model.edges[model.tier == "T0"]
    return set(e.source) | set(e.target)


@torch.no_grad()
def perron(struct, w, mask=None, tol=1e-13, max_iter=50000):
    """(rho, u, v, residual_v, residual_u) of |A| restricted to `mask` (default: loop edges)."""
    mask = loop_edge_mask(struct) if mask is None else mask
    aw = (w.abs() * mask).double()
    M, MT = struct.matrix(aw), struct.matrix_T(aw)
    v, lo, hi, _ = _perron_vector(M, struct.n, max_iter=max_iter, tol=tol, dtype=torch.float64, device=w.device)
    u, _, hi_u, _ = _perron_vector(MT, struct.n, max_iter=max_iter, tol=tol, dtype=torch.float64, device=w.device)
    rho = hi
    rv = float((M @ v[:, None]).squeeze(1).sub(rho * v).norm() / v.norm())
    ru = float((MT @ u[:, None]).squeeze(1).sub(rho * u).norm() / u.norm())
    return rho, u, v, rv, ru, hi_u


@torch.no_grad()
def sensitivities(model):
    st, w = model.struct, model.weights().detach()
    rho, u, v, rv, ru, rho_u = perron(st, w)
    loop = loop_edge_mask(st)
    s = (u[st.tgt] * v[st.src] / (u * v).sum()) * loop.double()
    load = (w.abs().double() * s)
    e = model.edges
    rn = route_nodes(model)
    touch = (e.source.isin(rn) | e.target.isin(rn)).values
    L, S = load.cpu().numpy(), s.cpu().numpy()
    tot = float(L.sum())
    by_tier = {}
    for t in ["T0", "T1", "T2", "T3"]:
        k = model.tier == t
        by_tier[t] = dict(n_edges=int(k.sum()), n_loop=int((loop.cpu().numpy() & k).sum()),
                          loading_share=float(L[k].sum() / tot),
                          median_sensitivity=float(np.median(S[k & (S > 0)])) if (k & (S > 0)).any() else 0.0)
    top = np.sort(L)[::-1]
    n1 = max(1, int(0.01 * len(L)))
    t0 = model.tier == "T0"
    return dict(
        rho=rho, rho_left=rho_u, residual_right=rv, residual_left=ru,
        euler_sum=tot, euler_rel_error=abs(tot - rho) / rho,
        u_min=float(u.min()), v_min=float(v.min()),
        by_tier=by_tier,
        route_touching=dict(n_edges=int(touch.sum()), loading_share=float(L[touch].sum() / tot)),
        bulk_T3_not_route=float(L[(model.tier == "T3") & ~touch].sum() / tot),
        top1pct_share=float(top[:n1].sum() / tot), top1pct_n=n1,
        route_T0=dict(loading_share=float(L[t0].sum() / tot), max_sensitivity=float(S[t0].max()),
                      median_sensitivity=float(np.median(S[t0])),
                      n_T0_outside_loops=int((~loop.cpu().numpy() & t0).sum())),
    ), pd.DataFrame(dict(source=e.source.values, target=e.target.values, tier=model.tier,
                         w=w.cpu().numpy(), sensitivity=S, loading=L, route_touching=touch))


def _giant_scc(src, tgt, n):
    g = sp.csr_matrix((np.ones(len(src)), (src, tgt)), shape=(n, n))
    _, comp = connected_components(g, directed=True, connection="strong")
    sizes = np.bincount(comp)
    big = int(sizes.argmax())
    inside = (comp[src] == comp[tgt])
    return int(sizes[big]), int(inside.sum()), len(sizes)


@torch.no_grad()
def feedback_arc_curve(model, string_score, fractions=FRACTIONS):
    """string_score: array aligned to model.edges (NaN where absent)."""
    st, w = model.struct, model.weights().detach()
    e = model.edges
    rn = route_nodes(model)
    touch = (e.source.isin(rn) | e.target.isin(rn)).values
    elig = np.where((model.tier == "T3") & ~touch)[0]
    order = elig[np.argsort(np.nan_to_num(string_score[elig], nan=-1.0), kind="stable")]
    src, tgt = st.src.cpu().numpy(), st.tgt.cpu().numpy()
    rows = []
    for f in fractions:
        k = int(round(f * len(order)))
        keep = np.ones(len(e), dtype=bool)
        keep[order[:k]] = False
        g_nodes, n_loop, n_scc = _giant_scc(src[keep], tgt[keep], st.n)
        km = torch.tensor(keep, device=w.device)
        rho, *_ = perron(st, w * km, mask=torch.ones_like(km), tol=1e-10, max_iter=20000)
        rows.append(dict(fraction_T3_cut=f, n_cut=k, max_score_cut=float(string_score[order[k - 1]]) if k else np.nan,
                         giant_scc_nodes=g_nodes, giant_scc_frac=g_nodes / st.n, n_scc=n_scc,
                         edges_in_loops_frac=n_loop / int(keep.sum()), rho_absA=rho))
    # reference: no T2/T3 at all (curated signed + route only)
    keep = np.isin(model.tier, ["T0", "T1"])
    g_nodes, n_loop, _ = _giant_scc(src[keep], tgt[keep], st.n)
    km = torch.tensor(keep, device=w.device)
    rho_c, *_ = perron(st, w * km, mask=torch.ones_like(km), tol=1e-10, max_iter=20000)
    ref = dict(T0_T1_only_giant_scc_frac=g_nodes / st.n, T0_T1_only_rho=rho_c, n_eligible=len(order))
    return pd.DataFrame(rows), ref


def loop_gain(rho):
    return {f"{r:.3f}": 1.0 / (1.0 - r) for r in [rho, 0.99, 0.995]}


@torch.no_grad()
def targeted_headroom(model, target=0.95):
    """How much uniform scale the non-T0 edges gain at rho <= target if the edges that
    close loops through the route are removed. T0 alone is acyclic (rho ~1e-3): the route's
    near-unit weights amplify every loop that passes through it, so this bounds what any
    Perron-targeted reallocation (12k) can buy without touching the route."""
    st, w = model.struct, model.weights().detach()
    e, rn = model.edges, route_nodes(model)
    dev = w.device
    t0 = torch.tensor(model.tier == "T0", device=dev)
    inner = torch.tensor(((model.tier != "T0") & e.source.isin(rn) & e.target.isin(rn)).values, device=dev)
    touch = torch.tensor(((model.tier != "T0") & (e.source.isin(rn) | e.target.isin(rn))).values, device=dev)
    ones = torch.ones_like(t0)

    def rho(ww):
        return perron(st, ww, mask=ones, tol=1e-10, max_iter=20000)[0]

    def max_scale(zero):
        base = torch.where(zero, torch.zeros_like(w), w)
        lo, hi = 0.0, 50.0
        for _ in range(30):
            s = (lo + hi) / 2
            lo, hi = (s, hi) if rho(torch.where(t0, base, base * s)) <= target else (lo, s)
        return lo

    out = dict(rho=rho(w), rho_T0_alone=rho(torch.where(t0, w, torch.zeros_like(w))))
    for name, z in [("none", torch.zeros_like(t0)), ("route_internal", inner), ("route_touching", touch)]:
        out[name] = dict(n_removed=int(z.sum()), rho_after=rho(torch.where(z, torch.zeros_like(w), w)),
                         max_scale=max_scale(z))
    return out
