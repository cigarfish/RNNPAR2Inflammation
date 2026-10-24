"""Step 7: LEMBAS core, written in torch (CPU or GPU, float32 or float64).

7a  MML activation (Nilsson et al. 2022, LEMBAS):
        f(z) = leak * z          z < 0
             = z                 0 <= z <= 0.5
             = 1 - 0.25 / z      z > 0.5
    continuous with continuous slope at 0.5, bounded above by 1. Its derivative can
    be computed from the pre-activation z or from the output y = f(z), which is what
    the implicit backward pass needs at the steady state.

7b  Sparse network + forward steady state. Edges src -> tgt with weight w form
    A[tgt, src] = w (CSR, rows = targets). For a batch of B simulations the state
    X (n x B) solves
        X = f(A (S * X) + b + U)
    by fixed-point iteration from X = 0, where b are node biases, U the inputs and
    S an optional output scale per node and simulation (1 = untouched; used for
    inhibition in step 8).

7c  Implicit backward (LEMBAS adjoint). At the fixed point X = f(Z),
    Z = A (S*X) + b + U. With G = dL/dX the adjoint lam = dL/dZ solves
        lam = f'(Z) * (G + S * (A^T lam))
    (iterated like the forward pass), and
        dL/dw_e = sum_k lam[tgt_e, k] * (S*X)[src_e, k]
        dL/db   = sum_k lam[:, k]
        dL/dU   = lam
        dL/dS   = X * (A^T lam)
    Memory does not grow with the number of forward iterations. Optional tanh
    clipping of lam (as in LEMBAS) is off by default.

7d  Stability. Because f is 1-Lipschitz and 0 <= S <= 1, |X - Y| <= |A| |X - Y|
    elementwise, so rho(|A|) < 1 guarantees a unique steady state that the
    fixed-point iteration (forward and adjoint) reaches from any start, for every
    input and inhibition pattern. rho(|A|) is the Perron root of a non-negative
    matrix: computed by power iteration on |A| + I (aperiodic, same eigenvectors),
    with Collatz-Wielandt bounds as a certificate, and differentiable:
        d rho / d w_e = sign(w_e) u[tgt_e] v[src_e] / (u . v)
    (u, v = left / right Perron vectors). It upper-bounds the signed spectral
    radius rho(A) used by the original LEMBAS, which is reported for comparison.
"""
import warnings
from dataclasses import dataclass

import numpy as np
import torch

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")

LEAK = 0.01


# ---------- 7a: activation ----------
def mml(z, leak=LEAK):
    """MML activation, elementwise."""
    hi = 1.0 - 0.25 / torch.clamp(z, min=0.5)          # clamp avoids division warnings
    return torch.where(z < 0, leak * z, torch.where(z > 0.5, hi, z))


def mml_grad(z, leak=LEAK):
    """df/dz from the pre-activation (leak at z <= 0, as in LEMBAS)."""
    hi = 0.25 / torch.clamp(z, min=0.5) ** 2
    return torch.where(z <= 0, torch.full_like(z, leak), torch.where(z > 0.5, hi, torch.ones_like(z)))


def mml_grad_from_output(y, leak=LEAK):
    """df/dz expressed through the output y = f(z): 4 (1 - y)^2 for y > 0.5."""
    return torch.where(y <= 0, torch.full_like(y, leak),
                       torch.where(y > 0.5, 4.0 * (1.0 - y) ** 2, torch.ones_like(y)))


def mml_inverse(y, leak=LEAK):
    """z such that f(z) = y (y < 1)."""
    hi = 0.25 / torch.clamp(1.0 - y, min=1e-12)
    return torch.where(y < 0, y / leak, torch.where(y > 0.5, hi, y))


# ---------- 7b: sparse structure and forward solver ----------
@dataclass
class Structure:
    """Fixed sparsity pattern. Edge k (in the caller's order) sits at CSR slot perm[k]."""
    n: int
    src: torch.Tensor      # (E,) long, caller order
    tgt: torch.Tensor      # (E,) long, caller order
    crow: torch.Tensor     # (n+1,) CSR row pointers (rows = targets)
    col: torch.Tensor      # (E,) CSR column index (sources), CSR order
    order: torch.Tensor    # (E,) caller index of each CSR slot
    colT_crow: torch.Tensor  # CSR of A^T (rows = sources), for the backward pass
    colT_col: torch.Tensor
    orderT: torch.Tensor

    @staticmethod
    def build(src, tgt, n, device="cpu"):
        src = torch.as_tensor(np.asarray(src), dtype=torch.long, device=device)
        tgt = torch.as_tensor(np.asarray(tgt), dtype=torch.long, device=device)
        if src.numel() and (int(torch.stack([src, tgt]).max()) >= n or int(torch.stack([src, tgt]).min()) < 0):
            raise ValueError("edge index out of range")
        key = tgt * n + src
        if torch.unique(key).numel() != key.numel():
            raise ValueError("duplicate edges")

        def csr(rows, cols):
            order = torch.argsort(rows * n + cols)
            counts = torch.bincount(rows, minlength=n)
            crow = torch.zeros(n + 1, dtype=torch.long, device=device)
            crow[1:] = torch.cumsum(counts, 0)
            return crow, cols[order], order

        crow, col, order = csr(tgt, src)
        crowT, colT, orderT = csr(src, tgt)
        return Structure(n, src, tgt, crow, col, order, crowT, colT, orderT)

    def matrix(self, w):
        """A with A[tgt, src] = w (w in caller order)."""
        return torch.sparse_csr_tensor(self.crow, self.col, w[self.order], size=(self.n, self.n))

    def matrix_T(self, w):
        """A^T, i.e. [src, tgt] = w."""
        return torch.sparse_csr_tensor(self.colT_crow, self.colT_col, w[self.orderT], size=(self.n, self.n))

    def to(self, device):
        return Structure(self.n, *(getattr(self, f).to(device) for f in
                                   ["src", "tgt", "crow", "col", "order", "colT_crow", "colT_col", "orderT"]))


@dataclass
class Solution:
    X: torch.Tensor          # (n, B) steady state
    Z: torch.Tensor          # (n, B) pre-activation at the steady state
    iterations: int
    converged: bool
    residual: float          # max |X - f(A S X + b + U)|


def steady_state(struct, w, b, U, S=None, leak=LEAK, max_iter=300, tol=1e-6, check_every=5):
    """Fixed-point iteration X <- f(A (S*X) + b + U) from X = 0 (no autograd)."""
    with torch.no_grad():
        A = struct.matrix(w)
        bias = b.reshape(-1, 1)
        X = torch.zeros_like(U)
        it, converged = 0, False
        for it in range(1, max_iter + 1):
            Xs = X if S is None else S * X
            Z = A @ Xs + bias + U
            Xn = mml(Z, leak)
            if it % check_every == 0 or it == max_iter:
                delta = (Xn - X).abs().max().item()
                if not torch.isfinite(Xn).all():
                    X = Xn
                    break
                if delta < tol:
                    X = Xn
                    converged = True
                    break
            X = Xn
        Xs = X if S is None else S * X
        Z = A @ Xs + bias + U
        residual = (mml(Z, leak) - X).abs().max().item()
    return Solution(X, Z, it, converged and residual < 10 * tol, residual)


# ---------- 7c: implicit backward ----------
def _clip(g, c):
    """LEMBAS-style soft clipping: |g| > c is squashed with tanh."""
    return torch.where(g > c, torch.tanh(g - c) + c, torch.where(g < -c, torch.tanh(g + c) - c, g))


def adjoint(struct, w, Z, S, G, leak=LEAK, max_iter=300, tol=1e-6, clip=None, check_every=5):
    """Solve lam = f'(Z) * (G + S * (A^T lam)); returns (lam, iterations, converged)."""
    with torch.no_grad():
        AT = struct.matrix_T(w)
        D = mml_grad(Z, leak)
        lam = D * G
        it, converged = 0, False
        for it in range(1, max_iter + 1):
            back = AT @ lam
            new = D * (G + (back if S is None else S * back))
            if clip is not None:
                new = _clip(new, clip)
            if it % check_every == 0 or it == max_iter:
                if (new - lam).abs().max().item() < tol:
                    lam = new
                    converged = True
                    break
            lam = new
    return lam, it, converged


class _SteadyState(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, b, U, S, struct, opts):
        sol = steady_state(struct, w, b, U, S, **opts["forward"])
        ctx.struct, ctx.opts = struct, opts
        ctx.save_for_backward(w, sol.X, sol.Z, S if S is not None else torch.empty(0))
        ctx.has_S = S is not None
        ctx.info = dict(iterations=sol.iterations, converged=sol.converged, residual=sol.residual)
        opts["last_forward"] = ctx.info
        return sol.X

    @staticmethod
    def backward(ctx, G):
        w, X, Z, S = ctx.saved_tensors
        S = S if ctx.has_S else None
        lam, it, conv = adjoint(ctx.struct, w, Z, S, G.contiguous(), **ctx.opts["backward"])
        ctx.opts["last_backward"] = dict(iterations=it, converged=conv)
        st = ctx.struct
        Xs = X if S is None else S * X
        gw = (lam[st.tgt] * Xs[st.src]).sum(dim=1) if ctx.needs_input_grad[0] else None
        gb = lam.sum(dim=1) if ctx.needs_input_grad[1] else None
        gU = lam if ctx.needs_input_grad[2] else None
        gS = None
        if ctx.has_S and ctx.needs_input_grad[3]:
            gS = X * (st.matrix_T(w) @ lam)
        return gw, gb, gU, gS, None, None


def _perron_vector(M, n, transpose_M=None, max_iter=2000, tol=1e-10, dtype=None, device=None):
    """Power iteration on the non-negative matrix M + I from a positive start.
    Returns (vector, lower, upper, iterations) with the Collatz-Wielandt bounds
    lower <= rho(M) <= upper. The upper bound converges to rho(M) also for
    reducible M (where the lower bound can stay loose), so it is the estimate."""
    v = torch.ones(n, 1, dtype=dtype, device=device)
    lo, hi, prev, still = 0.0, float("inf"), float("inf"), 0
    for it in range(1, max_iter + 1):
        Mv = M @ v + v
        ratio = Mv / v
        lo, hi = ratio.min().item() - 1.0, ratio.max().item() - 1.0
        v = Mv / Mv.max()
        if hi - lo < tol * max(1.0, abs(hi)):
            break
        # the upper bound can plateau transiently; require it to be stable for a while
        still = still + 1 if abs(hi - prev) < tol * max(1.0, abs(hi)) else 0
        if still >= 20:
            break
        prev = hi
    return v.squeeze(1), lo, hi, it


def loop_edge_mask(struct):
    """Edges inside strongly connected components (feedback loops). rho(|A|) depends
    only on these; edges between components (acyclic parts) cannot create gain."""
    if getattr(struct, "_loop_mask", None) is None:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
        src, tgt = struct.src.cpu().numpy(), struct.tgt.cpu().numpy()
        g = sp.csr_matrix((np.ones(len(src)), (src, tgt)), shape=(struct.n, struct.n))
        _, comp = connected_components(g, directed=True, connection="strong")
        struct._loop_mask = torch.tensor(comp[src] == comp[tgt], device=struct.src.device)
    return struct._loop_mask


class _PerronRadius(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, struct, opts):
        with torch.no_grad():
            # float64 internally: in float32 the normalised power vector underflows on
            # large unscaled networks (0/0 in the Collatz-Wielandt ratios)
            aw = (w.abs() * loop_edge_mask(struct)).double()
            M = struct.matrix(aw)
            MT = struct.matrix_T(aw)
            v, lo, hi, it_v = _perron_vector(M, struct.n, max_iter=opts["max_iter"], tol=opts["tol"],
                                            dtype=torch.float64, device=w.device)
            u, lo_u, hi_u, it_u = _perron_vector(MT, struct.n, max_iter=opts["max_iter"], tol=opts["tol"],
                                                dtype=torch.float64, device=w.device)
            u, v = u.to(w.dtype), v.to(w.dtype)
        rho = hi
        opts["last"] = dict(rho=rho, lower=lo, upper=hi, iterations=max(it_v, it_u))
        ctx.save_for_backward(w, u, v)
        ctx.struct = struct
        return torch.tensor(rho, dtype=w.dtype, device=w.device)

    @staticmethod
    def backward(ctx, g):
        w, u, v = ctx.saved_tensors
        st = ctx.struct
        denom = (u * v).sum()
        grad = torch.sign(w) * u[st.tgt] * v[st.src] / denom * loop_edge_mask(st)
        return g * grad, None, None


def perron_radius(struct, w, max_iter=2000, tol=1e-10, info=None):
    """Differentiable rho(|A|); pass a dict as `info` for bounds and iterations."""
    opts = info if info is not None else {}
    opts.update(max_iter=max_iter, tol=tol)
    return _PerronRadius.apply(w, struct, opts)


def signed_radius(struct, w):
    """rho(A) of the signed matrix (largest |eigenvalue|, scipy ARPACK; reporting only)."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigs
    st = struct
    A = sp.csr_matrix((w.detach().cpu().double().numpy(),
                       (st.tgt.cpu().numpy(), st.src.cpu().numpy())), shape=(st.n, st.n))
    if st.n < 50:
        return float(np.abs(np.linalg.eigvals(A.toarray())).max())
    vals = eigs(A, k=1, which="LM", return_eigenvectors=False, maxiter=5000, tol=1e-8)
    return float(np.abs(vals).max())


def center_biases(struct, w, U_ref, target=0.25, leak=LEAK):
    """Biases that put every node's steady-state pre-activation at `target` under the
    reference input U_ref (n x 1 or n x k; columns are averaged).

    Exact, no iteration: if every node has Z = target then every node has
    X = f(target), so Z = A (f(target) 1) + b + U = target gives
        b = target - f(target) * (A 1) - U.
    With rho(|A|) < 1 the steady state is unique, so this is the state the solver
    reaches. Returns (b, max |Z - target| after solving, as a check)."""
    with torch.no_grad():
        x0 = mml(torch.tensor(target, dtype=w.dtype), leak).item()
        ones = torch.ones(struct.n, 1, dtype=w.dtype, device=w.device)
        b = target - x0 * (struct.matrix(w) @ ones).squeeze(1) - U_ref.mean(dim=1)
        sol = steady_state(struct, w, b, U_ref.mean(dim=1, keepdim=True), leak=leak, max_iter=5000, tol=1e-10)
        err = (sol.Z[:, 0] - target).abs().max().item()
    return b, err


def simulate(struct, w, b, U, S=None, leak=LEAK, max_iter=300, tol=1e-6,
             adj_max_iter=None, adj_tol=None, clip=None, info=None):
    """Differentiable steady state X(w, b, U, S). Pass a dict as `info` to receive
    convergence diagnostics of the last forward / backward pass."""
    opts = info if info is not None else {}
    opts["forward"] = dict(leak=leak, max_iter=max_iter, tol=tol)
    opts["backward"] = dict(leak=leak, max_iter=adj_max_iter or max_iter, tol=adj_tol or tol, clip=clip)
    return _SteadyState.apply(w, b, U, S, struct, opts)
