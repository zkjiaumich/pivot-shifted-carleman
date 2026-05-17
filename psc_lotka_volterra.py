"""PSC solvers for  x1' = eta x1(1-x2),  x2' = -gamma x2(1-x1)."""

import numpy as np
from scipy.integrate import solve_ivp


def lv_rhs(t, x, a, g):
    # Original nonlinear vector field (used for reference RK45 solution).
    x1, x2 = x
    return np.array([a*x1 - a*x1*x2, -g*x2 + g*x1*x2])


def F_matrices(a, g, s):
    # Pivot shift: expand RHS around pivot s to get u' = F0 + F1 u + F2(u⊗u).
    sx, sy = s
    F0 = np.array([[a*sx - a*sx*sy], [-g*sy + g*sx*sy]])
    F1 = np.array([[a - a*sy, -a*sx], [g*sy, -g + g*sx]])
    F2 = np.zeros((2, 4)); F2[0, 1] = -a; F2[1, 1] = g
    return F0, F1, F2


def kron_pow(M, k):
    # Repeated Kronecker product M^{⊗k}; base case k=0 returns scalar 1.
    if k == 0: return np.array([[1.0]])
    out = np.array(M, dtype=float)
    for _ in range(k - 1): out = np.kron(out, M)
    return out


def tensor_sum(M, j, n):
    # Sums over all positions where M can sit among j identity factors.
    if j == 1: return M.copy()
    p, d = M.shape
    I = np.eye(n)
    out = np.zeros((p * n**(j-1), d * n**(j-1)))
    for ell in range(j):   # ell = position of M in the tensor product
        out += np.kron(np.kron(kron_pow(I, ell), M), kron_pow(I, j-1-ell))
    return out


def build_carleman_matrix(F0_s, F1_s, F2, N):
    # Assemble the block-structured Carleman matrix A_N and vector b_N.
    # Each block row j handles the ODE for the j-th Kronecker-power block;
    # off-diagonal blocks couple adjacent levels via F0 (down) and F2 (up).
    n      = F1_s.shape[0]
    sizes  = [n**j for j in range(1, N + 1)]   # block sizes: n, n^2, ..., n^N
    starts = np.cumsum([0] + sizes)             
    total  = starts[-1]
    A_N, b_N = np.zeros((total, total)), np.zeros(total)
    for j in range(1, N + 1):
        r0, r1 = starts[j-1], starts[j]
        A_N[r0:r1, r0:r1] = tensor_sum(F1_s, j, n)        
        if j >= 2:
            c0, c1 = starts[j-2], starts[j-1]
            A_N[r0:r1, c0:c1] = tensor_sum(F0_s.reshape(n, 1), j, n)  
        if j <= N - 1:
            c0, c1 = starts[j], starts[j+1]
            A_N[r0:r1, c0:c1] = tensor_sum(F2, j, n)      
    b_N[:n] = F0_s.reshape(-1)  
    return A_N, b_N


def lifted_ic(u0, N):
    # Build the initial condition for the lifted state.
    out, cur = [], u0.copy()
    for j in range(1, N + 1):
        if j == 1: out.append(u0.copy())
        else: cur = np.kron(cur, u0); out.append(cur.copy())
    return np.concatenate(out)


def solve_carleman(x0, t_eval, N, eta, gamma, pivot=None):
    # Full PSC pipeline: shift to pivot, build Carleman system,
    # integrate, then shift solution back to original coordinates.
    # pivot=None reduces to standard Carleman expansion about the origin.
    n  = len(x0)
    s  = np.zeros(n) if pivot is None else np.asarray(pivot, dtype=float)
    F0, F1, F2 = F_matrices(eta, gamma, s)
    A_N, b_N   = build_carleman_matrix(F0, F1, F2, N)
    Y0         = lifted_ic(x0 - s, N)
    sol = solve_ivp(lambda t, Y: A_N @ Y + b_N,
                    (t_eval[0], t_eval[-1]), Y0,
                    t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12)
    return sol.y[:n, :] + s[:, None]   # recover x(t) = u(t) + s
