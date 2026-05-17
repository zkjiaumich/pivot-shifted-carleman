"""PSC solvers for  x' = r x(1-x)."""

import numpy as np
from scipy.integrate import solve_ivp


def ps_coeffs(r, s):
    # Pivot shift: rewrite RHS in displaced variable u = x - s,
    return r*(s - s*s), r*(1.0 - 2.0*s), -r


def build_carleman_matrix_scalar(F0, F1, F2, N):
    # Carleman lifting: encode the ODE for each monomial y_j = u^j
    A, b = np.zeros((N, N)), np.zeros(N)
    for j in range(1, N + 1):
        k = j - 1
        A[k, k] = j * F1           
        if j < N: A[k, k + 1] = j * F2   
        if j > 1: A[k, k - 1] = j * F0   
    b[0] = F0   
    return A, b


def solve_carleman_scalar(F0, F1, F2, u0, T, N, Deltat, method="RK45"):
    # Build lifted initial condition y_j(0) = u0^j, then integrate
    # the truncated linear system forward in time.
    A, b = build_carleman_matrix_scalar(F0, F1, F2, N)
    y0   = np.array([u0**j for j in range(1, N + 1)])
    t_ev = np.linspace(0.0, T, int(T / Deltat) + 1)
    sol  = solve_ivp(lambda t, y: A @ y + b, (0.0, T), y0,
                     t_eval=t_ev, method=method, rtol=1e-7, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y, A, b
