# Pivot-Shifted Carleman Linearization
This repository implements numerical experiments for **pivot-shifted Carleman
linearization (PSC)** applied to two quadratic ordinary differential equations:
the scalar logistic equation and the two-dimensional Lotka–Volterra system. It
supports the experiments associated with
[arXiv:2605.20071](https://arxiv.org/abs/2605.20071).

Carleman linearisation replaces a nonlinear ODE with an infinite-dimensional
linear system. In computation, this lifted system is truncated at an
order $N$. The standard construction expands around the origin. PSC first
introduces a displaced variable

$$
u=x-s,
$$

where $s$ is a chosen pivot, and then constructs the Carleman system in
$u$. Changing $s$ changes the constant and linear terms of the displaced
ODE and can substantially affect stability and truncation error. This code
compares standard Carleman linearisation ($s=0$) with several nonzero pivots
and truncation orders.

## Repository structure

```
.
├── psc_logistic.py             # PSC algorithm — scalar logistic ODE
├── psc_lotka_volterra.py       # PSC algorithm — 2D Lotka-Volterra system
├── plot_results.ipynb          # Parameters, data, and all paper figures
├── sensitivity_analysis.ipynb  # Pivot sensitivity plots
├── requirements.txt            # pip-installable dependencies
├── environment.yml             # conda environment specification
└── README.md
```

---

## Models

### Logistic equation (1D)

$$\dot{x} = r\,x(1-x), \quad x(0)=x_0, \quad r=1,\; x_0=0.5$$

- Exact solution available in closed form.
- Pivot values compared: $s=0$ (standard Carleman), $s=x_0=0.5$, $s=1.2$.
- Truncation orders $N=4$ and $N=8$; error sweep over $N=1,\ldots,69$.

### Lotka-Volterra system (2D)

$$\dot{x}_1 = \eta x_1 - \eta x_1 x_2, \qquad \dot{x}_2 = -\gamma x_2 + \gamma x_1 x_2$$

with $\eta=1$, $\gamma=0.475$, $x_0=(0.5,\,0.5)^\top$.

- Reference trajectory computed by RK45.
- Pivot values compared: $s=\mathbf{0}$, $s=x_0$, $s=(0.7,\,0.3)^\top$.
- Truncation orders $N=4$ and $N=8$; error sweep over $N=1,\ldots,11$.


---

## How the implementation works

For a quadratic system written after shifting as

$$
\dot u=F_{0,s}+F_{1,s}u+F_{2,s}(u\otimes u),
$$

the lifted state contains successive tensor powers,

$$
Y=(u,\;u^{\otimes 2},\;\ldots,\;u^{\otimes N})^\top.
$$

Differentiating each level gives a block linear system

$$
\dot Y=A_NY+b_N.
$$

The diagonal blocks of \(A_N\) come from \(F_1\), the lower block diagonal
comes from \(F_0\), and the upper block diagonal comes from \(F_2\). Terms
requiring degree \(N+1\) are discarded. Both solver modules integrate this
finite affine system with `scipy.integrate.solve_ivp`.

---

## `psc_logistic.py`

This module handles

$$
\dot x=r x(1-x).
$$

With \(u=x-s\), its displaced form is

$$
\dot u=F_0+F_1u+F_2u^2,
\qquad
F_0=r(s-s^2),\quad F_1=r(1-2s),\quad F_2=-r.
$$

Because the problem is scalar, the lifted variables are simply
$(u,u^2,\ldots,u^N)$, and the truncated matrix has size $N\times N$.

### Functions

- `ps_coeffs(r, s)` computes the three coefficients $(F_0,F_1,F_2)$ after
  shifting by `s`.
- `build_carleman_matrix_scalar(F0, F1, F2, N)` constructs the affine lifted
  system. 
- `solve_carleman_scalar(F0, F1, F2, u0, T, N, Deltat, method="RK45")`
  constructs the lifted initial condition, integrates from `0` to `T`, and returns
  `(t, Y, A, b)`. 
## `psc_lotka_volterra.py`

This module handles

$$
\dot x_1=\eta x_1(1-x_2),\qquad
\dot x_2=-\gamma x_2(1-x_1).
$$

For this two-dimensional problem, level $j$ of the lift has $2^j$
entries. The complete order-\(N\) state therefore has

$$
2+2^2+\cdots+2^N=2^{N+1}-2
$$

entries. Memory and integration cost consequently grow quickly with `N`.

### Functions

- `lv_rhs(t, x, a, g)` evaluates the original nonlinear vector field. The
  notebook uses it with `solve_ivp` to generate a high-accuracy reference
  trajectory. Parameters `a` and `g` correspond to $\eta$ and $\gamma$.
- `F_matrices(a, g, s)` expands the vector field about the two-component pivot
  `s` and returns `F0`, `F1`, and `F2`.
- `kron_pow(M, k)` computes the repeated Kronecker product
  $M^{\otimes k}$; for `k == 0` it returns the scalar identity as a `(1, 1)`
  array.
- `tensor_sum(M, j, n)` forms the sum of all placements of `M` among the `j`
  tensor factors. This implements the product-rule contributions at one lifted
  level.
- `build_carleman_matrix(F0_s, F1_s, F2, N)` assembles the block matrix `A_N`
  and affine vector `b_N` for levels `1` through `N`.
- `lifted_ic(u0, N)` concatenates
  $u_0,u_0^{\otimes2},\ldots,u_0^{\otimes N}$ into the initial lifted state.
- `solve_carleman(x0, t_eval, N, eta, gamma, pivot=None)` performs the full
  workflow: choose the pivot, compute shifted coefficients, assemble and
  integrate the lifted system, and shift the first-level solution back to the
  original coordinates. It returns an array of shape `(2, len(t_eval))`.
  Passing `pivot=None` selects the zero pivot.

---
## Installation

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate psc-carleman
```

### Option B — pip

```bash
pip install -r requirements.txt
```
