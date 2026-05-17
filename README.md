# Pivot-Shifted Carleman Linearisation

## Repository structure

```
.
├── psc_logistic.py             # PSC algorithm — scalar logistic ODE
├── psc_lotka_volterra.py       # PSC algorithm — 2D Lotka-Volterra system
├── plot_results.ipynb          # Parameters, data, and all paper figures
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

