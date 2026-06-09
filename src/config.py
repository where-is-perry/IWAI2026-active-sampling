"""Shared configuration: parameters, priors, ground-truth, position grid.

No simulation or plotting code. `true_y` takes `model` as an argument so
the rest of the codebase doesn't have to mutate a global to switch
between the well-specified linear ground truth and the misspecified
sinc / dent ground truths.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Ground-truth parameters for the well-specified quadratic case:
#   y = θ₁ x² + θ₂ x + θ₃.  Matches the values reported in experiments.tex §Setup.
TRUE_THETA = np.array([0.4, -1.0, 0.5])


def true_y(x, model="quadratic"):
    """Ground-truth observation function.

    "quadratic"  — y = θ₁ x² + θ₂ x + θ₃ (well-specified for the filter)
    "sinusoidal" — y = sin(π x / 2 + 1)  (misspecified: smooth, asymmetric;
                    quadratic Taylor is a decent local approximation but the
                    residuals on [-1, 1] exceed the precise sensor's noise)
    "linear"     — y = θ₂ x + θ₃         (drops the quadratic coefficient)
    "sinc"       — y = sinc(x)            (misspecified)
    "dent"       — inverted Ricker wavelet (misspecified)
    """
    if model == "sinusoidal":
        return np.sin(np.pi * x / 2 + 1.0)
    if model == "sinc":
        return np.sinc(x)
    if model == "dent":
        s = 0.3
        return ((x / s) ** 2 - 1.0) * np.exp(-0.5 * (x / s) ** 2)
    if model == "linear":
        return TRUE_THETA[1] * x + TRUE_THETA[2]
    # default = quadratic
    return TRUE_THETA[0] * x**2 + TRUE_THETA[1] * x + TRUE_THETA[2]


def best_fit_theta(model, n_grid=1001):
    """L2-best-fit quadratic θ = (θ₁, θ₂, θ₃) for y_true(x; model) on [a, b].

    Useful when `model` is misspecified (e.g. sinusoidal): the filter can
    only ever reach this projection, so error metrics should measure
    distance from it, not from the unreachable true functional form.

    For "quadratic" the projection is exactly the truth, so we return
    `TRUE_THETA` without doing the lstsq.
    """
    if model == "quadratic":
        return TRUE_THETA.copy()
    grid = np.linspace(A_POS, B_POS, n_grid)
    Y = np.array([true_y(x, model) for x in grid])
    Phi = np.column_stack([grid**2, grid, np.ones_like(grid)])
    theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    return theta


# Feasible position interval [a, b] and discrete action grid.
A_POS, B_POS = -1.0, 1.0
POSITION_GRID = np.arange(A_POS, B_POS + 1e-9, 0.2)   # 11 positions
BOUNDARY_POSITIONS = np.array([A_POS, B_POS])


DEFAULT_PARAMS = dict(
    delta=np.array([1.0, 10.0]),        # measurement time for sensor 0 and sensor 1
    sigma=np.array([1e-2, 1e-3]),       # obs noise variance; ratio σ(0)/σ(1) = Δ[1]/Δ[0] = 10
    xi=15.0,                            # sensor switching cost (Ξ)
    sigma_t=1e-6,                       # time measurement noise variance
    zeta_x=1e-5,                        # tiny position process noise (avoids singular P)
    zeta_t=1e-5,                        # tiny time process noise (avoids singular P)
    travel_cost=1.0,                    # time per unit distance moved
    a_lim=A_POS, b_lim=B_POS,
)

DEFAULT_PRIORS = dict(
    m_theta0=np.zeros(3),               # prior mean for θ = [θ₁, θ₂, θ₃]
    V_theta0=np.eye(3),                 # prior covariance for θ
    m_x0=0.0,                           # scalar position (no bias-dummy now that φ(x) is explicit)
    v_x0=1e-3,                          # tiny prior variance on position (known start)
    m_t0=0.0,
    v_t0=1e-3,                          # tiny prior variance on time (known start)
)


FIGDIR = Path(__file__).resolve().parent.parent / "figures"


def save_fig(fig, name, subdir=None, dpi=150):
    """Save `fig` as a PNG under FIGDIR, optionally in a `subdir/` of it.

    `subdir` groups figures by type (e.g. "problem", "simulation",
    "diagnostics", "beta_sweep") so the figures folder stays organised.
    `dpi` defaults to 150; bump it for figures that need finer detail.
    """
    outdir = FIGDIR / subdir if subdir else FIGDIR
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
