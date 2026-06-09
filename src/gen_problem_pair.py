"""Figure: quadratic and sinusoidal problem statements side by side.

Self-contained (does not use plots._panel_problem_sinusoidal) so the sensor-band
styling here doesn't affect other figures.  The precise sensor (sensor 1,
sigma^2=1e-3) has a very narrow noise band, so it gets a stronger fill plus
explicit +/- std envelope lines to make its spread visible next to sensor 0.
"""
import numpy as np
import matplotlib.pyplot as plt
from config import (A_POS, B_POS, POSITION_GRID, DEFAULT_PARAMS, true_y,
                    best_fit_theta)
from plots import _PROBLEM_TITLES, save_fig

S0, S1 = DEFAULT_PARAMS["sigma"]
STD0, STD1 = np.sqrt(S0), np.sqrt(S1)


def draw_bands(ax, xs, ys):
    # sensor 0 — wide band, light fill
    ax.fill_between(xs, ys - STD0, ys + STD0, color="C0", alpha=0.18,
                    label=rf"sensor 0  ($\sigma_0^2={S0:.0e}$)", linewidth=0)
    # sensor 1 — narrow band: stronger fill + envelope lines so it stays visible
    ax.fill_between(xs, ys - STD1, ys + STD1, color="C1", alpha=0.55,
                    label=rf"sensor 1  ($\sigma_1^2={S1:.0e}$)", linewidth=0)
    ax.plot(xs, ys + STD1, color="C1", lw=1.0, alpha=0.9)
    ax.plot(xs, ys - STD1, color="C1", lw=1.0, alpha=0.9)


def panel_quadratic(ax):
    xs = np.linspace(A_POS, B_POS, 401)
    ys = np.array([true_y(x, "quadratic") for x in xs])
    draw_bands(ax, xs, ys)
    ax.plot(xs, ys, color="k", lw=2.0, label=r"true $y(x)$")
    ax.scatter(POSITION_GRID, np.full_like(POSITION_GRID, ax.get_ylim()[0]),
               marker="|", color="k", s=60, clip_on=False, zorder=5)
    ax.set_title(_PROBLEM_TITLES["quadratic"], fontsize=10)
    ax.set_xlabel(r"position $x$")
    ax.set_xlim(A_POS, B_POS)
    ax.set_ylabel(r"observation $y$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)


def panel_sinusoidal(ax):
    xs = np.linspace(A_POS, B_POS, 401)
    ys = np.array([true_y(x, "sinusoidal") for x in xs])
    theta_bf = best_fit_theta("sinusoidal")
    ys_quad = theta_bf[0]*xs**2 + theta_bf[1]*xs + theta_bf[2]
    draw_bands(ax, xs, ys)
    ax.plot(xs, ys, color="k", lw=2.0, label=r"true $y(x) = \sin(\pi x/2 + 1)$")
    ax.plot(xs, ys_quad, color="C3", lw=1.5, ls="--",
            label="best-fit quadratic\n(unreachable target)")
    ax.scatter(POSITION_GRID, np.full_like(POSITION_GRID, ax.get_ylim()[0]),
               marker="|", color="k", s=60, clip_on=False, zorder=5)
    ax.set_title(_PROBLEM_TITLES["sinusoidal"], fontsize=10)
    ax.set_xlabel(r"position $x$")
    ax.set_xlim(A_POS, B_POS)
    ax.set_ylabel(r"observation $y$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", fontsize=7, framealpha=0.9)


fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
panel_quadratic(axes[0])
panel_sinusoidal(axes[1])
fig.tight_layout()
save_fig(fig, "problem_quadratic_sinusoidal")
print("saved problem_quadratic_sinusoidal.png", flush=True)
