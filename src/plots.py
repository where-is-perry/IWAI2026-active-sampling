"""Figure-producing functions.

Each plot function takes pre-computed runner outputs (or just the
configuration it needs) and returns nothing — it writes a PNG via
`config.save_fig`. No simulation calls live in this module.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    A_POS, B_POS, POSITION_GRID,
    DEFAULT_PARAMS, TRUE_THETA, true_y,
    FIGDIR, save_fig,
)


# ---------------------------------------------------------------------------
# Style — kept here because it's a plotting concern, not a sim concern.
# ---------------------------------------------------------------------------

COLORS = {
    "Sensor 0 only": "C0",
    "Sensor 1 only": "C1",
    "EFE":           "C3",
    "Greedy":        "C2",
    "Planner H=1":   "C3",
    "Planner H=3":   "C4",
    "Planner H=5":   "C5",
    "Planner H=6":   "C6",
    "Planner H=10":  "C8",
    "Random":        "C7",
    r"Greedy $\beta=0$": "C9",
}

LINESTYLES = {
    "Sensor 0 only": "--",
    "Sensor 1 only": "--",
    "EFE":           "-",
    "Greedy":        "-",
    "Planner H=1":   "-",
    "Planner H=3":   "-",
    "Planner H=5":   "-",
    "Planner H=6":   "-",
    "Planner H=10":  "-",
    "Random":        "--",
    r"Greedy $\beta=0$": "-",
}

MARKERS = {
    "Sensor 0 only": "s",
    "Sensor 1 only": "D",
    "EFE":           "x",
    "Greedy":        "o",
    "Planner H=1":   "*",
    "Planner H=3":   "v",
    "Planner H=5":   "^",
    "Planner H=6":   "P",
    "Planner H=10":  "P",
    "Random":        "o",
}


# ---------------------------------------------------------------------------
# Aggregation helpers (multi-seed → curves on a common time grid).
# ---------------------------------------------------------------------------

def _error_stack(runs, t_grid, target_theta, metric="theta", model="quadratic"):
    """(N_seeds, len(t_grid)) array of per-seed error curves on t_grid."""
    from runners import error_series  # local import to avoid circular dep
    return np.array([
        np.interp(t_grid, r["time"],
                  error_series(r, target_theta, metric=metric, model=model))
        for r in runs
    ])


def _step_interp(t_grid, time, vals):
    """Previous-value (step) interpolation of a discrete series onto t_grid.

    For a piecewise-constant signal (e.g. the active sensor identity), linear
    np.interp would fabricate fractional values between switch points; this
    holds the last observed value instead.
    """
    time = np.asarray(time, dtype=float)
    vals = np.asarray(vals, dtype=float)
    idx = np.clip(np.searchsorted(time, t_grid, side="right") - 1, 0, len(vals) - 1)
    return vals[idx]


def _posterior_diagnostics(runs, target_theta, t_grid):
    """Per-seed (log q(θ*), entropy H[q(θ)], Mahalanobis²) curves on t_grid.

    Returns three (N_seeds, len(t_grid)) arrays.
    """
    d = len(target_theta)
    lpd_stack, ent_stack, mah_stack = [], [], []
    for r in runs:
        lpd, ent, mah = [], [], []
        for mth, V in zip(zip(r["theta1"], r["theta2"], r["theta3"]), r["V_theta"]):
            V = np.asarray(V)
            _, logdet = np.linalg.slogdet(V)
            diff = np.array(mth) - target_theta
            m2 = diff @ np.linalg.solve(V, diff)
            lpd.append(-0.5*d*np.log(2*np.pi) - 0.5*logdet - 0.5*m2)
            ent.append(0.5*d*np.log(2*np.pi*np.e) + 0.5*logdet)
            mah.append(m2)
        lpd_stack.append(np.interp(t_grid, r["time"], lpd))
        ent_stack.append(np.interp(t_grid, r["time"], ent))
        mah_stack.append(np.interp(t_grid, r["time"], mah))
    return np.array(lpd_stack), np.array(ent_stack), np.array(mah_stack)


# ---------------------------------------------------------------------------
# Main simulation figure (5 panels) + diagnostics figure (3 panels).
# ---------------------------------------------------------------------------

def plot_simulation(runs_by_name, summary_rows, t_grid, *, T, model, objective,
                    tag="", n_seeds=10, target_theta=None, metric="theta"):
    """Six-panel trajectory figure.

    Panels: θ̂₁, θ̂₂, θ̂₃ (median ± IQR over all seeds), sensor, position
    (single representative seed — a median trajectory washes out the
    boundary-hopping), error + tr V (median, with IQR band on the error).

    `target_theta` is the reference for the parameter-space axhlines and (when
    metric='theta') the error band.  `metric='output'` switches the error
    band to mean-squared output error against true_y(x; model).
    """
    from runners import error_series  # local import to avoid circular dep
    if target_theta is None:
        target_theta = TRUE_THETA
    fig, axes = plt.subplots(6, 1, figsize=(7, 9.2), sharex=True)
    ax_theta1, ax_theta2, ax_theta3, ax_sensor, ax_pos, ax_err = axes
    ax_post = ax_err.twinx()

    def _interp(time, vals):
        return np.interp(t_grid, time, vals)

    for name, runs in runs_by_name.items():
        r0 = runs[0]
        time = r0["time"]

        # θ̂ panels: median ± IQR over all seeds.
        for ax, key in zip((ax_theta1, ax_theta2, ax_theta3),
                            ("theta1", "theta2", "theta3")):
            stack = np.array([_interp(r["time"], r[key]) for r in runs])
            med = np.median(stack, axis=0)
            q1, q3 = np.quantile(stack, [0.25, 0.75], axis=0)
            ax.plot(t_grid, med, color=COLORS[name], ls=LINESTYLES[name],
                    lw=1.8, label=name)
            ax.fill_between(t_grid, q1, q3, color=COLORS[name], alpha=0.15,
                            linewidth=0)

        # sensor / position: a single representative seed (seed 0).  A median
        # over seeds would hide the boundary-hopping every run does.
        for ax, key in zip((ax_sensor, ax_pos), ("sensor", "position")):
            ax.plot(t_grid, _interp(time, r0[key]),
                    color=COLORS[name], ls=LINESTYLES[name], lw=1.8, label=name)
            ax.plot(time, r0[key], 'x', color=COLORS[name], markersize=5, alpha=0.4)

        # Multi-seed bands for error + posterior trace.
        err_stack = np.array([
            _interp(r["time"], error_series(r, target_theta, metric=metric, model=model))
            for r in runs
        ])
        post_stack = np.array([_interp(r["time"],
                               [np.trace(V) for V in r["V_theta"]]) for r in runs])
        err_med = np.median(err_stack, axis=0)
        err_q1, err_q3 = np.quantile(err_stack, [0.25, 0.75], axis=0)
        post_med = np.median(post_stack, axis=0)

        ax_err.plot(t_grid, err_med, color=COLORS[name], ls=LINESTYLES[name],
                    lw=1.8, label=f"{name} (n={n_seeds})")
        ax_err.fill_between(t_grid, err_q1, err_q3,
                            color=COLORS[name], alpha=0.15, linewidth=0)
        ax_post.plot(t_grid, post_med, color=COLORS[name], ls=":", lw=1.0, alpha=0.9)

    # axhlines show the *target* θ — either the true quadratic or the
    # best-fit projection if the truth lies outside the quadratic family.
    is_target_truth = np.allclose(target_theta, TRUE_THETA)
    target_label = "True" if is_target_truth else "Best-fit"
    ax_theta1.axhline(target_theta[0], color="k", ls=":", lw=1, alpha=0.5,
                      label=rf"{target_label} $\theta_1={target_theta[0]:.3f}$")
    ax_theta2.axhline(target_theta[1], color="k", ls=":", lw=1, alpha=0.5,
                      label=rf"{target_label} $\theta_2={target_theta[1]:.3f}$")
    ax_theta3.axhline(target_theta[2], color="k", ls=":", lw=1, alpha=0.5,
                      label=rf"{target_label} $\theta_3={target_theta[2]:.3f}$")

    ax_theta1.set_ylabel(r"$\hat{\theta}_1$")
    ax_theta1.set_title(f"Simulation  ($T={T}$, true model: {model}, objective: {objective})")
    ax_theta1.legend(fontsize=7); ax_theta1.grid(True, alpha=0.3)
    ax_theta2.set_ylabel(r"$\hat{\theta}_2$"); ax_theta2.legend(fontsize=7); ax_theta2.grid(True, alpha=0.3)
    ax_theta3.set_ylabel(r"$\hat{\theta}_3$"); ax_theta3.legend(fontsize=7); ax_theta3.grid(True, alpha=0.3)
    ax_sensor.set_ylabel(r"sensor choice"); ax_sensor.legend(fontsize=7); ax_sensor.grid(True, alpha=0.3)
    ax_pos.set_ylabel(r"position"); ax_pos.legend(fontsize=7); ax_pos.grid(True, alpha=0.3)
    err_ylabel = (r"$\|\hat\theta - \theta^\star\|^2$" if metric == "theta"
                  else r"$\mathbb{E}_x[(\hat y(x) - y(x))^2]$")
    ax_err.set_ylabel(err_ylabel); ax_err.set_yscale("log")
    ax_err.set_xlabel("Elapsed time")
    ax_post.set_ylabel(r"$\mathrm{tr}\,V_\theta$"); ax_post.set_yscale("log")

    h1, l1 = ax_err.get_legend_handles_labels()
    h2, l2 = ax_post.get_legend_handles_labels()
    ax_err.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2)
    ax_err.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    save_fig(fig, f"simulation{tag}_{model}_{objective}", subdir="simulation")


def plot_diagnostics(runs_by_name, t_grid, *, T, model, objective, tag="",
                     target_theta=None):
    """Three-panel probabilistic diagnostics (mean ± SE over all seeds):
    log-pred-density, entropy, Mahalanobis².

    The Mahalanobis² panel uses the *mean*, so the calibration target
    E[Mah²]=d is the right reference — the median of χ²_d sits below d even
    when the posterior is perfectly calibrated.
    """
    if target_theta is None:
        target_theta = TRUE_THETA
    fig, (ax_lpd, ax_ent, ax_mah) = plt.subplots(3, 1, figsize=(7, 6.0), sharex=True)
    d = len(target_theta)
    n_seeds = len(next(iter(runs_by_name.values()), []))

    for name, runs in runs_by_name.items():
        lpd_stack, ent_stack, mah_stack = _posterior_diagnostics(
            runs, target_theta, t_grid)

        n = len(runs)
        for ax, stack, logy in ((ax_lpd, lpd_stack, False),
                                (ax_ent, ent_stack, False),
                                (ax_mah, mah_stack, True)):
            mean = stack.mean(axis=0)
            se = (stack.std(axis=0, ddof=1) / np.sqrt(n)
                  if n > 1 else np.zeros_like(mean))
            lo, hi = mean - se, mean + se
            if logy:
                lo = np.clip(lo, 1e-9, None)   # keep the band valid on a log axis
            ax.plot(t_grid, mean, color=COLORS[name], ls=LINESTYLES[name],
                    lw=1.8, label=name)
            ax.fill_between(t_grid, lo, hi, color=COLORS[name], alpha=0.15,
                            linewidth=0)

    ax_mah.axhline(d, color="k", ls=":", lw=1, alpha=0.6,
                   label=f"calibrated value $=d={d}$")

    ax_lpd.set_ylabel(r"$\log q(\theta^\star)$" "\n(higher = better)")
    ax_lpd.set_title(f"Probabilistic diagnostics  ($T={T}$, true model: {model}, "
                     f"objective: {objective}, n={n_seeds})")
    ax_lpd.legend(fontsize=7); ax_lpd.grid(True, alpha=0.3)
    ax_ent.set_ylabel(r"$H[q(\theta)]$ (nats)" "\n(lower = better)")
    ax_ent.legend(fontsize=7); ax_ent.grid(True, alpha=0.3)
    ax_mah.set_ylabel(r"Mahalanobis$^2$" "\n($\\approx d$ if calibrated)")
    ax_mah.set_yscale("log")
    ax_mah.set_xlabel("Elapsed time")
    ax_mah.legend(fontsize=7); ax_mah.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    save_fig(fig, f"simulation{tag}_diagnostics_{model}_{objective}", subdir="diagnostics")


# ---------------------------------------------------------------------------
# Horizon-contrast figure — makes the H=1-vs-H=3 lookahead gap explicit.
# ---------------------------------------------------------------------------

def plot_horizon_contrast(runs_by_name, summary_rows, t_grid, *, T, model,
                          objective, threshold=1e-2, target_theta=None,
                          n_seeds=10, tag="_horizon_contrast", regime=None):
    """Two-panel H=1-vs-H=3 contrast built to make the lookahead effect legible.

    Unlike `plot_simulation` (single-seed trajectories that overlap), this
    aggregates over all seeds: median ± IQR error (top) and posterior trace
    (bottom), the convergence threshold η, the gap between the two median
    error curves shaded, and a summary box with reached-fraction / final-error.

    `regime`: optional dict with keys {"beta": float, "params": dict}.  When
    provided, a small settings annotation is drawn listing β, Ξ and σ for the
    regime.
    """
    from runners import error_series
    if target_theta is None:
        target_theta = TRUE_THETA
    rows_by_name = {row["strategy"]: row for row in summary_rows}

    fig, (ax_err, ax_post) = plt.subplots(2, 1, figsize=(7, 6.4), sharex=True)

    err_med = {}
    for name, runs in runs_by_name.items():
        err_stack = np.array([
            np.interp(t_grid, r["time"],
                      error_series(r, target_theta, metric="theta", model=model))
            for r in runs
        ])
        post_stack = np.array([
            np.interp(t_grid, r["time"], [np.trace(V) for V in r["V_theta"]])
            for r in runs
        ])
        e_med = np.median(err_stack, axis=0)
        e_q1, e_q3 = np.quantile(err_stack, [0.25, 0.75], axis=0)
        p_med = np.median(post_stack, axis=0)
        p_q1, p_q3 = np.quantile(post_stack, [0.25, 0.75], axis=0)
        err_med[name] = e_med

        c = COLORS[name]
        ax_err.plot(t_grid, e_med, color=c, ls=LINESTYLES.get(name, "-"),
                    lw=2.0, label=name)
        ax_err.fill_between(t_grid, e_q1, e_q3, color=c, alpha=0.15, linewidth=0)
        ax_post.plot(t_grid, p_med, color=c, ls=LINESTYLES.get(name, "-"),
                     lw=2.0, label=name)
        ax_post.fill_between(t_grid, p_q1, p_q3, color=c, alpha=0.15, linewidth=0)

        row = rows_by_name.get(name)
        if row is not None and not np.isnan(row["ttt_median"]):
            ax_err.axvline(row["ttt_median"], color=c, ls="--", lw=1.0, alpha=0.7)

    # Shade the gap between the two horizons' median error curves.
    if "Planner H=1" in err_med and "Planner H=3" in err_med:
        e1, e3 = err_med["Planner H=1"], err_med["Planner H=3"]
        ax_err.fill_between(t_grid, np.minimum(e1, e3), np.maximum(e1, e3),
                            where=(e1 != e3), color="gold", alpha=0.35,
                            linewidth=0, label="H=1 vs H=3 gap")

    ax_err.axhline(threshold, color="k", ls=":", lw=1.0, alpha=0.6,
                   label=rf"threshold $\eta={threshold:g}$")
    ax_err.set_yscale("log")
    ax_err.set_ylabel(r"$\|\hat\theta - \theta^\star\|^2$")
    ax_err.set_title(f"Horizon contrast: H=1 vs H=3  "
                     f"($T={T}$, {model}, {objective}, n={n_seeds})")
    ax_err.grid(True, alpha=0.3, which="both")
    ax_err.legend(fontsize=8, loc="upper right", ncol=2)

    ax_post.set_yscale("log")
    ax_post.set_ylabel(r"$\mathrm{tr}\,V_\theta$")
    ax_post.set_xlabel("Elapsed time")
    ax_post.grid(True, alpha=0.3, which="both")
    ax_post.legend(fontsize=8, loc="upper right")

    # Summary box of the gap (uses the same aggregated rows as the .tex table).
    r1, r3 = rows_by_name.get("Planner H=1"), rows_by_name.get("Planner H=3")
    if r1 and r3:
        fe1, fe3 = r1["final_err_median"], r3["final_err_median"]
        ratio = fe1 / fe3 if fe3 > 0 else float("nan")
        txt = (
            "H=3 vs H=1\n"
            f"reached:   {100*r3['reached_frac']:.0f}%  vs  {100*r1['reached_frac']:.0f}%\n"
            f"final err: {fe3:.2e}  vs  {fe1:.2e}  ({ratio:.1f}× lower)\n"
            f"t$\\to$<{threshold:g}: {r3['ttt_median']:.1f}  vs  {r1['ttt_median']:.1f}"
        )
        ax_err.text(0.02, 0.04, txt, transform=ax_err.transAxes, fontsize=7.5,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    # Settings annotation — surfaces the parameter values for this regime.
    if regime is not None:
        p = regime.get("params", {})
        beta_val = regime.get("beta", 0.0)
        sig_v = p.get("sigma", DEFAULT_PARAMS["sigma"])
        xi_v = p.get("xi", DEFAULT_PARAMS["xi"])
        lines = [
            "Settings:",
            f"  β = {beta_val:g}",
            f"  Ξ = {xi_v:g}",
            f"  σ = [{sig_v[0]:.0e}, {sig_v[1]:.0e}]",
        ]
        ax_err.text(0.98, 0.04, "\n".join(lines),
                    transform=ax_err.transAxes, fontsize=7, va="bottom", ha="right",
                    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    fig.tight_layout()
    save_fig(fig, f"horizon_contrast{tag.replace('_horizon_contrast', '')}_H1_vs_H3_"
                  f"{model}_{objective}", subdir="simulation")


# ---------------------------------------------------------------------------
# Paper-ready figures: compact result, sensor schedule, paired horizon.
# ---------------------------------------------------------------------------

def plot_result(runs_by_name, summary_rows, t_grid, *, T, model, objective,
                tag="", n_seeds=10, target_theta=None, metric="theta",
                threshold=1e-2):
    """Compact headline result (column-width, one panel).

    Median ± IQR of the error metric per strategy (log y), the convergence
    threshold η, a dashed tick per strategy at its median time-to-threshold;
    twin axis shows median tr V_θ (dotted).  The in-text replacement for the
    dense 6-panel `plot_simulation`.
    """
    if target_theta is None:
        target_theta = TRUE_THETA
    rows_by_name = {row["strategy"]: row for row in summary_rows}

    fig, ax_err = plt.subplots(1, 1, figsize=(5.0, 3.4))
    ax_post = ax_err.twinx()

    for name, runs in runs_by_name.items():
        err_stack = _error_stack(runs, t_grid, target_theta, metric=metric, model=model)
        post_stack = np.array([
            np.interp(t_grid, r["time"], [np.trace(V) for V in r["V_theta"]])
            for r in runs
        ])
        e_med = np.median(err_stack, axis=0)
        e_q1, e_q3 = np.quantile(err_stack, [0.25, 0.75], axis=0)
        p_med = np.median(post_stack, axis=0)

        c = COLORS[name]
        ax_err.plot(t_grid, e_med, color=c, ls=LINESTYLES.get(name, "-"),
                    lw=1.8, label=name)
        ax_err.fill_between(t_grid, e_q1, e_q3, color=c, alpha=0.15, linewidth=0)
        ax_post.plot(t_grid, p_med, color=c, ls=":", lw=1.0, alpha=0.8)

        row = rows_by_name.get(name)
        if row is not None and not np.isnan(row["ttt_median"]):
            ax_err.axvline(row["ttt_median"], color=c, ls="--", lw=0.9, alpha=0.6)

    ax_err.axhline(threshold, color="k", ls=":", lw=1.0, alpha=0.6,
                   label=rf"$\eta={threshold:g}$")
    err_tex = (r"\|\hat\theta-\theta^\star\|^2" if metric == "theta"
               else r"\mathbb{E}_x[(\hat y(x)-y(x))^2]")
    ax_err.set_yscale("log")
    ax_err.set_ylabel(rf"${err_tex}$")
    ax_err.set_xlabel("Elapsed time")
    ax_err.set_title(f"Result  ($T={T}$, {model}, {objective}, n={n_seeds})",
                     fontsize=9)
    ax_err.grid(True, alpha=0.3, which="both")
    ax_err.legend(fontsize=7, loc="upper right")
    ax_post.set_ylabel(r"$\mathrm{tr}\,V_\theta$ (dotted)")
    ax_post.set_yscale("log")

    fig.tight_layout()
    save_fig(fig, f"result{tag}_{model}_{objective}", subdir="simulation")


def plot_sensor_schedule(runs_by_name, t_grid, *, T, model, objective,
                         tag="", n_seeds=10):
    """Fraction of seeds on the precise sensor (l=1) over time, per strategy.

    Visualises the multi-modal switching policy: step-interpolate each seed's
    active-sensor identity, then average across seeds.  Single-sensor baselines
    appear as flat references at 0 / 1.  Legend reports mean #switches per run.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.0))

    for name, runs in runs_by_name.items():
        frac = np.mean([_step_interp(t_grid, r["time"], r["sensor"]) for r in runs],
                       axis=0)
        n_sw = np.mean([int(np.sum(np.abs(np.diff(r["sensor"])) > 0)) for r in runs])
        ax.plot(t_grid, frac, color=COLORS[name], ls=LINESTYLES.get(name, "-"),
                lw=1.8, label=f"{name}  ({n_sw:.1f} switches)")

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("fraction on precise sensor (l=1)")
    ax.set_xlabel("Elapsed time")
    ax.set_title(f"Sensor schedule  ($T={T}$, {model}, {objective}, n={n_seeds})",
                 fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    save_fig(fig, f"sensor_schedule{tag}_{model}_{objective}", subdir="simulation")


def plot_horizon_paired(runs_by_name, *, T, model, objective,
                        target_theta=None, metric="theta"):
    """Paired per-seed H=1-vs-H=3 comparison (same seeds run both horizons).

    Left: log-log scatter of final error, H=1 (x) vs H=3 (y), with the y=x
    line — points below are seeds where H=3 wins.  Right: histogram of the
    per-seed log-ratio log(err_H1 / err_H3), annotated with win-rate, median
    ratio and a numpy bootstrap 95% CI on the median ratio.
    """
    if target_theta is None:
        target_theta = TRUE_THETA
    if "Planner H=1" not in runs_by_name or "Planner H=3" not in runs_by_name:
        return

    from runners import error_series
    def final_err(r):
        return error_series(r, target_theta, metric=metric, model=model)[-1]
    e1 = np.array([final_err(r) for r in runs_by_name["Planner H=1"]])
    e3 = np.array([final_err(r) for r in runs_by_name["Planner H=3"]])
    n = min(len(e1), len(e3))
    e1, e3 = e1[:n], e3[:n]

    eps = 1e-12
    log_ratio = np.log((e1 + eps) / (e3 + eps))    # >0 ⇒ H=3 better
    win_rate = float(np.mean(e3 <= e1))
    med_ratio = float(np.median(np.exp(log_ratio)))
    rng = np.random.default_rng(0)
    boots = [np.median(np.exp(log_ratio[rng.integers(0, n, n)])) for _ in range(2000)]
    ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])

    fig, (ax_s, ax_h) = plt.subplots(1, 2, figsize=(7.5, 3.4))

    lim_lo, lim_hi = min(e1.min(), e3.min()) * 0.7, max(e1.max(), e3.max()) * 1.4
    ax_s.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="k", ls=":", lw=1,
              alpha=0.6, label="y = x")
    ax_s.scatter(e1, e3, s=22, color=COLORS["Planner H=3"], alpha=0.7,
                 edgecolor="none")
    ax_s.set_xscale("log"); ax_s.set_yscale("log")
    ax_s.set_xlim(lim_lo, lim_hi); ax_s.set_ylim(lim_lo, lim_hi)
    ax_s.set_xlabel(r"final $\|\hat\theta-\theta^\star\|^2$ — H=1")
    ax_s.set_ylabel(r"final — H=3")
    ax_s.set_title(f"Paired final error (n={n})", fontsize=9)
    ax_s.grid(True, alpha=0.3, which="both")
    ax_s.legend(fontsize=7, loc="upper left")

    ax_h.hist(log_ratio, bins=min(20, max(6, n // 3)),
              color=COLORS["Planner H=3"], alpha=0.7)
    ax_h.axvline(0.0, color="k", ls=":", lw=1, alpha=0.6)
    ax_h.set_xlabel(r"$\log(\mathrm{err}_{H1} / \mathrm{err}_{H3})$  ($>0$: H=3 better)")
    ax_h.set_ylabel("seeds")
    ax_h.set_title("Per-seed advantage", fontsize=9)
    ax_h.grid(True, alpha=0.3)
    txt = (f"H=3 win-rate: {100*win_rate:.0f}%\n"
           f"median ratio: {med_ratio:.2f}×\n"
           f"95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
    ax_h.text(0.03, 0.97, txt, transform=ax_h.transAxes, fontsize=7.5,
              va="top", ha="left",
              bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    fig.suptitle(f"H=1 vs H=3 paired  ($T={T}$, {model}, {objective})", fontsize=10)
    fig.tight_layout()
    save_fig(fig, f"horizon_paired_{model}_{objective}", subdir="simulation")


def plot_misspec_floor(runs_by_name, t_grid, *, T, model, objective, n_seeds=10):
    """Output-error convergence to the best-fit-quadratic floor (misspecified truth).

    Median ± IQR of E_x[(ŷ(x) − y_true(x))²] per strategy, plus a horizontal line
    at the irreducible best-fit-quadratic residual — the floor no θ can beat.
    """
    from config import best_fit_theta
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2))

    for name, runs in runs_by_name.items():
        stack = _error_stack(runs, t_grid, None, metric="output", model=model)
        med = np.median(stack, axis=0)
        q1, q3 = np.quantile(stack, [0.25, 0.75], axis=0)
        ax.plot(t_grid, med, color=COLORS[name], ls=LINESTYLES.get(name, "-"),
                lw=1.8, label=name)
        ax.fill_between(t_grid, q1, q3, color=COLORS[name], alpha=0.15, linewidth=0)

    grid = np.linspace(A_POS, B_POS, 401)
    theta_bf = best_fit_theta(model)
    y_bf = theta_bf[0]*grid**2 + theta_bf[1]*grid + theta_bf[2]
    y_true = np.array([true_y(x, model) for x in grid])
    floor = float(np.mean((y_bf - y_true)**2))
    ax.axhline(floor, color="k", ls="--", lw=1.2, alpha=0.7,
               label=f"best-fit floor = {floor:.2e}")

    ax.set_yscale("log")
    ax.set_ylabel(r"$\mathbb{E}_x[(\hat y(x) - y(x))^2]$")
    ax.set_xlabel("Elapsed time")
    ax.set_title(f"Misspecification floor  ($T={T}$, {model}, {objective}, n={n_seeds})",
                 fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    save_fig(fig, f"misspec_floor_{model}_{objective}", subdir="simulation")


def plot_efe_mechanism(runs_by_name, t_grid, *, T, model, objective, tag="",
                       n_seeds=10):
    """Mechanism view of the objective driving behaviour.

    Top: median per-step EIG (the planning MI the planner maximises) over time,
    per planner strategy.  Bottom: chosen position over time (single seed)
    coloured by active sensor.  Planner-only (strategies without 'eig' skipped).
    """
    planner = {name: runs for name, runs in runs_by_name.items()
               if runs and "eig" in runs[0]}
    if not planner:
        return

    fig, (ax_eig, ax_pol) = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True)

    for name, runs in planner.items():
        eig_stack = np.array([np.interp(t_grid, r["time"], r["eig"]) for r in runs])
        ax_eig.plot(t_grid, np.median(eig_stack, axis=0),
                    color=COLORS[name], ls=LINESTYLES.get(name, "-"),
                    lw=1.8, label=name)

    name0 = "Planner H=3" if "Planner H=3" in planner else next(iter(planner))
    r0 = planner[name0][0]
    ax_pol.plot(r0["time"], r0["position"], color="0.6", lw=0.8, alpha=0.6, zorder=2)
    sc = ax_pol.scatter(r0["time"], r0["position"], c=r0["sensor"],
                        cmap="coolwarm", vmin=0, vmax=1, s=28, zorder=3)
    cbar = fig.colorbar(sc, ax=ax_pol, ticks=[0, 1], pad=0.01)
    cbar.set_label("active sensor", fontsize=8)

    ax_eig.set_ylabel("per-step EIG\n(planning MI)")
    ax_eig.set_title(f"EFE mechanism  ($T={T}$, {model}, {objective}, n={n_seeds})",
                     fontsize=9)
    ax_eig.grid(True, alpha=0.3); ax_eig.legend(fontsize=7, loc="best")
    ax_pol.set_ylabel(f"position\n({name0}, seed 0)")
    ax_pol.set_xlabel("Elapsed time")
    ax_pol.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, f"efe_mechanism{tag}_{model}_{objective}", subdir="diagnostics")


def plot_calibration(runs_by_name, t_grid, *, T, model, objective, tag="",
                     n_seeds=10, target_theta=None):
    """Single-panel calibration appendix figure: Mahalanobis² mean ± SE per
    strategy vs time, with the d reference line (E[Mah²]=d when calibrated)."""
    if target_theta is None:
        target_theta = TRUE_THETA
    d = len(target_theta)
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2))

    for name, runs in runs_by_name.items():
        _, _, mah_stack = _posterior_diagnostics(runs, target_theta, t_grid)
        n = len(runs)
        mean = mah_stack.mean(axis=0)
        se = (mah_stack.std(axis=0, ddof=1) / np.sqrt(n)
              if n > 1 else np.zeros_like(mean))
        ax.plot(t_grid, mean, color=COLORS[name], ls=LINESTYLES.get(name, "-"),
                lw=1.8, label=name)
        ax.fill_between(t_grid, np.clip(mean - se, 1e-9, None), mean + se,
                        color=COLORS[name], alpha=0.15, linewidth=0)

    ax.axhline(d, color="k", ls=":", lw=1.2, alpha=0.7,
               label=f"calibrated $= d = {d}$")
    ax.set_yscale("log")
    ax.set_ylabel(r"Mahalanobis$^2$ (mean ± SE)")
    ax.set_xlabel("Elapsed time")
    ax.set_title(f"Calibration  ($T={T}$, {model}, {objective}, n={n_seeds})",
                 fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    save_fig(fig, f"calibration{tag}_{model}_{objective}", subdir="diagnostics")


# ---------------------------------------------------------------------------
# Compact "report" figures (landscape, ~½ page) — share panel helpers below.
# Old full-panel figures above are kept as the supplementary versions.
# ---------------------------------------------------------------------------

def _panel_convergence(ax, runs_by_name, t_grid, target_theta, *, metric="theta",
                       model="quadratic", threshold=1e-2, summary_rows=None):
    """Median ± IQR error per strategy on a log axis, with the η threshold line
    and (if summary_rows given) a faint vertical at each strategy's median TTT."""
    rows_by_name = {r["strategy"]: r for r in summary_rows} if summary_rows else {}
    for name, runs in runs_by_name.items():
        stack = _error_stack(runs, t_grid, target_theta, metric=metric, model=model)
        med = np.median(stack, axis=0)
        q1, q3 = np.quantile(stack, [0.25, 0.75], axis=0)
        ax.plot(t_grid, med, color=COLORS[name], ls=LINESTYLES.get(name, "-"),
                lw=1.6, label=name)
        ax.fill_between(t_grid, q1, q3, color=COLORS[name], alpha=0.15, linewidth=0)
        row = rows_by_name.get(name)
        if row is not None and not np.isnan(row["ttt_median"]):
            ax.axvline(row["ttt_median"], color=COLORS[name], ls="--", lw=0.8, alpha=0.5)
    ax.axhline(threshold, color="k", ls=":", lw=1.0, alpha=0.6)
    err_tex = (r"\|\hat\theta-\theta^\star\|^2" if metric == "theta"
               else r"\mathbb{E}_x[(\hat y(x)-y(x))^2]")
    ax.set_yscale("log")
    ax.set_ylabel(rf"${err_tex}$")
    ax.set_xlabel("elapsed time")
    ax.grid(True, alpha=0.3, which="both")


def _panel_sensor_fraction(ax, runs_by_name, t_grid):
    """Fraction of seeds on the precise sensor (l=1) over time, per strategy."""
    for name, runs in runs_by_name.items():
        frac = np.mean([_step_interp(t_grid, r["time"], r["sensor"]) for r in runs],
                       axis=0)
        ax.plot(t_grid, frac, color=COLORS[name], ls=LINESTYLES.get(name, "-"),
                lw=1.6, label=name)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("frac. on precise sensor")
    ax.set_xlabel("elapsed time")
    ax.grid(True, alpha=0.3)


def _panel_position(ax, runs_by_name, t_grid):
    """Single representative seed's position trajectory per strategy."""
    for name, runs in runs_by_name.items():
        r0 = runs[0]
        ax.plot(t_grid, np.interp(t_grid, r0["time"], r0["position"]),
                color=COLORS[name], ls=LINESTYLES.get(name, "-"), lw=1.4, label=name)
    ax.set_ylabel("position (seed 0)")
    ax.set_xlabel("elapsed time")
    ax.grid(True, alpha=0.3)


def _format_regime_inline(regime):
    """One-line settings string for the report figure header: β, Ξ, σ."""
    p = regime.get("params", {})
    beta_val = regime.get("beta", 0.0)
    sig_v = p.get("sigma", DEFAULT_PARAMS["sigma"])
    xi_v = p.get("xi", DEFAULT_PARAMS["xi"])
    return (rf"$\beta={beta_val:g}$    "
            # rf"$\Xi={xi_v:g}$    "
            # rf"$\sigma=[{sig_v[0]:.0e},\,{sig_v[1]:.0e}]$"
            )


def _shared_top_legend(fig, ax, suptitle, settings_line=None):
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(suptitle, fontsize=9, y=0.99)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.95))
    if settings_line:
        fig.text(0.5, 0.89, settings_line, ha="center", va="top",
                 fontsize=7, color="0.3")
        fig.tight_layout(rect=[0, 0, 1, 0.82])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.86])


def plot_horizon_report(runs_by_name, summary_rows, t_grid, *, T, beta, model, objective,
                        threshold=1e-2, target_theta=None, n_seeds=10,
                        include_position=True, regime=None, tag="", subdir="report"):
    """Compact landscape horizon figure: sensor choice, convergence, [position].
    H=1 vs H=3 in the contrast regime.

    `regime`: optional dict {"beta": float, "params": dict}.  When provided, a
    small settings line (β, Ξ, σ) is drawn under the legend so the reader can
    see which parameter values define the contrast regime."""
    if target_theta is None:
        target_theta = TRUE_THETA
    ncol = 3 if include_position else 2
    figsize = (10.0, 3.2) if include_position else (7.5, 3.4)
    fig, axes = plt.subplots(1, ncol, figsize=figsize)

    _panel_sensor_fraction(axes[0], runs_by_name, t_grid)
    axes[0].set_title("sensor choice", fontsize=9)
    _panel_convergence(axes[1], runs_by_name, t_grid, target_theta,
                       model=model, threshold=threshold, summary_rows=summary_rows)
    axes[1].set_title("convergence", fontsize=9)
    if include_position:
        _panel_position(axes[2], runs_by_name, t_grid)
        axes[2].set_title("position", fontsize=9)

    settings = _format_regime_inline(regime) if regime is not None else None
    _shared_top_legend(fig, axes[0],
                       f"Horizon (contrast regime)",
                       settings_line=rf"$\beta={beta}$, T={T}, {model}, n={n_seeds}")
    save_fig(fig, f"horizon_report{tag}_{model}_{objective}", subdir=subdir)


def plot_horizon_single_seed(runs_by_name, t_grid, *, T, model, objective,
                             summary_rows=None, threshold=1e-2, target_theta=None,
                             regime=None, seed=0, n_seeds=1, tag=""):
    """Horizon comparison: sensor | convergence | position.

    Sensor and convergence panels aggregate over the `n_seeds` runs in each
    `runs_by_name` entry (fraction-on-precise and median ± IQR error band); the
    position panel shows a single representative seed, since exploration paths
    don't aggregate meaningfully.  Reuses the shared `_panel_*` helpers so it
    stays in sync with `plot_horizon_report`."""
    if target_theta is None:
        target_theta = TRUE_THETA
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))

    _panel_sensor_fraction(axes[0], runs_by_name, t_grid)
    axes[0].set_title("sensor choice", fontsize=9)
    _panel_convergence(axes[1], runs_by_name, t_grid, target_theta,
                       model=model, threshold=threshold, summary_rows=summary_rows)
    axes[1].set_title("convergence", fontsize=9)
    _panel_position(axes[2], runs_by_name, t_grid)
    axes[2].set_title("position", fontsize=9)

    settings = _format_regime_inline(regime) if regime is not None else None
    _shared_top_legend(fig, axes[0],
                       f"Horizon (contrast regime) "
                       f"($T={T}$, {model}, n={n_seeds}; position seed {seed})",
                       settings_line=settings)
    save_fig(fig, f"horizon_single_seed{tag}_{model}_{objective}", subdir="report")


def plot_baseline_report(runs_by_name, summary_rows, t_grid, *, T, model, objective,
                         threshold=1e-2, target_theta=None, n_seeds=10,
                         metric="theta", include_position=True, tag="", subdir="report"):
    """Compact landscape baseline figure for the standard strategies:
    sensor choice, convergence, [position].  The sensor panel exposes the locked
    baselines (Sensor-0/1-only) and the planner's policy; convergence shows the
    Sensor-0-only penalty."""
    if target_theta is None:
        target_theta = TRUE_THETA
    ncol = 3 if include_position else 2
    figsize = (10.0, 3.2) if include_position else (7.5, 3.4)
    fig, axes = plt.subplots(1, ncol, figsize=figsize)

    _panel_sensor_fraction(axes[0], runs_by_name, t_grid)
    axes[0].set_title("sensor choice", fontsize=9)
    _panel_convergence(axes[1], runs_by_name, t_grid, target_theta,
                       metric=metric, model=model, threshold=threshold,
                       summary_rows=summary_rows)
    axes[1].set_title("convergence", fontsize=9)
    if include_position:
        _panel_position(axes[2], runs_by_name, t_grid)
        axes[2].set_title("position", fontsize=9)

    _shared_top_legend(fig, axes[0],
                       f"Baseline comparison", settings_line=f"$T={T}$, {model}, n={n_seeds}")
    save_fig(fig, f"baseline_report{tag}_{model}_{objective}", subdir=subdir)


def plot_results_combined(horizon_runs, horizon_rows, baseline_runs, baseline_rows,
                          t_grid, *, T, model, objective, threshold=1e-2,
                          target_theta=None, n_seeds=10, metric="theta", tag="", subdir="report"):
    """Single ½–⅔ page results grid: top row horizon (sensor | error),
    bottom row baseline (θ̂ | error).  Per-row legends (strategy sets differ)."""
    if target_theta is None:
        target_theta = TRUE_THETA
    fig, ((ax_hs, ax_he), (ax_bt, ax_be)) = plt.subplots(2, 2, figsize=(7.5, 6.0))

    _panel_sensor_fraction(ax_hs, horizon_runs, t_grid)
    ax_hs.set_title("Horizon (contrast): sensor choice", fontsize=9)
    ax_hs.legend(fontsize=7, loc="best")
    _panel_convergence(ax_he, horizon_runs, t_grid, target_theta, model=model,
                       threshold=threshold, summary_rows=horizon_rows)
    ax_he.set_title("Horizon: convergence", fontsize=9)

    _panel_sensor_fraction(ax_bt, baseline_runs, t_grid)
    ax_bt.set_title("Baseline (default): sensor choice", fontsize=9)
    ax_bt.legend(fontsize=7, loc="best")
    _panel_convergence(ax_be, baseline_runs, t_grid, target_theta, metric=metric,
                       model=model, threshold=threshold, summary_rows=baseline_rows)
    ax_be.set_title("Baseline: convergence", fontsize=9)

    fig.suptitle(f"Results  ($T={T}$, {model}, {objective}, n={n_seeds})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, f"results_combined{tag}_{model}_{objective}", subdir=subdir)


# ---------------------------------------------------------------------------
# Problem-statement figures.
# ---------------------------------------------------------------------------

def plot_problem(models=("linear", "sinc", "dent")):
    """Multi-panel: each model column, both sensor noise bands."""
    titles = {
        "linear": r"Well-specified:  $y = \theta_1 x + \theta_2$",
        "sinc":   r"Misspecified:  $y = \mathrm{sinc}(x)$",
        "dent":   r"Misspecified:  $y = ((x/\sigma)^2-1)\,e^{-x^2/2\sigma^2}$",
    }
    xs = np.linspace(A_POS, B_POS, 401)
    fig, axes = plt.subplots(1, len(models), figsize=(11, 3.2), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, m in zip(axes, models):
        ys = np.array([true_y(x, m) for x in xs])
        for sigma2, label, color, alpha in (
            (DEFAULT_PARAMS["sigma"][0],
             r"sensor 0  ($\sigma_0^2=$" f"{DEFAULT_PARAMS['sigma'][0]:.0e})", "C0", 0.15),
            (DEFAULT_PARAMS["sigma"][1],
             r"sensor 1  ($\sigma_1^2=$" f"{DEFAULT_PARAMS['sigma'][1]:.0e})", "C1", 0.30),
        ):
            std = np.sqrt(sigma2)
            ax.fill_between(xs, ys - std, ys + std, color=color, alpha=alpha,
                            label=label, linewidth=0)
        ax.plot(xs, ys, color="k", lw=2.0, label="true $y(x)$")
        ax.scatter(POSITION_GRID,
                   np.full_like(POSITION_GRID, fill_value=ax.get_ylim()[0]),
                   marker="|", color="k", s=60, clip_on=False, zorder=5)
        ax.set_title(titles[m], fontsize=10)
        ax.set_xlabel(r"position $x$")
        ax.set_xlim(A_POS, B_POS)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"observation $y$")
    axes[0].legend(loc="upper center", fontsize=8, ncol=1, framealpha=0.9)
    fig.suptitle("Problem setup: ground-truth observation models, two sensors",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "problem_statement", subdir="problem")


_PROBLEM_TITLES = {
    "linear":     r"Linear:  $y = \theta_2 x + \theta_3$",
    "quadratic":  r"Quadratic:  $y = \theta_1 x^2 + \theta_2 x + \theta_3$",
    "sinusoidal": r"Misspecified:  $y = \sin(\pi x/2 + 1)$",
    "sinc":       r"Misspecified:  $y = \mathrm{sinc}(x)$",
    "dent":       r"Misspecified:  $y = ((x/\sigma)^2-1)\,e^{-x^2/2\sigma^2}$",
}


def plot_problem_single(model="quadratic", filename=None):
    """Single-panel ground-truth visualisation for one model."""
    xs = np.linspace(A_POS, B_POS, 401)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.2))
    ys = np.array([true_y(x, model) for x in xs])
    for sigma2, label, color, alpha in (
        (DEFAULT_PARAMS["sigma"][0],
         r"sensor 0  ($\sigma_0^2=$" f"{DEFAULT_PARAMS['sigma'][0]:.0e})", "C0", 0.15),
        (DEFAULT_PARAMS["sigma"][1],
         r"sensor 1  ($\sigma_1^2=$" f"{DEFAULT_PARAMS['sigma'][1]:.0e})", "C1", 0.30),
    ):
        std = np.sqrt(sigma2)
        ax.fill_between(xs, ys - std, ys + std, color=color, alpha=alpha,
                        label=label, linewidth=0)
    ax.plot(xs, ys, color="k", lw=2.0, label="true $y(x)$")
    ax.scatter(POSITION_GRID,
               np.full_like(POSITION_GRID, fill_value=ax.get_ylim()[0]),
               marker="|", color="k", s=60, clip_on=False, zorder=5)
    ax.set_title(_PROBLEM_TITLES.get(model, model), fontsize=10)
    ax.set_xlabel(r"position $x$")
    ax.set_xlim(A_POS, B_POS)
    ax.set_ylabel(r"observation $y$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=1, framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, filename or f"problem_{model}", subdir="problem")


def plot_problem_linear():
    """Backwards-compatible wrapper (deprecated — use plot_problem_single)."""
    plot_problem_single("linear", filename="problem_linear")


def plot_problem_quadratic():
    plot_problem_single("quadratic", filename="problem_quadratic")


def _panel_problem_sinusoidal(ax):
    """Draw the sinusoidal-problem panel onto `ax`: true sinusoid, the best-fit
    quadratic (unreachable target), both sensor noise bands, and position ticks."""
    from config import best_fit_theta
    xs = np.linspace(A_POS, B_POS, 401)
    ys = np.array([true_y(x, "sinusoidal") for x in xs])
    theta_bf = best_fit_theta("sinusoidal")
    ys_quad = theta_bf[0]*xs**2 + theta_bf[1]*xs + theta_bf[2]
    for sigma2, label, color, alpha in (
        (DEFAULT_PARAMS["sigma"][0],
         r"sensor 0  ($\sigma_0^2=$" f"{DEFAULT_PARAMS['sigma'][0]:.0e})", "C0", 0.15),
        (DEFAULT_PARAMS["sigma"][1],
         r"sensor 1  ($\sigma_1^2=$" f"{DEFAULT_PARAMS['sigma'][1]:.0e})", "C1", 0.30),
    ):
        std = np.sqrt(sigma2)
        ax.fill_between(xs, ys - std, ys + std, color=color, alpha=alpha,
                        label=label, linewidth=0)
    ax.plot(xs, ys, color="k", lw=2.0, label=r"true $y(x) = \sin(\pi x/2 + 1)$")
    ax.plot(xs, ys_quad, color="C3", lw=1.5, ls="--",
            label="best-fit quadratic\n(unreachable target)")
    ax.scatter(POSITION_GRID,
               np.full_like(POSITION_GRID, fill_value=ax.get_ylim()[0]),
               marker="|", color="k", s=60, clip_on=False, zorder=5)
    ax.set_title(_PROBLEM_TITLES["sinusoidal"], fontsize=10)
    ax.set_xlabel(r"position $x$")
    ax.set_xlim(A_POS, B_POS)
    ax.set_ylabel(r"observation $y$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", fontsize=7, ncol=1, framealpha=0.9)


def plot_problem_sinusoidal():
    """Sinusoidal truth + the best-fit quadratic overlaid (so the misspecification is visible)."""
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.2))
    _panel_problem_sinusoidal(ax)
    fig.tight_layout()
    save_fig(fig, "problem_sinusoidal", subdir="problem")


def plot_sinusoidal_report(baseline_runs, baseline_rows, contrast_runs,
                           contrast_rows, t_grid, *, T, objective, n_seeds=10,
                           threshold=1e-2, beta_baseline=0.0, beta_contrast=0.20,
                           target_theta=None):
    """1×3 landscape report figure for the misspecified sinusoidal case:
    problem | baseline-regime convergence | contrast-regime convergence.

    Both convergence panels use the output-error metric on the sinusoidal model.
    The two regimes differ only in the time goal-prior weight β, called out in each
    panel title (baseline β=0 vs contrast β>0)."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))

    _panel_problem_sinusoidal(axes[0])

    for ax, runs, rows, beta in (
        (axes[1], baseline_runs, baseline_rows, beta_baseline),
        (axes[2], contrast_runs, contrast_rows, beta_contrast),
    ):
        _panel_convergence(ax, runs, t_grid, target_theta, metric="output",
                           model="sinusoidal", threshold=threshold,
                           summary_rows=rows)
        regime = "Baseline" if beta == beta_baseline else "Contrast"
        ax.set_title(rf"{regime} regime  ($\beta={beta:g}$)", fontsize=10)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"Sinusoidal (misspecified): problem & output-error convergence "
                 f"($T={T}$, {objective}, n={n_seeds})", fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, "sinusoidal_report", subdir="report")


# ---------------------------------------------------------------------------
# β sweep figure — consumes runners.run_beta_sweep output.
# ---------------------------------------------------------------------------

def plot_beta_sweep_compare(results, *, T, threshold=1e-2, metric="theta",
                            sensor_refs=None, filename=None):
    """Overlay TTT and final-err curves for multiple H values across β.

    `results` : dict[H] -> [(beta, ttts_array, finals_array), ...]
    `sensor_refs` : optional dict[name] -> (ttt_value, final_err_value) drawn
                    as horizontal reference lines (e.g. Sensor-1-only at β=0).
    """
    Hs = sorted(results.keys())
    betas = [b for b, _, _ in results[Hs[0]]]

    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True)
    h_colors = {1: "C3", 3: "C4", 5: "C5"}
    for H in Hs:
        rows = results[H]
        ts = [t for _, t, _ in rows]
        fs = [f for _, _, f in rows]
        t_med = [np.nanmedian(t) if np.any(~np.isnan(t)) else np.nan for t in ts]
        t_q1  = [np.nanquantile(t, 0.25) if np.any(~np.isnan(t)) else np.nan for t in ts]
        t_q3  = [np.nanquantile(t, 0.75) if np.any(~np.isnan(t)) else np.nan for t in ts]
        f_med = [np.median(f) for f in fs]
        f_q1  = [np.quantile(f, 0.25) for f in fs]
        f_q3  = [np.quantile(f, 0.75) for f in fs]
        color = h_colors.get(H, f"C{H}")
        ax_t.plot(betas, t_med, "o-", color=color, label=f"H={H}", lw=1.8)
        ax_t.fill_between(betas, t_q1, t_q3, color=color, alpha=0.18, linewidth=0)
        ax_f.plot(betas, f_med, "o-", color=color, label=f"H={H}", lw=1.8)
        ax_f.fill_between(betas, f_q1, f_q3, color=color, alpha=0.18, linewidth=0)

    if sensor_refs:
        for name, (ttt_val, ferr_val) in sensor_refs.items():
            color = COLORS.get(name, "k")
            if not np.isnan(ttt_val):
                ax_t.axhline(ttt_val, color=color, ls="--", lw=1.2, alpha=0.8,
                             label=name)
            if not np.isnan(ferr_val):
                ax_f.axhline(ferr_val, color=color, ls="--", lw=1.2, alpha=0.8,
                             label=name)

    err_tex = (r"\|\hat\theta-\theta^\star\|^2" if metric == "theta"
               else r"\mathbb{E}_x[(\hat y(x) - y(x))^2]")
    ax_t.set_ylabel(rf"time to ${err_tex} < {threshold:g}$")
    ax_t.set_title(f"β sweep — H=1 vs H=3  (T={T})")
    ax_t.set_xscale("symlog", linthresh=1e-2)
    ax_t.grid(True, alpha=0.3, which="both")
    ax_t.legend(fontsize=8, loc="best")
    ax_f.set_ylabel(rf"final ${err_tex}$")
    ax_f.set_xlabel(r"$\beta$ (time goal-prior weight)")
    ax_f.set_yscale("log")
    ax_f.set_xscale("symlog", linthresh=1e-2)
    ax_f.grid(True, alpha=0.3, which="both")
    ax_f.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_fig(fig, filename or "planning_H_vs_beta", subdir="beta_sweep")


def plot_beta_sweep(rows, *, H, T, threshold=1e-2, subdir="beta_sweep", tag=""):
    """Two-panel β sweep: time-to-threshold (top) and final error (bottom)."""
    bs    = [b for b, _, _ in rows]
    t_med = [np.nanmedian(t) for _, t, _ in rows]
    t_q1  = [np.nanquantile(t, 0.25) for _, t, _ in rows]
    t_q3  = [np.nanquantile(t, 0.75) for _, t, _ in rows]
    f_med = [np.median(f) for _, _, f in rows]
    f_q1  = [np.quantile(f, 0.25) for _, _, f in rows]
    f_q3  = [np.quantile(f, 0.75) for _, _, f in rows]

    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True)
    ax_t.plot(bs, t_med, "o-", color="C4")
    ax_t.fill_between(bs, t_q1, t_q3, color="C4", alpha=0.2)
    ax_t.set_ylabel(rf"time to $\|\hat\theta-\theta^\star\|^2<{threshold}$")
    ax_t.set_xscale("symlog", linthresh=1e-2)
    ax_t.set_title(f"β sweep  (H={H}, T={T})")
    ax_t.grid(True, alpha=0.3, which="both")

    ax_f.plot(bs, f_med, "o-", color="C4")
    ax_f.fill_between(bs, f_q1, f_q3, color="C4", alpha=0.2)
    ax_f.set_ylabel(r"final $\|\hat\theta-\theta^\star\|^2$")
    ax_f.set_xlabel(r"$\beta$")
    ax_f.set_yscale("log")
    ax_f.set_xscale("symlog", linthresh=1e-2)
    ax_f.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    save_fig(fig, f"beta_sweep_H{H}{tag}", subdir=subdir)
