"""Simulation runners + multi-seed aggregation + LaTeX-table writer.

No plotting code. Each runner returns a results dict; `run_strategies`
turns a list of (name, factory) tuples into a (runs_by_name,
summary_rows, t_grid) triple ready for plotting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bayes_filter import BayesFilter
from config import (
    A_POS, B_POS, POSITION_GRID,
    DEFAULT_PARAMS, DEFAULT_PRIORS,
    TRUE_THETA, true_y, best_fit_theta,
)


# ---------------------------------------------------------------------------
# Error metric.
# ---------------------------------------------------------------------------

_ERR_GRID = np.linspace(A_POS, B_POS, 401)
_ERR_PHI = np.column_stack([_ERR_GRID**2, _ERR_GRID, np.ones_like(_ERR_GRID)])
_ERR_Y_TRUE_CACHE = {}


def _y_true_grid(model):
    if model not in _ERR_Y_TRUE_CACHE:
        _ERR_Y_TRUE_CACHE[model] = np.array([true_y(x, model) for x in _ERR_GRID])
    return _ERR_Y_TRUE_CACHE[model]


def error_series(r, target_theta, metric="theta", model="quadratic"):
    """Per-step scalar error along a run.

    metric='theta'  — squared parameter mismatch ||θ̂ − target_theta||².
    metric='output' — mean over the position grid of (φ(x)ᵀθ̂ − y_true(x; model))².
                      Makes sense for misspecified truths (sinusoidal/sinc/dent)
                      where no θ recovers y_true exactly.
    """
    if metric == "theta":
        return [(t1 - target_theta[0])**2
                + (t2 - target_theta[1])**2
                + (t3 - target_theta[2])**2
                for t1, t2, t3 in zip(r["theta1"], r["theta2"], r["theta3"])]
    if metric == "output":
        y_true = _y_true_grid(model)
        thetas = np.array(list(zip(r["theta1"], r["theta2"], r["theta3"])))  # (K, 3)
        y_hat = thetas @ _ERR_PHI.T                                          # (K, N_grid)
        return list(np.mean((y_hat - y_true)**2, axis=1))
    raise ValueError(f"unknown metric: {metric!r}")


def _make_filter(params, priors):
    return BayesFilter(
        params["delta"], params["sigma"], params["xi"], params["sigma_t"],
        params["zeta_x"], params["zeta_t"], params["travel_cost"],
        priors["m_theta0"], priors["V_theta0"],
        priors["m_x0"], priors["v_x0"],
        priors["m_t0"], priors["v_t0"],
        params["a_lim"], params["b_lim"],
    )


def run_state_estimation(T, seed=0, model="quadratic",
                         params=None, priors=None):
    """Random-policy baseline: uniform position action, sensor switch with p=0.25."""
    params = params or DEFAULT_PARAMS
    priors = priors or DEFAULT_PRIORS
    bf = _make_filter(params, priors)

    rng = np.random.default_rng(seed=seed)
    results = {k: [] for k in
               ("time", "position", "sensor", "nll", "theta1", "theta2", "theta3", "V_theta")}

    r = 0.0    # true elapsed time
    x = 0.0    # true position
    l = 0      # active sensor

    while r < T:
        u = rng.uniform(A_POS - x, B_POS - x)
        s = rng.choice([0, 0, 0, 1])

        x = x + u + rng.normal(0, np.sqrt(params["zeta_x"]))
        l = abs(l - s)
        y = true_y(x, model) + rng.normal(0, np.sqrt(params["sigma"][l]))
        r = (r + params["delta"][l] + params["xi"]*s
             + params["travel_cost"]*abs(u)
             + rng.normal(0, np.sqrt(params["zeta_t"])))

        nll = bf.step(l, y, r, s, u)
        results["time"].append(r)
        results["position"].append(x)
        results["sensor"].append(l)   # ACTIVE sensor identity (matches run_planning)
        results["nll"].append(nll)
        results["theta1"].append(bf.m_theta[0])
        results["theta2"].append(bf.m_theta[1])
        results["theta3"].append(bf.m_theta[2])
        results["V_theta"].append(bf.V_theta)

    return results


def run_planning(T, H, fixed_sensor=None, beta=0.0, seed=0,
                 objective="joint", model="quadratic",
                 params=None, priors=None):
    """Active-inference planner; EFE-driven choice of (s, u) at each step."""
    params = params or DEFAULT_PARAMS
    priors = priors or DEFAULT_PRIORS
    bf = _make_filter(params, priors)

    rng = np.random.default_rng(seed=seed)
    results = {k: [] for k in
               ("time", "position", "sensor", "nll", "theta1", "theta2", "theta3", "V_theta", "eig")}

    r = 0.0
    x = 0.0
    l = 0
    first = True

    while r < T:
        s, u, eig = bf.plan_multistep_EFE(
            POSITION_GRID, H=H, T_budget=T, x=x,
            fixed_l=fixed_sensor, beta=beta,
            first_step=first, objective=objective,
        )

        x = x + u + rng.normal(0, np.sqrt(params["zeta_x"]))
        l = abs(l - s)
        y = true_y(x, model) + rng.normal(0, np.sqrt(params["sigma"][l]))
        switch_cost = 0.0 if first else params["xi"] * s
        r = (r + params["delta"][l] + switch_cost
             + params["travel_cost"]*abs(u)
             + rng.normal(0, np.sqrt(params["zeta_t"])))
        first = False

        nll = bf.step(l, y, r, s, u)
        results["time"].append(r)
        results["position"].append(x)
        results["sensor"].append(l)   # report ACTIVE sensor identity
        results["nll"].append(nll)
        results["theta1"].append(bf.m_theta[0])
        results["theta2"].append(bf.m_theta[1])
        results["theta3"].append(bf.m_theta[2])
        results["V_theta"].append(bf.V_theta)
        results["eig"].append(eig)

    return results


# ---------------------------------------------------------------------------
# Standard strategy registry — the five-way comparison.
# ---------------------------------------------------------------------------

def standard_strategies(T, H_main=3, beta=0.0, objective="joint", model="quadratic",
                        include_random=True, include_greedy=True):
    """Return [(name, fn(seed) -> result_dict), ...] for the standard comparison."""
    strats = [
        (f"Planner H={H_main}", lambda s, H=H_main: run_planning(
            T, H=H, beta=beta, objective=objective, model=model, seed=s)),
        ("Planner H=1",         lambda s: run_planning(
            T, H=1, beta=beta, objective=objective, model=model, seed=s)),
    ]
    if include_greedy:
        strats.append(
            (r"Greedy $\beta=0$", lambda s: run_planning(
                T, H=1, beta=0.0, objective=objective, model=model, seed=s)))
    strats.extend([
        ("Sensor 0 only",       lambda s: run_planning(
            T, H=1, beta=beta, fixed_sensor=0, objective=objective, model=model, seed=s)),
        ("Sensor 1 only",       lambda s: run_planning(
            T, H=1, beta=beta, fixed_sensor=1, objective=objective, model=model, seed=s)),
    ])
    if include_random:
        strats.append(("Random", lambda s: run_state_estimation(T, model=model, seed=s)))
    return strats


# ---------------------------------------------------------------------------
# Multi-seed runner + aggregation onto a common time grid.
# ---------------------------------------------------------------------------

def run_strategies(strategies, T, n_seeds=10, threshold=1e-2, n_tgrid=501,
                   target_theta=None, metric="theta", model="quadratic"):
    """Execute every (name, fn) for seeds 0..n_seeds-1 and aggregate.

    `target_theta` is the reference point for the parameter-space error metric;
    defaults to TRUE_THETA.  Ignored when metric='output'.

    `metric` selects the error functional — see `error_series()`.

    Returns:
      runs_by_name : dict[name] -> list of result dicts (length n_seeds)
      summary_rows : list of dicts with strategy/ttt_median/reached_frac/final_*_median
      t_grid       : np.ndarray, shape (n_tgrid,)
    """
    if target_theta is None:
        target_theta = TRUE_THETA
    t_grid = np.linspace(0.0, T, n_tgrid)
    seeds = list(range(n_seeds))
    runs_by_name = {name: [fn(s) for s in seeds] for name, fn in strategies}

    summary_rows = []
    for name, _ in strategies:
        err_stack, post_stack = [], []
        for r in runs_by_name[name]:
            err_raw = error_series(r, target_theta, metric=metric, model=model)
            post_raw = [np.trace(V) for V in r["V_theta"]]
            err_stack.append(np.interp(t_grid, r["time"], err_raw))
            post_stack.append(np.interp(t_grid, r["time"], post_raw))
        err_arr, post_arr = np.array(err_stack), np.array(post_stack)
        err_med = np.median(err_arr, axis=0)
        post_med = np.median(post_arr, axis=0)

        ttts = []
        for err_curve in err_stack:
            below = np.where(err_curve < threshold)[0]
            ttts.append(t_grid[below[0]] if below.size else np.nan)
        ttts = np.array(ttts)
        summary_rows.append({
            "strategy": name,
            "ttt_median": np.nanmedian(ttts) if np.any(~np.isnan(ttts)) else np.nan,
            "reached_frac": float(np.mean(~np.isnan(ttts))),
            "final_err_median": err_med[-1],
            "final_trV_median": post_med[-1],
        })

    return runs_by_name, summary_rows, t_grid


def print_summary(summary_rows, *, tag="", T, n_seeds, threshold=1e-2):
    print(f"\n=== Summary for {tag.lstrip('_')}  (T={T}, n_seeds={n_seeds}, threshold={threshold}) ===")
    print(f"{'strategy':<18} {'t→<'+str(threshold):>12} {'reached%':>10} {'final err':>12} {'final trV':>12}")
    for row in summary_rows:
        ttt = row['ttt_median']
        ttt_str = f"{ttt:.2f}" if not np.isnan(ttt) else "  (never)"
        print(f"{row['strategy']:<18} {ttt_str:>12} {100*row['reached_frac']:>9.0f}% "
              f"{row['final_err_median']:>12.2e} {row['final_trV_median']:>12.2e}")


_ERR_TEX = {
    "theta":  r"\|\hat\theta-\theta^\star\|^2",
    "output": r"\mathbb{E}_x[(\hat y(x) - y(x))^2]",
}


def write_summary_tex(summary_rows, path, threshold=1e-2, metric="theta"):
    """LaTeX tabular: strategy / time-to-threshold / reached% / final err / final trV."""
    err_tex = _ERR_TEX[metric]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write(f"Strategy & $t_{{<{threshold:g}}}$ & reached & "
                f"final ${err_tex}$ & "
                "final $\\operatorname{tr} V_\\theta$ \\\\\n\\midrule\n")
        for row in summary_rows:
            ttt = row['ttt_median']
            ttt_str = f"{ttt:.1f}" if not np.isnan(ttt) else "---"
            f.write(f"{row['strategy']} & {ttt_str} & "
                    f"{100*row['reached_frac']:.0f}\\% & "
                    f"{row['final_err_median']:.2e} & "
                    f"{row['final_trV_median']:.2e} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------------------
# β sweep — simulation half (the plotter consumes `rows`).
# ---------------------------------------------------------------------------

def run_beta_sweep(T=100, H=3, betas=(0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0),
                   n_seeds=10, threshold=1e-2, objective="joint", model="quadratic",
                   n_tgrid=501, target_theta=None, metric="theta"):
    """Run run_planning across β values; return list of (beta, ttt_array, final_err_array)."""
    if target_theta is None:
        target_theta = best_fit_theta(model)
    t_grid = np.linspace(0.0, T, n_tgrid)
    rows = []
    for beta in betas:
        ttts, finals = [], []
        for s in range(n_seeds):
            r = run_planning(T, H, beta=beta, seed=s, objective=objective, model=model)
            err_raw = error_series(r, target_theta, metric=metric, model=model)
            err = np.interp(t_grid, r["time"], err_raw)
            below = np.where(err < threshold)[0]
            ttts.append(t_grid[below[0]] if below.size else np.nan)
            finals.append(err[-1])
        rows.append((beta, np.array(ttts), np.array(finals)))
    return rows


def print_beta_sweep(rows, *, H, T, n_seeds, threshold=1e-2):
    print(f"\n=== β sweep (H={H}, T={T}, n_seeds={n_seeds}, threshold={threshold}) ===")
    print(f"{'beta':>8} {'t→thresh (med)':>16} {'reached%':>10} {'final err (med)':>18}")
    for beta, t, f in rows:
        ttt_med = np.nanmedian(t) if np.any(~np.isnan(t)) else np.nan
        ttt_str = f"{ttt_med:.2f}" if not np.isnan(ttt_med) else "(never)"
        print(f"{beta:>8.3f} {ttt_str:>16} {100*np.mean(~np.isnan(t)):>9.0f}% "
              f"{np.median(f):>18.2e}")


# ---------------------------------------------------------------------------
# β sweep with H=1 vs H=3 (or any set of H's) overlaid.
# ---------------------------------------------------------------------------

def run_beta_sweep_compare(T=100, Hs=(1, 3),
                           betas=(0.0, 0.01, 0.02, 0.05, 0.10, 0.20),
                           n_seeds=20, threshold=1e-2, objective="joint",
                           model="quadratic", n_tgrid=501, target_theta=None,
                           metric="theta"):
    """Run run_planning across (H, β) values.

    Returns: dict[H] -> list of (beta, ttts_array, finals_array).
    """
    if target_theta is None:
        target_theta = best_fit_theta(model)
    t_grid = np.linspace(0.0, T, n_tgrid)
    out = {}
    for H in Hs:
        rows = []
        for beta in betas:
            ttts, finals = [], []
            for s in range(n_seeds):
                r = run_planning(T, H, beta=beta, seed=s, objective=objective,
                                 model=model)
                err_raw = error_series(r, target_theta, metric=metric, model=model)
                err = np.interp(t_grid, r["time"], err_raw)
                below = np.where(err < threshold)[0]
                ttts.append(t_grid[below[0]] if below.size else np.nan)
                finals.append(err[-1])
            rows.append((beta, np.array(ttts), np.array(finals)))
        out[H] = rows
    return out


def print_beta_sweep_compare(results, *, T, n_seeds, threshold=1e-2):
    print(f"\n=== β sweep compare (T={T}, n_seeds={n_seeds}, threshold={threshold}) ===")
    Hs = sorted(results.keys())
    head = f"{'beta':>6}"
    for H in Hs:
        head += f"  {'H='+str(H)+' TTT':>10}  {'reach%':>7}  {'finalErr':>10}"
    print(head)
    betas = [b for b, _, _ in results[Hs[0]]]
    for i, beta in enumerate(betas):
        row = f"{beta:>6.3f}"
        for H in Hs:
            _, t, f = results[H][i]
            ttt = np.nanmedian(t) if np.any(~np.isnan(t)) else np.nan
            ttt_s = f"{ttt:.2f}" if not np.isnan(ttt) else "(never)"
            row += f"  {ttt_s:>10}  {100*np.mean(~np.isnan(t)):>6.0f}%  {np.median(f):>10.2e}"
        print(row)
