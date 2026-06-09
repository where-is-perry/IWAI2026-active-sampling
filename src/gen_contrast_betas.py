"""Condensed contrast-regime figure with one row per beta (0.1 and 0.2).

Same contrast regime as gen_contrast_det.py (sigma=[0.01,1e-3], delta=[1,10],
xi=15, joint_det_full); stacks two beta rows x three panels (sensor | convergence
| position) into a single compact figure.  Reuses the _panel_* helpers so it
stays in sync with plot_horizon_report.

Runs are cached to /tmp so cosmetic re-plots are instant (delete the cache file
to force a recompute).
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import config, runners
from plots import (_panel_sensor_fraction, _panel_convergence, _panel_position,
                   save_fig)

OBJ = "joint_det_full"; T = 100; NS = 20; MODEL = "quadratic"
BETAS = [0.1, 0.2]
THRESH = 1e-2
target = config.best_fit_theta(MODEL)

params = dict(config.DEFAULT_PARAMS)
params["sigma"] = np.array([0.01, 1e-3])
params["delta"] = np.array([1.0, 10.0])
params["xi"] = 15.0


def strategies_for(beta):
    return [
        ("Planner H=1", lambda s: runners.run_planning(
            T=T, H=1, beta=beta, seed=s, objective=OBJ, model=MODEL,
            params=params, priors=config.DEFAULT_PRIORS)),
        ("Planner H=3", lambda s: runners.run_planning(
            T=T, H=3, beta=beta, seed=s, objective=OBJ, model=MODEL,
            params=params, priors=config.DEFAULT_PRIORS)),
        ("Sensor 1 only", lambda s: runners.run_planning(
            T=T, H=1, beta=beta, fixed_sensor=1, seed=s, objective=OBJ, model=MODEL,
            params=params, priors=config.DEFAULT_PRIORS)),
    ]


# --- compute (or load) the runs -------------------------------------------
cache_path = Path("/tmp/contrast_betas_cache.pkl")
cache_key = (OBJ, T, NS, MODEL, tuple(BETAS), float(params["xi"]),
             tuple(params["sigma"]), tuple(params["delta"]))
data = None
if cache_path.exists():
    blob = pickle.loads(cache_path.read_bytes())
    if blob.get("key") == cache_key:
        data = blob["data"]
        print("loaded cached runs (delete /tmp/contrast_betas_cache.pkl to recompute)")
if data is None:
    data = {}
    for b in BETAS:
        runs, rows, tg = runners.run_strategies(
            strategies_for(b), T=T, n_seeds=NS, target_theta=target)
        runners.print_summary(rows, tag=f"_contrast_beta{b}", T=T, n_seeds=NS)
        data[b] = (runs, rows, tg)
    cache_path.write_bytes(pickle.dumps({"key": cache_key, "data": data}))

# --- condensed 2 (beta) x 3 (panel) grid ----------------------------------
fig, axes = plt.subplots(len(BETAS), 3, figsize=(9.0, 3.6), sharex="col")
col_titles = ["sensor choice", "convergence", "position"]
ylabels = ["frac. on precise sensor", r"$\|\hat\theta-\theta^\star\|^2$",
           "position (seed 0)"]
last = len(BETAS) - 1
for i, b in enumerate(BETAS):
    runs, rows, tg = data[b]
    _panel_sensor_fraction(axes[i, 0], runs, tg)
    _panel_convergence(axes[i, 1], runs, tg, target, model=MODEL,
                       threshold=THRESH, summary_rows=rows)
    _panel_position(axes[i, 2], runs, tg)
    for j in range(3):
        axes[i, j].set_ylabel("")            # re-added once, centered, below
        if i == 0:
            axes[i, j].set_title(col_titles[j], fontsize=9)
        if i != last:
            axes[i, j].set_xlabel("")        # x-label only on the bottom row

fig.tight_layout(rect=[0.12, 0, 1, 0.96])
fig.canvas.draw()

# one y-label per column, vertically centered across both rows
for j in range(3):
    p_top = axes[0, j].get_position()
    p_bot = axes[last, j].get_position()
    yc = (p_bot.y0 + p_top.y1) / 2
    frac = (yc - p_bot.y0) / (p_bot.y1 - p_bot.y0)   # bottom-axes fraction
    axes[last, j].set_ylabel(ylabels[j], fontsize=9)
    axes[last, j].yaxis.set_label_coords(-0.20, frac)

# beta row labels at the far left, centered on each row
for i, b in enumerate(BETAS):
    p = axes[i, 0].get_position()
    fig.text(p.x0 - 0.085, (p.y0 + p.y1) / 2, rf"$\beta={b}$", rotation=90,
             ha="center", va="center", fontsize=12, fontweight="bold")

# legend + title centered over the axes region, kept tight to the plots
xc = (axes[0, 0].get_position().x0 + axes[0, 2].get_position().x1) / 2
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8,
           frameon=False, bbox_to_anchor=(xc, 1.005))
fig.suptitle("Horizon (contrast regime)", fontsize=9, x=xc, y=1.05)
save_fig(fig, f"horizon_report_contrast_betas_{MODEL}")
print(f"saved horizon_report_contrast_betas_{MODEL}.png", flush=True)
