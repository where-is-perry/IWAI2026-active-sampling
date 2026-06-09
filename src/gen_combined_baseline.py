"""Combined baseline comparison (joint_det_full), 1x3:
  1) sensor choice — quadratic (solid) + sinusoidal (dashed)
  2) convergence — quadratic (theta-error metric)
  3) convergence — sinusoidal (output-error metric)

Quadratic + sinusoidal baseline runs are cached to /tmp (the sinusoidal cache is
shared with gen_baseline_sin.py)."""
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import config, runners
from plots import _panel_convergence, _step_interp, COLORS, save_fig

OBJ = "joint_det_full"; T = 100; NS = 20; THRESH = 1e-2


def baseline_runs(model, metric, cache_file):
    key = (OBJ, T, NS, model, metric)
    cache_path = Path(cache_file)
    if cache_path.exists():
        blob = pickle.loads(cache_path.read_bytes())
        if blob.get("key") == key:
            print(f"loaded {model} baseline cache")
            return blob["payload"]
    target = config.best_fit_theta(model)
    base = runners.standard_strategies(
        T=T, H_main=3, beta=0.0, objective=OBJ, model=model,
        include_random=True, include_greedy=False)
    runs, rows, tg = runners.run_strategies(
        base, T=T, n_seeds=NS, target_theta=target, metric=metric,
        model=model, threshold=THRESH)
    payload = (runs, rows, tg)
    cache_path.write_bytes(pickle.dumps({"key": key, "payload": payload}))
    return payload


q_runs, q_rows, tg = baseline_runs("quadratic", "theta", "/tmp/baseline_quad_cache.pkl")
s_runs, s_rows, _ = baseline_runs("sinusoidal", "output", "/tmp/baseline_sin_cache.pkl")
q_target = config.best_fit_theta("quadratic")
s_target = config.best_fit_theta("sinusoidal")

fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.1))

# --- panel 1: sensor choice, all 7 traces ----------------------------------
# At baseline (beta=0) every planner pins to the precise sensor (frac=1) for both
# problems, so the four planner curves coincide at the top.
ax = axes[0]

def frac_of(runs, name):
    return np.mean([_step_interp(tg, r["time"], r["sensor"]) for r in runs[name]], axis=0)

for runs, mlabel, mls in ((q_runs, "quadratic", "-"), (s_runs, "sinusoidal", "--")):
    for name in ("Planner H=3", "Planner H=1"):
        ax.plot(tg, frac_of(runs, name),
                color=COLORS[name], ls=mls, lw=1.6, label=f"{name} ({mlabel})")
ax.plot(tg, frac_of(q_runs, "Sensor 1 only"), color=COLORS["Sensor 1 only"],
        ls=":", lw=1.6, label="Sensor 1 only")
ax.plot(tg, frac_of(q_runs, "Sensor 0 only"), color=COLORS["Sensor 0 only"],
        ls=":", lw=1.6, label="Sensor 0 only")
ax.plot(tg, frac_of(q_runs, "Random"), color=COLORS["Random"],
        ls="--", lw=1.3, alpha=0.85, label="Random")
ax.set_ylim(-0.08, 1.12)
ax.set_ylabel("frac. on precise sensor", fontsize=9)
ax.set_xlabel("elapsed time")
ax.set_title("sensor choice", fontsize=9)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="best", framealpha=0.9)

# --- panel 2: convergence, quadratic ---------------------------------------
_panel_convergence(axes[1], q_runs, tg, q_target, metric="theta", model="quadratic",
                   threshold=THRESH, summary_rows=q_rows)
axes[1].set_title("convergence — quadratic", fontsize=9)
axes[1].yaxis.label.set_fontsize(9)
axes[1].legend(fontsize=8, loc="upper right", framealpha=0.9)

# --- panel 3: convergence, sinusoidal --------------------------------------
_panel_convergence(axes[2], s_runs, tg, s_target, metric="output", model="sinusoidal",
                   threshold=THRESH, summary_rows=s_rows)
axes[2].set_title("convergence — sinusoidal", fontsize=9)
axes[2].yaxis.label.set_fontsize(9)
axes[2].legend(fontsize=8, loc="upper right", framealpha=0.9)

fig.tight_layout(rect=[0.03, 0, 1, 0.95])
fig.canvas.draw()
xc = (axes[0].get_position().x0 + axes[2].get_position().x1) / 2
fig.suptitle("Baseline comparison", fontsize=9, x=xc, y=1.02)
save_fig(fig, "baseline_combined_quad_sin", dpi=204)
print("saved baseline_combined_quad_sin.png", flush=True)
