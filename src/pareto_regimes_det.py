"""Report figure: a Pareto front over regimes showing adaptive planning.

Across a grid of regimes (β, switching cost Ξ, sensor times δ, noises σ) we run
four planners and place each (regime × planner) as a point in the
speed-vs-accuracy plane:

    x = time-to-threshold (median)   — how fast a good θ-estimate is reached
    y = final θ-error ‖m_θ−θ*‖²      — how good the final estimate is
    (bottom-left = fast AND accurate)

Story: the ADAPTIVE planners (H=1, H=3) pick a sensor/timing strategy suited to
each regime, so they trace the accuracy/speed frontier; the FIXED-sensor
baselines (Sensor-0-only, Sensor-1-only) are each good only in part of the
space and are dominated.  Adaptivity is shown twice: a Pareto envelope over the
adaptive points, and H=3 colored by `frac_l1` (which sensor it chose) — sensor-1
in accuracy-favouring regimes, sensor-0 under time pressure.

Writes a NEW file figures/report/pareto_regimes_<model>_<objective>.png; it does
not touch any existing committed plot.

Run:  python src/pareto_regimes.py
"""

from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_PARAMS, DEFAULT_PRIORS, best_fit_theta, save_fig
from runners import run_planning

# --- experiment knobs (trim here to go faster) -----------------------------
T = 100
MODEL = "quadratic"
OBJECTIVE = "joint_det_full"
SEEDS = tuple(range(10))
THRESHOLD = 1e-2

BETAS  = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)   # time goal-prior strength
XIS    = (3.0, 15.0)                  # switching cost Ξ
DELTA1 = (10.0, 15.0)                 # sensor-1 measurement time (sensor-0 = 1)
SIGMA0 = (1e-2, 5e-2)                 # sensor-0 noise (sensor-1 = 1e-3)

# The four planners. Adaptive ones choose s/u freely; baselines lock the sensor.
PLANNERS = {
    "Planner H=1":   dict(H=1, fixed_sensor=None, adaptive=True),
    "Planner H=3":   dict(H=3, fixed_sensor=None, adaptive=True),
    "Sensor 0 only": dict(H=1, fixed_sensor=0,    adaptive=False),
    "Sensor 1 only": dict(H=1, fixed_sensor=1,    adaptive=False),
}
MARKERS = {"Planner H=1": "o", "Planner H=3": "s",
           "Sensor 0 only": "v", "Sensor 1 only": "^"}


def run_metrics(params, beta, H, fixed_sensor, seed, target):
    r = run_planning(T=T, H=H, beta=beta, seed=seed, fixed_sensor=fixed_sensor,
                     objective=OBJECTIVE, model=MODEL, params=params,
                     priors=DEFAULT_PRIORS)
    errs = np.array([(t1-target[0])**2 + (t2-target[1])**2 + (t3-target[2])**2
                     for t1, t2, t3 in zip(r["theta1"], r["theta2"], r["theta3"])])
    times = np.array(r["time"])
    crossed = np.where(errs < THRESHOLD)[0]
    ttt = float(times[crossed[0]]) if crossed.size else np.nan
    sensors = np.array(r["sensor"])
    frac_l1 = float(np.mean(sensors)) if sensors.size else 0.0
    n_switches = int(np.sum(np.abs(np.diff(sensors)))) if sensors.size > 1 else 0
    return dict(final=float(errs[-1]), ttt=ttt, frac_l1=frac_l1,
                n_switches=n_switches)


def agg(ms):
    ttts = np.array([m["ttt"] for m in ms])
    reached = ~np.isnan(ttts)
    return dict(
        final_med=float(np.median([m["final"] for m in ms])),
        ttt_med=float(np.nanmedian(ttts)) if np.any(reached) else np.nan,
        reached=float(np.mean(reached)),
        frac_l1_med=float(np.median([m["frac_l1"] for m in ms])),
        switches_med=float(np.median([m["n_switches"] for m in ms])),
    )


def main():
    target = best_fit_theta(MODEL)
    regimes = list(product(BETAS, XIS, DELTA1, SIGMA0))
    print(f"{MODEL}  T={T}  objective={OBJECTIVE}  seeds={SEEDS}  "
          f"{len(regimes)} regimes x {len(PLANNERS)} planners\n")

    # Cache the sweep keyed on the config so plot tweaks don't recompute it.
    import json
    cache_key = json.dumps([BETAS, XIS, DELTA1, SIGMA0, list(SEEDS), T,
                            MODEL, OBJECTIVE])
    cache_path = Path("/tmp/pareto_rows_cache.json")
    rows = None
    if cache_path.exists():
        blob = json.loads(cache_path.read_text())
        if blob.get("key") == cache_key:
            rows = blob["rows"]
            print("loaded cached sweep results (delete /tmp/pareto_rows_cache.json"
                  " to force recompute)\n")
    if rows is None:
        rows = []   # one per (regime, planner)
        t0 = time.perf_counter()
        for beta, xi, d1, s0 in regimes:
            params = {**DEFAULT_PARAMS,
                      "xi": xi,
                      "delta": np.array([1.0, d1]),
                      "sigma": np.array([s0, 1e-3])}
            for name, cfg in PLANNERS.items():
                a = agg([run_metrics(params, beta, cfg["H"], cfg["fixed_sensor"],
                                     s, target) for s in SEEDS])
                rows.append(dict(planner=name, adaptive=cfg["adaptive"],
                                 beta=beta, xi=xi, d1=d1, s0=s0, **a))
            print(f"  done β={beta:<5g} Ξ={xi:<4g} δ1={d1:<4g} σ0={s0:<5g}  "
                  f"[{time.perf_counter()-t0:.0f}s]")
        cache_path.write_text(json.dumps({"key": cache_key, "rows": rows}))

    # --- table -------------------------------------------------------------
    print(f"\n{'planner':14s} {'β':>5s} {'Ξ':>4s} {'δ1':>4s} {'σ0':>6s} "
          f"{'final':>10s} {'ttt':>6s} {'reach':>6s} {'fl1':>5s} {'sw':>4s}")
    for r in rows:
        print(f"{r['planner']:14s} {r['beta']:>5g} {r['xi']:>4g} {r['d1']:>4g} "
              f"{r['s0']:>6g} {r['final_med']:>10.2e} "
              f"{r['ttt_med']:>6.1f} {r['reached']:>6.0%} "
              f"{r['frac_l1_med']:>5.2f} {r['switches_med']:>4.0f}")

    # --- figure: (L) speed/accuracy front, (R) adaptive sensor choice ------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    PCOLOR = {"Planner H=1": "tab:green", "Planner H=3": "tab:blue",
              "Sensor 0 only": "0.55", "Sensor 1 only": "0.2"}
    JITTER = {"Sensor 0 only": -1.2, "Planner H=1": -0.4,
              "Planner H=3": 0.4, "Sensor 1 only": 1.2}

    def xy(r):  # true coords; non-reachers parked at x=T
        x = r["ttt_med"] if not np.isnan(r["ttt_med"]) else float(T)
        return x, max(r["final_med"], 1e-4)

    # ---- LEFT: speed vs accuracy, one point per (regime, planner) ----------
    for name in PLANNERS:
        sub = [r for r in rows if r["planner"] == name]
        xs = [xy(r)[0] + JITTER[name] for r in sub]
        ys = [xy(r)[1] for r in sub]
        fc = ["none" if np.isnan(r["ttt_med"]) else PCOLOR[name] for r in sub]
        axL.scatter(xs, ys, marker=MARKERS[name], s=48, facecolors=fc,
                    edgecolors=PCOLOR[name], linewidths=1.2, alpha=0.85,
                    label=name, zorder=3)

    # global Pareto front over ALL points (minimise both ttt and error).
    pts = sorted(rows, key=lambda r: (xy(r)[0], xy(r)[1]))
    front, best_y = [], np.inf
    for r in pts:
        x, y = xy(r)
        if y < best_y - 1e-12:
            front.append((x, y, r)); best_y = y
    if len(front) > 1:
        fx, fy, _ = zip(*front)
        axL.step(fx, fy, where="post", color="k", lw=1.3, alpha=0.55,
                 zorder=1, label="global Pareto front")

    axL.axhline(THRESHOLD, color="k", ls=":", lw=0.9, alpha=0.6)
    axL.set_yscale("log")
    axL.set_xlabel("time-to-threshold (median)")
    axL.set_ylabel(r"final $\|m_\theta-\theta^\star\|^2$ (median)")
    axL.set_title("Speed vs accuracy across regimes")
    axL.grid(True, alpha=0.25, which="both")
    # "better" cue: text in the empty top-right (slow+inaccurate is sparse there)
    # with an arrow pointing down-left toward the good (fast & accurate) corner.
    axL.annotate("better\n(fast & accurate)",
                 xy=(0.10, 0.06), xytext=(0.74, 0.40),
                 xycoords="axes fraction", textcoords="axes fraction",
                 ha="center", va="center", fontsize=8.5, color="tab:red",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.3,
                                 connectionstyle="arc3,rad=0.15"))
    # only mention hollow markers in the legend if any actually exist
    has_hollow = any(np.isnan(r["ttt_med"]) for r in rows)
    leg_kw = dict(fontsize=8, loc="upper right")
    if has_hollow:
        leg_kw.update(title=f"hollow marker = never reached η (parked at T={T})",
                      title_fontsize=7)
    axL.legend(**leg_kw)

    # ---- RIGHT: sensor choice adapts to regime (frac_l1 vs beta, by Xi) ----
    betas = sorted(set(r["beta"] for r in rows))
    xis = sorted(set(r["xi"] for r in rows))
    XI_COLOR = {xis[0]: "tab:orange", xis[-1]: "tab:purple"}
    PL_LS   = {"Planner H=1": "-",  "Planner H=3": "--"}
    PL_MARK = {"Planner H=1": "o",  "Planner H=3": "s"}
    for name in ("Planner H=1", "Planner H=3"):
        for xi in xis:
            ys, los, his = [], [], []
            for b in betas:
                vals = [r["frac_l1_med"] for r in rows
                        if r["planner"] == name and r["xi"] == xi and r["beta"] == b]
                ys.append(np.median(vals))
                los.append(np.quantile(vals, 0.25)); his.append(np.quantile(vals, 0.75))
            axR.plot(betas, ys, ls=PL_LS[name], marker=PL_MARK[name], ms=7,
                     color=XI_COLOR[xi], lw=2.0, label=f"{name}  (Ξ={xi:g})")
            axR.fill_between(betas, los, his, color=XI_COLOR[xi], alpha=0.08, lw=0)
    axR.set_xlabel(r"$\beta$   (time pressure — higher = hurry more)")
    axR.set_ylabel("fraction of steps on the precise (slow) sensor")
    axR.set_ylim(-0.05, 1.10)
    axR.set_title("Sensor choice adapts to regime")
    axR.grid(True, alpha=0.25)
    axR.legend(fontsize=8, loc="upper right", handlelength=3.2, framealpha=0.9)

    fig.suptitle("Adaptive planning across regimes", y=1.07, fontsize=13)
    fig.text(0.5, 1.01, f"{len(regimes)} regimes = β×Ξ×δ₁×σ₀, n={len(SEEDS)}",
             ha="center", va="bottom", fontsize=9, color="0.35")
    fig.tight_layout()
    save_fig(fig, f"pareto_regimes_{MODEL}")
    print(f"\nsaved figures/pareto_regimes_{MODEL}.png")


if __name__ == "__main__":
    main()
