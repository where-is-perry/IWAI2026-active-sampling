"""Reproduce the paper figures.

Running `python src/main.py` regenerates the four figures used in the paper
(IWAI 2026, sensor switching) into `figures/`:

  1. problem_quadratic_sinusoidal.png            (gen_problem_pair.py)
  2. baseline_combined_quad_sin.png              (gen_combined_baseline.py)
  3. horizon_report_contrast_betas_quadratic.png (gen_contrast_betas.py)
  4. pareto_regimes_quadratic.png                (pareto_regimes_det.py)

All four use the log-determinant `joint_det_full` planning objective with the
time-budget fix (plans that exceed T are infeasible, G = +inf). Each generator
is also runnable on its own, e.g. `python src/gen_contrast_betas.py`.

The planner runs are cached under /tmp (keyed on their config) so re-runs are
fast; delete the matching `/tmp/*cache*` file to force a full recompute.
"""
from __future__ import annotations

import runpy
from pathlib import Path

SRC = Path(__file__).resolve().parent

# (generator script, the figure it writes into figures/)
GENERATORS = [
    ("gen_problem_pair.py",      "problem_quadratic_sinusoidal.png"),
    ("gen_combined_baseline.py", "baseline_combined_quad_sin.png"),
    ("gen_contrast_betas.py",    "horizon_report_contrast_betas_quadratic.png"),
    ("pareto_regimes_det.py",    "pareto_regimes_quadratic.png"),
]


def main():
    n = len(GENERATORS)
    for i, (script, fig) in enumerate(GENERATORS, 1):
        print(f"\n===== [{i}/{n}] {script}  ->  figures/{fig} =====", flush=True)
        runpy.run_path(str(SRC / script), run_name="__main__")
    print("\nAll paper figures regenerated into figures/.", flush=True)


if __name__ == "__main__":
    main()
