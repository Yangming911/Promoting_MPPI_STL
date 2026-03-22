"""
Run our adaptive-cooling PI solver on Tasks II / IV / V / VI and save results to JSON.

Usage (from repo/ dir, conda mppi):
    python run_ours_json.py --tasks 2 4 5 6 --runs 100 --out-dir ../results
    python run_ours_json.py --tasks 2 4 5 6 --runs 1  --out-dir ../results  # trajectory only
"""

import sys
import os
import argparse
import json
import time
import contextlib
import numpy as np
from tqdm import tqdm

# Make sure repo/ Python package is on path
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

import matplotlib
matplotlib.use('Agg')

# ── problem registry ─────────────────────────────────────────────────────────
from problems.problem_II  import Problem_II
from problems.problem_IV  import Problem_IV
from problems.problem_V   import Problem_V
from problems.problem_VI  import Problem_VI

PROBLEM_MAP = {2: Problem_II, 4: Problem_IV, 5: Problem_V, 6: Problem_VI}


@contextlib.contextmanager
def suppress_output():
    """Redirect fd-level stdout/stderr to /dev/null (silences C++ prints)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = [os.dup(1), os.dup(2)]
    os.dup2(devnull, 1); os.dup2(devnull, 2)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout, sys.stderr = old_out, old_err
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        os.close(saved[0]); os.close(saved[1]); os.close(devnull)


def run_task(task_id: int, n_runs: int, out_dir: str):
    ProblemClass = PROBLEM_MAP[task_id]
    solver_name  = 'PI'          # our adaptive-cooling PI
    method_label = 'Adaptive-PI'

    costs, rhos, times, successes = [], [], [], []
    cost_lists   = []   # convergence curves
    traj_x = None       # save one representative trajectory

    real_stdout = sys.stdout
    for i in tqdm(range(n_runs), desc=f'Task {task_id} [{method_label}]', file=real_stdout):
        with suppress_output():
            p = ProblemClass()
            p.solve(solver_name, use_pi_cpp_implementation=True)

        sol = p.solutions[solver_name]
        costs.append(float(sol.cost))
        rhos.append(float(sol.rho))
        times.append(float(sol.solve_time))
        successes.append(1 if sol.rho >= 0 else 0)

        if hasattr(p, 'cost_list') and p.cost_list is not None:
            cost_lists.append([float(c) for c in p.cost_list])

        # Save trajectory from the first successful run (or last run)
        if traj_x is None or (sol.rho >= 0 and traj_x is None):
            traj_x = sol.x.tolist() if sol.x is not None else None

    result = {
        'task'         : task_id,
        'method'       : method_label,
        'solver'       : solver_name,
        'n_runs'       : n_runs,
        # per-run arrays
        'costs'        : costs,
        'rhos'         : rhos,
        'times'        : times,
        'successes'    : successes,
        # summary statistics
        'cost_mean'    : float(np.mean(costs)),
        'cost_std'     : float(np.std(costs)),
        'rho_mean'     : float(np.mean(rhos)),
        'rho_std'      : float(np.std(rhos)),
        'time_mean'    : float(np.mean(times)),
        'time_std'     : float(np.std(times)),
        'success_rate' : float(sum(successes) / n_runs),
        # convergence (list of cost_lists, one per run)
        'cost_lists'   : cost_lists,
        # representative trajectory
        'traj_x'       : traj_x,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'ours_task{task_id}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    tqdm.write(
        f'  Task {task_id} | cost={result["cost_mean"]:.4f}±{result["cost_std"]:.4f} '
        f'| time={result["time_mean"]:.2f}s±{result["time_std"]:.2f} '
        f'| success={sum(successes)}/{n_runs}',
        file=real_stdout
    )
    tqdm.write(f'  → saved {out_path}', file=real_stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks',   type=int, nargs='+', default=[2, 4, 5, 6])
    parser.add_argument('--runs',    type=int, default=100)
    parser.add_argument('--out-dir', type=str, default='../results')
    args = parser.parse_args()

    for t in args.tasks:
        if t not in PROBLEM_MAP:
            print(f'[WARN] Task {t} not in PROBLEM_MAP, skipping.')
            continue
        run_task(t, args.runs, args.out_dir)


if __name__ == '__main__':
    main()
