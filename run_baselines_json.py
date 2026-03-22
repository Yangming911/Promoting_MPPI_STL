"""
Run baseline solvers (PI fixed-psi, STL-GUIDED-PI, GRAD) on Tasks II / IV / V / VI.

Usage (from repo_origin/ dir, conda mppi):
    python run_baselines_json.py --tasks 2 4 5 6 --runs 100 --solvers PI STL-GUIDED-PI GRAD
    python run_baselines_json.py --tasks 2 4 5 6 --runs 1  --solvers PI
"""

import sys
import os
import argparse
import json
import numpy as np
from tqdm import tqdm
import contextlib

ORIGIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ORIGIN_DIR)

import matplotlib
matplotlib.use('Agg')

from problems.problem_II  import Problem_II
from problems.problem_IV  import Problem_IV
from problems.problem_V   import Problem_V
from problems.problem_VI  import Problem_VI

PROBLEM_MAP = {2: Problem_II, 4: Problem_IV, 5: Problem_V, 6: Problem_VI}

AVAILABLE_SOLVERS = ['PI', 'STL-GUIDED-PI', 'GRAD', 'CMA-ES']


@contextlib.contextmanager
def suppress_output():
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


def run_task_solver(task_id: int, solver_name: str, n_runs: int, out_dir: str):
    ProblemClass = PROBLEM_MAP[task_id]

    use_cpp = solver_name in ('PI', 'STL-GUIDED-PI')
    method_label = {
        'PI'           : 'PI-FixedPsi',
        'STL-GUIDED-PI': 'STL-Guided-PI',
        'GRAD'         : 'GRAD',
        'CMA-ES'       : 'CMA-ES',
    }[solver_name]

    costs, rhos, times, successes = [], [], [], []
    cost_lists = []
    traj_x = None

    real_stdout = sys.stdout
    for i in tqdm(range(n_runs), desc=f'Task {task_id} [{method_label}]', file=real_stdout):
        p = None
        try:
            with suppress_output():
                p = ProblemClass()
                p.solve(solver_name, use_pi_cpp_implementation=use_cpp)
        except AttributeError as e:
            if 'cost_list' not in str(e):
                tqdm.write(f'  [ERROR] run {i}: {e}', file=real_stdout)
                continue
            # solve() completed but returned self.cost_list which doesn't exist
            # for non-PI solvers — solution is still in p.solutions, proceed
        except Exception as e:
            tqdm.write(f'  [ERROR] run {i}: {e}', file=real_stdout)
            continue
        if p is None:
            continue

        sol = p.solutions.get(solver_name)
        if sol is None:
            continue

        costs.append(float(sol.cost))
        rhos.append(float(sol.rho))
        times.append(float(sol.solve_time))
        successes.append(1 if sol.rho >= 0 else 0)

        if hasattr(p, 'cost_list') and p.cost_list is not None:
            cost_lists.append([float(c) for c in p.cost_list])

        if traj_x is None:
            traj_x = sol.x.tolist() if sol.x is not None else None

    if not costs:
        tqdm.write(f'  [SKIP] No valid runs for Task {task_id} / {method_label}', file=real_stdout)
        return

    result = {
        'task'         : task_id,
        'method'       : method_label,
        'solver'       : solver_name,
        'n_runs'       : len(costs),
        'costs'        : costs,
        'rhos'         : rhos,
        'times'        : times,
        'successes'    : successes,
        'cost_mean'    : float(np.mean(costs)),
        'cost_std'     : float(np.std(costs)),
        'rho_mean'     : float(np.mean(rhos)),
        'rho_std'      : float(np.std(rhos)),
        'time_mean'    : float(np.mean(times)),
        'time_std'     : float(np.std(times)),
        'success_rate' : float(sum(successes) / len(costs)),
        'cost_lists'   : cost_lists,
        'traj_x'       : traj_x,
    }

    os.makedirs(out_dir, exist_ok=True)
    safe_name = solver_name.lower().replace('-', '_')
    out_path = os.path.join(out_dir, f'baseline_{safe_name}_task{task_id}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    tqdm.write(
        f'  Task {task_id} [{method_label}] | cost={result["cost_mean"]:.4f}±{result["cost_std"]:.4f} '
        f'| time={result["time_mean"]:.2f}s±{result["time_std"]:.2f} '
        f'| success={sum(successes)}/{len(costs)}',
        file=real_stdout
    )
    tqdm.write(f'  → saved {out_path}', file=real_stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks',   type=int,  nargs='+', default=[2, 4, 5, 6])
    parser.add_argument('--runs',    type=int,  default=100)
    parser.add_argument('--solvers', type=str,  nargs='+',
                        default=['PI', 'STL-GUIDED-PI', 'GRAD'],
                        choices=AVAILABLE_SOLVERS)
    parser.add_argument('--out-dir', type=str,  default='../results')
    args = parser.parse_args()

    for t in args.tasks:
        if t not in PROBLEM_MAP:
            print(f'[WARN] Task {t} not in PROBLEM_MAP, skipping.')
            continue
        for s in args.solvers:
            run_task_solver(t, s, args.runs, args.out_dir)


if __name__ == '__main__':
    main()
