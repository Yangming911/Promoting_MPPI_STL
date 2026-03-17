from problems.problem_I import Problem_I
from problems.problem_II import Problem_II
from problems.problem_III import Problem_III
from problems.problem_IV import Problem_IV
from problems.problem_V import Problem_V

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import os
import sys
import contextlib
from datetime import datetime
from tqdm import tqdm


@contextlib.contextmanager
def redirect_to_log(log_path):
    """Redirect Python + C++ stdout/stderr (fd1, fd2) to log file."""
    with open(log_path, 'a') as log_file:
        fd = log_file.fileno()
        old_fd1 = os.dup(1)
        old_fd2 = os.dup(2)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = log_file
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            os.dup2(old_fd1, 1)
            os.dup2(old_fd2, 2)
            os.close(old_fd1)
            os.close(old_fd2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1, help='Number of runs to average over')
    parser.add_argument('--task', type=int, default=2, choices=[1, 2, 3, 4, 5], help='Task to run (default: 2)')
    args = parser.parse_args()

    task_map = {1: Problem_I, 2: Problem_II, 3: Problem_III, 4: Problem_IV, 5: Problem_V}
    ProblemClass = task_map[args.task]

    solvers = {0: 'MIP',
               1: 'GRAD',
               2: 'SGRAD',
               3: 'CMA-ES',
               4: 'PI',
               5: 'STL-GUIDED-PI'
               }

    selected_solver = 4

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S') + '.log')
    # Save real terminal stdout before any redirection
    real_stdout = sys.stdout

    tqdm.write('Task %d | runs=%d | log -> %s' % (args.task, args.runs, log_path), file=real_stdout)

    if args.runs == 1:
        problem = ProblemClass()
        with redirect_to_log(log_path):
            problem.solve(solvers[selected_solver], use_pi_cpp_implementation=True)
            problem.visualize(solvers[selected_solver], plot_all_time_steps=False)
        sol = problem.solutions[solvers[selected_solver]]
        tqdm.write('Cost: %.4f  rho: %.4f  Time: %.2fs  %s' % (
            sol.cost, sol.rho, sol.solve_time, 'OK' if sol.rho >= 0 else 'FAIL'), file=real_stdout)
    else:
        costs, rhos, times, successes = [], [], [], []
        with open(log_path, 'a') as lf:
            lf.write('Task %d | solver=%s | runs=%d\n' % (args.task, solvers[selected_solver], args.runs))
        for i in tqdm(range(args.runs), desc='Task %d' % args.task, file=real_stdout):
            with redirect_to_log(log_path):
                p = ProblemClass()
                p.solve(solvers[selected_solver], use_pi_cpp_implementation=True)
            sol = p.solutions[solvers[selected_solver]]
            costs.append(sol.cost)
            rhos.append(sol.rho)
            times.append(sol.solve_time)
            successes.append(1 if sol.rho >= 0 else 0)
            with open(log_path, 'a') as lf:
                lf.write('Run %d: cost=%.4f rho=%.4f time=%.2fs %s\n' % (
                    i+1, sol.cost, sol.rho, sol.solve_time, 'OK' if sol.rho >= 0 else 'FAIL'))
        tqdm.write('\n=== Average over %d runs ===' % args.runs, file=real_stdout)
        tqdm.write('Cost:        %.4f  (std=%.4f)' % (np.mean(costs),  np.std(costs)), file=real_stdout)
        tqdm.write('rho:         %.4f  (std=%.4f)' % (np.mean(rhos),   np.std(rhos)),  file=real_stdout)
        tqdm.write('SolveTime:   %.2fs (std=%.2f)'  % (np.mean(times),  np.std(times)), file=real_stdout)
        tqdm.write('SuccessRate: %d/%d'              % (sum(successes),  args.runs),     file=real_stdout)

    # # Plot cost_list if available
    # if hasattr(problem, 'cost_list') and problem.cost_list is not None:
    #     plt.figure(figsize=(8, 4.5))
    #     plt.plot(problem.cost_list, '-o', label=f'{solvers[selected_solver]}')
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Cost')
    #     plt.title('Cost over Iterations')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()

    # if solvers[selected_solver] in ['PI', 'STL-GUIDED-PI']:
    #     problem.visualize_pi_steps()

    # 直接按各自索引画在同一张图上（简单）
    # x1 = np.arange(len(cost_list1))
    # x2 = np.arange(len(cost_list2))

    # plt.figure(figsize=(8, 4.5))
    # plt.plot(x1, cost_list1, '-o', label=f'{solvers[4]} (len={len(cost_list1)})')
    # plt.plot(x2, cost_list2, '-x', label=f'{solvers[5]} (len={len(cost_list2)})')
    # plt.xlabel('Iteration')
    # plt.ylabel('Cost')
    # plt.title('Cost comparison')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()