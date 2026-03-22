"""
Trace: run ONE episode capturing per-iteration trajectory for baseline PI methods.
Run from repo_origin/ directory.
"""
import sys, os, json, argparse, contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg')


@contextlib.contextmanager
def suppress_output():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = [os.dup(1), os.dup(2)]
    os.dup2(devnull,1); os.dup2(devnull,2)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull,'w')
    try: yield
    finally:
        sys.stdout.close()
        sys.stdout, sys.stderr = old_out, old_err
        os.dup2(saved[0],1); os.dup2(saved[1],2)
        os.close(saved[0]); os.close(saved[1]); os.close(devnull)


def trace_task(task_id, solver_name, out_dir):
    from problems.problem_II import Problem_II as P2
    from problems.problem_IV import Problem_IV as P4
    from problems.problem_V  import Problem_V  as P5
    from problems.problem_VI import Problem_VI as P6
    MAP = {2:P2, 4:P4, 5:P5, 6:P6}

    p = MAP[task_id]()
    p.cost_threshold_pi = -1e9
    use_cpp = solver_name in ('PI', 'STL-GUIDED-PI')
    try:
        with suppress_output():
            p.solve(solver_name, use_pi_cpp_implementation=use_cpp)
    except AttributeError as e:
        if 'cost_list' not in str(e): raise

    sol = p.solutions.get(solver_name)
    if sol is None:
        print(f'[trace] {solver_name} Task {task_id}: no solution'); return

    traj_per_iter = []
    if hasattr(p,'pi_record_y_opt') and p.pi_record_y_opt is not None:
        for y_opt in p.pi_record_y_opt:
            traj_per_iter.append(np.array(y_opt).tolist())

    safe = solver_name.lower().replace('-','_')
    result = {
        'task'          : task_id,
        'method'        : solver_name,
        'cost_list'     : [float(c) for c in p.cost_list] if hasattr(p,'cost_list') and p.cost_list else [],
        'final_cost'    : float(sol.cost),
        'final_rho'     : float(sol.rho),
        'traj_per_iter' : traj_per_iter,
        'final_x'       : sol.x.tolist() if sol.x is not None else None,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'trace_{safe}_task{task_id}.json')
    with open(path,'w') as f: json.dump(result, f)
    print(f'[trace] {solver_name} Task {task_id}: {len(traj_per_iter)} iters, rho={sol.rho:.3f} → {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks',    type=int, nargs='+', default=[2,4,5,6])
    parser.add_argument('--solvers',  nargs='+', default=['PI','STL-GUIDED-PI'])
    parser.add_argument('--out-dir',  default='../results')
    args = parser.parse_args()
    for t in args.tasks:
        for s in args.solvers:
            trace_task(t, s, args.out_dir)


if __name__ == '__main__':
    main()
