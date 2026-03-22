"""
Capture per-iteration ESS and entropy (H) from C++ stdout for PI-FixedPsi baseline.
"""
import sys, os, json, argparse, tempfile, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def capture_entropy_trace(task_id, out_dir):
    from problems.problem_II import Problem_II as P2
    from problems.problem_IV import Problem_IV as P4
    from problems.problem_V  import Problem_V  as P5
    from problems.problem_VI import Problem_VI as P6
    MAP = {2:P2, 4:P4, 5:P5, 6:P6}

    p = MAP[task_id]()
    p.cost_threshold_pi = -1e9

    tmpf = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
    tmpf.close()
    real_stdout_fd = os.dup(1)
    tmp_fd = os.open(tmpf.name, os.O_WRONLY | os.O_TRUNC)
    os.dup2(tmp_fd, 1)
    os.close(tmp_fd)

    old_err = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        p.solve('PI', use_pi_cpp_implementation=True)
    except AttributeError as e:
        if 'cost_list' not in str(e):
            raise
    finally:
        sys.stderr.close()
        sys.stderr = old_err
        os.dup2(real_stdout_fd, 1)
        os.close(real_stdout_fd)

    with open(tmpf.name) as f:
        lines = f.readlines()
    os.unlink(tmpf.name)

    ess_per_iter  = []
    h_per_iter    = []
    perp_per_iter = []
    wmax_per_iter = []

    re_ess  = re.compile(r'ESS:\s*([\d.eE+-]+)\s*/\s*([\d.eE+-]+)')
    re_diag = re.compile(r'H\(w\)=([\d.eE+-]+)\s+perp=([\d.eE+-]+)\s+w_max=([\d.eE+-]+)')

    for line in lines:
        m = re_ess.search(line)
        if m:
            ess_per_iter.append(float(m.group(1)))
        m2 = re_diag.search(line)
        if m2:
            h_per_iter.append(float(m2.group(1)))
            perp_per_iter.append(float(m2.group(2)))
            wmax_per_iter.append(float(m2.group(3)))

    sol = p.solutions.get('PI')
    result = {
        'task'          : task_id,
        'method'        : 'PI-FixedPsi',
        'ess_per_iter'  : ess_per_iter,
        'h_per_iter'    : h_per_iter,
        'perp_per_iter' : perp_per_iter,
        'wmax_per_iter' : wmax_per_iter,
        'cost_list'     : [float(c) for c in p.cost_list] if hasattr(p,'cost_list') and p.cost_list else [],
        'final_cost'    : float(sol.cost) if sol else float('nan'),
        'final_rho'     : float(sol.rho) if sol else float('nan'),
        'n_samples'     : p.n_samples_pi,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'entropy_trace_pi_task{task_id}.json')
    with open(path,'w') as f:
        json.dump(result, f)
    print(f'[entropy-trace] PI-FixedPsi Task {task_id}: {len(ess_per_iter)} ESS, '
          f'{len(h_per_iter)} H entries → {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks',   type=int, nargs='+', default=[2, 4])
    parser.add_argument('--out-dir', default='../results')
    args = parser.parse_args()
    for t in args.tasks:
        capture_entropy_trace(t, args.out_dir)


if __name__ == '__main__':
    main()
