import numpy as np
import promoting_pi_planner_c as stl_pi_planner_c

from stl_pi_planner.problems.problem_base import ProblemBase
from stl_pi_planner.problems.common import (
    inside_rectangle_formula,
    outside_rectangle_formula,
    make_rectangle_patch
)
from stl_pi_planner.systems.linear import PointMass

import rtamt


class Problem_VI(ProblemBase):
    def __init__(self):
        self.goal_bounds = None
        self.corridor_bounds = None
        self.wall_low_bounds = None
        self.wall_up_bounds = None

        super().__init__('Problem_VI')

        self.robustness_cost_fct = 'max'        # 'max' or 'viol'
        self.cost_threshold_pi = -1e9           # disable early stop — Task VI needs all iterations

        self.Q = 0 * np.diag([0, 0, 0, 0])      # Quadratic state cost
        self.R = 20 * np.eye(2)                 # Quadratic input cost
        self.P = 0 * np.array([0, 0, 0, 0])     # Terminal cost
        self.gamma = 10

    def init_scenario(self):
        self.K = 40
        self.x0 = np.array([1.0, 1.0, 0.0, 0.0])

        # final goal
        self.goal_bounds = (7.0, 8.5, 7.0, 8.5)

        # two rectangular buildings / walls
        self.wall_low_bounds = (3.0, 5.0, 0.0, 3.8)
        self.wall_up_bounds  = (3.0, 5.0, 4.2, 9.0)

        # corridor between the two walls
        # self.corridor_bounds = (3.5, 4.5, 3.0, 5.0)
        self.corridor_bounds = (3.0, 5.0, 3.8, 4.2)

        # time window for corridor traversal
        self.t_corridor_lb = 8
        self.t_corridor_ub = 30

    def init_system(self):
        dt = 0.5
        # x := [p_x, p_y, v_x, v_y], u := [a_x, a_y]
        return PointMass(dt), stl_pi_planner_c.PointMass(dt)

    def init_mip_solver(self):
        pass

    def init_grad_solver(self):
        self.maxiter_grad = 100
        self.ftol_grad = 2.2e-7
        self.eps_grad = 1.6e-7

    def init_sgrad_solver(self):
        self.maxiter_sgrad = 30
        self.ftol_sgrad = 6.7e-7
        self.eps_sgrad = 4.8e-7
        self.k_sgrad = 490
        self.scaling_sgrad = 11

    def init_cma_es_solver(self):
        self.sigma_cmaes = 0.037
        self.maxfevals_cmaes = 9300
        self.pop_size_cmaes = 13
        self.noise_change_sigma_exponent_cmaes = 0.555
        self.n_iter_cmaes = 773

    def init_pi_solver(self):
        alpha = 6.8
        self.cov_pi = alpha * np.array([[0.5, 0],
                                        [0, 0.5]])
        self.lamb_pi = alpha * 10.0
        self.n_samples_pi = 2000
        self.nu_pi = 0.85
        self.n_iterations_pi = 200

    def init_STL_guided_pi_solver(self):
        alpha = 6.8
        self.cov_pi = alpha * np.array([[0.5, 0],
                                        [0, 0.5]])
        self.lamb_pi = alpha * 10.0
        self.n_samples_pi = 1700
        self.nu_pi = 0.8
        self.n_iterations_pi = 100

    def mip_param_optimization(self, trial):
        self.tune_mip = True

    def grad_param_optimization(self, trial):
        self.maxiter_grad = trial.suggest_int('maxiter_grad', 10, 100, step=10)
        self.ftol_grad = trial.suggest_float('ftol_grad', 1e-8, 1e-6, log=True)
        self.eps_grad = trial.suggest_float('eps_grad', 1e-8, 1e-6, log=True)

    def sgrad_param_optimization(self, trial):
        self.maxiter_sgrad = trial.suggest_int('maxiter_sgrad', 10, 150, step=10)
        self.ftol_sgrad = trial.suggest_float('ftol_sgrad', 1e-8, 1e-6, log=True)
        self.eps_sgrad = trial.suggest_float('eps_sgrad', 1e-8, 1e-6, log=True)
        self.k_sgrad = trial.suggest_int('k_sgrad', 1, 500)
        self.scaling_sgrad = trial.suggest_int('scaling_sgrad', 10, 1000, log=True)

    def cma_es_param_optimization(self, trial):
        self.sigma_cmaes = trial.suggest_float('sigma_cmaes', 0.01, 0.5, log=True)
        self.maxfevals_cmaes = trial.suggest_int('maxfevals_cmaes', 1e2, 1e5, log=True)
        self.pop_size_cmaes = trial.suggest_int('pop_size_cmaes', 2, 1e2)
        self.noise_change_sigma_exponent_cmaes = trial.suggest_float('noise_change_sigma_exponent_cmaes', 0.01, 1)
        self.n_iter_cmaes = trial.suggest_int('n_iter_cmaes', 100, 1e5, log=True)

    def pi_param_optimization(self, trial):
        alpha = trial.suggest_float("alpha", 0.5, 10.0, step=0.1)
        self.cov_pi = alpha * np.array([[0.5, 0],
                                        [0, 0.5]])
        self.lamb_pi = alpha * 10.0
        self.n_samples_pi = trial.suggest_int("n_samples_pi", 100, 2000)
        self.nu_pi = trial.suggest_float("nu_pi", 0.1, 1.0, step=0.1)
        self.n_iterations_pi = trial.suggest_int("n_iterations_pi", 1, 100)

    def param_objective(self, solver):
        sol = self.solutions[solver]
        cost = round(sol.cost, 3)
        elapsed_time = round(sol.solve_time, 3)
        theta = 1.0
        return cost + theta * elapsed_time

    def init_specification(self):
        # safe with respect to rectangular walls
        wall_low_safe = outside_rectangle_formula(self.wall_low_bounds, 0, 1, self.sys.p)
        wall_up_safe  = outside_rectangle_formula(self.wall_up_bounds, 0, 1, self.sys.p)

        safe = wall_low_safe & wall_up_safe

        # corridor and final goal
        at_corridor = inside_rectangle_formula(self.corridor_bounds, 0, 1, self.sys.p)
        at_goal = inside_rectangle_formula(self.goal_bounds, 0, 1, self.sys.p)

        # STL specification
        spec = (
            safe.always(0, self.K)
            & at_corridor.eventually(self.t_corridor_lb, self.t_corridor_ub)
            & at_goal.eventually(self.t_corridor_ub, self.K)
        )

        # rtamt monitor
        monitor = rtamt.StlDiscreteTimeSpecification()
        monitor.name = 'PI Problem VI STL monitor'

        monitor.declare_var('px', 'float')
        monitor.declare_var('py', 'float')
        monitor.declare_var('out', 'float')

        K = self.K
        t1 = self.t_corridor_lb
        t2 = self.t_corridor_ub

        gx1, gx2, gy1, gy2 = self.goal_bounds
        cx1, cx2, cy1, cy2 = self.corridor_bounds
        wl_x1, wl_x2, wl_y1, wl_y2 = self.wall_low_bounds
        wu_x1, wu_x2, wu_y1, wu_y2 = self.wall_up_bounds

        wall_low_expr = (
            f"not (((px >= {wl_x1}) and (px <= {wl_x2}) and "
            f"(py >= {wl_y1}) and (py <= {wl_y2})))"
        )
        wall_up_expr = (
            f"not (((px >= {wu_x1}) and (px <= {wu_x2}) and "
            f"(py >= {wu_y1}) and (py <= {wu_y2})))"
        )
        corridor_expr = (
            f"((px >= {cx1}) and (px <= {cx2}) and "
            f"(py >= {cy1}) and (py <= {cy2}))"
        )
        goal_expr = (
            f"((px >= {gx1}) and (px <= {gx2}) and "
            f"(py >= {gy1}) and (py <= {gy2}))"
        )

        monitor.spec = (
            "out = ( "
            f"always[0,{K}](({wall_low_expr}) and ({wall_up_expr})) "
            "and "
            f"eventually[{t1},{t2}]({corridor_expr}) "
            "and "
            f"eventually[{t2},{K}]({goal_expr}) "
            ")"
        )

        print(monitor.spec)

        try:
            monitor.parse()
            monitor.pastify()
        except rtamt.RTAMTException as err:
            print('RTAMT parse error:', err)
            raise

        return spec, monitor

    def plot_scenario(self, ax, k=0):
        wall_low = make_rectangle_patch(
            *self.wall_low_bounds,
            edgecolor='black',
            facecolor='gray',
            alpha=1.0
        )
        wall_up = make_rectangle_patch(
            *self.wall_up_bounds,
            edgecolor='black',
            facecolor='gray',
            alpha=1.0
        )
        corridor = make_rectangle_patch(
            *self.corridor_bounds,
            edgecolor='green',
            facecolor='lightgreen',
            alpha=0.25
        )
        goal = make_rectangle_patch(
            *self.goal_bounds,
            edgecolor='black',
            facecolor='skyblue',
            alpha=0.5
        )

        ax.add_patch(wall_low)
        ax.add_patch(wall_up)
        ax.add_patch(corridor)
        ax.add_patch(goal)

        ax.set_xlim((0, 9))
        ax.set_ylim((0, 9))
        ax.set_aspect('equal')