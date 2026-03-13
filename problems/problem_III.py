import numpy as np
import stl_pi_planner_c

from stl_pi_planner.STL.predicate import CirclePredicate
from stl_pi_planner.problems.problem_base import ProblemBase
from stl_pi_planner.problems.common import make_circle_patch, outside_rectangle_formula, make_rectangle_patch, \
    inside_rectangle_formula
from stl_pi_planner.systems.nonlinear import Bicycle


class Problem_III(ProblemBase):
    def __init__(self):
        self.tasks = None
        self.obstacles = None

        super().__init__('Problem_III')

        self.robustness_cost_fct = 'viol'         # 'max' or 'viol'

        self.Q = 0 * np.diag([0, 0, 0, 0, 0])     # Quadratic state cost
        self.R = 100 * np.eye(2)                  # Quadratic input cost
        self.P = 1 * np.array([-1, -1, 0, 0, 0])  # Terminal cost
        self.gamma = 30

    def init_scenario(self):
        self.K = 50                                         # Time horizon
        self.x0 = np.array([1.0, 1.0, 0.0, 0.0, np.pi/4])   # Initial state
        self.tasks = [([2.5, 8.5], 1.25),                   # ([x, y], radius)
                      ([3.5, 6.75], 1.45),
                      ([8.5, 1.75], 1)
                      ]
        self.obstacles = [(-3, 1.5, 2, 6.5),                 # (xmin, xmax, ymin, ymax)
                          (2.25, 6.5, -3, 1.5),
                          (2.25, 5.5, 2.5, 5),
                          (5.5, 10, 7, 11),
                          (7.5, 9.25, 3.5, 5.5)
                          ]

    def init_system(self):
        dt = 0.5    # [s]
        l_wb = 0.25  # [m]
        # x:= [p_x, p_y, delta, v, psi], u:= [v_delta, a_long] y:= x
        return Bicycle(dt, l_wb), stl_pi_planner_c.Bicycle(dt, l_wb)

    def init_mip_solver(self):
        pass

    def init_grad_solver(self):
        self.maxiter_grad = 100000
        self.ftol_grad = 1e-6
        self.eps_grad = 1.49e-8

    def init_sgrad_solver(self):
        self.maxiter_sgrad = 100000
        self.ftol_sgrad = 1e-6
        self.eps_sgrad = 1.49e-8
        self.k_sgrad = 400
        self.scaling_sgrad = 100

    def init_cma_es_solver(self):
        self.sigma_cmaes =  0.022
        self.maxfevals_cmaes = 1000000
        self.pop_size_cmaes = 35
        self.noise_change_sigma_exponent_cmaes =  0.644
        self.n_iter_cmaes = 11240

    def init_pi_solver(self):
        alpha = 0.1
        # alpha = 1
        self.cov_pi = alpha * np.array([[0.02, 0],
                                        [0, 0.02]])
        self.lamb_pi = alpha * 2.0
        self.n_samples_pi = 81650
        self.nu_pi = 0.8
        # self.nu_pi = 0.9
        # self.n_iterations_pi = 40
        self.n_iterations_pi = 80

    def init_STL_guided_pi_solver(self):
        alpha = 0.1
        self.cov_pi = alpha * np.array([[0.02, 0],
                                        [0, 0.02]])
        self.lamb_pi = alpha * 2.0
        self.n_samples_pi = 81650
        self.nu_pi = 0.8
        self.n_iterations_pi = 40

    def mip_param_optimization(self, trial):
        self.tune_mip = True

    def grad_param_optimization(self, trial):
        self.maxiter_grad = trial.suggest_int('maxiter_grad', 100, 1000, log=True)
        self.ftol_grad = trial.suggest_float('ftol_grad', 1e-8, 1e-6, log=True)
        self.eps_grad = trial.suggest_float('eps_grad', 1e-8, 1e-6, log=True)

    def sgrad_param_optimization(self, trial):
        self.maxiter_sgrad = trial.suggest_int('maxiter_sgrad', 100, 1000, log=True)
        self.ftol_sgrad = trial.suggest_float('ftol_sgrad', 1e-8, 1e-6, log=True)
        self.eps_sgrad = trial.suggest_float('eps_sgrad', 1e-8, 1e-6, log=True)
        self.k_sgrad = trial.suggest_int('k_sgrad', 1, 1e5, log=True)
        self.scaling_sgrad = trial.suggest_int('scaling_sgrad', 10, 1000, log=True)

    def cma_es_param_optimization(self, trial):
        self.sigma_cmaes = trial.suggest_float('sigma_cmaes', 0.01, 0.5, log=True)
        self.maxfevals_cmaes = trial.suggest_int('maxfevals_cmaes', 1e6, 1e6)
        self.pop_size_cmaes = trial.suggest_int('pop_size_cmaes', 2, 50)
        self.noise_change_sigma_exponent_cmaes = trial.suggest_float('noise_change_sigma_exponent_cmaes', 0.1, 1)
        self.n_iter_cmaes = trial.suggest_int('n_iter_cmaes', 1e3, 1e5, log=True)

    def pi_param_optimization(self, trial):
        alpha = trial.suggest_float("alpha", 0.1, 3.0, step=0.1)
        self.cov_pi = alpha * np.array([[0.02, 0],
                                        [0, 0.02]])
        self.lamb_pi = alpha * 2.0
        self.n_samples_pi = trial.suggest_int("n_samples_pi", 10000, 100000, log=True)
        self.nu_pi = trial.suggest_float("nu_pi", 0.1, 1.0, step=0.1)
        self.n_iterations_pi = trial.suggest_int("n_iterations_pi", 2, 100)

    def param_objective(self, solver):
        sol = self.solutions[solver]
        cost = round(sol.cost, 3)
        elapsed_time = round(sol.solve_time, 3)
        theta = 0.1

        return cost + theta * elapsed_time

    def init_specification(self):
        # inside domain
        inside_domain = inside_rectangle_formula((0, 10, 0, 10), 0, 1, self.sys.p).always(0, self.K)

        # Always not at obstacles
        obstacle_formulas = []
        for obstacle_bounds in self.obstacles:
            not_at_obstacle = outside_rectangle_formula(obstacle_bounds, 0, 1, self.sys.p)
            obstacle_formulas.append(not_at_obstacle)

        not_at_obstacles = obstacle_formulas[0]
        for i in range(1, len(obstacle_formulas)):
            not_at_obstacles = not_at_obstacles & obstacle_formulas[i]

        always_not_at_obstacles = not_at_obstacles.always(0, self.K)

        # Tasks
        task_formulas = []
        for task in self.tasks:
            in_task_area = CirclePredicate(task[0], task[1], 0, 1, self.sys.p)
            task_formulas.append(in_task_area)

        do_tasks = (task_formulas[0].always(0, 1)).eventually(0, self.K)
        for i in range(1, len(task_formulas)):
            do_tasks = do_tasks & (task_formulas[i].always(0, 1)).eventually(0, self.K)

        # No parallel task execution
        not_tasks_parallel = (task_formulas[0].negation() | task_formulas[1].negation()).always(0, self.K)

        # Put everything together
        spec = inside_domain & always_not_at_obstacles & do_tasks & not_tasks_parallel

        return spec, None

    def plot_scenario(self, ax, k=0):
        for task in self.tasks:
            ax.add_patch(make_circle_patch(task[0], task[1], edgecolor='black', facecolor='skyblue', alpha=0.5))
        for obstacle in self.obstacles:
            ax.add_patch(make_rectangle_patch(*obstacle, edgecolor='black', facecolor='gray', alpha=1.0))

        # set the field of view
        ax.set_xlim((0, 10))
        ax.set_ylim((0, 10))
        ax.set_aspect('equal')
