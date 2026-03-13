import numpy as np
import stl_pi_planner_c

from stl_pi_planner.STL import LinearPredicate
from stl_pi_planner.problems.problem_base import ProblemBase
from stl_pi_planner.systems import SingleIntegrator

from stl_pi_planner.STL.syntax_tree import STLNode, STLSyntaxTree


class Problem_I(ProblemBase):
    def __init__(self):
        self.gate = None
        super().__init__('Problem_I')

        self.robustness_cost_fct = 'viol'       # 'max' or 'viol'

        self.Q = 0 * np.diag([1])           # Quadratic state cost
        self.R = 2 * np.eye(1)              # Quadratic input cost
        self.P = 2 * np.array([-1])         # Terminal cost
        self.gamma = 5

    def init_scenario(self):
        self.K = 10                         # Time horizon
        self.x0 = np.array([5.0])           # Initial state
        self.gate = 1.0                     # Gate value

    def init_system(self):
        dt = 1.0 # [s]
        # x:=[pos, v], u:=[a], y:= x
        return SingleIntegrator(dt), stl_pi_planner_c.SingleIntegrator(dt)

    def init_mip_solver(self):
        pass

    def init_grad_solver(self):
        self.maxiter_grad = 100
        self.ftol_grad = 7.0e-5
        self.eps_grad = 4.7e-5

    def init_sgrad_solver(self):
        self.maxiter_sgrad = 100
        self.ftol_sgrad =  3.4e-6
        self.eps_sgrad = 1.5e-5
        self.k_sgrad = 186
        self.scaling_sgrad = 10

    def init_cma_es_solver(self):
        self.sigma_cmaes = 0.03
        self.maxfevals_cmaes = 4060
        self.pop_size_cmaes = 17
        self.noise_change_sigma_exponent_cmaes = 0.887
        self.n_iter_cmaes = 2140

    def init_pi_solver(self):
        alpha = 2.8
        self.cov_pi = alpha * np.array([[2.]])
        self.lamb_pi = alpha * 4.0
        self.n_samples_pi = 955
        self.nu_pi = 0.3
        self.n_iterations_pi = 19

    def init_STL_guided_pi_solver(self):
        alpha = 2.8
        self.cov_pi = alpha * np.array([[2.]])
        self.lamb_pi = alpha * 4.0
        self.n_samples_pi = 955
        self.nu_pi = 0.3
        # self.nu_pi = 0.5
        self.n_iterations_pi = 19

    def mip_param_optimization(self, trial):
        self.tune_mip = True

    def grad_param_optimization(self, trial):
        self.maxiter_grad = trial.suggest_int('maxiter_grad', 100, 100, step=10)
        self.ftol_grad = trial.suggest_float('ftol_grad', 1e-7, 1e-4, log=True)
        self.eps_grad = trial.suggest_float('eps_grad', 1e-7, 1e-4, log=True)

    def sgrad_param_optimization(self, trial):
        self.maxiter_sgrad = trial.suggest_int('maxiter_sgrad', 100, 100, step=10)
        self.ftol_sgrad = trial.suggest_float('ftol_sgrad', 1e-7, 1e-4, log=True)
        self.eps_sgrad = trial.suggest_float('eps_sgrad', 1e-7, 1e-4, log=True)
        self.k_sgrad = trial.suggest_int('k_sgrad', 1, 200)
        self.scaling_sgrad = trial.suggest_int('scaling_sgrad', 10, 1000, log=True)

    def cma_es_param_optimization(self, trial):
        self.sigma_cmaes = trial.suggest_float('sigma_cmaes', 0.01, 0.5, log=True)
        self.maxfevals_cmaes = trial.suggest_int('maxfevals_cmaes', 1, 1e4, log=True)
        self.pop_size_cmaes = trial.suggest_int('pop_size_cmaes', 2, 1e2)
        self.noise_change_sigma_exponent_cmaes = trial.suggest_float('noise_change_sigma_exponent_cmaes', 0.01, 1)
        self.n_iter_cmaes = trial.suggest_int('n_iter_cmaes', 100, 1e5, log=True)

    def pi_param_optimization(self, trial):
        alpha = trial.suggest_float("alpha", 0.5, 10.0, step=0.1)
        self.cov_pi = alpha * np.array([[2.0]])
        self.lamb_pi = alpha * 4.0

        self.n_samples_pi = trial.suggest_int("n_samples_pi", 100, 2000)
        self.nu_pi = trial.suggest_float("nu_pi", 0.1, 1.0, step=0.1)
        self.n_iterations_pi = trial.suggest_int("n_iterations_pi", 1, 50)

    def param_objective(self, solver):
        sol = self.solutions[solver]
        cost = round(sol.cost, 3)
        elapsed_time = round(sol.solve_time, 3)
        theta = 0.1

        return cost + theta * elapsed_time

    def init_specification(self):
        a = np.zeros((1, self.sys.p))
        a[:, 0] = 1
        gate = LinearPredicate(-a, -self.gate)

        spec = (gate & gate.eventually(1, self.K)).eventually(0, self.K)

        # 把 predicate 包成 STLNode
        gate_syntax = STLSyntaxTree.from_predicate(gate, name="y <= 1")
        # 构造公式：□[0,a]( ¬(y>0) ∨ ◇[b,c](x>0) )
        spec_syntax = (gate_syntax & gate_syntax.eventually((1, self.K))).eventually((0, self.K))
        spec_syntax = STLSyntaxTree(spec_syntax)
        # print(spec)
        return spec, spec_syntax

    def plot_scenario(self, ax, k=0):
        # set the field of view
        ax.set_xlim((0, 10))
        ax.set_ylim((0, 10))
        ax.set_aspect('equal')
