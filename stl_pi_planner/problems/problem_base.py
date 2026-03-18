from abc import ABC, abstractmethod

import optuna
import promoting_pi_planner_c as stl_pi_planner_c
# import stl_guided_pi_planner_c
from tabulate import tabulate

from stl_pi_planner.common.common import convert_to_cpp_stltree
from stl_pi_planner.common.visualization import visualize_solution, visualize_pi_steps, save_table, \
    visualize_pi_steps_single_integrator
from stl_pi_planner.solvers.cma_es import CMA_ES
from stl_pi_planner.solvers.grad import Grad
from stl_pi_planner.solvers.mip import MIP
from stl_pi_planner.solvers.pi import PISolver
from stl_pi_planner.solvers.STL_guided_pi import STLGuidedPISolver
from stl_pi_planner.systems import DoubleIntegrator, SingleIntegrator


class SolutionData:
    def __init__(self, x, u, rho, cost, solve_time):
        self.x = x
        self.u = u
        self.rho = rho
        self.cost = cost
        self.solve_time = solve_time


class ProblemBase(ABC):
    def __init__(self, name: str):
        self.name = name
        self.H = None
        self.K = None
        self.k0 = None
        self.x0 = None
        self.solutions = {}

        self.init_scenario()
        self.sys, self.cpp_sys = self.init_system()
        self.spec, self.spec_syntax = self.init_specification()
        self.cpp_spec = convert_to_cpp_stltree(self.spec)

        # Cost function parameters
        self.Q = None
        self.R = None
        self.P = None
        self.gamma = None

        # Solver specific parameters
        self.tune_mip = False

        self.maxiter_grad = None
        self.ftol_grad = None
        self.eps_grad = None

        self.maxiter_sgrad = None
        self.ftol_sgrad = None
        self.eps_sgrad = None
        self.k_sgrad = None
        self.scaling_sgrad = None

        self.sigma_cmaes = None
        self.maxfevals_cmaes = None
        self.pop_size_cmaes = None
        self.noise_change_sigma_exponent_cmaes = None
        self.n_iter_cmaes = None

        self.n_iterations_pi = None
        self.n_samples_pi = None
        self.cov_pi = None
        self.lamb_pi = None
        self.nu_pi = None
        self.cost_threshold_pi = 13.5

        # Additional recordings for PI solver
        self.pi_record_y_opt = None
        self.pi_record_y = None
        self.pi_record_best_sample_idx = None

        self.robustness_cost_fct = 'max'    # 'max' or 'viol'

    @abstractmethod
    def init_scenario(self):
        """
        Initialize scenario
        """
        pass

    @abstractmethod
    def init_system(self):
        """
        Initialize system
        @return: Python system instance, C++ system instance
        """
        return None, None

    @abstractmethod
    def init_specification(self):
        """
        Initialize specification
        @return: STL specification
        """
        return None

    def init_mip_solver(self):
        """
        Initialize MIP solver
        """
        pass

    @abstractmethod
    def init_grad_solver(self):
        """
        Initialize GRAD solver
        """
        pass

    @abstractmethod
    def init_sgrad_solver(self):
        """
        Initialize SGRAD solver
        """
        pass

    @abstractmethod
    def init_cma_es_solver(self):
        """
        Initialize CMA-ES solver
        """
        pass

    @abstractmethod
    def init_pi_solver(self):
        """
        Initialize PI Solver
        """
        pass

    @abstractmethod
    def init_STL_guided_pi_solver(self):
        """
        Initialize STL-GUIDED-PI Solver
        """
        pass

    def mip_param_optimization(self, trial):
        """
        Parameters to be optimized for MIP solver
        @param trial: Optuna trial instance
        """
        pass

    def grad_param_optimization(self, trial):
        """
        Parameters to be optimized for GRAD solver
        @param trial: Optuna trial instance
        """
        pass

    def sgrad_param_optimization(self, trial):
        """
        Parameters to be optimized for SGRAD solver
        @param trial: Optuna trial instance
        """
        pass

    def cma_es_param_optimization(self, trial):
        """
        Parameters to be optimized for CMA-ES solver
        @param trial: Optuna trial instance
        """
        pass

    def pi_param_optimization(self, trial):
        """
        Parameters to be optimized for PI solver
        @param trial: Optuna trial instance
        """
        pass

    def param_objective(self, solver):
        """
        Objective function for parameter optimization
        @param solver: Used solver
        """
        pass

    @abstractmethod
    def plot_scenario(self, ax, k=0):
        """
        Plot scenario
        @param ax: axis
        @param k: time step
        """
        pass

    def optimize_parameters(self, solver: str):
        """
        Optimizes the hyperparameters
        @param solver: The solver for which the hyperparameters shall be optimized
        """

        # For mip we use the gurobi internal parameter tuner
        if solver == 'MIP':
            self.mip_param_optimization(None)
            self.solve_mip()
        else:
            def objective(trial):
                if solver == 'GRAD':
                    self.grad_param_optimization(trial)
                    self.solve_grad()
                elif solver == 'SGRAD':
                    self.sgrad_param_optimization(trial)
                    self.solve_sgrad()
                elif solver == 'CMA-ES':
                    self.cma_es_param_optimization(trial)
                    self.solve_cma_es()
                elif solver == 'PI':
                    self.pi_param_optimization(trial)
                    self.solve_pi()
                return self.param_objective(solver)

            # Create a study and run optimization
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=1e3, timeout=3600)

            # Iterate over all Pareto-optimal trials
            print('Best Configuration ----------------------------')
            # Print the best hyperparameters
            print(f"Best parameters: {study.best_params}")
            print(f"Best score: {study.best_value}")
            print('--------------------------------------------')

    def solve(self, solver: str, use_pi_cpp_implementation=False):
        """
        Solves the optimization problem
        @param use_pi_cpp_implementation: Flag for using C++ implementation of PI solver
        @param solver: The solver to be used
        """
        if solver == 'MIP':
            self.init_mip_solver()
            self.solve_mip()
        elif solver == 'GRAD':
            self.init_grad_solver()
            self.solve_grad()
        elif solver == 'SGRAD':
            self.init_sgrad_solver()
            self.solve_sgrad()
        elif solver == 'CMA-ES':
            self.init_cma_es_solver()
            self.solve_cma_es()
        elif solver == 'PI':
            self.init_pi_solver()
            self.solve_pi(use_pi_cpp_implementation)
        elif solver == 'STL-GUIDED-PI':
            self.init_STL_guided_pi_solver()
            self.solve_stl_guided_pi(use_pi_cpp_implementation)

        print('Optimal cost', self.solutions[solver].cost)
        print('Optimal rob ', self.solutions[solver].rho)
        print('Solver time ', self.solutions[solver].solve_time)
        return self.cost_list

    def solve_all(self):
        """
        Solves the optimization once with each solver
        """
        self.init_scenario()
        self.init_grad_solver()
        self.init_sgrad_solver()
        self.init_cma_es_solver()
        self.init_pi_solver()
        self.init_STL_guided_pi_solver()

        self.solve_mip()
        self.solve_grad()
        self.solve_sgrad()
        self.solve_cma_es()
        self.solve_pi()

    def solve_mip(self):
        """
        Solves the optimization problem with the MIP solver
        """
        solver = MIP(self.spec, self.sys, self.x0, self.K, self.Q, self.R, self.P, self.gamma, self.robustness_cost_fct,
                     tune=self.tune_mip, verbose=True)
        x, u, rho, cost, solve_time = solver.solve()
        self.solutions['MIP'] = SolutionData(x, u, rho, cost, solve_time)

    def solve_grad(self):
        """
        Solves the optimization problem with the GRAD solver
        """
        solver = Grad(self.spec, self.sys, self.x0, self.K, self.Q, self.R, self.P, self.gamma, self.robustness_cost_fct,
                      maxiter=self.maxiter_grad, ftol=self.ftol_grad, eps=self.eps_grad, verbose=True)
        x, u, rho, cost, solve_time = solver.solve()
        self.solutions['GRAD'] = SolutionData(x, u, rho, cost, solve_time)

    def solve_sgrad(self):
        """
        Solves the optimization problem with the SGRAD solver
        """
        solver = Grad(self.spec, self.sys, self.x0, self.K, self.Q, self.R, self.P, self.gamma, self.robustness_cost_fct,
                      smooth=True, scaling=self.scaling_sgrad, maxiter=self.maxiter_sgrad, ftol=self.ftol_sgrad,
                      eps=self.eps_sgrad, k_smooth=self.k_sgrad, verbose=True)
        x, u, rho, cost, solve_time = solver.solve()
        self.solutions['SGRAD'] = SolutionData(x, u, rho, cost, solve_time)

    def solve_cma_es(self):
        """
        Solves the optimization problem with the CMA-ES solver
        """
        solver = CMA_ES(self.spec, self.cpp_spec, self.sys, self.cpp_sys, self.x0, self.K, self.Q, self.R, self.P,
                        self.gamma, self.robustness_cost_fct, self.sigma_cmaes, self.maxfevals_cmaes,
                        self.pop_size_cmaes, self.noise_change_sigma_exponent_cmaes, self.n_iter_cmaes)
        x, u, rho, cost, solve_time = solver.solve()
        self.solutions['CMA-ES'] = SolutionData(x, u, rho, cost, solve_time)

    def solve_pi(self, use_pi_cpp_implementation=False):
        """
        Solves the optimization problem with the PI solver
        """
        if not use_pi_cpp_implementation:
            solver = PISolver(self.spec, self.sys, self.x0, self.K, self.n_samples_pi, self.cov_pi,
                              self.lamb_pi, self.nu_pi, self.n_iterations_pi, self.Q,
                              self.P, self.R, self.gamma, self.robustness_cost_fct)
        else:
            solver = stl_pi_planner_c.PISolver(self.cpp_spec, self.cpp_sys, self.x0, self.K, self.n_samples_pi,
                                               self.cov_pi, self.lamb_pi, self.nu_pi, self.n_iterations_pi,
                                               self.Q, self.P, self.R, self.gamma, self.robustness_cost_fct, True,
                                               True, True, self.cost_threshold_pi)

        x, u, rho, cost, solve_time, record_y_opt, record_y, record_best_sample_idx, cost_list = solver.solve()
        # x, u, rho, cost, solve_time, record_y_opt, record_y, record_best_sample_idx = solver.solve()
        self.solutions['PI'] = SolutionData(x, u, rho, cost, solve_time)

        self.pi_record_y_opt = record_y_opt
        self.pi_record_y = record_y
        self.pi_record_best_sample_idx = record_best_sample_idx
        self.cost_list = cost_list

    def solve_stl_guided_pi(self, use_pi_cpp_implementation=False):
        """
        Solves the optimization problem with the STL-GUIDED-PI solver
        """
        if not use_pi_cpp_implementation:
            solver = STLGuidedPISolver(self.spec, self.spec_syntax, self.sys, self.x0, self.K, self.n_samples_pi, self.cov_pi,
                                self.lamb_pi, self.nu_pi, self.n_iterations_pi, self.Q,
                                self.P, self.R, self.gamma, self.robustness_cost_fct)
        else:
            solver = stl_pi_planner_c.PISolver(self.cpp_spec, self.cpp_sys, self.x0, self.K, self.n_samples_pi,
                                               self.cov_pi, self.lamb_pi, self.nu_pi, self.n_iterations_pi,
                                               self.Q, self.P, self.R, self.gamma, self.robustness_cost_fct, True,
                                               True, True, self.cost_threshold_pi)
            stl_pi_planner_c.PISolver.set_use_stl_guided(solver, True)

        # x, u, rho, cost, solve_time, record_y_opt, record_y, record_best_sample_idx = solver.solve()
        x, u, rho, cost, solve_time, record_y_opt, record_y, record_best_sample_idx, cost_list = solver.solve()
        self.solutions['STL-GUIDED-PI'] = SolutionData(x, u, rho, cost, solve_time)

        self.pi_record_y_opt = record_y_opt
        self.pi_record_y = record_y
        self.pi_record_best_sample_idx = record_best_sample_idx
        self.cost_list = cost_list

    def visualize(self, solver: str, plot_all_time_steps=False):
        """
        Visualizes the solution of the optimization problem
        @param solver: The used solver
        @param plot_all_time_steps: Flag to indicate whether to plot all time steps
        """
        sol = self.solutions[solver]
        if sol.x is not None:
            visualize_solution(sol.x, sol.u, self.x0, self.K, self.name, self.plot_scenario, self.sys, solver, plot_all_time_steps)
        else:
            print(f'There is no solution for solver {solver}. Either the optimization failed or was not yet executed.')

    def visualize_pi_steps(self):
        """
        Visualizes the iterations of the solution from the PI solver
        """
        # sol = self.solutions['PI']
        if isinstance(self.sys, DoubleIntegrator):
            print('PI visualization not yet implemented for DoubleIntegrator system!')
        elif isinstance(self.sys, SingleIntegrator):
            visualize_pi_steps_single_integrator(self.name, self.K, self.plot_scenario, self.x0, self.n_samples_pi,
                                                 self.pi_record_y, self.pi_record_y_opt, self.pi_record_best_sample_idx)
        # elif sol.x is not None:
        elif self.pi_record_y is not None:
            visualize_pi_steps(self.name, self.K, self.plot_scenario, self.x0, self.n_samples_pi,
                               self.pi_record_y, self.pi_record_y_opt, self.pi_record_best_sample_idx)
        else:
            print('The optimization was not yet executed for PI solver')

    def print_solutions_details(self):
        """
        Print the details of the solutions
        """
        # Prepare data for tabulation
        table_data = [(key, sol.rho, sol.solve_time) for key, sol in self.solutions.items()]
        headers = ["Solver", "Opt Rho [-]", "Solver Time [s]"]

        # Print the tabulated data
        table = tabulate(table_data, headers=headers, tablefmt="github", floatfmt=".4f")

        save_table(table, self.name)
        print(table)
