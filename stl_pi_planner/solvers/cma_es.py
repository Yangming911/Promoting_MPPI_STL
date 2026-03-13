
import time

import numpy as np
import stl_pi_planner_c
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from stl_pi_planner.solvers.stl_solver_base import STLSolverBase


# Pymoo Optimization Problem
class PymooProblem(Problem):
    def __init__(self, cost_fct, n_var, xl, xu):
        self.cost_fct = cost_fct
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.apply_along_axis(self.cost_fct, 1, x)


class CMA_ES(STLSolverBase):
    """
    CMA-ES Solver
    """
    def __init__(self, spec, spec_cpp, sys, sys_cpp, x0, K, Q, R, P, gamma, robustness_cost_fct='max',
                 sigma=0.1, maxfevals=1e6, pop_size=100, noise_change_sigma_exponent=1, n_iter=1000, verbose=True):
        super().__init__(spec, sys, x0, K, verbose)
        self.Q = Q
        self.R = R
        self.P = P
        self.gamma = gamma
        self.robustness_cost_fct = robustness_cost_fct

        self.spec_cpp = spec_cpp
        self.sys_cpp = sys_cpp

        # Parameters
        self.sigma = sigma
        self.maxfevals = maxfevals
        self.pop_size = pop_size
        self.noise_change_sigma_exponent = noise_change_sigma_exponent
        self.n_iter = n_iter

        self.xl = -1e6
        self.xu = 1e6

    def solve(self):
        """
        Solves the optimization problem using the CMA-ES algorithm.

        Initializes the cost function and problem, runs the optimization, and returns the solution variables and
        statistics.

        @return: A tuple (x, u, rho, cost, solve_time), where:
            - x: State trajectory array.
            - u: Control input array.
            - rho: Robustness measure.
            - cost: Final value of the cost function.
            - solve_time: Time taken to solve the optimization problem.
        """
        cost_fct = stl_pi_planner_c.RobCostFunction(self.spec_cpp, self.sys_cpp, self.x0, self.K, self.Q, self.P,
                                                    self.R, self.gamma, self.robustness_cost_fct, True)

        problem = PymooProblem(cost_fct.evaluate, self.sys.m * self.K, self.xl, self.xu)

        # Run the optimization
        start_time = time.time()
        algorithm = CMAES(sigma=self.sigma, maxfevals=self.maxfevals, pop_size=self.pop_size,
                          noise_change_sigma_exponent=self.noise_change_sigma_exponent)
        res = minimize(problem, algorithm=algorithm,
                       termination = ('n_iter', self.n_iter),
                       #termination=('time', 3600),
                       verbose=self.verbose)
        solve_time = time.time() - start_time

        if self.verbose:
            print("Solve Time: ", solve_time)

        u = res.X.reshape((self.sys.m, self.K))

        x, y = cost_fct.forward_rollout(u)
        rho = self.spec_cpp.robustness(y, 0)
        cost = cost_fct.evaluate(res.X)

        return x, u, rho, cost, solve_time
