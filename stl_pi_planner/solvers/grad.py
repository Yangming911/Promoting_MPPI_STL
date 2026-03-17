# Some functions are take from the stlpy package: https://github.com/vincekurtz/stlpy/tree/main


import time
import warnings

import numpy as np
from scipy.optimize import minimize

from stl_pi_planner.solvers.stl_solver_base import STLSolverBase


class Grad(STLSolverBase):
    """
    Gradient-based Solver
    """
    def __init__(self, spec, sys, x0, K, Q, R, P, gamma, robustness_cost_fct, method="SLSQP", smooth=False, scaling = 1.0,
                 maxiter=100, ftol=0.01, eps=0.01, k_smooth=100, verbose=True):
        super().__init__(spec, sys, x0, K, verbose)
        self.Q = Q
        self.R = R
        self.P = P
        self.K = K
        self.gamma = gamma
        self.robustness_cost_fct = robustness_cost_fct
        self.method = method
        self.smooth = smooth
        self.k_smooth = k_smooth
        self.scaling = scaling

        # SLSQP parameters
        self.maxiter = maxiter
        self.ftol = ftol
        self.eps = eps

        # suppress warnings  (required when smooth robustness calculation is used)
        warnings.filterwarnings('ignore')

    def solve(self):
        """
        Solves the optimization problem using numerical optimization.

        Initializes a control input guess, minimizes the cost function, computes the state trajectory, and evaluates
        the robustness measure.

        @return: A tuple (x, u, rho, cost, solve_time), where:
            - x: State trajectory array, or None if optimization failed.
            - u: Control input array, or None if optimization failed.
            - rho: Robustness measure, or negative infinity if optimization failed.
            - cost: Final value of the cost function.
            - solve_time: Time taken to solve the optimization problem.
        """

        # Set an initial guess
        np.random.seed(0)  # for reproducibility
        u_guess = np.random.uniform(-0.1, 0.1,(self.sys.m, self.K+1))

        # Solve
        start_time = time.time()
        res = minimize(self.cost, u_guess.flatten(), method=self.method, options = {'maxiter': self.maxiter,
                                                                                    'ftol': self.ftol,
                                                                                    'eps': self.eps,
                                                                                    'disp': self.verbose})
        solve_time = time.time() - start_time

        if self.verbose:
            print(res.message)

        if res.success:
            u = res.x.reshape((self.sys.m, self.K+1))
            x, y = self.forward_rollout(u)

            rho = self.spec.robustness(y, 0, self.smooth, self.k_smooth, self.scaling)
            cost = self.cost(u.flatten())

            if np.isnan(rho):
                rho = -float('inf')

        else:
            x = None
            u = None
            rho = -np.inf
            cost = np.inf
            print('Optimization failed')

        return x, u, rho, cost, solve_time

    def forward_rollout(self, u):
        """
        Performs a forward integration of the system dynamics
        @param u: Input trajectory
        @return: Tuple of state trajectory and output trajectory
        """
        x = np.full((self.sys.n, self.K+1),np.nan)
        y = np.full((self.sys.p, self.K+1),np.nan)

        x[:, 0] = self.x0

        for k in range(self.K):
            x[:,k+1] = self.sys.f(x[:,k], u[:,k])
            y[:,k] = self.sys.g(x[:,k], u[:,k])

        y[:, self.K] = self.sys.g(x[:, self.K], u[:, self.K])

        return x, y

    def cost(self, u_flat):
        """
        The cost function
        @param u_flat: The input trajectory flattened (aka as vector)
        @return: Cost value
        """
        cost = 0
        u = u_flat.reshape((self.sys.m, self.K+1))

        # Do a forward rollout to compute the state and output trajectories
        x, y = self.forward_rollout(u)

        # Add additional state and control costs
        for k in range(self.K):
            cost += x[:, k].T@self.Q@x[:, k] + 0.5 * u[:, k].T @ self.R @ u[:, k]

        # Add linear end costs
        cost += self.P @ x[:, self.K]

        if self.robustness_cost_fct == "viol":
            cost += self.gamma * self.scaling * -min(0, self.spec.robustness(y, 0, self.smooth, self.k_smooth, self.scaling))
        else:
            cost += self.gamma * self.scaling * -self.spec.robustness(y, 0, self.smooth, self.k_smooth, self.scaling)
        return cost
