# Some functions are take from the stlpy package: https://github.com/vincekurtz/stlpy/tree/main

import os

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from stl_pi_planner.STL import LinearPredicate, NonlinearPredicate
from stl_pi_planner.STL.predicate import DynamicLinearPredicate, CirclePredicate
from stl_pi_planner.solvers.stl_solver_base import STLSolverBase
from stl_pi_planner.systems import LinearSystem
from stl_pi_planner.systems.nonlinear import Bicycle


class MIP(STLSolverBase):
    """
    MIP Solver
    """

    def __init__(self, spec, sys, x0, K, Q, R, P, gamma, robustness_cost_fct, M=1000, tune=False, presolve=False,
                 verbose=True):
        assert M > 0, "M should be a (large) positive scalar"
        super().__init__(spec, sys, x0, K, verbose)

        self.Q = Q
        self.R = R
        self.P = P

        self.M = float(M)
        self.gamma = gamma
        self.robustness_cost_fct = robustness_cost_fct
        self.presolve = presolve
        self.tune = tune

        # Set up the optimization problem
        self.model = gp.Model("STL_MIP")
        
        # Store the cost function, which will added to self.model right before solving
        self.cost = 0.0

        if not self.verbose:
            self.model.setParam('OutputFlag', 0)

        self.model.setParam('TuneTimeLimit', 3600)
        self.model.setParam('TimeLimit', 3600)

        # Create optimization variables
        self.y = self.model.addMVar((self.sys.p, self.K), lb=-float('inf'), name='y')
        self.x = self.model.addMVar((self.sys.n, self.K), lb=-float('inf'), name='x')
        self.u = self.model.addMVar((self.sys.m, self.K), lb=-float('inf'), name='u')
        self.rho = self.model.addMVar(1,name="rho", lb=-float('inf'))

        # Add cost and constraints to the optimization problem
        if isinstance(self.sys, Bicycle):
            self.add_bicycle_dynamics_constraints()
        elif isinstance(self.sys, LinearSystem):
            self.add_dynamics_constraints()
        else:
            raise NotImplementedError("Only Bicycle Model and Linear System are supported!")

        self.add_stl_constraints()
        self.add_robustness_cost()
        self.add_quadratic_cost(self.Q, self.R)
        self.add_terminal_cost(self.P)

    def add_quadratic_cost(self, Q, R):
        """
        Add the quatratic cost function to the optimization problem
        @param Q: The Q matrix
        @param R: The R matrix
        """
        for k in range(self.K-1):
            self.cost += self.x[:,k] @ Q @ self.x[:,k] + 0.5 * self.u[:,k] @ R @self.u[:,k]

    def add_terminal_cost(self, P):
        """
        Add the terminal cost function to the optimization problem
        @param P: The P matrix
        """
        self.cost += P @ self.x[:, self.K-1]

    def add_robustness_cost(self):
        """
        Add the robustness cost function to the optimization problem
        """
        if self.robustness_cost_fct == 'max':
            self.cost += self.gamma * -self.rho  # TODO Has gamma any effect?
        else: # 'viol'
            # Introduce a new auxiliary variable to represent min(0, rho)
            self.min_rho = self.model.addVar(name="min_rho")

            # Add constraints to define min_rho as min(0, rho)
            self.model.addConstr(self.min_rho <= 0, name="min_rho_leq_0")
            self.model.addConstr(self.min_rho <= self.rho, name="min_rho_leq_rho")

            # Update the cost function by subtracting min_rho
            self.cost += self.gamma * -self.min_rho

    def solve(self):
        """
        Solves the optimization problem and returns the results.

        Sets the objective function, optionally tunes the model parameters, runs the optimization, and retrieves the
        solution variables and statistics.

        @return: A tuple (x, u, rho, obj_val, runtime), where:
            - x: State trajectory array or None if optimization failed.
            - u: Control input array or None if optimization failed.
            - rho: Dual variables array or negative infinity if optimization failed.
            - obj_val: Objective value of the solution.
            - runtime: Total runtime of the optimization.
        """

        # Set the cost function now, right before we solve. This is needed since model.setObjective resets the cost.
        self.model.setObjective(self.cost, GRB.MINIMIZE)

        if self.tune:
            print('########### Start tuning')
            self.model.tune()
            if self.model.tuneResultCount > 0:
                self.model.getTuneResult(0)

                # Save the tuned parameters
                script_dir = os.path.dirname(os.path.abspath(__file__))
                param_file_path = os.path.join(script_dir, 'tuned_params_gurobi.prm')
                self.model.write(param_file_path)

        print('########### Start optimizing')
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            if self.verbose:
                print("\nOptimal Solution Found!\n")
            x = self.x.X
            u = self.u.X
            rho = self.rho.X

        else:
            if self.verbose:
                print(f"\nOptimization failed with status {self.model.status}.\n")
            x = None
            u = None
            rho = -np.inf

        return x, u, rho, self.model.ObjVal, self.model.Runtime

    def add_dynamics_constraints(self):
        """
        Adds the system dynamic as constraints to the model
        """
        # Initial condition
        self.model.addConstr( self.x[:,0] == self.x0 )

        # Dynamics
        for k in range(self.K-1):
            self.model.addConstr(
                    self.x[:,k+1] == self.sys.A@self.x[:,k] + self.sys.B@self.u[:,k] )

            self.model.addConstr(
                    self.y[:,k] == self.sys.C@self.x[:,k] + self.sys.D@self.u[:,k] )

        self.model.addConstr(
                self.y[:,self.K-1] == self.sys.C@self.x[:,self.K-1] + self.sys.D@self.u[:,self.K-1] )

    def add_bicycle_dynamics_constraints(self):
        """
        Adds the system dynamic of the bicycle model as constraints to the model
        """

        # Initial condition
        self.model.addConstr(self.x[:, 0] == self.x0)

        # Dynamics
        for k in range(self.K-1):
            delta = self.x[2, k]
            v = self.x[3, k]
            psi = self.x[4, k]
            v_delta = self.u[0, k]
            a_long = self.u[1, k]

            # Define auxiliary variables for trigonometric and tangent functions
            sin_psi = self.model.addVar(lb=-1, ub=1, name="sin_psi")
            cos_psi = self.model.addVar(lb=-1, ub=1, name="cos_psi")
            tan_delta = self.model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="tan_delta")

            # Add general function constraints for sin, cos, and tan
            self.model.addGenConstrSin(psi, sin_psi, name=f"sin_psi_{k}")
            self.model.addGenConstrCos(psi, cos_psi, name=f"cos_psi_{k}")
            self.model.addGenConstrTan(delta, tan_delta, name=f"tan_delta_{k}")

            self.model.addConstr(self.x[0, k + 1] == self.x[0, k] + self.sys.dt * v * cos_psi)
            self.model.addConstr(self.x[1, k + 1] == self.x[1, k] + self.sys.dt * v * sin_psi)
            self.model.addConstr(self.x[2, k + 1] == self.x[2, k] + self.sys.dt * v_delta)
            self.model.addConstr(self.x[3, k + 1] == self.x[3, k] + self.sys.dt * a_long)
            self.model.addConstr(self.x[4, k + 1] == self.x[4, k] + self.sys.dt * v / self.sys.l_wb * tan_delta)

            self.model.addConstr(
                    self.y[:, k] == self.x[:, k])

        self.model.addConstr(
                self.y[:, self.K-1] == self.x[:, self.K-1])

    def add_stl_constraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Recursively traverse the tree defined by the specification
        # to add binary variables and constraints that ensure that
        # rho is the robustness value
        z_spec = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
        self.add_subformula_constraints(self.spec, z_spec, 0)
        self.model.addConstr( z_spec == 1 )

    def add_subformula_constraints(self, formula, z, k):
        """
        Given an STLFormula (formula) and a binary variable (z),
        add constraints to the optimization problem such that z
        takes value 1 only if the formula is satisfied (at time k).

        If the formula is a predicate, this constraint uses the "big-M"
        formulation

            A[x(k);u(k)] - b + (1-z)M >= 0,

        which enforces A[x;u] - b >= 0 if z=1, where (A,b) are the
        linear constraints associated with this predicate.

        If the formula is not a predicate, we recursively traverse the
        subformulas associated with this formula, adding new binary
        variables z_i for each subformula and constraining

            z <= z_i  for all i

        if the subformulas are combined with conjunction (i.e. all
        subformulas must hold), or otherwise constraining

            z <= sum(z_i)

        if the subformulas are combined with disjuction (at least one
        subformula must hold).
        """
        # We're at the bottom of the tree, so add the big-M constraints
        if isinstance(formula, LinearPredicate):
            # a.T*y - b + (1-z)*M >= rho
            self.model.addConstr( formula.a.T@self.y[:,k] - formula.b + (1-z)*self.M >= self.rho )

            # Force z to be binary
            b = self.model.addMVar(1,vtype=GRB.BINARY)
            self.model.addConstr(z == b)

        elif isinstance(formula, DynamicLinearPredicate):
            # a.T*y - b + (1-z)*M >= rho
            self.model.addConstr(formula.a[:, k].T @ self.y[:, k] - formula.b[k] + (1 - z) * self.M >= self.rho)

            # Force z to be binary
            b = self.model.addMVar(1, vtype=GRB.BINARY)
            self.model.addConstr(z == b)

        elif isinstance(formula, CirclePredicate):
            # a.T*y - b + (1-z)*M >= rho
            self.model.addConstr(formula.factor * (formula.radius**2
                                                   - (self.y[0, k]-formula.center[0]) * (self.y[0, k]-formula.center[0])
                                                   - (self.y[1, k]-formula.center[1]) * (self.y[1, k]-formula.center[1]))
                                                   + (1 - z) * self.M >= self.rho)

            # Force z to be binary
            b = self.model.addMVar(1, vtype=GRB.BINARY)
            self.model.addConstr(z == b)
        
        elif isinstance(formula, NonlinearPredicate):
            raise TypeError("Mixed integer programming does not support nonlinear predicates")

        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            if formula.combination_type == "and":
                for i, subformula in enumerate(formula.subformula_list):
                    k_sub = formula.timesteps[i]   # the timestep at which this formula
                                                   # should hold
                    if k+k_sub >= self.K:
                        continue
                    z_sub = self.model.addMVar(1, vtype=GRB.CONTINUOUS)
                    self.add_subformula_constraints(subformula, z_sub, k + k_sub)
                    self.model.addConstr( z <= z_sub )

            else:  # combination_type == "or":
                z_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    k_sub = formula.timesteps[i]
                    if k+k_sub >= self.K:
                        continue
                    z_sub = self.model.addMVar(1, vtype=GRB.CONTINUOUS)
                    z_subs.append(z_sub)
                    self.add_subformula_constraints(subformula, z_sub, k + k_sub)
                self.model.addConstr( z <= sum(z_subs) )
