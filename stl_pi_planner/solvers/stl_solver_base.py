from abc import ABC, abstractmethod

class STLSolverBase(ABC):
    """
    A simple abstract base class defining a common solver interface
    for different optimization-based STL optimization methods.
    """

    def __init__(self, spec, sys, x0, K, verbose):
        # Store the relevant data
        self.sys = sys
        self.spec = spec
        self.x0 = x0
        self.K = K+1  # needed to be consistent with how we've defined STLFormula
        self.verbose = verbose

    @abstractmethod
    def solve(self):
        """
        Solves the optimization problem
        """
        pass

