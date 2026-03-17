# These functions are take from the stlpy package: https://github.com/vincekurtz/stlpy/tree/main

from .nonlinear import NonlinearSystem
import numpy as np

class LinearSystem(NonlinearSystem):
    """
    A linear discrete-time system of the form

    .. math::

        x_{k+1} = A x_t + B u_t

        y_t = C x_t + D u_t

    where

        - :math:`x_t \in \mathbb{R}^n` is a system state,
        - :math:`u_t \in \mathbb{R}^m` is a control input,
        - :math:`y_t \in \mathbb{R}^p` is a system output.

    :param A: A ``(n,n)`` numpy array representing the state transition matrix
    :param B: A ``(n,m)`` numpy array representing the control input matrix
    :param C: A ``(p,n)`` numpy array representing the state output matrix
    :param D: A ``(p,m)`` numpy array representing the control output matrix
    """
    def __init__(self, A, B, C, D):
        self.n = A.shape[1]
        self.m = B.shape[1]
        self.p = C.shape[0]

        # Sanity checks on matrix sizes
        assert A.shape == (self.n, self.n), "A must be an (n,n) matrix"
        assert B.shape == (self.n, self.m), "B must be an (n,m) matrix"
        assert C.shape == (self.p, self.n), "C must be an (p,n) matrix"
        assert D.shape == (self.p, self.m), "D must be an (p,m) matrix"

        # Store dynamics parameters
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def dynamics_fcn(self, x, u):
        return self.A @ x + self.B @ u

    def output_fcn(self, x, u):
        return self.C @ x + self.D @ u


class PointMass(LinearSystem):
    """
    A linear system describing a point mass model in two dimensions

    .. math::

        A = \\begin{bmatrix} I_{d \\times d}  & I_{d \\times d} \\\ 0_{d \\times d} & I_{d \\times d}  \\end{bmatrix}
        \quad
        B = \\begin{bmatrix} 0_{d \\times d} \\\ I_{d \\times d}  \\end{bmatrix}

    .. math::
        C = \\begin{bmatrix} I_{2d \\times 2d} \\\ 0_{d \\times 2d} \\end{bmatrix}
        \quad
        D = \\begin{bmatrix} 0_{2d \\times d} \\\ I_{d \\times d} \\end{bmatrix}

    :param dt: Float describing the time increment
    """

    def __init__(self, dt: float):
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0.5 * dt ** 2, 0],
                      [0, 0.5 * dt ** 2],
                      [dt, 0],
                      [0, dt]])

        I = np.eye(2)
        z = np.zeros((2, 2))

        C = np.block([[I, z],
                      [z, I],
                      [z, z]])
        D = np.block([[z],
                      [z],
                      [I]])

        LinearSystem.__init__(self, A, B, C, D)


class DoubleIntegrator(LinearSystem):
    def __init__(self, dt: float):
        A = np.array([[1, dt],
                      [0, 1]])

        B = np.array([[0.5 * dt**2],
                      [dt]])

        C = np.array([[1, 0],
                      [0, 1],
                      [0, 0]])

        D = np.array([[0],
                      [0],
                      [1]])

        LinearSystem.__init__(self, A, B, C, D)


class SingleIntegrator(LinearSystem):
    def __init__(self, dt: float):
        A = np.array([[1]])

        B = np.array([[dt]])

        C = np.array([[1],
                      [0]])

        D = np.array([[0],
                      [1]])

        # Initialize the parent class with the system matrices
        LinearSystem.__init__(self, A, B, C, D)
