# Some of these functions are take from the stlpy package: https://github.com/vincekurtz/stlpy/tree/main

import numpy as np

from .formula import STLFormula


class NonlinearPredicate(STLFormula):
    """
    A nonlinear STL predicate:math:`\pi` defined by

    .. math::

        g(y_t) \geq 0

    where :math:`y_t \in \mathbb{R}^d` is the value of the signal
    at a given timestep :math:`k`, and :math:`g : \mathbb{R}^d \\to \mathbb{R}`.
    
    :param g:       A function mapping the signal at a given timestep to 
                    a scalar value. 
    :param d:       An integer expressing the dimension of the signal y.
    :param name:    (optional) a string used to identify this predicate.
    """
    def __init__(self, g, d, name=None):
        self.d = d
        self.name = name
        self.g = g

    def negation(self):
        if self.name is None:
            newname = None
        else:
            newname = "not " + self.name

        negative_g = lambda y : -self.g(y)
        return NonlinearPredicate(negative_g, self.d, name=newname)

    def robustness(self, y, k, smooth=False, k_smooth=100, scaling=1.0):
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert isinstance(k, int), "timestep k must be an integer"
        assert y.shape[0] == self.d, "y must be of shape (d,K)"
        assert y.shape[1] > k, "requested timestep %s, but y only has %s timesteps" % (k, y.shape[1])

        return np.array([self.g(y[:,k]) / scaling])

    def is_predicate(self):
        return True

    def is_state_formula(self):
        return True

    def is_disjunctive_state_formula(self):
        return True

    def is_conjunctive_state_formula(self):
        return True
    
    def get_all_inequalities(self):
        raise NotImplementedError("linear inequalities are not defined for nonlinear predicates")

    def __str__(self):
        if self.name is None:
            return "{ Nonlinear Predicate }"
        else:
            return "{ Predicate " + self.name + " }"


class CirclePredicate(STLFormula):
    """
    A circle STL predicate :math:`\pi` defined by

    .. math::

        r^2 - (y_x-c_x)^2 - (y_y-c_y)^2

    """
    def __init__(self, center, radius, y1_index, y2_index, d, negated=False, name=None):
        self.center = np.asarray(center)
        self.radius = radius
        self.y1_index = y1_index
        self.y2_index = y2_index

        # Store the dimensionality of y_t
        self.d = d

        # A unique string describing this predicate
        self.name = name

        self.negated = negated
        if negated:
            self.factor = -1.0
        else:
            self.factor = 1.0

    def negation(self):
        if self.name is None:
            newname = None
        else:
            newname = "not " + self.name
        return CirclePredicate(self.center, self.radius, self.y1_index, self.y2_index, self.d,
                               negated=True, name=newname)

    def robustness(self, y, k, smooth=False, k_smooth=100, scaling=1.0):
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert isinstance(k, int), "timestep k must be an integer"
        assert y.shape[0] == self.d, "y must be of shape (d,K)"
        assert y.shape[1] > k, "requested timestep %s, but y only has %s timesteps" % (k, y.shape[1])
        y1 = y[self.y1_index, k]
        y2 = y[self.y2_index, k]
        return self.factor/scaling * (self.radius**2 - (y1-self.center[0])**2 - (y2-self.center[1])**2)

    def is_predicate(self):
        return True

    def is_state_formula(self):
        return True

    def is_disjunctive_state_formula(self):
        return True

    def is_conjunctive_state_formula(self):
        return True

    def get_all_inequalities(self):
        raise NotImplementedError("linear inequalities are not defined for nonlinear predicates")

    def __str__(self):
        if self.name is None:
            return "{ Circle Predicate }"
        else:
            return "{ Predicate " + self.name + " }"


class DynamicCirclePredicate(STLFormula):
    """
    A dynamic circle STL predicate :math:`\pi` defined by

    .. math::

        r^2 - (y_x-c_x)^2 - (y_y-c_y)^2

    """
    def __init__(self, centers, radius, y1_index, y2_index, d, negated=False, name=None):
        self.centers = centers
        self.radius = radius
        self.y1_index = y1_index
        self.y2_index = y2_index

        # Store the dimensionality of y_t
        self.d = d

        # A unique string describing this predicate
        self.name = name

        self.negated = negated
        if negated:
            self.factor = -1.0
        else:
            self.factor = 1.0

    def negation(self):
        if self.name is None:
            newname = None
        else:
            newname = "not " + self.name
        return DynamicCirclePredicate(self.centers, self.radius, self.y1_index, self.y2_index, self.d, negated=True,
                                      name=newname)

    def robustness(self, y, k, smooth=False, k_smooth=100, scaling=1.0):
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert isinstance(k, int), "timestep k must be an integer"
        assert y.shape[0] == self.d, "y must be of shape (d,K)"
        assert y.shape[1] > k, "requested timestep %s, but y only has %s timesteps" % (k, y.shape[1])
        y1 = y[self.y1_index, k]
        y2 = y[self.y2_index, k]
        c1 = self.centers[k, 0]
        c2 = self.centers[k, 1]
        return np.array([self.factor/scaling * (self.radius**2 - (y1-c1)**2 - (y2-c2)**2)])

    def is_predicate(self):
        return True

    def is_state_formula(self):
        return True

    def is_disjunctive_state_formula(self):
        return True

    def is_conjunctive_state_formula(self):
        return True

    def get_all_inequalities(self):
        raise NotImplementedError("linear inequalities are not defined for nonlinear predicates")

    def __str__(self):
        if self.name is None:
            return "{ Circle Predicate }"
        else:
            return "{ Predicate " + self.name + " }"


class LinearPredicate(STLFormula):
    """
    A linear STL predicate :math:`\pi` defined by

    .. math::

        a^Ty_t - b \geq 0

    where :math:`y_t \in \mathbb{R}^d` is the value of the signal
    at a given timestep :math:`k`, :math:`a \in \mathbb{R}^d`,
    and :math:`b \in \mathbb{R}`.

    :param a:       a numpy array or list representing the vector :math:`a`
    :param b:       a list, numpy array, or scalar representing :math:`b`
    :param name:    (optional) a string used to identify this predicate.
    """
    def __init__(self, a, b, name=None):
        # Convert provided constraints to numpy arrays
        self.a = np.asarray(a).reshape((-1,1))
        self.b = np.atleast_1d(b)

        # Some dimension-related sanity checks
        assert (self.a.shape[1] == 1), "a must be of shape (d,1)"
        assert (self.b.shape == (1,)), "b must be of shape (1,)"

        # Store the dimensionality of y_t
        self.d = self.a.shape[0]

        # A unique string describing this predicate
        self.name = name

    def negation(self):
        if self.name is None:
            newname = None
        else:
            newname = "not " + self.name
        return LinearPredicate(-self.a, -self.b, name=newname)

    def robustness(self, y, k, smooth=False, k_smooth=100, scaling=1.0):
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert isinstance(k, int), "timestep k must be an integer"
        assert y.shape[0] == self.d, "y must be of shape (d,K)"
        assert y.shape[1] > k, "requested timestep %s, but y only has %s timesteps" % (k, y.shape[1])

        return (self.a.T@y[:, k] - self.b)[0] / scaling

    def is_predicate(self):
        return True

    def is_state_formula(self):
        return True

    def is_disjunctive_state_formula(self):
        return True

    def is_conjunctive_state_formula(self):
        return True

    def get_all_inequalities(self):
        A = -self.a.T
        b = -self.b
        return (A,b)

    def __str__(self):
        if self.name is None:
            return "{ Predicate %s*y >= %s }" % (self.a, self.b)
        else:
            return "{ Predicate " + self.name + " }"


class DynamicLinearPredicate(STLFormula):
    """
    A dynamic linear STL predicate :math:`\pi` defined by

    .. math::

        a^Ty_t - b \geq 0

    where :math:`y_t \in \mathbb{R}^d` is the value of the signal
    at a given timestep :math:`k`, :math:`a \in \mathbb{R}^d`,
    and :math:`b \in \mathbb{R}`.

    :param a:       a numpy array or list representing the vector :math:`a`
    :param b:       a list, numpy array, or scalar representing :math:`b`
    :param name:    (optional) a string used to identify this predicate.
    """
    def __init__(self, a, b, name=None):
        # Convert provided constraints to numpy arrays
        self.a = a
        self.b = b

        # Store the dimensionality of y_t
        self.d = self.a[:, 0].shape[0]

        # A unique string describing this predicate
        self.name = name

    def negation(self):
        if self.name is None:
            newname = None
        else:
            newname = "not " + self.name
        return DynamicLinearPredicate(-self.a, -self.b, name=newname)

    def robustness(self, y, k, smooth=False, k_smooth=100, scaling=1.0):
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert isinstance(k, int), "timestep k must be an integer"
        assert y.shape[0] == self.d, "y must be of shape (d,K)"
        assert y.shape[1] > k, "requested timestep %s, but y only has %s timesteps" % (k, y.shape[1])

        return (self.a[:, k].T@y[:, k] - self.b[k]) / scaling

    def is_predicate(self):
        return True

    def is_state_formula(self):
        return True

    def is_disjunctive_state_formula(self):
        return True

    def is_conjunctive_state_formula(self):
        return True

    def get_all_inequalities(self):
        A = -self.a.T
        b = -self.b
        return (A, b)

    def __str__(self):
        if self.name is None:
            return "{ Predicate %s*y >= %s }" % (self.a, self.b)
        else:
            return "{ Predicate " + self.name + " }"