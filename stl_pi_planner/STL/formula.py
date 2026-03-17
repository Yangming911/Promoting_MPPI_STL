# Some of these functions are take from the stlpy package: https://github.com/vincekurtz/stlpy/tree/main

from abc import ABC, abstractmethod

import numpy as np
from treelib import Tree


class STLFormula(ABC):
    """
    An abstract class which encompasses represents all kinds of STL formulas :math:`\\varphi`, including
    predicates (the simplest possible formulas) and standard formulas (made up of logical operations over
    predicates and other formulas).
    """
    @abstractmethod
    def robustness(self, y, k, smooth=False, k_smooth=100, scaling=1.0):
        """
        Compute the robustness measure :math:`\\rho^\\varphi(y,k)` of this formula for the
        given signal :math:`y = y_0,y_1,\\dots,y_T`, evaluated at timestep :math:`k`.

        :param y:    A ``(d,K)`` numpy array representing the signal
                     to evaluate, where ``d`` is the dimension of
                     the signal and ``K`` is the number of timesteps
        :param k:    The timestep :math:`k` to evaluate the signal at. This
                     is typically 0 for the full formula.

        :param smooth: If ``True``, then the smooth robustness measure is used.

        :return:    The robustness measure :math:`\\rho^\\varphi(y,k)` which is positive only
                    if the signal satisfies the specification.
        """
        pass

    @abstractmethod
    def is_predicate(self):
        """
        Indicate whether this formula is a predicate.

        :return:    A boolean which is ``True`` only if this is a predicate.
        """
        pass

    @abstractmethod
    def is_state_formula(self):
        """
        Indicate whether this formula is a state formula, e.g.,
        a predicate or the result of boolean operations over
        predicates.

        :return:    A boolean which is ``True`` only if this is a state formula.
        """
        pass

    @abstractmethod
    def is_disjunctive_state_formula(self):
        """
        Indicate whether this formula is a state formula defined by
        only disjunctions (or) over predicates.

        :return:     A boolean which is ``True`` only if this is a disjunctive state formula.
        """
        pass

    @abstractmethod
    def is_conjunctive_state_formula(self):
        """
        Indicate whether this formula is a state formula defined by
        only conjunctions (and) over predicates.

        :return:     A boolean which is ``True`` only if this is a conjunctive state formula.
        """
        pass

    @abstractmethod
    def get_all_inequalities(self):
        """
        Return all inequalities associated with this formula stacked into vector form

        .. math::

            Ay \le b

        where each row of :math:`A` and :math:`b` correspond to a predicate in this formula.

        .. note::

            This method is really only useful for conjunctive state formulas.

        :return A:  An (n,m) numpy array representing :math:`A`
        :return b:  An (n,) numpy array representing :math:`b`
        """
        pass

    @abstractmethod
    def negation(self):
        """
        Return a new :class:`.STLFormula` :math:`\\varphi_{new}` which represents
        the negation of this formula:

        .. math::

            \\varphi_{new} = \lnot \\varphi

        :return: An :class:`.STLFormula` representing :math:`\\varphi_{new}`

        .. note::

            For now, only formulas in positive normal form are supported. That means that negation
            (:math:`\lnot`) can only be applied to predicates (:math:`\\pi`).

        """
        pass

    def conjunction(self, other):
        """
        Return a new :class:`.STLTree` :math:`\\varphi_{new}` which represents the conjunction
        (and) of this formula (:math:`\\varphi`) and another one (:math:`\\varphi_{other}`):

        .. math::

            \\varphi_{new} = \\varphi \land \\varphi_{other}

        :param other:   The :class:`.STLFormula` :math:`\\varphi_{other}`

        :return: An :class:`.STLTree` representing :math:`\\varphi_{new}`

        .. note::

            Conjuction can also be represented with the ``&`` operator, i.e.,
            ::

                c = a & b

            is equivalent to
            ::

                c = a.conjuction(b)

        """
        return STLTree([self, other], "and", [0, 0])

    def __and__(self, other):
        """
        Syntatic sugar so we can write `one_and_two = one & two`
        """
        return self.conjunction(other)

    def disjunction(self, other):
        """
        Return a new :class:`.STLTree` :math:`\\varphi_{new}` which represents the disjunction
        (or) of this formula (:math:`\\varphi`) and another one (:math:`\\varphi_{other}`):

        .. math::

            \\varphi_{new} = \\varphi \lor \\varphi_{other}

        :param other:   The :class:`.STLFormula` :math:`\\varphi_{other}`

        :return: An :class:`.STLTree` representing :math:`\\varphi_{new}`

        .. note::

            Disjunction can also be represented with the ``|`` operator, i.e.,
            ::

                c = a | b

            is equivalent to
            ::

                c = a.disjunction(b)

        """
        return STLTree([self, other], "or", [0, 0])

    def __or__(self, other):
        """
        Syntatic sugar so we can write `one_or_two = one | two`
        """
        return self.disjunction(other)

    def always(self, k1, k2):
        """
        Return a new :class:`.STLTree` :math:`\\varphi_{new}` which ensures that this
        formula (:math:`\\varphi`) holds for all of the timesteps between
        :math:`k_1` and :math:`k_2`:

        .. math::

            \\varphi_{new} = G_{[k_1,k_2]}(\\varphi)


        :param k1:  An integer representing the delay :math:`k_1`
        :param k2:  An integer representing the deadline :math:`k_2`

        :return: An :class:`.STLTree` representing :math:`\\varphi_{new}`
        """
        time_interval = [k for k in range(k1, k2 + 1)]
        subformula_list = [self for k in time_interval]
        formula = STLTree(subformula_list, "and", time_interval)
        if self.name is not None:
            formula.name = "always [%s,%s] %s" % (k1, k2, self.name)
        return formula

    def historically(self, k1, k2):
        """
        Return a new :class:`.STLTree` :math:`\\varphi_{new}` which ensures that this
        formula (:math:`\\varphi`) holds for all of the timesteps between
        :math:`-k_2` and :math:`-k_1`:

        .. math::

            \\varphi_{new} = H_{[k_1,k_2]}(\\varphi)


        :param k1:  An integer representing the time :math:`k_1`
        :param k2:  An integer representing the time :math:`k_2`

        :return: An :class:`.STLTree` representing :math:`\\varphi_{new}`
        """
        time_interval = [k for k in range(-k2, -k1 + 1)]
        subformula_list = [self for k in time_interval]
        formula = STLTree(subformula_list, "and", time_interval)
        if self.name is not None:
            formula.name = "historically [%s,%s] %s" % (k1, k2, self.name)
        return formula

    def eventually(self, k1, k2):
        """
        Return a new :class:`.STLTree` :math:`\\varphi_{new}` which ensures that this
        formula (:math:`\\varphi`) holds for at least one timestep between
        :math:`k_1` and :math:`k_2`:

        .. math::

            \\varphi_{new} = F_{[k_1,k_2]}(\\varphi)


        :param k1:  An integer representing the delay :math:`k_1`
        :param k2:  An integer representing the deadline :math:`k_2`

        :return: An :class:`.STLTree` representing :math:`\\varphi_{new}`
        """
        time_interval = [k for k in range(k1, k2 + 1)]
        subformula_list = [self for k in time_interval]
        formula = STLTree(subformula_list, "or", time_interval)
        if self.name is not None:
            formula.name = "eventually [%s,%s] %s" % (k1, k2, self.name)
        return formula

    def once(self, k1, k2):
        """
        Return a new :class:`.STLTree` :math:`\\varphi_{new}` which ensures that this
        formula (:math:`\\varphi`) holds for at least one timestep between
        :math:`-k_2` and :math:`-k_1`:

        .. math::

            \\varphi_{new} = O_{[k_1,k_2]}(\\varphi)


        :param k1:  An integer representing the time :math:`k_1`
        :param k2:  An integer representing the time :math:`k_2`

        :return: An :class:`.STLTree` representing :math:`\\varphi_{new}`
        """
        time_interval = [k for k in range(-k2, -k1 + 1)]
        subformula_list = [self for k in time_interval]
        formula = STLTree(subformula_list, "or", time_interval)
        if self.name is not None:
            formula.name = "once [%s,%s] %s" % (k1, k2, self.name)
        return formula

    def until(self, other, k1, k2):
        """
        Return a new :class:`.STLTree` :math:`\\varphi_{new}` which ensures that the
        given formula :math:`\\varphi_{other}` holds for at least one timestep between
        :math:`k_1` and :math:`k_2`, and that this formula (:math:`\\varphi`) holds
        at all timesteps until then:

        .. math::

            \\varphi_{new} = \\varphi U_{[k_1,k_2]}(\\varphi_{other})

        :param other:   A :class:`.STLFormula` representing :math:`\\varphi_{other`
        :param k1:  An integer representing the delay :math:`k_1`
        :param k2:  An integer representing the deadline :math:`k_2`

        :return: An :class:`.STLTree` representing :math:`\\varphi_{new}`
        """
        # For every candidate swiching time (k_prime), construct a subformula
        # representing 'self' holding until k_prime, at which point 'other' holds.
        # TODO Check the inverals and the interval boundaries! E.g. is k_prime part of the interval or not
        self_until_k_prime = []

        for k_prime in range(k1, k2 + 1):
            time_interval = [k for k in range(k1, k_prime + 1)]
            subformula_list = [self for k in
                               range(0, k_prime)]  # NOTE: This describes now the official definition of until
            subformula_list.append(other)
            self_until_k_prime.append(STLTree(subformula_list, "and", time_interval))

        # Then we take the disjunction over each of these formulas
        return STLTree(self_until_k_prime, "or", [0 for i in range(len(self_until_k_prime))])

    def since(self, other, k1, k2):
        """
        Return a new :class:`.STLTree` :math:`\\varphi_{new}` which ensures that
        this formula (:math:`\\varphi`) holds for all time steps from the time step
        when the given formula :math:`\\varphi_{other}` holds at least one time step
        between :math:`k_1` and :math:`k_2`, until the current time step

        .. math::

            \\varphi_{new} = \\varphi S_{[k_1,k_2]}(\\varphi_{other})

        :param other:   A :class:`.STLFormula` representing :math:`\\varphi_{other`
        :param k1:  An integer representing the time :math:`k_1`
        :param k2:  An integer representing the time :math:`k_2`

        :return: An :class:`.STLTree` representing :math:`\\varphi_{new}`
        """
        # For every candidate swiching time (k_prime), construct a subformula
        # representing 'self' holding until k_prime, at which point 'other' holds.
        self_since_k_prime = []

        for k_prime in range(-k2, -k1 + 1):
            time_interval = [k for k in range(k_prime, 0 + 1)]
            subformula_list.append(other)
            subformula_list = [self for k in range(k_prime + 1, 0 + 1)]

            self_since_k_prime.append(STLTree(subformula_list, "and", time_interval))

        # Then we take the disjunction over each of these formulas
        return STLTree(self_since_k_prime, "or", [0 for i in range(len(self_since_k_prime))])

    def get_all_conjunctive_state_formulas(self):
        """
        Return a list of all of the (unique) conjunctive state
        formulas that make up this specification.

        :return:    A list of STLFormula objects
        """
        CSFs = []
        if self.is_conjunctive_state_formula():
            CSFs.append(self)
        else:
            for subformula in self.subformula_list:
                new_csf_list = subformula.get_all_conjunctive_state_formulas()

                # TODO: Deal with special case where there is only one formula in the
                # conjunction list: this comes up when the UNTIL operator is used

                # Only add those formulas that we don't already have to the list
                for csf in new_csf_list:
                    if not csf in CSFs:
                        CSFs.append(csf)

        return CSFs


class STLTree(STLFormula):
    """
    Describes an STL formula :math:`\\varphi` which is made up of
    operations over :class:`.STLFormula` objects. This defines a tree structure,
    so that, for example, the specification

    .. math::

        \\varphi = G_{[0,3]} \\pi_1 \land F_{[0,3]} \\pi_2

    is represented by the tree

    .. graphviz::

        digraph tree {
            root [label="phi"];
            G [label="G"];
            F [label="F"];
            n1 [label="pi_1"];
            n2 [label="pi_1"];
            n3 [label="pi_1"];
            n4 [label="pi_2"];
            n5 [label="pi_2"];
            n6 [label="pi_2"];

            root -> G;
            root -> F;
            G -> n1;
            G -> n2;
            G -> n3;
            F -> n4;
            F -> n5;
            F -> n6;
        }

    where each node is an :class:`.STLFormula` and the leaf nodes are :class:`.LinearPredicate` objects.


    Each :class:`.STLTree` is defined by a list of :class:`.STLFormula` objects
    (the child nodes in the tree) which are combined together using either conjunction or
    disjunction.

    :param subformula_list:     A list of :class:`.STLFormula` objects (formulas or
                                predicates) that we'll use to construct this formula.
    :param combination_type:    A string representing the type of operation we'll use
                                to combine the child nodes. Must be either ``"and"`` or ``"or"``.
    :param timesteps:           A list of timesteps that the subformulas must hold at.
                                This is needed to define the temporal operators.
    """

    def __init__(self, subformula_list, combination_type, timesteps, name=None):
        # Record the dimension of the signal this formula is defined over
        self.d = subformula_list[0].d

        # Run some type check on the inputs
        assert (combination_type == "and") or (combination_type == "or"), "Invalid combination type"
        assert isinstance(subformula_list, list), "subformula_list must be a list of STLTree or LinearPredicate objects"
        assert isinstance(timesteps, list), "timesteps must be a list of integers"
        assert len(timesteps) == len(subformula_list), "a timestep must be provided for each subformula"
        for formula in subformula_list:
            assert isinstance(formula, STLFormula), "subformula_list must be a list of STLTree or LinearPredicate objects"
            # assert formula.d == self.d, "all subformulas must be defined over same dimension of signal"  TODO: Otherwise problems when having different Predicate Domains
        for k in timesteps:
            assert isinstance(k, int), "each timestep must be an integer"

        # Simply save the input arguments. We will parse these recursively later on to
        # determine, for example, the formula robustness.
        self.subformula_list = subformula_list
        self.combination_type = combination_type
        self.timesteps = timesteps

        # Save the given name for pretty printing
        self.name = name

    def negation(self):
        raise NotImplementedError("Only formulas in positive normal form are supported at this time")

    def robustness(self, y, k, smooth=False, k_smooth=100, scaling=1.0):
        # The robustness values are bounded by K
        rob_values = np.array([formula.robustness(y, k + self.timesteps[i], smooth, k_smooth, scaling) for i, formula in
                               enumerate(self.subformula_list) if k + self.timesteps[i] < y.shape[1]])

        if len(rob_values) == 0:
            return np.nan

        rob_values = rob_values[~np.isnan(rob_values)]  # remove nans

        if smooth:
            if len(rob_values) == 1:
                return rob_values[0]
            if np.isinf(rob_values).any():
                print("Warning: Array contains inf or -inf values!")

            if self.combination_type == "and":
                k1 = k_smooth
                out = -(1 / k1) * np.log(np.sum(np.exp(-k1 * rob_values)))
                return out

            else:  # combination_type == "or"
                k2 = k_smooth
                out = np.sum(rob_values * np.exp(k2 * rob_values)) / (np.sum(np.exp(k2 * rob_values)))
                return out

        else:
            if self.combination_type == "and":
                return np.min(rob_values)
            else:  # combination_type == "or"
                return np.max(rob_values)

    def is_predicate(self):
        return False

    def is_state_formula(self):
        boolean_operation = all([self.timesteps[i] == self.timesteps[0] for i in range(len(self.timesteps))])
        children_are_state_formulas = all([subformula.is_state_formula() for subformula in self.subformula_list])

        return boolean_operation and children_are_state_formulas

    def is_disjunctive_state_formula(self):
        boolean_operation = all([self.timesteps[i] == self.timesteps[0] for i in range(len(self.timesteps))])
        children_match = all([s.is_disjunctive_state_formula() for s in self.subformula_list])

        return boolean_operation and children_match and self.combination_type == "or"

    def is_conjunctive_state_formula(self):
        boolean_operation = all([self.timesteps[i] == self.timesteps[0] for i in range(len(self.timesteps))])
        children_match = all([s.is_conjunctive_state_formula() for s in self.subformula_list])

        return boolean_operation and children_match and self.combination_type == "and"

    def simplify(self):
        """
        Modify this formula to reduce the depth of the formula tree while preserving
        logical equivalence.

        A shallower formula tree can result in a more efficient binary encoding in some
        cases.
        """
        mod = True
        while mod:
            # Just keep trying to flatten until we don't get any improvement
            mod = self.flatten(self)

    def flatten(self, formula):
        """
        Reduce the depth of the given :class:`STLFormula` by combining adjacent
        layers with the same logical operation. This preserves the meaning of the
        formula, since, for example,

        ..math::

            (a \land b) \land (c \land d) = a \land b \land c \land d

        :param formula: The formula to modify

        :return made_modification: boolean flag indicating whether the formula was changed.l

        """
        made_modification = False

        for subformula in formula.subformula_list:
            if subformula.is_predicate():
                pass
            else:
                if formula.combination_type == subformula.combination_type:
                    # Remove the subformula
                    i = formula.subformula_list.index(subformula)
                    formula.subformula_list.pop(i)
                    st = formula.timesteps.pop(i)

                    # Add all the subformula's subformulas instead
                    formula.subformula_list += subformula.subformula_list
                    formula.timesteps += [k + st for k in subformula.timesteps]
                    made_modification = True

                made_modification = self.flatten(subformula) or made_modification

        return made_modification

    def get_all_inequalities(self):
        As = []
        bs = []
        for subformula in self.subformula_list:
            A, b = subformula.get_all_inequalities()
            As.append(A)
            bs.append(b)
        A = np.vstack(As)
        b = np.hstack(bs)

        return A, b

    def __str__(self):
        """
        Return a string representing this formula. This string displays
        the tree structure of the formula, where each node represents either
        a conjuction or disjuction of subformulas, and leaves are state formulas.
        """
        tree = Tree()
        root = tree.create_node(self.combination_type)

        for subformula in self.subformula_list:
            self._add_subformula_to_tree(tree, root, subformula)

        return tree.__str__()

    def _add_subformula_to_tree(self, tree, root, formula):
        """
        Helper function for recursively parsing subformulas to create
        a Tree object for visualizing this formula.
        """
        if formula.is_predicate():
            tree.create_node(formula.__str__(), parent=root)
        else:
            new_node = tree.create_node(formula.combination_type, parent=root)
            for subformula in formula.subformula_list:
                self._add_subformula_to_tree(tree, new_node, subformula)
