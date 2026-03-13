import stl_pi_planner_c

from stl_pi_planner.STL import LinearPredicate
from stl_pi_planner.STL.predicate import CirclePredicate, DynamicCirclePredicate, DynamicLinearPredicate


def convert_to_cpp_stltree(py_stltree):
    """
    Convert the python stl tree instance to a C++ stl tree instance
    @param py_stltree: Python stl tree instance
    @return: C++ stl tree instance
    """
    cpp_stl_tree = stl_pi_planner_c.STLTree(py_stltree.combination_type, py_stltree.timesteps)
    for sub_formula in py_stltree.subformula_list:
        if isinstance(sub_formula, LinearPredicate):
            cpp_lin_pred = stl_pi_planner_c.LinearPredicate(sub_formula.a, sub_formula.b)
            cpp_stl_tree.add_subformula(cpp_lin_pred)

        elif isinstance(sub_formula, DynamicLinearPredicate):
            cpp_dyn_lin_pred = stl_pi_planner_c.DynamicLinearPredicate(sub_formula.a, sub_formula.b)
            cpp_stl_tree.add_subformula(cpp_dyn_lin_pred)

        elif isinstance(sub_formula, CirclePredicate):
            cpp_circ_pred = stl_pi_planner_c.CirclePredicate(sub_formula.radius, sub_formula.center[0], sub_formula.center[1],
                                                     sub_formula.negated)
            cpp_stl_tree.add_subformula(cpp_circ_pred)

        elif isinstance(sub_formula, DynamicCirclePredicate):
            cpp_dyn_circ_pred = stl_pi_planner_c.DynamicCirclePredicate(sub_formula.radius, sub_formula.centers, sub_formula.negated)
            cpp_stl_tree.add_subformula(cpp_dyn_circ_pred)

        else:
            cpp_sub_formula = convert_to_cpp_stltree(sub_formula)
            cpp_stl_tree.add_subformula(cpp_sub_formula)
    return cpp_stl_tree
