from problems.problem_I import Problem_I
from problems.problem_II import Problem_II
from problems.problem_III import Problem_III


def main():

    problem = Problem_I()
    # problem = Problem_II()
    # problem = Problem_III()

    solvers = {0: 'MIP',
               1: 'GRAD',
               2: 'SGRAD',
               3: 'CMA-ES',
               4: 'PI'}

    selected_solver = 0

    problem.optimize_parameters(solvers[selected_solver])


if __name__ == '__main__':
    main()
