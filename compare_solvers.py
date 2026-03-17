from problems.problem_I import Problem_I
from problems.problem_II import Problem_II
from problems.problem_III import Problem_III
from stl_pi_planner.common.visualization import visualize_solution_comparison


def main():
    problem = Problem_I()
    # problem = Problem_II()
    # problem = Problem_III()

    # Solve problem with all solvers
    problem.solve_all()
    problem.print_solutions_details()

    # Make some visualizations
    visualize_solution_comparison(problem.solutions, problem.K, problem.name, problem.plot_scenario)
    problem.visualize_pi_steps()

if __name__ == '__main__':
    main()
