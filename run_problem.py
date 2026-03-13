from problems.problem_I import Problem_I
from problems.problem_II import Problem_II
from problems.problem_III import Problem_III
from problems.problem_IV import Problem_IV
from problems.problem_V import Problem_V

import matplotlib.pyplot as plt
import numpy as np
import time

def main():

    # problem = Problem_I()
    problem = Problem_II()
    # problem = Problem_III()
    # problem = Problem_IV()
    # problem = Problem_V()

    solvers = {0: 'MIP',
               1: 'GRAD',
               2: 'SGRAD',
               3: 'CMA-ES',
               4: 'PI',
               5: 'STL-GUIDED-PI'
               }

    selected_solver = 4
    # selected_solver = 4

    # t1 = time.time()
    # for i in range(100):
    #     problem.solve(solvers[selected_solver], use_pi_cpp_implementation=True)
    # t2 = time.time()
    # print(f"Solved in {(t2 - t1)/100} seconds.")
    problem.solve(solvers[selected_solver], use_pi_cpp_implementation=True)
    # cost_list1 = problem.solve(solvers[4], use_pi_cpp_implementation=False)
    # cost_list2 = problem.solve(solvers[5], use_pi_cpp_implementation=False)
    # problem.solve(solvers[selected_solver], use_pi_cpp_implementation=True)
    # problem.visualize(solvers[selected_solver], plot_all_time_steps=False)
    problem.visualize(solvers[selected_solver], plot_all_time_steps=False)

    # # Plot cost_list if available
    # if hasattr(problem, 'cost_list') and problem.cost_list is not None:
    #     plt.figure(figsize=(8, 4.5))
    #     plt.plot(problem.cost_list, '-o', label=f'{solvers[selected_solver]}')
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Cost')
    #     plt.title('Cost over Iterations')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()

    # if solvers[selected_solver] in ['PI', 'STL-GUIDED-PI']:
    #     problem.visualize_pi_steps()

    # 直接按各自索引画在同一张图上（简单）
    # x1 = np.arange(len(cost_list1))
    # x2 = np.arange(len(cost_list2))

    # plt.figure(figsize=(8, 4.5))
    # plt.plot(x1, cost_list1, '-o', label=f'{solvers[4]} (len={len(cost_list1)})')
    # plt.plot(x2, cost_list2, '-x', label=f'{solvers[5]} (len={len(cost_list2)})')
    # plt.xlabel('Iteration')
    # plt.ylabel('Cost')
    # plt.title('Cost comparison')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()