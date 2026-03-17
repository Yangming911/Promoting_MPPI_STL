import numpy as np
import promoting_pi_planner_c as stl_pi_planner_c

from stl_pi_planner.STL.predicate import CirclePredicate
from stl_pi_planner.problems.problem_base import ProblemBase
from stl_pi_planner.problems.common import inside_rectangle_formula, make_rectangle_patch, \
    make_circle_patch
from stl_pi_planner.systems.linear import PointMass

from stl_pi_planner.STL.syntax_tree import STLNode, STLSyntaxTree

import rtamt

class Problem_II(ProblemBase):
    def __init__(self):
        self.goal_bounds = None
        self.obstacle_center = None
        self.obstacle_radius = None

        super().__init__('Problem_II')

        self.robustness_cost_fct = 'max'        # 'max' or 'viol'

        self.Q = 0 * np.diag([0, 0, 0, 0])      # Quadratic state cost
        self.R = 20 * np.eye(2)                 # Quadratic input cost
        self.P = 0 * np.array([0, 0, 0, 0])     # Terminal cost
        self.gamma = 10
        # self.gamma = 20

    def init_scenario(self):
        self.K = 15                                 # Time horizon
        self.x0 = np.array([1.0, 1.0, 0.0, 0.0])    # Initial state
        self.goal_bounds = (6, 8, 6, 8)             # (x_min, x_max, y_min, y_max)
        self.obstacle_center = (4, 4)               # obstacle center and radius
        self.obstacle_radius = 1.5                  # obstacle radius

    def init_system(self):
        dt = 0.5  # [s]
        # x:=[p_x, p_y, v_x, v_y], u:=[a_x, a_y], y:= [p_x, p_y, v_x, v_y, a_x, a_y]
        return PointMass(dt), stl_pi_planner_c.PointMass(dt)

    def init_mip_solver(self):
        pass

    def init_grad_solver(self):
        self.maxiter_grad = 100
        self.ftol_grad = 2.2e-7
        self.eps_grad = 1.6e-7

    def init_sgrad_solver(self):
        self.maxiter_sgrad = 30
        self.ftol_sgrad = 6.7e-7
        self.eps_sgrad = 4.8e-7
        self.k_sgrad = 490
        self.scaling_sgrad = 11

    def init_cma_es_solver(self):
        self.sigma_cmaes =  0.037
        self.maxfevals_cmaes = 9300
        self.pop_size_cmaes = 13
        self.noise_change_sigma_exponent_cmaes =  0.555
        self.n_iter_cmaes = 773

    def init_pi_solver(self):
        # alpha = 6.8
        alpha = 1
        self.cov_pi = alpha * np.array([[0.5, 0],
                                        [0, 0.5]])
        self.lamb_pi = alpha * 10.0
        # self.n_samples_pi = 1140
        self.n_samples_pi = 1700
        # self.n_samples_pi = 50000
        # self.nu_pi = 0.9
        self.nu_pi = 0.8
        # self.nu_pi = 0.7
        # self.n_iterations_pi = 75
        self.n_iterations_pi = 100

    def init_STL_guided_pi_solver(self):
        alpha = 6.8
        # alpha = 1
        self.cov_pi = alpha * np.array([[0.5, 0],
                                        [0, 0.5]])
        self.lamb_pi = alpha * 10.0
        # self.n_samples_pi = 1140
        self.n_samples_pi = 1700
        self.nu_pi = 0.8
        # self.nu_pi = 0.9
        # self.n_iterations_pi = 75
        self.n_iterations_pi = 100

    def mip_param_optimization(self, trial):
        self.tune_mip = True

    def grad_param_optimization(self, trial):
        self.maxiter_grad = trial.suggest_int('maxiter_grad', 10, 100, step=10)
        self.ftol_grad = trial.suggest_float('ftol_grad', 1e-8, 1e-6, log=True)
        self.eps_grad = trial.suggest_float('eps_grad', 1e-8, 1e-6, log=True)

    def sgrad_param_optimization(self, trial):
        self.maxiter_sgrad = trial.suggest_int('maxiter_sgrad', 10, 150, step=10)
        self.ftol_sgrad = trial.suggest_float('ftol_sgrad', 1e-8, 1e-6, log=True)
        self.eps_sgrad = trial.suggest_float('eps_sgrad', 1e-8, 1e-6, log=True)
        self.k_sgrad = trial.suggest_int('k_sgrad', 1, 500)
        self.scaling_sgrad = trial.suggest_int('scaling_sgrad', 10, 1000, log=True)

    def cma_es_param_optimization(self, trial):
        self.sigma_cmaes = trial.suggest_float('sigma_cmaes', 0.01, 0.5, log=True)
        self.maxfevals_cmaes = trial.suggest_int('maxfevals_cmaes', 1e2, 1e5, log=True)
        self.pop_size_cmaes = trial.suggest_int('pop_size_cmaes', 2, 1e2)
        self.noise_change_sigma_exponent_cmaes = trial.suggest_float('noise_change_sigma_exponent_cmaes', 0.01, 1)
        self.n_iter_cmaes = trial.suggest_int('n_iter_cmaes', 100, 1e5, log=True)

    def pi_param_optimization(self, trial):
        alpha = trial.suggest_float("alpha", 0.5, 10.0, step=0.1)
        self.cov_pi = alpha * np.array([[0.5, 0],
                                          [0, 0.5]])
        self.lamb_pi = alpha * 10.0
        self.n_samples_pi = trial.suggest_int("n_samples_pi", 100, 2000)
        self.nu_pi = trial.suggest_float("nu_pi", 0.1, 1.0, step=0.1)
        self.n_iterations_pi = trial.suggest_int("n_iterations_pi", 1, 100)

    def param_objective(self, solver):
        sol = self.solutions[solver]
        cost = round(sol.cost, 3)
        elapsed_time = round(sol.solve_time, 3)
        theta = 1.0

        return cost + theta * elapsed_time

    def init_specification(self):
        # Goal Reaching
        at_goal = inside_rectangle_formula(self.goal_bounds, 0, 1, self.sys.p)

        # Obstacle Avoidance
        at_obstacle = CirclePredicate(self.obstacle_center, self.obstacle_radius, 0, 1, self.sys.p)
        not_at_obstacle = at_obstacle.negation()

        # Put all the constraints together in one specification
        spec = not_at_obstacle.always(0, self.K) & at_goal.eventually(0, self.K)

        # at_goal_syntax = STLSyntaxTree.from_predicate(at_goal, name="at_goal")
        # not_at_obstacle_syntax = STLSyntaxTree.from_predicate(not_at_obstacle, name="not_at_obstacle")
        # spec_syntax = not_at_obstacle_syntax.always((0, self.K)) & at_goal_syntax.eventually((0, self.K))
        # spec_syntax = STLSyntaxTree(spec_syntax)

        # -------- 2. 新增 rtamt 的在线监控规格 --------
        # 离散时间规格：时间索引就是 0,1,2,...,K
        monitor = rtamt.StlDiscreteTimeSpecification()
        monitor.name = 'PI Problem II STL monitor'

        # 声明信号变量：这里假设你在线监控时会喂给监视器 x,y 位置
        monitor.declare_var('px', 'float')
        monitor.declare_var('py', 'float')
        monitor.declare_var('dist2', 'float')
        monitor.declare_var('out', 'float')  # 输出鲁棒度

        # 把一些常数先在 Python 里算好，避免在 STL 里面出现 "*"
        K = self.K
        x_min, x_max, y_min, y_max = self.goal_bounds
        cx, cy = self.obstacle_center
        r2 = self.obstacle_radius ** 2
        print(f"Obstacle center: ({cx}, {cy}), radius^2: {r2}")
        print(x_min, x_max, y_min, y_max)

        # 规范：
        #   1) always[0:K-1] 远离障碍物：(px-cx)^2 + (py-cy)^2 >= r^2
        #   2) eventually[0:K-1] 落在目标矩形内
        #
        # 注意：
        #   - 区间端点 [0:K-1] 里只放整数，不要写 K*dt 之类
        #   - 逻辑运算用 and / or
        #   - 末尾加分号 ';'
        # monitor.spec = (
        #     "out = ( "
        #     f"always[0,{K}]( dist2 >= {r2} ) "
        #     "and "
        #     f"eventually[0,{K}]( (px >= {x_min}) and (px <= {x_max}) "
        #     f"and (py >= {y_min}) and (py <= {y_max}) ) "
        #     ")"
        # )
        monitor.spec = (
            "out = ( "
            f"always[0,{K}]( dist2 >= {r2} ) "
            ")"
        )

        print(monitor.spec)

        try:
            monitor.parse()
            # 对离散时间在线监控，官方推荐再 pastify 一下
            monitor.pastify()
        except rtamt.RTAMTException as err:
            print('RTAMT parse error:', err)
            raise

        # 返回两个东西：
        #   1) 原来框架用的 spec（stlpy）
        #   2) 用于在线监控的 rtamt monitor（你可以在 STL_guided_pi 里用它替换 self.spec_syntax）
        return spec, monitor

    def plot_scenario(self, ax, k=0):
        obstacle = make_circle_patch(self.obstacle_center, self.obstacle_radius, edgecolor='black', facecolor='gray', alpha=1.0)
        goal = make_rectangle_patch(*self.goal_bounds,  edgecolor='black', facecolor='skyblue', alpha=0.5)
        ax.add_patch(obstacle)
        ax.add_patch(goal)

        # set the field of view
        ax.set_xlim((0, 9))
        ax.set_ylim((0, 9))
        ax.set_aspect('equal')
