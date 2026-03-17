import time

import numpy as np

from stl_pi_planner.solvers.stl_solver_base import STLSolverBase
from stl_pi_planner.STL.predicate import CirclePredicate

np.random.seed(42)

class STLGuidedPISolver(STLSolverBase):
    """
    PI Solver
    """
    def __init__(self, spec, spec_monitor, sys, x0, K, n_samples: int = 10, cov=np.array([]), lamb: float = 2.0,
                 nu: float = 0.6, num_iterations: int = 10, Q=np.eye(0), P=np.eye(0), R=np.eye(0), gamma=1.0,
                 robustness_cost_fct='max', pi_weighting=True, verbose=True):
        super().__init__(spec, sys, x0, K, verbose)

        # self.spec_syntax = spec_syntax  # STL 语法树
        self.monitor = spec_monitor  # rtamt StlDiscreteTimeSpecification 实例

        # Parameters
        self.n_samples = n_samples                          # nof trajectory samples
        self.lamb = lamb                                    # initial inverse? temperature
        self.cov = cov                                      # covariance
        self.nu = nu                                        # factor reducing the temperature and the covariance
        self.num_iterations = num_iterations                # number of pi iterations

        # Cost function
        self.Q = Q          # State cost
        self.P = P          # Input cost
        self.R = R          # Terminal cost
        self.gamma = gamma  # STL-cost weighting

        # Check R = lamb * conv^-1
        if not np.array_equal(R, lamb * np.linalg.inv(cov)):
            raise ValueError("R != lambda_init * cov_init^-1")

        self.rob_calc_time = 0.0

        # Select robustness cost function:
        if robustness_cost_fct == 'max':
            self.robustness_cost_fct = self.maximize_robustness_cost
        elif robustness_cost_fct == 'viol':
            self.robustness_cost_fct = self.violation_robustness_cost
        else:
            raise ValueError('robustness_cost_fct must be either max or viol')

        self.pi_weighting = pi_weighting   # When turned off -> pure random search algorithm with shrinking horizon

    def solve(self):
        """
        Solves the optimization problem using the Path-Integral algorithm.

        Performs iterative optimization steps to update the control input trajectory, scales covariance and inverse
        temperature, and records intermediate solutions at each iteration.

        @return: A tuple (x, u, rho, cost, solve_time, record_y_opt, record_y, record_best_sample_idx), where:
            - x: State trajectory array.
            - u: Control input trajectory array.
            - rho: Robustness measure.
            - cost: Final value of the cost function.
            - solve_time: Time taken to solve the optimization problem.
            - record_y_opt: List of optimal output trajectories at each iteration.
            - record_y: List of sampled output trajectories at each iteration.
            - record_best_sample_idx: List of indices of the best sample at each iteration.
        """

        # Initialize lists for recordings
        record_y_opt, record_y, record_best_sample_idx = [], [], []

        # Initialize input trajectory
        u = np.zeros([self.sys.m, self.K])

        # Initialize covariance and inverse temperature
        cov = self.cov
        lamb = self.lamb

        # Start timer
        t_start = time.time()

        # Perform the optimization for num_iterations
        cost_list = []
        for i in range(self.num_iterations+1):
            time1 = time.time()
            if self.verbose:
                print(f'Iteration: {i}/{self.num_iterations} --------------------------------------------')

            # Execute step of the PI algorithm
            time3 = time.time()
            u, y_opt, y, best_sample_idx = self.pi_step(u, cov, lamb)
            time4 = time.time()
            print(f"PI step time: {time4 - time3:.4f}s")

            # Scale covariance and lambda
            cov = self.nu * cov
            lamb = self.nu * lamb

            # Record intermediate solutions
            record_y_opt.append(y_opt)
            record_y.append(y)
            record_best_sample_idx.append(best_sample_idx)

            # 计算当前最优解的 cost 并记录
            x, y = self.forward_rollout(u)
            cost = self.calculate_cost(x, y, u)
            print(f"cost:{cost}")
            cost_list.append(cost)

            time2 = time.time()
            print(f"Iteration time: {time2 - time1:.4f}s")

        # End timer
        solve_time = time.time() - t_start

        # Get optimal state and output trajectory
        x, y = self.forward_rollout(u)
        rho = self.spec.robustness(y, 0)
        cost = self.calculate_cost(x, y, u)

        # return x, u, rho, cost, solve_time, record_y_opt, record_y, record_best_sample_idx, cost_list
        return x, u, rho, cost, solve_time, record_y_opt, record_y, record_best_sample_idx

    def pi_step(self, u, cov, lamb):
        """
        Performs one path-integral step (=iteration)
        @param u: Input trajectory
        @param cov: Covariance matrix
        @param lamb: inverse temperature
        @return: Tuple of updated input trajectory, optimal output trajectory, all samples, and cost-optimal (=best) sample
        """
        time_start = time.time()
        x_dim = self.sys.n
        y_dim = self.sys.p
        u_dim = self.sys.m

        eps = np.zeros([self.n_samples, u_dim, self.K])         # initialize epsilons
        u_samples = np.zeros([self.n_samples, u_dim, self.K])   # initialize u_samples
        x = np.zeros([self.n_samples, x_dim, self.K])           # initialize state trajectories
        y = np.zeros([self.n_samples, y_dim, self.K])           # initialize output trajectories
        cost = np.zeros(self.n_samples)                         # initialize path costs

        cov_inv = np.linalg.inv(cov)                            # Invert covariance
        time_end = time.time()
        # print(f"Pre-sampling time: {time_end - time_start:.4f}s")

        total_time1 = 0.0
        total_time2 = 0.0
        total_time3 = 0.0
        for n in range(self.n_samples):
            time1 = time.time()
            # Generate epsilons
            eps[n, ...] = np.random.multivariate_normal(mean=np.zeros(u_dim), cov=cov, size=self.K).T

            # Get state and output trajectory
            u_sample = u + eps[n, ...]
            u_samples[n, ...] = u_sample
            time2 = time.time()
            # x[n, ...], y[n, ...] = self.forward_rollout(u_sample)
            x[n, ...], y[n, ...], violated, rob = self.forward_rollout_with_monitor(u_sample)
            # print(violated)
            time3 = time.time()

            # Calculate cost
            if violated:
                # cost[n] = float('inf')
                cost[n] = 10000.0
                time4 = time.time()
                total_time1 += (time2 - time1)
                total_time2 += (time3 - time2)
                total_time3 += (time4 - time3)
                continue
            else:
                cost[n] = self.calculate_path_cost(x[n, ...], y[n, ...], eps[n, ...], u, lamb, cov_inv)
            time4 = time.time()
            total_time1 += (time2 - time1)
            total_time2 += (time3 - time2)
            total_time3 += (time4 - time3)
        print(f"Total times: {total_time1:.4f}s, {total_time2:.4f}s, {total_time3:.4f}s")

        time5 = time.time()
        # get trajectory with the lowest costs
        best_sample = np.argmin(cost)
        psi = cost[best_sample]

        if self.pi_weighting:
            # calculate weightings for each sample
            eta = np.sum(np.exp(-1 / lamb * (cost - psi)))
            omega = 1 / eta * np.exp(-1 / lamb * (cost - psi))

            for k in range(self.K):
                u[:, k] += omega @ eps[:, :, k]

            # _, y_opt = self.forward_rollout_with_monitor(u)
            _, y_opt = self.forward_rollout(u)
            time6 = time.time()
            # print(f"Weighting time: {time6 - time5:.4f}s")
            # print(f"total step time: {time6 - time_start:.4f}s")

            return u, y_opt, y, best_sample

        else:
            _, y_opt = self.forward_rollout(u_samples[best_sample])
            return u_samples[best_sample], y_opt, y, best_sample


    def forward_rollout(self, u):
        """
        Performs an integration step of the system dynamics
        @param u: Input trajectory
        @return: The state trajectory and output trajectory
        """
        x = np.full((self.sys.n, self.K), np.nan)
        y = np.full((self.sys.p, self.K), np.nan)

        x[:, 0] = self.x0

        for k in range(self.K-1):
            x[:, k+1] = self.sys.f(x[:, k], u[:, k])
            y[:, k] = self.sys.g(x[:, k], u[:, k])

        y[:, self.K-1] = self.sys.g(x[:, self.K-1], u[:, self.K-1])

        return x, y
    

    def forward_rollout_with_monitor(self, u):
        """
        Roll out system with given control sequence u_sample,
        and use rtamt monitor to compute STL robustness online.
        返回:
            x:  (K+1, nx)
            y:  (K+1, ny)
            violated: 是否提前判定明显违反（只根据当前 rob<0 的前缀粗略剪枝）
            rob: 最终得到的 STL robustness（rtamt 的结果）
        """
        K = self.K
        x = np.full((self.sys.n, self.K), np.nan)
        y = np.full((self.sys.p, self.K), np.nan)
        x[:, 0] = self.x0

        # 每个 rollout 前必须 reset 监控器的内部状态
        try:
            self.monitor.reset()
        except AttributeError:
            # 如果你用的 rtamt 版本没有 reset，可以注释掉这一段，
            # 然后在 Problem_II 里把 monitor 改成“构造函数”而不是实例（后面我们可以再细调）
            pass

        violated = False
        rob = None  # 存最后一次有效的robustness

        for k in range(self.K - 1):
            x[:, k+1] = self.sys.f(x[:, k], u[:, k])
            y[:, k]   = self.sys.g(x[:, k], u[:, k])

            # 当前时刻的观测 —— 这里只用位置
            px = float(y[0, k])
            py = float(y[1, k])
            # print(f"Step {k}: position=({px:.2f}, {py:.2f})")

            # 调用 rtamt 在线监控
            rob_k = self.monitor.update(k + 1, [['px', px], ['py', py], ['dist2', (px - 4)**2 + (py - 4)**2]])
            # print(f"  robustness={rob_k}")

            # rtamt 的 online monitor 是有固定 delay 的，
            # 有时候前几步会返回 None，你可以只在 rob_k 不是 None 时更新 rob
            if rob_k is not None:
                rob = rob_k

                # 非严格，但可以做一个简单剪枝：
                # 对于 "always not_obstacle and eventually goal" 这种规格，
                # 一旦 robustness 明显 < 0，就极大概率已经不可能恢复了，
                # 这里可以选择提前结束该 sample。
                if rob_k < 0.0:
                    violated = True
                    # 如果你想节省一点内存，可以在这里截断 x,y：
                    # x = x[:k+2, :]
                    # y = y[:k+2, :]
                    break
        
        # 最后一步输出
        if not violated:
            y[:, self.K-1] = self.sys.g(x[:, self.K-1], u[:, self.K-1])

        return x, y, violated, rob


    def state_cost(self, x):
        """
        Calculates the sate costs
        @param x: State trajectory
        @return: The state cost
        """
        return 0.5 * x @ self.Q @ x

    def terminal_cost(self, x):
        """
        Calculates the terminal cost
        @param x: State trajectory
        @return: The terminal cost
        """
        return self.P @ x

    def maximize_robustness_cost(self, y):
        """
        Cost function to maximize the robustness
        @param y: State trajectory
        @return: The robustness cost
        """
        return self.spec.robustness(y, 0)

    def violation_robustness_cost(self, y):
        """
        Cost function to minimize the bounded robustness
        @param y: State trajectory
        @return: The robustness cost
        """
        return min(self.spec.robustness(y, 0), 0)

    def calculate_path_cost(self, x, y, w, u, lamb, cov_inv):
        """
        Calculate the PI path costs
        @param x: State trajectory
        @param y: Output trajectory
        @param w: Weight
        @param u: Input trajectory
        @param lamb: Inverse temperature
        @param cov_inv: Inverse covariance matrix
        @return: The path cost
        """
        cost = 0
        for k in range(self.K - 1):
            cost += self.state_cost(x[..., k]) + lamb * np.sum(u[:, k] @ cov_inv @ w[:, k])     # State cost
        cost += self.terminal_cost(x[:, -1])                                                    # Terminal cost
        t_start = time.time()
        rob_costs = self.robustness_cost_fct(y)                                                 # Robustness cost
        self.rob_calc_time += time.time() - t_start
        cost += self.gamma * -rob_costs
        return cost

    def calculate_cost(self, x, y, u):
        """
        Calculates the cost as defined in the cost function (aka no PI related stuff)
        @param x: State trajectory
        @param y: Output trajectory
        @param u: Input trajectory
        @return: Cost
        """
        cost = 0
        for k in range(self.K - 1):
            cost += self.state_cost(x[..., k]) + 0.5 * u[:, k] @ self.R @ u[:, k]
        cost += self.terminal_cost(x[:, -1])
        cost += self.gamma * -self.robustness_cost_fct(y)
        return cost
