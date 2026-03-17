#include "stl_pi_planner_c.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <tuple>
#include <random>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <cassert>

#include <limits>
#include <cmath>
#include <numeric>

namespace {

inline void weight_stats(const Eigen::VectorXd& omega,
                         double& H, double& perp, double& w_max) {
    H = 0.0;
    w_max = 0.0;
    for (int i = 0; i < omega.size(); ++i) {
        const double w = omega[i];
        if (w > 0.0) H -= w * std::log(w);
        if (w > w_max) w_max = w;
    }
    perp = std::exp(H);
}

inline double linear_r2(const std::vector<double>& x,
                        const std::vector<double>& y) {
    const int N = (int)x.size();
    if (N < 2) return std::numeric_limits<double>::quiet_NaN();

    const double x_mean = std::accumulate(x.begin(), x.end(), 0.0) / N;
    const double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / N;

    double Sxx = 0.0, Sxy = 0.0, Syy = 0.0;
    for (int i = 0; i < N; ++i) {
        const double dx = x[i] - x_mean;
        const double dy = y[i] - y_mean;
        Sxx += dx * dx;
        Sxy += dx * dy;
        Syy += dy * dy;
    }
    if (Sxx <= 1e-30 || Syy <= 1e-30) return 0.0;

    const double b = Sxy / Sxx;

    double SSE = 0.0;
    for (int i = 0; i < N; ++i) {
        const double y_hat = y_mean + b * (x[i] - x_mean);
        const double e = y[i] - y_hat;
        SSE += e * e;
    }
    return 1.0 - SSE / Syy;
}

} // namespace



// Definitions for STLFormula
double STLFormula::robustness(const MatrixXd &y, int k) {
    return 0.0;
}

// Definitions for STLTree
STLTree::STLTree(std::string comb_type, std::vector<int> time_steps)
        : combination_type_(comb_type), time_steps_(time_steps) {}

void STLTree::addSubformula(std::shared_ptr <STLFormula> subformula) {
    subformula_list_.push_back(subformula);
}

double STLTree::robustness(const MatrixXd &y, int k) {
    bool has_valid_value = false;
    double result;

    if (combination_type_ == "and") {
        result = std::numeric_limits<double>::infinity();  // Initialize to positive infinity for min
        for (size_t i = 0; i < subformula_list_.size(); ++i) {
            int adjusted_time = k + time_steps_[i];

            if (adjusted_time < y.cols()) {
                double robustness_i = subformula_list_[i]->robustness(y, adjusted_time);

                if (!std::isnan(robustness_i)) {
                    result = std::min(result, robustness_i);
                    has_valid_value = true;
                }
            }
        }
    } else if (combination_type_ == "or") {
        result = -std::numeric_limits<double>::infinity();  // Initialize to negative infinity for max
        for (size_t i = 0; i < subformula_list_.size(); ++i) {
            int adjusted_time = k + time_steps_[i];

            if (adjusted_time < y.cols()) {
                double robustness_i = subformula_list_[i]->robustness(y, adjusted_time);

                if (!std::isnan(robustness_i)) {
                    result = std::max(result, robustness_i);
                    has_valid_value = true;
                }
            }
        }
    } else {
        // Handle other combination types if necessary
        return std::numeric_limits<double>::quiet_NaN();
    }

    // If we found at least one valid robustness value, return the result
    if (has_valid_value) {
        return result;
    } else {
        // If no valid robustness values were found, return NaN
        return std::numeric_limits<double>::quiet_NaN();
    }
}

// Definitions for LinearPredicate
LinearPredicate::LinearPredicate(const Eigen::VectorXd &a, const double b)
        : a_transpose_(a.transpose()), b_(b) {}

double LinearPredicate::robustness(const Eigen::MatrixXd &y, int k) {
    return a_transpose_.dot(y.col(k)) - b_;
}

// Definitions for DynamicLinearPredicate
DynamicLinearPredicate::DynamicLinearPredicate(const Eigen::MatrixXd &a, const Eigen::VectorXd &b)
        : a_transpose_(a.transpose()), b_(b) {}

double DynamicLinearPredicate::robustness(const Eigen::MatrixXd &y, int k) {
    return a_transpose_.row(k).dot(y.col(k)) - b_[k];
}

// Definitions for CirclePredicate
CirclePredicate::CirclePredicate(const double radius, const double center_x, const double center_y, bool negation)
        : radius_(radius), center_x_(center_x), center_y_(center_y), negation_(negation)
{
    negation ? factor_ = -1.0 : factor_ = 1.0;
}

double CirclePredicate::robustness(const Eigen::MatrixXd &y, int k) {
    return factor_ * (pow(radius_, 2) - pow(y.col(k)[0] - center_x_, 2) - pow(y.col(k)[1] - center_y_, 2));
}

// Definitions for DynamicCirclePredicate
DynamicCirclePredicate::DynamicCirclePredicate(const double radius, const MatrixXd& centers, bool negation)
        : radius_(radius), centers_(centers), negation_(negation)
{
    negation ? factor_ = -1.0 : factor_ = 1.0;
}

double DynamicCirclePredicate::robustness(const Eigen::MatrixXd &y, int k) {
    return factor_ * (pow(radius_, 2) - pow(y.col(k)[0] - centers_.row(k)[0], 2) - pow(y.col(k)[1] - centers_.row(k)[1], 2));
}

// Definitions for DynamicSystem
DynamicSystem::DynamicSystem(int x_dim, int u_dim, int y_dim)
        : x_dim_(x_dim), u_dim_(u_dim), y_dim_(y_dim) {}

VectorXd DynamicSystem::f(const VectorXd &x, const VectorXd &u) {
    return VectorXd::Zero(x_dim_);
}

VectorXd DynamicSystem::g(const VectorXd &x, const VectorXd &u) {
    return VectorXd::Zero(y_dim_);
}

int DynamicSystem::getXDim() const { return x_dim_; }

int DynamicSystem::getYDim() const { return y_dim_; }

int DynamicSystem::getUDim() const { return u_dim_; }

// Definitions for SingleIntegrator
SingleIntegrator::SingleIntegrator(double dt)
    : DynamicSystem(1, 1, 2), dt_(dt) {

    A_.resize(1, 1);
    A_ << 1;

    B_.resize(1, 1);
    B_ << dt ;

    C_.resize(2, 1);
    C_ << 1, 0;

    D_.resize(2, 1);
    D_ << 0, 1;
}

VectorXd SingleIntegrator::f(const VectorXd &x, const VectorXd &u) {
    return A_ * x + B_ * u;
}

VectorXd SingleIntegrator::g(const VectorXd &x, const VectorXd &u) {
    return C_ * x + D_ * u;
}

// Definitions for DoubleIntegrator
DoubleIntegrator::DoubleIntegrator(double dt)
    : DynamicSystem(2, 1, 3), dt_(dt) {

    A_.resize(2, 2);
    A_ << 1, dt, 0, 1;

    B_.resize(2, 1);
    B_ << 0.5 * dt * dt, dt;

    C_.resize(3, 2);
    C_ << 1, 0, 0, 1, 0, 0;

    D_.resize(3, 1);
    D_ << 0, 0, 1;
}

VectorXd DoubleIntegrator::f(const VectorXd &x, const VectorXd &u) {
    return A_ * x + B_ * u;
}

VectorXd DoubleIntegrator::g(const VectorXd &x, const VectorXd &u) {
    return C_ * x + D_ * u;
}

// Definitions for PointMass
PointMass::PointMass(double dt)
        : DynamicSystem(4, 2, 6), dt_(dt) {

    A_.resize(4, 4);
    A_ << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1;

    B_.resize(4, 2);
    B_ << 0.5 * dt * dt, 0, 0, 0.5 * dt * dt, dt, 0, 0, dt;

    MatrixXd I = MatrixXd::Identity(2, 2);
    MatrixXd z = MatrixXd::Zero(2, 2);

    C_.resize(6, 4);
    C_ << I, z, z, I, z, z;

    D_.resize(6, 2);
    D_ << z, z, I;
}

VectorXd PointMass::f(const VectorXd &x, const VectorXd &u) {
    return A_ * x + B_ * u;
}

VectorXd PointMass::g(const VectorXd &x, const VectorXd &u) {
    return C_ * x + D_ * u;
}

// Definitions for Unicycle
Unicycle::Unicycle(double dt)
        : DynamicSystem(3, 2, 3), dt_(dt) {}

VectorXd Unicycle::f(const VectorXd &x, const VectorXd &u) {
    Eigen::VectorXd x_dot(3);
    x_dot[0] = u[0] * std::cos(x[2]); // v * cos(theta)
    x_dot[1] = u[0] * std::sin(x[2]); // v * sin(theta)
    x_dot[2] = u[1];                  // omega (angular velocity)
    return x + dt_ * x_dot;
}

VectorXd Unicycle::g(const VectorXd &x, const VectorXd &u) {
    return x;
}

// Definitions for Bicycle
Bicycle::Bicycle(double dt, double l_wb)
        : DynamicSystem(5, 2, 5), dt_(dt), l_wb_(l_wb) {}

VectorXd Bicycle::f(const VectorXd &x, const VectorXd &u) {
    Eigen::VectorXd x_dot(5);
    x_dot[0] = x[3] * std::cos(x[4]);
    x_dot[1] = x[3] * std::sin(x[4]);
    x_dot[2] = u[0];
    x_dot[3] = u[1];
    x_dot[4] = x[3]/l_wb_ * std::tan(x[2]);
    return x + dt_ * x_dot;
}

VectorXd Bicycle::g(const VectorXd &x, const VectorXd &u) {
    return x;
}

// Definitions for PI Solver
PISolver::PISolver(STLTree &spec, DynamicSystem &sys, const VectorXd &x0, int K, int n_samples, const MatrixXd &cov,
                       double lamb, double psi, int num_iterations, const MatrixXd &Q, const MatrixXd &P,
                       const MatrixXd &R, double gamma, std::string robustness_cost_fct, bool use_parallel,
                       bool pi_weighting, bool verbose, double cost_threshold)
        : spec_(spec), sys_(sys), x0_(x0), K_(K + 1), n_samples_(n_samples), cov_(cov), lamb_(lamb), psi_(psi),
          num_iterations(num_iterations), Q_(Q), P_(P), R_(R), gamma_(gamma), use_parallel_(use_parallel),
          pi_weighting_(pi_weighting), verbose_(verbose), use_stl_guided_(false), cost_threshold_(cost_threshold) {

    x_dim_ = sys_.getXDim();
    y_dim_ = sys_.getYDim();
    u_dim_ = sys_.getUDim();

    // Check if the matrix is diagonal using Eigen's built-in method
    assert(cov_.isDiagonal() && "The cov matrix must be diagonal! Arbitrary covariance matrix not supported yet.");
    // std::cout << "PISolver constructor: cov_ =\n" << cov_ << std::endl;
    // double lambda_cov = 3.4;
    // cov_ = lambda_cov * MatrixXd::Identity(u_dim_, u_dim_);
    // std::cout << "PISolver constructor: cov_ set to lambda_cov * I, lambda_cov = " << lambda_cov << "\n"
    //           << cov_ << std::endl;

    // Check R = lamb * conv^-1
    assert(R_ == lamb_ * cov.inverse() && "R != lambda_init * cov_init^-1");
    std::cout << "R_ : " << R_ << std::endl;

    // Select cost functions
    select_robustness_cost_function(robustness_cost_fct);

    // Enable dynamic adjustment of threads
    // omp_set_dynamic(1);
    omp_set_dynamic(0);
    omp_set_num_threads(1);
}

void PISolver::select_robustness_cost_function(const std::string& cost_function) {
    if (cost_function == "max") {
        robustness_cost_fct_ = [this](const MatrixXd &y) { return this->maximize_robustness_cost(y); };
    } else if (cost_function == "viol") {
        robustness_cost_fct_ = [this](const MatrixXd &y) { return this->violation_robustness_cost(y); };
    } else {
        throw std::invalid_argument("robustness_cost_fct must be either 'max' or 'viol'");
    }
}

std::tuple<MatrixXd, MatrixXd, double, double, double, std::vector<MatrixXd>, std::vector<std::vector < MatrixXd>>, std::vector<int>, std::vector<double>> PISolver::solve() {
    // Initialize lists for recordings
    std::vector <MatrixXd> record_y_opt;
    std::vector <std::vector<MatrixXd>> record_y;
    std::vector<int> record_best_sample_idx;
    std::vector<double> cost_list;

    // Initialize input trajectory
    MatrixXd u(u_dim_, K_);
    u.setZero();

    // Initialize covariance and inverse temperature
    MatrixXd cov_curr = cov_;
    double lamb_curr = lamb_;

    // Initialize previous cost for adaptive psi
    double prev_cost = std::numeric_limits<double>::infinity();

    // Start timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Perform the optimization for num_iterations
    for (int i = 0; i <= num_iterations; ++i) {
        if (verbose_) {
            std::cout << "Iteration: " << i << "/" << num_iterations << " ---------------------------" << std::endl;
        }

        // Execute step of the PI algorithm
        auto start = std::chrono::high_resolution_clock::now();

        MatrixXd y_opt;
        std::vector <MatrixXd> y;
        int best_sample_idx;
        double this_cost;
        std::vector<MatrixXd> eps;
        MatrixXd cov_new;
        double E1_hat_1;
        
        double H;
        std::tie(u, y_opt, y, best_sample_idx, this_cost, cov_new, E1_hat_1, H) = pi_step(u, cov_curr, lamb_curr, i);
        std::cout << "E1_hat_1: " << E1_hat_1 << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        std::cout << "PI step " << i << " time: " << duration << " seconds" << std::endl;

        // double cost_bi_lambda = this_cost/lamb_curr;

        // Calculate and print SNR
        // double snr = u.squaredNorm() / (K_ * cov_new.trace()/cov_curr(0,0));  // Normalize by time steps and average variance
        // std::cout << "SNR: " << snr << std::endl;

        // Adaptive psi based on cost improvement
        // double psi_ = 0.9;
        // double improvement = 0.0;
        // if (prev_cost != std::numeric_limits<double>::infinity() && prev_cost != 0.0) {
        //     improvement = (prev_cost - this_cost) / prev_cost;
        // }
        // std::cout << "Cost improvement: " << improvement << std::endl;
        // double threshold = 0.01;
        // double accelerate_factor = 0.5;
        // double decelerate_factor = 1;
        // double max_psi = 0.999;
        // if (improvement < 0) {
        //     psi_ = std::min(psi_ * accelerate_factor, max_psi);
        // } else {
        //     psi_ = std::min(psi_ * decelerate_factor, max_psi);
        // }
        // prev_cost = this_cost;

        // AIS
        // if(i<=10 && use_stl_guided_){
        // if(i<=10 && use_stl_guided_){
        //     double new_psi = 0.0;
        //     new_psi = cov_new(0, 0)/cov_curr(0, 0);
        //     cov_curr = cov_new;
        //     lamb_curr *= new_psi;
        // }
        // new scale
        // if(i<=10 && use_stl_guided_){
        // if(i<=5{
        // if(i<=10){
        //     double new_psi = 0.5;
        //     cov_curr *= new_psi;
        //     lamb_curr *= new_psi;
        // }
        // double new_psi = 0.0;
        // new_psi = cov_new(0, 0)/cov_curr(0, 0);
        // cov_curr = cov_new;
        // lamb_curr *= new_psi;

        int shrink_flag = 1; // 1: new 0: old
        if ( i> 5) {//需要一个判断来决定是否开始保护过程，防止玻璃化
            // double new_psi;
            // double a;
            // a = std::log(E1_hat_1);
            // std::cout << "********a: " << a << std::endl;
            // if (a < -400) {
            //     a = -400;
            // }
            // new_psi = std::exp(-(400+a) + std::log(0.5)) + 0.4;
            // std::cout << "********New psi: " << new_psi << std::endl;
            // cov_curr *= new_psi;
            // lamb_curr *= new_psi;
            //-----------------------------------
            double new_psi;
            double a;
            a = std::log(E1_hat_1);
            std::cout << "********a: " << a << std::endl;
            new_psi = -0.1*a + 0.1;//温和过程防止玻璃化，仍需改进
            std::cout << "********New psi: " << new_psi << std::endl;
            cov_curr *= new_psi;
            lamb_curr *= new_psi;
        } else {
            // Scale covariance and lambda
            //double new_psi;
            //new_psi = 0.813838;            
            //std::cout << "********New psi: " << new_psi << std::endl;
            //cov_curr *= new_psi;
            //lamb_curr *= new_psi;
            double new_psi;
            double a;
            double logN = std::log(std::max(2, n_samples_));
            double H_hat = std::max(0.0, std::min(1.0, H / logN));

            a = H_hat;
            std::cout << "********a(H_hat): " << a << std::endl;

            new_psi = std::max(0.3, 0.6 * (1.0 - H_hat)); //根据熵的极冷过程，H_hat是信息熵
            std::cout << "********New psi: " << new_psi << std::endl;

            cov_curr *= new_psi;
            lamb_curr *= new_psi;
        }

        std::cout << cov_curr << std::endl;

        // Record intermediate solutions
        record_y_opt.push_back(y_opt);
        record_y.push_back(y);  // TODO: This gives memory problems for large n_samples_
        record_best_sample_idx.push_back(best_sample_idx);
        cost_list.push_back(this_cost);
        // cost_list.push_back(cost_bi_lambda);

        if (this_cost < cost_threshold_) {
            break;
        }
    }

    // End timer
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    // Get optimal state and output trajectory
    MatrixXd x, y;
    std::tie(x, y) = forward_rollout(u);
    double rho = spec_.robustness(y, 0);
    double cost = calculate_cost(x, y, u);

    // Return results
    return std::make_tuple(x, u, rho, cost, solve_time, record_y_opt, record_y, record_best_sample_idx, cost_list);
}

std::tuple<MatrixXd, MatrixXd, std::vector<MatrixXd>, int, double, MatrixXd, double, double>
PISolver::pi_step(MatrixXd u, const MatrixXd &cov_curr, double lamb_curr, int iteration) {
    std::vector <MatrixXd> eps(n_samples_, MatrixXd::Zero(u_dim_, K_));         // initialize epsilons
    std::vector <MatrixXd> u_samples(n_samples_, MatrixXd::Zero(u_dim_, K_));   // initialize u_samples
    std::vector <MatrixXd> x(n_samples_, MatrixXd::Zero(x_dim_, K_));           // initialize state trajectories
    std::vector <MatrixXd> y(n_samples_, MatrixXd::Zero(y_dim_, K_));           // initialize output trajectories
    VectorXd cost(n_samples_);                                                  // initialize path costs

    // An intermediate value saving computation time in cost calculation
    MatrixXd cov_curr_inv = cov_curr.inverse();
    std::vector <RowVectorXd> alpha;
    for (int k = 0; k < K_ - 1; ++k) {
        alpha.push_back(lamb_curr * u.col(k).transpose() * cov_curr_inv);
    }

    MatrixXd sqrt_cov_curr = cov_curr.array().sqrt();

#pragma omp parallel if(use_parallel_)
    {
        // Create a local generator for each thread
        std::mt19937 gen_local(std::random_device{}());
        std::normal_distribution<double> dist_local(0.0, 1.0);
        bool violated = false;

#pragma omp for
        for (int n = 0; n < n_samples_; ++n) {
            // Generate random samples
            eps[n] = generateRandomScaledMatrix(dist_local, gen_local, sqrt_cov_curr);
            MatrixXd u_sample = u + eps[n];
            u_samples[n] = u_sample;

            // Get state and output trajectory (optionally STL-guided)
            // if (use_stl_guided_ && iteration > 30) {
            if (use_stl_guided_ && iteration <= 30) {
                // std::cout<< "Using STL-guided rollout for sample " << std::endl;
                bool violated = false;
                double rob = 0.0;
                std::tie(x[n], y[n], violated, rob) = forward_rollout_with_monitor(u, eps[n], sqrt_cov_curr, gen_local, dist_local);
                // printf("Sample %d: robustness = %f\n", n, rob);

                if (violated) {
                    cost[n] = 1e7;
                    // std::cout << "violated=" << std::boolalpha << violated << std::endl;
                    eps[n].setZero();
                    continue;
                }
            } else {
                std::tie(x[n], y[n]) = forward_rollout(u_sample);
            }

            // Calculate cost for non-rejected samples
            cost[n] = calculate_path_cost(x[n], y[n], eps[n], alpha);
        };
    };

    std::cout << "Expected cost over samples: " << cost.mean() << std::endl;
    // std::cout << "all costs: " << cost.transpose() << std::endl;

    // get trajectory with the lowest costs
    int best_sample_idx;
    cost.minCoeff(&best_sample_idx);
    double psi = cost(best_sample_idx);

    if (pi_weighting_) {
        // calculate weightings for each sample
        VectorXd costs_exp = (-(cost.array() - psi) / lamb_curr).exp();
        VectorXd costs_exp_1 = (-(cost.array()+10) / lamb_curr).exp();
        std::cout << "min cost: " << psi << std::endl;
        // std::cout << "Costs_exp_1: " << costs_exp_1.transpose() << std::endl;
        double E1_hat = costs_exp.sum() / n_samples_;
        double E1_hat_1 = costs_exp_1.sum() / n_samples_;
        std::cout << "%%%%%%%%%E1_hat: " << E1_hat << std::endl;
        double eta = costs_exp.sum();
        VectorXd omega = costs_exp / eta;
        // double E1 = eta / n_samples_;
        // std::cout << "E1: " << E1 << std::endl;
        // Debug: print weights
        // if (verbose_) {
        //     std::cout << "Weights (omega): " << omega.transpose() << std::endl;
        // }

        // Calculate ESS
        double ess = 0.0;
        for (int n = 0; n < n_samples_; ++n) {
            ess += omega[n] * omega[n];
        }
        ess = 1.0 / ess;
        std::cout << "ESS: " << ess << " / " << n_samples_ << std::endl;

        // ==================== Diagnostics (whitened, scale-aware) ====================

        // valid sample indices (skip rejected samples whose eps is zero)
        std::vector<int> valid_idx;
        valid_idx.reserve(n_samples_);
        for (int n = 0; n < n_samples_; ++n) {
            if (eps[n].norm() > 0.0) valid_idx.push_back(n);
        }
        const int valid_cnt = (int)valid_idx.size();

        // 0) Weight stats: entropy/perplexity/max weight
        double H = 0.0, perp = 0.0, w_max = 0.0;
        weight_stats(omega, H, perp, w_max);

        // 1) Whitened effective update magnitude: ||ΔU~||_2
        // ΔU = sum_i ω_i ε_i;  ΔU~(d,k) = ΔU(d,k)/sqrt(cov_curr(d,d))
        Eigen::MatrixXd delta_u = Eigen::MatrixXd::Zero(u_dim_, K_);
        for (int n = 0; n < n_samples_; ++n) {
            // eps[n] rejected -> all zeros, contributes nothing anyway
            delta_u.noalias() += omega[n] * eps[n];
        }

        double delta_u_tilde_norm2 = 0.0;
        for (int d = 0; d < u_dim_; ++d) {
            const double sigma2 = std::max(1e-18, cov_curr(d, d));
            const double inv_sigma = 1.0 / std::sqrt(sigma2);
            for (int k = 0; k < K_; ++k) {
                const double v = delta_u(d, k) * inv_sigma;
                delta_u_tilde_norm2 += v * v;
            }
        }
        const double delta_u_tilde_norm = std::sqrt(delta_u_tilde_norm2);

        // 2) Elite set by cost among valid samples (top-q)
        const double elite_q = 0.10; // 10% elites; feel free to change
        int elite_cnt = 0;
        double trace_cov_elite_norm = std::numeric_limits<double>::quiet_NaN();

        if (valid_cnt >= 2) {
            std::vector<int> elite_idx = valid_idx;
            std::sort(elite_idx.begin(), elite_idx.end(),
                    [&](int a, int b){ return cost[a] < cost[b]; });

            elite_cnt = std::max(2, (int)std::floor(elite_q * valid_cnt));
            elite_cnt = std::min(elite_cnt, valid_cnt);

            // Compute mean of whitened eps over elites, then trace(cov) = E||x-mu||^2
            Eigen::VectorXd mu = Eigen::VectorXd::Zero(u_dim_ * K_);
            Eigen::VectorXd tmp(u_dim_ * K_);

            auto flatten_whiten = [&](int n, Eigen::VectorXd& out) {
                out.resize(u_dim_ * K_);
                int t = 0;
                for (int d = 0; d < u_dim_; ++d) {
                    const double sigma2 = std::max(1e-18, cov_curr(d, d));
                    const double inv_sigma = 1.0 / std::sqrt(sigma2);
                    for (int k = 0; k < K_; ++k) {
                        out(t++) = eps[n](d, k) * inv_sigma;
                    }
                }
            };

            for (int j = 0; j < elite_cnt; ++j) {
                flatten_whiten(elite_idx[j], tmp);
                mu += tmp;
            }
            mu /= (double)elite_cnt;

            double trace_cov_elite = 0.0;
            for (int j = 0; j < elite_cnt; ++j) {
                flatten_whiten(elite_idx[j], tmp);
                tmp -= mu;
                trace_cov_elite += tmp.squaredNorm();
            }
            trace_cov_elite /= (double)elite_cnt;

            // normalize by dimension (D*K), so values are comparable across K/u_dim
            trace_cov_elite_norm = trace_cov_elite / (double)(u_dim_ * K_);
        }

        // 3) R^2 of (cost-psi) vs ||eps~||^2 (over valid samples)
        double R2 = std::numeric_limits<double>::quiet_NaN();
        if (valid_cnt >= 2) {
            std::vector<double> x_r2; x_r2.reserve(valid_cnt);
            std::vector<double> y_r2; y_r2.reserve(valid_cnt);

            for (int idx_i = 0; idx_i < valid_cnt; ++idx_i) {
                const int n = valid_idx[idx_i];

                double x2 = 0.0; // ||eps~||^2
                for (int d = 0; d < u_dim_; ++d) {
                    const double sigma2 = std::max(1e-18, cov_curr(d, d));
                    const double inv_sigma = 1.0 / std::sqrt(sigma2);
                    for (int k = 0; k < K_; ++k) {
                        const double v = eps[n](d, k) * inv_sigma;
                        x2 += v * v;
                        // x2 += abs(v);  // L1 norm variant
                    }
                }
                x_r2.push_back(x2);
                y_r2.push_back(cost[n] - psi);
            }
            R2 = linear_r2(x_r2, y_r2);
        }

        // Print all diagnostics in one line
        std::cout
            << "Diag(valid=" << valid_cnt << ", elite=" << elite_cnt << ") "
            << "||dU~||2=" << delta_u_tilde_norm
            << " traceCovElite~=" << trace_cov_elite_norm
            << " R2=" << R2
            << " H(w)=" << H
            << " perp=" << perp
            << " w_max=" << w_max
            << std::endl;

        // ==================== End diagnostics ====================


        // Update covariance matrix by ESS
        // MatrixXd cov_new = MatrixXd::Zero(u_dim_, u_dim_);
        // if(ess/n_samples_ > 0.01)
        // {
        //     cov_new = cov_curr * 0.5;
        // }
        // else
        // {
        //     cov_new = cov_curr;
        // }

        // Save old u
        MatrixXd u_old = u;

        // record distribution
        std::vector<MatrixXd> matrices(n_samples_, MatrixXd::Zero(u_dim_, K_));
        for (int n = 0; n < n_samples_; ++n) {
            matrices[n] = omega[n] * eps[n];
        }
        double total_mean = 0.0;
        int total_elements = n_samples_ * u_dim_ * K_;
        for (int i = 0; i < n_samples_; ++i) {
            total_mean += matrices[i].sum();
        }
        total_mean /= total_elements;
        double total_variance = 0.0;
        for (int i = 0; i < n_samples_; ++i) {
            for (int d = 0; d < u_dim_; ++d) {
                for (int k = 0; k < K_; ++k) {
                    double diff = matrices[i](d, k) - total_mean;
                    total_variance += diff * diff;
                }
            }
        }
        total_variance /= total_elements;
        total_variance = n_samples_ * total_variance;
        std::cout << "--------Variance: " << total_variance << std::endl;

        // Update u
        MatrixXd weighted_eps_sum = MatrixXd::Zero(u_dim_, K_);
        double sum_delta_u = 0.0;
        for (int n = 0; n < n_samples_; ++n) {
            weighted_eps_sum += omega[n] * eps[n];
        }
        u += weighted_eps_sum;
        for (int d = 0; d < u_dim_; ++d) {
            for (int k = 0; k < K_; ++k) {
                sum_delta_u += std::abs(u_old(d, k) - u(d, k));
            }
        }
        sum_delta_u /= (u_dim_ * K_);
        sum_delta_u /= std::sqrt(cov_curr(0,0)); // normalize by standard deviation
        std::cout << "Sum of |delta_u|: " << sum_delta_u << std::endl;

        // 3) R^2 of (cost-psi) vs ||eps~||^2 (over valid samples)
        double R2_post = std::numeric_limits<double>::quiet_NaN();
        if (valid_cnt >= 2) {
            std::vector<double> x_r2; x_r2.reserve(valid_cnt);
            std::vector<double> y_r2; y_r2.reserve(valid_cnt);

            for (int idx_i = 0; idx_i < valid_cnt; ++idx_i) {
                const int n = valid_idx[idx_i];
                double x2 = 0.0; // ||eps~||^2
                for (int d = 0; d < u_dim_; ++d) {
                    // const double sigma2 = std::max(1e-18, cov_curr(d, d));
                    // const double inv_sigma = 1.0 / std::sqrt(sigma2);
                    for (int k = 0; k < K_; ++k) {
                        // const double v = eps[n](d, k) * inv_sigma;
                        const double v = eps[n](d, k) - eps[best_sample_idx](d, k); // centered at best sample
                        // x2 += v * v;
                        x2 += abs(v);  // L1 norm variant
                    }
                }
                x2 = x2 * x2; // square L1 norm to get something comparable to L2 norm squared
                x_r2.push_back(x2);
                y_r2.push_back(cost[n] - psi);
                // std::cout << "Sample " << n << ": x2 = " << x2 << ", cost - psi = " << cost[n] - psi << std::endl; 
            }
            R2_post = linear_r2(x_r2, y_r2);
        }
        std::cout << "Post-update R2: " << R2_post << std::endl;

        // Update cov using weighted variance of eps (diagonal covariance)
        MatrixXd cov_new = MatrixXd::Zero(u_dim_, u_dim_);
        for (int d = 0; d < u_dim_; ++d) {
            double weighted_var = 0.0;
            double sum_omega = 0.0;
            for (int n = 0; n < n_samples_; ++n) {
                if (eps[n].norm() > 0.0) {  // Only use non-discarded samples
                    double eps_sum_sq = 0.0;
                    double eqs_for_new_u = 0.0;
                    for (int k = 0; k < K_; ++k) {
                        eqs_for_new_u = u_old(d, k) + eps[n](d, k) - u(d, k);
                        // eps_sum_sq += eps[n](d, k) * eps[n](d, k);
                        // eps_sum_sq += abs(eps[n](d, k));
                        eps_sum_sq += eqs_for_new_u * eqs_for_new_u;
                    }
                    weighted_var += omega[n] * (eps_sum_sq / K_);  // Average over time steps
                    sum_omega += omega[n];
                    // weighted_var = weighted_var * weighted_var;  // Square the average
                }
            }
            // std::cout<< "sum_omega: " << sum_omega << std::endl;
            // weighted_var = weighted_var * weighted_var;  // Square the average
            if (sum_omega > 0) {
                cov_new(d, d) = weighted_var / sum_omega;
                // double beta = 0.5; // 0~1
                double beta = 1; // 0~1
                cov_new(d, d) = (1.0 - beta) * cov_curr(d,d) + beta * cov_new(d,d);
            } else {
                cov_new(d, d) = cov_curr(d, d);  // Fallback to current cov
            }
        }
        double cov_avg = 0.0;
        for (int d = 0; d < u_dim_; ++d) {
            cov_avg += cov_new(d, d);
        }
        cov_avg /= u_dim_;
        for (int d = 0; d < u_dim_; ++d) {
            cov_new(d, d) = cov_avg;
        }

        // Get output trajectory
        MatrixXd x_opt;
        MatrixXd y_opt;
        double final_cost;
        // std::tie(std::ignore, y_opt) = forward_rollout(u);
        std::tie(x_opt, y_opt) = forward_rollout(u);
        final_cost = calculate_cost(x_opt, y_opt, u);
        printf("Final cost after weighting: %f\n", final_cost);

        // return std::make_tuple(u, y_opt, y, best_sample_idx, final_cost, cov_new, E1_hat_1);
        return std::make_tuple(u, y_opt, y, best_sample_idx, final_cost, cov_new, E1_hat, H);
    } else {
        MatrixXd y_opt;
        std::tie(std::ignore, y_opt) = forward_rollout(u_samples[best_sample_idx]);
        double final_cost = calculate_cost(x[best_sample_idx], y[best_sample_idx], u_samples[best_sample_idx]);
        return std::make_tuple(u_samples[best_sample_idx], y_opt, y, best_sample_idx, final_cost, cov_curr, 0.0, 0.0);
    }
}

Eigen::MatrixXd PISolver::generateRandomScaledMatrix(std::normal_distribution<double> &dist, std::mt19937 &gen,
                                                       const Eigen::MatrixXd &sqrt_cov) {
    Eigen::MatrixXd eps(u_dim_, K_);

    for (int n = 0; n < u_dim_; ++n) {
        for (int k = 0; k < K_; ++k) {
            eps(n, k) = dist(gen) * sqrt_cov(n, n);
        }
    }
    return eps;
}

std::tuple <MatrixXd, MatrixXd> PISolver::forward_rollout(const MatrixXd &u) {
    MatrixXd x(x_dim_, K_);
    MatrixXd y(y_dim_, K_);

    x.col(0) = x0_;
    for (int k = 0; k < K_ - 1; ++k) {
        x.col(k + 1) = sys_.f(x.col(k), u.col(k));
        y.col(k) = sys_.g(x.col(k), u.col(k));
    }

    y.col(K_ - 1) = sys_.g(x.col(K_ - 1), u.col(K_ - 1));

    return std::make_tuple(x, y);
}

std::tuple <MatrixXd, MatrixXd, bool, double> PISolver::forward_rollout_with_monitor(const MatrixXd &u, MatrixXd &eps, const MatrixXd &sqrt_cov, std::mt19937 &gen, std::normal_distribution<double> &dist) {
    MatrixXd x(x_dim_, K_);
    MatrixXd y(y_dim_, K_);
    bool violated = false;

    const double cx = 4.0; 
    const double cy = 4.0; 
    // const double cx2 = 4.0; 
    // const double cy2 = 6.0; 
    // const double cx3 = 6.0; 
    // const double cy3 = 4.0;
    const double r2 = 2.25;
    // const double r2 = 2;  
    // const double r2 = 0.25;

    MatrixXd u_sample = u + eps;

    x.col(0) = x0_;
    for (int k = 0; k < K_ - 1; ++k) {
        bool success = false;
        int attempts = 0;
        const int max_attempts = 5;  // Maximum attempts to resample at this step

        while (!success && attempts < max_attempts) {
            x.col(k + 1) = sys_.f(x.col(k), u_sample.col(k));
            y.col(k) = sys_.g(x.col(k), u_sample.col(k));
            // std::cout << "Step " << k << ": y = [" << y(0, k) << ", " << y(1, k) << "]" << std::endl;
            // std::cout << "x = [" << x(0, k) << ", " << x(1, k) << ", " << x(2, k) << ", " << x(3, k) << "]" << std::endl;
            double dx = x(0, k+1) - cx;
            double dy = x(1, k+1) - cy;
            // double dx2 = x(0, k+1) - cx2;
            // double dy2 = x(1, k+1) - cy2;
            // double dx3 = x(0, k+1) - cx3;
            // double dy3 = x(1, k+1) - cy3;
            double dist2 = dx * dx + dy * dy;
            // double dist22 = dx2 * dx2 + dy2 * dy2;
            // double dist23 = dx3 * dx3 + dy3 * dy3;
            bool inside_circle = (dist2 <= r2);
            // bool inside_circle2 = (dist22 <= r2);
            // bool inside_circle3 = (dist23 <= r2);
            // if (inside_circle || inside_circle2 || inside_circle3) {
            if (inside_circle) {
                // Resample eps at step k
                for (int d = 0; d < u_dim_; ++d) {
                    eps(d, k) = dist(gen) * sqrt_cov(d, d);
                }
                u_sample.col(k) = u.col(k) + eps.col(k);
                attempts++;
                // std::cout << "Resampling at step " << k << ", attempt " << attempts << std::endl;
            } else {
                // if(attempts > 0) {
                //     std::cout << "Successful sample at step " << k << " after " << attempts << " resamples." << std::endl;
                // }
                success = true;
            }
        }

        if (!success) {
            violated = true;
            break;
        }
    }

    if (!violated){
        y.col(K_ - 1) = sys_.g(x.col(K_ - 1), u_sample.col(K_ - 1));
    }

    double rob = 0.0;

    return std::make_tuple(x, y, violated, rob);
}

void PISolver::set_use_stl_guided(bool flag) {
    use_stl_guided_ = flag;
}


double PISolver::state_cost(const VectorXd &x) {
    return 0.5 * x.transpose() * Q_ * x;
}

double PISolver::maximize_robustness_cost(const MatrixXd &y) {
    return spec_.robustness(y, 0);
}

double PISolver::violation_robustness_cost(const MatrixXd &y) {
    return std::min(spec_.robustness(y, 0), 0.0);
}

double PISolver::terminal_cost(const VectorXd &x) {
    Eigen::VectorXd P_vector = P_.reshaped();  // Flattens the matrix to a vector
    return P_vector.dot(x);
}

double PISolver::calculate_path_cost(const MatrixXd &x, const MatrixXd &y, const MatrixXd &eps,
                                       const std::vector <RowVectorXd> &alpha) {
    double cost = 0.0;
    for (int k = 0; k < K_ - 1; ++k) {
        cost += state_cost(x.col(k)) + (alpha[k] * eps.col(k)).sum(); // State cost
    }
    cost += terminal_cost(x.col(K_ - 1));       // Terminal cost
    cost += gamma_ * -robustness_cost_fct_(y);       // Robustness cost
    return cost;
}

double PISolver::calculate_cost(const MatrixXd &x, const MatrixXd &y, const MatrixXd &u) {
    double cost = 0.0;
    for (int k = 0; k < K_ - 1; ++k) {
        cost += state_cost(x.col(k)) + 0.5 * u.col(k).transpose() * R_ * u.col(k); // State cost
    }
    cost += terminal_cost(x.col(K_ - 1));            // Terminal cost
    cost += gamma_ * -robustness_cost_fct_(y);       // Robustness cost
    return cost;
}

// Definitions for RobCostFunction
RobCostFunction::RobCostFunction(STLTree &spec, DynamicSystem &sys, const VectorXd &x0, int K, const MatrixXd &Q,
                                 const MatrixXd &P, const MatrixXd &R, double gamma, std::string robustness_cost_fct,
                                 bool verbose)

        : spec_(spec), sys_(sys), x0_(x0), K_(K), Q_(Q), P_(P), R_(R), gamma_(gamma), verbose_(verbose) {

    x_dim_ = sys_.getXDim();
    y_dim_ = sys_.getYDim();
    u_dim_ = sys_.getUDim();

    // Select cost functions
    select_robustness_cost_function(robustness_cost_fct);
}


void RobCostFunction::select_robustness_cost_function(const std::string& cost_function) {
    if (cost_function == "max") {
        robustness_cost_fct_ = [this](const MatrixXd &y) { return this->maximize_robustness_cost(y); };
    } else if (cost_function == "viol") {
        robustness_cost_fct_ = [this](const MatrixXd &y) { return this->violation_robustness_cost(y); };
    } else {
        throw std::invalid_argument("robustness_cost_fct must be either 'max' or 'viol'");
    }
}

double RobCostFunction::evaluate(const VectorXd &u)
{
    Eigen::Map<const Eigen::Matrix<double, Dynamic, Dynamic, Eigen::RowMajor>> u_reshaped(u.data(), u_dim_, K_); //reshape
    return calculate_cost(u_reshaped);
}

std::tuple <MatrixXd, MatrixXd> RobCostFunction::forward_rollout(const MatrixXd &u) {
    MatrixXd x(x_dim_, K_);
    MatrixXd y(y_dim_, K_);

    x.col(0) = x0_;
    for (int k = 0; k < K_ - 1; ++k) {
        x.col(k + 1) = sys_.f(x.col(k), u.col(k));
        y.col(k) = sys_.g(x.col(k), u.col(k));
    }
    y.col(K_ - 1) = sys_.g(x.col(K_ - 1), u.col(K_ - 1));

    return std::make_tuple(x, y);
}

double RobCostFunction::state_cost(const VectorXd &x) {
    return 0.5 * x.transpose() * Q_ * x;
}

double RobCostFunction::maximize_robustness_cost(const MatrixXd &y) {
    return spec_.robustness(y, 0);
}

double RobCostFunction::violation_robustness_cost(const MatrixXd &y) {
    return std::min(spec_.robustness(y, 0), 0.0);
}

double RobCostFunction::terminal_cost(const VectorXd &x) {
    Eigen::VectorXd P_vector = P_.reshaped();  // Flattens the matrix to a vector
    return P_vector.dot(x);
}

double RobCostFunction::calculate_cost(const MatrixXd &u) {
    MatrixXd x, y;
    std::tie(x, y) = forward_rollout(u);

    double cost = 0.0;
    for (int k = 0; k < K_-1; ++k) {
        cost += state_cost(x.col(k)) + 0.5 * u.col(k).transpose() * R_ * u.col(k); // State cost
    }
    cost += terminal_cost(x.col(K_-1));             // Terminal cost
    cost += gamma_ * -robustness_cost_fct_(y);      // Robustness cost
    return cost;
}
