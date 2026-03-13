#ifndef STL_PI_PLANNER_C_H
#define STL_PI_PLANNER_C_H

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <tuple>
#include <random>
#include <string>
#include <tuple>
#include <chrono>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <omp.h>

using namespace Eigen;

// STL Formula base class
class STLFormula {
public:
    STLFormula() = default;

    virtual ~STLFormula() = default;

    virtual double robustness(const MatrixXd &y, int k);
};

// STL Tree for combining formulas
class STLTree : public STLFormula {
private:
    std::string combination_type_;
    std::vector<int> time_steps_;
    std::vector <std::shared_ptr<STLFormula>> subformula_list_;

public:
    STLTree(std::string comb_type, std::vector<int> time_steps);

    void addSubformula(std::shared_ptr <STLFormula> subformula);

    double robustness(const MatrixXd &y, int k) override;

    double robustness_old(const MatrixXd &y, int k);
};

// Linear Predicate class
class LinearPredicate : public STLFormula {
private:
    VectorXd a_transpose_;
    double b_;

public:
    LinearPredicate(const VectorXd &a, const double b);

    double robustness(const MatrixXd &y, int k) override;
};

// Dynamic Linear Predicate class
class DynamicLinearPredicate : public STLFormula {
private:
    MatrixXd a_transpose_;
    VectorXd b_;

public:
    DynamicLinearPredicate(const MatrixXd &a, const VectorXd &b);

    double robustness(const MatrixXd &y, int k) override;
};

// Circle Predicate class
class CirclePredicate : public STLFormula {
private:
    bool negation_;
    double radius_;
    double center_x_;
    double center_y_;
    double factor_;

public:
    CirclePredicate(const double radius, const double center_x, const double center_y, bool negation);

    double robustness(const MatrixXd &y, int k) override;
};

// Dynamic Circle Predicate class
class DynamicCirclePredicate : public STLFormula {
private:
    bool negation_;
    double radius_;
    MatrixXd centers_;
    double factor_;

public:
    DynamicCirclePredicate(const double radius, const MatrixXd& centers, bool negation);

    double robustness(const MatrixXd &y, int k) override;
};

// Dynamic System base class
class DynamicSystem {
private:
    int x_dim_, y_dim_, u_dim_;

public:
    DynamicSystem(int x_dim, int y_dim, int u_dim);

    virtual VectorXd f(const VectorXd &x, const VectorXd &u);

    virtual VectorXd g(const VectorXd &x, const VectorXd &u);

    int getXDim() const;

    int getYDim() const;

    int getUDim() const;
};

// Derived classes for specific systems
class SingleIntegrator : public DynamicSystem {
private:
    double dt_;
    MatrixXd A_, B_, C_, D_;

public:
    SingleIntegrator(double dt);

    VectorXd f(const VectorXd &x, const VectorXd &u) override;

    VectorXd g(const VectorXd &x, const VectorXd &u) override;
};

class DoubleIntegrator : public DynamicSystem {
private:
    double dt_;
    MatrixXd A_, B_, C_, D_;

public:
    DoubleIntegrator(double dt);

    VectorXd f(const VectorXd &x, const VectorXd &u) override;

    VectorXd g(const VectorXd &x, const VectorXd &u) override;
};

class PointMass : public DynamicSystem {
private:
    double dt_;
    MatrixXd A_, B_, C_, D_;

public:
    PointMass(double dt);

    VectorXd f(const VectorXd &x, const VectorXd &u) override;

    VectorXd g(const VectorXd &x, const VectorXd &u) override;
};

class Unicycle : public DynamicSystem {
private:
    double dt_;

public:
    Unicycle(double dt);

    VectorXd f(const VectorXd &x, const VectorXd &u) override;

    VectorXd g(const VectorXd &x, const VectorXd &u) override;
};

class Bicycle : public DynamicSystem {
private:
    double dt_;
    double l_wb_;

public:
    Bicycle(double dt, double l_wb);

    VectorXd f(const VectorXd &x, const VectorXd &u) override;

    VectorXd g(const VectorXd &x, const VectorXd &u) override;
};

// PI Solver class for trajectory optimization
class PISolver {
private:
    STLTree &spec_;
    DynamicSystem &sys_;
    int x_dim_, y_dim_, u_dim_, K_, n_samples_, num_iterations;
    VectorXd x0_;
    MatrixXd cov_, Q_, P_, R_;
    double lamb_, psi_, gamma_;
    std::function<double(MatrixXd)> robustness_cost_fct_;
    bool verbose_, use_parallel_, pi_weighting_, use_stl_guided_;


public:
    PISolver(STLTree &spec, DynamicSystem &sys, const VectorXd &x0, int K, int n_samples, const MatrixXd &cov,
               double lamb, double psi, int num_iterations, const MatrixXd &Q, const MatrixXd &P,
               const MatrixXd &R, double gamma, std::string robustness_cost_fct, bool use_parallel, bool pi_weighting,
               bool verbose);

    std::tuple<MatrixXd, MatrixXd, double, double, double, std::vector<MatrixXd>, std::vector<
            std::vector < MatrixXd>>, std::vector<int>, std::vector<double>> solve();

    
    // std::tuple <Eigen::MatrixXd, Eigen::MatrixXd, bool, double> forward_rollout_with_monitor(const Eigen::MatrixXd &u);
    std::tuple <Eigen::MatrixXd, Eigen::MatrixXd, bool, double> forward_rollout_with_monitor(const MatrixXd &u, MatrixXd &eps, const MatrixXd &sqrt_cov, std::mt19937 &gen, std::normal_distribution<double> &dist);

    void set_use_stl_guided(bool flag);

private:
    void select_robustness_cost_function(const std::string& cost_function);

    std::tuple<MatrixXd, MatrixXd, std::vector<MatrixXd>, int, double, MatrixXd, double>
    pi_step(MatrixXd u, const MatrixXd &cov_curr, double lamb_curr, int iteration);


    MatrixXd
    generateRandomScaledMatrix(std::normal_distribution<double> &dist, std::mt19937 &gen, const MatrixXd &sqrt_cov);

    std::tuple <MatrixXd, MatrixXd> forward_rollout(const MatrixXd &u);

    double state_cost(const VectorXd &x);

    double maximize_robustness_cost(const MatrixXd &y);

    double violation_robustness_cost(const MatrixXd &y);

    double terminal_cost(const VectorXd &x);

    double calculate_path_cost(const MatrixXd &x, const MatrixXd &y, const MatrixXd &eps,
                               const std::vector <RowVectorXd> &alpha);

    double calculate_cost(const MatrixXd &x, const MatrixXd &y, const MatrixXd &u);
};

// Cost function containing robustness term
class RobCostFunction {
private:
    STLTree &spec_;
    DynamicSystem &sys_;
    int x_dim_, y_dim_, u_dim_, K_;
    VectorXd x0_;
    MatrixXd Q_, P_, R_;
    double gamma_;
    std::function<double(MatrixXd)> robustness_cost_fct_;
    bool verbose_;


public:
    RobCostFunction(STLTree &spec, DynamicSystem &sys, const VectorXd &x0, int K, const MatrixXd &Q, const MatrixXd &P,
                    const MatrixXd &R, double gamma, std::string robustness_cost_fct, bool verbose);

    double evaluate(const VectorXd &u);
    std::tuple <MatrixXd, MatrixXd> forward_rollout(const MatrixXd &u);

private:
    void select_robustness_cost_function(const std::string& cost_function);

    double state_cost(const VectorXd &x);

    double maximize_robustness_cost(const MatrixXd &y);

    double violation_robustness_cost(const MatrixXd &y);

    double terminal_cost(const VectorXd &x);

    double calculate_cost(const MatrixXd &u);
};


#endif // STL_PI_PLANNER_C_H
