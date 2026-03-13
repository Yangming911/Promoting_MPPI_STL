#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../cpp/stl_pi_planner_c.cpp"

namespace py = pybind11;

PYBIND11_MODULE(stl_pi_planner_c, m) {
    py::class_<STLFormula, std::shared_ptr<STLFormula>>(m, "STLFormula")
        .def(py::init<>())
        .def("robustness", &STLFormula::robustness);

    py::class_<STLTree, STLFormula, std::shared_ptr<STLTree>>(m, "STLTree")
        .def(py::init<std::string, std::vector<int>>())
        .def("robustness", &STLTree::robustness)
        .def("add_subformula", &STLTree::addSubformula);

    py::class_<LinearPredicate, STLFormula, std::shared_ptr<LinearPredicate>>(m, "LinearPredicate")
        .def(py::init<Eigen::VectorXd&, double>())
        .def("robustness", &LinearPredicate::robustness);

    py::class_<DynamicLinearPredicate, STLFormula, std::shared_ptr<DynamicLinearPredicate>>(m, "DynamicLinearPredicate")
        .def(py::init<Eigen::MatrixXd&, Eigen::VectorXd&>())
        .def("robustness", &DynamicLinearPredicate::robustness);

    py::class_<CirclePredicate, STLFormula, std::shared_ptr<CirclePredicate>>(m, "CirclePredicate")
        .def(py::init<double, double, double, bool>())
        .def("robustness", &CirclePredicate::robustness);

    py::class_<DynamicCirclePredicate, STLFormula, std::shared_ptr<DynamicCirclePredicate>>(m, "DynamicCirclePredicate")
        .def(py::init<double, Eigen::MatrixXd&, bool>())
        .def("robustness", &DynamicCirclePredicate::robustness);

    py::class_<DynamicSystem, std::shared_ptr<DynamicSystem>>(m, "DynamicSystem")
        .def(py::init<int, int, int>());

    py::class_<SingleIntegrator, DynamicSystem, std::shared_ptr<SingleIntegrator>>(m, "SingleIntegrator")
        .def(py::init<double>());

    py::class_<DoubleIntegrator, DynamicSystem, std::shared_ptr<DoubleIntegrator>>(m, "DoubleIntegrator")
        .def(py::init<double>());

    py::class_<PointMass, DynamicSystem, std::shared_ptr<PointMass>>(m, "PointMass")
        .def(py::init<double>());

    py::class_<Unicycle, DynamicSystem, std::shared_ptr<Unicycle>>(m, "Unicycle")
        .def(py::init<double>());

    py::class_<Bicycle, DynamicSystem, std::shared_ptr<Bicycle>>(m, "Bicycle")
        .def(py::init<double, double>());

    py::class_<PISolver>(m, "PISolver")
        .def(py::init<STLTree&, DynamicSystem&, Eigen::VectorXd, int, int, Eigen::MatrixXd, double, double, int,
            Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, double, std::string, bool, bool, bool>())
        .def("solve", &PISolver::solve)
        .def("set_use_stl_guided", &PISolver::set_use_stl_guided);

    py::class_<RobCostFunction>(m, "RobCostFunction")
        .def(py::init<STLTree&, DynamicSystem&, Eigen::VectorXd, int,
             Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, double, std::string, bool>())
        .def("evaluate", &RobCostFunction::evaluate)
        .def("forward_rollout", &RobCostFunction::forward_rollout);
}
