#include "hpipm-cpp/hpipm-cpp.hpp"

#include <chrono>
#include <iostream>
#include <vector>

#include "Eigen/Core"

int main() {
  int dim_var = 2;
  int dim_eq = 1;
  int dim_ineq = 2;

  Eigen::MatrixXd H(dim_var, dim_var);
  H << 4.0, 1.0,
       1.0, 2.0;

  Eigen::VectorXd g = Eigen::VectorXd::Zero(dim_var);
  g << 1.0, 1.0;

  Eigen::MatrixXd A(dim_eq, dim_var);
  A << 1.0, 1.0;
  Eigen::VectorXd b = Eigen::VectorXd::Zero(dim_eq);
  b << 1.0;

  Eigen::MatrixXd C(2, 2);
  C << 1.0, 0.0,
       0.0, 1.0;

  Eigen::VectorXd lbg = Eigen::VectorXd::Zero(dim_ineq);
  lbg << 0.0, 0.0;
  Eigen::VectorXd ubg = Eigen::VectorXd::Zero(dim_ineq);
  ubg << 0.7, 0.7;

  Eigen::VectorXd lbx = std::numeric_limits<double>::lowest() * Eigen::VectorXd::Ones(dim_var);
  Eigen::VectorXd ubx = std::numeric_limits<double>::max() * Eigen::VectorXd::Ones(dim_var);

  hpipm::DenseQpSolver solver(dim_var, dim_eq, dim_ineq);
  solver.solve(H, g, A, b, C, lbg, ubg, lbx, ubx);
  std::cout << "Optimal solution: " << solver.getOptX().transpose() << std::endl;
  std::cout << "Optimal dual (BOX): " << solver.getOptLamB().transpose() << std::endl;
  std::cout << "Optimal dual (INEQ): " << solver.getOptLamG().transpose() << std::endl;
}
