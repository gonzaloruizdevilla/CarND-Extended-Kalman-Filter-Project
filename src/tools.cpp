#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    cout << "Invalid estimation or ground_truth data." << endl;
    return rmse;
  }

  //accumulate squared residuals
  for (unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    //coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse / estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Hj(3, 4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //check division by zero
  if (px == 0 && py == 0) {

    cout << "Error: positions x and y with 0 value. Unable to calculate Jacobian.";

  } else {

    float p2 = px * px + py * py;
    float p2_1_2 = sqrt(p2);
    float p2_3_2 = p2 * p2_1_2;
  
    float vxpy_vypx = vx * py - vy * px;

    Hj << px / p2_1_2, py / p2_1_2, 0, 0,
        -py / p2, px / p2, 0, 0,
        py * (vxpy_vypx) / p2_3_2, -px * (vxpy_vypx) / p2_3_2, px / p2_1_2, py / p2_1_2;
  }

  return Hj;
}

VectorXd Tools::ToPolar(const VectorXd &x)
{
  VectorXd polar_x(3);

  float px = x[0];
  float py = x[1];
  float vx = x[2];
  float vy = x[3];

  if (px == 0. && py == 0.)
  {
    //if px and py are both zero, neigher angle nor rhodot can be calculated
    polar_x << 0, 0, 0;
  }
  else
  {
    float rho = sqrt(px * px + py * py);
    float theta = atan2(py, px);
    float rho_dot = (px * vx + py * vy) / rho;
    polar_x << rho, theta, rho_dot;
  }

  return polar_x;
}

VectorXd Tools::ToCartesian(const VectorXd &polar_x){
  VectorXd x(4);
  float rho = polar_x[0];
  float theta = polar_x[1];
  float rho_dot = polar_x[2];
  x << rho * cos(theta),
      rho * sin(theta),
      rho_dot * cos(theta),
      rho_dot * sin(theta);

  return x;
}
