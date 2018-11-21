#include "kalman_filter.h"
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  UpdateWith_y(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) 
{
  VectorXd h = tools.ToPolar(x_);
  VectorXd y = z - h;
  double y1 = y(1);
  if (y1 > M_PI)
  {
    y[1] -= 2. * M_PI;
  }
  if (y1 < - M_PI)
  {
    y[1] += 2. *  M_PI;
  }

  UpdateWith_y(y);
}

void KalmanFilter::UpdateWith_y(const Eigen::VectorXd &y)
{
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
  //new estimates
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(4, 4);
  P_ = (I - K * H_) * P_;
}
