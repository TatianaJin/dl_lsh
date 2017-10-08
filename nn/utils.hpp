#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

using Eigen::VectorXd;
using SparseVectorXd = Eigen::SparseVector<double>;

VectorXd vectorize(const std::vector<double>& data, int offset, int length);

VectorXd vectorize(const std::vector<double>& data);

SparseVectorXd sparse_vectorize(const std::vector<double>& data);

#endif  // UTILS_HPP
