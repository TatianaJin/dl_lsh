#include <Eigen/Dense>
#include <Eigen/Sparse>

using Eigen::VectorXd;
using SparseVectorXd = Eigen::SparseVector<double>;

VectorXd vectorize(const std::vector<double>& data, int offset, int length) {
    VectorXd vector = VectorXd::Zero(length);
    for (int idx = 0; idx < length; ++idx) {
        vector(idx) = data[offset + idx];
    }
    return vector;
}

VectorXd vectorize(const std::vector<double>& data) { return vectorize(data, 0, data.size()); }

SparseVectorXd sparse_vectorize(const std::vector<double>& data) {
    int length = data.size();
    SparseVectorXd vector(length);
    for (int idx = 0; idx < length; ++idx) {
        if (data[idx] != 0) {
            vector.coeffRef(idx) = data[idx];
        }
    }
    return vector;
}
