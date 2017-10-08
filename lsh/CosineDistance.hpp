#ifndef COSINEDISTANCE_HPP
#define COSINEDISTANCE_HPP

#include <Eigen/Dense>
#include <vector>

#include "LSH.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class CosineDistance : public LSH {
   public:
    CosineDistance() = default;
    CosineDistance(int b, int L, int d);
    std::vector<int> hashSignature(VectorXd data);

   private:
    int m_L;
    std::vector<int> hashes;
    std::vector<MatrixXd> randomMatrix;
    std::mt19937 rng;
};

#endif  // COSINEDISTANCE_HPP
