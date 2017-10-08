#include "CosineDistance.hpp"

#include <iostream>

#include "RandomProjection.hpp"

CosineDistance::CosineDistance(int b, int L, int d) {
    m_L = L;
    hashes.resize(m_L);
    randomMatrix.reserve(m_L);
    for (int i = 0; i < m_L; ++i) {
        // the assignment of coefficients follows uniform distrbution [-1,1]
        randomMatrix.push_back(MatrixXd::Zero(b, d).unaryExpr([& rng = this->rng](double a) {
            std::normal_distribution<> nd(0.0, 1.0);
            return nd(rng);
        }));
    }
}

std::vector<int> CosineDistance::hashSignature(VectorXd data) {
    RandomSignProjection::run(randomMatrix, data, &hashes);
    return hashes;
}
