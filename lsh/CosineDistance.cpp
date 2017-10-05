#include "CosineDistance.hpp"

#include <iostream>

#include "RandomProjection.hpp"

CosineDistance::CosineDistance(int b, int L, int d) {
    m_L = L;
    hashes.resize(m_L);
    randomMatrix.reserve(m_L);
    for (int i = 0; i < m_L; ++i) {
        // the assignment of coefficients follows uniform distrbution [-1,1]
        randomMatrix.push_back(MatrixXd::Random(b, d));
    }
}

std::vector<int> CosineDistance::hashSignature(VectorXd data) {
    RandomSignProjection::run(randomMatrix, data, &hashes);
    return hashes;
}
