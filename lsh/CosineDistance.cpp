#include "CosineDistance.hpp"

#include <iostream>

#include "RandomProjection.hpp"

CosineDistance::CosineDistance(int b, int L, int d) {
    m_L = L;
    hashes.resize(m_L);
    randomMatrix.reserve(m_L);
    for (int i = 0; i < m_L; ++i) {
        MatrixXd rand_m = MatrixXd::Random(b, d);
        randomMatrix.push_back(rand_m);  // TODO(tatiana): check copy / move constructor of matrixxd
    }
}

std::vector<int> CosineDistance::hashSignature(VectorXd data) {  // TODO check if data need copy
    return RandomProjection(&hashes, &randomMatrix, &data).run();
}
