#include <iostream>
#include "RandomProjection.hpp"
#include "CosineDistance.hpp"

CosineDistance::CosineDistance(int b, int L, int d)
{
    m_L = L;
    hashes.resize(m_L);
    for (int i = 0; i < m_L; ++i)
    {
        MatrixXd rand_m = MatrixXd::Random(b, d);
        randomMatrix.push_back(rand_m);
    }
}

vector<int> CosineDistance::hashSignature(VectorXd data)
{
    return RandomProjection(hashes, randomMatrix, data).run();
}
