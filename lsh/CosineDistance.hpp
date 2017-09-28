#ifndef COSINEDISTANCE_HPP
#define COSINEDISTANCE_HPP

#include <vector>
#include <Eigen/Dense>
#include "LSH.hpp"

using namespace std;
using namespace Eigen;

class CosineDistance : public LSH
{
public:
    CosineDistance() = default;
    CosineDistance(int b, int L, int d);
    vector<int> hashSignature(VectorXd data);

private:
    int m_L;
    vector<int> hashes;
    vector<MatrixXd> randomMatrix;
};

#endif // COSINEDISTANCE_HPP
