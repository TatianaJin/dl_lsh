#ifndef RANDOMPROJECTION_HPP
#define RANDOMPROJECTION_HPP

#include <Eigen/Dense>
#include <vector>

using namespace std;
using namespace Eigen;

class RandomProjection
{
public:
    RandomProjection(vector<int> hashes, vector<MatrixXd> projection_matrix, VectorXd query);

    vector<int> run();

private:
    int sign(double value);

    vector<MatrixXd> m_projection_matrix;
    VectorXd m_query;
    vector<int> m_hashes;
};

#endif // RANDOMPROJECTION_HPP