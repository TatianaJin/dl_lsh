#ifndef POOLING_HPP
#define POOLING_HPP

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class Pooling
{
public:
    static VectorXd compress(int size, VectorXd data);

private:
    static double sum(VectorXd data, int start, int end);
};

#endif // POOLING_HPP
