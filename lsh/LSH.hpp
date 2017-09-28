#ifndef LSH_HPP
#define LSH_HPP

#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class LSH
{
public:
    virtual vector<int> hashSignature(VectorXd data) = 0;
    virtual ~LSH() = default;
};

#endif // LSH_HPP