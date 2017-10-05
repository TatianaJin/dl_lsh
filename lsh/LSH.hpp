#ifndef LSH_HPP
#define LSH_HPP

#include <Eigen/Dense>
#include <vector>

using Eigen::VectorXd;

class LSH {
   public:
    virtual std::vector<int> hashSignature(VectorXd data) = 0;
    virtual ~LSH() = default;
};

#endif  // LSH_HPP
