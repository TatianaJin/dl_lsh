#ifndef RANDOMPROJECTION_HPP
#define RANDOMPROJECTION_HPP

#include <Eigen/Dense>
#include <vector>

using Eigen::VectorXd;
using Eigen::MatrixXd;

class RandomSignProjection {
   public:
    static void run(const std::vector<MatrixXd>& projection_matrixi, const VectorXd& queryi, std::vector<int>* hashes);

   private:
    static int sign(double value);
};

#endif  // RANDOMPROJECTION_HPP
