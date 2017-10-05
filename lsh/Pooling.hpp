#ifndef POOLING_HPP
#define POOLING_HPP

#include <Eigen/Dense>

using Eigen::VectorXd;

class Pooling {
   public:
    static VectorXd compress(int size, const VectorXd& data);

   private:
    static double mean(const VectorXd& data, int start, int end);
};

#endif  // POOLING_HPP
