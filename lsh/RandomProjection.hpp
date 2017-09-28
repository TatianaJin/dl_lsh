#ifndef RANDOMPROJECTION_HPP
#define RANDOMPROJECTION_HPP

#include <Eigen/Dense>
#include <vector>

using Eigen::VectorXd;
using Eigen::MatrixXd;

class RandomProjection {  // TODO(Tatiana): just a functor, no need to have states
   public:
    RandomProjection(std::vector<int>* hashes, std::vector<MatrixXd>* projection_matrix, VectorXd* query);

    std::vector<int> run();

   private:
    int sign(double value);

    // not owned, got from CosineDistance
    std::vector<MatrixXd>* m_projection_matrix;
    VectorXd* m_query;
    std::vector<int>* m_hashes;
};

#endif  // RANDOMPROJECTION_HPP
