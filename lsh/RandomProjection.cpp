#include "RandomProjection.hpp"

#include <iostream>

#include "exp/colors.hpp"

void RandomSignProjection::run(const std::vector<MatrixXd>& projection_matrix, const VectorXd& query,
                               std::vector<int>* hashes) {  // random sign projection for cosine distance
    int hash_idx = -1;
    assert(hashes->size() == projection_matrix.size());
    for (size_t i = 0; i < projection_matrix.size(); ++i) {
        const MatrixXd& projection = projection_matrix[i];
        assert(projection.cols() == query.size());
        VectorXd dotProduct = projection * query;

        int signature = 0;
        signature |= sign(dotProduct(0));
        for (int idx = 1; idx < dotProduct.size(); ++idx) {
            signature <<= 1;
            signature |= sign(dotProduct(idx));
        }
        (*hashes)[++hash_idx] = signature;
    }
}

int RandomSignProjection::sign(double value) { return (value >= 0) ? 1 : 0; }
