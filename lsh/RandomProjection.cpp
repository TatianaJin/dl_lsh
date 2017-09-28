#include "RandomProjection.hpp"

RandomProjection::RandomProjection(std::vector<int>* hashes, std::vector<MatrixXd>* projection_matrix,
                                   VectorXd* query) {
    m_projection_matrix = projection_matrix;
    m_query = query;
    m_hashes = hashes;
}

std::vector<int> RandomProjection::run() {  // random sign projection for cosine distance
    int hash_idx = -1;
    assert(m_hashes->size() == m_projection_matrix->size());
    for (size_t i = 0; i < m_projection_matrix->size(); ++i) {
        MatrixXd& projection = (*m_projection_matrix)[i];
        assert(projection.cols() == m_query->size());
        VectorXd dotProduct = projection * (*m_query);

        int signature = 0;
        for (int idx = 0; idx < dotProduct.size(); ++idx) {
            signature |= sign(dotProduct(idx));
            signature <<= 1;
        }
        (*m_hashes)[++hash_idx] = signature;
    }
    return *m_hashes;
}

int RandomProjection::sign(double value) { return (value >= 0) ? 1 : 0; }
