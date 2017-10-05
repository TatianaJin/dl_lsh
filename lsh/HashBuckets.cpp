#include "HashBuckets.hpp"

#include "Histogram.hpp"
#include "Pooling.hpp"

HashBuckets::HashBuckets(double sizeLimit, int poolDim, int L, const CosineDistance& hashFunction) {
    m_hashFunction = hashFunction;
    m_poolDim = poolDim;
    m_nn_sizeLimit = sizeLimit;
    m_L = L;
    m_Tables.resize(m_L);
}

void HashBuckets::LSHAdd(int recIndex, const VectorXd& data) { LSHAdd(recIndex, generateHashSignature(data)); }

std::set<int> HashBuckets::histogramLSH(const VectorXd& data) { return histogramLSH(generateHashSignature(data)); }

std::set<int> HashBuckets::histogramLSH(const std::vector<int>& hashes) {
    assert(hashes.size() == m_L);

    Histogram hist;

    for (int idx = 0; idx < m_L; ++idx) {
        if (m_Tables[idx].find(hashes[idx]) != m_Tables[idx].end()) {
            hist.add(m_Tables[idx][hashes[idx]]);
        }
    }
    return hist.thresholdSet(m_nn_sizeLimit);
}

std::vector<int> HashBuckets::generateHashSignature(const VectorXd& data) {
    return m_hashFunction.hashSignature(Pooling::compress(m_poolDim, data));
}

void HashBuckets::clear() {
    m_Tables.clear();
    m_Tables.resize(m_L);
}

void HashBuckets::LSHAdd(int recIndex, const std::vector<int>& hashes) {
    assert(hashes.size() == m_L);
    for (int idx = 0; idx < m_L; ++idx) {
        m_Tables[idx][hashes[idx]].insert(recIndex);
    }
}
