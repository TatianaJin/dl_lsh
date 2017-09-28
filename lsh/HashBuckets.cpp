#include "HashBuckets.hpp"

#include <iostream>

#include "Histogram.hpp"
#include "Pooling.hpp"

HashBuckets::HashBuckets(double sizeLimit, int poolDim, int L,
                         const CosineDistance& hashFunction) {  // TODO(tatiana): move instead of copy cosine distance
    m_hashFunction = hashFunction;
    m_poolDim = poolDim;
    m_nn_sizeLimit = sizeLimit;
    m_L = L;
    m_Tables.resize(m_L);
}

void HashBuckets::LSHAdd(int recIndex, VectorXd data) { LSHAdd(recIndex, generateHashSignature(data)); }

std::set<int> HashBuckets::histogramLSH(VectorXd data) { return histogramLSH(generateHashSignature(data)); }

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

std::vector<int> HashBuckets::generateHashSignature(VectorXd data) {
    return m_hashFunction.hashSignature(Pooling::compress(m_poolDim, data));
}

void HashBuckets::LSHAdd(int recIndex, std::vector<int> hashes) {
    assert(hashes.size() == m_L);
    for (int idx = 0; idx < m_L; ++idx) {
        if (m_Tables[idx].find(hashes[idx]) == m_Tables[idx].end()) {
            std::set<int> set;
            set.insert(recIndex);
            m_Tables[idx].insert(make_pair(hashes[idx], set));
        } else {
            m_Tables[idx][hashes[idx]].insert(recIndex);
        }
    }
}
