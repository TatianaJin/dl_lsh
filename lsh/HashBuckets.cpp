#include <iostream>
#include "Histogram.hpp"
#include "Pooling.hpp"
#include "HashBuckets.hpp"

HashBuckets::HashBuckets(double sizeLimit, int poolDim, int L, CosineDistance hashFunction)
{
    m_hashFunction = hashFunction;
    m_poolDim = poolDim;
    m_nn_sizeLimit = sizeLimit;
    m_L = L;
    m_Tables.resize(m_L);
}

void HashBuckets::LSHAdd(int recIndex, VectorXd data)
{
    LSHAdd(recIndex, generateHashSignature(data));
}

set<int> HashBuckets::histogramLSH(VectorXd data)
{
    return histogramLSH(generateHashSignature(data));
}

set<int> HashBuckets::histogramLSH(vector<int> hashes)
{
    assert(hashes.size() == m_L);

    Histogram hist;

    for (int idx = 0; idx < m_L; ++idx)
    {
        if (m_Tables[idx].find(hashes[idx]) != m_Tables[idx].end())
        {
            hist.add(m_Tables[idx][hashes[idx]]);
        }
    }
    return hist.thresholdSet(m_nn_sizeLimit);
}

vector<int> HashBuckets::generateHashSignature(VectorXd data)
{
    return m_hashFunction.hashSignature(Pooling::compress(m_poolDim, data));
}

void HashBuckets::LSHAdd(int recIndex, vector<int> hashes)
{
    assert(hashes.size() == m_L);
    for (int idx = 0; idx < m_L; ++idx)
    {
        if (m_Tables[idx].find(hashes[idx]) == m_Tables[idx].end())
        {
            set<int> set;
            set.insert(recIndex);
            m_Tables[idx].insert(make_pair(hashes[idx], set));
        }
        else
        {
            m_Tables[idx][hashes[idx]].insert(recIndex);
        }
    }
}