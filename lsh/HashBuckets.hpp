#ifndef HASHBUCKETS_HPP
#define HASHBUCKETS_HPP

#include <Eigen/Dense>
#include <set>
#include <unordered_map>
#include <vector>

#include "CosineDistance.hpp"

using Eigen::VectorXd;

class HashBuckets {
   public:
    HashBuckets() = default;

    HashBuckets(double sizeLimit, int poolDim, int L, const CosineDistance& hashFunction);

    void LSHAdd(int recIndex, VectorXd data);

    std::set<int> histogramLSH(VectorXd data);

    std::set<int> histogramLSH(const std::vector<int>& hashes);

    std::vector<int> generateHashSignature(VectorXd data);

   private:
    void LSHAdd(int recIndex, std::vector<int> hashes);

    double m_nn_sizeLimit;
    int m_L;
    int m_poolDim;
    CosineDistance m_hashFunction;
    std::vector<unordered_map<int, std::set<int>>> m_Tables;
};

#endif  // HASHBUCKETS_HPP
