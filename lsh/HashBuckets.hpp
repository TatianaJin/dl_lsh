#ifndef HASHBUCKETS_HPP
#define HASHBUCKETS_HPP

#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include <set>
#include "CosineDistance.hpp"

using namespace std;
using namespace Eigen;

class HashBuckets
{
public:
    HashBuckets() = default;
    
    HashBuckets(double sizeLimit, int poolDim, int L, CosineDistance hashFunction);

    void LSHAdd(int recIndex, VectorXd data);

    set<int> histogramLSH(VectorXd data);

    set<int> histogramLSH(vector<int> hashes);

    vector<int> generateHashSignature(VectorXd data);

private:
    void LSHAdd(int recIndex, vector<int> hashes);

    double m_nn_sizeLimit;
    int m_L;
    int m_poolDim;
    CosineDistance m_hashFunction;
    vector<unordered_map<int, set<int>>> m_Tables;
};

#endif // HASHBUCKETS_HPP
