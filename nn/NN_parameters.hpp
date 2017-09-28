#ifndef NN_PARAMETERS_HPP
#define NN_PARAMETERS_HPP

#include <set>
#include <vector>
#include <Eigen/Dense>

#include "NeuronLayer.hpp"
#include "../lsh/HashBuckets.hpp"

using namespace std;
using Eigen::VectorXd;

class NN_parameters
{
public:
    NN_parameters() = default;

    NN_parameters(vector<NeuronLayer*> NN_structure, vector<int> poolDim, vector<int> b, vector<int> L, double learning_rate, vector<double> size_limit); // : m_layers(NN_structure), m_learning_rate(learning_rate), m_poolDim(poolDim), m_b(b), m_L(L), m_size_limit(size_limit);

    void construct(vector<NeuronLayer*> NN_structure);

    int epoch_offset();

    void weight_initialization(vector<NeuronLayer*> NN_structure);

    vector<vector<int>> computeHashes(vector<VectorXd> data);

    void rebuildTables();

    void createLSHTable(vector<HashBuckets> tables, vector<int> poolDim, vector<int> b, vector<int> L, vector<double> size_limit);

    // consider replacing vector<double> with VectorXd
    vector<double> getWeight(int layer, int node);

    set<int> retrieveNodes(int layer, VectorXd input);

    set<int> retrieveNodes(int layer, vector<int> hashes);

    void timeStep();

    int size();

    double getGradient(int idx);

    double getTheta(int idx);

    void setTheta(int idx, double value);

    double getWeight(int layer, int row, int col);

    void setWeight(int layer, int row, int col, double value);

    vector<double> getWeightVector(int layer, int node);

    double getBias(int layer, int idx);

    void setBias(int layer, int idx, double value);

    double L2_regularization();

    int weightOffset(int layer, int row, int column);

    int biasOffset(int layer, int idx);

    void clear_gradient();

    void stochasticGradientDescent(int idx, double gradient);

    VectorXd vectorize(vector<double> data, int offset, int length);

    VectorXd vectorize(vector<double> data);

private:
    int m_epoch_offset;
    vector<NeuronLayer*> m_layers;

    vector<int> m_weight_idx;
    vector<int> m_bias_idx;
    int m_size = 0;
    int m_layer_count = 0;
    vector<int> m_layer_row;
    vector<int> m_layer_col;

    // stochastic gradient decsent
    vector<double> m_theta;
    vector<double> m_gradient;

    // momentum
    vector<double> m_momentum;
    double m_momentum_lambda = 0.50;
    double momentum_max = 0.90;
    double momentum_rate = 1.00;

    // learning rate - adagrad
    double m_learning_rate; // this one is final in java implementation
    vector<double> m_learning_rates;

    // LSH
    vector<HashBuckets> m_tables;
    vector<int> m_poolDim;
    vector<int> m_b; // final
    vector<int> m_L; // final
    vector<double> m_size_limit;
};

#endif // NN_PARAMETERS_HPP