#ifndef NN_PARAMETERS_HPP
#define NN_PARAMETERS_HPP

#include <Eigen/Dense>

#include <memory>
#include <set>
#include <vector>

#include "lsh/HashBuckets.hpp"
#include "nn/NeuronLayer.hpp"

using Eigen::VectorXd;

class NN_parameters {
   public:
    NN_parameters() = default;

    NN_parameters(const std::vector<std::shared_ptr<NeuronLayer>>& NN_structure, const std::vector<int>& pool_dim,
                  const std::vector<double>& size_limit, const std::vector<int>& b, const std::vector<int>& L,
                  double learning_rate);

    void construct(const std::vector<std::shared_ptr<NeuronLayer>>& NN_structure);

    int epoch_offset();

    void weight_initialization(const std::vector<std::shared_ptr<NeuronLayer>>& NN_structure);

    std::vector<std::vector<int>> computeHashes(std::vector<VectorXd>& data);

    void rebuildTables();

    void createLSHTable(std::vector<HashBuckets>* tables, const std::vector<int>& poolDim, const std::vector<int>& b,
                        const std::vector<int>& L, const std::vector<double>& size_limit);

    // consider replacing std::vector<double> with VectorXd
    std::vector<double> getWeight(int layer, int node);

    std::set<int> retrieveNodes(int layer, const VectorXd& input);

    std::set<int> retrieveNodes(int layer, const std::vector<int>& hashes);

    void timeStep();

    int size();

    double getGradient(int idx);

    double getTheta(int idx);

    void setTheta(int idx, double value);

    double getWeight(int layer, int row, int col);

    void setWeight(int layer, int row, int col, double value);

    std::vector<double> getWeightVector(int layer, int node);

    double getBias(int layer, int idx);

    void setBias(int layer, int idx, double value);

    double L2_regularization();

    int weightOffset(int layer, int row, int column);

    int biasOffset(int layer, int idx);

    void clear_gradient();

    void stochasticGradientDescent(int idx, double gradient);

    VectorXd vectorize(const std::vector<double>& data, int offset, int length);

    VectorXd vectorize(const std::vector<double>& data);

   private:
    int m_epoch_offset = 0;
    std::vector<std::shared_ptr<NeuronLayer>> m_layers;

    std::vector<int> m_weight_idx;
    std::vector<int> m_bias_idx;
    int m_size = 0;
    int m_layer_count = 0;
    std::vector<int> m_layer_row;
    std::vector<int> m_layer_col;

    // stochastic gradient decsent
    std::vector<double> m_theta;
    std::vector<double> m_gradient;

    // momentum
    std::vector<double> m_momentum;
    double m_momentum_lambda = 0.50;
    double momentum_max = 0.90;
    double momentum_rate = 1.00;

    // learning rate - adagrad
    double m_learning_rate;  // this one is final in java implementation
    std::vector<double> m_learning_rates;

    // LSH
    std::vector<HashBuckets> m_tables;
    std::vector<int> m_poolDim;
    std::vector<int> m_b;  // final
    std::vector<int> m_L;  // final
    std::vector<double> m_size_limit;
};

#endif  // NN_PARAMETERS_HPP
