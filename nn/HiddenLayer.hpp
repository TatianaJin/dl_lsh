#ifndef HIDDENLAYER_HPP
#define HIDDENLAYER_HPP

#include <iostream>
#include <set>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "NN_parameters.hpp"
#include "NeuronLayer.hpp"

using Eigen::VectorXd;
using SparseVectorXd = Eigen::SparseVector<double>;

class HiddenLayer : public NeuronLayer<VectorXd> {
   public:
    HiddenLayer(int prev_layer_size, int layer_size, double L2);

    virtual double derivative(double input) = 0;

    VectorXd forwardPropagation(const VectorXd& input, NN_parameters* m_theta) override;
    VectorXd forwardPropagation(const VectorXd& input, bool training, NN_parameters* m_theta);

    VectorXd calculateDelta(const VectorXd& prev_layer_delta, NN_parameters* m_theta) override;
    void calculateGradient(NN_parameters* m_theta) override;
};

class HiddenLayerWithDropOut : public NeuronLayer<SparseVectorXd> {
   public:
    HiddenLayerWithDropOut(int prev_layer_size, int layer_size, double L2);

    virtual double derivative(double input) = 0;

    SparseVectorXd forwardPropagation(const std::vector<double>& input, bool training, NN_parameters* m_theta);
    SparseVectorXd forwardPropagation(const SparseVectorXd& input, bool training, NN_parameters* m_theta);
    SparseVectorXd forwardPropagation(const SparseVectorXd& input, NN_parameters* m_theta) override;

    SparseVectorXd forwardPropagation(const SparseVectorXd& input, std::set<int> nn_node_set, bool training,
                                      NN_parameters* m_theta);
    SparseVectorXd forwardPropagation(const SparseVectorXd& input, const std::vector<int>& hashes, bool training,
                                      NN_parameters* m_theta);

    SparseVectorXd calculateDelta(const SparseVectorXd& prev_layer_delta, NN_parameters* m_theta) override;
    void calculateGradient(NN_parameters* m_theta) override;

    std::set<int> m_node_set;  // active node set in one forward propagation
    unsigned long long m_total_nn_set_size = 0;
    unsigned long long m_total_multiplication = 0;
};

#endif  // HIDDENLAYER_HPP
