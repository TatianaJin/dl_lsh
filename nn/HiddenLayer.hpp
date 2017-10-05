#ifndef HIDDENLAYER_HPP
#define HIDDENLAYER_HPP

#include <iostream>
#include <set>
#include <vector>

#include <Eigen/Dense>

#include "NN_parameters.hpp"
#include "NeuronLayer.hpp"

using Eigen::VectorXd;

class HiddenLayer : public NeuronLayer {
   public:
    HiddenLayer(int prev_layer_size, int layer_size, double L2);

    virtual double derivative(double input) = 0;

    VectorXd forwardPropagation(const VectorXd& input, std::set<int> nn_node_set, bool training,
                                           NN_parameters* m_theta);

    // some forwardPropagation functions should be added later
    VectorXd forwardPropagation(const VectorXd& input, NN_parameters* m_theta);

    VectorXd forwardPropagation(const std::vector<double>& input, bool training, NN_parameters* m_theta);

    VectorXd forwardPropagation(const VectorXd& input, bool training, NN_parameters* m_theta);

    VectorXd forwardPropagation(const VectorXd& input, const std::vector<int>& hashes, bool training,
                                           NN_parameters* m_theta);

    VectorXd calculateDelta(const VectorXd& prev_layer_delta, NN_parameters* m_theta);

    void calculateGradient(NN_parameters* m_theta);

    void updateHashTables(double size);

    VectorXd vectorize(const std::vector<double>& data, int offset, int length);

    VectorXd vectorize(const std::vector<double>& data);

    std::set<int> m_node_set;  // active node set in one forward propagation
    unsigned long long m_total_nn_set_size = 0;
    unsigned long long m_total_multiplication = 0;
};

#endif  // HIDDENLAYER_HPP
