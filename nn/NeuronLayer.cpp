#include "NeuronLayer.hpp"
#include <iostream>
#include "NN_parameters.hpp"

NeuronLayer::NeuronLayer(int prev_layer_size, int layer_size, double L2)
    : m_weightedSum(layer_size, 0.0), m_prev_layer_size(prev_layer_size), m_layer_size(layer_size), L2_lambda(L2) {}

VectorXd NeuronLayer::activationFunction() { return activationFunction(m_weightedSum); }

VectorXd NeuronLayer::activationFunction(const std::vector<double>& input) { return VectorXd::Map(input.data(), input.size()); }

int NeuronLayer::numWeights() { return m_prev_layer_size * m_layer_size; }

int NeuronLayer::numBias() { return m_layer_size; }
