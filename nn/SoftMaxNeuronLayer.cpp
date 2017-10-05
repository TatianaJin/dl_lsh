#include <memory>

#include "SoftMaxNeuronLayer.hpp"

SoftMaxNeuronLayer::SoftMaxNeuronLayer(int prev_layer_size, int layer_size, double L2)
    : NeuronLayer(prev_layer_size, layer_size, L2) {}

std::shared_ptr<NeuronLayer> SoftMaxNeuronLayer::clone() const {
    auto copy = std::make_shared<SoftMaxNeuronLayer>(m_prev_layer_size, m_layer_size, L2_lambda);
    copy->m_pos = this->m_pos;
    return copy;
}

VectorXd SoftMaxNeuronLayer::forwardPropagation(const VectorXd& input, NN_parameters* m_theta) {
    assert(input.size() == m_prev_layer_size);
    m_input = input;

    for (int jdx = 0; jdx < m_layer_size; ++jdx) {
        m_weightedSum[jdx] = 0.0;
        for (int idx = 0; idx < m_prev_layer_size; ++idx) {
            m_weightedSum[jdx] += m_theta->getWeight(m_pos, jdx, idx) * m_input(idx);
        }
        m_weightedSum[jdx] += m_theta->getBias(m_pos, jdx);
    }
    return activationFunction(m_weightedSum);
}

VectorXd SoftMaxNeuronLayer::calculateDelta(const VectorXd& prev_layer_delta,
                                                       NN_parameters* m_theta) {
    assert(prev_layer_delta.size() == m_layer_size);
    m_delta = prev_layer_delta;
    return m_delta;
}

void SoftMaxNeuronLayer::calculateGradient(NN_parameters* m_theta) {
    assert(m_delta.size() == m_layer_size);
    for (int idx = 0; idx < m_layer_size; ++idx) {
        // set weight gradient
        for (int jdx = 0; jdx < m_prev_layer_size; ++jdx) {
            m_theta->stochasticGradientDescent(
                m_theta->weightOffset(m_pos, idx, jdx),
                m_delta[idx] * m_input(jdx) + L2_lambda * m_theta->getWeight(m_pos, idx, jdx));
        }
        // set bias gradient
        m_theta->stochasticGradientDescent(m_theta->biasOffset(m_pos, idx), m_delta[idx]);
    }
}

double SoftMaxNeuronLayer::weightInitialization() const {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double interval = 2.0 * sqrt(6.0 / (m_prev_layer_size + m_layer_size));
    return distribution(generator) * (2 * interval) - interval;
}

VectorXd SoftMaxNeuronLayer::activationFunction(const std::vector<double>& input) {
    double sum = 0.0;
    VectorXd output(input.size());
    int idx = 0;
    for (auto& ele : input) {
        double temp = std::exp(ele);
        output(idx) = temp;
        sum += temp;
    }
    output = output / sum;
    return output;
}
