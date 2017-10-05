#include "ReLUNeuronLayer.hpp"

#include <memory>

ReLUNeuronLayer::ReLUNeuronLayer(int prev_layer_size, int layer_size, double L2)
    : HiddenLayer(prev_layer_size, layer_size, L2) {}

double ReLUNeuronLayer::weightInitialization() const {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double interval = 2.0 * sqrt(6.0 / (m_prev_layer_size + m_layer_size));  // TODO(Tatiana) why is this interval?
    return distribution(generator) * (2 * interval) - interval;
}

std::shared_ptr<NeuronLayer> ReLUNeuronLayer::clone() const {
    auto copy = std::make_shared<ReLUNeuronLayer>(m_prev_layer_size, m_layer_size, L2_lambda);
    copy->m_pos = m_pos;
    return copy;
}

VectorXd ReLUNeuronLayer::activationFunction(const std::vector<double>& input) {
    VectorXd output(input.size());
    int idx = 0;
    for (auto& ele : input) {
        output(idx) = std::max(ele, 0.0);
        idx += 1;
    }
    return output;
}

double ReLUNeuronLayer::derivative(double input) { return (input > 0) ? 1.0 : 0.0; }
