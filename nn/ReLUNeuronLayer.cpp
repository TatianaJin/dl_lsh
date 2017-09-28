#include <memory>

#include "ReLUNeuronLayer.hpp"

ReLUNeuronLayer::ReLUNeuronLayer(int prev_layer_size, int layer_size, double L2)
    : HiddenLayer(prev_layer_size, layer_size, L2) {}

double ReLUNeuronLayer::weightInitialization() const {
    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<double> distribution(0.0, 1.0);
    double interval = 2.0 * sqrt(6.0 / (m_prev_layer_size + m_layer_size));  // TODO(Tatiana) why is this interval?
    return distribution(generator) * (2 * interval) - interval;
}

std::shared_ptr<NeuronLayer> ReLUNeuronLayer::clone() const {
    return std::make_shared<ReLUNeuronLayer>(m_prev_layer_size, m_layer_size, L2_lambda);
}

std::vector<double> ReLUNeuronLayer::activationFunction(std::vector<double> input) {
    std::vector<double> output;
    output.reserve(input.size());
    for (auto& ele : input) {
        output.push_back(max(ele, 0.0));
    }
    return output;
}

double ReLUNeuronLayer::derivative(double input) { return (input > 0) ? 1.0 : 0.0; }
