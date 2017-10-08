#include "ReLUNeuronLayer.hpp"

#include <memory>

ReLUNeuronLayer::ReLUNeuronLayer(int prev_layer_size, int layer_size, double L2)
    : HiddenLayer(prev_layer_size, layer_size, L2) {}

double ReLUNeuronLayer::weightInitialization() const {
    static std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double interval = 2.0 * sqrt(6.0 / (m_prev_layer_size + m_layer_size));  // TODO(Tatiana) why is this interval?
    return distribution(generator) * (2 * interval) - interval;
}

std::shared_ptr<NeuronLayer<VectorXd>> ReLUNeuronLayer::clone() const {
    auto copy = std::make_shared<ReLUNeuronLayer>(m_prev_layer_size, m_layer_size, L2_lambda);
    copy->m_pos = m_pos;
    return copy;
}

VectorXd ReLUNeuronLayer::activationFunction(const VectorXd& input) {
    return input.unaryExpr([](double a) { return std::max(a, 0.); });
}

double ReLUNeuronLayer::derivative(double input) { return (input > 0) ? 1.0 : 0.0; }

ReLUNeuronLayerWithDropOut::ReLUNeuronLayerWithDropOut(int prev_layer_size, int layer_size, double L2)
    : HiddenLayerWithDropOut(prev_layer_size, layer_size, L2) {}

double ReLUNeuronLayerWithDropOut::weightInitialization() const {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double interval = 2.0 * sqrt(6.0 / (m_prev_layer_size + m_layer_size));
    return distribution(generator) * (2 * interval) - interval;
}

std::shared_ptr<NeuronLayer<SparseVectorXd>> ReLUNeuronLayerWithDropOut::clone() const {
    auto copy = std::make_shared<ReLUNeuronLayerWithDropOut>(m_prev_layer_size, m_layer_size, L2_lambda);
    copy->m_pos = m_pos;
    return copy;
}

SparseVectorXd ReLUNeuronLayerWithDropOut::activationFunction(const SparseVectorXd& input) {
    return input.unaryExpr([](double a) { return std::max(a, 0.); });
}

double ReLUNeuronLayerWithDropOut::derivative(double input) { return (input > 0) ? 1.0 : 0.0; }
