#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(NN_parameters* params, std::vector<std::shared_ptr<NeuronLayer>> layers, double L2,
                             ICostFunction* cf)
    : m_params(params), m_layers(std::move(layers)), L2_lambda(L2), m_cf(cf) {}

void NeuralNetwork::execute(VectorXd& input, double labels, bool training) {
    VectorXd y_hat = forwardPropagation(input, training);
    backPropagation(y_hat, labels);
    // TODO m_train_correct += m_cf->correct(y_hat, labels);
}

double NeuralNetwork::test(std::vector<std::vector<int>>& input_hashes, std::vector<VectorXd>& data,
                           std::vector<double>& labels) {
    std::vector<VectorXd> y_hat(labels.size());
    for (size_t idx = 0; idx < labels.size(); idx++) {
        y_hat[idx] = forwardPropagation(data[idx], input_hashes[idx], false);
    }
    return m_cf->accuracy(y_hat, labels);
}

VectorXd NeuralNetwork::forwardPropagation(VectorXd& input, std::vector<int>& hashes, bool training) {
    auto it = m_layers.begin();
    VectorXd data =
        static_cast<HiddenLayer*>(it->get())->forwardPropagation(input, hashes, training, m_params);

    while (++it != m_layers.end() - 1) {
        data = static_cast<HiddenLayer*>(it->get())->forwardPropagation(data, training, m_params);
    }
    return m_layers[m_layers.size() - 1]->forwardPropagation(data, m_params);
}

VectorXd NeuralNetwork::forwardPropagation(VectorXd& input, bool training) {
    auto it = m_layers.begin();
    VectorXd data = static_cast<HiddenLayer*>(it->get())->forwardPropagation(input, training, m_params);

    while (++it != m_layers.end() - 1) {
        data = static_cast<HiddenLayer*>(it->get())->forwardPropagation(data, training, m_params);
    }
    return m_layers[m_layers.size() - 1]->forwardPropagation(data, m_params);
}

void NeuralNetwork::backPropagation(VectorXd& y_hat, double labels) {
    // square loss function // tatiana: looks like log loss plus regularization?
    m_cost = m_cf->costFunction(y_hat, labels) + L2_lambda * m_params->L2_regularization();

    auto& outputLayer = m_layers[m_layers.size() - 1];

    // cost function derivatives
    VectorXd delta = m_cf->outputDelta(y_hat, labels, outputLayer);

    // calculate gradient for output layer, calculate delta for hidden layers
    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
        delta = (*it)->calculateDelta(delta, m_params);
    }

    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
        (*it)->calculateGradient(m_params);
    }
}

int NeuralNetwork::calculateActiveNodes() {  // FIXME the active node number is incorrect
    long total = 0;
    for (auto layer = m_layers.begin(); layer != m_layers.end() - 1; ++layer) {
        auto l = static_cast<HiddenLayer*>(layer->get());
        total += l->m_total_nn_set_size;
        std::cout << "HiddenLayer total nn set size = " << l->m_total_nn_set_size << std::endl;
        l->m_total_nn_set_size = 0;
    }
    total += m_layers[m_layers.size() - 1]->getLayerSize();  // no dropout in the output layer
    return total;
}

int NeuralNetwork::calculateMultiplications() {
    long total = 0;
    for (auto layer = m_layers.begin(); layer != m_layers.end() - 1; ++layer) {
        auto l = static_cast<HiddenLayer*>(layer->get());
        total += l->m_total_multiplication;
        l->m_total_multiplication = 0;
    }
    total += m_layers[m_layers.size() - 1]->numWeights();
    return total;
}

std::vector<std::vector<int>> NeuralNetwork::computeHashes(std::vector<VectorXd>& data) {
    return m_params->computeHashes(data);
}

void NeuralNetwork::updateHashTables(int miniBatch_size) {  // TODO obsolete
    for (auto& layer : m_layers) {
        auto l = static_cast<HiddenLayer*>(layer.get());
        l->updateHashTables(miniBatch_size);
    }
}

double NeuralNetwork::getGradient(int idx) { return m_params->getGradient(idx); }

double NeuralNetwork::getCost() { return m_cost; }

int NeuralNetwork::numTheta() { return m_params->size(); }

double NeuralNetwork::getTheta(int idx) { return m_params->getTheta(idx); }
