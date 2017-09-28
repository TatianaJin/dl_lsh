#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(NN_parameters* params, std::vector<std::shared_ptr<NeuronLayer>> layers,
                             std::vector<std::shared_ptr<HiddenLayer>> hiddenLayers, double L2, ICostFunction* cf)
    : m_params(params),
      m_layers(std::move(layers)),
      m_hidden_layers(std::move(hiddenLayers)),
      L2_lambda(L2),
      m_cf(cf) {}

void NeuralNetwork::execute(std::vector<int> hashes, VectorXd input, double labels, bool training) {
    std::vector<double> y_hat = forwardPropagation(input, hashes, training);
    backPropagation(y_hat, labels);
    m_train_correct += m_cf->correct(y_hat, labels);
}

double NeuralNetwork::test(std::vector<std::vector<int>> input_hashes, std::vector<VectorXd> data,
                           std::vector<double> labels) {
    std::vector<std::vector<double>> y_hat(labels.size());
    for (size_t idx = 0; idx < labels.size(); idx++) {
        y_hat[idx] = forwardPropagation(data[idx], input_hashes[idx], false);
    }
    return m_cf->accuracy(y_hat, labels);
}

std::vector<double> NeuralNetwork::forwardPropagation(VectorXd input, std::vector<int> hashes, bool training) {
    auto it = m_hidden_layers.begin();
    std::vector<double> data = (*it++)->forwardPropagation(input, hashes, training, m_params);

    while (it != m_hidden_layers.end()) {
        data = (*it++)->forwardPropagation(data, training, m_params);
    }
    return m_layers[m_layers.size() - 1]->forwardPropagation(data, m_params);
}

void NeuralNetwork::backPropagation(std::vector<double> y_hat, double labels) {
    // square loss function
    m_cost = m_cf->costFunction(y_hat, labels) + L2_lambda * m_params->L2_regularization();

    auto outputLayer = m_layers[m_layers.size() - 1];

    // cost function derivatives
    std::vector<double> delta = m_cf->outputDelta(y_hat, labels, outputLayer);

    // calculate gradient for output layer, calculate delta for hidden layers
    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
        delta = (*it)->calculateDelta(delta, m_params);
    }

    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
        (*it)->calculateGradient(m_params);
    }
}

long NeuralNetwork::calculateActiveNodes() {
    long total = 0;
    for (auto& l : m_hidden_layers) {
        total += l->m_total_nn_set_size;  // to be defined
        l->m_total_nn_set_size = 0;
    }
    total += m_layers[m_layers.size() - 1]->m_layer_size;
    return total;
}

long NeuralNetwork::calculateMultiplications() {
    long total = 0;
    for (auto& l : m_hidden_layers) {
        total += l->m_total_multiplication;
        l->m_total_multiplication = 0;
    }
    total += m_layers[m_layers.size() - 1]->numWeights();
    return total;
}

std::vector<std::vector<int>> NeuralNetwork::computeHashes(std::vector<VectorXd> data) {
    return m_params->computeHashes(data);
}

void NeuralNetwork::updateHashTables(int miniBatch_size) {
    for (auto& l : m_hidden_layers) {
        l->updateHashTables(miniBatch_size);
    }
}

//------------- double check ---------------------

double NeuralNetwork::getGradient(int idx) { return m_params->getGradient(idx); }

double NeuralNetwork::getCost() { return m_cost; }

int NeuralNetwork::numTheta() { return m_params->size(); }

double NeuralNetwork::getTheta(int idx) { return m_params->getTheta(idx); }
