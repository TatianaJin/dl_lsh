#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>
#include "HiddenLayer.hpp"
#include "ICostFunction.hpp"
#include "NeuronLayer.hpp"

using Eigen::VectorXd;

template <typename Vector>
class NeuralNetwork {
   public:
    NeuralNetwork() = default;

    NeuralNetwork(NN_parameters* params, std::vector<std::shared_ptr<NeuronLayer<Vector>>> layers, double L2,
                  ICostFunction<Vector>* cf)
        : m_params(params), m_layers(std::move(layers)), L2_lambda(L2), m_cf(cf) {}

    void execute(VectorXd& input, double labels, bool training) {
        Vector y_hat = forwardPropagation(input, training);
        backPropagation(y_hat, labels);
        m_train_correct += m_cf->correct(y_hat, labels);
    }

    double test(std::vector<VectorXd>& data, std::vector<double>& labels) {
        std::vector<Vector> y_hat(labels.size());
        for (size_t idx = 0; idx < labels.size(); idx++) {
            y_hat[idx] = forwardPropagation(data[idx], false);
        }
        return m_cf->accuracy(y_hat, labels);
    }

    auto forwardPropagation(VectorXd& input, bool training) {
        std::cout << input << "\ntraining: " << training << std::endl;
        std::cout << RED("[WARN] ") << "forwardPropagation function supports only template specialization" << std::endl;
    }

    void backPropagation(Vector& y_hat, double labels) {
        // square loss function // tatiana: looks like log loss plus regularization?
        m_cost = m_cf->costFunction(y_hat, labels) + L2_lambda * m_params->L2_regularization();

        auto& outputLayer = m_layers[m_layers.size() - 1];

        // cost function derivatives
        auto delta = m_cf->outputDelta(y_hat, labels, outputLayer);

        // calculate gradient for output layer, calculate delta for hidden layers
        for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
            delta = (*it)->calculateDelta(delta, m_params);
        }

        for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
            (*it)->calculateGradient(m_params);
        }
    }

    std::vector<std::vector<int>> computeHashes(std::vector<VectorXd>& data) { return m_params->computeHashes(data); }

    // getters for stats
    int calculateActiveNodes() {
        int total = 0;
        for (auto& layer : m_layers) total += layer->getLayerSize();
        return total;
    }

    int calculateMultiplications() {
        int total = 0;
        for (auto& layer : m_layers) total += layer->numWeights();
        return total;
    }

    double getGradient(int idx) { return m_params->getGradient(idx); }
    double getCost() { return m_cost; }
    double getTheta(int idx) { return m_params->getTheta(idx); }

    int numTheta() { auto tmp = m_train_correct; m_train_correct = 0; return tmp; }
    void set_dropout(bool b) { m_dropout = b; }

   private:
    NN_parameters* m_params;                                     // shared among neural networks, not owned
    std::vector<std::shared_ptr<NeuronLayer<Vector>>> m_layers;  // the last layer is output layer
    double L2_lambda;
    ICostFunction<Vector>* m_cf;  // shared
    double m_cost;
    double m_train_correct;
    bool m_dropout = false;
};

template <>
int NeuralNetwork<SparseVectorXd>::calculateActiveNodes() {
    if (!m_dropout) return -1;  // passive checking, bad
    long total = 0;
    for (auto layer = m_layers.begin(); layer != m_layers.end() - 1; ++layer) {
        auto l = static_cast<HiddenLayerWithDropOut*>(layer->get());
        total += l->m_total_nn_set_size;
        l->m_total_nn_set_size = 0;
    }
    total += m_layers[m_layers.size() - 1]->getLayerSize();  // no dropout in the output layer
    return total;
}

template <>
int NeuralNetwork<SparseVectorXd>::calculateMultiplications() {
    if (!m_dropout) return -1;  // passive checking, bad
    long total = 0;
    for (auto layer = m_layers.begin(); layer != m_layers.end() - 1; ++layer) {
        auto l = static_cast<HiddenLayerWithDropOut*>(layer->get());
        total += l->m_total_multiplication;
        l->m_total_multiplication = 0;
    }
    total += m_layers[m_layers.size() - 1]->numWeights();
    return total;
}

template <>
auto NeuralNetwork<VectorXd>::forwardPropagation(VectorXd& input, bool training) {
    auto it = m_layers.begin();
    auto data = static_cast<HiddenLayer*>(it->get())->forwardPropagation(input, training, m_params);
    while (++it != m_layers.end() - 1) {
        data = static_cast<HiddenLayer*>(it->get())->forwardPropagation(data, training, m_params);
    }
    return m_layers[m_layers.size() - 1]->forwardPropagation(data, m_params);
}

template <>
auto NeuralNetwork<SparseVectorXd>::forwardPropagation(VectorXd& input, bool training) {
    auto it = m_layers.begin();
    auto data =
        static_cast<HiddenLayerWithDropOut*>(it->get())->forwardPropagation(input.sparseView(), training, m_params);
    while (++it != m_layers.end() - 1) {
        data = static_cast<HiddenLayerWithDropOut*>(it->get())->forwardPropagation(data, training, m_params);
    }
    return m_layers[m_layers.size() - 1]->forwardPropagation(data, m_params);
}

#endif  // NEURALNETWORK_HPP
