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

class NeuralNetwork {
   public:
    NeuralNetwork() = default;

    NeuralNetwork(NN_parameters* params, std::vector<std::shared_ptr<NeuronLayer>> layers, double L2,
                  ICostFunction* cf);

    void execute(VectorXd& input, double labels, bool training);

    double test(std::vector<std::vector<int>>& input_hashes, std::vector<VectorXd>& data, std::vector<double>& labels);

    VectorXd forwardPropagation(VectorXd& input, std::vector<int>& hashes, bool training);
    VectorXd forwardPropagation(VectorXd& input, bool training);

    void backPropagation(VectorXd& y_hat, double labels);

    std::vector<std::vector<int>> computeHashes(std::vector<VectorXd>& data);

    void updateHashTables(int miniBatch_size);

    // getters for stats
    int calculateActiveNodes();
    int calculateMultiplications();
    double getGradient(int idx);
    double getCost();
    double getTheta(int idx);
    double getTrainAccuracy() { return m_train_correct; }
    int numTheta();

   private:
    NN_parameters* m_params;                             // shared among neural networks, not owned
    std::vector<std::shared_ptr<NeuronLayer>> m_layers;  // the last layer is output layer
    double L2_lambda;
    ICostFunction* m_cf;  // shared
    double m_cost;
    double m_train_correct;
};

#endif  // NEURALNETWORK_HPP
