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

    NeuralNetwork(NN_parameters* params, std::vector<std::shared_ptr<NeuronLayer>> layers,
                  std::vector<std::shared_ptr<HiddenLayer>> hiddenLayers, double L2, ICostFunction* cf);

    void execute(std::vector<int> hashes, VectorXd input, double labels, bool training);

    double test(std::vector<std::vector<int>> input_hashes, std::vector<VectorXd> data, std::vector<double> labels);

    std::vector<double> forwardPropagation(VectorXd input, std::vector<int> hashes, bool training);

    void backPropagation(std::vector<double> y_hat, double labels);

    long calculateActiveNodes();

    long calculateMultiplications();

    std::vector<std::vector<int>> computeHashes(std::vector<VectorXd> data);

    void updateHashTables(int miniBatch_size);

    //------------- double check ---------------------

    double getGradient(int idx);

    double getCost();

    int numTheta();

    double getTheta(int idx);
    // ------------ double check -------------------

   protected:
    double m_train_correct;

   private:
    NN_parameters* m_params;                             // shared among neural networks, not owned
    std::vector<std::shared_ptr<NeuronLayer>> m_layers;  // m_layers - m_hidden_layers = outputLayer
    std::vector<std::shared_ptr<HiddenLayer>> m_hidden_layers;
    double L2_lambda;
    std::shared_ptr<ICostFunction> m_cf;  // shared
    double m_cost;
};

#endif  // NEURALNETWORK_HPP
