#ifndef NEURALCOORDINATOR_HPP
#define NEURALCOORDINATOR_HPP

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "HiddenLayer.hpp"
#include "ICostFunction.hpp"
#include "NN_parameters.hpp"
#include "NeuralNetwork.hpp"
#include "NeuronLayer.hpp"

using Eigen::VectorXd;

class NeuralCoordinator {
   public:
    static int LAYER_THREADS;  // number of threads to run concurrently
    static int UPDATE_SIZE;    // number of hash table updates in one epoch

    NeuralCoordinator() = default;
    NeuralCoordinator(NN_parameters* params, std::vector<std::shared_ptr<NeuronLayer>> layers, double L2,
                      ICostFunction* cf);

    void test(std::vector<VectorXd>& data, std::vector<double>& labels);

    void train(int max_epoch, std::vector<VectorXd> data, std::vector<double> labels, std::vector<VectorXd> test_data,
               std::vector<double> test_labels);

   private:
    std::vector<int> initIndices(int length);

    void shuffle(std::vector<int>& indices);

    double calculateActiveNodes(double total);

    NN_parameters* m_params;  // not owned
    std::vector<NeuralNetwork> m_networks;
    double m_total_nodes = 0.;
    int update_threshold = 20;
};

#endif  // NEURALCOORDINATOR_HPP
