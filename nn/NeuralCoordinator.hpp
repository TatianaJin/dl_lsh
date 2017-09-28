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

using namespace Eigen;

class NeuralCoordinator {
   public:
    static int INT_SIZE;
    static int LAYER_THREADS;
    static int UPDATE_SIZE;

    NeuralCoordinator(NN_parameters* params, std::vector<std::shared_ptr<NeuronLayer>> layers,
                      std::vector<std::shared_ptr<HiddenLayer>> hiddenLayers, double L2, ICostFunction* cf);

    void test(std::vector<VectorXd> data, std::vector<double> labels);

    void train(int max_epoch, std::vector<VectorXd> data, std::vector<double> labels, std::vector<VectorXd> test_data,
               std::vector<double> test_labels);

   private:
    std::vector<int> initIndices(int length);

    void shuffle(std::vector<int> indices);

    double calculateActiveNodes(double total);

    // void run(int start, int end, NeuralNetwork network, std::vector<std::vector<int>>
    // input_hashes, std::vector<VectorXd> data, std::vector<double> labels);

    int min(int ele1, int ele2);

    std::vector<NeuralNetwork> m_networks;
    NN_parameters* m_params;  // not owned
    string m_modelTitle;
    string m_model_path;
    string m_train_path;
    string m_test_path;
    double m_total_nodes;
    int update_threshold = 20;
};

#endif  // NEURALCOORDINATOR_HPP
