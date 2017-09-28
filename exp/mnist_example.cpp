#include <cassert>
#include <chrono>
#include <ctime>
#include <memory>
#include <string>

#include <Eigen/Dense>

#include "dataset/MNISTDataSet.hpp"
#include "nn/HiddenLayer.hpp"
#include "nn/ICostFunction.hpp"
#include "nn/NeuralCoordinator.hpp"
#include "nn/NeuronLayer.hpp"
#include "nn/ReLUNeuronLayer.hpp"
#include "nn/SoftMaxNeuronLayer.hpp"

using Eigen::VectorXd;

int min_layers = 1;
int max_layers = 1;
int hidden_layer_size = 1000;
int hidden_pool_size = hidden_layer_size * 0.1;

// Neural Network Parameters TODO avoid using global variables in this way
std::string dataset;
int training_size;
int test_size;
int inputLayer;
int outputLayer;
int k;
int b = 6;
int L = 100;

int max_epoch = 25;
double L2_Lambda = 0.003;
std::vector<int> hiddenLayers;
std::vector<double> learning_rates;
std::vector<double> size_limits = {0.05, 0.10, 0.25, 0.5, 0.75, 1.0};

std::vector<std::shared_ptr<HiddenLayer>> hidden_layers;
std::vector<std::shared_ptr<NeuronLayer>> NN_layers;

std::string make_title() {
    auto title = dataset + '_' + "LSH" + '_' + std::to_string(inputLayer) + '_';
    for (size_t idx = 0; idx < hiddenLayers.size(); ++idx) {
        title += std::to_string(hiddenLayers[idx]) + '_';
    }
    title += std::to_string(outputLayer);
    return title;
}

void construct(int inputLayer, int outputLayer) {
    // Clear up
    std::vector<std::shared_ptr<HiddenLayer>>().swap(hidden_layers);
    std::vector<std::shared_ptr<NeuronLayer>>().swap(NN_layers);

    // input layer
    hidden_layers.push_back(std::make_shared<ReLUNeuronLayer>(inputLayer, hiddenLayers[0], L2_Lambda));
    for (size_t idx = 0; idx < hiddenLayers.size() - 1; ++idx) {
        hidden_layers.push_back(std::make_shared<ReLUNeuronLayer>(hiddenLayers[idx], hiddenLayers[idx + 1], L2_Lambda));
    }
    for (auto& hidden_layer : hidden_layers) {
        NN_layers.push_back(std::shared_ptr<NeuronLayer>(hidden_layer));
    }
    // Output Layers
    NN_layers.push_back(
        std::make_shared<SoftMaxNeuronLayer>(hiddenLayers[hiddenLayers.size() - 1], outputLayer, L2_Lambda));
}

void execute(std::vector<VectorXd> train_data, std::vector<double> train_labels, std::vector<VectorXd> test_data,
             std::vector<double> test_labels) {
    assert(size_limits.size() == learning_rates.size());

    for (int size = min_layers; size <= max_layers; ++size) {
        // Network structures
        hiddenLayers.resize(size);
        hiddenLayers.assign(size, hidden_layer_size);

        // For hashing
        std::vector<int> sum_pool(size, hidden_pool_size);
        sum_pool[0] = k;
        std::vector<int> bits(size, b);
        std::vector<int> tables(size, L);

        for (size_t idx = 0; idx < size_limits.size(); idx++) {
            // every layer has the same size limits
            std::vector<double> sl(size, size_limits[idx]);

            // Construct the neural network
            construct(inputLayer, outputLayer);
            // only one copy of parameters
            NN_parameters parameters(NN_layers, sum_pool, sl, bits, tables, learning_rates[idx]);
            ICostFunction cost_func;
            NeuralCoordinator NN(&parameters, std::move(NN_layers), std::move(hidden_layers), L2_Lambda, &cost_func);
            std::cout << "Neural network deployed." << std::endl;
            // Start training
            auto startTime = std::chrono::high_resolution_clock::now();
            NN.train(max_epoch, train_data, train_labels, test_data, test_labels);
            auto endTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
            std::cout << "Training time: " << elapsed << "ms." << std::endl;
        }
    }
}

void testMNIST(std::string training_label_path, std::string training_image_path, std::string test_label_path,
               std::string test_image_path) {
    dataset = "MNIST";
    training_size = 60000;
    test_size = 10000;
    inputLayer = 784;
    outputLayer = 10;
    k = 98;
    learning_rates = {1e-2, 1e-2, 1e-2, 5e-3, 1e-3, 1e-3};

    auto title = make_title();
    std::cout << title << std::endl;

    // Read MNIST test and training data
    std::pair<std::vector<VectorXd>, std::vector<double>> training =
        MNISTDataSet::loadDataSet(training_label_path, training_image_path);
    std::pair<std::vector<VectorXd>, std::vector<double>> test =
        MNISTDataSet::loadDataSet(test_label_path, test_image_path);

    execute(training.first, training.second, test.first, test.second);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: MNISTExample <training_label_path> "
                     "<training_image_path> <test_label_path> "
                     "<test_image_path>" << std::endl;
        return 0;
    }

    // Read program args
    std::string training_label_path = argv[1];
    std::string training_image_path = argv[2];
    std::string test_label_path = argv[3];
    std::string test_image_path = argv[4];

    testMNIST(training_label_path, training_image_path, test_label_path, test_image_path);
    return 0;
}
