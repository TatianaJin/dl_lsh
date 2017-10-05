#include <Eigen/Dense>
#include <cassert>
#include <chrono>
#include <ctime>
#include <memory>
#include <string>

#include "dataset/MNISTDataSet.hpp"
#include "nn/HiddenLayer.hpp"
#include "nn/ICostFunction.hpp"
#include "nn/NeuralCoordinator.hpp"
#include "nn/NeuronLayer.hpp"
#include "nn/ReLUNeuronLayer.hpp"
#include "nn/SoftMaxNeuronLayer.hpp"

#include "exp/colors.hpp"

using Eigen::VectorXd;

int min_layers = 1;
int max_layers = 1;

class FCLExample {
   public:
    FCLExample() {}
    FCLExample(int hidden_layer_size, int hidden_pool_size, int num_hidden_layers, int b, int L, double size_limit,
               double L2_Lambda, double learning_rate)
        : hidden_layer_size_(hidden_layer_size),
          hidden_pool_size_(hidden_pool_size),
          num_hidden_layers_(num_hidden_layers),
          b_(b),
          L_(L),
          size_limit_(size_limit),
          L2_Lambda_(L2_Lambda),
          learning_rate_(learning_rate) {}

    void initialize() {
        // 1. Construct the neural network
        construct(inputLayer_, outputLayer_, num_hidden_layers_);

        // 2. Initialize weights and hash tables
        // For hashing
        std::vector<int> sum_pool(num_hidden_layers_, hidden_pool_size_);
        sum_pool[0] = k_;
        std::vector<double> sl(num_hidden_layers_, size_limit_);  // every layer has the same size limits
        std::vector<int> bits(num_hidden_layers_, b_);
        std::vector<int> tables(num_hidden_layers_, L_);
        parameters_ = NN_parameters(NN_layers_, sum_pool, sl, bits, tables, learning_rate_);  // only one copy

        // 3. Build controller
        coordinator_ = NeuralCoordinator(&parameters_, NN_layers_, L2_Lambda_, &cost_func_);
        std::cout << GREEN("[INFO] ") << "Neural network deployed." << std::endl;
    }

    void train(std::vector<VectorXd> train_data, std::vector<double> train_labels, std::vector<VectorXd> test_data,
               std::vector<double> test_labels, int max_epoch) {
        auto startTime = std::chrono::high_resolution_clock::now();
        coordinator_.train(max_epoch, train_data, train_labels, test_data, test_labels);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << GREEN("[INFO] ") << "Training time: " << elapsed << "ms." << std::endl;
    }

   protected:
    std::string make_title(const std::string& dataset, const std::vector<int>& hiddenLayers) {
        auto title = dataset + '_' + "LSH" + '_' + std::to_string(inputLayer_) + '_';
        for (size_t idx = 0; idx < hiddenLayers.size(); ++idx) {
            title += std::to_string(hiddenLayers[idx]) + '_';
        }
        title += std::to_string(outputLayer_);
        return title;
    }

    void construct(int inputLayer, int outputLayer, int num_hidden_layer) {
        std::vector<int> hiddenLayers(num_hidden_layer, hidden_layer_size_);
        std::cout << make_title(dataset_, hiddenLayers) << std::endl;
        // Clear up
        NN_layers_.clear();

        NN_layers_.reserve(num_hidden_layer + 1);
        // input layer
        NN_layers_.push_back(std::make_shared<ReLUNeuronLayer>(inputLayer, hiddenLayers[0], L2_Lambda_));
        for (int idx = 0; idx < num_hidden_layer - 1; ++idx) {
            NN_layers_.push_back(
                std::make_shared<ReLUNeuronLayer>(hiddenLayers[idx], hiddenLayers[idx + 1], L2_Lambda_));
        }
        // Output Layer
        NN_layers_.push_back(
            std::make_shared<SoftMaxNeuronLayer>(hiddenLayers[num_hidden_layer - 1], outputLayer, L2_Lambda_));
    }

    NN_parameters parameters_;
    NeuralCoordinator coordinator_;
    std::vector<std::shared_ptr<NeuronLayer>> NN_layers_;
    ICostFunction cost_func_;

    std::string dataset_;
    int training_size_ = 0;
    int test_size_ = 0;

    int hidden_layer_size_ = 1000;
    int hidden_pool_size_ = 100;
    int num_hidden_layers_ = 1;
    int inputLayer_;
    int outputLayer_;

    int k_;  // input layer pools
    int b_ = 6;
    int L_ = 100;
    double size_limit_ = 0.05;  // dropout remaining percentage

    double L2_Lambda_ = 0.003;
    double learning_rate_ = 1e-2;

};  // class FCLExample

class MNISTExample : public FCLExample {
   public:
    MNISTExample() : FCLExample() {
        training_size_ = 60000;
        test_size_ = 10000;
        inputLayer_ = 784;
        outputLayer_ = 10;
        k_ = 98;
        dataset_ = "MNIST";
    }
    MNISTExample(int hidden_layer_size, int hidden_pool_size, int num_hidden_layers, int b, int L, double size_limit,
                 double L2_Lambda, double learning_rate)
        : FCLExample(hidden_layer_size, hidden_pool_size, num_hidden_layers, b, L, size_limit, L2_Lambda,
                     learning_rate) {
        training_size_ = 60000;
        test_size_ = 10000;
        inputLayer_ = 784;
        outputLayer_ = 10;
        k_ = 98;
        dataset_ = "MNIST";
    }

    void testMNIST(std::string training_label_path, std::string training_image_path, std::string test_label_path,
                   std::string test_image_path, int max_epoch) {
        // Read MNIST test and training data
        std::pair<std::vector<VectorXd>, std::vector<double>> training =
            MNISTDataSet::loadDataSet(training_label_path, training_image_path);
        std::pair<std::vector<VectorXd>, std::vector<double>> test =
            MNISTDataSet::loadDataSet(test_label_path, test_image_path);

        initialize();
        train(training.first, training.second, test.first, test.second, max_epoch);
    }
};  // class MNISTExample

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cout << "Usage: MNISTExample <training_label_path> "
                     "<training_image_path> <test_label_path> "
                     "<test_image_path> <max_epoch> [<hidden_layer_size>] [<hidden_pool_size_>] [<num_hidden_layers>] "
                     "[<b>] [<L>] [<size_limit>] [<L2_Lambda>] [<learning_rate>]"
                  << std::endl;
        return 0;
    }

    // Read program args
    std::string training_label_path = argv[1];
    std::string training_image_path = argv[2];
    std::string test_label_path = argv[3];
    std::string test_image_path = argv[4];

    int max_epoch = std::atoi(argv[5]);
    if (argc == 14) {
        int hidden_layer_size = std::atoi(argv[6]);
        int hidden_pool_size = std::atoi(argv[7]);
        int num_hidden_layers = std::atoi(argv[8]);
        int b = std::atoi(argv[9]);
        int L = std::atoi(argv[10]);
        double size_limit = std::atof(argv[11]);
        double L2_Lambda = std::atoi(argv[12]);
        double learning_rate = std::atoi(argv[13]);
        MNISTExample mnist(hidden_layer_size, hidden_pool_size, num_hidden_layers, b, L, size_limit, L2_Lambda,
                           learning_rate);
        mnist.testMNIST(training_label_path, training_image_path, test_label_path, test_image_path, max_epoch);
    } else {
        MNISTExample mnist;
        mnist.testMNIST(training_label_path, training_image_path, test_label_path, test_image_path, max_epoch);
    }
    return 0;
}
