#include <Eigen/Dense>
#include <cassert>
#include <chrono>
#include <ctime>
#include <memory>
#include <string>

#include "dataset/MNISTDataSet.hpp"
#include "nn/HiddenLayer.hpp"
#include "nn/ICostFunction.hpp"
#include "nn/NN_parameters.hpp"
#include "nn/NeuralCoordinator.hpp"
#include "nn/NeuronLayer.hpp"
#include "nn/ReLUNeuronLayer.hpp"
#include "nn/SoftMaxNeuronLayer.hpp"

#include "exp/colors.hpp"

using Eigen::VectorXd;
using SparseVectorXd = Eigen::SparseVector<double>;

template <typename Vector>
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

    void initialize(int num_threads, int update_size, bool dropout = false) {
        // 1. Construct the neural network
        construct(inputLayer_, outputLayer_, num_hidden_layers_);

        // 2. Initialize weights and hash tables
        // For hashing
        std::vector<int> sum_pool(num_hidden_layers_, hidden_pool_size_);
        sum_pool[0] = k_;
        std::vector<double> sl(num_hidden_layers_, size_limit_);  // every layer has the same size limits
        std::vector<int> bits(num_hidden_layers_, b_);
        std::vector<int> tables(num_hidden_layers_, L_);
        parameters_ = NN_parameters(NN_layers_, sum_pool, sl, bits, tables, learning_rate_, dropout);  // only one copy

        // 3. Build controller
        coordinator_ = NeuralCoordinator<Vector>(&parameters_, NN_layers_, L2_Lambda_, &cost_func_, num_threads);
        coordinator_.setUpdatesPerEpoch(update_size);
        std::cout << GREEN("[INFO] ") << "Neural network deployed." << std::endl;
    }

    void train(std::vector<VectorXd> train_data, std::vector<double> train_labels, std::vector<VectorXd> test_data,
               std::vector<double> test_labels, int max_epoch, bool dropout) {
        auto startTime = std::chrono::high_resolution_clock::now();
        coordinator_.train(max_epoch, train_data, train_labels, test_data, test_labels, dropout);
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
        std::cout << inputLayer << ", " << outputLayer << ", " << num_hidden_layer << std::endl;
        std::cout << RED("[WARN] ") << "construct funtion supports only template specialization" << std::endl;
    }

    NN_parameters parameters_;
    NeuralCoordinator<Vector> coordinator_;
    std::vector<std::shared_ptr<NeuronLayer<Vector>>> NN_layers_;
    ICostFunction<Vector> cost_func_;

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

template <>
void FCLExample<SparseVectorXd>::construct(int inputLayer, int outputLayer, int num_hidden_layer) {
    std::vector<int> hiddenLayers(num_hidden_layer, hidden_layer_size_);
    std::cout << make_title(dataset_, hiddenLayers) << std::endl;
    // Clear up
    NN_layers_.clear();

    NN_layers_.reserve(num_hidden_layer + 1);
    // input layer
    NN_layers_.push_back(std::make_shared<ReLUNeuronLayerWithDropOut>(inputLayer, hiddenLayers[0], L2_Lambda_));
    for (int idx = 0; idx < num_hidden_layer - 1; ++idx) {
        NN_layers_.push_back(
            std::make_shared<ReLUNeuronLayerWithDropOut>(hiddenLayers[idx], hiddenLayers[idx + 1], L2_Lambda_));
    }
    // Output Layer
    NN_layers_.push_back(std::make_shared<SoftMaxNeuronLayer<SparseVectorXd>>(hiddenLayers[num_hidden_layer - 1],
                                                                              outputLayer, L2_Lambda_));
}

template <>
void FCLExample<VectorXd>::construct(int inputLayer, int outputLayer, int num_hidden_layer) {
    std::vector<int> hiddenLayers(num_hidden_layer, hidden_layer_size_);
    std::cout << make_title(dataset_, hiddenLayers) << std::endl;
    // Clear up
    NN_layers_.clear();

    NN_layers_.reserve(num_hidden_layer + 1);
    // input layer
    NN_layers_.push_back(std::make_shared<ReLUNeuronLayer>(inputLayer, hiddenLayers[0], L2_Lambda_));
    for (int idx = 0; idx < num_hidden_layer - 1; ++idx) {
        NN_layers_.push_back(std::make_shared<ReLUNeuronLayer>(hiddenLayers[idx], hiddenLayers[idx + 1], L2_Lambda_));
    }
    // Output Layer
    NN_layers_.push_back(
        std::make_shared<SoftMaxNeuronLayer<VectorXd>>(hiddenLayers[num_hidden_layer - 1], outputLayer, L2_Lambda_));
}

template <typename Vector>
class MNISTExample : public FCLExample<Vector> {
   public:
    MNISTExample() : FCLExample<Vector>() {
        this->training_size_ = 60000;
        this->test_size_ = 10000;
        this->inputLayer_ = 784;
        this->outputLayer_ = 10;
        this->k_ = 98;
        this->dataset_ = "MNIST";
    }
    MNISTExample(int hidden_layer_size, int hidden_pool_size, int num_hidden_layers, int b, int L, double size_limit,
                 double L2_Lambda, double learning_rate)
        : FCLExample<Vector>(hidden_layer_size, hidden_pool_size, num_hidden_layers, b, L, size_limit, L2_Lambda,
                             learning_rate) {
        this->training_size_ = 60000;
        this->test_size_ = 10000;
        this->inputLayer_ = 784;
        this->outputLayer_ = 10;
        this->k_ = 98;
        this->dataset_ = "MNIST";
    }

    void testMNIST(std::string training_label_path, std::string training_image_path, std::string test_label_path,
                   std::string test_image_path, int max_epoch, int num_threads = 1, bool dropout = false,
                   int update_size = 1) {
        // Read MNIST test and training data
        std::pair<std::vector<VectorXd>, std::vector<double>> training =
            MNISTDataSet::loadDataSet(training_label_path, training_image_path);
        std::pair<std::vector<VectorXd>, std::vector<double>> test =
            MNISTDataSet::loadDataSet(test_label_path, test_image_path);

        this->initialize(num_threads, update_size, dropout);
        this->train(training.first, training.second, test.first, test.second, max_epoch, dropout);
    }
};  // class MNISTExample

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cout << "Usage: MNISTExample <training_label_path> "
                     "<training_image_path> <test_label_path> "
                     "<test_image_path> <max_epoch> [<hidden_layer_size>] [<hidden_pool_size_>] [<num_hidden_layers>] "
                     "[<b>] [<L>] [<size_limit>] [<L2_Lambda>] [<learning_rate>] [<num_threads>] [if_dropout]"
                  << std::endl;
        return 0;
    }

    for (int i = 0; i < argc; ++i) {
        std::cout << std::string(argv[i]) << " ";
    }
    std::cout << std::endl;

    // Read program args
    std::string training_label_path = argv[1];
    std::string training_image_path = argv[2];
    std::string test_label_path = argv[3];
    std::string test_image_path = argv[4];

    int max_epoch = std::atoi(argv[5]);
    if (argc >= 15) {
        int hidden_layer_size = std::atoi(argv[6]);
        int hidden_pool_size = std::atoi(argv[7]);
        int num_hidden_layers = std::atoi(argv[8]);
        int b = std::atoi(argv[9]);
        int L = std::atoi(argv[10]);
        double size_limit = std::atof(argv[11]);
        double L2_Lambda = std::atof(argv[12]);
        double learning_rate = std::atof(argv[13]);
        int num_threads = std::atoi(argv[14]);
        bool dropout = false;
        int update_per_epoch = 1;
        if (argc >= 16 && std::atoi(argv[15]) == 1) {
            dropout = true;
            std::cout << RED("[INFO] dropout enabled") << std::endl;
            if (argc >= 17) {
                update_per_epoch = std::atoi(argv[16]);
            }
        }
        if (dropout) {
            MNISTExample<SparseVectorXd> mnist(hidden_layer_size, hidden_pool_size, num_hidden_layers, b, L, size_limit,
                                               L2_Lambda, learning_rate);
            mnist.testMNIST(training_label_path, training_image_path, test_label_path, test_image_path, max_epoch,
                            num_threads, dropout, update_per_epoch);
        } else {
            MNISTExample<VectorXd> mnist(hidden_layer_size, hidden_pool_size, num_hidden_layers, b, L, size_limit,
                                         L2_Lambda, learning_rate);
            mnist.testMNIST(training_label_path, training_image_path, test_label_path, test_image_path, max_epoch,
                            num_threads, dropout, update_per_epoch);
        }
    } else {
        MNISTExample<VectorXd> mnist;
        mnist.testMNIST(training_label_path, training_image_path, test_label_path, test_image_path, max_epoch);
    }
    return 0;
}
