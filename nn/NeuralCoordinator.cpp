#include "NeuralCoordinator.hpp"

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <thread>
#include <vector>

#include "exp/colors.hpp"

int NeuralCoordinator::LAYER_THREADS = 5;
int NeuralCoordinator::UPDATE_SIZE = 1;

NeuralCoordinator::NeuralCoordinator(NN_parameters* params, std::vector<std::shared_ptr<NeuronLayer>> layers, double L2,
                                     ICostFunction* cf)
    : m_params(params) {
    // calculate the total number of neurons
    for (auto& layer : layers) {
        m_total_nodes += layer->getLayerSize();
    }

    // construct LAYER_THREADS neural networks for parallel training
    m_networks.reserve(LAYER_THREADS);
    for (int idx = 1; idx < LAYER_THREADS; idx++) {
        std::vector<std::shared_ptr<NeuronLayer>> layers1;
        layers1.reserve(layers.size());
        for (auto& layer : layers) {
            layers1.push_back(layer->clone());
        }
        m_networks.push_back(NeuralNetwork(params, std::move(layers1), L2, cf));
    }
    m_networks.push_back(NeuralNetwork(params, std::move(layers), L2, cf));
}

void run(int start, int end, NeuralNetwork& network, std::vector<VectorXd>& data, std::vector<double>& labels) {
    for (int pos = start; pos < end; ++pos) {
        network.execute(data[pos], labels[pos], true);
    }
}

void NeuralCoordinator::test(std::vector<VectorXd>& data, std::vector<double>& labels) {
    std::vector<std::vector<int>> test_hashes = m_params->computeHashes(data);
    std::cout << "Finished Pre-Computing Testing Hashes" << std::endl;
    std::cout << m_networks[0].test(test_hashes, data, labels) << std::endl;
}

/*
 * the data and labels may get transformed according to the hashing scheme
 */
void NeuralCoordinator::train(int max_epoch, std::vector<VectorXd> data, std::vector<double> labels,
                              std::vector<VectorXd> test_data, std::vector<double> test_labels) {
    assert(data.size() == labels.size());            // debug
    assert(test_data.size() == test_labels.size());  // debug

    // TODO(Tatiana): parallelize the hashing process?
    int cardinality = data.size();
    int m_examples_per_thread = cardinality / (UPDATE_SIZE * LAYER_THREADS);

    for (int epoch_count = 0; epoch_count < max_epoch; epoch_count++) {
        m_params->clear_gradient();
        int count = 0;
        while (count < cardinality) {
            // std::vector<std::vector<int>> input_hashes = m_params->computeHashes(data);
            // std::cout << GREEN("[INFO] ") << "Finished Pre-Computing Training Hashes" << std::endl;

            std::vector<std::thread> threads;
            threads.reserve(m_networks.size());
            for (auto& network : m_networks) {  // TODO(tatiana): use thread pool?
                if (count < cardinality) {
                    int start = count;
                    count = std::min(cardinality, count + m_examples_per_thread);
                    int end = count;
                    threads.push_back(
                        std::thread(run, start, end, std::ref(network), std::ref(data), std::ref(labels)));
                } else
                    break;
            }
            for (auto& t : threads) t.join();  // barrier
            std::cout << GREEN("[INFO] ") << "Rebuilding hash tables" << std::endl;
            m_params->rebuildTables();  // TODO(tatiana): adjust hash table update frequency?
        }

        std::vector<std::vector<int>> test_hashes = m_params->computeHashes(test_data);
        std::cout << GREEN("[INFO] ") << "Finished Pre-Computing Testing Hashes" << std::endl;
        int epoch = m_params->epoch_offset() + epoch_count;
        double activeNodes = calculateActiveNodes(m_total_nodes * data.size());
        double test_accuracy = m_networks[0].test(test_hashes, test_data, test_labels);
        std::cout << GREEN("[INFO] ") << "Epoch " << epoch << "\tAccuracy: " << test_accuracy
                  << "\n\tActive Nodes: " << activeNodes << std::endl;

        m_params->timeStep();
    }
}

std::vector<int> NeuralCoordinator::initIndices(int length) {
    std::vector<int> indices(length);
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}

void NeuralCoordinator::shuffle(std::vector<int>& indices) {  // TODO obsolete
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
}

double NeuralCoordinator::calculateActiveNodes(double total) {
    double active = 0;
    for (auto& network : m_networks) {
        active += network.calculateActiveNodes();
    }
    return active / total;
}
