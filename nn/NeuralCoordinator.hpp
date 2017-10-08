#ifndef NEURALCOORDINATOR_HPP
#define NEURALCOORDINATOR_HPP

#include <Eigen/Dense>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "exp/colors.hpp"
#include "nn/HiddenLayer.hpp"
#include "nn/ICostFunction.hpp"
#include "nn/NN_parameters.hpp"
#include "nn/NeuralNetwork.hpp"
#include "nn/NeuronLayer.hpp"

using Eigen::VectorXd;

template <typename Vector>
void run(int start, int end, NeuralNetwork<Vector>& network, std::vector<VectorXd>& data, std::vector<double>& labels) {
    for (int pos = start; pos < end; ++pos) {
        network.execute(data[pos], labels[pos], true);
    }
    std::cout << GREEN("[INFO] ") << "Train correct count = " << network.numTheta() << std::endl;
}

template <typename Vector>
class NeuralCoordinator {
   public:
    NeuralCoordinator() = default;
    NeuralCoordinator(NN_parameters* params, std::vector<std::shared_ptr<NeuronLayer<Vector>>> layers, double L2,
                      ICostFunction<Vector>* cf, int num_threads)
        : m_params(params), num_threads_(num_threads) {
        // calculate the total number of neurons
        for (auto& layer : layers) {
            m_total_nodes += layer->getLayerSize();
        }

        // construct num_threads_ neural networks for parallel training
        m_networks.reserve(num_threads_);
        std::cout << GREEN("[INFO] ") << num_threads_ << " thread(s) are to be used" << std::endl;
        for (int idx = 1; idx < num_threads_; idx++) {
            std::vector<std::shared_ptr<NeuronLayer<Vector>>> layers1;
            layers1.reserve(layers.size());
            for (auto& layer : layers) {
                layers1.push_back(layer->clone());
            }
            m_networks.push_back(NeuralNetwork<Vector>(params, std::move(layers1), L2, cf));
        }
        m_networks.push_back(NeuralNetwork<Vector>(params, std::move(layers), L2, cf));
    }

    void setUpdatesPerEpoch(int num) { this->m_update_size = num; }

    void test(std::vector<VectorXd>& data, std::vector<double>& labels) {
        std::cout << m_networks[0].test(data, labels) << std::endl;
    }

    /*
     * the data and labels may get transformed according to the hashing scheme
     */
    void train(int max_epoch, std::vector<VectorXd> data, std::vector<double> labels, std::vector<VectorXd> test_data,
               std::vector<double> test_labels, bool dropout) {
        assert(data.size() == labels.size());            // debug
        assert(test_data.size() == test_labels.size());  // debug

        // TODO(Tatiana): parallelize the hashing process?
        int cardinality = data.size();
        int m_examples_per_thread = cardinality / (m_update_size * num_threads_);
        std::cout << GREEN("[INFO] ") << "Each thread uses " << m_examples_per_thread << " samples" << std::endl;

        for (auto& network : m_networks) {  // TODO(tatiana): use thread pool?
            network.set_dropout(dropout);
        }
        for (int epoch_count = 0; epoch_count < max_epoch; epoch_count++) {
            m_params->clear_gradient();
            int count = 0;
            while (count < cardinality) {
                std::vector<std::thread> threads;
                threads.reserve(m_networks.size());
                assert(!m_networks.empty());
                for (auto& network : m_networks) {  // TODO(tatiana): use thread pool?
                    if (count < cardinality) {
                        int start = count;
                        count = std::min(cardinality, count + m_examples_per_thread);
                        int end = count;
                        threads.push_back(
                            std::thread(run<Vector>, start, end, std::ref(network), std::ref(data), std::ref(labels)));
                    } else
                        break;
                }
                for (auto& t : threads) t.join();  // barrier
                if (dropout) {
                    // std::cout << GREEN("[INFO] ") << "Rebuilding hash tables" << std::endl;
                    m_params->rebuildTables();  // TODO(tatiana): adjust hash table update frequency?
                }
            }

            int epoch = m_params->epoch_offset() + epoch_count;
            double activeNodes = calculateActiveNodes(m_total_nodes * data.size());
            double test_accuracy = m_networks[0].test(test_data, test_labels);
            std::cout << GREEN("[INFO] ") << "Epoch " << epoch << "\tAccuracy: " << test_accuracy
                      << "\n\tActive Nodes: " << activeNodes << std::endl;

            m_params->timeStep();
        }
    }

    void set_parallelism(int n) { num_threads_ = n; }

   private:
    std::vector<int> initIndices(int length) {
        std::vector<int> indices(length);
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }

    double calculateActiveNodes(double total) {
        double active = 0;
        for (auto& network : m_networks) {
            active += network.calculateActiveNodes();
        }
        return active / total;
    }

    NN_parameters* m_params;  // not owned
    std::vector<NeuralNetwork<Vector>> m_networks;
    double m_total_nodes = 0.;
    int update_threshold = 20;

    int num_threads_ = 1;   // number of threads to run concurrently
    int m_update_size = 1;  // number of hash table updates in one epoch
};

#endif  // NEURALCOORDINATOR_HPP
