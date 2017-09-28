#include "NeuralCoordinator.hpp"

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <vector>

int NeuralCoordinator::INT_SIZE = 32;
int NeuralCoordinator::LAYER_THREADS = 1;
int NeuralCoordinator::UPDATE_SIZE = 10;

NeuralCoordinator::NeuralCoordinator(NN_parameters* params, std::vector<std::shared_ptr<NeuronLayer>> layers,
                                     std::vector<std::shared_ptr<HiddenLayer>> hiddenLayers, double L2,
                                     ICostFunction* cf) {
    for (auto& layer : layers) {
        m_total_nodes += layer->m_layer_size;
    }

    m_params = params;
    m_networks.resize(LAYER_THREADS);  // construct LAYER_THREADS neural networks for multi-core training

    for (int idx = 1; idx < LAYER_THREADS; idx++) {
        std::vector<std::shared_ptr<HiddenLayer>> hiddenLayers1(hiddenLayers.size());
        for (auto& hiddenLayer : hiddenLayers) {
            hiddenLayers1.push_back(
                std::shared_ptr<HiddenLayer>(static_cast<HiddenLayer*>(hiddenLayer->clone().get())));
        }

        std::vector<std::shared_ptr<NeuronLayer>> layers1(layers.size());
        for (auto& hiddenLayer1 : hiddenLayers1) {
            layers1.push_back(hiddenLayer1);
        }
        layers1.push_back(layers[layers.size() - 1]->clone());
        m_networks.push_back(NeuralNetwork(params, std::move(layers1), std::move(hiddenLayers1), L2, cf));
    }

    m_networks.push_back(NeuralNetwork(params, std::move(layers), std::move(hiddenLayers), L2, cf));
}

void run(int start, int end, NeuralNetwork network, std::vector<std::vector<int>> input_hashes,
         std::vector<VectorXd> data, std::vector<double> labels) {
    for (int pos = start; pos < end; pos++) {
        network.execute(input_hashes[pos], data[pos], labels[pos], true);
    }
}

void NeuralCoordinator::test(std::vector<VectorXd> data, std::vector<double> labels) {
    std::vector<std::vector<int>> test_hashes = m_params->computeHashes(data);
    std::cout << "Finished Pre-Computing Training Hashes" << std::endl;
    std::cout << m_networks[0].test(test_hashes, data, labels) << std::endl;
}

/*
 * the data and labels may get transformed according to the hashing scheme
 */
void NeuralCoordinator::train(int max_epoch, std::vector<VectorXd> data, std::vector<double> labels,
                              std::vector<VectorXd> test_data, std::vector<double> test_labels) {
    assert(data.size() == labels.size());
    assert(test_data.size() == test_labels.size());

    std::vector<std::vector<int>> input_hashes = m_params->computeHashes(data);  // TODO(Tatiana): may parallelize
    std::cout << "Finished Pre-Computing Training Hashes" << std::endl;

    std::vector<std::vector<int>> test_hashes = m_params->computeHashes(test_data);
    std::cout << "Finished Pre-Computing Testing Hashes" << std::endl;

    std::vector<int> data_idx = initIndices(labels.size());
    int m_examples_per_thread = data.size() / (UPDATE_SIZE * LAYER_THREADS);
    assert(data_idx.size() == labels.size());

    // BufferedWriter train_writer = new BufferedWriter(new
    // FileWriter(m_train_path, true));
    // BufferedWriter test_writer = new BufferedWriter(new
    // FileWriter(m_test_path, true));
    for (int epoch_count = 0; epoch_count < max_epoch; epoch_count++) {
        m_params->clear_gradient();
        shuffle(data_idx);
        size_t count = 0;
        while (count < data_idx.size()) {
            // List<Thread> threads = new LinkedList<>();
            for (auto& network : m_networks) {
                if (count < data_idx.size()) {
                    int start = count;
                    count = min(data_idx.size(), count + m_examples_per_thread);
                    int end = count;

                    thread t(run, start, end, network, input_hashes, data, labels);
                    t.join();
                }
            }
            // Util.join(threads);
            if (epoch_count <= update_threshold && epoch_count % (epoch_count / 10 + 1) == 0) {
                m_params->rebuildTables();
            }
        }

        // Console Debug Output
        int epoch = m_params->epoch_offset() + epoch_count;
        // m_networks.stream().forEach(e -> e.updateHashTables(labels.length /
        // Util.LAYER_THREADS));
        double activeNodes = calculateActiveNodes(m_total_nodes * data.size());
        double test_accuracy = m_networks[0].test(test_hashes, test_data, test_labels);
        std::cout << "Epoch " << epoch << "\tAccuracy: " << test_accuracy << "\n\tActive Nodes: " << activeNodes
                  << std::endl;

        /*
        // Test Output
        DecimalFormat df = new DecimalFormat("#.###");
        df.setRoundingMode(RoundingMode.FLOOR);
        test_writer.write(m_modelTitle + " " + epoch + " " +
        df.format(activeNodes) + " " + test_accuracy);
        test_writer.newLine();

        // Train Output
        train_writer.write(m_modelTitle + " " + epoch + " " +
        df.format(activeNodes) + " " + calculateTrainAccuracy(data.size()));
        train_writer.newLine();

        test_writer.flush();
        train_writer.flush();
                */
        m_params->timeStep();
    }
    /*
    test_writer.close();
    train_writer.close();
    save_model(max_epoch, m_model_path);
    */
}

int NeuralCoordinator::min(int ele1, int ele2) { return ele1 > ele2 ? ele2 : ele1; }

/*void NeuralCoordinator::run(int start, int end, NeuralNetwork network,
std::vector<std::vector<int>> input_hashes, std::vector<VectorXd> data,
std::vector<double> labels)
{
    for (int pos = start; pos < end; pos ++)
    {
        network.execute(input_hashes[pos], data[pos], labels[pos], true);
    }
}
*/

std::vector<int> NeuralCoordinator::initIndices(int length) {
    std::vector<int> indices;
    for (int idx = 0; idx < length; idx++) {
        indices.push_back(idx);
    }
    return indices;
}

void NeuralCoordinator::shuffle(std::vector<int> indices) {
    for (int idx = 0; idx < indices.size(); idx++) {
        srand(time(NULL));
        int random = rand() % indices.size();
        int value = indices[idx];
        indices[idx] = indices[random];
        indices[random] = value;
    }
}

double NeuralCoordinator::calculateActiveNodes(double total) {
    long active = 0;
    for (auto& network : m_networks) {
        active += network.calculateActiveNodes();
    }
    return active / total;
}
