#include "nn/HiddenLayer.hpp"
#include "nn/NeuronLayer.hpp"
#include "nn/ReLUNeuronLayer.hpp"
#include "nn/SoftMaxNeuronLayer.hpp"
#include "nn/ICostFunction.hpp"
#include "nn/NeuralCoordinator.hpp"
#include "dataset/MNISTDataSet.hpp"
#include <ctime>

int min_layers = 1;
int max_layers = 1;
int hidden_layer_size = 1000;
int hidden_pool_size = hidden_layer_size * 0.1;

// Neural Network Parameters
string title;
string dataset;
int training_size;
int test_size;
int inputLayer;
int outputLayer;
int k;
int b = 6;
int L = 100;

int max_epoch = 25;
double L2_Lambda = 0.003;
vector<int> hiddenLayers;
vector<double> learning_rates;
vector<double> size_limits = {0.05, 0.10, 0.25, 0.5, 0.75, 1.0};

vector<HiddenLayer*> hidden_layers;
vector<NeuronLayer*> NN_layers;

string make_title()
{
    title = dataset + '_' + "LSH" + '_' + to_string(inputLayer) + '_';
    for(int idx = 0; idx < hiddenLayers.size(); idx ++)
    {
        title += to_string(hiddenLayers[idx]);
        title += '_';
    }
    title += to_string(outputLayer);
    return title;
}

/*
void testNORB()
{
    dataset = "NORB_SMALL"
    training_size = 20000
    test_size = 24300
    inputLayer = 2048
    outputLayer = 5
    k = 128
    learning_rates = [1e-2, 1e-2, 1e-2, 5e-3, 1e-3, 1e-3]

    // Read NORB training, validation, test data
    final String training_path = Util.DATAPATH + dataset + "/norb-small-train.bz2"
    final String test_path = Util.DATAPATH + dataset + "/norb-small-test.bz2"

    Pair<List<DoubleMatrix>, double[]> training = NORBDataSet.loadDataSet(Util.readerBZ2(training_path), training_size)
    Pair<List<DoubleMatrix>, double[]> test = NORBDataSet.loadDataSet(Util.readerBZ2(test_path), test_size)
    execute(training.getLeft(), training.getRight(), test.getLeft(), test.getRight());
}

void testRectangles()
{
    test_size = 50000
    inputLayer = 784
    outputLayer = 2
    k = 98

    // Rectangles Data
    dataset = "Rectangles";
    training_size = 12000;
    final String training_path = Util.DATAPATH + dataset + "/rectangles_im_train.amat.bz2";
    final String test_path = Util.DATAPATH + dataset + "/rectangles_im_test.amat.bz2";
    learning_rates = [1e-2, 1e-2, 5e-3, 1e-3, 1e-3, 1e-3]

    Pair<List<DoubleMatrix>, double[]> training = DLDataSet.loadDataSet(Util.readerBZ2(training_path), training_size, inputLayer)
    Pair<List<DoubleMatrix>, double[]> test = DLDataSet.loadDataSet(Util.readerBZ2(test_path), test_size, inputLayer)
    execute(training.getLeft(), training.getRight(), test.getLeft(), test.getRight());
}

void testConvex()
{
    test_size = 50000
    inputLayer = 784
    outputLayer = 2
    k = 98

    // Convex
    dataset = "Convex";
    training_size = 8000;
    final String training_path = Util.DATAPATH + dataset + "/convex_train.amat.bz2";
    final String test_path = Util.DATAPATH + dataset + "/convex_test.amat.bz2";
    learning_rates = [1e-2, 1e-2, 5e-3, 1e-3, 1e-3, 1e-3]

    Pair<List<DoubleMatrix>, double[]> training = DLDataSet.loadDataSet(Util.readerBZ2(training_path), training_size, inputLayer)
    Pair<List<DoubleMatrix>, double[]> test = DLDataSet.loadDataSet(Util.readerBZ2(test_path), test_size, inputLayer)
    execute(training.getLeft(), training.getRight(), test.getLeft(), test.getRight());
}
*/

void construct(int inputLayer, int outputLayer)
{
    // Hidden Layers
    ReLUNeuronLayer input_layer(inputLayer, hiddenLayers[0], L2_Lambda);
    hidden_layers.push_back(&input_layer);
    for(int idx = 0; idx < hiddenLayers.size()-1; idx ++)
    {
        ReLUNeuronLayer hidden_layer(hiddenLayers[idx], hiddenLayers[idx+1], L2_Lambda);
        hidden_layers.push_back(&hidden_layer);
    }
    for(auto& hidden_layer : hidden_layers)
    {
        NN_layers.push_back(hidden_layer);
    }
    // Output Layers
    SoftMaxNeuronLayer output_layer(hiddenLayers[hiddenLayers.size()-1], outputLayer, L2_Lambda);
    NN_layers.push_back(&output_layer);
}

void execute(vector<VectorXd> training_data, vector<double> training_labels, vector<VectorXd> test_data, vector<double> test_labels)
{
    assert(size_limits.size() == learning_rates.size());

    for(int size = min_layers; size <= max_layers; size ++)
    {
        hiddenLayers.resize(size);
        hiddenLayers.assign(size, hidden_layer_size);

        vector<int> sum_pool(size);
        sum_pool.assign(size, hidden_pool_size);
        sum_pool[0] = k;

        vector<int> bits(size);
        bits.assign(size, b);

        vector<int> tables(size);
        tables.assign(size, L);

        for(int idx = 0; idx < size_limits.size(); idx ++)
        {
            vector<double> sl(size);
            sl.assign(size, size_limits[idx]);

            cout << make_title();
            construct(inputLayer, outputLayer);
            /*try
            {
                parameters = new NN_parameters(Util.readerBZ2(Util.DATAPATH + dataset + "/" + Util.MODEL + title), NN_layers, sum_pool, bits, tables, learning_rates[idx], sl)
            }
            catch (Exception ignore)
            {
                parameters = new NN_parameters(NN_layers, sum_pool, bits, tables, learning_rates[idx], sl)
            }*/

            NN_parameters parameters(NN_layers, sum_pool, bits, tables, learning_rates[idx], sl);
            NeuralCoordinator NN(to_string(size_limits[idx]), title, dataset, parameters, NN_layers, hidden_layers, L2_Lambda, ICostFunction());
            time_t startTime = time(0);
            NN.train(max_epoch, training_data, training_labels, test_data, test_labels);
            time_t endTime = time(0);
            double estimatedTime = (double)(endTime - startTime) / 1000;
            cout << estimatedTime;
        }
    }
}

void testMNIST()
{
    dataset = "MNIST";
    training_size = 60000;
    test_size = 10000;
    inputLayer = 784;
    outputLayer = 10;
    k = 98;
    learning_rates = {1e-2, 1e-2, 1e-2, 5e-3, 1e-3, 1e-3};

    // Read MNIST test and training data
    string training_label_path = "/Users/mac/Desktop/ALSH_DL2/train-labels.idx1-ubyte";
    string training_image_path = "/Users/mac/Desktop/ALSH_DL2/train-images.idx3-ubyte";
    string test_label_path = "/Users/mac/Desktop/ALSH_DL2/t10k-labels.idx1-ubyte";
    string test_image_path = "/Users/mac/Desktop/ALSH_DL2/t10k-images.idx3-ubyte";

    pair<vector<VectorXd>, vector<double>> training = MNISTDataSet::loadDataSet(training_label_path, training_image_path);
    pair<vector<VectorXd>, vector<double>> test = MNISTDataSet::loadDataSet(test_label_path, test_image_path);

    execute(training.first, training.second, test.first, test.second);
}

int main()
{
    testMNIST();
    return 0;
}