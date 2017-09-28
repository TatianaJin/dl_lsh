#include "NeuralCoordinator.hpp"
#include <cstdlib>
#include <ctime>
#include <thread>

int NeuralCoordinator::INT_SIZE = 32;
int NeuralCoordinator::LAYER_THREADS = 1;
int NeuralCoordinator::UPDATE_SIZE = 10;

void run(int start, int end, NeuralNetwork network, vector<vector<int>> input_hashes, vector<VectorXd> data, vector<double> labels)
{
    for (int pos = start; pos < end; pos ++)
    {
        network.execute(input_hashes[pos], data[pos], labels[pos], true);
    }
}

NeuralCoordinator::NeuralCoordinator(string model_title, string title, string dataset, NN_parameters params, vector<NeuronLayer*> layers, vector<HiddenLayer*> hiddenLayers, double L2, ICostFunction cf)
{
    /*m_modelTitle = model_title;
    m_model_path = Util.DATAPATH + dataset + "/" + Util.MODEL + title + "_" + model_title;
    m_train_path = Util.DATAPATH + dataset + "/" + Util.TRAIN + title;
    m_test_path = Util.DATAPATH + dataset + "/" + Util.TEST + title;
	*/

    for(auto& layer : layers)
    {
        m_total_nodes += layer->m_layer_size;
    }

    m_params = params;
    m_networks.resize(LAYER_THREADS);
    m_networks.push_back(NeuralNetwork(params, layers, hiddenLayers, L2, cf));

    for(int idx = 1; idx < LAYER_THREADS; idx ++)
    {
        vector<HiddenLayer*> hiddenLayers1(hiddenLayers.size());
        for(auto& hiddenLayer : hiddenLayers)
        {
        	hiddenLayers1.push_back(hiddenLayer);
        }

        vector<NeuronLayer*> layers1(layers.size());
        for(auto& hiddenLayer1 : hiddenLayers1)
        {
        	layers1.push_back(hiddenLayer1);
        }
        layers1.push_back(layers[layers.size()-1]);
        m_networks.push_back(NeuralNetwork(params, layers1, hiddenLayers1, L2, cf));
    }	
}

void NeuralCoordinator::test(vector<VectorXd> data, vector<double> labels)
{   
	vector<vector<int>> test_hashes = m_params.computeHashes(data);
    cout << "Finished Pre-Computing Training Hashes" << endl;
    cout << m_networks[0].test(test_hashes, data, labels) << endl;
}

void NeuralCoordinator::train(int max_epoch, vector<VectorXd> data, vector<double> labels, vector<VectorXd> test_data, vector<double> test_labels)
{
    assert(data.size() == labels.size());
    assert(test_data.size() == test_labels.size());

    vector<vector<int>> input_hashes = m_params.computeHashes(data);
    cout << "Finished Pre-Computing Training Hashes" << endl;

    vector<vector<int>> test_hashes = m_params.computeHashes(test_data);
    cout << "Finished Pre-Computing Testing Hashes" << endl;

    vector<int> data_idx = initIndices(labels.size());
    int m_examples_per_thread = data.size() / (UPDATE_SIZE * LAYER_THREADS);
    assert(data_idx.size() == labels.size());

    // BufferedWriter train_writer = new BufferedWriter(new FileWriter(m_train_path, true));
    // BufferedWriter test_writer = new BufferedWriter(new FileWriter(m_test_path, true));
    for(int epoch_count = 0; epoch_count < max_epoch; epoch_count ++)
    {
        m_params.clear_gradient();
        shuffle(data_idx);
        int count = 0;
        while(count < data_idx.size())
        {
            // List<Thread> threads = new LinkedList<>();
            for(auto& network : m_networks)
            {
                if(count < data_idx.size())
                {
                    int start = count;
                    count = min(data_idx.size(), count + m_examples_per_thread);
                    int end = count;

                    thread t(run, start, end, network, input_hashes, data, labels);
                    t.join();
                }
            }
            // Util.join(threads);
            if(epoch_count <= update_threshold && epoch_count % (epoch_count / 10 + 1) == 0)
            {
                m_params.rebuildTables();
            }

        }

        // Console Debug Output
        int epoch = m_params.epoch_offset() + epoch_count;
        //m_networks.stream().forEach(e -> e.updateHashTables(labels.length / Util.LAYER_THREADS));
        double activeNodes = calculateActiveNodes(m_total_nodes * data.size());
        double test_accuracy = m_networks[0].test(test_hashes, test_data, test_labels);
        cout << "Epoch " << epoch << " Accuracy: " << test_accuracy << endl;

        /*
        // Test Output
        DecimalFormat df = new DecimalFormat("#.###");
        df.setRoundingMode(RoundingMode.FLOOR);
        test_writer.write(m_modelTitle + " " + epoch + " " + df.format(activeNodes) + " " + test_accuracy);
        test_writer.newLine();

        // Train Output
        train_writer.write(m_modelTitle + " " + epoch + " " + df.format(activeNodes) + " " + calculateTrainAccuracy(data.size()));
        train_writer.newLine();

        test_writer.flush();
        train_writer.flush();
		*/
        m_params.timeStep();
    }
    /*
    test_writer.close();
    train_writer.close();
    save_model(max_epoch, m_model_path);
    */
}

int NeuralCoordinator::min(int ele1, int ele2)
{
	return ele1 > ele2 ? ele2 : ele1;
}

/*void NeuralCoordinator::run(int start, int end, NeuralNetwork network, vector<vector<int>> input_hashes, vector<VectorXd> data, vector<double> labels)
{
    for (int pos = start; pos < end; pos ++)
    {
        network.execute(input_hashes[pos], data[pos], labels[pos], true);
    }
}
*/

vector<int> NeuralCoordinator::initIndices(int length)
{
    vector<int> indices;
    for(int idx = 0; idx < length; idx ++)
    {
        indices.push_back(idx);
    }
    return indices;
}

void NeuralCoordinator::shuffle(vector<int> indices)
{
    for(int idx = 0; idx < indices.size(); idx ++)
    {
    	srand(time(NULL));
    	int random = rand() % indices.size();
        int value = indices[idx];
        indices[idx] = indices[random];
        indices[random] = value;
    }
}

double NeuralCoordinator::calculateActiveNodes(double total)
{
   	long active = 0;
    for(auto& network : m_networks)
    {
        active += network.calculateActiveNodes();
    }
    return active / total;
}