#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(NN_parameters params, vector<NeuronLayer*> layers, vector<HiddenLayer*> hiddenLayers, double L2, ICostFunction cf) :
    m_params(params), m_layers(layers), m_hidden_layers(hiddenLayers), L2_lambda(L2), m_cf(cf) {}

void NeuralNetwork::execute(vector<int> hashes, VectorXd input, double labels, bool training)
{
    vector<double> y_hat = forwardPropagation(input, hashes, training);
    backPropagation(y_hat, labels);
    m_train_correct += m_cf.correct(y_hat, labels);
}

double NeuralNetwork::test(vector<vector<int>> input_hashes, vector<VectorXd> data, vector<double> labels)
{
    vector<vector<double>> y_hat(labels.size());
    for(int idx = 0; idx < labels.size(); idx ++)
    {
        y_hat[idx] = forwardPropagation(data[idx], input_hashes[idx], false);
    }
    return m_cf.accuracy(y_hat, labels);
}

vector<double> NeuralNetwork::forwardPropagation(VectorXd input, vector<int> hashes, bool training)
{
    vector<HiddenLayer*>::iterator it = m_hidden_layers.begin();
    vector<double> data = (*it++)->forwardPropagation(input, hashes, training, m_params);

    while(it != m_hidden_layers.end())
    {
        data = (*it++)->forwardPropagation(data, training, m_params);
    }
    return m_layers[m_layers.size() - 1]->forwardPropagation(data, m_params);
}

void NeuralNetwork::backPropagation(vector<double> y_hat, double labels)
{
    // square loss function
    m_cost = m_cf.costFunction(y_hat, labels) + L2_lambda * m_params.L2_regularization();

    NeuronLayer* outputLayer = m_layers[m_layers.size() - 1];

    // cost function derivatives
    vector<double> delta = m_cf.outputDelta(y_hat, labels, *outputLayer);

    // calculate gradient for output layer, calculate delta for hidden layers
    for (vector<NeuronLayer*>::reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); ++it)
    {
        delta = (*it)->calculateDelta(delta, m_params);
    }

    for (vector<NeuronLayer*>::reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); ++it)
    {
        (*it)->calculateGradient(m_params);
    }
}

long NeuralNetwork::calculateActiveNodes()
{
    long total = 0;
    for (auto& l : m_hidden_layers)
    {
        total += l->m_total_nn_set_size; // to be defined
        l->m_total_nn_set_size = 0;
    }
    total += m_layers[m_layers.size() - 1]->m_layer_size;
    return total;
}

long NeuralNetwork::calculateMultiplications()
{
    long total = 0;
    for (auto& l : m_hidden_layers)
    {
        total += l->m_total_multiplication;
        l->m_total_multiplication = 0;
    }
    total += m_layers[m_layers.size() - 1]->numWeights();
    return total;
}

vector<vector<int>> NeuralNetwork::computeHashes(vector<VectorXd> data)
{
    return m_params.computeHashes(data);
}

void NeuralNetwork::updateHashTables(int miniBatch_size)
{
    for (auto& l : m_hidden_layers)
    {
        l->updateHashTables(miniBatch_size);
    }
}

//------------- double check ---------------------

double NeuralNetwork::getGradient(int idx)
{
    return m_params.getGradient(idx);
}

double NeuralNetwork::getCost()
{
    return m_cost;
}

int NeuralNetwork::numTheta()
{
    return m_params.size();
}

double NeuralNetwork::getTheta(int idx)
{
    return m_params.getTheta(idx);
}