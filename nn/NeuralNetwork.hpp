#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "NeuronLayer.hpp"
#include "HiddenLayer.hpp"
#include "ICostFunction.hpp"

using namespace std;
using namespace Eigen;

class NeuralNetwork
{
public:
    NeuralNetwork() = default;

    NeuralNetwork(NN_parameters params, vector<NeuronLayer*> layers, vector<HiddenLayer*> hiddenLayers, double L2, ICostFunction cf);

    void execute(vector<int> hashes, VectorXd input, double labels, bool training);

    double test(vector<vector<int>> input_hashes, vector<VectorXd> data, vector<double> labels);

    vector<double> forwardPropagation(VectorXd input, vector<int> hashes, bool training);

    void backPropagation(vector<double> y_hat, double labels);

    long calculateActiveNodes();

    long calculateMultiplications();

    vector<vector<int>> computeHashes(vector<VectorXd> data);

    void updateHashTables(int miniBatch_size);

    //------------- double check ---------------------

    double getGradient(int idx);

    double getCost();

    int numTheta();

    double getTheta(int idx);
    // ------------ double check -------------------

protected:
    double m_train_correct;

private:
    vector<NeuronLayer*> m_layers; // m_layers - m_hidden_layers = outputLayer
    vector<HiddenLayer*> m_hidden_layers;
    double L2_lambda;
    ICostFunction m_cf;
    NN_parameters m_params;
    double m_cost;
};

#endif // NEURALNETWORK_HPP