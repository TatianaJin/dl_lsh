#ifndef NEURALCOORDINATOR_HPP
#define NEURALCOORDINATOR_HPP

#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "NN_parameters.hpp"
#include "NeuronLayer.hpp"
#include "HiddenLayer.hpp"
#include "NeuralNetwork.hpp"
#include "ICostFunction.hpp"

using namespace std;
using namespace Eigen;

class NeuralCoordinator
{
public:
	static int INT_SIZE;
    static int LAYER_THREADS;
    static int UPDATE_SIZE;

	NeuralCoordinator(string model_title, string title, string dataset, NN_parameters params, vector<NeuronLayer*> layers, vector<HiddenLayer*> hiddenLayers, double L2, ICostFunction cf);

	void test(vector<VectorXd> data, vector<double> labels);

	void train(int max_epoch, vector<VectorXd> data, vector<double> labels, vector<VectorXd> test_data, vector<double> test_labels);

private:
	vector<int> initIndices(int length);

	void shuffle(vector<int> indices);

	double calculateActiveNodes(double total);

	// void run(int start, int end, NeuralNetwork network, vector<vector<int>> input_hashes, vector<VectorXd> data, vector<double> labels);

	int min(int ele1, int ele2);
 
    vector<NeuralNetwork> m_networks;
    NN_parameters m_params;
    string m_modelTitle;
    string m_model_path;
    string m_train_path;
    string m_test_path;
    double m_total_nodes;
    int update_threshold = 20;
};

#endif // NEURALCOORDINATOR_HPP