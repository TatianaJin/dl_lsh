#ifndef HIDDENLAYER_HPP
#define HIDDENLAYER_HPP

#include <iostream>
#include <set>
#include <vector>
#include <Eigen/Dense>
#include "NeuronLayer.hpp"
#include "NN_parameters.hpp"

using namespace std;
using Eigen::VectorXd;

class HiddenLayer : public NeuronLayer
{
public:
    HiddenLayer(int prev_layer_size, int layer_size, double L2);

    virtual double derivative(double input) = 0;

    vector<double> forwardPropagation(VectorXd input, set<int> nn_node_set, bool training, NN_parameters m_theta);

    // some forwardPropagation functions should be added later
    vector<double> forwardPropagation(vector<double> input, NN_parameters m_theta);

    vector<double> forwardPropagation(vector<double> input, bool training, NN_parameters m_theta);

    vector<double> forwardPropagation(VectorXd input, bool training, NN_parameters m_theta);

    vector<double> forwardPropagation(VectorXd input, vector<int> hashes, bool training, NN_parameters m_theta);

    vector<double> calculateDelta(vector<double> prev_layer_delta, NN_parameters m_theta);

    void calculateGradient(NN_parameters m_theta);

    void updateHashTables(double size);

    VectorXd vectorize(vector<double> data, int offset, int length);

    VectorXd vectorize(vector<double> data);

    set<int> m_node_set;       // number of nodes that have been activated in that layer?
    set<int> m_total_node_set; // total number of nodes in that layer??
    long m_total_nn_set_size;
    long m_total_multiplication;
};

#endif // HIDDENLAYER_HPP
