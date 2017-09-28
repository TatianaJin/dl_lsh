#include <iostream>
#include "NN_parameters.hpp"
#include "NeuronLayer.hpp"

using namespace std;

//NN_parameters m_theta;

NeuronLayer::NeuronLayer(int prev_layer_size, int layer_size, double L2)
    : m_prev_layer_size(prev_layer_size), m_layer_size(layer_size), L2_lambda(L2)
{
    m_weightedSum.resize(m_layer_size, 0.0);  // pay more attention to this line, may need to be commented
}

vector<double> NeuronLayer::activationFunction()
{
    return activationFunction(m_weightedSum);
}

vector<double> NeuronLayer::activationFunction(vector<double> input) 
{
	return input;
}

/*
vector<double> NeuronLayer::forwardPropagation(vector<double> input) 
{
	return input;
}*/

/*
vector<double> NeuronLayer::calculateDelta(vector<double> prev_layer_delta) 
{
	return prev_layer_delta;
}*/

/* void NeuronLayer::calculateGradient() {}*/

int NeuronLayer::numWeights()
{
    return m_prev_layer_size * m_layer_size;
}

int NeuronLayer::numBias()
{
    return m_layer_size;
}

/* double NeuronLayer::weightInitialization() 
{
	return 0;
}*/

/*NN_parameters* NeuronLayer::get_theta() 
{
	return &m_theta;
}*/