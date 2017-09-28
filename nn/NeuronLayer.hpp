#ifndef NEURONLAYER_HPP
#define NEURONLAYER_HPP

#include <vector>

using namespace std;

class NN_parameters;

class NeuronLayer
{
public:
    NeuronLayer(int prev_layer_size, int layer_size, double L2); // : m_prev_layer_size(prev_layer_size), m_layer_size(layer_size), L2_lambda(L2);

    vector<double> activationFunction();
    
    virtual vector<double> forwardPropagation(vector<double> input, NN_parameters m_theta) = 0;
    
    virtual vector<double> calculateDelta(vector<double> prev_layer_delta, NN_parameters m_theta) = 0;
    
    virtual void calculateGradient(NN_parameters m_theta) = 0;

    int numWeights();
    
    int numBias();

    virtual double weightInitialization() = 0;

    virtual vector<double> activationFunction(vector<double> input) = 0;

    int m_pos = -1;
    vector<double> m_input;
    vector<double> m_weightedSum;
    vector<double> m_delta;
    int m_prev_layer_size;
    int m_layer_size;
    double L2_lambda;
};

#endif // NEURONLAYER_HPP