#ifndef RELUNEURONLAYER_HPP
#define RELUNEURONLAYER_HPP

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include "HiddenLayer.hpp"

using namespace std;

class ReLUNeuronLayer : public HiddenLayer
{
public:
    ReLUNeuronLayer(int prev_layer_size, int layer_size, double L2);

protected:
    double weightInitialization();

    vector<double> activationFunction(vector<double> input);

    double derivative(double input);
};

#endif // RELUNEURONLAYER_HPP
