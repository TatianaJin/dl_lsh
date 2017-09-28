#include <cmath>
#include <random>
#include <vector>
#include "HiddenLayer.hpp"

using namespace std;

class SoftMaxNeuronLayer : public HiddenLayer
{
public:
    SoftMaxNeuronLayer(int prev_layer_size, int layer_size, double L2);

    vector<double> forwardPropagation(vector<double> input, NN_parameters m_theta);

    vector<double> calculateDelta(vector<double> prev_layer_size, NN_parameters m_theta);

    void calculateGradient(NN_parameters m_theta);

protected:
    double weightInitialization();

    vector<double> activationFunction(vector<double> input);

    double derivative(double input);
};
