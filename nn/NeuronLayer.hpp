#ifndef NEURONLAYER_HPP
#define NEURONLAYER_HPP

#include <memory>
#include <vector>

class NN_parameters;

class NeuronLayer {
   public:
    NeuronLayer(int prev_layer_size, int layer_size, double L2);

    std::vector<double> activationFunction();

    virtual std::vector<double> forwardPropagation(std::vector<double> input, NN_parameters* m_theta) = 0;

    virtual std::vector<double> calculateDelta(std::vector<double> prev_layer_delta, NN_parameters* m_theta) = 0;

    virtual void calculateGradient(NN_parameters* m_theta) = 0;

    virtual std::shared_ptr<NeuronLayer> clone() const = 0;

    int numWeights();

    int numBias();

    virtual double weightInitialization() const = 0;

    virtual std::vector<double> activationFunction(std::vector<double> input) = 0;

    int m_pos = -1;
    std::vector<double> m_input;
    std::vector<double> m_weightedSum;
    std::vector<double> m_delta;
    int m_prev_layer_size;
    int m_layer_size;
    double L2_lambda;
};

#endif  // NEURONLAYER_HPP
