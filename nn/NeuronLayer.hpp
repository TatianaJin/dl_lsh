#ifndef NEURONLAYER_HPP
#define NEURONLAYER_HPP

#include <Eigen/Dense>
#include <memory>
#include <vector>

using Eigen::VectorXd;

class NN_parameters;

class NeuronLayer {
   public:
    NeuronLayer(int prev_layer_size, int layer_size, double L2);

    VectorXd activationFunction();

    virtual VectorXd forwardPropagation(const VectorXd& input, NN_parameters* m_theta) = 0;

    virtual VectorXd calculateDelta(const VectorXd& prev_layer_delta, NN_parameters* m_theta) = 0;

    virtual void calculateGradient(NN_parameters* m_theta) = 0;

    virtual std::shared_ptr<NeuronLayer> clone() const = 0;

    int numWeights();

    int numBias();

    virtual double weightInitialization() const = 0;

    virtual VectorXd activationFunction(const std::vector<double>& input) = 0;

    int getLayerSize() const { return m_layer_size; }
    int getPrevLayerSize() const { return m_prev_layer_size; }
    int getPos() const { return m_pos; }

    void setPos(int pos) { m_pos = pos; }

   protected:
    VectorXd m_input;                   // input to the layer
    std::vector<double> m_weightedSum;  // pre-activation
    VectorXd m_delta;
    int m_pos = -1;  // the index of layer in the whole neural network, i.e. indexed with 0 is input layer
    int m_prev_layer_size;
    int m_layer_size;
    double L2_lambda;
};

#endif  // NEURONLAYER_HPP
