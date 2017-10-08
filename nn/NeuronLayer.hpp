#ifndef NEURONLAYER_HPP
#define NEURONLAYER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <vector>

class NN_parameters;

template <typename Vector>
class NeuronLayer {
   public:
    NeuronLayer(int prev_layer_size, int layer_size, double L2)
        : m_prev_layer_size(prev_layer_size),
          m_layer_size(layer_size),
          L2_lambda(L2),
          m_delta(layer_size),
          m_weightedSum(layer_size) {}

    virtual Vector forwardPropagation(const Vector& input, NN_parameters* m_theta) = 0;

    virtual Vector calculateDelta(const Vector& prev_layer_delta, NN_parameters* m_theta) = 0;

    virtual void calculateGradient(NN_parameters* m_theta) = 0;

    virtual std::shared_ptr<NeuronLayer> clone() const = 0;

    int numWeights() { return m_prev_layer_size * m_layer_size; }

    int numBias() { return m_layer_size; }

    virtual double weightInitialization() const = 0;

    virtual Vector activationFunction(const Vector& input) { return input; };

    int getLayerSize() const { return m_layer_size; }
    int getPrevLayerSize() const { return m_prev_layer_size; }
    int getPos() const { return m_pos; }

    void setPos(int pos) { m_pos = pos; }

   protected:
    Vector m_input;  // input to the layer
    Vector m_delta;
    Vector m_weightedSum;
    int m_pos = -1;  // the index of layer in the whole neural network, i.e. indexed with 0 is input layer
    int m_prev_layer_size;
    int m_layer_size;
    double L2_lambda;
};

#endif  // NEURONLAYER_HPP
