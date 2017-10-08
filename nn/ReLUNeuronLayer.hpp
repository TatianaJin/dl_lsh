#ifndef RELUNEURONLAYER_HPP
#define RELUNEURONLAYER_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "HiddenLayer.hpp"

class ReLUNeuronLayer : public HiddenLayer {
   public:
    ReLUNeuronLayer(int prev_layer_size, int layer_size, double L2);
    std::shared_ptr<NeuronLayer<VectorXd>> clone() const override;

   protected:
    double weightInitialization() const override;

    VectorXd activationFunction(const VectorXd& input) override;

    double derivative(double input) override;
};

class ReLUNeuronLayerWithDropOut : public HiddenLayerWithDropOut {
   public:
    ReLUNeuronLayerWithDropOut(int prev_layer_size, int layer_size, double L2);
    std::shared_ptr<NeuronLayer<SparseVectorXd>> clone() const override;

   protected:
    double weightInitialization() const override;

    SparseVectorXd activationFunction(const SparseVectorXd& input) override;

    double derivative(double input) override;
};

#endif  // RELUNEURONLAYER_HPP
