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
    std::shared_ptr<NeuronLayer> clone() const override;

   protected:
    double weightInitialization() const override;

    VectorXd activationFunction(const std::vector<double>& input) override;

    double derivative(double input) override;
};

#endif  // RELUNEURONLAYER_HPP
