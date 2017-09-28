#ifndef RELUNEURONLAYER_HPP
#define RELUNEURONLAYER_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>
#include "HiddenLayer.hpp"

using namespace std;

class ReLUNeuronLayer : public HiddenLayer {
   public:
    ReLUNeuronLayer(int prev_layer_size, int layer_size, double L2);
    std::shared_ptr<NeuronLayer> clone() const override;

   protected:
    double weightInitialization() const override;

    std::vector<double> activationFunction(std::vector<double> input) override;

    double derivative(double input) override;
};

#endif  // RELUNEURONLAYER_HPP
