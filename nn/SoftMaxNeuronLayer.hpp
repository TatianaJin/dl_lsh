#include <cmath>
#include <random>
#include <vector>
#include "HiddenLayer.hpp"

class SoftMaxNeuronLayer : public HiddenLayer {
   public:
    SoftMaxNeuronLayer(int prev_layer_size, int layer_size, double L2);

    std::vector<double> forwardPropagation(std::vector<double> input, NN_parameters* m_theta) override;

    std::vector<double> calculateDelta(std::vector<double> prev_layer_size, NN_parameters* m_theta) override;

    void calculateGradient(NN_parameters* m_theta) override;

    std::shared_ptr<NeuronLayer> clone() const override;

   protected:
    double weightInitialization() const override;

    std::vector<double> activationFunction(std::vector<double> input) override;

    double derivative(double input) override;
};
