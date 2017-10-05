#include <cmath>
#include <random>
#include <vector>
#include "HiddenLayer.hpp"

class SoftMaxNeuronLayer : public NeuronLayer {
   public:
    SoftMaxNeuronLayer(int prev_layer_size, int layer_size, double L2);

    VectorXd forwardPropagation(const VectorXd& input, NN_parameters* m_theta) override;

    VectorXd calculateDelta(const VectorXd& prev_layer_delta, NN_parameters* m_theta) override;

    void calculateGradient(NN_parameters* m_theta) override;

    std::shared_ptr<NeuronLayer> clone() const override;

   protected:
    double weightInitialization() const override;

    VectorXd activationFunction(const std::vector<double>& input) override;
};
