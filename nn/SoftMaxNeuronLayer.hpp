#include <cmath>
#include <random>
#include <vector>

#include "NN_parameters.hpp"
#include "NeuronLayer.hpp"

template <typename Vector>
class SoftMaxNeuronLayer : public NeuronLayer<Vector> {
   public:
    SoftMaxNeuronLayer(int prev_layer_size, int layer_size, double L2)
        : NeuronLayer<Vector>(prev_layer_size, layer_size, L2) {}

    Vector forwardPropagation(const Vector& input, NN_parameters* m_theta) override {
        assert(input.size() == this->m_prev_layer_size);
        this->m_input = input;

        setZero(this->m_weightedSum, this->m_layer_size);
        for (int jdx = 0; jdx < this->m_layer_size; ++jdx) {
            this->m_weightedSum.coeffRef(jdx) = this->m_input.dot(m_theta->getWeightVector(this->m_pos, jdx));
            this->m_weightedSum.coeffRef(jdx) += m_theta->getBias(this->m_pos, jdx);
        }
        return activationFunction(this->m_weightedSum);
    }

    void setZero(Eigen::SparseVector<double>& vec, int size) {
        vec.resize(size);
        vec.setZero();
    }
    void setZero(Eigen::VectorXd& vec, int size) {
        vec.resize(size);
        vec.setZero();
    }

    Vector calculateDelta(const Vector& prev_layer_delta, NN_parameters* m_theta) override {
        assert(prev_layer_delta.size() == this->m_layer_size);
        this->m_delta = prev_layer_delta;
        return this->m_delta;
    }

    void calculateGradient(NN_parameters* m_theta) override {
        assert(this->m_delta.size() == this->m_layer_size);
        for (int idx = 0; idx < this->m_layer_size; ++idx) {
            // set weight gradient
            for (int jdx = 0; jdx < this->m_prev_layer_size; ++jdx) {
                m_theta->stochasticGradientDescent(m_theta->weightOffset(this->m_pos, idx, jdx),
                                                   this->m_delta.coeff(idx) * this->m_input.coeff(jdx) +
                                                       this->L2_lambda * m_theta->getWeight(this->m_pos, idx, jdx));
            }
            // set bias gradient
            m_theta->stochasticGradientDescent(m_theta->biasOffset(this->m_pos, idx), this->m_delta.coeff(idx));
        }
    }

    std::shared_ptr<NeuronLayer<Vector>> clone() const override {
        auto copy =
            std::make_shared<SoftMaxNeuronLayer<Vector>>(this->m_prev_layer_size, this->m_layer_size, this->L2_lambda);
        copy->m_pos = this->m_pos;
        return copy;
    }

   protected:
    double weightInitialization() const override {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double interval = 2.0 * sqrt(6.0 / (this->m_prev_layer_size + this->m_layer_size));
        return distribution(generator) * (2 * interval) - interval;
    }

    Vector activationFunction(const Vector& input) override {
        Vector output = input.unaryExpr([](double ele) { return std::exp(ele); });
        output /= output.sum();
        return output;
    }
};
