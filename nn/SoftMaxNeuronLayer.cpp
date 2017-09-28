#include "SoftMaxNeuronLayer.hpp"

SoftMaxNeuronLayer::SoftMaxNeuronLayer(int prev_layer_size, int layer_size, double L2) : HiddenLayer(prev_layer_size, layer_size, L2) {}

vector<double> SoftMaxNeuronLayer::forwardPropagation(vector<double> input, NN_parameters m_theta)
{
    assert(input.size() == m_prev_layer_size);
    m_input = input;

    for (int jdx = 0; jdx < m_layer_size; ++jdx)
    {
        m_weightedSum[jdx] = 0.0;
        for (int idx = 0; idx < m_prev_layer_size; ++idx)
        {
            m_weightedSum[jdx] += m_theta.getWeight(m_pos, jdx, idx) * m_input[idx];
        }
        m_weightedSum[jdx] += m_theta.getBias(m_pos, jdx);
    }
    return activationFunction(m_weightedSum);
}

vector<double> SoftMaxNeuronLayer::calculateDelta(vector<double> prev_layer_delta, NN_parameters m_theta)
{
    assert(prev_layer_delta.size() == m_layer_size);
    m_delta = prev_layer_delta;
    return m_delta;
}

void SoftMaxNeuronLayer::calculateGradient(NN_parameters m_theta)
{
    assert(m_delta.size() == m_layer_size);
    for (int idx = 0; idx < m_layer_size; ++idx)
    {
        // set weight gradient
        for (int jdx = 0; jdx < m_prev_layer_size; ++jdx)
        {
            m_theta.stochasticGradientDescent(m_theta.weightOffset(m_pos, idx, jdx), m_delta[idx] * m_input[jdx] + L2_lambda * m_theta.getWeight(m_pos, idx, jdx));
        }
        // set bias gradient
        m_theta.stochasticGradientDescent(m_theta.biasOffset(m_pos, idx), m_delta[idx]);
    }
}

double SoftMaxNeuronLayer::weightInitialization()
{
    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<double> distribution(0.0, 1.0);
    double interval = 2.0 * sqrt(6.0 / (m_prev_layer_size + m_layer_size));
    return distribution(generator) * (2 * interval) - interval;
}

vector<double> SoftMaxNeuronLayer::activationFunction(vector<double> input)
{
    double sum = 0.0;
    vector<double> output(input.size());
    for (auto& ele : input)
    {
        double temp = exp(ele);
        output.push_back(temp);
        sum += temp;
    }
    for (auto& ele : output)
    {
        ele /= sum;
    }
    return output;
}

double SoftMaxNeuronLayer::derivative(double input)
{
    return (input > 0) ? 1.0 : 0.0;
}