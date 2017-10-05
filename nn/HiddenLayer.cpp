#include "HiddenLayer.hpp"

HiddenLayer::HiddenLayer(int prev_layer_size, int layer_size, double L2)
    : NeuronLayer(prev_layer_size, layer_size, L2) {
    m_delta.resize(m_layer_size);
}

VectorXd HiddenLayer::forwardPropagation(const VectorXd& input, std::set<int> nn_node_set, bool training,
                                                    NN_parameters* m_theta) {
    assert(nn_node_set.size() <= m_layer_size);  // debug
    assert(input.rows() == m_prev_layer_size);   // debug

    m_node_set = std::move(nn_node_set);
    m_input = input;

    // Update statistics
    if (training) {
        m_total_nn_set_size += m_node_set.size();
        assert(m_total_nn_set_size >= 0);  // debug
        m_total_multiplication += m_node_set.size() * m_prev_layer_size;
    }

    m_weightedSum.resize(m_layer_size, 0.0);
    for (auto& idx : m_node_set) {
        double prod = 0;
        auto weights = m_theta->getWeightVector(m_pos, idx);
        for (int i = 0; i < input.size(); i++) prod += weights[i] * input(i);
        m_weightedSum[idx] = prod + m_theta->getBias(m_pos, idx);
    }

    return activationFunction(m_weightedSum);
}

VectorXd HiddenLayer::forwardPropagation(const VectorXd& input, NN_parameters* m_theta) {
    return forwardPropagation(input, false, m_theta);
}

VectorXd HiddenLayer::forwardPropagation(const std::vector<double>& input, bool training,
                                                    NN_parameters* m_theta) {
    return forwardPropagation(vectorize(input), training, m_theta);
}

// some forwardPropagation functions should be added later

VectorXd HiddenLayer::forwardPropagation(const VectorXd& input, bool training, NN_parameters* m_theta) {
    return forwardPropagation(input, m_theta->retrieveNodes(m_pos, input), training, m_theta);
}

VectorXd HiddenLayer::forwardPropagation(const VectorXd& input, const std::vector<int>& hashes,
                                                    bool training, NN_parameters* m_theta) {
    return forwardPropagation(input, m_theta->retrieveNodes(m_pos, hashes), training, m_theta);
}

VectorXd HiddenLayer::calculateDelta(const VectorXd& prev_layer_delta, NN_parameters* m_theta) {
    for (int idx : m_node_set) {
        for (int jdx = 0; jdx < prev_layer_delta.size(); ++jdx) {
            m_delta[idx] += m_theta->getWeight(m_pos + 1, jdx, idx) * prev_layer_delta[jdx];
        }
        m_delta[idx] *= derivative(m_weightedSum[idx]);
    }
    return m_delta;
}

void HiddenLayer::calculateGradient(NN_parameters* m_theta) {
    assert(m_delta.size() == m_layer_size);  // debug

    for (int idx : m_node_set) {
        // Set Weight Gradient
        for (int jdx = 0; jdx < m_prev_layer_size; ++jdx) {
            m_theta->stochasticGradientDescent(m_theta->weightOffset(m_pos, idx, jdx), m_delta[idx] * m_input(jdx));
        }

        // Set Bias Gradient
        m_theta->stochasticGradientDescent(m_theta->biasOffset(m_pos, idx), m_delta[idx]);
    }
}

void HiddenLayer::updateHashTables(double size) {
    std::cout << m_pos << " : " << m_total_nn_set_size / size << std::endl;
    std::cout << "[WARNING] function HiddenLayer::updateHashTables not implemented" << std::endl;
}

VectorXd HiddenLayer::vectorize(const std::vector<double>& data) { return vectorize(data, 0, data.size()); }

VectorXd HiddenLayer::vectorize(const std::vector<double>& data, int offset, int length) {
    VectorXd vector = VectorXd::Zero(length);
    for (int idx = 0; idx < length; ++idx) {
        vector(idx) = data[offset + idx];
    }
    return vector;
}
