#include "nn/HiddenLayer.hpp"

#include "exp/colors.hpp"
#include "nn/utils.hpp"

HiddenLayerWithDropOut::HiddenLayerWithDropOut(int prev_layer_size, int layer_size, double L2)
    : NeuronLayer<SparseVectorXd>(prev_layer_size, layer_size, L2) {}

SparseVectorXd HiddenLayerWithDropOut::forwardPropagation(const SparseVectorXd& input, std::set<int> nn_node_set,
                                                          bool training, NN_parameters* m_theta) {
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

    m_weightedSum.setZero();
    for (auto& idx : m_node_set) {
        auto weights = m_theta->getWeightVector(m_pos, idx);
        double prod = input.dot(weights);
        m_weightedSum.coeffRef(idx) = prod + m_theta->getBias(m_pos, idx);
    }

    return activationFunction(m_weightedSum);
}

SparseVectorXd HiddenLayerWithDropOut::forwardPropagation(const SparseVectorXd& input, bool training,
                                                          NN_parameters* m_theta) {
    return forwardPropagation(input, m_theta->retrieveNodes(m_pos, input), training, m_theta);
}

SparseVectorXd HiddenLayerWithDropOut::forwardPropagation(const SparseVectorXd& input, NN_parameters* m_theta) {
    return forwardPropagation(input, false, m_theta);
}

SparseVectorXd HiddenLayerWithDropOut::forwardPropagation(const std::vector<double>& input, bool training,
                                                          NN_parameters* m_theta) {
    return forwardPropagation(sparse_vectorize(input), training, m_theta);
}

SparseVectorXd HiddenLayerWithDropOut::forwardPropagation(const SparseVectorXd& input, const std::vector<int>& hashes,
                                                          bool training, NN_parameters* m_theta) {
    return forwardPropagation(input, m_theta->retrieveNodes(m_pos, hashes), training, m_theta);
}

SparseVectorXd HiddenLayerWithDropOut::calculateDelta(const SparseVectorXd& prev_layer_delta, NN_parameters* m_theta) {
    m_delta.setZero();
    for (int idx : m_node_set) {
        m_delta.coeffRef(idx) =
            prev_layer_delta.dot(m_theta->getBackWeightVector(m_pos, idx)) * derivative(m_weightedSum.coeffRef(idx));
    }
    return m_delta;
}

void HiddenLayerWithDropOut::calculateGradient(NN_parameters* m_theta) {
    assert(m_delta.size() == m_layer_size);  // debug

    for (int idx : m_node_set) {
        // Set Weight Gradient
        for (int jdx = 0; jdx < m_prev_layer_size; ++jdx) {  // TODO: use vectorxd
            m_theta->stochasticGradientDescent(m_theta->weightOffset(m_pos, idx, jdx),
                                               m_delta.coeff(idx) * m_input.coeff(jdx));
        }

        // Set Bias Gradient
        m_theta->stochasticGradientDescent(m_theta->biasOffset(m_pos, idx), m_delta.coeff(idx));
    }
}

// ================ Hidden Layer ================ //
HiddenLayer::HiddenLayer(int prev_layer_size, int layer_size, double L2)
    : NeuronLayer(prev_layer_size, layer_size, L2) {
    m_delta.resize(m_layer_size);
    m_delta.setZero();
}

VectorXd HiddenLayer::forwardPropagation(const VectorXd& input, NN_parameters* m_theta) {
    return forwardPropagation(input, false, m_theta);
}
VectorXd HiddenLayer::forwardPropagation(const VectorXd& input, bool training, NN_parameters* m_theta) {
    m_input = input;

    m_weightedSum = VectorXd::Zero(m_layer_size);
    for (int idx = 0; idx < m_layer_size; ++idx) {
        double prod = 0;
        auto weights = m_theta->getWeightVector(m_pos, idx);
        for (int i = 0; i < input.size(); i++) prod += weights[i] * input(i);
        m_weightedSum.coeffRef(idx) = prod + m_theta->getBias(m_pos, idx);
    }

    return activationFunction(m_weightedSum);
}

VectorXd HiddenLayer::calculateDelta(const VectorXd& prev_layer_delta, NN_parameters* m_theta) {
    m_delta = VectorXd::Zero(m_layer_size);
    for (int idx = 0; idx < m_layer_size; ++idx) {
        for (int jdx = 0; jdx < prev_layer_delta.size(); ++jdx) {
            m_delta[idx] += m_theta->getWeight(m_pos + 1, jdx, idx) * prev_layer_delta[jdx];
        }
        m_delta[idx] *= derivative(m_weightedSum.coeffRef(idx));
    }
    return m_delta;
}

void HiddenLayer::calculateGradient(NN_parameters* m_theta) {
    assert(m_delta.size() == m_layer_size);  // debug

    for (int idx = 0; idx < m_layer_size; ++idx) {
        // Set Weight Gradient
        for (int jdx = 0; jdx < m_prev_layer_size; ++jdx) {
            m_theta->stochasticGradientDescent(m_theta->weightOffset(m_pos, idx, jdx), m_delta[idx] * m_input(jdx));
        }

        // Set Bias Gradient
        m_theta->stochasticGradientDescent(m_theta->biasOffset(m_pos, idx), m_delta[idx]);
    }
}
