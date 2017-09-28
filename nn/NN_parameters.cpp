#include "NN_parameters.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

using Eigen::VectorXd;

NN_parameters::NN_parameters(const std::vector<std::shared_ptr<NeuronLayer>>& NN_structure,
                             const std::vector<int>& pool_dim, const std::vector<double>& size_limit,
                             const std::vector<int>& b, const std::vector<int>& L, double learning_rate)
    : m_learning_rate(learning_rate), m_poolDim(pool_dim), m_b(b), m_L(L), m_size_limit(size_limit) {
    construct(NN_structure);
    weight_initialization(NN_structure);
    createLSHTable(m_tables, pool_dim, b, L, size_limit);

    auto n_layers = NN_structure.size();                                                  // debug
    assert(m_layer_col.size() == n_layers);                                               // debug
    assert(m_layer_row.size() == n_layers);                                               // debug
    assert(m_weight_idx.size() == n_layers);                                              // debug
    assert(m_bias_idx.size() == n_layers);                                                // debug
    assert(m_tables.size() == n_layers - 1);                                              // debug
    std::cout << "n_layers = " << m_layer_count << ", m_size = " << m_size << std::endl;  // debug
    std::cout << "Finished initializing parameters" << endl;
}

void NN_parameters::construct(const std::vector<std::shared_ptr<NeuronLayer>>& NN_structure) {
    auto n_layers = NN_structure.size();
    m_layer_row.reserve(n_layers);
    m_layer_col.reserve(n_layers);
    m_weight_idx.reserve(n_layers);
    m_bias_idx.reserve(n_layers);

    for (auto& l : NN_structure) {
        m_layer_row.push_back(l->m_layer_size);       // No. of neurons in each layer
        m_layer_col.push_back(l->m_prev_layer_size);  // Dimensionality of each layer

        // The weights and bias are stored in a flat array
        m_weight_idx.push_back(m_size);  // Starting index of weights of each layer
        m_size += l->numWeights();

        m_bias_idx.push_back(m_size);  // Starting index of bias of each layer
        m_size += l->numBias();
        l->m_pos = m_layer_count++;
        // TODO l->m_theta = this;
    }
    m_theta.resize(m_size);
    m_gradient.resize(m_size);
    m_momentum.resize(m_size);
    m_learning_rates.resize(m_size);
}

int NN_parameters::epoch_offset() { return m_epoch_offset; }

void NN_parameters::weight_initialization(const std::vector<std::shared_ptr<NeuronLayer>>& NN_structure) {
    int global_idx = -1;
    for (auto& l : NN_structure) {
        for (int idx = 0; idx < l->m_layer_size; ++idx) {
            for (int jdx = 0; jdx < l->m_prev_layer_size; ++jdx) {
                m_theta[++global_idx] = l->weightInitialization();
            }
        }
        global_idx += l->m_layer_size;
    }
    assert(global_idx == m_size - 1);  // debug
}

std::vector<std::vector<int>> NN_parameters::computeHashes(std::vector<VectorXd> data) {
    int interval = data.size() / 10;
    std::vector<std::vector<int>> hashes(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        // Progress log
        if (i % interval == 0) {
            std::cout << "Completed " << i << " / " << data.size() << std::endl;
        }
        // input layer neuron selection TODO
        hashes[i] = m_tables[0].generateHashSignature(data[i]);
    }
    return hashes;
}

void NN_parameters::rebuildTables() {
    // to be defined
}

void NN_parameters::createLSHTable(std::vector<HashBuckets>& tables, const std::vector<int>& poolDim,
                                   const std::vector<int>& b, const std::vector<int>& L,
                                   const std::vector<double>& size_limit) {
    int global_idx = 0;
    for (int layer_idx = 0; layer_idx < m_layer_count - 1; ++layer_idx) {
        HashBuckets table(size_limit[layer_idx], poolDim[layer_idx], L[layer_idx],
                          CosineDistance(b[layer_idx], L[layer_idx], m_layer_col[layer_idx] / poolDim[layer_idx]));
        for (int idx = 0; idx < m_layer_row[layer_idx]; ++idx) {
            std::vector<double> row(m_theta.begin() + global_idx,
                                    m_theta.begin() + global_idx + m_layer_col[layer_idx]);
            // initialize a Eigen::VectorXd using std::vector<double>
            table.LSHAdd(idx, vectorize(m_theta, global_idx, m_layer_col[layer_idx]));
            global_idx += m_layer_col[layer_idx];
        }
        tables.push_back(table);  // TODO avoid copy, may implement a move constructor
        global_idx += m_layer_row[layer_idx];
    }
}

// consider replacing std::vector<double> with VectorXd
std::vector<double> NN_parameters::getWeight(int layer, int node) {
    assert(layer >= 0 && layer < m_layer_count);
    assert(node >= 0 && node < m_layer_row[layer]);

    std::vector<double>::const_iterator first = m_theta.begin() + node * m_layer_col[layer];
    std::vector<double>::const_iterator last = m_theta.begin() + (node + 1) * m_layer_col[layer];
    std::vector<double> row(first, last);
    return row;
}

set<int> NN_parameters::retrieveNodes(int layer, VectorXd input) { return m_tables[layer].histogramLSH(input); }

set<int> NN_parameters::retrieveNodes(int layer, std::vector<int> hashes) {
    return m_tables[layer].histogramLSH(hashes);
}

void NN_parameters::timeStep() {
    m_momentum_lambda *= momentum_rate;
    m_momentum_lambda = min(m_momentum_lambda, momentum_max);
}

int NN_parameters::size() { return m_size; }

double NN_parameters::getGradient(int idx) {
    assert(idx >= 0 && idx < m_theta.size());
    return m_momentum[idx] / m_learning_rate;
}

double NN_parameters::getTheta(int idx) {
    assert(idx >= 0 && idx < m_theta.size());
    return m_theta[idx];
}

void NN_parameters::setTheta(int idx, double value) {
    assert(idx >= 0 && idx < m_theta.size());
    m_theta[idx] = value;
}

double NN_parameters::getWeight(int layer, int row, int col) {
    assert(layer >= 0 && layer < m_layer_count);
    assert(row >= 0 && row < m_layer_row[layer]);
    assert(col >= 0 && col < m_layer_col[layer]);

    int idx = row * m_layer_col[layer] + col;
    return m_theta[m_weight_idx[layer] + idx];
}

void NN_parameters::setWeight(int layer, int row, int col, double value) {
    assert(layer >= 0 && layer < m_layer_count);
    assert(row >= 0 && row < m_layer_row[layer]);
    assert(col >= 0 && col < m_layer_col[layer]);

    int idx = row * m_layer_col[layer] + col;
    m_theta[m_weight_idx[layer] + idx] = value;
}

std::vector<double> NN_parameters::getWeightVector(int layer, int node) {
    assert(layer >= 0 && layer < m_layer_count);
    assert(node >= 0 && node < m_layer_row[layer]);

    std::vector<double>::const_iterator first = m_theta.begin() + node * m_layer_col[layer];
    std::vector<double>::const_iterator last = m_theta.begin() + (node + 1) * m_layer_col[layer];
    std::vector<double> row(first, last);
    return row;
}

double NN_parameters::getBias(int layer, int idx) { return m_theta[m_bias_idx[layer] + idx]; }

void NN_parameters::setBias(int layer, int idx, double value) { m_theta[m_bias_idx[layer] + idx] = value; }

double NN_parameters::L2_regularization() {
    double L2 = 0.0;
    for (int layer_idx = 0; layer_idx < m_layer_count; ++layer_idx) {
        for (int i = m_weight_idx[layer_idx]; i < m_bias_idx[layer_idx]; ++i) {
            L2 += pow(m_theta[i], 2.0);
        }
    }
    return 0.5 * L2;
}

int NN_parameters::weightOffset(int layer, int row, int column) {
    int idx = row * m_layer_col[layer] + column;
    return m_weight_idx[layer] + idx;
}

int NN_parameters::biasOffset(int layer, int idx) { return m_bias_idx[layer] + idx; }

void NN_parameters::stochasticGradientDescent(int idx, double gradient) {
    m_gradient[idx] = gradient;
    m_learning_rates[idx] += pow(gradient, 2.0);
    double learning_rate = m_learning_rate / (pow(10, -6) + sqrt(m_learning_rates[idx]));
    m_momentum[idx] *= m_momentum_lambda;
    m_momentum[idx] += learning_rate * gradient;
    m_theta[idx] -= m_momentum[idx];
}

void NN_parameters::clear_gradient() { m_gradient.resize(m_size, 0); }

VectorXd NN_parameters::vectorize(std::vector<double> data) { return vectorize(data, 0, data.size()); }

VectorXd NN_parameters::vectorize(std::vector<double> data, int offset, int length) {
    VectorXd vector = VectorXd::Zero(length);
    for (int idx = 0; idx < length; ++idx) {
        vector(idx) = data[offset + idx];
    }
    return vector;
}
