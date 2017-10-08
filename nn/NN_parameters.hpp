#ifndef NN_PARAMETERS_HPP
#define NN_PARAMETERS_HPP

#include <glog/logging.h>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#include "exp/colors.hpp"
#include "lsh/HashBuckets.hpp"
#include "nn/utils.hpp"

using Eigen::VectorXd;

class NN_parameters {
   public:
    NN_parameters() = default;

    template <typename NeuronLayer>
    NN_parameters(const std::vector<std::shared_ptr<NeuronLayer>>& NN_structure, const std::vector<int>& pool_dim,
                  const std::vector<double>& size_limit, const std::vector<int>& b, const std::vector<int>& L,
                  double learning_rate, bool dropout)
        : m_learning_rate(learning_rate), m_poolDim(pool_dim), m_b(b), m_L(L), m_size_limit(size_limit) {
        construct(NN_structure);
        weight_initialization(NN_structure);
        if (dropout) createLSHTable(&m_tables, pool_dim, b, L, size_limit);

        DLOG(INFO) << GREEN("[INFO] ") << "Finished initializing parameters";
    }

    int epoch_offset() { return m_epoch_offset; }
    void set_epoch_offset(int offset) { m_epoch_offset = offset; }

    std::vector<std::vector<int>> computeHashes(std::vector<VectorXd>& data) {  // unused
        int interval = data.size() / 10;
        std::vector<std::vector<int>> hashes(data.size());

        for (size_t i = 0; i < data.size(); ++i) {
            // Progress log
            if (i % interval == 0) {
                std::cout << "Completed " << i << " / " << data.size() << std::endl;
            }
            // input layer neuron selection
            hashes[i] = m_tables[0].generateHashSignature(data[i]);
        }
        return hashes;
    }

    void rebuildTables() {
        int global_idx = 0;
        for (int layer_idx = 0; layer_idx < m_layer_count - 1; ++layer_idx) {
            m_tables[layer_idx].clear();
            for (int idx = 0; idx < m_layer_row[layer_idx]; ++idx) {
                m_tables[layer_idx].LSHAdd(idx, vectorize(m_theta, global_idx, m_layer_col[layer_idx]));
                global_idx += m_layer_col[layer_idx];
            }
            global_idx += m_layer_row[layer_idx];  // skip bias
        }
    }

    // consider replacing std::vector<double> with VectorXd
    std::vector<double> getWeight(int layer, int node) {
        assert(layer >= 0 && layer < m_layer_count);
        assert(node >= 0 && node < m_layer_row[layer]);

        std::vector<double>::const_iterator first = m_theta.begin() + m_weight_idx[layer] + node * m_layer_col[layer];
        std::vector<double>::const_iterator last = first + m_layer_col[layer];
        std::vector<double> row(first, last);
        return row;
    }

    template <typename Vector>
    std::set<int> retrieveNodes(int layer, const Vector& input) {
        return m_tables[layer].histogramLSH(input);
    }

    std::set<int> retrieveNodes(int layer, const std::vector<int>& hashes) {
        return m_tables[layer].histogramLSH(hashes);
    }

    void timeStep() {
        m_momentum_lambda *= momentum_rate;
        m_momentum_lambda = std::min(m_momentum_lambda, momentum_max);
    }

    int size() { return m_size; }

    double getGradient(int idx) {  // unused
        assert(idx >= 0 && idx < m_theta.size());
        return m_momentum[idx] / m_learning_rate;
    }

    double getTheta(int idx) {  // for test
        assert(idx >= 0 && idx < m_theta.size());
        return m_theta[idx];
    }

    void setTheta(int idx, double value) {  // for test
        assert(idx >= 0 && idx < m_theta.size());
        m_theta[idx] = value;
    }

    double getWeight(int layer, int row, int col) {
        assert(layer >= 0 && layer < m_layer_count);
        assert(row >= 0 && row < m_layer_row[layer]);
        assert(col >= 0 && col < m_layer_col[layer]);

        int idx = row * m_layer_col[layer] + col;
        return m_theta[m_weight_idx[layer] + idx];
    }

    void setWeight(int layer, int row, int col, double value) {
        assert(layer >= 0 && layer < m_layer_count);
        assert(row >= 0 && row < m_layer_row[layer]);
        assert(col >= 0 && col < m_layer_col[layer]);

        int idx = row * m_layer_col[layer] + col;
        m_theta[m_weight_idx[layer] + idx] = value;
    }

    VectorXd getWeightVector(int layer, int node) {
        assert(layer >= 0 && layer < m_layer_count);
        assert(node >= 0 && node < m_layer_row[layer]);

        auto offset = m_weight_idx[layer] + node * m_layer_col[layer];
        return vectorize(m_theta, offset, m_layer_col[layer]);
    }

    VectorXd getBackWeightVector(int layer, int node) {  // reverse direction weight vector for BP
        auto length = m_layer_row[layer + 1];
        VectorXd vector = VectorXd::Zero(length);
        for (int idx = 0; idx < length; ++idx) {
            vector[idx] = m_theta[idx * m_layer_col[layer + 1] + node + m_weight_idx[layer + 1]];
        }
        return vector;
    }

    double getBias(int layer, int idx) { return m_theta[m_bias_idx[layer] + idx]; }

    void setBias(int layer, int idx, double value) { m_theta[m_bias_idx[layer] + idx] = value; }

    double L2_regularization() {
        double L2 = 0.0;
        for (int layer_idx = 0; layer_idx < m_layer_count; ++layer_idx) {
            for (int i = m_weight_idx[layer_idx]; i < m_bias_idx[layer_idx]; ++i) {
                L2 += m_theta[i] * m_theta[i];
            }
        }
        return 0.5 * L2;
    }

    int weightOffset(int layer, int row, int column) {
        int idx = row * m_layer_col[layer] + column;
        return m_weight_idx[layer] + idx;
    }

    int biasOffset(int layer, int idx) { return m_bias_idx[layer] + idx; }

    void clear_gradient() { m_gradient.resize(m_size, 0); }

    void stochasticGradientDescent(int idx, double gradient) {
        m_gradient[idx] = gradient;
        m_learning_rates[idx] += gradient * gradient;
        double learning_rate = m_learning_rate / (1e-6 + sqrt(m_learning_rates[idx]));
        m_momentum[idx] *= m_momentum_lambda;
        m_momentum[idx] += learning_rate * gradient;
        m_theta[idx] -= m_momentum[idx];
    }

   private:
    template <typename NeuronLayer>
    void construct(const std::vector<std::shared_ptr<NeuronLayer>>& NN_structure) {
        auto n_layers = NN_structure.size();
        m_layer_row.reserve(n_layers);
        m_layer_col.reserve(n_layers);
        m_weight_idx.reserve(n_layers);
        m_bias_idx.reserve(n_layers);

        for (auto& l : NN_structure) {
            m_layer_row.push_back(l->getLayerSize());      // No. of neurons in each layer
            m_layer_col.push_back(l->getPrevLayerSize());  // Dimensionality of each layer

            // The weights and bias are stored in a flat array
            m_weight_idx.push_back(m_size);  // Starting index of weights of each layer
            m_size += l->numWeights();

            m_bias_idx.push_back(m_size);  // Starting index of bias of each layer
            m_size += l->numBias();
            l->setPos(m_layer_count++);
        }
        m_theta.resize(m_size, 0.0);
        m_gradient.resize(m_size, 0.0);
        m_momentum.resize(m_size, 0.0);
        m_learning_rates.resize(m_size, 0.0);
    }

    template <typename NeuronLayer>
    void weight_initialization(const std::vector<std::shared_ptr<NeuronLayer>>& NN_structure) {
        int global_idx = -1;
        for (auto& l : NN_structure) {
            for (int idx = 0; idx < l->getLayerSize(); ++idx) {
                for (int jdx = 0; jdx < l->getPrevLayerSize(); ++jdx) {
                    m_theta[++global_idx] = l->weightInitialization();
                }
            }
            global_idx += l->getLayerSize();
        }
        assert(global_idx == m_size - 1);  // debug
    }

    void createLSHTable(std::vector<HashBuckets>* tables, const std::vector<int>& poolDim, const std::vector<int>& b,
                        const std::vector<int>& L, const std::vector<double>& size_limit) {
        int global_idx = 0;
        tables->reserve(m_layer_count - 1);
        for (int layer_idx = 0; layer_idx < m_layer_count - 1; ++layer_idx) {
            tables->push_back(
                HashBuckets(size_limit[layer_idx] * m_layer_row[layer_idx], poolDim[layer_idx], L[layer_idx],
                            CosineDistance(b[layer_idx], L[layer_idx], m_layer_col[layer_idx] / poolDim[layer_idx])));
            auto& table = (*tables)[layer_idx];
            for (int idx = 0; idx < m_layer_row[layer_idx]; ++idx) {
                // initialize a Eigen::VectorXd using std::vector<double>
                table.LSHAdd(idx, vectorize(m_theta, global_idx, m_layer_col[layer_idx]));
                global_idx += m_layer_col[layer_idx];
            }
            global_idx += m_layer_row[layer_idx];
        }
    }

    int m_epoch_offset = 0;

    std::vector<int> m_weight_idx;
    std::vector<int> m_bias_idx;
    int m_size = 0;
    int m_layer_count = 0;
    std::vector<int> m_layer_row;
    std::vector<int> m_layer_col;

    // stochastic gradient decsent
    std::vector<double> m_theta;
    std::vector<double> m_gradient;

    // momentum
    std::vector<double> m_momentum;
    double m_momentum_lambda = 0.50;
    double momentum_max = 0.90;
    double momentum_rate = 1.00;

    // learning rate - adagrad
    double m_learning_rate;  // this one is final in java implementation
    std::vector<double> m_learning_rates;

    // LSH
    std::vector<HashBuckets> m_tables;
    std::vector<int> m_poolDim;
    std::vector<int> m_b;  // final
    std::vector<int> m_L;  // final
    std::vector<double> m_size_limit;
};

#endif  // NN_PARAMETERS_HPP
