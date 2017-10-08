#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "nn/NeuronLayer.hpp"
#include "nn/ReLUNeuronLayer.hpp"
#include "nn/SoftMaxNeuronLayer.hpp"

using Eigen::VectorXd;
using SparseVectorXd = Eigen::SparseVector<double>;

class TestNNParameters : public testing::Test {
   protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(TestNNParameters, InitDropout) {
    // construct layers
    std::vector<std::shared_ptr<NeuronLayer<SparseVectorXd>>> layers;
    layers.reserve(2);
    layers.push_back(std::make_shared<ReLUNeuronLayerWithDropOut>(5, 2, 0.1));
    layers.push_back(std::make_shared<SoftMaxNeuronLayer<SparseVectorXd>>(2, 2, 0.1));
    NN_parameters params(layers, {2}, {0.05}, {1}, {1}, 0.01, true);
    EXPECT_EQ(params.size(), 18);
}

TEST_F(TestNNParameters, InitNormal) {
    std::vector<std::shared_ptr<NeuronLayer<VectorXd>>> layers;
    // construct layers
    layers.reserve(2);
    layers.push_back(std::make_shared<ReLUNeuronLayer>(5, 2, 0.1));
    layers.push_back(std::make_shared<SoftMaxNeuronLayer<VectorXd>>(2, 2, 0.1));
    NN_parameters params(layers, {2}, {0.05}, {1}, {1}, 0.01, false);
    EXPECT_EQ(params.size(), 18);
}

TEST_F(TestNNParameters, SetGetTheta) {
    // construct layers
    std::vector<std::shared_ptr<NeuronLayer<SparseVectorXd>>> layers;
    layers.reserve(2);
    layers.push_back(std::make_shared<ReLUNeuronLayerWithDropOut>(5, 2, 0.1));
    layers.push_back(std::make_shared<SoftMaxNeuronLayer<SparseVectorXd>>(2, 2, 0.1));
    NN_parameters params(layers, {2}, {0.05}, {1}, {1}, 0.01, true);

    for (int i = 0; i < 18; ++i) {
        params.setTheta(i, i);
        EXPECT_EQ(params.getTheta(i), i);
    }
}

TEST_F(TestNNParameters, SetGetWeight) {
    // construct layers
    std::vector<std::shared_ptr<NeuronLayer<SparseVectorXd>>> layers;
    layers.reserve(2);
    layers.push_back(std::make_shared<ReLUNeuronLayerWithDropOut>(5, 2, 0.1));
    layers.push_back(std::make_shared<SoftMaxNeuronLayer<SparseVectorXd>>(2, 2, 0.1));
    NN_parameters params(layers, {2}, {0.05}, {1}, {1}, 0.01, true);

    for (int i = 0; i < 18; ++i) {
        params.setTheta(i, i);
    }
    const auto& weights_0_1 = params.getWeight(0, 1);  // layer 0, node 1
    EXPECT_EQ(weights_0_1, std::vector<double>({5, 6, 7, 8, 9}));

    params.setWeight(1, 1, 0, 100);
    EXPECT_EQ(params.getWeight(1, 1, 0), 100);
    const auto& weights_1_1 = params.getWeight(1, 1);
    EXPECT_EQ(weights_1_1, std::vector<double>({100, 15}));
}

TEST_F(TestNNParameters, SetGetBias) {
    // construct layers
    std::vector<std::shared_ptr<NeuronLayer<SparseVectorXd>>> layers;
    layers.reserve(2);
    layers.push_back(std::make_shared<ReLUNeuronLayerWithDropOut>(5, 2, 0.1));
    layers.push_back(std::make_shared<SoftMaxNeuronLayer<SparseVectorXd>>(2, 2, 0.1));
    NN_parameters params(layers, {2}, {0.05}, {1}, {1}, 0.01, true);

    for (int i = 0; i < 18; ++i) {
        params.setTheta(i, i);
    }
    EXPECT_EQ(params.getBias(0, 1), 11);
    params.setBias(1, 1, 100);
    EXPECT_EQ(params.getBias(1, 1), 100);
}

TEST_F(TestNNParameters, GetWeightVector) {
    // construct layers
    std::vector<std::shared_ptr<NeuronLayer<SparseVectorXd>>> layers;
    layers.reserve(2);
    layers.push_back(std::make_shared<ReLUNeuronLayerWithDropOut>(5, 2, 0.1));
    layers.push_back(std::make_shared<SoftMaxNeuronLayer<SparseVectorXd>>(2, 2, 0.1));
    NN_parameters params(layers, {2}, {0.05}, {1}, {1}, 0.01, true);

    for (int i = 0; i < 18; ++i) {
        params.setTheta(i, i);
    }

    const auto& vec = params.getWeightVector(0, 0);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

TEST_F(TestNNParameters, GetBackWeightVector) {
    // construct layers
    std::vector<std::shared_ptr<NeuronLayer<SparseVectorXd>>> layers;
    layers.reserve(2);
    layers.push_back(std::make_shared<ReLUNeuronLayerWithDropOut>(5, 2, 0.1));
    layers.push_back(std::make_shared<SoftMaxNeuronLayer<SparseVectorXd>>(2, 2, 0.1));
    NN_parameters params(layers, {2}, {0.05}, {1}, {1}, 0.01, true);

    for (int i = 0; i < 18; ++i) {
        params.setTheta(i, i);
    }

    const auto& vec = params.getBackWeightVector(0, 0);
    ASSERT_EQ(vec.size(), 2);
    EXPECT_EQ(vec[0], 12);
    EXPECT_EQ(vec[1], 14);
}

TEST_F(TestNNParameters, L2Regularization) {
    // construct layers
    std::vector<std::shared_ptr<NeuronLayer<SparseVectorXd>>> layers;
    layers.reserve(2);
    layers.push_back(std::make_shared<ReLUNeuronLayerWithDropOut>(5, 2, 0.1));
    layers.push_back(std::make_shared<SoftMaxNeuronLayer<SparseVectorXd>>(2, 2, 0.1));
    NN_parameters params(layers, {2}, {0.05}, {1}, {1}, 0.01, true);

    for (int i = 0; i < 18; ++i) {
        params.setTheta(i, 2);
    }

    EXPECT_EQ(params.L2_regularization(), 28);  // 2^2 * 14 * 0.5
}

TEST_F(TestNNParameters, Offsets) {
    // construct layers
    std::vector<std::shared_ptr<NeuronLayer<SparseVectorXd>>> layers;
    layers.reserve(2);
    layers.push_back(std::make_shared<ReLUNeuronLayerWithDropOut>(5, 2, 0.1));
    layers.push_back(std::make_shared<SoftMaxNeuronLayer<SparseVectorXd>>(2, 2, 0.1));
    NN_parameters params(layers, {2}, {0.05}, {1}, {1}, 0.01, true);

    EXPECT_EQ(params.weightOffset(0, 0, 0), 0);
    EXPECT_EQ(params.weightOffset(0, 1, 0), 5);
    EXPECT_EQ(params.weightOffset(1, 0, 0), 12);
    EXPECT_EQ(params.weightOffset(1, 1, 0), 14);
    EXPECT_EQ(params.biasOffset(0, 0), 10);
    EXPECT_EQ(params.biasOffset(0, 1), 11);
    EXPECT_EQ(params.biasOffset(1, 0), 16);
    EXPECT_EQ(params.biasOffset(1, 1), 17);
}

TEST_F(TestNNParameters, SGD) {
    // construct layers
    std::vector<std::shared_ptr<NeuronLayer<SparseVectorXd>>> layers;
    layers.reserve(2);
    layers.push_back(std::make_shared<ReLUNeuronLayerWithDropOut>(5, 2, 0.1));
    layers.push_back(std::make_shared<SoftMaxNeuronLayer<SparseVectorXd>>(2, 2, 0.1));
    NN_parameters params(layers, {2}, {0.05}, {1}, {1}, 0.01, true);

    params.setTheta(0, 1);

    params.stochasticGradientDescent(0, 0.1);
    EXPECT_DOUBLE_EQ(params.getTheta(0), 1 - 0.1 * (0.01 / (1e-6 + 0.1)));
}

TEST_F(TestNNParameters, RetrieveNodesByData) {
    // construct layers
    std::vector<std::shared_ptr<NeuronLayer<SparseVectorXd>>> layers;
    layers.reserve(2);
    layers.push_back(std::make_shared<ReLUNeuronLayerWithDropOut>(5, 2, 0.1));
    layers.push_back(std::make_shared<SoftMaxNeuronLayer<SparseVectorXd>>(2, 2, 0.1));
    NN_parameters params(layers, {2}, {1}, {1}, {1}, 0.01, true);

    SparseVectorXd input(5);
    input.coeffRef(0) = 1;
    input.coeffRef(3) = 2;
    auto node_set = params.retrieveNodes(0, input);
    auto node_set1 = params.retrieveNodes(0, input);
    EXPECT_TRUE(node_set.size() <= 5);
    EXPECT_EQ(node_set, node_set1);
}

TEST_F(TestNNParameters, RebuildTable) {
    // construct layers
    std::vector<std::shared_ptr<NeuronLayer<SparseVectorXd>>> layers;
    layers.reserve(2);
    layers.push_back(std::make_shared<ReLUNeuronLayerWithDropOut>(5, 2, 0.1));
    layers.push_back(std::make_shared<SoftMaxNeuronLayer<SparseVectorXd>>(2, 2, 0.1));
    NN_parameters params(layers, {2}, {0.05}, {1}, {1}, 0.01, true);

    params.rebuildTables();
}
