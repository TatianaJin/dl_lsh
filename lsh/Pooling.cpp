#include "Pooling.hpp"

VectorXd Pooling::compress(int size, const VectorXd& data) {
    int compressSize = data.size() / size;  // should be divisible?

    VectorXd compressData = VectorXd::Zero(compressSize);
    for (int idx = 0; idx < compressSize; idx++) {
        int offset = idx * size;
        compressData(idx) = mean(data, offset, offset + size);
    }
    return compressData;
}

double Pooling::mean(const VectorXd& data, int start, int end) { return data.block(start, 0, end - start, 1).mean(); }
