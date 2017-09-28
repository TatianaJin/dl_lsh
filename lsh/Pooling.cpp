#include <iostream>
#include "Pooling.hpp"

VectorXd Pooling::compress(int size, VectorXd data)
{
    int compressSize = data.size() / size; // should be divisible?

    VectorXd compressData = VectorXd::Zero(compressSize);
    for(int idx = 0; idx < compressSize; idx ++)
    {
        int offset = idx * size;
        compressData(idx) = sum(data, offset, offset + size);
    }
    return compressData;
}

double Pooling::sum(VectorXd data, int start, int end)
{
    double value = 0;
    for(int idx = start; idx < end; idx ++)
    {
        value += data(idx);
    }
    return value / (end - start);
}