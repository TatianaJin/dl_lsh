#include "ICostFunction.hpp"

#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>

double ICostFunction::correct(std::vector<double> y_hat, double labels) {
    return (max_idx(y_hat) == (int)labels) ? 1.0 : 0.0;
}

double ICostFunction::accuracy(std::vector<std::vector<double>> y_hat, std::vector<double> labels) {
    double correct = 0;
    for (size_t idx = 0; idx < labels.size(); ++idx) {
        if (max_idx(y_hat[idx]) == (int)labels[idx]) {
            ++correct;
        }
    }
    return correct / labels.size();
}

double ICostFunction::costFunction(std::vector<double> y_hat, double labels) { return -log(y_hat[(int)labels]); }

std::vector<double> ICostFunction::outputDelta(std::vector<double> y_hat, double labels,
                                               std::shared_ptr<NeuronLayer> l) {
    std::vector<double> delta = y_hat;
    delta[(int)labels] -= 1.0;
    return delta;
}

int ICostFunction::max_idx(std::vector<double> v) {
    int max_idx = 0;
    double max_value = DBL_MIN;
    for (size_t i = 0; i < v.size(); ++i) {
        if (max_value < v[i]) {
            max_idx = i;
            max_value = v[i];
        }
    }
    return max_idx;
}
