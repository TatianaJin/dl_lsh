#ifndef ICOSTFUNCTION_HPP
#define ICOSTFUNCTION_HPP

#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>
#include "NeuronLayer.hpp"

using namespace std;

class ICostFunction
{
public:
    ICostFunction() = default;

    double correct(vector<double> y_hat, double labels);

    double accuracy(vector<vector<double>> y_hat, vector<double> labels);

    double costFunction(vector<double> y_hat, double labels);

    vector<double> outputDelta(vector<double> y_hat, double labels, NeuronLayer l);

private:
    int max_idx(vector<double> v);
};

#endif // ICOSTFUNCTION_HPP