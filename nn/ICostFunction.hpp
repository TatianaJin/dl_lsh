#ifndef ICOSTFUNCTION_HPP
#define ICOSTFUNCTION_HPP

#include <memory>
#include <vector>

#include "NeuronLayer.hpp"

using namespace std;

class ICostFunction {
   public:
    ICostFunction() = default;

    double correct(std::vector<double> y_hat, double labels);

    double accuracy(std::vector<std::vector<double>> y_hat, std::vector<double> labels);

    double costFunction(std::vector<double> y_hat, double labels);

    std::vector<double> outputDelta(std::vector<double> y_hat, double labels, std::shared_ptr<NeuronLayer> l);

   private:
    int max_idx(std::vector<double> v);
};

#endif  // ICOSTFUNCTION_HPP
