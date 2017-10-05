#ifndef ICOSTFUNCTION_HPP
#define ICOSTFUNCTION_HPP

#include <memory>
#include <vector>

#include "NeuronLayer.hpp"

class ICostFunction {
   public:
    ICostFunction() = default;

    double correct(const std::vector<double>& y_hat, double labels);

    double accuracy(const std::vector<VectorXd>& y_hat, const std::vector<double>& labels);

    double costFunction(const VectorXd& y_hat, double labels);

    VectorXd outputDelta(const VectorXd& y_hat, double labels, std::shared_ptr<NeuronLayer> l);

   private:
    int max_idx(const std::vector<double>& v);
    int max_idx(const VectorXd& v);
};

#endif  // ICOSTFUNCTION_HPP
