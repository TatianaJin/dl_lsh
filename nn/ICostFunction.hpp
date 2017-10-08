#ifndef ICOSTFUNCTION_HPP
#define ICOSTFUNCTION_HPP

#include <cfloat>
#include <cmath>
#include <memory>
#include <vector>

#include "NeuronLayer.hpp"

template <typename Vector>
class ICostFunction {
   public:
    ICostFunction() = default;

    double correct(const Vector& y_hat, double labels) {
        return (max_idx(y_hat) == (int)labels) ? 1.0 : 0.0;
    }

    double accuracy(const std::vector<Vector>& y_hat, const std::vector<double>& labels) {
        double correct = 0;
        for (size_t idx = 0; idx < labels.size(); ++idx) {
            if (max_idx(y_hat[idx]) == (int)labels[idx]) {
                ++correct;
            }
        }
        return correct / labels.size();
    }

    double costFunction(const Vector& y_hat, double labels) { return -std::log(y_hat.coeff((int)labels)); }

    Vector outputDelta(const Vector& y_hat, double labels, std::shared_ptr<NeuronLayer<Vector>> l) {
        Vector delta = y_hat;
        delta.coeffRef((int)labels) -= 1.0;
        return delta;
    }

   private:
    int max_idx(const std::vector<double>& v) {
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
    int max_idx(const Vector& v) {
        int max_idx = 0;
        double max_value = DBL_MIN;
        for (int i = 0; i < v.size(); ++i) {
            if (max_value < v.coeff(i)) {
                max_idx = i;
                max_value = v.coeff(i);
            }
        }
        return max_idx;
    }
};

#endif  // ICOSTFUNCTION_HPP
