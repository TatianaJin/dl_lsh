#include "ICostFunction.hpp"

double ICostFunction::correct(vector<double> y_hat, double labels)
{
    return (max_idx(y_hat) == (int) labels) ? 1.0 : 0.0;
}

double ICostFunction::accuracy(vector<vector<double>> y_hat, vector<double> labels)
{
    double correct = 0;
    for (int idx = 0; idx < labels.size(); ++idx)
    {
        if (max_idx(y_hat[idx]) == (int) labels[idx])
        {
            ++correct;
        }
    }
    return correct / labels.size();
}

double ICostFunction::costFunction(vector<double> y_hat, double labels)
{
    return - log(y_hat[(int) labels]);
}

vector<double> ICostFunction::outputDelta(vector<double> y_hat, double labels, NeuronLayer l)
{
    vector<double> delta = y_hat;
    delta[(int)labels] -= 1.0;
    return delta;
}

int ICostFunction::max_idx(vector<double> v)
{
    int max_idx = 0;
    double max_value = DBL_MIN;
    for (int i = 0; i < v.size(); ++i)
    {
        if (max_value < v[i])
        {
            max_idx = i;
            max_value = v[i];
        }
    }
    return max_idx;
}