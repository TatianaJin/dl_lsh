#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include "Histogram.hpp"

using namespace Eigen;

void Histogram::add(set<int> data)
{
    for(set<int>::iterator it = data.begin(); it != data.end(); it ++)
    {
        if(histogram.find(*it) == histogram.end())
        {
            histogram.insert(make_pair(*it, 1));
        }
        else
        {
            histogram.find(*it)->second ++;   // increment the mutable int, tells that it has one more value.
        }
    }
}

set<int> Histogram::thresholdSet(double count)
{
    vector<pair<int, int>> list(histogram.begin(), histogram.end());
    count = min(int(count), int(list.size()));
    sort(list.begin(), list.end(), [](const pair<int,int> &left, const pair<int,int> &right) {return left.second > right.second;});
    set<int> retrieved;
    vector<pair<int,int>>::iterator it;
    for(int i = 0; i < count; i++, it++)
    {
        retrieved.insert(it -> first);
    }
    return retrieved;
}