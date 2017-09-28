#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include <unordered_map>
#include <set>

using namespace std;

class Histogram
{
public:
    void add(set<int> data);

    set<int> thresholdSet(double count);

private:
    unordered_map<int, int> histogram;
};

#endif // HISTOGRAM_HPP