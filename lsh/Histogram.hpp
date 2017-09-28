#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include <set>
#include <unordered_map>

class Histogram {
   public:
    void add(std::set<int> data);

    std::set<int> thresholdSet(double count);

   private:
    std::unordered_map<int, int> histogram;
};

#endif  // HISTOGRAM_HPP
