#include "Histogram.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <vector>

using namespace Eigen;

void Histogram::add(std::set<int> data) {
    for (std::set<int>::iterator it = data.begin(); it != data.end(); it++) {
        if (histogram.find(*it) == histogram.end()) {
            histogram.insert(std::make_pair(*it, 1));
        } else {
            histogram.find(*it)->second++;  // increment the mutable int, tells that it has one more value.
        }
    }
}

std::set<int> Histogram::thresholdSet(double count) {
    std::vector<std::pair<int, int>> list(histogram.begin(), histogram.end());
    count = std::min(int(count), int(list.size()));
    sort(list.begin(), list.end(),
         [](const std::pair<int, int> &left, const std::pair<int, int> &right) { return left.second > right.second; });
    std::set<int> retrieved;
    std::vector<std::pair<int, int>>::iterator it = list.begin();
    for (int i = 0; i < count; i++, it++) {
        retrieved.insert(it->first);
    }
    return retrieved;
}
