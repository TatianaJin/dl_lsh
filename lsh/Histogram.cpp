#include "Histogram.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "exp/colors.hpp"

void Histogram::add(const std::set<int>& data) {
    if (data.empty()) return;
    for (auto it = data.begin(); it != data.end(); ++it) {
        // auto value = *it; debug
        const auto& pos = histogram.find(*it);
        if (pos == histogram.end()) {
            histogram[*it] = 1;
        } else {
            pos->second++;  // increment the mutable int, tells that it has one more value.
        }
    }
}

std::set<int> Histogram::thresholdSet(double count) {  // get at most [count] nodes
    std::vector<int> node_indices;
    node_indices.reserve(histogram.size());
    for (auto& kv : histogram) {
        node_indices.push_back(kv.first);
    }
    count = std::min(int(count), int(histogram.size()));
    std::sort(node_indices.begin(), node_indices.end(),
              [& histogram = this->histogram](int a, int b) { return histogram[a] > histogram[b]; });
    std::set<int> retrieved(node_indices.begin(), node_indices.begin() + count);
    assert(retrieved.size() == count);
    return retrieved;
}
