#ifndef MNISTDATASET_HPP
#define MNISTDATASET_HPP

#include <vector>
#include <Eigen/Dense>
#include <string>

using Eigen::VectorXd;

class MNISTDataSet
{
public:
    static std::pair<std::vector<VectorXd>, std::vector<double>> loadDataSet(std::string label_path, std::string image_path);

private:
    static int LABEL_MAGIC;
    static int IMAGE_MAGIC;
};

#endif // MNISTDATASET_HPP
