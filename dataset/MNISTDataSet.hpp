#ifndef MNISTDATASET_HPP
#define MNISTDATASET_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>

using Eigen::VectorXd;

class MNISTDataSet {
   public:
    /**
     * load mnist dataset from the path and return pair <flattened feature vectors, labels>
     * check the mnist dataset specification at http://yann.lecun.com/exdb/mnist/
     *
     * @param label_path   input file path for labels
     * @param image_path   input file path for image pixel maps
     */
    static std::pair<std::vector<VectorXd>, std::vector<double>> loadDataSet(std::string label_path,
                                                                             std::string image_path);

   private:
    static int LABEL_MAGIC;
    static int IMAGE_MAGIC;
};

#endif  // MNISTDATASET_HPP
