#ifndef MNISTDATASET_HPP
#define MNISTDATASET_HPP

#include <vector>
#include <Eigen/Dense>
#include <string>

using namespace std;
using namespace Eigen;

class MNISTDataSet
{
public:
    static pair<vector<VectorXd>, vector<double>> loadDataSet(string label_path, string image_path);

private:
    static int LABEL_MAGIC;
    static int IMAGE_MAGIC;
};

#endif // MNISTDATASET_HPP
