#include "MNISTDataSet.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int MNISTDataSet::LABEL_MAGIC = 2049;  // the magic number given in the dataset
int MNISTDataSet::IMAGE_MAGIC = 2051;

int ReverseInt(int i)  // the integers are stored in MSB first manner
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

std::pair<std::vector<VectorXd>, std::vector<double>> MNISTDataSet::loadDataSet(std::string label_path,
                                                                                std::string image_path) {
    // read MNIST data
    std::ifstream label_stream;
    std::ifstream image_stream;
    label_stream.open(label_path);
    image_stream.open(image_path);

    int label_magicNumber;
    label_stream.read((char*)&label_magicNumber, sizeof(int));
    label_magicNumber = ReverseInt(label_magicNumber);

    if (label_magicNumber != LABEL_MAGIC) {
        std::cerr << "Label file has wrong magic number: " << label_magicNumber << " expected: " << LABEL_MAGIC
                  << std::endl;
    }

    int image_magicNumber;
    image_stream.read((char*)&image_magicNumber, sizeof(int));
    // image_stream >> image_magicNumber;
    image_magicNumber = ReverseInt(image_magicNumber);

    if (image_magicNumber != IMAGE_MAGIC) {
        std::cerr << "Image file has wrong magic number: " << image_magicNumber << " expected: " << IMAGE_MAGIC
                  << std::endl;
    }

    int numLabels;
    label_stream.read((char*)&numLabels, sizeof(int));
    numLabels = ReverseInt(numLabels);
    int numImages;
    image_stream.read((char*)&numImages, sizeof(int));
    numImages = ReverseInt(numImages);
    int numRows;
    image_stream.read((char*)&numRows, sizeof(int));
    numRows = ReverseInt(numRows);
    int numCols;
    image_stream.read((char*)&numCols, sizeof(int));
    numCols = ReverseInt(numCols);
    if (numLabels != numImages) {
        std::cerr << "Image file and label file do not contain the same number of entries." << std::endl;
        std::cerr << "  Label file contains: " << numLabels << std::endl;
        std::cerr << "  Image file contains: " << numImages << std::endl;
    } else {
        std::cout << "Loading dataset with " << numLabels << " records\n";
        std::cout << "\tlabels:" << label_path << ", images: " << image_path << "\n";
        std::cout << "\tno. columns: " << numCols << ", no. rows: " << numRows << std::endl;
    }

    int label_idx = 0;
    int numImagesRead = 0;
    std::vector<double> label_list(numLabels);
    std::vector<VectorXd> image_list(numImages);

    char label;
    unsigned char pixel;
    while (label_stream.is_open() && label_idx < numLabels) {
        label_stream.read(&label, sizeof(char));
        image_list[numImagesRead] = VectorXd(numRows * numCols);
        label_list[label_idx++] = (double)label;
        int image_idx = 0;

        for (int colIdx = 0; colIdx < numCols; colIdx++) {
            for (int rowIdx = 0; rowIdx < numRows; rowIdx++)  // the data is actually stored row-wise...
            {
                assert(image_stream.is_open());  // debug
                image_stream >> pixel;
                image_list[numImagesRead](image_idx++) = ((double)pixel) / 255.0;
            }
        }
        ++numImagesRead;
    }
    assert(label_idx == numImagesRead);  // debug
    label_stream.close();
    image_stream.close();
    return std::make_pair(std::move(image_list), std::move(label_list));
}
