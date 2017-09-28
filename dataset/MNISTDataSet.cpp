#include "MNISTDataSet.hpp"
#include <fstream>
#include <iostream>
#include <utility>

int MNISTDataSet::LABEL_MAGIC = 2049;
int MNISTDataSet::IMAGE_MAGIC = 2051;

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

pair<vector<VectorXd>, vector<double>> MNISTDataSet::loadDataSet(string label_path, string image_path)
{
    // read MNIST data
    ifstream label_stream;
    ifstream image_stream;
    label_stream.open(label_path);
    image_stream.open(image_path);

    int label_magicNumber;
    label_stream.read((char*)&label_magicNumber,sizeof(int));
    label_magicNumber = ReverseInt(label_magicNumber);
    
    /*while(label_stream >> label_magicNumber)
    {
    	cout << label_magicNumber;
    }*/

    if (label_magicNumber != LABEL_MAGIC)
    {
        cerr << "Label file has wrong magic number: " << label_magicNumber << " expected: " << LABEL_MAGIC << endl;
    }

    int image_magicNumber;
    image_stream.read((char*)&image_magicNumber,sizeof(int));
    //image_stream >> image_magicNumber;
    image_magicNumber = ReverseInt(image_magicNumber);

    if (image_magicNumber != IMAGE_MAGIC)
    {
        cerr << "Image file has wrong magic number: " <<  image_magicNumber << " expected: " << IMAGE_MAGIC << endl;
    }

    int numLabels;
    label_stream.read((char*)&numLabels,sizeof(int));
    numLabels = ReverseInt(numLabels);
    int numImages;
    image_stream.read((char*)&numImages,sizeof(int));
    numImages = ReverseInt(numImages);
    int numRows;
    image_stream.read((char*)&numRows,sizeof(int));
    numRows = ReverseInt(numRows);
    int numCols;
    image_stream.read((char*)&numCols,sizeof(int));
    numCols = ReverseInt(numCols);
    if (numLabels != numImages)
    {
        cerr << "Image file and label file do not contain the same number of entries." << endl;
        cerr << "  Label file contains: " << numLabels << endl;
        cerr << "  Image file contains: " << numImages << endl;
    }

    int label_idx = 0;
    int numImagesRead = 0;
    vector<double> label_list(numLabels);
    vector<VectorXd> image_list(numImages);
    //vector<VectorXd> image_list;
    //vector<VectorXd>::iterator image_iterator = image_list.begin();
    char label;
    unsigned char pixel;
    while (label_stream.is_open() && label_idx < numLabels)
    {
    	label_stream.read(&label,sizeof(char));
        //VectorXd* image = new VectorXd(numRows*numCols);
        image_list[numImagesRead] = * new VectorXd(numRows*numCols);
        label_list[label_idx++] = (double)ReverseInt(label);
        int image_idx = 0;

        for (int colIdx = 0; colIdx < numCols; colIdx++)
        {
            for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
            {
                image_stream.read((char*)&pixel,sizeof(unsigned char));
                //(*image)(image_idx++) = (double)ReverseInt(pixel) / 255.0;
				image_list[numImagesRead](image_idx++) = (double)ReverseInt(pixel) / 255.0;
            }
        }
        //image_list.insert(image_iterator,image);
        //image_list.push_back(*image);
        //image_iterator ++;
        numImagesRead ++;
    }
    assert(label_idx == numImagesRead);
    label_stream.close();
    image_stream.close();
    return make_pair(image_list, label_list);
}