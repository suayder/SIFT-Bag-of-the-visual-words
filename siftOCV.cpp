/***********************************************************************************************
 * From the dictionary for bag of the words this generate features for machine learning iMput
 * Compile like this: g++ -std=c++11 siftOCV.cpp `pkg-config --cflags --libs opencv`
 * Author: Suayder
***********************************************************************************************/
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/features2d.hpp"
#include "opencv4/opencv2/xfeatures2d.hpp"
#include "opencv4/opencv2/highgui.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void readme();

int main( int argc, char** argv )
{
    if( argc != 2 ){ 
        readme();
        return -1;
    }

    Mat img = imread( argv[1], IMREAD_GRAYSCALE);

    if( !img.data){
        std::cout<< " --(!) Error reading images " << std::endl;
        return -1;
    }

    Mat dictionary;
    FileStorage fs("dictionary.yml", FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");

    Ptr<SIFT> detector = SIFT::create(400);

    BOWImgDescriptorExtractor bowDetector(detector,matcher);

    bowDetector.setVocabulary(dictionary);
 
    //open the file to write the resultant descriptor
    ofstream fileResult("descriptor.txt");

    vector<KeyPoint> keypoints;

    detector->detect(img,keypoints);

    Mat bowDescriptor;

    bowDetector.compute(img,keypoints,bowDescriptor);

    fileResult<<bowDescriptor;

    Mat img_keypoints;

    drawKeypoints( img, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    //-- Show detected (drawn) keypoints
    imwrite("result.jpg", img_keypoints);
}

/** @function readme */
void readme(){
    cout << "\nUsage: ./a.out image_name" <<endl;
}
