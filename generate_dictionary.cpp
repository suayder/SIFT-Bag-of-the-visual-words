/***********************************************************************************************
 * This class generate dictionary for bag of the words
 * Compile like this: g++ -std=c++11 generate_dictionary.cpp `pkg-config --cflags --libs opencv`
 * Author: Suayder
***********************************************************************************************/

#include <stdio.h>
#include <iostream>
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/features2d.hpp"
#include "opencv4/opencv2/xfeatures2d.hpp"
#include "opencv4/opencv2/highgui.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main(){
    char * filename = new char[100];
    Mat input;
    vector<KeyPoint> keypoints; //keypoints that will be extracted by SIFT
    Mat descriptor;
    Mat setOfFeatures;
    Ptr<SIFT> detector = SIFT::create(400);

    const char* image_list[] = {"1.JPG", "2.JPG", "3.JPG", "4.JPG"};

    for(int i=0;i<4;i++){ //Go through all listed images
        sprintf(filename, image_list[i]);
        input = imread(filename, IMREAD_GRAYSCALE);
        detector->detect( input, keypoints);
        detector->compute(input, keypoints,descriptor);

        setOfFeatures.push_back(descriptor);
    }

    int dictionarySize=256;

    TermCriteria tc(1,100,0.001);
    int retries=1;

    int flags=KMEANS_PP_CENTERS;
    BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
    Mat dictionary=bowTrainer.cluster(setOfFeatures);    
    FileStorage fs("dictionary.yml", FileStorage::WRITE);
    fs << "vocabulary" << dictionary;
    fs.release();

    return 0;
}