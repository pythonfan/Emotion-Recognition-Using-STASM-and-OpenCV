#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include<iostream>
#include<string>
using namespace std;
using namespace cv;
void svminit(string modelpath);
float getPrediction(Mat landmarkdata);