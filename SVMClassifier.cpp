#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include<fstream>
#include<iostream>
#include<windows.h>
#include<atlstr.h>
#include<string>
#include"SVMClassifier.h"
using namespace std;
using namespace cv;
CvSVM svm;
void svminit(string modelpath)
{
	const char* modelfile = modelpath.c_str();
	svm.load(modelfile);
}
float getPrediction(Mat landmarkext)
{
	float prediction = svm.predict(landmarkext, false);
	//cout << "Prediction in prediction function:" << prediction<<"\n";
	return prediction;
}