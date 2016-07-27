/*
Landmarks:
0-10: face outline (lower)
15-30: Eyebrows
58-73: mouth
58-left mouth corner
*/
/*
0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
*/
/*
116 ,121, 134 - classify neutral and peak
*/
#include <stdio.h>
#include <stdlib.h>
#include "opencv/highgui.h"
#include "stasm_lib.h"
#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream>
#include "SVMClassifier.h"
using namespace cv;
using namespace std;

Mat populateLandmarkTest(Mat landmarktestMat, ofstream& landmarkinfo, float landmarks[]);
int main()
{
	static const char* path = "C:/Users/Shakti/Downloads/stasm4.1.0/data/myface.jpg";

	VideoCapture cap;
	//cap.open("C:/Users/Shakti/Downloads/emotions.avi");
	cap.open(0);
	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}
	Mat imgcap;
	cap >> imgcap;
	
	imwrite("C:/Users/Shakti/Downloads/stasm4.1.0/data/myface.jpg", imgcap);
	//Declare csv file to store landmark info
	ofstream landmarkinfo;
	landmarkinfo.open("C:\\Users\\Shakti\\Downloads\\CK+Dataset\\emotionCateg\\landmarkinfo.csv");

	
	
	for (;;)
	{
		cv::Mat_<unsigned char> img(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE));
		Mat allLandmarktest(1, stasm_NLANDMARKS * 2, CV_32F);
		

		if (!img.data)
		{
			printf("Cannot load %s\n", path);
			exit(1);
		}
		int foundface;
		float landmarks[2 * stasm_NLANDMARKS]; // x,y coords
		if (!stasm_search_single(&foundface, landmarks,
			(char*)img.data, img.cols, img.rows, path, "C:/Users/Shakti/Downloads/stasm4.1.0/data"))
		{
			printf("Error in stasm_search_single: %s\n", stasm_lasterr());
			exit(1);
		}
		if (!foundface)
			printf("No face found in %s\n", path);
		else
		{
			// draw the landmarks on the image as white dots
		

			stasm_force_points_into_image(landmarks, img.cols, img.rows);

			for (int i = 0; i < stasm_NLANDMARKS; i++)
			{
				img(cvRound(landmarks[i * 2 + 1]), cvRound(landmarks[i * 2])) = 255;
				float xscale = abs(landmarks[47] - landmarks[15]), yscale = abs(landmarks[22] - landmarks[0]);
				float tmpvalx = (landmarks[i * 2 + 1] - landmarks[47]), tmpvaly = (landmarks[i * 2] - landmarks[0]);
				
			}
		}
		

		//extract face
		int xrows = (landmarks[15] - landmarks[47]);
		int yrows = (landmarks[22] - landmarks[0]);
		//cout << "xrows: " << xrows << " ycols: " << yrows;
		Mat faceimg = Mat::zeros(xrows, yrows, img.type());;

		for (int x = landmarks[47]; x < landmarks[15]; x++)
		{
			for (int y = landmarks[0]; y < landmarks[22]; y++)
			{
				circle(img, Point(cvRound(landmarks[22]), cvRound(landmarks[23])), 3, Scalar(255, 100, 100), 3, 8);	
				faceimg.at<uchar>(x - landmarks[47], y - landmarks[0]) = img.at<uchar>(x, y);

			}
		}

		
		//Write captured landmark info to test file
		Mat landmarkext(1, 32, CV_32F);			//1 row, 12 column matrix to store extracted landmark feature data
		landmarkext = populateLandmarkTest(landmarkext, landmarkinfo, landmarks);
		landmarkinfo << "\n";
		
//**************************************************************************************************		
		//Load svm and train
		svminit("C:\\Users\\Shakti\\Downloads\\CK+Dataset\\neutralData\\svm_happy_ratios_flipped.xml");
		float predicted = getPrediction(landmarkext);
		if (predicted == 1)
		{
			cout << "Predicted frame as happy\n";
		}
		else
		{
			//Load svm and test
			svminit("C:\\Users\\Shakti\\Downloads\\CK+Dataset\\neutralData\\svm_sad_ratios_flipped.xml");
			predicted = getPrediction(landmarkext);
			if (predicted == 1)
				cout << "Predicted frame as sad\n";
			else
			{
				svminit("C:\\Users\\Shakti\\Downloads\\CK+Dataset\\neutralData\\svm_anger_ratios_flipped.xml");
				predicted = getPrediction(landmarkext);
				if (predicted == 1)
					cout << "Predicted frame as anger\n";
				else
					cout << "Predicted frame as neutral\n";
			}

		}
		namedWindow("stasm minimal", CV_WINDOW_NORMAL);
		cv::imshow("stasm minimal", img);
		namedWindow("Face", CV_WINDOW_NORMAL);
		imshow("Face", faceimg);
		for (int j = 0; j < 2;j++)
		cap >> imgcap;
		imwrite("C:/Users/Shakti/Downloads/stasm4.1.0/data/myface.jpg", imgcap);
		//for (int k = 0; k < 10000; k++);
		cv::waitKey(1);

	}
	landmarkinfo.close();
	cv::waitKey();
	return 0;
}

Mat populateLandmarkTest(Mat landmarktestMat, ofstream & landmarkinfo, float landmarks[])
{
	float scalex = max(abs(landmarks[25] - landmarks[1]), abs(landmarks[24] - landmarks[0]));
	float scaley = max(abs(landmarks[12] - landmarks[28]), abs(landmarks[13] - landmarks[29]));	
	landmarkinfo << to_string((landmarks[118] - landmarks[130])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[119] - landmarks[131])/ scaley) + ",";

	landmarktestMat.at<float>(0, 0) = ((landmarks[118] - landmarks[130])/ scalex);
	landmarktestMat.at<float>(0, 1) = ((landmarks[119] - landmarks[131])/ scaley);
	//2
	landmarkinfo << to_string((landmarks[124] - landmarks[148])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[125] - landmarks[149])/ scaley) + ",";

	landmarktestMat.at<float>(0, 2) = ((landmarks[124] - landmarks[148])/ scalex);
	landmarktestMat.at<float>(0, 3) = ((landmarks[125] - landmarks[149])/ scaley);
	//3
	landmarkinfo << to_string((landmarks[112] - landmarks[124])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[113] - landmarks[125])/ scaley) + ",";

	landmarktestMat.at<float>(0, 4) = ((landmarks[112] - landmarks[124])/ scalex);
	landmarktestMat.at<float>(0, 5) = ((landmarks[113] - landmarks[125])/ scaley);
	//4
	landmarkinfo << to_string((landmarks[112] - landmarks[148])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[113] - landmarks[149])/ scaley) + ",";

	landmarktestMat.at<float>(0, 6) = ((landmarks[112] - landmarks[148])/ scalex);
	landmarktestMat.at<float>(0, 7) = ((landmarks[113] - landmarks[149])/ scaley);
	//5
	landmarkinfo << to_string((landmarks[118] - landmarks[116])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[119] - landmarks[117])/ scaley) + ",";

	landmarktestMat.at<float>(0, 8) = ((landmarks[118] - landmarks[116])/ scalex);
	landmarktestMat.at<float>(0, 9) = ((landmarks[119] - landmarks[117])/ scaley);
	//6
	landmarkinfo << to_string((landmarks[130] - landmarks[110])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[131] - landmarks[111])/ scaley) + ",";

	landmarktestMat.at<float>(0, 10) = ((landmarks[130] - landmarks[110])/ scalex);
	landmarktestMat.at<float>(0, 11) = ((landmarks[131] - landmarks[111])/ scaley);
	//7
	landmarkinfo << to_string((landmarks[60] - landmarks[118])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[61] - landmarks[119])/ scaley) + ",";

	landmarktestMat.at<float>(0, 12) = ((landmarks[60] - landmarks[118])/ scalex);
	landmarktestMat.at<float>(0, 13) = ((landmarks[61] - landmarks[119])/ scaley);
	//8
	landmarkinfo << to_string((landmarks[80] - landmarks[130])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[81] - landmarks[131])/ scaley) + ",";

	landmarktestMat.at<float>(0, 14) = ((landmarks[80] - landmarks[130])/ scalex);
	landmarktestMat.at<float>(0, 15) = ((landmarks[81] - landmarks[131])/ scaley);
	//9
	landmarkinfo << to_string((landmarks[68] - landmarks[60])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[69] - landmarks[61])/ scaley) + ",";

	landmarktestMat.at<float>(0, 16) = ((landmarks[68] - landmarks[60])/ scalex);
	landmarktestMat.at<float>(0, 17) = ((landmarks[69] - landmarks[61])/ scaley);
	//10
	landmarkinfo << to_string((landmarks[80] - landmarks[88])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[81] - landmarks[89])/ scaley) + ",";

	landmarktestMat.at<float>(0, 18) = ((landmarks[80] - landmarks[88])/ scalex);
	landmarktestMat.at<float>(0, 19) = ((landmarks[81] - landmarks[89])/ scaley);
	//11
	landmarkinfo << to_string((landmarks[64] - landmarks[72])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[65] - landmarks[73])/ scaley) + ",";

	landmarktestMat.at<float>(0, 20) = ((landmarks[64] - landmarks[72])/ scalex);
	landmarktestMat.at<float>(0, 21) = ((landmarks[65] - landmarks[73])/ scaley);
	//12
	landmarkinfo << to_string((landmarks[84] - landmarks[92])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[85] - landmarks[93])/ scaley) + ",";

	landmarktestMat.at<float>(0, 22) = ((landmarks[84] - landmarks[92])/ scalex);
	landmarktestMat.at<float>(0, 23) = ((landmarks[85] - landmarks[93])/ scaley);
	//13
	landmarkinfo << to_string((landmarks[42] - landmarks[104])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[43] - landmarks[105])/ scaley) + ",";

	landmarktestMat.at<float>(0, 24) = ((landmarks[42] - landmarks[104])/ scalex);
	landmarktestMat.at<float>(0, 25) = ((landmarks[43] - landmarks[105])/ scaley);
	//14
	landmarkinfo << to_string((landmarks[44] - landmarks[104])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[45] - landmarks[105])/ scaley) + ",";

	landmarktestMat.at<float>(0, 26) = ((landmarks[44] - landmarks[104])/ scalex);
	landmarktestMat.at<float>(0, 27) = ((landmarks[45] - landmarks[105])/ scaley);
	//15
	landmarkinfo << to_string((landmarks[42] - landmarks[44])/ scalex) + ",";
	landmarkinfo << to_string((landmarks[43] - landmarks[45])/ scaley) + ",";

	landmarktestMat.at<float>(0, 28) = ((landmarks[42] - landmarks[44])/ scalex);
	landmarktestMat.at<float>(0, 29) = ((landmarks[43] - landmarks[45])/ scaley);

	//16
	
	landmarkinfo << to_string((landmarks[25] - landmarks[1])/ scalex) + ",";

	landmarktestMat.at<float>(0, 30) = ((landmarks[25] - landmarks[1])/ scalex);
	//17
	//15
	landmarkinfo << to_string((landmarks[12] - landmarks[28])/ scaley) + ",";
	
	landmarktestMat.at<float>(0, 31) = ((landmarks[12] - landmarks[28])/ scalex);
	return landmarktestMat;
}