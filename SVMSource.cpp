#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include<fstream>
#include<iostream>
#include<windows.h>
#include<atlstr.h>
#include<string>
using namespace std;
using namespace cv;
int main()
{
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	float labels[4] = { 1.0, -1.0, -1.0, -1.0 };
	Mat labelsMat(4, 1, CV_32FC1, labels);

	float trainingData[4][2] = { { 501, 10 }, { 255, 10 }, { 501, 255 }, { 10, 501 } };
	Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

	//Create 900 x (68 x 2) matrix of training data
	CvMLData cvml;
	//cvml.read_csv("C:\\Users\\Shakti\\Downloads\\CK+Dataset\\mytrainSelFeaturesHappy.csv");
	//cvml.read_csv("C:\\Users\\Shakti\\Downloads\\trainlandmarks2.csv");
	cvml.read_csv("C:\\Users\\Shakti\\Downloads\\EmotionTrainData\\newSel_anger_ratios_flipped.csv");

	// Indicate the column that has the response
	const CvMat* temp = cvml.get_values();
	int numfeatures = temp->cols - 1;
	cvml.set_response_idx(numfeatures);
	const CvMat* rs = cvml.get_responses();
	//split train data
	CvTrainTestSplit spl((float)0.66);
	cvml.set_train_test_split(&spl);

	// get the respective indices of the training and testing data and store it in the cv::Mat format.
	const CvMat* traindata_idx = cvml.get_train_sample_idx();
	const CvMat* testdata_idx = cvml.get_test_sample_idx();
	Mat mytraindataidx(traindata_idx);
	Mat mytestdataidx(testdata_idx);

	Mat all_Data(temp);
	Mat all_responses = cvml.get_responses();
	Mat traindata(mytraindataidx.cols, numfeatures, CV_32F);
	Mat trainresponse(mytraindataidx.cols, 1, CV_32S);
	Mat testdata(mytestdataidx.cols, numfeatures, CV_32F);
	Mat testresponse(mytestdataidx.cols, 1, CV_32S);

	//Populate train and test data
	for (int i = 0; i<mytraindataidx.cols; i++)
	{
		trainresponse.at<int>(i) = all_responses.at<float>(mytraindataidx.at<int>(i));
		for (int j = 0; j < numfeatures; j++)
		{
			traindata.at<float>(i, j) = all_Data.at<float>(mytraindataidx.at<int>(i), j);
		}
	}
	cout << "Train data initialized";
	for (int i = 0; i<mytestdataidx.cols; i++)
	{
		testresponse.at<int>(i) = all_responses.at<float>(mytestdataidx.at<int>(i));
		for (int j = 0; j < numfeatures; j++)
		{
			testdata.at<float>(i, j) = all_Data.at<float>(mytestdataidx.at<int>(i), j);
		}
	}
	cout << "Test data initialized";

	//Setup SVM parameters
	CvSVM opencv_svm;
	CvSVMParams params;
	params.svm_type = CvSVM::NU_SVC;
	params.C = 0.1;
	params.nu=0.9;
	params.gamma = 3;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);
	//Train
	if (opencv_svm.train_auto(traindata, trainresponse, Mat(), Mat(), params))
	{
		cout << "Train complete";

	}
	else
		cout << "Train complete";
	//save model to file
	//opencv_svm.save("C:\\Users\\Shakti\\Downloads\\CK+Dataset\\neutralData\\svmNeutral.xml");
	//opencv_svm.save("C:\\Users\\Shakti\\Downloads\\CK+Dataset\\neutralData\\svmEmotion.xml");
	
	
	opencv_svm.save("C:\\Users\\Shakti\\Downloads\\CK+Dataset\\neutralData\\svm_anger_ratios_flipped.xml");

	//Testing
	int k = 0;
	for (int i = 0; i<testdata.rows; i++)
	{
		Mat samplemat(testdata, Range(i, i + 1));
		float response = opencv_svm.predict(samplemat, false);
		k = (response == testresponse.at<int>(i)) ? ++k : k;
		cout << "Test Response: " << response << " Expected" << testresponse.at<int>(i)<<  "\n";
	}
	cout << "accuracy by the opencv svm is ..." << 100.0 * (float)k / testdata.rows << endl;

	//////////////////////////////////////////
	/*
	//Perform predictions
	CvMLData landmarksreadinfo;
	landmarksreadinfo.read_csv("C:\\Users\\Shakti\\Downloads\\CK+Dataset\\emotionCateg\\landmarkinfo.csv");
	const CvMat* landmarkValues = landmarksreadinfo.get_values();
	Mat myLandmarkValues(landmarkValues);
	Mat LandmarkTestData(myLandmarkValues.rows, numfeatures, CV_32F);


	for (int j = 0; j < LandmarkTestData.rows; j++)
	{
	Mat tmpLandmarkTestData(LandmarkTestData, Range(j, j + 1));
	float predictions = opencv_svm.predict(tmpLandmarkTestData);
	cout << "Prediction: " << predictions << "\n";
	}
	*/
	/*

	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Train the SVM
	CvSVM SVM;
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	Vec3b green(0, 255, 0), blue(255, 0, 0);
	// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
	for (int j = 0; j < image.cols; ++j)
	{
	Mat sampleMat = (Mat_<float>(1, 2) << j, i);
	float response = SVM.predict(sampleMat);

	if (response == 1)
	image.at<Vec3b>(i, j) = green;
	else if (response == -1)
	image.at<Vec3b>(i, j) = blue;
	}

	// Show the training data
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

	// Show support vectors
	thickness = 2;
	lineType = 8;
	int c = SVM.get_support_vector_count();

	for (int i = 0; i < c; ++i)
	{
	const float* v = SVM.get_support_vector(i);
	circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
	}

	imwrite("result.png", image);        // save the image

	imshow("SVM Simple Example", image); // show it to the user
	waitKey(0);
	*/

}
